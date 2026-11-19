import json
import os
import re
from pathlib import Path
import numpy as np
import requests
import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi

if __package__:
    from .scientific_prompt import RAG_SYSTEM_PROMPT_TEMPLATE
else:
    from scientific_prompt import RAG_SYSTEM_PROMPT_TEMPLATE

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN FILE INDEX TRÊN MÁY LOCAL & SYSTEM PROMPT
# ==============================================================================
# Đặt 2 file tải từ Google Drive (06_rag_index) cùng thư mục với file app_local.py
BASE_DIR = Path(__file__).resolve().parent
EMB_FILE = BASE_DIR / "chunk_embeddings_lora.npy"
META_FILE = BASE_DIR / "chunks_simplified.json"

# Giá trị mặc định (có thể dán trực tiếp trên giao diện Sidebar khi chạy)
DEFAULT_VLLM_URL = os.environ.get("BAFUSE_RAG_VLLM_URL","")
DEFAULT_EMBED_URL = os.environ.get("BAFUSE_RAG_EMBED_URL", "")
DEFAULT_MODEL_NAME = "qwen25-battery-lora"


# ==============================================================================
# 2. CÁC HÀM XỬ LÝ VECTOR INDEX LOCAL (CPU) & GỌI API TỪ XA
# ==============================================================================
def tokenize_vi(text: str):
    return re.findall(r"\w+", text.lower())


def normalize_vllm_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def normalize_embed_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.endswith("/embed"):
        url = f"{url}/embed"
    return url


@st.cache_resource(show_spinner="Đang nạp Vector Index & BM25 trên RAM (0 GB VRAM)...")
def load_local_rag_index(meta_path: str, emb_path: str):
    meta_file = Path(meta_path)
    emb_file = Path(emb_path)
    if not meta_file.exists() or not emb_file.exists():
        return None, None, None

    chunks = json.loads(meta_file.read_text(encoding="utf-8"))
    emb_matrix = np.load(emb_file)

    bm25_corpus = [tokenize_vi(c["embed_text"]) for c in chunks]
    bm25 = BM25Okapi(bm25_corpus)
    return chunks, emb_matrix, bm25


def embed_query_via_api(query: str, embed_endpoint: str) -> np.ndarray:
    """Gọi API tới Colab/Server để nhúng câu hỏi bằng Qwen2.5-7B-LoRA."""
    resp = requests.post(
        embed_endpoint,
        json={"query": query},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    vec = np.array(data["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def retrieve_top_k_chunks(
    query: str,
    chunks_data: list,
    chunk_embeddings: np.ndarray,
    bm25_index: BM25Okapi,
    embed_endpoint: str,
    top_k: int = 4,
    alpha_dense: float = 0.65,
):
    """Kết hợp Cosine Similarity từ API Fine-tuned LoRA Embedding và điểm BM25 local."""
    q_vec = embed_query_via_api(query, embed_endpoint)
    dense_scores = np.dot(chunk_embeddings, q_vec)
    dense_norm = (dense_scores - dense_scores.min()) / (
        dense_scores.max() - dense_scores.min() + 1e-8
    )

    bm25_scores = np.array(bm25_index.get_scores(tokenize_vi(query)))
    bm25_norm = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + 1e-8
    )

    final_scores = alpha_dense * dense_norm + (1.0 - alpha_dense) * bm25_norm
    for idx, c in enumerate(chunks_data):
        if c.get("block_type") == "nomenclature" or len(c["page_content"]) < 250:
            final_scores[idx] -= 0.5

    top_indices = np.argsort(final_scores)[::-1][:top_k]
    retrieved = []
    for idx in top_indices:
        item = dict(chunks_data[idx])
        item["score"] = float(final_scores[idx])
        item["dense_score"] = float(dense_scores[idx])
        retrieved.append(item)
    return retrieved


def format_retrieved_context(chunks: list) -> str:
    blocks = []
    for idx, c in enumerate(chunks, 1):
        blocks.append(
            f"--- Đoạn tài liệu #{idx} ---\n"
            f"Nguồn: {c['doc_name']} ({c.get('display_name', '')})\n"
            f"Đề mục: {c.get('header_breadcrumb', '')}\n"
            f"Nội dung:\n{c['page_content']}"
        )
    return "\n\n".join(blocks)


# ==============================================================================
# 3. HÀM HIỂN THỊ GIAO DIỆN CHAT (CÓ THỂ CHẠY ĐỘC LẬP HOẶC GỌI TỪ DASHBOARD KHÁC)
# ==============================================================================
def render_rag_chat_page():
    with st.sidebar:
        st.title("⚙️ Cấu hình Kết nối API")
        vllm_raw_url = st.text_input(
            "🔗 Link vLLM API (Port 8000):",
            value=st.session_state.get("vllm_url", DEFAULT_VLLM_URL),
            help="Configured vLLM endpoint; the client appends /v1 when needed.",
        )
        embed_raw_url = st.text_input(
            "🔗 Link Embed API (Port 8001):",
            value=st.session_state.get("embed_url", DEFAULT_EMBED_URL),
            help="Configured embedding endpoint; the client appends /embed when needed.",
        )
        model_name = st.text_input(
            "🤖 Tên Model trên vLLM:",
            value=DEFAULT_MODEL_NAME,
        )

        st.session_state["vllm_url"] = vllm_raw_url
        st.session_state["embed_url"] = embed_raw_url

        vllm_base_url = normalize_vllm_url(vllm_raw_url)
        embed_endpoint = normalize_embed_url(embed_raw_url)

        if st.button("🔌 Kiểm tra kết nối API", use_container_width=True):
            try:
                r_vllm = requests.get(f"{vllm_base_url}/models", timeout=10)
                r_vllm.raise_for_status()
                st.success("Kết nối vLLM Server thành công!")
            except Exception as e:
                st.error(f"Lỗi kết nối vLLM API: {e}")

            try:
                r_emb = requests.post(embed_endpoint, json={"query": "SOH"}, timeout=15)
                r_emb.raise_for_status()
                dim = len(r_emb.json().get("embedding", []))
                st.success(f"Kết nối Embed API thành công! (Vector dim: {dim})")
            except Exception as e:
                st.error(f"Lỗi kết nối Embed API: {e}")

        st.divider()
        st.subheader("🎛️ Tham số Truy xuất & Sinh")
        top_k = st.slider("Số đoạn tài liệu truy xuất (Top-K)", min_value=2, max_value=6, value=4)
        temperature = st.slider("Temperature", min_value=0.0, max_value=0.8, value=0.1, step=0.05)
        single_retrieval_mode = st.checkbox(
            "Giữ cố định ngữ cảnh từ Lượt 1 (Single-Retrieval)",
            value=True,
            help="Đúng theo thiết kế tập huấn luyện: Chỉ truy xuất tài liệu ở câu hỏi đầu tiên của phiên chat.",
        )

        if st.button("🗑️ Xóa lịch sử / Hội thoại mới", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.retrieved_chunks = []
            st.rerun()

    # Giao diện chính
    st.title("🔋 Trợ lý Chẩn đoán & Phân tích Suy thoái Pin Lithium-ion (RAG)")
    st.caption(
        "Chạy Local (0 GB VRAM) • Kết nối API từ xa tới Qwen2.5-7B-Instruct (LoRA BF16) & Kho 15 Bài báo Tổng quan (P01–P15)"
    )

    chunks_data, chunk_embeddings, bm25_index = load_local_rag_index(
        str(META_FILE), str(EMB_FILE)
    )

    if chunks_data is None:
        st.warning(
            f"⚠️ Không tìm thấy 2 file dữ liệu Index tại thư mục local:\n"
            f"- `{META_FILE}`\n"
            f"- `{EMB_FILE}`\n\n"
            f"Vui lòng tải 2 file này từ thư mục `06_rag_index` trên Google Drive về đặt cùng thư mục với `app_local.py`."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input(
        "Nhập câu hỏi về cơ chế suy thoái (LLI, LAM, SEI, mạ Li), chẩn đoán EIS/IC-DV hoặc BMS..."
    ):
        with st.chat_message("user"):
            st.markdown(user_query)

        if not st.session_state.retrieved_chunks or not single_retrieval_mode:
            with st.spinner("Đang gọi API nhúng Qwen2.5-LoRA và truy xuất tài liệu..."):
                try:
                    st.session_state.retrieved_chunks = retrieve_top_k_chunks(
                        query=user_query,
                        chunks_data=chunks_data,
                        chunk_embeddings=chunk_embeddings,
                        bm25_index=bm25_index,
                        embed_endpoint=embed_endpoint,
                        top_k=top_k,
                    )
                except Exception as e:
                    st.error(
                        f"❌ Không thể gọi Embed API tại `{embed_endpoint}`. "
                        f"Hãy kiểm tra lại link Cloudflare Tunnel ở thanh bên. Chi tiết lỗi: {e}"
                    )
                    st.stop()

        st.session_state.messages.append({"role": "user", "content": user_query})

        ctx_str = format_retrieved_context(st.session_state.retrieved_chunks)
        sys_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.replace("{retrieved_context}", ctx_str)
        api_messages = [{"role": "system", "content": sys_prompt}] + st.session_state.messages

        client = OpenAI(base_url=vllm_base_url, api_key="EMPTY")
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=1024,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(
                    f"❌ Lỗi khi gọi vLLM API tại `{vllm_base_url}`: {e}. "
                    f"Vui lòng kiểm tra lại link Cloudflare Tunnel cổng 8000."
                )
                st.stop()

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.session_state.retrieved_chunks:
        with st.expander(
            f"📚 Xem {len(st.session_state.retrieved_chunks)} đoạn tài liệu tham khảo đã truy xuất (RAG Context)",
            expanded=False,
        ):
            for i, c in enumerate(st.session_state.retrieved_chunks, 1):
                st.markdown(
                    f"**Đoạn #{i} — Nguồn: `{c['doc_name']}`** "
                    f"*(Điểm tổng hợp: `{c['score']:.3f}` | Cosine LoRA: `{c['dense_score']:.3f}`)*  \n"
                    f"**Đề mục:** {c['header_breadcrumb']}"
                )
                st.info(c["page_content"])


if __name__ == "__main__":
    st.set_page_config(
        page_title="Battery Health RAG - Local Client",
        page_icon="🔋",
        layout="wide",
    )
    render_rag_chat_page()
