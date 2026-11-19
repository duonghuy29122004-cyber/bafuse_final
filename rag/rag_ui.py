from __future__ import annotations

import os
from pathlib import Path
from typing import Any, MutableMapping

import streamlit as st

from rag.rag_service import (
    DEFAULT_EMBEDDING_PATH,
    DEFAULT_META_PATH,
    DEFAULT_MODEL_NAME,
    RagApiConfig,
    RagIndex,
    RagResponseError,
    RagServiceUnavailable,
    check_embedding,
    check_vllm,
    endpoint_approval_target,
    generate_answer,
    load_rag_index,
    normalize_endpoint,
    retrieve_top_k,
)

SUGGESTED_QUESTION = (
    "Based on the predicted SoH of this battery, explain its health condition and "
    "discuss possible degradation mechanisms using the retrieved scientific literature."
)


@st.cache_resource(show_spinner="Loading local scientific RAG index...")
def _cached_rag_index(meta_path: str, embedding_path: str) -> RagIndex:
    return load_rag_index(meta_path, embedding_path)


def reset_rag_conversation(state: MutableMapping[str, Any]) -> None:
    state["rag_messages"] = []
    state["rag_retrieved_chunks"] = []


def _render_references(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    st.markdown("**Retrieved scientific references**")
    for rank, chunk in enumerate(chunks, start=1):
        source = str(chunk["doc_name"])
        section = str(chunk["header_breadcrumb"])
        score = float(chunk["score"])
        st.markdown(f"**{rank}. {source}** · {section} · score `{score:.3f}`")
        with st.expander(f"Show retrieved passage · {source} · {rank}", expanded=False):
            st.write(chunk["page_content"])


def _format_context(context: dict[str, Any]) -> None:
    st.subheader("Battery Analysis Context")
    if context.get("part"):
        st.text(f"Dataset: Oxford Part {context['part']}")
    st.text(f"Cell: G{context['group']}-{context['cell']}")
    st.text(f"RPT: RPT{context['rpt']}")
    st.text(f"Model: {context['model']}")
    for label, key in (
        ("Predicted SoH", "predicted_soh"),
        ("Label SoH", "saved_label_soh"),
        ("PyDMA LLI", "lli_percent"),
        ("PyDMA anode LAM", "lam_anode_percent"),
        ("PyDMA cathode LAM", "lam_cathode_percent"),
    ):
        value = context.get(key)
        st.text(f"{label}: {value:.3f}%" if value is not None else f"{label}: Unavailable")
    st.caption(
        "Prediction is a model output; saved SoH is a provisional label; LLI/LAM "
        "values are saved PyDMA diagnostics, not independently verified causes."
    )


def _save_api_field(key: str) -> None:
    st.session_state[key] = st.session_state[f"_{key}"]
    st.session_state.pop("rag_connection_test", None)
    if key in {"rag_generation", "rag_embedding"}:
        st.session_state.pop(f"{key}_approval", None)


def _api_field(key: str, label: str, default: str, placeholder: str = "") -> str:
    value = st.session_state.setdefault(key, default)
    return st.text_input(
        label, value=value, placeholder=placeholder, key=f"_{key}",
        on_change=_save_api_field, args=(key,),
    ).strip()


def _resolve_api_endpoint(value: str, service: str) -> tuple[str | None, str | None]:
    if not value:
        return None, "Enter the API URL."
    key = f"rag_{service}_approval"
    target = endpoint_approval_target(value)
    if target and st.session_state.get(key) != target:
        if st.button(f"Approve {service} host {target}", key=key + "_button"):
            st.session_state[key] = target
            st.session_state.pop("rag_connection_test", None)
    try:
        return normalize_endpoint(value, service, approved_target=st.session_state.get(key)), None
    except RagServiceUnavailable as error:
        return None, str(error)


def _render_api_settings(index: RagIndex | None) -> RagApiConfig | None:
    generation_default = os.getenv("BAFUSE_RAG_VLLM_URL", "").strip()
    embedding_default = os.getenv("BAFUSE_RAG_EMBED_URL", "").strip()
    with st.sidebar:
        st.subheader("RAG API settings")
        generation_raw = _api_field(
            "rag_generation", "vLLM API URL (Port 8000)", generation_default,
            "https://your-vllm.trycloudflare.com",
        )
        embedding_raw = _api_field(
            "rag_embedding", "Embedding API URL (Port 8001)", embedding_default,
            "https://your-embedding.trycloudflare.com",
        )
        model = _api_field(
            "rag_model", "vLLM Model Name",
            os.getenv("BAFUSE_RAG_MODEL_NAME", "").strip() or DEFAULT_MODEL_NAME,
        )
        generation, generation_error = _resolve_api_endpoint(generation_raw, "generation")
        embedding, embedding_error = _resolve_api_endpoint(embedding_raw, "embedding")
        if generation_error or embedding_error:
            st.caption("Approve each remote HTTPS host before sending requests.")

        # Keep an environment credential confined to the configured endpoint pair.
        api_key = None
        if (generation_raw, embedding_raw) == (generation_default, embedding_default):
            api_key = os.getenv("BAFUSE_RAG_API_KEY", "").strip() or None
        dimensions = None if index is None else int(index.embeddings.shape[1])
        signature = (generation_raw, embedding_raw, generation, embedding, model, dimensions)
        if st.button("Test API Connections", key="rag_test_api"):
            results = {}
            try:
                if generation is None or not model:
                    raise RagServiceUnavailable(generation_error or "Enter the vLLM model name.")
                check_vllm(generation, model, api_key=api_key)
                results["vLLM"] = (True, f"Model {model} found.")
            except (RagServiceUnavailable, RagResponseError) as error:
                results["vLLM"] = (False, str(error))

            try:
                if embedding is None or dimensions is None:
                    raise RagServiceUnavailable(embedding_error or "The local index is unavailable.")
                received = check_embedding(embedding, dimensions, api_key=api_key)
                results["Embedding"] = (True, f"{received:,} dimensions match the local index.")
            except (RagServiceUnavailable, RagResponseError) as error:
                results["Embedding"] = (False, str(error))
            st.session_state["rag_connection_test"] = (signature, results)

        tested = st.session_state.get("rag_connection_test")
        if tested and tested[0] == signature:
            for service, (connected, detail) in tested[1].items():
                if connected:
                    st.success(f"{service} connected. {detail}")
                else:
                    st.error(f"{service} failed. {detail}")
        else:
            st.caption("API connections have not been tested for these settings.")

    if generation is None or embedding is None or not model:
        return None
    return RagApiConfig(embedding, generation, model, api_key)


def render_rag_page(
    battery_context: dict[str, Any] | None,
    *,
    meta_path: str | Path = DEFAULT_META_PATH,
    embedding_path: str | Path = DEFAULT_EMBEDDING_PATH,
) -> None:
    st.markdown('<div class="section-title">RAG Assistant</div>', unsafe_allow_html=True)
    if battery_context:
        st.session_state["rag_last_context"] = dict(battery_context)
        _format_context(battery_context)
        st.caption("Suggested question")
        st.code(SUGGESTED_QUESTION, language=None)
    else:
        st.session_state["rag_last_context"] = None
        st.caption("No valid battery prediction is attached. General scientific Q&A remains available.")

    try:
        index = _cached_rag_index(str(meta_path), str(embedding_path))
    except (RagServiceUnavailable, RagResponseError) as error:
        st.error(f"RAG local index unavailable: {error}")
        index = None

    config = _render_api_settings(index)

    if index is not None:
        st.caption(
            f"Local scientific index ready: {len(index.chunks):,} passages · "
            f"{index.embeddings.shape[1]:,}-dimension embeddings · hybrid retrieval · Top-K 4."
        )
    if config is None:
        st.info("Enter both API URLs and approve the remote hosts in the sidebar to use RAG.")

    messages = st.session_state.setdefault("rag_messages", [])
    st.session_state.setdefault("rag_retrieved_chunks", [])
    if st.button("New RAG conversation", key="rag_new_conversation"):
        reset_rag_conversation(st.session_state)
        st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("retrieved_chunks"):
                _render_references(message["retrieved_chunks"])

    question = st.chat_input(
        "Ask a battery science question or about the attached prediction",
        disabled=config is None or index is None,
        key="rag_chat_input",
    )
    if question and config is not None and index is not None:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        retrieved: list[dict[str, Any]] = []
        try:
            with st.spinner("Retrieving scientific references and generating an answer..."):
                retrieved = retrieve_top_k(question, index, config)
                st.session_state["rag_retrieved_chunks"] = retrieved
                answer = generate_answer(
                    question,
                    retrieved,
                    battery_context,
                    messages[:-1],
                    config,
                )
        except (RagServiceUnavailable, RagResponseError, ValueError) as error:
            assistant_message = {
                "role": "assistant",
                "content": (
                    f"RAG service not connected — {error}"
                    if isinstance(error, RagServiceUnavailable)
                    else f"RAG request failed: {error}"
                ),
                "retrieved_chunks": retrieved,
            }
        else:
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "retrieved_chunks": retrieved,
            }
        messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(assistant_message["content"])
            _render_references(assistant_message.get("retrieved_chunks", []))
