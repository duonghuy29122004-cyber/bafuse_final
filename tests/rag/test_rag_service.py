from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests
from rank_bm25 import BM25Okapi

from rag.rag_service import (
    DEFAULT_MODEL_NAME,
    RagIndex,
    RagResponseError,
    RagServiceUnavailable,
    embed_query,
    generate_answer,
    load_api_config,
    load_rag_index,
    rag_service_available,
    retrieve_top_k,
)
from rag.rag_ui import reset_rag_conversation


ROOT = Path(__file__).resolve().parents[2]


def _chunk(name: str, text: str) -> dict[str, str]:
    return {
        "doc_name": name,
        "page_content": text,
        "embed_text": text,
        "header_breadcrumb": "Battery degradation",
        "block_type": "text",
    }


def _index() -> RagIndex:
    chunks = [
        _chunk("P01.md", "Lithium inventory loss can occur through SEI growth." * 6),
        _chunk("P02.md", "Active material loss changes battery capacity." * 6),
    ]
    return RagIndex(
        chunks,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        BM25Okapi([["lithium", "inventory", "sei"], ["active", "material", "capacity"]]),
    )


class RagServiceTests(unittest.TestCase):
    def test_real_index_count_dimensions_and_metadata(self) -> None:
        index = load_rag_index(
            ROOT / "rag" / "chunks_simplified.json",
            ROOT / "rag" / "chunk_embeddings_lora.npy",
        )
        self.assertEqual(len(index.chunks), 931)
        self.assertEqual(index.embeddings.shape, (931, 3584))
        self.assertTrue(all(chunk["doc_name"].startswith("P") for chunk in index.chunks))

    def test_loader_rejects_chunk_embedding_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks.json").write_text(
                json.dumps([_chunk("P01.md", "A sufficiently long passage.")]),
                encoding="utf-8",
            )
            np.save(root / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))
            with self.assertRaisesRegex(RagResponseError, "do not match"):
                load_rag_index(root / "chunks.json", root / "embeddings.npy")

    def test_loader_rejects_non_2d_embedding_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks.json").write_text(
                json.dumps([_chunk("P01.md", "A sufficiently long passage.")]),
                encoding="utf-8",
            )
            np.save(root / "embeddings.npy", np.zeros(4, dtype=np.float32))
            with self.assertRaisesRegex(RagResponseError, "two-dimensional"):
                load_rag_index(root / "chunks.json", root / "embeddings.npy")

    def test_api_configuration_requires_both_endpoints_and_secures_remote_hosts(self) -> None:
        self.assertIsNone(load_api_config({}))
        local = load_api_config(
            {
                "BAFUSE_RAG_EMBED_URL": "http://127.0.0.1:8001",
                "BAFUSE_RAG_VLLM_URL": "http://localhost:8000",
            }
        )
        self.assertIsNotNone(local)
        self.assertEqual(local.embed_url, "http://127.0.0.1:8001/embed")
        self.assertEqual(local.vllm_url, "http://localhost:8000/v1")
        self.assertEqual(local.model_name, DEFAULT_MODEL_NAME)
        self.assertFalse(rag_service_available({"BAFUSE_RAG_EMBED_URL": "http://localhost:1"}))

        with self.assertRaisesRegex(RagServiceUnavailable, "allowlisted HTTPS"):
            load_api_config(
                {
                    "BAFUSE_RAG_EMBED_URL": "https://embed.example",
                    "BAFUSE_RAG_VLLM_URL": "https://llm.example",
                }
            )
        remote = load_api_config(
            {
                "BAFUSE_RAG_EMBED_URL": "https://embed.example",
                "BAFUSE_RAG_VLLM_URL": "https://llm.example/v1",
                "BAFUSE_RAG_ALLOW_REMOTE": "true",
                "BAFUSE_RAG_ALLOWED_HOSTS": "embed.example,llm.example",
                "BAFUSE_RAG_API_KEY": "test-key",
            }
        )
        self.assertEqual(remote.api_key, "test-key")
        self.assertEqual(remote.vllm_url, "https://llm.example/v1")
        with self.assertRaisesRegex(RagServiceUnavailable, "allowlisted HTTPS"):
            load_api_config(
                {
                    "BAFUSE_RAG_EMBED_URL": "https://evil.example",
                    "BAFUSE_RAG_VLLM_URL": "https://llm.example",
                    "BAFUSE_RAG_ALLOW_REMOTE": "true",
                    "BAFUSE_RAG_ALLOWED_HOSTS": "llm.example",
                }
            )

    def test_embedding_response_contract_and_dimensions(self) -> None:
        config = SimpleNamespace(embed_url="http://localhost/embed", api_key=None)
        with patch(
            "rag.rag_service._post_json",
            return_value={"embedding": [3.0, 4.0]},
        ) as post:
            vector = embed_query("battery", config, expected_dimensions=2)
        np.testing.assert_allclose(vector, [0.6, 0.8])
        post.assert_called_once()
        with patch(
            "rag.rag_service._post_json",
            return_value={"embedding": [1.0]},
        ):
            with self.assertRaisesRegex(RagResponseError, "dimensions"):
                embed_query("battery", config, expected_dimensions=2)

    def test_api_transport_blocks_redirects_and_reports_unreachable_service_safely(self) -> None:
        config = SimpleNamespace(embed_url="https://embed.example/embed", api_key="secret-token")
        with patch(
            "rag.rag_service.requests.post",
            side_effect=requests.ConnectionError("secret-token must not be surfaced"),
        ) as post:
            with self.assertRaisesRegex(RagServiceUnavailable, "could not be reached") as raised:
                embed_query("battery", config, expected_dimensions=2)
        self.assertNotIn("secret-token", str(raised.exception))
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer secret-token",
        )

        redirect_response = SimpleNamespace(
            is_redirect=True,
            is_permanent_redirect=False,
            ok=False,
            status_code=302,
        )
        with patch("rag.rag_service.requests.post", return_value=redirect_response):
            with self.assertRaisesRegex(RagServiceUnavailable, "redirects"):
                embed_query("battery", config, expected_dimensions=2)

    def test_hybrid_retrieval_returns_finite_ranked_citations(self) -> None:
        config = SimpleNamespace(embed_url="http://localhost/embed", api_key=None)
        with patch("rag.rag_service.embed_query", return_value=np.array([1.0, 0.0])):
            results = retrieve_top_k("lithium inventory", _index(), config, top_k=2)
        self.assertEqual([item["doc_name"] for item in results], ["P01.md", "P02.md"])
        self.assertTrue(all(np.isfinite(item["score"]) for item in results))
        self.assertTrue(all(item["page_content"] for item in results))
        self.assertIn("dense_score", results[0])
        self.assertIn("bm25_score", results[0])

    def test_generation_contract_includes_context_history_and_cited_sources(self) -> None:
        config = SimpleNamespace(
            vllm_url="http://localhost/v1",
            model_name="test-model",
            api_key=None,
        )
        chunk = _chunk("P06.md", "SEI growth consumes cyclable lithium.")
        context = {"predicted_soh": 91.0, "saved_label_soh": None}
        with patch(
            "rag.rag_service._post_json",
            return_value={"choices": [{"message": {"content": "Evidence (P06.md)."}}]},
        ) as post:
            answer = generate_answer(
                "Explain the result",
                [chunk],
                context,
                [{"role": "user", "content": "Earlier question"}],
                config,
            )
        self.assertEqual(answer, "Evidence (P06.md).")
        payload = post.call_args.args[1]
        self.assertEqual(payload["model"], "test-model")
        self.assertFalse(payload["stream"])
        system = payload["messages"][0]["content"]
        self.assertIn("P06.md", system)
        self.assertIn('"saved_label_soh": null', system)
        self.assertIn("cannot establish what occurred in this cell", system)
        self.assertEqual(payload["messages"][1]["content"], "Earlier question")
        self.assertEqual(payload["messages"][2]["content"], "Explain the result")

    def test_generation_rejects_malformed_response(self) -> None:
        config = SimpleNamespace(
            vllm_url="http://localhost/v1",
            model_name="test-model",
            api_key=None,
        )
        with patch("rag.rag_service._post_json", return_value={"choices": []}):
            with self.assertRaisesRegex(RagResponseError, "choices"):
                generate_answer("question", [], None, [], config)

    def test_new_rag_conversation_does_not_clear_battery_state(self) -> None:
        state = {
            "battery_result": {"predicted_soh_percent": 90.0},
            "rag_battery_context": {"predicted_soh": 90.0},
            "rag_last_context": {"predicted_soh": 90.0},
            "rag_messages": [{"role": "user", "content": "old"}],
            "rag_retrieved_chunks": [{"doc_name": "P01.md"}],
        }
        reset_rag_conversation(state)
        self.assertEqual(state["rag_messages"], [])
        self.assertEqual(state["rag_retrieved_chunks"], [])
        self.assertIn("battery_result", state)
        self.assertIn("rag_battery_context", state)
        self.assertIn("rag_last_context", state)


if __name__ == "__main__":
    unittest.main()
