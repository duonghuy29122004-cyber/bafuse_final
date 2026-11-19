from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class RagServiceUnavailable(RuntimeError):
    pass


class RagResponseError(ValueError):
    pass


def _configured_endpoint(environ: dict[str, str] | None = None) -> str | None:
    values = os.environ if environ is None else environ
    endpoint = values.get("BAFUSE_RAG_ENDPOINT", "").strip()
    if not endpoint:
        return None
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RagServiceUnavailable("RAG endpoint must be an HTTP(S) URL.")
    if parsed.hostname.casefold() not in {"localhost", "127.0.0.1", "::1"}:
        raise RagServiceUnavailable(
            "For data safety, the RAG endpoint must run on this computer (localhost)."
        )
    return endpoint.rstrip("/")


def rag_service_available(environ: dict[str, str] | None = None) -> bool:
    try:
        return _configured_endpoint(environ) is not None
    except RagServiceUnavailable:
        return False


def build_rag_request(
    question: str, battery_context: dict[str, Any] | None
) -> dict[str, Any]:
    cleaned = str(question).strip()
    if not cleaned:
        raise ValueError("A non-empty RAG question is required.")
    return {
        "question": cleaned,
        "battery_context": dict(battery_context or {}),
    }


def ask_rag(
    question: str,
    battery_context: dict[str, Any] | None,
    *,
    environ: dict[str, str] | None = None,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    endpoint = _configured_endpoint(environ)
    if endpoint is None:
        raise RagServiceUnavailable("RAG service not connected")
    url = endpoint if endpoint.endswith("/rag/explain") else endpoint + "/rag/explain"
    values = os.environ if environ is None else environ
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = values.get("BAFUSE_RAG_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(
            build_rag_request(question, battery_context),
            allow_nan=False,
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagServiceUnavailable(f"RAG request failed: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        raise RagResponseError("RAG response must contain an answer string.")
    documents = payload.get("retrieved_documents", [])
    citations = payload.get("citations", [])
    if not isinstance(documents, list) or not isinstance(citations, list):
        raise RagResponseError("RAG response documents and citations must be lists.")
    return {
        "answer": payload["answer"],
        "retrieved_documents": documents,
        "citations": citations,
    }
