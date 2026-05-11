from __future__ import annotations

import os
import re
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:  # pragma: no cover
    import requests

OPENAI_API_BASE = "https://api.openai.com/v1"
_OPENAI_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_DEFAULT_BATCH_REQUEST_OVERHEAD_TOKENS = 64


def _resolve_api_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI Batch requires an API key. Set OPENAI_API_KEY or pass api_key explicitly."
        )
    return key


def _normalize_api_base(api_base: str) -> str:
    return api_base.rstrip("/")


def _validate_batch_id(batch_id: str) -> None:
    if not _OPENAI_BATCH_ID_RE.fullmatch(batch_id):
        raise RuntimeError(
            "Invalid OpenAI batch id. Use the exact id returned by submit-openai-batch "
            "or stored under batch.id in the submission JSON; do not pass placeholders "
            f"such as 'batch_...'. Got: {batch_id!r}"
        )


def estimate_batch_request_input_tokens(request: dict[str, Any]) -> int:
    # OpenAI enforces an organization-level enqueued-token limit before a Batch
    # starts. Tokenizer-perfect accounting is unnecessary; a stable upper-ish
    # estimate keeps shards comfortably below the cap.
    body = request.get("body")
    serialized_body = json_dumps_compact(body if isinstance(body, dict) else request)
    return math.ceil(len(serialized_body) / 4) + _DEFAULT_BATCH_REQUEST_OVERHEAD_TOKENS


def json_dumps_compact(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def shard_batch_requests(
    requests: list[dict[str, Any]],
    *,
    max_estimated_input_tokens: int,
    max_requests: int | None = None,
) -> list[dict[str, Any]]:
    if max_estimated_input_tokens < 1:
        raise ValueError(f"max_estimated_input_tokens must be >= 1; got {max_estimated_input_tokens}")
    if max_requests is not None and max_requests < 1:
        raise ValueError(f"max_requests must be >= 1; got {max_requests}")

    shards: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    current_estimated_tokens = 0

    def flush() -> None:
        nonlocal current_rows, current_estimated_tokens
        if current_rows:
            shards.append(
                {
                    "rows": current_rows,
                    "request_count": len(current_rows),
                    "estimated_input_tokens": current_estimated_tokens,
                }
            )
        current_rows = []
        current_estimated_tokens = 0

    for request in requests:
        estimated_tokens = estimate_batch_request_input_tokens(request)
        would_exceed_tokens = current_rows and current_estimated_tokens + estimated_tokens > max_estimated_input_tokens
        would_exceed_count = max_requests is not None and len(current_rows) >= max_requests
        if would_exceed_tokens or would_exceed_count:
            flush()
        current_rows.append(request)
        current_estimated_tokens += estimated_tokens
    flush()
    return shards


def _json_or_raise(response: requests.Response, context: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - API should return JSON
        raise RuntimeError(
            f"{context} failed with HTTP {response.status_code}: {response.text[:500]}"
        ) from exc
    if not response.ok:
        raise RuntimeError(f"{context} failed with HTTP {response.status_code}: {payload}")
    return dict(payload)


def upload_batch_input_file(
    path: Path,
    *,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> dict[str, Any]:
    import requests

    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    with path.open("rb") as handle:
        response = requests.post(
            f"{base}/files",
            headers={"Authorization": f"Bearer {key}"},
            data={"purpose": "batch"},
            files={"file": (path.name, handle, "application/jsonl")},
            timeout=300,
        )
    return _json_or_raise(response, f"Upload batch input file {path}")


def create_batch(
    *,
    input_file_id: str,
    endpoint: str,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> dict[str, Any]:
    import requests

    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    response = requests.post(
        f"{base}/batches",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    return _json_or_raise(response, "Create batch")


def retrieve_batch(
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> dict[str, Any]:
    import requests

    _validate_batch_id(batch_id)
    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    response = requests.get(
        f"{base}/batches/{batch_id}",
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    )
    return _json_or_raise(response, f"Retrieve batch {batch_id}")


def download_file_content(
    file_id: str,
    output_path: Path,
    *,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> Path:
    import requests

    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    response = requests.get(
        f"{base}/files/{file_id}/content",
        headers={"Authorization": f"Bearer {key}"},
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(
            f"Download file content {file_id} failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path
