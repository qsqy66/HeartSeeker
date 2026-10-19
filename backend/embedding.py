"""DashScope (Qwen) embedding + local LanceDB vector store.

The embedding provider is always DashScope (Qwen), because DeepSeek has no
embedding endpoint. Vectors are computed remotely but stored locally in a
LanceDB table under APP_HOME/vectors, so retrieval and provenance stay on the
user's machine. Every vector row carries page/bbox/section metadata so a recall
result can always be traced back to its exact spot in the source PDF.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any, Optional

DASHSCOPE_EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
DEFAULT_EMBED_MODEL = "text-embedding-v3"
BATCH_SIZE = 10  # DashScope compatible-mode caps batched inputs at 10 per request.


class EmbeddingError(RuntimeError):
    """Raised when the remote embedding service is unavailable or misbehaves."""


def embed_texts(
    texts: list[str],
    api_key: str,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[list[float]]:
    """Embed a list of texts via DashScope, preserving input order."""
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            DASHSCOPE_EMBED_URL,
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise EmbeddingError("DashScope 嵌入服务暂不可用") from error
        rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        vectors.extend(row["embedding"] for row in rows)
    if len(vectors) != len(texts):
        raise EmbeddingError("嵌入返回数量与输入不一致")
    return vectors


def normalize(vector: list[float]) -> list[float]:
    """Unit-normalize a vector so L2 distance behaves like cosine similarity."""
    norm = math.sqrt(sum(float(x) * float(x) for x in vector)) or 1.0
    return [float(x) / norm for x in vector]


def _connect(vectors_dir):
    import lancedb  # lazy import: app runs fine (FTS-only) without lancedb

    return lancedb.connect(str(vectors_dir))


def upsert_chunks(vectors_dir, rows: list[dict[str, Any]]) -> None:
    """Insert or append vector rows. Schema is inferred from the first batch."""
    if not rows:
        return
    db = _connect(vectors_dir)
    if "chunks" in db.table_names():
        db.open_table("chunks").add(rows)
    else:
        db.create_table("chunks", data=rows)


def search_chunks(vectors_dir, paper_id: int, query_vector: list[float], top_k: int = 8) -> list[dict[str, Any]]:
    db = _connect(vectors_dir)
    if "chunks" not in db.table_names():
        return []
    table = db.open_table("chunks")
    try:
        return (
            table.search(query_vector)
            .where(f"paper_id = {int(paper_id)}", prefilter=True)
            .limit(top_k)
            .to_list()
        )
    except Exception:
        # Fall back to an unfiltered scan when the where clause is unsupported.
        results = table.search(query_vector).limit(top_k).to_list()
        return [row for row in results if row.get("paper_id") == int(paper_id)][:top_k]


def delete_paper_vectors(vectors_dir, paper_id: int) -> None:
    db = _connect(vectors_dir)
    if "chunks" in db.table_names():
        db.open_table("chunks").delete(f"paper_id = {int(paper_id)}")
