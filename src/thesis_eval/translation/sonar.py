from __future__ import annotations

from pathlib import Path


def prewarm_sonar_text_encoder(cache_dir: Path | None = None) -> dict[str, object]:
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    except ImportError as exc:
        raise RuntimeError(
            "SONAR prewarm requires `uv sync --extra qc` and Python 3.12. "
            "Install libsndfile on macOS if fairseq2 complains: `brew install libsndfile`."
        ) from exc

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    embedder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
    )
    embeddings = embedder.predict(["SONAR cache prewarm."], source_lang="eng_Latn")
    return {
        "encoder": "text_sonar_basic_encoder",
        "tokenizer": "text_sonar_basic_encoder",
        "embedding_shape": list(embeddings.shape),
    }
