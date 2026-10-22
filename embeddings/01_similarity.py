"""Rank passages for one question using two billable embedding requests.

    uv run embeddings/01_similarity.py
    uv run embeddings/01_similarity.py "Your question" "passage one" "passage two"

No training examples are saved. Keys come from REOPENLY_API_KEY.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal

import numpy as np
from openai import OpenAI

BASE_MODEL = "ReOpenly/Qwen3-Embedding-0.6B"
BASES = {BASE_MODEL, "ReOpenly/Qwen3-Embedding-4B"}
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "help_center.json"


def main(argv: list[str] | None = None) -> None:
    """Rank the supplied passages, or the demo passages, for one question."""
    args = parse_args(argv)
    model = os.environ.get("REOPENLY_EMBEDDING_MODEL", BASE_MODEL)
    if args.texts:
        question, *passages = args.texts
    else:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        question = data["searches"][0]
        passages = [passage["text"] for passage in data["passages"][:3]]

    with client_for(model) as client:
        question_vector = embed(client, [question], "query", model)[0]
        document_vectors = embed(client, passages, "document", model)
    # Unit vectors turn the dot product into cosine similarity: larger is closer.
    scores = document_vectors @ question_vector
    print(f"Question: {question}\n")
    for score, text in sorted(zip(scores, passages, strict=True), reverse=True):
        print(f"  {score:.3f}  {text}")


# Supporting helpers: settings and response checks stay local to this example.


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line options before constructing API clients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "texts", nargs="*", help="a question followed by at least one passage"
    )
    args = parser.parse_args(argv)
    if len(args.texts) == 1:
        parser.error(
            "Supply a question and at least one passage, or no arguments for the demo."
        )
    return args


def project_path() -> str:
    """Return the validated owner/project path for a hosted model or feedback."""
    value = os.environ.get("REOPENLY_PROJECT", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value):
        raise ValueError(
            "Set REOPENLY_PROJECT to owner/project, using the two slugs in the app URL."
        )
    return value


def client_for(model: str) -> OpenAI:
    """Validate settings locally before constructing a base or hosted client."""
    key = os.environ.get("REOPENLY_API_KEY", "").strip()
    if not key:
        raise ValueError("Set REOPENLY_API_KEY, or use uv run --env-file .env <file>.")
    base = "https://reopenly.com/api"
    if model in BASES:
        route = "openai/v1"
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
            raise ValueError("Use a listed ReOpenly base id or a hosted model slug.")
        route = f"projects/{project_path()}/openai/v1"
    # A lost response may already have been billed. Never automatically resend it.
    return OpenAI(api_key=key, base_url=f"{base}/{route}", max_retries=0, timeout=300.0)


def embed(
    client: OpenAI,
    texts: list[str],
    input_type: Literal["query", "document"],
    model: str,
) -> np.ndarray:
    """Return finite, normalized vectors in input order."""
    response = client.embeddings.create(
        model=model, input=texts, extra_body={"input_type": input_type}
    )
    rows = sorted(response.data, key=lambda row: row.index)
    if [row.index for row in rows] != list(range(len(texts))):
        raise ValueError("The embeddings response did not match the inputs.")
    vectors = np.array([row.embedding for row in rows], dtype=float)
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("The embeddings response contained invalid vectors.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("The embeddings response contained a zero vector.")
    return vectors / norms


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
