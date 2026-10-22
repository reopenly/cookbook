"""Report four illustrative searches to a dataset you choose.

    Set REOPENLY_PROJECT=owner/project and REOPENLY_DATASET_ID=42 in .env, then:
    uv run --env-file .env --locked embeddings/03_report_searches.py --report

Five billable embedding requests and four persistent feedback events. Reporting itself is free.
Use a disposable embedding dataset. Review results in Search feedback before adding training pairs.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal
from uuid import uuid4

import httpx
import numpy as np
from openai import OpenAI

BASE_MODEL = "ReOpenly/Qwen3-Embedding-0.6B"
BASES = {BASE_MODEL, "ReOpenly/Qwen3-Embedding-4B"}
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "help_center.json"


def main(argv: list[str] | None = None) -> None:
    """Report four demo searches after verifying their destination dataset."""
    parse_args(argv)  # Reporting requires an explicit --report flag.
    project, dataset_id = feedback_settings()
    model = os.environ.get("REOPENLY_EMBEDDING_MODEL", BASE_MODEL)
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    passages = data["passages"]

    with client_for(model) as client, feedback_client(project) as api:
        # Check the destination before any embedding charge.
        validate_dataset(api, dataset_id)
        document_vectors = embed(
            client, [passage["text"] for passage in passages], "document", model
        )
        for query in data["searches"]:
            question_vector = embed(client, [query], "query", model)[0]
            scores = document_vectors @ question_vector
            event = build_event(model, query, passages, scores)
            # If delivery is uncertain, retain this exact id AND payload before retrying it.
            api.post(
                f"/datasets/{dataset_id}/retrieval-events", json=event
            ).raise_for_status()
            print(f"reported: {query}")
    print(
        f"\nOpen Search feedback for dataset {dataset_id}, keep what answered, then add the training pairs."
    )


def build_event(
    model: str, query: str, passages: list[dict[str, str]], scores: np.ndarray
) -> dict[str, object]:
    """Attach the five closest passages to one uniquely identified search."""
    results = []
    top_indices = np.argsort(-scores, stable=True)[:5]  # Highest similarity first.
    for rank, passage_index in enumerate(top_indices, start=1):
        passage = passages[passage_index]
        results.append(
            {
                "rank": rank,
                "text": passage["text"],
                "metadata": {
                    "score": round(float(scores[passage_index]), 4),
                    "title": passage["title"],
                    "external_id": passage["id"],
                },
            }
        )
    return {
        "event_id": f"cookbook-{uuid4().hex}",
        "model": model,
        "query": query,
        "metadata": {
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "search_target": "help-center",
        },
        "results": results,
    }


def validate_dataset(api: httpx.Client, dataset_id: int) -> None:
    """Refuse missing, archived or non-embedding destinations before paid calls."""
    datasets = api.get("/datasets").raise_for_status().json()["data"]
    active = any(
        dataset["id"] == dataset_id
        and dataset.get("kind") == "embedding"
        and not dataset.get("archived_at")
        for dataset in datasets
    )
    if not active:
        raise ValueError("Choose an active embedding dataset in REOPENLY_PROJECT.")


# Supporting helpers: explicit opt-in, settings and response checks.


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line options before constructing API clients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="allow four persistent demo feedback events",
    )
    args = parser.parse_args(argv)
    if not args.report:
        parser.error(
            "Pass --report to allow billable embeddings and four persistent feedback events."
        )
    return args


def feedback_client(project: str) -> httpx.Client:
    """Construct the feedback client after model-client settings validation."""
    base = "https://reopenly.com/api"
    return httpx.Client(
        base_url=f"{base}/v1/projects/{project}",
        headers={"Authorization": f"Bearer {os.environ['REOPENLY_API_KEY'].strip()}"},
        timeout=300.0,
    )


def feedback_settings() -> tuple[str, int]:
    """Return the selected project and positive feedback dataset id."""
    project = project_path()
    raw_id = os.environ.get("REOPENLY_DATASET_ID", "")
    if not raw_id.isascii() or not raw_id.isdecimal() or int(raw_id) < 1:
        raise ValueError(
            "Set REOPENLY_DATASET_ID to the positive dataset id shown on Search feedback."
        )
    return project, int(raw_id)


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
