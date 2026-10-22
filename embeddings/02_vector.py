"""Project six embeddings onto two principal components and save vector.png.

    uv run embeddings/02_vector.py

One billable request. A 2D projection can distort the original distances.
"""

import os
import re
import sys
from typing import Literal

import numpy as np
from openai import OpenAI

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_MODEL = "ReOpenly/Qwen3-Embedding-0.6B"
BASES = {BASE_MODEL, "ReOpenly/Qwen3-Embedding-4B"}

TEXTS = [
    ("refunds", "How long do refunds take to arrive?"),
    ("refunds", "Refunds arrive in 5 to 7 business days."),
    ("passwords", "How do I reset my password?"),
    ("passwords", "Choose Forgot password on the sign-in page."),
    ("shipping", "When will my order ship?"),
    ("shipping", "Orders ship within two business days."),
]


def main() -> None:
    """Embed the sample texts and save their two-dimensional projection."""
    model = os.environ.get("REOPENLY_EMBEDDING_MODEL", BASE_MODEL)
    with client_for(model) as client:
        vectors = embed(client, [text for _, text in TEXTS], "document", model)

    # Center the vectors, then keep the two axes with the most variation (PCA).
    centered = vectors - vectors.mean(axis=0)
    _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)
    points = centered @ principal_axes[:2].T

    colors = {"refunds": "#22c55e", "passwords": "#3b82f6", "shipping": "#f59e0b"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for (topic, text), (x, y) in zip(TEXTS, points, strict=True):
        ax.scatter(x, y, s=80, color=colors[topic])
        ax.annotate(text, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set(
        title="Six texts: a 2D projection (distances may be distorted)",
        xticks=[],
        yticks=[],
    )
    fig.tight_layout()
    fig.savefig("vector.png", dpi=150)
    plt.close(fig)
    for (topic, text), (x, y) in zip(TEXTS, points, strict=True):
        print(f"  {topic:<10} ({x:+.3f}, {y:+.3f})  {text}")
    print("\nSaved vector.png")


# Supporting helpers: settings and response checks stay local to this example.


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
