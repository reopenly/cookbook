"""Compare retrieval on held-out questions before switching models.

    uv run embeddings/04_compare_before_switching.py
    Set REOPENLY_PROJECT=owner/project in .env for your hosted model, then:
    uv run --env-file .env --locked embeddings/04_compare_before_switching.py --model your-model

Two billable requests per model. Ten illustrative questions show the method; use your own held-out
searches to decide. Nothing here trains a model or promises a quality gain.
"""

import argparse
from contextlib import ExitStack
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
    """Evaluate the base and an optional hosted model on the same held-out questions."""
    args = parse_args(argv)
    models = [BASE_MODEL] + ([args.model] if args.model else [])
    # Validate every model's settings before the first paid request.
    with ExitStack() as stack:
        clients = [stack.enter_context(client_for(model)) for model in models]
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        print(
            f"{len(data['held_out'])} held-out searches over {len(data['passages'])} passages\n"
        )
        print(
            f"  {'model':<34} {'hit@1':>6} {'MRR':>6}  rank of the right passage, per search"
        )
        for model, client in zip(models, clients, strict=True):
            result = evaluate(model, data["passages"], data["held_out"], client)
            print(
                f"  {model:<34} {result['hit@1']:>6.2f} {result['mrr']:>6.3f}  {result['ranks']}"
            )
            if model == BASE_MODEL and result["hit@1"] == 1.0:
                print(
                    "  The base ranks every answer first here; your own held-out searches decide whether training helps."
                )
    if not args.model:
        print(
            "\nAfter training on reviewed searches and hosting the result, compare with --model <its slug>."
        )


def evaluate(
    model: str,
    passages: list[dict[str, str]],
    held_out: list[dict[str, str]],
    client: OpenAI,
) -> dict[str, float | list[int]]:
    """Rank the expected answer for each question, then compute hit@1 and MRR."""
    passage_ids = [passage["id"] for passage in passages]
    if (
        not passages
        or not held_out
        or len(passage_ids) != len(set(passage_ids))
        or any(test_case["answer"] not in passage_ids for test_case in held_out)
    ):
        raise ValueError(
            "Evaluation needs unique passage ids and held-out answers matching those ids."
        )
    document_vectors = embed(
        client, [passage["text"] for passage in passages], "document", model
    )
    question_vectors = embed(
        client, [test_case["query"] for test_case in held_out], "query", model
    )
    ranks = []
    for test_case, question_vector in zip(held_out, question_vectors, strict=True):
        scores = document_vectors @ question_vector
        ranked_ids = [passage_ids[index] for index in np.argsort(-scores, stable=True)]
        ranks.append(ranked_ids.index(test_case["answer"]) + 1)  # Ranks start at 1.
    answer_ranks = np.array(ranks)
    # hit@1 counts rank-1 answers; MRR gives rank 1 a score of 1, rank 2 a score of 1/2.
    return {
        "hit@1": float(np.mean(answer_ranks == 1)),
        "mrr": float(np.mean(1 / answer_ranks)),
        "ranks": answer_ranks.tolist(),
    }


# Supporting helpers: settings and response checks stay local to this example.


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line options before constructing API clients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("REOPENLY_MODEL", ""),
        help="hosted model slug (or REOPENLY_MODEL)",
    )
    return parser.parse_args(argv)


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
