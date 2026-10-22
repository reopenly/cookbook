"""Offline release checks. --live additionally permits paid requests and demo feedback writes."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

from scan_secrets import release_files, scan

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "d203a277f19a8f7d",
    "e3ce9cc5efa9354c",
    "f3f2184b56f946c6",
    "cfc35d18e7332fde",
    "920510199770f4d6",
    "72e8088a05b38ae3",
    "92e96a480708316d",
    "ea3f2c381cdbe297",
    "7f871cbf905f6e0c",
    "5d72436256ada538",
    "1e3bd9448dc9acaa",
    "c70eca6b0f88f44d",
    "531e67a59846756e",
    "763268b8dbdcf327",
    "2086d0f715be4a4d",
    "31d1b3d540ecf852",
    "665e05102ac943a0",
    "e2fde0483fc8158c",
}


def load_notebook(root: Path = ROOT) -> dict[str, Any]:
    """Read the notebook itself, the source of the public teaching example."""
    return json.loads(
        (root / "embeddings/try_embeddings.ipynb").read_text(encoding="utf-8")
    )


def output_issues(notebook: dict[str, Any], *, required: bool = True) -> list[str]:
    """Permit reviewed text and PNG output; refuse active content and saved exceptions."""
    issues = []
    for c in notebook["cells"]:
        if c.get("attachments"):
            issues.append("Notebook contains attachments.")
        for output in c.get("outputs", []):
            kind = output.get("output_type")
            if kind == "stream":
                if output.get("name") != "stdout":
                    issues.append(
                        "Notebook contains stderr output; review it before saving."
                    )
            elif kind in {"display_data", "execute_result"}:
                if not set(output.get("data", {})) <= {"text/plain", "image/png"}:
                    issues.append(
                        "Notebook contains active or unsupported output content."
                    )
            else:
                issues.append("Notebook contains an error or unsupported output type.")
    if required:
        by_id = {c["id"]: c for c in notebook["cells"]}
        for name in [
            "similarity",
            "vector",
            "comparison",
            "hosted-comparison",
            "feedback",
        ]:
            if not by_id.get(name, {}).get("outputs"):
                issues.append(
                    f"Notebook needs saved {name} output from a reviewed run."
                )
        for name in ["hosted-comparison", "feedback"]:
            outputs = by_id.get(name, {}).get("outputs", [])
            if any(
                "skipped" in str(output.get("text", "")).lower() for output in outputs
            ):
                issues.append(
                    f"Notebook needs executed {name} output, not a skipped section."
                )
        if not any(
            o.get("data", {}).get("image/png")
            for o in by_id.get("vector", {}).get("outputs", [])
        ):
            issues.append("Notebook needs a saved PNG chart.")
    return issues


def notebook_issues(notebook: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Validate notebook structure, minimum Python syntax, dependency pins and saved outputs."""
    import nbformat

    try:
        nbformat.validate(nbformat.from_dict(notebook))
    except nbformat.ValidationError:
        return ["Notebook structure is invalid."]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    issues = []
    install = next((c for c in notebook["cells"] if c.get("id") == "install"), {})
    requirements = None
    try:
        for node in ast.parse("".join(install.get("source", []))).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "requirements"
                for target in node.targets
            ):
                requirements = ast.literal_eval(node.value)
    except (SyntaxError, ValueError):
        pass  # Syntax errors are reported per cell below as well.
    if requirements != project["project"]["dependencies"]:
        issues.append("Notebook install cell must match pyproject.toml dependencies.")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%matplotlib inline\n"):
            source = source.partition("\n")[2]
        try:
            ast.parse(source, feature_version=(3, 12))
        except SyntaxError:
            issues.append(
                f"Notebook cell {cell.get('id', '?')}: Python 3.12 syntax error."
            )
    return issues + output_issues(notebook)


def named(text: str) -> set[str]:
    """Find prohibited release names without storing those names in this checker."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams = set(words) | {a + " " + b for a, b in zip(words, words[1:])}
    return {
        g for g in grams if hashlib.sha256(g.encode()).hexdigest()[:16] in FORBIDDEN
    }


def static_checks(root: Path = ROOT) -> list[str]:
    """Check release source, sensitive content, and the standalone notebook."""
    failures = scan(root)
    for path in release_files(root):
        if path.suffix == ".py":
            try:
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 12),
                )
            except SyntaxError:
                failures.append(f"{path.relative_to(root)}: Python syntax error")
        if path.suffix not in {".lock", ".png"}:
            if named(path.read_text(encoding="utf-8", errors="ignore")):
                failures.append(f"{path.relative_to(root)}: naming rule")
    failures.extend(notebook_issues(load_notebook(root), root))
    return failures


def live_checks() -> None:
    """Run explicitly authorized paid examples and eight persistent feedback writes."""
    from nbclient import NotebookClient
    import nbformat

    # Validate every setting before invoking a script or issuing a request.
    for name in ("REOPENLY_PROJECT", "REOPENLY_DATASET_ID", "REOPENLY_MODEL"):
        if not os.environ.get(name):
            raise ValueError(
                f"Live checks require {name}; choose a disposable embedding dataset and hosted model."
            )
    feedback = runpy.run_path(str(ROOT / "embeddings/03_report_searches.py"))
    feedback["feedback_settings"]()
    comparison = runpy.run_path(str(ROOT / "embeddings/04_compare_before_switching.py"))
    for model in [
        comparison["BASE_MODEL"],
        os.environ["REOPENLY_MODEL"],
        os.environ.get("REOPENLY_EMBEDDING_MODEL", comparison["BASE_MODEL"]),
    ]:
        with comparison["client_for"](model):
            pass
    print(
        "LIVE: embedding requests spend credits; eight demo events will be retained in the chosen dataset.",
        flush=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        for script in sorted((ROOT / "embeddings").glob("0*.py")):
            args = ["--report"] if script.name.startswith("03") else []
            result = subprocess.run(
                [sys.executable, str(script), *args],
                cwd=tmp,
                capture_output=True,
                timeout=1800,
            )
            if result.returncode:
                raise ValueError(
                    f"Live example failed: {script.name}; inspect the selected project before retrying."
                )
            print(f"ok live {script.name}")
        notebook = nbformat.read(ROOT / "embeddings/try_embeddings.ipynb", as_version=4)
        previous = os.environ.get("REOPENLY_REPORT_SEARCHES")
        os.environ["REOPENLY_REPORT_SEARCHES"] = "1"
        try:
            NotebookClient(
                notebook,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": tmp}},
            ).execute()
        finally:
            if previous is None:
                os.environ.pop("REOPENLY_REPORT_SEARCHES", None)
            else:
                os.environ["REOPENLY_REPORT_SEARCHES"] = previous
        print("ok live notebook; execution output was kept in memory only")


def main(argv: list[str] | None = None) -> int:
    """Run offline gates, followed by live checks only when explicitly requested."""
    print(f"Python {sys.version.split()[0]}", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="permit paid requests and eight persistent feedback events",
    )
    args = parser.parse_args(argv)
    failures = static_checks()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    # The test process receives no credentials and denies socket connections.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("REOPENLY_") and k not in {"OPENAI_API_KEY", "UV_ENV_FILE"}
    }
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        env=env,
    )
    if result.returncode:
        return result.returncode
    print(
        "Offline checks passed: source, secrets/history, notebook and regression tests.",
        flush=True,
    )
    if args.live:
        live_checks()
    else:
        print(
            "Live API and hosted-model acceptance were not run. Use uv run --env-file .env --locked scripts/check.py --live with a disposable dataset."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(str(error)) from None
