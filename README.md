# <img src="assets/reopenly-logo.svg" alt="" width="40" height="40" align="top"> ReOpenly cookbook

Runnable examples for [ReOpenly](https://reopenly.com): fine-tune open models on your own data and
serve them through an OpenAI-compatible API. Each example runs independently with one command.
All examples call ReOpenly's cloud API at `https://reopenly.com/api`.

## Start in five minutes

You need Python **3.12 or newer** and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The commands below work in PowerShell or Command Prompt on Windows and in a terminal on macOS or
Linux. They require no environment activation, `make`, or Unix shell. Git is needed to clone the
repository and run its release checks.

From the repository root, install the locked environment once:

```text
uv sync --locked
```

All examples and checks use this project's `.venv`, `pyproject.toml` and shared `uv.lock`.

1. Sign in and open **profile menu → [API keys](https://reopenly.com/app/api-keys)**.
   Create a key and copy its full value; it is shown only once.
   See the [API-key creation guide](https://reopenly.com/docs/getting-started/authentication/#create-an-account-api-key).
   New accounts start with free credits.
2. Copy `.env.example` to `.env` and enter the key locally. Never commit that file.
3. Run the first example:

   ```text
   uv run --env-file .env --locked embeddings/01_similarity.py
   ```

Alternatively, set `REOPENLY_API_KEY` in your shell and omit `--env-file .env`:

| Shell | Set the key for this session |
| --- | --- |
| PowerShell | `$env:REOPENLY_API_KEY = "your-key"` |
| Command Prompt | `set "REOPENLY_API_KEY=your-key"` |
| macOS/Linux shell | `export REOPENLY_API_KEY="your-key"` |

Run commands from the repository root; uv loads the project environment automatically.

## Platform checks

Run the credential-free check on any OS:

```text
uv run --locked scripts/check.py
```

The CI matrix is configured for Linux x64, Windows x64, macOS Apple Silicon and macOS Intel on
Python 3.12 and 3.14. Each job installs the locked dependencies from wheels and runs all examples
and the notebook with mocked API calls. Native macOS and Windows execution is accepted only after
their CI jobs pass; the matrix configuration alone is not a test result. Python versions between
these endpoints and other OS versions or CPU architectures are not separately qualified.

## Embeddings: try this first

Three things to try before training, then a comparison before switching models. The default is the
public base `ReOpenly/Qwen3-Embedding-0.6B`; the small help center in `data/` is illustrative.

| Example | What it shows | Requests and requirements |
| --- | --- | --- |
| [`01_similarity.py`](embeddings/01_similarity.py) | Rank three passages for a question | two embedding requests; API key |
| [`02_vector.py`](embeddings/02_vector.py) | Plot six texts in 2D; projection can distort distances | one embedding request; API key |
| [`03_report_searches.py`](embeddings/03_report_searches.py) | Report searches for owner review | five embedding requests and four stored events; explicit project, dataset and `--report` |
| [`04_compare_before_switching.py`](embeddings/04_compare_before_switching.py) | Compare hit@1 and MRR on held-out searches | two embedding requests per model; hosted comparison also needs its project |
| [`try_embeddings.ipynb`](embeddings/try_embeddings.ipynb) | A compact version of all four, with saved example outputs and a chart | two embedding requests by default; API key to rerun; optional reporting is disabled |

**Embedding requests spend credits.** Reporting itself is free. Nothing here starts training.
A rerun of reporting adds new events to the dataset; use a disposable embedding dataset for the demo.
These events are unreviewed feedback until you review and promote useful results into training pairs.

The notebook's saved ranking, chart, metrics, hosted comparison and reporting outputs come from
a real ReOpenly cloud run. That capture made five embedding requests and stored four synthetic
feedback events in a separate demo dataset. Four illustrative questions do not establish a
training gain. Capture provenance is recorded in notebook metadata.
You can download the notebook on its own: its data and helpers are included. Its main
path covers ranking, visualization and evaluation, followed by optional comparison and reporting.
Run its setup cell first. It reuses the installed dependency versions, or installs the required
versions into the selected kernel with uv (falling back to pip when uv is unavailable).

For feedback, find the owner and project slugs in the app URL, and the dataset id on Search feedback.
Set `REOPENLY_PROJECT=owner/project` and `REOPENLY_DATASET_ID=42` in `.env`, then run:

```text
uv run --env-file .env --locked embeddings/03_report_searches.py --report
```

After training on reviewed searches and hosting the result, compare it with the base:

```text
uv run --env-file .env --locked embeddings/04_compare_before_switching.py --model your-hosted-model
```

Use searches training never saw. Ten illustrative questions show the method; your own held-out
searches decide whether training helps. No example promises a quality gain.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `REOPENLY_API_KEY` | required | account API key; keep it on your server or local machine |
| `REOPENLY_PROJECT` | required for hosted models and reporting | `owner/project`, using the two slugs in the app URL |
| `REOPENLY_DATASET_ID` | required for reporting | positive id of the chosen active embedding dataset |
| `REOPENLY_EMBEDDING_MODEL` | `ReOpenly/Qwen3-Embedding-0.6B` | base id or hosted slug for examples 01–03 |
| `REOPENLY_MODEL` | empty | hosted model for example 04 and the notebook comparison; `--model` overrides it |
| `REOPENLY_REPORT_SEARCHES` | `0` | notebook only: `1` explicitly enables four persistent feedback events |

All clients disable automatic retries and use a 300-second request timeout. A lost response may
already have been billed: inspect the outcome before deciding to send another request. For an
uncertain feedback delivery, preserve the exact event id and payload before retrying it.

Your queries and passages are transmitted to the configured API; feedback text is retained.
Replace the sample data only with text you may share. Notebook outputs may contain your searches:
clear them before sharing. Never put an API key in browser code, a notebook cell, or a public issue.

## Use with an AI coding agent

Try [ReOpenly skills](https://github.com/reopenly/skills) to import Q&A datasets,
fine-tune and deploy text models, and run chat inference with your agent. Skills are optional;
see their repository for installation instructions. Training and inference can spend credits.

## More

The docs: [reopenly.com/docs](https://reopenly.com/docs/). Nonsensitive questions and ideas can go in
issues. For vulnerabilities, follow [SECURITY.md](SECURITY.md).
Contributions and checks: [CONTRIBUTING.md](CONTRIBUTING.md).
