# Contributing

An example earns its place when a reader can run it in minutes and learn one thing from it.
It must be useful, correct, clear, complete about costs and prerequisites, and honest about results.

## Rules

- **One file, one command.** Use the shared uv project. Keep each example
  runnable without importing another example or a local helper. Show the main workflow before
  supporting helpers. Keys come from `REOPENLY_API_KEY`.
  Examples use only `https://reopenly.com/api`; do not expose local or custom API destinations.
- **Python readability.** Use syntax supported by the minimum Python version, currently 3.12.
  Annotate function parameters and return types, and give each function a concise purpose docstring.
  Prefer descriptive intermediate variables and explain unfamiliar numerical steps.
- **Lock dependencies.** Use Python 3.12 or newer. Runtime dependencies belong in `pyproject.toml`;
  test and notebook tooling belongs in its `dev` dependency group. Use `uv add "library==version"`
  or `uv add --dev "library==version"`, and commit `pyproject.toml` and `uv.lock` together.
  Run `uv sync --locked` after checking out dependency changes. The standalone notebook's
  install cell's `requirements` list must match the project's runtime dependencies; update it
  alongside dependency changes. Setup skips installation when the pinned versions are present.
  Otherwise it uses uv with the kernel's interpreter, following
  [uv's environment guidance](https://docs.astral.sh/uv/pip/environments/), or falls back to that
  interpreter's pip when uv is unavailable. Without either installer it gives an actionable error.
  Restart the kernel after installing or changing versions. Tests execute the unchanged setup
  cell with installation mocked or forbidden; never replace it with `pass`.
  Test and audit updates before merging.
  Do not add inline script dependency metadata or per-script locks: every repo command must use
  the shared project environment.
- **Readable notebooks.** Edit `embeddings/try_embeddings.ipynb` directly; the notebook is the
  source of the lesson, with no generator or second copy of its teaching code. Keep stable cell ids,
  short meaningful steps, one setup and an embedding helper. Finish ranking, plotting and evaluation
  before optional hosted comparison and feedback writes.
  Inline sample data and helpers so a downloaded notebook runs without the checkout. Do not import
  numbered scripts or require local helper modules; shared behavior is covered by common tests.
  Explain what to notice after each output, and offer one bounded question-editing exercise with
  its extra request cost stated.
  Use descriptive names and readable Python rather than compressing code to meet a line count.
  Keep implementation asides in brief Markdown notes (`> **Note:** ...`); costs and persistent
  writes belong before the affected action. These conventions follow the
  [NumPy tutorial style guide](https://numpy.org/numpy-tutorials/tutorial-style-guide/)
  and [Jupyter's teaching guidance](https://jupyter4edu.github.io/jupyter-edu-book/notebooks-in-teaching-and-learning.html).
- **Reviewed outputs.** Commit the main notebook with its ranking, PNG chart, metrics, hosted
  comparison and reported-search outputs. Skipped optional sections do not satisfy this requirement.
  After code edits, clear stale outputs and execution counts, restart the kernel, rerun, and review
  before saving. Readers' default run makes two billable base embedding requests. Capturing every
  section makes five embedding requests and stores four feedback events; use an existing hosted
  embedding model and a separate demo dataset, with explicit operator approval for that capture.
  Use ReOpenly's cloud API, keep keys in the environment, and record model and execution provenance
  in notebook metadata. Illustrative outputs do not establish a training gain or billing correctness.
  Saved output may contain only reviewed illustrative text and PNG images; errors, stderr,
  attachments, active HTML and credentials fail the gate. Do not publish mocked results as real outputs.
- **Small, illustrative data.** Larger data needs its source. Never add customer queries or secrets.
- **ReOpenly only.** Name only ReOpenly, the OpenAI SDK, imported libraries and required development
  tools. The naming check examines release content, excluding dependency-lock metadata.
- **Honest measurements.** Use your own held-out data; never promise a quality gain.
- **Regression tests.** Guards need a failing case. Test API behavior offline, including hosted
  routing, ranking metrics, preflight errors and uncertain-delivery handling.

## Before committing

Before initial publication, work and commit directly on `main`. At the owner's finalization step,
squash the preparation history into one initial commit before the first push.

After publication, every change must come from a branch and pull request, including documentation,
dependency updates and fixes. Do not commit or push changes directly to the default branch.
Merge through the PR after the required offline checks pass.

```text
uv run --locked scripts/check.py
```

This command needs no API key and makes no API requests. It checks source syntax, release-file
names, credential patterns in release files and all locally reachable Git history, notebook structure, Python 3.12 syntax, install-cell dependency pins,
saved-output policy, and offline tests. The tests use synthetic data and deny socket connections.
Ignored files such as `.venv` are excluded; forced tracked secret files still fail.
Historical filenames are checked in every reachable Git tree independently of blob content, so
removed private filenames are still checked when another file shares the same bytes.
The scanner reports rule names and file/object identifiers, never matched secret values. It detects
common credential patterns, not every possible secret; review new data and outputs before committing.

The same gate is configured in CI on Python 3.12 and 3.14 for Linux x64, Windows x64, macOS Apple
Silicon and macOS Intel, with read-only repository permissions and no application credentials.
CI installs wheels only, so an absent platform wheel fails without requiring a native compiler.
Keep runner-dependent temporary-directory settings in the check step's `env`: the `runner`
context is unavailable in job-level `env`. See [GitHub's context availability
reference](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability).
Native platform acceptance requires the corresponding CI jobs to pass. Tests exercise UTF-8 text,
paths containing spaces and non-ASCII characters, and the standalone notebook kernel. CI does not
establish hosted or production acceptance. Keep file encodings explicit and subprocess arguments
as lists. `make check` remains an optional shortcut where `make` is installed.
The kernel test builds the platform font cache before teaching cells run, so font-discovery startup
messages do not become notebook output; the saved-output policy still rejects stderr.

## Optional live acceptance

Use a **disposable embedding dataset** and hosted embedding model. Set `.env` with
`REOPENLY_API_KEY`, `REOPENLY_PROJECT`, `REOPENLY_DATASET_ID` and `REOPENLY_MODEL`, then explicitly run:

```text
uv run --env-file .env --locked scripts/check.py --live
```

Offline checks must pass first. Live checks spend embedding credits, run the hosted comparison,
and store **eight demo feedback events** in the chosen dataset (four from example 03 and four from
the notebook). A rerun adds new events; there is no automatic cleanup or replay. Review the selected
dataset before retrying a partial failure. Notebook outputs stay in memory and are not written back.

## Publication

Use the public author/committer identity you intend to expose before creating commits. Deleting a
secret from the latest tree does not remove it from history. Enable private vulnerability reporting
on the destination repository before publication. Once the initial publication checks have run,
apply [the prepared branch rules](.github/branch-ruleset.json) to the destination repository's
default branch. They require PRs and all eight OS/Python checks, with no bypass actors, and prevent force
pushes and deletion. The file prepares the policy; it does not enable remote enforcement by itself.
