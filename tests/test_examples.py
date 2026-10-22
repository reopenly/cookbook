"""Offline behavior and release-guard regressions; real socket connections are denied."""

import contextlib
import hashlib
import io
from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import httpx
import httpx2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check import load_notebook, output_issues
import check
import scan_secrets


def sample_outputs(notebook):
    """Small synthetic fixtures for output-policy tests, never published model results."""
    for index, c in enumerate(notebook["cells"], start=1):
        if c["id"] in {"similarity", "comparison", "hosted-comparison", "feedback"}:
            c.update(
                execution_count=index,
                outputs=[
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": "Synthetic test fixture\n",
                    }
                ],
            )
        elif c["id"] == "vector":
            c.update(
                execution_count=index,
                outputs=[
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6fk0AAAAASUVORK5CYII=",
                            "text/plain": "Synthetic test fixture",
                        },
                    }
                ],
            )
    return notebook


class MockAPI:
    def __init__(self):
        self.requests = []
        self.events = []
        self.vector_override = None
        self.timeout = False
        self.datasets = [
            {"id": 1, "kind": "embedding"},
            {"id": 22, "kind": "embedding"},
        ]
        data = json.loads((ROOT / "data/help_center.json").read_text(encoding="utf-8"))
        ids = {p["id"]: i for i, p in enumerate(data["passages"])}
        self.lookup = {p["text"]: i for i, p in enumerate(data["passages"])}
        self.lookup.update({h["query"]: ids[h["answer"]] for h in data["held_out"]})

    def send(self, client, request, **kwargs):
        self.requests.append(request)
        module = httpx2 if isinstance(client, httpx2.Client) else httpx
        if self.timeout:
            raise module.ReadTimeout("Simulated lost reply", request=request)
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            rows = []
            for i, text in enumerate(body["input"]):
                vector = [0.0] * 16
                vector[
                    self.lookup.get(
                        text, int(hashlib.sha256(text.encode()).hexdigest(), 16) % 16
                    )
                ] = 1.0
                rows.append({"object": "embedding", "index": i, "embedding": vector})
            if self.vector_override is not None:
                rows = self.vector_override
            return module.Response(
                200,
                request=request,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": rows[::-1],
                    "usage": {"prompt_tokens": len(rows), "total_tokens": len(rows)},
                },
            )
        if request.url.path.endswith("/datasets"):
            return module.Response(200, request=request, json={"data": self.datasets})
        if request.url.path.endswith("/retrieval-events"):
            body = json.loads(request.content)
            self.events.append(body)
            return module.Response(
                201, request=request, json={"event_id": body["event_id"]}
            )
        raise AssertionError(f"Unexpected API path: {request.url.path}")

    def install(self, stack):
        def send(client, request, **kwargs):
            return self.send(client, request, **kwargs)

        stack.enter_context(patch.object(httpx.Client, "send", send))
        stack.enter_context(patch.object(httpx2.Client, "send", send))
        stack.enter_context(
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("Real networking is forbidden"),
            )
        )
        stack.enter_context(
            patch.object(
                socket.socket,
                "connect_ex",
                side_effect=AssertionError("Real networking is forbidden"),
            )
        )


class Examples(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        temp_root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.temp = temp_root / "cookbook space café 数据"
        self.temp.mkdir()
        # Preserve OS runtime paths (especially SystemRoot on Windows), never API credentials.
        system_env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            in {
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "SYSTEMDRIVE",
                "PATH",
                "PATHEXT",
                "HOME",
                "USERPROFILE",
                "APPDATA",
                "LOCALAPPDATA",
                "TMP",
                "TEMP",
                "TMPDIR",
            }
        }
        self.stack.enter_context(
            patch.dict(
                os.environ,
                {
                    **system_env,
                    "REOPENLY_API_KEY": "review-dummy",
                    "REOPENLY_PROJECT": "owner/project",
                    "REOPENLY_DATASET_ID": "22",
                    "REOPENLY_BASE_URL": "https://review.invalid/api",
                    "MPLCONFIGDIR": str(self.temp / "mpl"),
                    "MPLBACKEND": "Agg",
                },
                clear=True,
            )
        )
        self.api = MockAPI()
        self.api.install(self.stack)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        previous = Path.cwd()
        os.chdir(self.temp)
        self.addCleanup(os.chdir, previous)

    def load(self, number):
        script = next((ROOT / "embeddings").glob(f"{number:02}_*.py"))
        return runpy.run_path(str(script))

    def test_all_scripts_and_response_order(self):
        for number in [1, 2, 3, 4]:
            module = self.load(number)
            args = (
                ["--report"]
                if number == 3
                else ["--model", "trained"] if number == 4 else []
            )
            module["main"]() if number == 2 else module["main"](args)
        self.assertTrue((self.temp / "vector.png").is_file())
        self.assertEqual(len(self.api.events), 4)
        self.assertTrue(
            all(
                "/datasets/22/" in str(r.url)
                for r in self.api.requests
                if r.method == "POST" and r.url.path.endswith("retrieval-events")
            )
        )
        self.assertTrue(
            any(
                "/projects/owner/project/openai/v1/embeddings" in r.url.path
                for r in self.api.requests
            )
        )

    def test_each_example_keeps_its_embedding_request_count(self):
        for number, expected in [(1, 2), (2, 1), (3, 5), (4, 2)]:
            with self.subTest(example=number):
                self.api.requests.clear()
                module = self.load(number)
                if number == 2:
                    module["main"]()
                else:
                    module["main"](["--report"] if number == 3 else [])
                requests = [
                    r for r in self.api.requests if r.url.path.endswith("/embeddings")
                ]
                self.assertEqual(len(requests), expected)

    def test_feedback_event_contains_top_five_passages_in_rank_order(self):
        passages = [
            {"id": f"p{i}", "title": f"Title {i}", "text": f"Text {i}"}
            for i in range(6)
        ]
        scores = np.array([0.1, 0.8, 0.6, 0.7, 0.5, 0.9])
        event = self.load(3)["build_event"]("model", "question", passages, scores)
        self.assertEqual(
            [r["metadata"]["external_id"] for r in event["results"]],
            ["p5", "p1", "p3", "p2", "p4"],
        )
        self.assertEqual([r["rank"] for r in event["results"]], [1, 2, 3, 4, 5])
        self.assertEqual(event["results"][0]["text"], "Text 5")
        self.assertEqual(event["results"][0]["metadata"]["score"], 0.9)
        self.assertEqual(event["query"], "question")
        self.assertEqual(self.api.requests, [])

    def test_scripts_and_notebook_restore_order_and_normalize_vectors(self):
        # MockAPI reverses these rows on the wire; callers must restore input order.
        self.api.vector_override = [
            {"object": "embedding", "index": 0, "embedding": [3.0, 0.0]},
            {"object": "embedding", "index": 1, "embedding": [0.0, 4.0]},
        ]
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        for number in [1, 2, 3, 4]:
            with self.subTest(example=number):
                module = self.load(number)
                with module["client_for"](module["BASE_MODEL"]) as client:
                    vectors = module["embed"](
                        client, ["first", "second"], "query", module["BASE_MODEL"]
                    )
                np.testing.assert_array_equal(vectors, expected)
        scope = {}
        for cell in load_notebook()["cells"]:
            if cell["id"] in {
                "imports",
                "settings",
                "connect",
                "optional-settings",
                "embed-helper",
            }:
                exec("".join(cell["source"]), scope)
        try:
            np.testing.assert_array_equal(
                scope["embed"](["first", "second"], input_type="query"), expected
            )
        finally:
            scope["client"].close()

    def test_hosted_override_routes_all_examples(self):
        os.environ["REOPENLY_EMBEDDING_MODEL"] = "trained"
        for number in [1, 2, 3]:
            m = self.load(number)
            (
                m["main"]()
                if number == 2
                else m["main"](["--report"] if number == 3 else [])
            )
        embeddings = [
            r for r in self.api.requests if r.url.path.endswith("/embeddings")
        ]
        self.assertTrue(embeddings)
        self.assertTrue(
            all(
                r.url.path == "/api/projects/owner/project/openai/v1/embeddings"
                for r in embeddings
            )
        )
        self.assertTrue(all(e["model"] == "trained" for e in self.api.events))

    def test_scripts_always_connect_to_reopenly_cloud(self) -> None:
        """An old endpoint override cannot redirect a reader's key away from ReOpenly."""
        os.environ["REOPENLY_BASE_URL"] = "http://localhost:8000/api"
        for number in [1, 2, 3, 4]:
            module = self.load(number)
            for model in [module["BASE_MODEL"], "trained"]:
                with module["client_for"](model) as client:
                    self.assertEqual(client.base_url.host, "reopenly.com")
                    self.assertEqual(client.base_url.scheme, "https")
        with self.load(3)["feedback_client"]("owner/project") as api:
            self.assertEqual(api.base_url.host, "reopenly.com")
            self.assertEqual(api.base_url.scheme, "https")

    def test_missing_hosted_project_fails_before_calls(self):
        os.environ.pop("REOPENLY_PROJECT")
        with self.assertRaisesRegex(ValueError, "REOPENLY_PROJECT"):
            self.load(4)["main"](["--model", "trained"])
        self.assertEqual(self.api.requests, [])

    def test_feedback_requires_explicit_opt_in_and_dataset(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.load(3)["main"]([])
        os.environ.pop("REOPENLY_DATASET_ID")
        with self.assertRaisesRegex(ValueError, "REOPENLY_DATASET_ID"):
            self.load(3)["main"](["--report"])
        self.assertEqual(self.api.requests, [])

    def test_unknown_or_archived_dataset_spends_no_embedding_credits(self):
        self.api.datasets = [
            {"id": 22, "kind": "embedding", "archived_at": "2026-01-01"}
        ]
        with self.assertRaisesRegex(ValueError, "active embedding dataset"):
            self.load(3)["main"](["--report"])
        self.assertEqual(len(self.api.requests), 1)
        self.assertEqual(self.api.requests[0].method, "GET")

    def test_feedback_ids_are_unique_when_clock_does_not_advance(self):
        import time

        with patch.object(time, "time", return_value=1800000000):
            for _ in range(2):
                self.load(3)["main"](["--report"])
        self.assertEqual(len({e["event_id"] for e in self.api.events}), 8)
        for event in self.api.events:
            self.assertEqual([r["rank"] for r in event["results"]], list(range(1, 6)))
            self.assertTrue(
                all(isinstance(r["metadata"]["score"], float) for r in event["results"])
            )

    def test_lost_reply_is_not_retried(self):
        self.api.timeout = True
        with self.assertRaises(Exception) as caught:
            self.load(1)["main"]([])
        self.assertIn("Timeout", type(caught.exception).__name__)
        self.assertEqual(len(self.api.requests), 1)

    def test_all_clients_have_bounded_timeouts_and_no_retries(self):
        for number in [1, 2, 3, 4]:
            m = self.load(number)
            for model in [m["BASE_MODEL"], "trained"]:
                with m["client_for"](model) as client:
                    self.assertEqual(client.max_retries, 0)
                    timeout = getattr(client.timeout, "read", client.timeout)
                    self.assertGreater(timeout, 240)
                    self.assertLessEqual(timeout, 300)

    def test_key_project_and_model_validation(self):
        m = self.load(1)
        for variable, value, model in [
            ("REOPENLY_API_KEY", "", m["BASE_MODEL"]),
            ("REOPENLY_PROJECT", "owner/project/extra", "trained"),
            ("REOPENLY_PROJECT", "owner/../project", "trained"),
        ]:
            with (
                patch.dict(os.environ, {variable: value}),
                self.assertRaises(ValueError),
            ):
                m["client_for"](model)
        with self.assertRaises(ValueError):
            m["client_for"]("unknown/model")
        self.assertEqual(self.api.requests, [])

    def test_single_question_argument_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.load(1)["main"](["one question"])
        self.assertEqual(self.api.requests, [])

    def test_cli_supports_unicode_paths_data_and_ascii_stdout(self):
        repo = self.temp / "repo with spaces"
        (repo / "embeddings").mkdir(parents=True)
        (repo / "data").mkdir()
        source = next((ROOT / "embeddings").glob("01_*.py"))
        script = repo / "embeddings" / source.name
        shutil.copy(source, script)
        data = json.loads((ROOT / "data/help_center.json").read_text(encoding="utf-8"))
        data["searches"][0] = "Como funciona a devolução? 🔎"
        data["passages"][0]["text"] = "Informação sobre devoluções 🧠"
        (repo / "data/help_center.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        bootstrap = f"""
import contextlib, runpy, sys
sys.path.insert(0, {str(ROOT / "tests")!r})
from test_examples import MockAPI
api = MockAPI()
with contextlib.ExitStack() as patches:
    api.install(patches)
    sys.argv = [{str(script)!r}]
    runpy.run_path({str(script)!r}, run_name="__main__")
"""
        result = subprocess.run(
            [sys.executable, "-c", bootstrap],
            cwd=repo,
            env={**os.environ, "PYTHONIOENCODING": "ascii"},
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode, 0, result.stderr.decode("utf-8", errors="replace")
        )
        output = result.stdout.decode("utf-8")
        self.assertIn(data["searches"][0], output)
        self.assertIn(data["passages"][0]["text"], output)

    def test_metrics_against_known_ranks(self):
        m = self.load(4)
        self.api.lookup.update(
            {"passage-a": 0, "passage-b": 1, "query-a": 0, "query-b": 1}
        )
        with m["client_for"](m["BASE_MODEL"]) as client:
            result = m["evaluate"](
                m["BASE_MODEL"],
                [{"id": "a", "text": "passage-a"}, {"id": "b", "text": "passage-b"}],
                [
                    {"query": "query-a", "answer": "a"},
                    {"query": "query-b", "answer": "a"},
                ],
                client,
            )
        self.assertEqual(result, {"hit@1": 0.5, "mrr": 0.75, "ranks": [1, 2]})

    def test_invalid_response_vectors_are_rejected(self):
        m = self.load(1)
        for rows in [
            [],
            [{"object": "embedding", "index": 4, "embedding": [1.0]}],
            [{"object": "embedding", "index": 0, "embedding": [0.0]}],
        ]:
            self.api.vector_override = rows
            with (
                m["client_for"](m["BASE_MODEL"]) as client,
                self.assertRaises(ValueError),
            ):
                m["embed"](client, ["text"], "query", m["BASE_MODEL"])

    def test_notebook_executes_all_branches_and_skips_reporting_by_default(self):
        for report in ["0", "1"]:
            self.api.events.clear()
            os.environ.update(REOPENLY_REPORT_SEARCHES=report, REOPENLY_MODEL="trained")
            scope = {"__name__": "notebook", "__doc__": None}
            for c in load_notebook()["cells"]:
                source = "".join(c["source"])
                if c["cell_type"] == "code" and not source.startswith("%"):
                    exec(compile(source, f"<notebook:{c['id']}>", "exec"), scope)
            self.assertEqual(len(self.api.events), 4 if report == "1" else 0)

    def test_notebook_preflight_fails_before_first_paid_call(self):
        os.environ["REOPENLY_MODEL"] = "trained"
        os.environ.pop("REOPENLY_PROJECT")
        scope = {"__doc__": None}
        with self.assertRaisesRegex(ValueError, "REOPENLY_PROJECT"):
            for c in load_notebook()["cells"]:
                source = "".join(c["source"])
                if c["cell_type"] == "code" and not source.startswith("%"):
                    exec(source, scope)
        self.assertEqual(self.api.requests, [])

    def test_notebook_always_connects_to_reopenly_cloud(self) -> None:
        """The notebook ignores custom endpoints for base, hosted and feedback calls."""
        setup = [
            c
            for c in load_notebook()["cells"]
            if c["id"] in {"imports", "settings", "connect"}
        ]
        for url in ["http://localhost:8000/api", "https://other.invalid/api"]:
            with self.subTest(url=url):
                os.environ["REOPENLY_BASE_URL"] = url
                scope = {}
                for c in setup:
                    exec("".join(c["source"]), scope)
                self.assertEqual(scope["client"].base_url.host, "reopenly.com")
                self.assertEqual(scope["client"].base_url.scheme, "https")
                scope["client"].close()
        self.assertEqual(self.api.requests, [])

    def test_default_notebook_reuses_two_embedding_requests(self):
        scope = {"__name__": "notebook", "__doc__": None}
        for c in load_notebook()["cells"]:
            source = "".join(c["source"])
            if c["cell_type"] == "code" and not source.startswith("%"):
                exec(compile(source, f"<notebook:{c['id']}>", "exec"), scope)
        self.assertEqual(len(self.api.requests), 2)
        self.assertTrue(
            all(r.url.path == "/api/openai/v1/embeddings" for r in self.api.requests)
        )
        self.assertEqual(self.api.events, [])
        self.assertEqual(
            scope["metrics"](
                scope["scores"], scope["passage_ids"], scope["expected_answers"]
            ),
            (1.0, 1.0),
        )
        self.assertEqual(scope["client"].max_retries, 0)
        self.assertEqual(
            getattr(scope["client"].timeout, "read", scope["client"].timeout), 300
        )
        self.assertTrue(set(scope["searches"]).isdisjoint(scope["questions"]))

    def test_notebook_invalid_feedback_destination_makes_no_extra_embedding_call(self):
        os.environ["REOPENLY_REPORT_SEARCHES"] = "1"
        self.api.datasets = []
        scope = {"__name__": "notebook", "__doc__": None}
        with self.assertRaisesRegex(ValueError, "active embedding dataset"):
            for cell in load_notebook()["cells"]:
                source = "".join(cell["source"])
                if cell["cell_type"] == "code" and not source.startswith("%"):
                    exec(compile(source, f"<notebook:{cell['id']}>", "exec"), scope)
        embeddings = [
            r for r in self.api.requests if r.url.path.endswith("/embeddings")
        ]
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(self.api.events, [])

    def test_notebook_metrics_match_known_ranks(self):
        scope = {"np": np}
        source = "".join(
            next(
                c["source"]
                for c in load_notebook()["cells"]
                if c["id"] == "metric-helper"
            )
        )
        exec(source, scope)
        self.assertEqual(
            scope["metrics"](np.array([[1, 0], [0, 1]]), ["a", "b"], ["a", "a"]),
            (0.5, 0.75),
        )


class NotebookInstall(unittest.TestCase):
    def setUp(self) -> None:
        """Read the actual setup cell and the independently declared project pins."""
        self.source = "".join(
            next(c["source"] for c in load_notebook()["cells"] if c["id"] == "install")
        )
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.requirements = project["project"]["dependencies"]
        self.versions = dict(item.split("==") for item in self.requirements)

    def test_locked_environment_needs_no_installer_or_pip(self) -> None:
        """An already synced kernel must not try to install anything."""
        with (
            patch("importlib.metadata.version", side_effect=self.versions.__getitem__),
            patch("importlib.util.find_spec", return_value=None),
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            exec(self.source, {})
        run.assert_not_called()

    def test_uv_installs_missing_or_mismatched_packages_in_this_kernel(self) -> None:
        """A pip-free kernel uses uv with its own interpreter, even in a spaced path."""
        for installed in (None, "0.0.0"):
            with self.subTest(installed=installed):

                def version(name: str) -> str:
                    """Simulate a missing or outdated runtime dependency."""
                    if name == "openai":
                        if installed is None:
                            raise PackageNotFoundError(name)
                        return installed
                    return self.versions[name]

                uv = str(Path("tools with spaces") / "uv")
                with (
                    patch("importlib.metadata.version", side_effect=version),
                    patch("importlib.util.find_spec", return_value=None),
                    patch("shutil.which", return_value=uv),
                    patch("subprocess.run") as run,
                ):
                    exec(self.source, {})
                run.assert_called_once_with(
                    [
                        uv,
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        *self.requirements,
                    ],
                    check=True,
                )

    def test_standalone_kernel_can_fall_back_to_pip(self) -> None:
        """A standalone notebook remains usable where only pip is available."""
        with (
            patch("importlib.metadata.version", side_effect=PackageNotFoundError),
            patch("importlib.util.find_spec", return_value=object()),
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            exec(self.source, {})
        run.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", *self.requirements], check=True
        )

    def test_missing_installers_give_actionable_error(self) -> None:
        """Missing dependencies and installers must stop setup with guidance."""
        with (
            patch("importlib.metadata.version", side_effect=PackageNotFoundError),
            patch("importlib.util.find_spec", return_value=None),
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
            self.assertRaisesRegex(RuntimeError, "Install uv"),
        ):
            exec(self.source, {})
        run.assert_not_called()

    def test_install_failure_stops_notebook_setup(self) -> None:
        """An unsuccessful installer cannot silently continue to paid calls."""
        with (
            patch("importlib.metadata.version", side_effect=PackageNotFoundError),
            patch("shutil.which", return_value="uv"),
            patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "uv")),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            exec(self.source, {})


class NotebookKernel(unittest.TestCase):
    def test_standalone_notebook_in_a_real_kernel_with_mocked_api(self):
        import nbformat
        from nbclient import NotebookClient

        with tempfile.TemporaryDirectory() as tmp:
            # Build the platform font cache outside teaching cells. On macOS the
            # first scan can emit a startup message on stderr after several seconds.
            subprocess.run(
                [sys.executable, "-c", "import matplotlib.font_manager"],
                env={**os.environ, "MPLCONFIGDIR": str(Path(tmp) / "mpl")},
                capture_output=True,
                check=True,
                timeout=60,
            )
            notebook = nbformat.reads(json.dumps(load_notebook()), as_version=4)
            bootstrap = f"""
import contextlib, os, sys
from unittest.mock import patch
sys.path.insert(0, {str(ROOT / "tests")!r})
from test_examples import MockAPI
os.environ.update(REOPENLY_API_KEY="review-dummy", REOPENLY_PROJECT="owner/project",
    REOPENLY_DATASET_ID="22", REOPENLY_MODEL="trained", REOPENLY_REPORT_SEARCHES="1",
    MPLBACKEND="Agg", MPLCONFIGDIR={str(Path(tmp) / "mpl")!r})
offline_api = MockAPI()
offline_patches = contextlib.ExitStack()
offline_api.install(offline_patches)
setup_patches = contextlib.ExitStack()
installer_guard = setup_patches.enter_context(patch("subprocess.run",
    side_effect=AssertionError("A synced kernel must not invoke an installer")))
"""
            notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap))
            install_index = next(
                i for i, cell in enumerate(notebook.cells) if cell.id == "install"
            )
            notebook.cells.insert(
                install_index + 1,
                nbformat.v4.new_code_cell(
                    "installer_guard.assert_not_called()\nsetup_patches.close()"
                ),
            )
            notebook.cells.append(
                nbformat.v4.new_code_cell(
                    'assert len(offline_api.events) == 4\nassert any("/projects/owner/project/openai/" in r.url.path for r in offline_api.requests)'
                )
            )
            NotebookClient(
                notebook,
                timeout=60,
                startup_timeout=60,
                kernel_name="python3",
                resources={"metadata": {"path": tmp}},
            ).execute()
            stderr = [
                (cell.id, output.get("text", ""))
                for cell in notebook.cells
                for output in cell.get("outputs", [])
                if output.get("output_type") == "stream"
                and output.get("name") == "stderr"
            ]
            self.assertEqual(output_issues(notebook), [], msg=stderr)


class Guards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo space café 数据"
        self.root.mkdir()
        # A tiny isolated repository proves working-tree and committed-history behavior.
        for name in ["embeddings", "data", "scripts"]:
            (self.root / name).mkdir()
        for path in (ROOT / "embeddings").glob("*.py"):
            shutil.copy(path, self.root / "embeddings" / path.name)
        shutil.copy(ROOT / "data/help_center.json", self.root / "data/help_center.json")
        shutil.copy(ROOT / "pyproject.toml", self.root / "pyproject.toml")
        (self.root / "embeddings/try_embeddings.ipynb").write_text(
            json.dumps(sample_outputs(load_notebook())), encoding="utf-8"
        )
        (self.root / ".gitignore").write_text(
            ".venv/\n.env\n.env.*\n!.env.example\n", encoding="utf-8"
        )
        self.git("init", "-q")
        self.git("config", "user.name", "Offline test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgSign", "false")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], stderr=subprocess.DEVNULL
        )

    def test_secret_scan_detects_worktree_and_removed_history_without_values(self):
        token = "fnf_" + "a" * 32
        secret = self.root / "credentials.txt"
        secret.write_text(token, encoding="utf-8")
        self.assertTrue(scan_secrets.scan(self.root))
        self.git("add", ".")
        self.git("commit", "-qm", "synthetic fixture")
        secret.unlink()
        self.git("add", "-u")
        self.git("commit", "-qm", "removed fixture")
        findings = scan_secrets.scan(self.root)
        self.assertTrue(any("history" in issue for issue in findings))
        self.assertNotIn(token, "\n".join(findings))

    def test_scanner_token_families_and_placeholder(self):
        samples = [
            "fnf_" + "a" * 32,
            "AKIA" + "A" * 16,
            "sk-" + "a" * 32,
            "-----BEGIN " + "PRIVATE KEY-----",
            "https://user:" + "secret@host.invalid",
            "PASSWORD=" + "a" * 20,
        ]
        for sample in samples:
            self.assertTrue(scan_secrets.findings(sample.encode()))
        self.assertEqual(scan_secrets.findings(scan_secrets.PLACEHOLDER), set())

    def test_history_checks_sensitive_names_even_when_blobs_are_shared(self) -> None:
        """Removed private filenames must be found despite Git deduplicating their bytes."""
        content = "identical nonsensitive fixture\n"
        (self.root / "public.txt").write_text(content, encoding="utf-8")
        private = [".env.private", "keys/client.pem", "keys/client copy café.key"]
        for name in private:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.git("add", "-f", "public.txt", *private)
        self.git("commit", "-qm", "shared synthetic blobs")
        for name in private:
            (self.root / name).unlink()
        self.git("add", "-u")
        self.git("commit", "-qm", "removed private aliases")
        self.assertEqual(scan_secrets.scan(self.root, history=False), [])
        issues = scan_secrets.scan(self.root)
        self.assertGreaterEqual(
            sum(
                "history" in issue and "sensitive filename" in issue for issue in issues
            ),
            len(private),
        )

    def test_secrets_in_commit_and_tag_messages_are_detected(self):
        token = "fnf_" + "a" * 32
        self.git("commit", "--allow-empty", "-qm", token)
        self.git("tag", "-a", "synthetic-tag", "-m", token)
        issues = scan_secrets.scan(self.root)
        self.assertEqual(sum("reopenly_key" in issue for issue in issues), 2)
        self.assertNotIn(token, "\n".join(issues))

    def test_ignored_virtualenv_does_not_affect_release_checks(self):
        ignored = self.root / ".venv/vendor.py"
        ignored.parent.mkdir()
        ignored.write_text("fnf_" + "x" * 32 + "\na" + "nthropic\n", encoding="utf-8")
        self.assertEqual(check.static_checks(self.root), [])

    def test_forced_tracked_env_is_refused(self):
        (self.root / ".env.local").write_text("not a secret", encoding="utf-8")
        self.git("add", "-f", ".env.local")
        self.assertTrue(
            any("sensitive filename" in issue for issue in scan_secrets.scan(self.root))
        )

    def test_notebook_syntax_and_unsafe_outputs_are_rejected(self):
        path = self.root / "embeddings/try_embeddings.ipynb"
        notebook = sample_outputs(load_notebook(self.root))
        next(c for c in notebook["cells"] if c["id"] == "similarity")["source"] = [
            "if\n"
        ]
        path.write_text(json.dumps(notebook), encoding="utf-8")
        self.assertTrue(
            any("syntax error" in issue for issue in check.static_checks(self.root))
        )
        notebook = sample_outputs(load_notebook())
        next(c for c in notebook["cells"] if c["id"] == "similarity")["outputs"] = [
            {
                "output_type": "error",
                "ename": "RuntimeError",
                "evalue": "fixture",
                "traceback": [],
            }
        ]
        path.write_text(json.dumps(notebook), encoding="utf-8")
        self.assertTrue(
            any("error" in issue for issue in check.static_checks(self.root))
        )

    def test_notebook_structure_and_install_dependencies_are_valid(self):
        import nbformat

        notebook = sample_outputs(load_notebook())
        nbformat.validate(nbformat.from_dict(notebook))
        self.assertEqual(check.notebook_issues(notebook), [])

    def test_invalid_notebook_structure_is_refused(self):
        notebook = sample_outputs(load_notebook())
        notebook["cells"][0]["cell_type"] = "unknown"
        self.assertEqual(
            check.notebook_issues(notebook), ["Notebook structure is invalid."]
        )

    def test_minimum_python_syntax_is_enforced_for_scripts_and_notebook(self):
        # Type parameter defaults require Python 3.13; the cookbook minimum is 3.12.
        future_syntax = "type Example[T = str] = T\n"
        (self.root / "future.py").write_text(future_syntax, encoding="utf-8")
        notebook = sample_outputs(load_notebook(self.root))
        next(c for c in notebook["cells"] if c["id"] == "similarity")["source"] = [
            future_syntax
        ]
        (self.root / "embeddings/try_embeddings.ipynb").write_text(
            json.dumps(notebook), encoding="utf-8"
        )
        issues = check.static_checks(self.root)
        self.assertTrue(
            any("future.py: Python syntax error" in issue for issue in issues)
        )
        self.assertTrue(
            any(
                "Notebook cell similarity: Python 3.12 syntax error" in issue
                for issue in issues
            )
        )

    def test_reviewed_outputs_are_required(self):
        notebook = load_notebook()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell.update(outputs=[], execution_count=None)
        self.assertTrue(output_issues(notebook))
        self.assertEqual(output_issues(sample_outputs(notebook)), [])

    def test_optional_sections_need_executed_saved_outputs(self) -> None:
        """A default run with skipped optionals is insufficient for the published notebook."""
        for name in ["hosted-comparison", "feedback"]:
            notebook = sample_outputs(load_notebook())
            cell = next(c for c in notebook["cells"] if c["id"] == name)
            cell["outputs"] = []
            self.assertTrue(any(name in issue for issue in output_issues(notebook)))
            cell["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": "Skipped."}
            ]
            self.assertTrue(any(name in issue for issue in output_issues(notebook)))

    def test_notebook_is_the_source_without_generator_parity(self):
        notebook = sample_outputs(load_notebook(self.root))
        next(c for c in notebook["cells"] if c["id"] == "similarity")["source"] = [
            "print('Edited example')\n"
        ]
        (self.root / "embeddings/try_embeddings.ipynb").write_text(
            json.dumps(notebook), encoding="utf-8"
        )
        self.assertEqual(check.static_checks(self.root), [])

    def test_saved_output_rejects_active_content_and_secrets(self):
        notebook = sample_outputs(load_notebook(self.root))
        vector = next(c for c in notebook["cells"] if c["id"] == "vector")
        vector["outputs"][0]["data"]["text/html"] = "<script>alert('fixture')</script>"
        self.assertTrue(any("active" in issue for issue in output_issues(notebook)))
        notebook = sample_outputs(load_notebook(self.root))
        next(c for c in notebook["cells"] if c["id"] == "similarity")["outputs"][0][
            "text"
        ] = ("fnf_" + "a" * 32)
        (self.root / "embeddings/try_embeddings.ipynb").write_text(
            json.dumps(notebook), encoding="utf-8"
        )
        self.assertTrue(
            any("reopenly_key" in issue for issue in check.static_checks(self.root))
        )

    def test_notebook_cells_stay_short_and_do_not_embed_cli_scaffolding(self):
        code = [
            "".join(c["source"])
            for c in load_notebook()["cells"]
            if c["cell_type"] == "code"
        ]
        self.assertLessEqual(sum(len(s.splitlines()) for s in code), 220)
        self.assertLessEqual(max(len(s.splitlines()) for s in code), 24)
        self.assertLessEqual(
            max(len(line) for source in code for line in source.splitlines()), 100
        )
        self.assertNotIn("ArgumentParser", "\n".join(code))
        self.assertNotIn("__file__", "\n".join(code))

    def test_notebook_install_must_match_project_dependencies(self):
        project = self.root / "pyproject.toml"
        project.write_text(
            project.read_text(encoding="utf-8").replace(
                "openai==3.35.1", "openai==3.35.2"
            ),
            encoding="utf-8",
        )
        self.assertTrue(
            any("install cell" in issue for issue in check.static_checks(self.root))
        )

    def test_repo_commands_use_shared_uv_project(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["requires-python"], ">=3.12")
        self.assertTrue((ROOT / "uv.lock").is_file())
        self.assertFalse((ROOT / "requirements.txt").exists())
        scripts = [*ROOT.glob("embeddings/*.py"), *ROOT.glob("scripts/*.py")]
        for script in scripts:
            self.assertNotIn("# /// script", script.read_text(encoding="utf-8"))
            self.assertFalse(Path(str(script) + ".lock").exists())

    def test_static_failure_stops_tests_and_live_calls(self):
        with (
            patch.object(check, "static_checks", return_value=["synthetic failure"]),
            patch.object(check.subprocess, "run") as run,
            patch.object(check, "live_checks") as live,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(check.main(["--live"]), 1)
            run.assert_not_called()
            live.assert_not_called()

    def test_live_preflight_requires_all_settings_before_calls(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "REOPENLY_PROJECT"),
        ):
            check.live_checks()

    def test_ci_has_no_application_credentials_or_write_permissions(self):
        source = (ROOT / ".github/workflows/check.yml").read_text(encoding="utf-8")
        self.assertNotIn("secrets.", source)
        self.assertNotIn("pull_request_target", source)
        self.assertIn("contents: read", source)

    def test_ci_runner_context_is_scoped_to_steps(self) -> None:
        """GitHub must accept the workflow before it can schedule any OS checks."""
        source = (ROOT / ".github/workflows/check.yml").read_text(encoding="utf-8")
        job_environments = re.findall(r"(?m)^    env:\n((?:      .*\n)*)", source)
        for environment in job_environments:
            self.assertNotRegex(environment, r"\$\{\{\s*runner[.\[]")
        self.assertRegex(source, r"(?m)^          TMPDIR: \$\{\{ runner.temp \}\}$")

    def test_ci_covers_all_native_platforms_without_building_dependencies(self):
        source = (ROOT / ".github/workflows/check.yml").read_text(encoding="utf-8")
        platforms = re.search(r"^\s+os: (\[.*\])$", source, re.MULTILINE)
        self.assertIsNotNone(platforms)
        self.assertEqual(
            set(json.loads(platforms[1].replace("'", '"'))),
            {
                "ubuntu-latest",
                "windows-latest",
                "macos-latest",
                "macos-15-intel",
            },
        )
        self.assertIn("runs-on: ${{ matrix.os }}", source)
        self.assertIn("uv sync --locked --no-build", source)

    def test_prepared_default_branch_policy_requires_prs_and_checks(self):
        policy = json.loads(
            (ROOT / ".github/branch-ruleset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(policy["enforcement"], "active")
        self.assertEqual(policy["bypass_actors"], [])
        self.assertEqual(
            policy["conditions"]["ref_name"]["include"], ["~DEFAULT_BRANCH"]
        )
        rules = {rule["type"]: rule for rule in policy["rules"]}
        self.assertIn("pull_request", rules)
        self.assertIn("non_fast_forward", rules)
        self.assertIn("deletion", rules)
        contexts = {
            item["context"]
            for item in rules["required_status_checks"]["parameters"][
                "required_status_checks"
            ]
        }
        self.assertEqual(
            contexts,
            {
                f"Offline checks ({runner}, Python {version})"
                for runner in [
                    "ubuntu-latest",
                    "windows-latest",
                    "macos-latest",
                    "macos-15-intel",
                ]
                for version in ["3.12", "3.14"]
            },
        )


if __name__ == "__main__":
    unittest.main()
