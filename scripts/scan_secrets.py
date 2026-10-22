"""Credential and sensitive-file checks. Findings never include matched secret values."""

import re
import subprocess
from pathlib import Path
from typing import Any

RULES = {
    "reopenly_key": re.compile(rb"\bfnf_[A-Za-z0-9_-]{16,}\b"),
    "private_key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    ),
    "cloud_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "service_token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b"
    ),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    "credential_url": re.compile(rb"[a-z]+://[^\s/:]+:[^\s/@]+@"),
    "credential_assignment": re.compile(
        rb"(?im)^\s*(?:[A-Z][A-Z0-9_]*_)?(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|PASSWORD|CLIENT_SECRET|TOKEN)\s*=\s*['\"]?([A-Za-z0-9_+./=-]{12,})"
    ),
}
# The original example placeholder is present in the initial Git history; it was never a key.
PLACEHOLDER = b"fnf_your_api_key_here"


def findings(content: bytes) -> set[str]:
    """Return matching rule names without disclosing credential values."""
    return {
        name
        for name, pattern in RULES.items()
        for match in pattern.finditer(content)
        if match.group(0) != PLACEHOLDER
        and not (name == "credential_assignment" and match.group(1) == PLACEHOLDER)
    }


def sensitive_path(name: str) -> bool:
    """Identify filenames that should not appear in a public release."""
    path = Path(name)
    return (
        (
            path.name != ".env.example"
            and (path.name == ".env" or path.name.startswith(".env."))
        )
        or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
        or path.name in {"credentials.json", "id_rsa", "id_ed25519"}
    )


def git(root: Path, *args: str, **kwargs: Any) -> bytes:
    """Read Git state as bytes with arguments passed without a shell."""
    return subprocess.check_output(["git", "-C", str(root), *args], **kwargs)


def release_files(root: Path) -> list[Path]:
    """Tracked plus new, nonignored files; a local virtualenv is never release content."""
    return [
        root / name
        for name in git(
            root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        )
        .decode()
        .split("\0")
        if name and (root / name).is_file()
    ]


def scan(root: Path, history: bool = True) -> list[str]:
    """Scan release files and reachable history, reporting identifiers and rules only."""
    issues = []
    for path in release_files(root):
        name = str(path.relative_to(root))
        if sensitive_path(name):
            issues.append(f"{name}: sensitive filename")
        if path.is_symlink():
            issues.append(f"{name}: release symlink requires review")
            continue
        issues.extend(f"{name}: {rule}" for rule in sorted(findings(path.read_bytes())))
    if history:
        objects = set(
            git(root, "rev-list", "--objects", "--no-object-names", "--all")
            .decode()
            .splitlines()
        )
        # Include annotated tag messages as well as file and commit contents.
        objects.update(
            git(root, "for-each-ref", "--format=%(objectname)").decode().splitlines()
        )
        for oid in sorted(objects):
            kind = git(root, "cat-file", "-t", oid).strip().decode()
            if kind == "tree":
                # Blob ids deduplicate contents; tree entries preserve every filename.
                for entry in git(root, "ls-tree", "-z", oid).split(b"\0"):
                    if not entry:
                        continue
                    header, _, name = entry.partition(b"\t")
                    if header.split()[1] == b"blob" and sensitive_path(
                        name.decode("utf-8", errors="surrogateescape")
                    ):
                        issues.append(f"history {oid[:12]}: sensitive filename")
                continue
            if kind not in {"blob", "commit", "tag"}:
                continue
            issues.extend(
                f"history {oid[:12]}: {rule}"
                for rule in sorted(findings(git(root, "cat-file", kind, oid)))
            )
    return issues


if __name__ == "__main__":
    problems = scan(Path(__file__).resolve().parents[1])
    for problem in problems:
        print(problem)
    print(
        "Secret scan failed."
        if problems
        else "Secret scan passed (release files and Git history)."
    )
    raise SystemExit(bool(problems))
