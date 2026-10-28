#!/usr/bin/env python
"""Fail if private data, models or notebook outputs are committed.

Checks (standard library only, so it runs anywhere with git and Python 3):

1. No tracked file lives under the private directories: data/, models/, artifacts/,
   checkpoints/, outputs/, eda/, demo/ (all gitignored by design).
2. No tracked file looks like a private artifact: Messages databases, trained-model weights, or
   Parquet files outside examples/ (the synthetic example data is allowed).
3. No tracked notebook has saved cell outputs (outputs often contain message text).

With ``--base REF`` the same checks also cover every commit in ``REF..HEAD``, so a file that is
added in one commit and deleted in a later commit of the same pull request is still caught
(deleting it later does not remove it from Git history).

The script never prints file contents, only paths.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

PRIVATE_DIRS = ("data/", "models/", "artifacts/", "checkpoints/", "outputs/", "eda/", "demo/")
ALLOWED_PREFIXES = ("examples/",)  # synthetic example data only
PRIVATE_SUFFIXES = (
    ".db", ".sqlite", ".sqlite3",  # Messages / sqlite databases
    ".safetensors", ".pt", ".pth", ".ckpt", ".joblib", ".onnx", ".mlmodel", ".gguf",  # weights
)
PRIVATE_NAMES = ("pytorch_model.bin", "chat.db")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def forbidden_reason(path: str) -> str | None:
    """Why ``path`` must not be committed, or None if it is fine."""
    if path.startswith(ALLOWED_PREFIXES):
        return None
    for d in PRIVATE_DIRS:
        if path.startswith(d):
            return f"is inside the private directory {d}"
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if name in PRIVATE_NAMES or lower.endswith(PRIVATE_SUFFIXES):
        return "looks like a private database or trained-model file"
    if lower.endswith(".parquet"):
        return "is a Parquet data file outside examples/"
    return None


def notebook_has_outputs(blob: str) -> bool:
    try:
        nb = json.loads(blob)
    except json.JSONDecodeError:
        return False
    return any(
        c.get("cell_type") == "code" and c.get("outputs") for c in nb.get("cells", [])
    )


def check(base: str | None) -> list[str]:
    problems: list[str] = []

    for path in git("ls-files").splitlines():
        reason = forbidden_reason(path)
        if reason:
            problems.append(f"tracked: {path} {reason}")
        elif path.endswith(".ipynb") and notebook_has_outputs(git("show", f"HEAD:{path}")):
            problems.append(f"tracked: {path} has saved cell outputs; clear them before committing")

    if base:
        seen: set[tuple[str, str]] = set()
        log = git("log", "--format=%H", "--name-only", "--diff-filter=AM", f"{base}..HEAD")
        commit = ""
        for line in log.splitlines():
            if not line:
                continue
            if len(line) == 40 and all(ch in "0123456789abcdef" for ch in line):
                commit = line
                continue
            if (commit, line) in seen:
                continue
            seen.add((commit, line))
            reason = forbidden_reason(line)
            if reason:
                problems.append(f"commit {commit[:8]}: {line} {reason}")
            elif line.endswith(".ipynb"):
                try:
                    blob = git("show", f"{commit}:{line}")
                except subprocess.CalledProcessError:
                    continue
                if notebook_has_outputs(blob):
                    problems.append(f"commit {commit[:8]}: {line} has saved cell outputs")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", help="also check every commit in BASE..HEAD (for pull requests)")
    args = p.parse_args()

    problems = check(args.base)
    if not problems:
        print("OK: no private data, models or notebook outputs are committed.")
        return 0

    print("PRIVATE FILE CHECK FAILED:\n", file=sys.stderr)
    for line in problems:
        print(f"  - {line}", file=sys.stderr)
    print(
        "\nVeto's data, models and artifacts are private and must never be committed.\n"
        "Fix: `git rm --cached <path>` for the current tree. If a commit in your branch added\n"
        "the file, removing it in a later commit is NOT enough: rewrite your branch\n"
        "(for example squash or `git rebase -i`) so the file never appears in its history,\n"
        "and email dev.meisel+veto@gmail.com if anything private was already pushed.\n"
        "See CONTRIBUTING.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
