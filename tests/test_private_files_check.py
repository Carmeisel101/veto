"""Tests for scripts/check_private_files.py, using throwaway git repos with synthetic files."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_private_files.py"
spec = importlib.util.spec_from_file_location("check_private_files", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo, check=True, capture_output=True,
    )


def commit(repo: Path, files: dict[str, str], message: str = "c") -> None:
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    run_git(repo, "add", "-f", "-A")
    run_git(repo, "commit", "-m", message)


def run_check(repo: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra], cwd=repo, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    run_git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, {"README.md": "hello", "src/veto/models/code.py": "x = 1"}, "base")
    return tmp_path


def nb(outputs: list) -> str:
    return json.dumps({"cells": [{"cell_type": "code", "source": [], "outputs": outputs}]})


def test_clean_repo_passes(repo):
    r = run_check(repo)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("path", [
    "data/raw/data.parquet", "data/notes.txt", "models/veto/model.safetensors",
    "artifacts/evaluation.json", "checkpoints/step1.ckpt", "outputs/x.txt", "eda/n.ipynb",
    "demo/corpus.parquet", "dist/coreml/metadata.json", "release/veto.mlpackage/Manifest.json",
    "chat.db", "weights/model.joblib", "tables/export.parquet",
])
def test_forbidden_paths_fail(repo, path):
    commit(repo, {path: "synthetic"})
    r = run_check(repo)
    assert r.returncode == 1
    assert path in r.stderr
    assert "synthetic" not in r.stderr.replace("synthetic data", "")  # never prints contents


def test_examples_dir_is_allowed(repo):
    commit(repo, {"examples/synthetic/raw.parquet": "fake", "examples/synthetic/labels.csv": "x"})
    assert run_check(repo).returncode == 0


def test_nested_dirs_named_like_private_dirs_are_fine(repo):
    commit(repo, {"src/veto/data/helpers.py": "x = 1", "docs/artifacts.md": "notes"})
    assert run_check(repo).returncode == 0


def test_notebook_outputs_fail_and_clean_notebook_passes(repo):
    commit(repo, {"notebooks/clean.ipynb": nb([])})
    assert run_check(repo).returncode == 0
    commit(repo, {"notebooks/dirty.ipynb": nb([{"output_type": "stream", "text": ["x"]}])})
    r = run_check(repo)
    assert r.returncode == 1 and "dirty.ipynb" in r.stderr


def test_base_check_catches_file_added_then_removed_in_branch(repo):
    run_git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, {"data/raw/data.parquet": "private"}, "oops")
    run_git(repo, "rm", "-q", "-r", "--cached", "data")
    run_git(repo, "commit", "-q", "-m", "remove it again")
    # The current tree is clean, so without --base nothing is flagged...
    assert run_check(repo).returncode == 0
    # ...but the file is still in the branch history, so --base flags it.
    r = run_check(repo, "--base", "main")
    assert r.returncode == 1
    assert "data/raw/data.parquet" in r.stderr and "oops" not in r.stdout


def test_base_check_catches_notebook_outputs_cleared_later(repo):
    run_git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, {"notebooks/a.ipynb": nb([{"output_type": "stream", "text": ["x"]}])})
    commit(repo, {"notebooks/a.ipynb": nb([])})
    assert run_check(repo).returncode == 0
    assert run_check(repo, "--base", "main").returncode == 1


def test_base_check_passes_for_clean_branch(repo):
    run_git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, {"src/new.py": "x = 1"})
    assert run_check(repo, "--base", "main").returncode == 0


@pytest.mark.parametrize("path,expected", [
    ("data/x", True), ("dist/coreml/veto.mlpackage/Data/w.bin", True), ("models/a/b", True),
    ("artifacts/e.json", True), ("src/a.py", False),
    ("examples/synthetic/raw.parquet", False), ("README.md", False),
])
def test_forbidden_reason(path, expected):
    assert (checker.forbidden_reason(path) is not None) is expected
