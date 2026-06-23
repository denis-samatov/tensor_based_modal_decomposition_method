import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    ".cfg",
    ".example",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _git_files(*args: str) -> list[str]:
    """Return git-managed file lists or skip when git metadata is unavailable."""
    result = subprocess.run(
        ["git", "ls-files", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("git metadata is unavailable")
    return [line for line in result.stdout.splitlines() if line]


def _repository_files() -> list[str]:
    """Return tracked and pending files, excluding deleted files."""
    files = set(_git_files("--cached", "--others", "--exclude-standard"))
    deleted = set(_git_files("--deleted"))
    return sorted(files - deleted)


def test_documentation_entry_points_exist():
    """Verify that documented entry points use case-correct paths."""
    expected = [
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "SECURITY.md",
        ".env.example",
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        "docs/product/overview.md",
        "docs/architecture/overview.md",
        "docs/setup/local-development.md",
        "docs/interfaces/python-api.md",
        "docs/development/contribution-guide.md",
        "docs/research-system/reconstruction-pipeline.md",
    ]

    missing = [path for path in expected if not (PROJECT_ROOT / path).is_file()]

    assert missing == []


def test_no_tracked_generated_or_local_artifacts():
    """Generated outputs, local data, and local environment files must stay untracked."""
    forbidden_prefixes = (
        "data/",
        "results/",
        "scripts/plots/",
    )
    forbidden_files = {
        ".env",
    }

    offenders = sorted(
        path
        for path in _repository_files()
        if path in forbidden_files or path.startswith(forbidden_prefixes)
    )

    assert offenders == []


def test_no_cyrillic_text_in_tracked_text_files():
    """Tracked documentation and source comments should be written in English."""
    offenders: list[str] = []

    for relative_path in _repository_files():
        path = PROJECT_ROOT / relative_path
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES and path.name != ".env.example":
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if any("\u0400" <= char <= "\u04ff" for char in text):
            offenders.append(relative_path)

    assert offenders == []


def test_no_personal_absolute_paths_in_tracked_text_files():
    """Generated metadata must not leak local workstation paths into tracked files."""
    forbidden_fragments = (
        "/" + "Users/",
        "C:" + "\\Users\\",
        "/" + "home/",
    )
    offenders: list[str] = []

    for relative_path in _repository_files():
        path = PROJECT_ROOT / relative_path
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES and path.name != ".env.example":
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if any(fragment in text for fragment in forbidden_fragments):
            offenders.append(relative_path)

    assert offenders == []


def test_import_dag():
    """Verify basic core package imports."""
    import TBMD.core.decomposition
    import TBMD.core.geometry
    import TBMD.core.reconstruction
    import TBMD.core.sensor_placement

    assert TBMD.core.decomposition is not None
    assert TBMD.core.geometry is not None
    assert TBMD.core.reconstruction is not None
    assert TBMD.core.sensor_placement is not None


def test_deprecation_warning():
    """Verify that importing TBMD.modules emits a deprecation warning."""
    code = "import warnings; warnings.simplefilter('always'); import TBMD.modules"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert "DeprecationWarning" in result.stderr
    assert "TBMD.modules" in result.stderr
    assert "deprecated" in result.stderr
