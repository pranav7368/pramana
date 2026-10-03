"""Check the repository for files and credential patterns unsuitable for publication.

This is a small guardrail for CI and release review. It deliberately reports
only paths and counts; it never prints matching lines or credential material.
It is not a replacement for GitHub secret scanning or incident response.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SCAN_BYTES = 2_000_000
SECRET_PATTERNS = {
    "Google API token": re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
    "Groq API token": re.compile(rb"gsk_[0-9A-Za-z_-]{20,}"),
    "GitHub token": re.compile(rb"gh[pousr]_[0-9A-Za-z]{20,}"),
    "OpenAI-style token": re.compile(rb"sk-[0-9A-Za-z_-]{20,}"),
}
ALLOWED_ENV_FILES = {".env.example", ".env.pilot.example"}
BLOCKED_PREFIXES = (
    ".codex-finalizer/",
    ".pptbuild/",
    "deliverables/",
    "output/",
    "tmp/",
)


def git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def nul_paths(data: bytes) -> list[str]:
    return [item.decode("utf-8", "surrogateescape") for item in data.split(b"\0") if item]


def candidate_paths() -> list[Path]:
    tracked = nul_paths(git("ls-files", "--cached", "-z"))
    untracked = nul_paths(git("ls-files", "--others", "--exclude-standard", "-z"))
    paths = {Path(name) for name in tracked + untracked}
    return sorted(path for path in paths if (ROOT / path).is_file())


def blocked_path(path: Path) -> str | None:
    name = path.as_posix()
    if path.name.startswith(".env") and path.name not in ALLOWED_ENV_FILES:
        return "environment file"
    if any(name == prefix.rstrip("/") or name.startswith(prefix) for prefix in BLOCKED_PREFIXES):
        return "generated/private directory"
    if path.name.endswith(".tex") and "ArXiv" in path.name:
        return "private paper draft"
    if path.name.endswith("_Readiness.md") and "ArXiv" in path.name:
        return "private paper checklist"
    return None


def scan_bytes(path: Path, data: bytes) -> list[str]:
    if len(data) > MAX_SCAN_BYTES or b"\0" in data:
        return []
    return [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(data)]


def scan_worktree(paths: bool = True) -> tuple[list[tuple[str, str]], dict[str, int]]:
    violations: list[tuple[str, str]] = []
    matches: dict[str, int] = {}
    for relative in candidate_paths():
        reason = blocked_path(relative) if paths else None
        if reason:
            violations.append((relative.as_posix(), reason))
            continue
        try:
            data = (ROOT / relative).read_bytes()
        except OSError:
            violations.append((relative.as_posix(), "unreadable file"))
            continue
        for label in scan_bytes(relative, data):
            matches[label] = matches.get(label, 0) + 1
            violations.append((relative.as_posix(), label))
    return violations, matches


def scan_history() -> tuple[list[tuple[str, str]], dict[str, int]]:
    violations: list[tuple[str, str]] = []
    matches: dict[str, int] = {}
    for commit in git("rev-list", "--all").decode().splitlines():
        for name in nul_paths(git("ls-tree", "-r", "--name-only", "-z", commit)):
            data = git("show", f"{commit}:{name}")
            for label in scan_bytes(Path(name), data):
                matches[label] = matches.get(label, 0) + 1
                violations.append((name, f"{label} in history"))
    return violations, matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", action="store_true", help="also scan every reachable Git blob"
    )
    parser.add_argument(
        "--secrets-only", action="store_true",
        help="private archive: allow private directories, still refuse credentials",
    )
    args = parser.parse_args()

    try:
        violations, counts = scan_worktree(paths=not args.secrets_only)
        if args.history:
            history_violations, history_counts = scan_history()
            violations.extend(history_violations)
            for label, count in history_counts.items():
                counts[label] = counts.get(label, 0) + count
    except subprocess.CalledProcessError as exc:
        print(f"publication check could not run git command (exit {exc.returncode})", file=sys.stderr)
        return 2

    if violations:
        print("Publication check failed:")
        for path, reason in sorted(set(violations)):
            print(f"- {path}: {reason}")
        return 1

    scanned = len(candidate_paths())
    suffix = ", history included" if args.history else ""
    print(f"Publication check passed: {scanned} candidate files scanned{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
