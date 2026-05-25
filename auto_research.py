#!/usr/bin/env python3
"""Small autoresearch controller for the persona-drift benchmark workspace.

The script does not decide scientific truth. It creates run folders, captures
commands, parses a few stable metric lines, and writes a TSV ledger so agents can
iterate without losing the audit trail.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REQUIRED_FILES = (
    "AGENTS.MD",
    "IMPLEMENTATION_PLAN.md",
    "guide.MD",
    "program.md",
    "auto_research.py",
)

RESULT_FIELDS = (
    "timestamp_utc",
    "tag",
    "commit",
    "dirty_worktree",
    "status",
    "primary_metric_name",
    "primary_metric",
    "planned_generation_calls",
    "duration_s",
    "command",
    "log_path",
    "description",
)

TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
METRIC_PATTERNS = (
    ("primary_metric", re.compile(r"^primary_metric\s*[:=]\s*([-+0-9.eE]+)\s*$", re.MULTILINE)),
    ("bc_f1", re.compile(r"^(?:BC_F1|BC-F1|bc_f1)\s*[:=]\s*([-+0-9.eE]+)\s*$", re.MULTILINE)),
    ("pa_mean", re.compile(r"^(?:PA_mean|PA-mean|pa_mean)\s*[:=]\s*([-+0-9.eE]+)\s*$", re.MULTILINE)),
)
PLANNED_CALLS_PATTERN = re.compile(r"^planned_generation_calls\s*=\s*(\d+)\s*$", re.MULTILINE)
PYTEST_PASSED_PATTERN = re.compile(r"(?P<count>\d+)\s+passed")
UNITTEST_OK_PATTERN = re.compile(r"^OK\s*$", re.MULTILINE)


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def validate_tag(tag: str) -> str:
    if not TAG_PATTERN.match(tag):
        raise ValueError(
            "tag must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen"
        )
    return tag


def run_dir(root: Path, tag: str) -> Path:
    validate_tag(tag)
    return root / "results" / "autoresearch" / tag


def ensure_run_dir(root: Path, tag: str) -> Path:
    directory = run_dir(root, tag)
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    results_path = directory / "results.tsv"
    if not results_path.exists():
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
            writer.writeheader()
    return directory


def git_value(root: Path, args: Iterable[str], fallback: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    return completed.stdout.strip() or fallback


def current_commit(root: Path) -> str:
    return git_value(root, ("rev-parse", "--short", "HEAD"), "no_commit")


def dirty_worktree(root: Path) -> str:
    status = git_value(root, ("status", "--porcelain"), "")
    return "true" if status else "false"


def check_required_files(root: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (root / name).exists()]


def parse_metrics(text: str) -> dict[str, str]:
    metrics = {
        "primary_metric_name": "not_available",
        "primary_metric": "not_available",
        "planned_generation_calls": "not_available",
    }

    calls_match = PLANNED_CALLS_PATTERN.search(text)
    if calls_match:
        metrics["planned_generation_calls"] = calls_match.group(1)

    for name, pattern in METRIC_PATTERNS:
        match = pattern.search(text)
        if match:
            metrics["primary_metric_name"] = name
            metrics["primary_metric"] = match.group(1)
            break

    pytest_match = PYTEST_PASSED_PATTERN.search(text)
    if metrics["primary_metric"] == "not_available" and pytest_match:
        metrics["primary_metric_name"] = "tests_passed"
        metrics["primary_metric"] = pytest_match.group("count")

    if metrics["primary_metric"] == "not_available" and UNITTEST_OK_PATTERN.search(text):
        metrics["primary_metric_name"] = "unittest_status"
        metrics["primary_metric"] = "1"

    return metrics


def append_result(directory: Path, row: dict[str, str]) -> None:
    results_path = directory / "results.tsv"
    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow(row)


def shell_join(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def command_after_dash(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_log_name() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def cmd_init(args: argparse.Namespace) -> int:
    root = repo_root()
    directory = ensure_run_dir(root, args.tag)
    print(f"tag={args.tag}")
    print(f"results={directory / 'results.tsv'}")
    print(f"logs={directory / 'logs'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    del args
    root = repo_root()
    missing = check_required_files(root)
    if missing:
        for name in missing:
            print(f"missing_required_file={name}", file=sys.stderr)
        return 1
    print("autoresearch_check=ok")
    print(f"commit={current_commit(root)}")
    print(f"dirty_worktree={dirty_worktree(root)}")
    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    root = repo_root()
    command = command_after_dash(args.command)
    if not command:
        print("run-once requires a command after --", file=sys.stderr)
        return 2

    directory = ensure_run_dir(root, args.tag)
    log_path = directory / "logs" / f"{safe_log_name()}.log"
    start = time.monotonic()

    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_s,
        )
        output = completed.stdout
        return_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\nTIMEOUT after {args.timeout_s}s\n"
        return_code = 124
        timed_out = True

    duration_s = time.monotonic() - start
    log_path.write_text(output, encoding="utf-8")
    metrics = parse_metrics(output)

    status = "pass" if return_code == 0 else "fail"
    if timed_out:
        status = "timeout"

    row = {
        "timestamp_utc": utc_stamp(),
        "tag": args.tag,
        "commit": current_commit(root),
        "dirty_worktree": dirty_worktree(root),
        "status": status,
        "primary_metric_name": metrics["primary_metric_name"],
        "primary_metric": metrics["primary_metric"],
        "planned_generation_calls": metrics["planned_generation_calls"],
        "duration_s": f"{duration_s:.3f}",
        "command": shell_join(command),
        "log_path": str(log_path.relative_to(root)),
        "description": args.description,
    }
    append_result(directory, row)

    print(f"status={status}")
    print(f"return_code={return_code}")
    print(f"log_path={row['log_path']}")
    print(f"primary_metric_name={row['primary_metric_name']}")
    print(f"primary_metric={row['primary_metric']}")
    print(f"planned_generation_calls={row['planned_generation_calls']}")
    return return_code


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def cmd_summarize(args: argparse.Namespace) -> int:
    root = repo_root()
    directory = run_dir(root, args.tag)
    rows = read_results(directory / "results.tsv")
    if not rows:
        print(f"no_results_for_tag={args.tag}")
        return 1

    total = len(rows)
    passed = sum(1 for row in rows if row["status"] == "pass")
    failed = sum(1 for row in rows if row["status"] == "fail")
    timed_out = sum(1 for row in rows if row["status"] == "timeout")
    last = rows[-1]

    print(f"tag={args.tag}")
    print(f"total_runs={total}")
    print(f"pass={passed}")
    print(f"fail={failed}")
    print(f"timeout={timed_out}")
    print(f"last_status={last['status']}")
    print(f"last_metric={last['primary_metric_name']}:{last['primary_metric']}")
    print(f"last_log={last['log_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persona autoresearch controller")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    init_parser = subparsers.add_parser("init", help="create a tagged autoresearch run folder")
    init_parser.add_argument("--tag", required=True, type=validate_tag)
    init_parser.set_defaults(func=cmd_init)

    check_parser = subparsers.add_parser("check", help="check autoresearch readiness")
    check_parser.set_defaults(func=cmd_check)

    run_parser = subparsers.add_parser("run-once", help="run a command and append a TSV record")
    run_parser.add_argument("--tag", required=True, type=validate_tag)
    run_parser.add_argument("--description", required=True)
    run_parser.add_argument("--timeout-s", type=int, default=600)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=cmd_run_once)

    summarize_parser = subparsers.add_parser("summarize", help="summarize a tagged run ledger")
    summarize_parser.add_argument("--tag", required=True, type=validate_tag)
    summarize_parser.set_defaults(func=cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
