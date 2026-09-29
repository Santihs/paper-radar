"""paper-radar CLI.

paper-radar fetch       latest arXiv papers -> data/ (promptfoo dataset)
paper-radar eval        governance check, then promptfoo eval -> data/runs/<timestamp>/
paper-radar view        promptfoo web viewer (local)
paper-radar consensus   pass rate per model x task for the latest run (or --run NAME)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
from collections.abc import Sequence
from pathlib import Path

import truststore
from rich.console import Console

from paper_radar import config
from paper_radar.arxiv import fetch_by_ids, fetch_papers
from paper_radar.consensus import load_runs
from paper_radar.dataset import assemble, build_tests, load_papers, save_dataset
from paper_radar.governance import find_violations, load_config
from paper_radar.report import render
from paper_radar.runs import latest_run, new_run_dir

console = Console()

# Unknown flags on these commands go straight to promptfoo (e.g. --no-cache).
_PASSTHROUGH_COMMANDS = frozenset({"eval", "view"})


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        with console.status("Fetching recent papers and planted classics from arXiv..."):
            recent = fetch_papers(config.ARXIV_CATEGORIES, max_results=args.max_results)
            time.sleep(3)  # arXiv asks for >= 3 s between API calls
            classics = fetch_by_ids([p.arxiv_id for p in config.PLANTED])
    except urllib.error.HTTPError as err:
        kept = "existing dataset kept" if config.TESTS_FILE.exists() else "no dataset yet"
        console.print(
            f"[red]arXiv answered HTTP {err.code}[/] ({kept}). Try again in a few minutes."
        )
        return 1
    papers = assemble(recent, classics)
    save_dataset(papers, build_tests(papers), config.PAPERS_FILE, config.TESTS_FILE)
    console.print(
        f"[green]{len(papers)} public papers saved[/] "
        f"({len(papers) - len(config.PLANTED)} recent + {len(config.PLANTED)} planted) "
        f"-> {config.TESTS_FILE}"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    violations = find_violations(load_config(config.PROMPTFOO_CONFIG))
    if violations:
        for v in violations:
            console.print(f"[bold red]Blocked:[/] {v.provider}: {v.reason}")
        return 1
    if not config.TESTS_FILE.exists():
        console.print("[red]No dataset yet. Run `paper-radar fetch` first.[/]")
        return 1
    run = new_run_dir(config.RUNS_DIR)
    # Snapshot the exact config, so every result can be traced to the models that produced it.
    shutil.copy2(config.PROMPTFOO_CONFIG, run / config.PROMPTFOO_CONFIG.name)
    # JSON feeds `consensus`; CSV opens in Excel for whoever makes the decision.
    outputs = [str(run / config.EVAL_JSON), str(run / config.EVAL_CSV)]
    cli_args = ["-c", str(config.PROMPTFOO_CONFIG), "-o", *outputs]
    code = _promptfoo(["eval", *cli_args, "--no-share", *args.extra])
    console.print(f"Run saved in [bold]{run}[/]")
    return code


def cmd_view(args: argparse.Namespace) -> int:
    return _promptfoo(["view", "--yes", *args.extra])


def cmd_consensus(args: argparse.Namespace) -> int:
    run = config.RUNS_DIR / args.run if args.run else latest_run(config.RUNS_DIR, config.EVAL_JSON)
    if run is None or not (run / config.EVAL_JSON).is_file():
        console.print("[red]No eval results yet. Run `paper-radar eval` first.[/]")
        return 1
    console.print(f"Run: [bold]{run.name}[/]")
    eval_output = json.loads((run / config.EVAL_JSON).read_text(encoding="utf-8"))
    render(load_runs(eval_output), load_papers(config.PAPERS_FILE), console)
    return 0


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (comments, blanks and optional quotes allowed)."""
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _promptfoo(args: Sequence[str]) -> int:
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        console.print("[red]pnpm not found on PATH[/]")
        return 1
    # Secrets go through the process env: pnpm dlx would swallow a --env-file flag.
    env = {
        **read_env_file(config.ENV_FILE),
        **os.environ,
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_SHARING": "1",
        "PROMPTFOO_DISABLE_UPDATE": "true",
        # python asserts must run in this venv so they can import paper_radar
        "PROMPTFOO_PYTHON": sys.executable,
    }
    cmd = [pnpm, "dlx", f"promptfoo@{config.PROMPTFOO_VERSION}", *args]
    return subprocess.run(cmd, cwd=config.ROOT, env=env, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-radar",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download latest arXiv papers")
    fetch.add_argument("--max-results", type=int, default=config.RECENT_COUNT, help="recent papers")
    fetch.set_defaults(func=cmd_fetch)

    for name, func in (("eval", cmd_eval), ("view", cmd_view)):
        sub.add_parser(name, help=f"promptfoo {name} (extra flags pass through)").set_defaults(
            func=func
        )

    consensus = sub.add_parser("consensus", help="cross-model agreement (latest run)")
    consensus.add_argument("--run", help="run folder name under data/runs (default: latest)")
    consensus.set_defaults(func=cmd_consensus)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    truststore.inject_into_ssl()  # trust the OS store (corporate TLS inspection)
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command not in _PASSTHROUGH_COMMANDS:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
