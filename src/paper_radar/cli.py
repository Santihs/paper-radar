"""paper-radar CLI.

paper-radar fetch       latest arXiv papers -> data/ (promptfoo dataset)
paper-radar eval        governance check, then promptfoo eval (extra args pass through)
paper-radar view        promptfoo web viewer (local)
paper-radar consensus   pass rate per model x task, and agreement on picks
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence

import truststore
from rich.console import Console

from paper_radar import config
from paper_radar.arxiv import fetch_by_ids, fetch_papers
from paper_radar.consensus import load_runs
from paper_radar.dataset import assemble, build_tests, load_papers, save_dataset
from paper_radar.governance import find_violations, load_config
from paper_radar.report import render

console = Console()

# Unknown flags on these commands go straight to promptfoo (e.g. --no-cache).
_PASSTHROUGH_COMMANDS = frozenset({"eval", "view"})


def cmd_fetch(args: argparse.Namespace) -> int:
    with console.status("Fetching recent papers and planted classics from arXiv..."):
        recent = fetch_papers(config.ARXIV_CATEGORIES, max_results=args.max_results)
        classics = fetch_by_ids([p.arxiv_id for p in config.PLANTED])
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
    # JSON feeds `consensus`; CSV opens in Excel for whoever makes the decision.
    outputs = [str(config.EVAL_OUTPUT_FILE), str(config.EVAL_CSV_FILE)]
    cli_args = ["-c", str(config.PROMPTFOO_CONFIG), "-o", *outputs]
    return _promptfoo(["eval", *cli_args, "--no-share", *args.extra])


def cmd_view(args: argparse.Namespace) -> int:
    return _promptfoo(["view", "--yes", *args.extra])


def cmd_consensus(args: argparse.Namespace) -> int:
    if not config.EVAL_OUTPUT_FILE.exists():
        console.print("[red]No eval results yet. Run `paper-radar eval` first.[/]")
        return 1
    eval_output = json.loads(config.EVAL_OUTPUT_FILE.read_text(encoding="utf-8"))
    render(load_runs(eval_output), load_papers(config.PAPERS_FILE), console)
    return 0


def _promptfoo(args: Sequence[str]) -> int:
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        console.print("[red]pnpm not found on PATH[/]")
        return 1
    if config.ENV_FILE.exists():
        args = [*args, "--env-file", str(config.ENV_FILE)]
    env = {
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

    sub.add_parser("consensus", help="cross-model agreement").set_defaults(func=cmd_consensus)
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
