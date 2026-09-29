# Paper Radar

Same public data, same tasks, several LLMs through one API: which model fits which task,
how much does it cost, and does it make things up?

| Task | Question | Graded by |
|---|---|---|
| `pick` | Which 3 papers help choose a model by cost and quality? | Known answer (RouteLLM, FrugalGPT, LLM-as-a-Judge planted among recent papers); famous distractor *Attention Is All You Need* |
| `trap` | Summarize a paper that is **not** in the list | Must reply `NOT_IN_LIST` |
| `tools` | Build the team Tech Radar with `search_papers` / `save_to_sheet` | First tool call: right tool, on topic, schema-valid args, no saving invented rows |

| Layer | Tool |
|---|---|
| Model access | [OpenRouter](https://openrouter.ai) (one key, many models) |
| Evaluation | [promptfoo](https://www.promptfoo.dev) (side-by-side runs, asserts, cost, latency) |
| Domain (this repo) | Dataset with planted answers, task asserts, pass rate per model x task, governance check |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and pnpm (promptfoo runs via `pnpm dlx`, pinned).

```sh
uv sync
cp .env.example .env   # add your OPENROUTER_API_KEY
```

## Run

```sh
uv run paper-radar fetch              # 14 recent arXiv papers + 6 planted classics -> data/
uv run paper-radar eval --repeat 3    # policy check, then promptfoo (flags pass through)
uv run paper-radar view               # promptfoo web viewer (local)
uv run paper-radar consensus          # pass rate per model x task + agreement on picks
```

Each eval gets its own folder, `data/runs/<YYYYMMDD-HHMMSS>/`, with `eval.json`,
`results.csv` (opens in Excel) and a copy of the `promptfooconfig.yaml` that produced them.
`consensus` reads the latest run; `consensus --run <name>` reads an older one.
Swap a model: edit one `id` in `promptfooconfig.yaml`. No code changes.

This is a directional signal, not a benchmark: 3 tasks, few cases. The planted classics are in
the models' training data, so `pick` measures judgment, not comprehension of new work.

## Guardrails

- Public data only (arXiv). No company, client or personal data.
- `paper-radar eval` refuses to run unless every provider goes through OpenRouter, belongs to an
  allowed vendor (`ALLOWED_VENDORS` in `src/paper_radar/config.py`) and sets
  `provider.data_collection: deny`. The same rule is a test (`tests/test_governance.py`).
- promptfoo is pinned, telemetry and sharing are disabled, and `--no-share` is always passed.

## Tests

```sh
uv run pytest                   # offline unit tests
uv run pytest -m e2e            # promptfoo end-to-end with fake providers (no key, no cost)
uv run pytest -m live           # real arXiv call
uv run ruff check . && uv run mypy
```
