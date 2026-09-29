"""Offline stand-in for OpenRouter, used by the e2e test (no API key, no cost).

`honest` answers every task correctly; `hallucinating` fails every task the way real
models do: picks the famous distractor, invents a summary, answers without tools.
"""

import json
from typing import Any


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    v = context["vars"]
    honest = options["config"].get("mode") == "honest"
    task = v["task"]

    if task == "pick":
        planted = (v["expected_ids"] + "," + v["distractor_ids"]).split(",")
        ids = v["expected_ids"].split(",") if honest else v["distractor_ids"].split(",")[:1]
        if not honest:  # famous distractor + two unrelated recent papers
            recent = [i for i in v["valid_ids"].split(",") if not i.startswith(tuple(planted))]
            ids += recent[:2]
        picks = [{"id": i, "why": "relevant"} for i in ids[:3]]
        return {"output": {"picks": picks, "hype_warning": "Attention is famous, not actionable"}}

    if task == "trap":
        text = (
            "NOT_IN_LIST" if honest else f"{v['trap_title']} proposes routing laws that cut cost."
        )
        return {"output": text}

    if honest:
        args = json.dumps({"query": "LLM routing strong vs cheap models", "max_results": 10})
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_papers", "arguments": args},
        }
        return {"output": [call]}
    return {"output": "Here are 3 great routing papers: RouterX, CheapNet and SmartSwitch."}
