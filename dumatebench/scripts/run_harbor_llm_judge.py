#!/usr/bin/env python3
"""Run DuMateBench's unified judge from a Harbor verifier container."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dumatebench.evaluator.llm_judge.unified import run_llm_judge_score
from dumatebench.evaluator.scoring import final_score


def _outputs(task_dir: Path, reward: dict) -> list[str]:
    candidates: list[str] = []
    checks = reward.get("checks", [])
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, dict):
            continue
        detail = check.get("detail", "")
        if isinstance(detail, str):
            try:
                value = json.loads(detail)
            except json.JSONDecodeError:
                value = {}
            if isinstance(value, dict) and isinstance(value.get("file"), str):
                candidates.append(value["file"])
    checks_file = task_dir / "evaluator" / "checks.yaml"
    if checks_file.is_file():
        candidates.extend(re.findall(r'["\']?file["\']?\s*:\s*["\']([^"\']+)', checks_file.read_text(errors="ignore")))
    result: list[str] = []
    for value in candidates:
        if value.startswith("run_outputs/") and value not in result and (task_dir / value).is_file():
            result.append(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("DUMATE_LLM_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    reward_path = task_dir / "run_outputs" / "reward.json"
    reward = json.loads(reward_path.read_text(encoding="utf-8"))
    scores: list[float] = []
    reports: list[dict] = []
    for index, output_file in enumerate(_outputs(task_dir, reward), start=1):
        report = run_llm_judge_score(task_dir, {
            "output_file": output_file,
            "rule_result": reward,
            "model": args.model,
            "base_url": args.base_url,
            "judge_output_file": f"run_outputs/llm_judge_scores/{index}.json",
            "reference_dir": "workspace_seed",
            "web_reference_dir": "web_reference",
        })
        reports.append(report)
        if report.get("judge_score") is not None:
            scores.append(float(report["judge_score"]))
    if not scores:
        return 0
    judge = round(sum(scores) / len(scores), 4)
    merged = dict(reward)
    merged.update({
        "llm_judge_score": judge,
        "final_score": final_score(reward.get("complete_pass", 0), reward.get("partial_pass", 0), judge),
        "llm_judge": {"status": "ok", "reports": reports},
    })
    (task_dir / "run_outputs" / "reward_with_llm_judge.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
