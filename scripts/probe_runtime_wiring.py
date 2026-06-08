from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.local_env import load_local_env
from scripts.run_benchmark_harness import make_arms, make_scorers


def main() -> None:
    load_local_env(PROJECT_ROOT)
    args = parse_probe_args()
    harness_args = SimpleNamespace(
        arms=args.arms,
        model=args.model,
        prompt_id=args.prompt_id,
        reasoning_effort=args.reasoning_effort,
        worker_model=args.worker_model,
        worker_reasoning_effort=args.worker_reasoning_effort,
        experiment_id=args.experiment_id,
        oolong_synth_official=True,
        benchmark=None,
    )
    arms = make_arms(harness_args)
    scorers = make_scorers(harness_args)
    payload = {
        "arms": [_describe_arm(arm) for arm in arms],
        "scorers": [
            {
                "class": scorer.__class__.__name__,
                "score_backend": getattr(scorer, "score_backend", None),
            }
            for scorer in (scorers or [])
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_probe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["direct_gpt", "graph_rlm"])
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-reasoning-effort")
    parser.add_argument("--experiment-id")
    parser.add_argument("--prompt-id", default="probe_prompt")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/probe"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/system_review/runtime_wiring_probe.json"),
    )
    return parser.parse_args()


def _describe_arm(arm) -> dict:
    return {
        "class": arm.__class__.__name__,
        "name": getattr(arm, "name", None),
        "model_name": getattr(arm, "model_name", None),
        "reasoning_effort": getattr(arm, "reasoning_effort", None),
        "worker_model_name": getattr(arm, "worker_model_name", None),
        "worker_reasoning_effort": getattr(arm, "worker_reasoning_effort", None),
        "prompt_id": getattr(arm, "prompt_id", None),
        "experiment_id": getattr(arm, "experiment_id", None),
        "top_k": getattr(arm, "top_k", None),
    }


if __name__ == "__main__":
    main()
