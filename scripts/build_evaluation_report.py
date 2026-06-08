from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.metrics import build_evaluation_report, write_evaluation_report


def main() -> None:
    args = parse_args()
    artifact_dirs = _artifact_dirs(args)
    report = build_evaluation_report(artifact_dirs, score_name=args.score_name)
    write_evaluation_report(report, args.output_json, args.output_markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, action="append", default=[])
    parser.add_argument("--glob", type=str, action="append", default=[])
    parser.add_argument("--score-name", default="task_score")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def _artifact_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = list(args.artifact_dir)
    for pattern in args.glob:
        dirs.extend(Path().glob(pattern))
    return sorted({path for path in dirs if path.is_dir()})


if __name__ == "__main__":
    main()
