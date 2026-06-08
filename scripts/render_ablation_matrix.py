from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.metrics.ablation_matrix import build_ablation_matrix, write_ablation_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/research_matrix/oolong_synth_stratified5/ablation_matrix_v1"),
    )
    args = parser.parse_args()

    payload = build_ablation_matrix()
    outputs = write_ablation_matrix(payload, args.output_dir)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
