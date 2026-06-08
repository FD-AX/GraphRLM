from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.metrics.graph_ops_forensics import build_graph_ops_forensics, write_graph_ops_forensics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        action="append",
        type=Path,
        default=[
            Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph_ops_v2/repeat_0"),
            Path("artifacts/research_matrix/oolong_synth_stratified5/graph_rlm_semantic_graph_ops_v2/repeat_1"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/research_matrix/oolong_synth_stratified5/graph_ops_forensics_v2"),
    )
    parser.add_argument("--encoder-backend", choices=["hashing", "transformer"], default="hashing")
    args = parser.parse_args()

    payload = build_graph_ops_forensics(args.artifact_dir, encoder_backend=args.encoder_backend)
    outputs = write_graph_ops_forensics(payload, args.output_dir)
    print(json.dumps({"outputs": outputs, "summary": payload["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
