from __future__ import annotations

import json
from pathlib import Path


DEFAULT_REGISTRY_PATH = Path("artifacts/artifact_registry.json")


def load_artifact_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    if not path.exists():
        return {"artifacts": {}, "arms": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_status(registry: dict, artifact_dir: Path, arm: str) -> dict:
    artifact_entry = registry.get("artifacts", {}).get(artifact_dir.name)
    if artifact_entry:
        return artifact_entry
    return registry.get("arms", {}).get(
        arm,
        {
            "status": "unknown",
            "research_eligible": False,
            "reason": "not_registered",
        },
    )
