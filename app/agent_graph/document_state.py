from typing import NotRequired, TypedDict

from app.core.graph_models import ChunkNode, DocumentNode, LocalGraphPatch
from app.rlm.state import RLMState, RLMTransition


class DocumentGraphState(TypedDict):
    document: DocumentNode
    chunks: list[ChunkNode]
    current_chunk_index: int

    rlm_state: RLMState

    current_chunk: NotRequired[ChunkNode]
    graph_patch: NotRequired[LocalGraphPatch]
    rlm_transition: NotRequired[RLMTransition]

    errors: list[str]
    done: bool
