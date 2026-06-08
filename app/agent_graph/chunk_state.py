from typing import TypedDict, NotRequired

from app.core.graph_models import ChunkNode, LocalGraphPatch
from app.rlm.state import RLMState
from app.agent_graph.chunk_schemas import LLMGraphExtraction


class ChunkGraphState(TypedDict):
    chunk: ChunkNode
    rlm_state: RLMState

    llm_extraction: NotRequired[LLMGraphExtraction]
    graph_patch: NotRequired[LocalGraphPatch]

    errors: list[str]