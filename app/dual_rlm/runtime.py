from __future__ import annotations

from uuid import uuid4

from app.dual_rlm.arbiter import EvidenceArbiter
from app.dual_rlm.models import DualRLMResult, GraphViewRef, RetrievalArm


class IndependentDualRLMRuntime:
    def __init__(
        self,
        graph_arm: RetrievalArm,
        text_arm: RetrievalArm,
        arbiter: EvidenceArbiter | None = None,
    ) -> None:
        self.graph_arm = graph_arm
        self.text_arm = text_arm
        self.arbiter = arbiter or EvidenceArbiter()

    def run(
        self,
        query: str,
        graph_view: GraphViewRef,
        run_id: str | None = None,
    ) -> DualRLMResult:
        resolved_run_id = run_id or f"dual_rlm_{uuid4().hex[:12]}"
        graph_result = self.graph_arm.run(query, run_id=resolved_run_id)
        text_result = self.text_arm.run(query, run_id=resolved_run_id)
        arbitration = self.arbiter.arbitrate(query, graph_result, text_result)
        return DualRLMResult(
            run_id=resolved_run_id,
            query=query,
            graph_view=graph_view,
            graph_result=graph_result,
            text_result=text_result,
            arbitration=arbitration,
            graph_mutation_allowed=False,
        )
