from typing import Any, Literal, TypedDict
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    user_request: str
    plan: list[dict]
    messages: list[dict]
    artifacts: dict[str, Any]
    current_depth: int
    max_depth: int
    status: Literal["running", "done", "failed"]
    last_error: str | None
    final_answer: str | None

class AgentAction(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)

    reason: str | None = None
    expected_result: str | None = None

    requires_confirmation: bool = False
    evidence_refs: list[str] = Field(default_factory=list)