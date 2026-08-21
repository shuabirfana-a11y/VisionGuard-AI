from typing import Protocol

from app.schemas import KnowledgeEvidence, ReasoningResult, RiskAssessment, VisionResult


class Reasoner(Protocol):
    name: str
    model: str

    async def explain(
        self,
        task: str,
        vision: VisionResult,
        knowledge: list[KnowledgeEvidence],
        risk: RiskAssessment,
    ) -> ReasoningResult:
        ...

