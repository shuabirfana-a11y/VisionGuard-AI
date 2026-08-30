from app.schemas import KnowledgeEvidence, ReasoningResult, RiskAssessment, VisionResult
from app.services.reasoning.base import Reasoner


class FallbackReasoner:
    name = "llm-with-deterministic-fallback"

    def __init__(
        self,
        primary: Reasoner | None,
        fallback: Reasoner,
        startup_reason: str | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.startup_reason = startup_reason
        self.model = primary.model if primary else fallback.model

    async def explain(self, task, vision, knowledge, risk) -> ReasoningResult:
        if self.primary is not None:
            try:
                return await self.primary.explain(task, vision, knowledge, risk)
            except Exception as exc:
                detail = str(exc).strip().replace("\n", " ")[:160]
                reason = f"大模型推理不可用：{type(exc).__name__}"
                if detail:
                    reason += f"：{detail}"
        else:
            reason = self.startup_reason or "大模型推理未配置"
        result = await self.fallback.explain(task, vision, knowledge, risk)
        result.fallback_used = True
        result.fallback_reason = reason
        result.uncertainties.insert(0, f"推理环节已切换为确定性回退：{reason}")
        return result
