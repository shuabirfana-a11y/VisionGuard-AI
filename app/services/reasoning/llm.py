import json
from typing import Any

import httpx

from app.schemas import KnowledgeEvidence, ReasoningResult, RiskAssessment, VisionResult


class ReasoningValidationError(ValueError):
    pass


class CompatibleLLMReasoner:
    """Evidence-constrained adapter for an OpenAI-compatible chat endpoint."""

    name = "compatible-llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 20,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def explain(
        self,
        task: str,
        vision: VisionResult,
        knowledge: list[KnowledgeEvidence],
        risk: RiskAssessment,
    ) -> ReasoningResult:
        evidence_ids = {item.evidence_id for item in vision.detections}
        knowledge_ids = {item.rule_id for item in knowledge}
        context = {
            "task": task,
            "visual_evidence": [item.model_dump() for item in vision.detections],
            "knowledge_evidence": [
                {
                    "rule_id": item.rule_id,
                    "title": item.title,
                    "basis": item.basis,
                    "citation_id": item.citation_id,
                }
                for item in knowledge
            ],
            "risk_assessment": risk.model_dump(),
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是工业安全风险辅助分析器。只能引用输入中存在的evidence_id和rule_id；"
                        "不得把未检出表述为安全，不得虚构法规、现场状态或处置结果。"
                        "返回JSON字段：explanation、evidence_ids、knowledge_ids、uncertainties、"
                        "follow_up_questions、safety_boundary。"
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
        }
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.timeout_seconds)
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            raw: Any = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
        finally:
            if owns_client:
                await client.aclose()
        used_evidence = list(parsed.get("evidence_ids", []))
        used_knowledge = list(parsed.get("knowledge_ids", []))
        if not set(used_evidence).issubset(evidence_ids):
            raise ReasoningValidationError("大模型引用了不存在的视觉证据")
        if not set(used_knowledge).issubset(knowledge_ids):
            raise ReasoningValidationError("大模型引用了不存在的知识依据")
        explanation = str(parsed.get("explanation", "")).strip()
        if not explanation:
            raise ReasoningValidationError("大模型未返回有效解释")
        return ReasoningResult(
            provider=self.name,
            model=self.model,
            used_llm=True,
            explanation=explanation,
            evidence_ids=used_evidence,
            knowledge_ids=used_knowledge,
            uncertainties=[str(item) for item in parsed.get("uncertainties", [])],
            follow_up_questions=[str(item) for item in parsed.get("follow_up_questions", [])],
            safety_boundary=str(parsed.get("safety_boundary", "必须由现场安全人员复核。")),
        )
