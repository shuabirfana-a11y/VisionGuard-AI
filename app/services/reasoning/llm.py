import json
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.schemas import KnowledgeEvidence, ReasoningResult, RiskAssessment, VisionResult


class ReasoningValidationError(ValueError):
    pass


class LLMReasoningPayload(BaseModel):
    explanation: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    knowledge_ids: list[str] = Field(default_factory=list, max_length=20)
    uncertainties: list[str] = Field(default_factory=list, max_length=10)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=10)
    safety_boundary: str = Field(min_length=1, max_length=500)


def _decode_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ReasoningValidationError("大模型未返回JSON对象")
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ReasoningValidationError("大模型返回的JSON无法解析") from exc
    if not isinstance(parsed, dict):
        raise ReasoningValidationError("大模型返回值不是JSON对象")
    return parsed


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
        if vision.fire_classification and vision.fire_classification.available:
            evidence_ids.add(vision.fire_classification.evidence_id)
        knowledge_ids = {item.rule_id for item in knowledge}
        context = {
            "task": task,
            "visual_evidence": [item.model_dump() for item in vision.detections],
            "fire_classification": (
                vision.fire_classification.model_dump()
                if vision.fire_classification
                else None
            ),
            "vision_fusion": vision.fusion.model_dump() if vision.fusion else None,
            "knowledge_evidence": [
                {
                    "rule_id": item.rule_id,
                    "title": item.title,
                    "basis": item.basis,
                    "citation_id": item.citation_id,
                    "authority_level": item.authority_level,
                    "source_title": item.source_title,
                    "source_section": item.source_section,
                    "applicability": item.applicability,
                }
                for item in knowledge
            ],
            "risk_assessment": risk.model_dump(),
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            # The model sometimes needs roughly 220 tokens to finish all six
            # required fields. Leave enough headroom for the closing brace so a
            # valid, grounded response is not rejected merely due to truncation.
            "max_tokens": 320,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是工业安全风险辅助分析器。只能引用输入中存在的evidence_id和rule_id；"
                        "不得把未检出表述为安全，不得虚构法规、现场状态或处置结果。"
                        "必须遵守每条知识依据的applicability，不得把视觉疑似结果写成法律认定。"
                        "返回JSON字段：explanation、evidence_ids、knowledge_ids、uncertainties、"
                        "follow_up_questions、safety_boundary；所有字段必须存在，只输出JSON对象。"
                        "explanation不超过120字，uncertainties和follow_up_questions各不超过2项；"
                        "safety_boundary用一句话明确包含人工复核要求。"
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
            parsed = _decode_json_object(raw)
        finally:
            if owns_client:
                await client.aclose()
        try:
            validated = LLMReasoningPayload.model_validate(parsed)
        except ValidationError as exc:
            raise ReasoningValidationError("大模型返回结构不符合可信推理协议") from exc
        used_evidence = validated.evidence_ids
        used_knowledge = validated.knowledge_ids
        if not set(used_evidence).issubset(evidence_ids):
            raise ReasoningValidationError("大模型引用了不存在的视觉证据")
        if not set(used_knowledge).issubset(knowledge_ids):
            raise ReasoningValidationError("大模型引用了不存在的知识依据")
        explanation = validated.explanation.strip()
        if evidence_ids and not used_evidence:
            raise ReasoningValidationError("大模型解释未引用已有视觉证据")
        if knowledge_ids and not used_knowledge:
            raise ReasoningValidationError("大模型解释未引用已有知识依据")
        if not any(
            marker in validated.safety_boundary
            for marker in ("人工", "复核", "不得替代", "现场人员")
        ):
            raise ReasoningValidationError("大模型未给出明确人工复核边界")
        return ReasoningResult(
            provider=self.name,
            model=self.model,
            used_llm=True,
            explanation=explanation,
            evidence_ids=used_evidence,
            knowledge_ids=used_knowledge,
            uncertainties=validated.uncertainties,
            follow_up_questions=validated.follow_up_questions,
            safety_boundary=validated.safety_boundary,
        )
