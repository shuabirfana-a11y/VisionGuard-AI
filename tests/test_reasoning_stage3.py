import asyncio
import json

import httpx

from app.schemas import (
    BoundingBox,
    DetectionEvidence,
    InferenceMetadata,
    KnowledgeEvidence,
    RiskAssessment,
    RiskItem,
    VisionResult,
)
from app.services.reasoning.deterministic import DeterministicReasoner
from app.services.reasoning.fallback import FallbackReasoner
from app.services.reasoning.llm import CompatibleLLMReasoner, ReasoningValidationError


def context():
    vision = VisionResult(
        image_width=100,
        image_height=100,
        detector="test",
        inference=InferenceMetadata(
            backend="yolo",
            model_name="test-yolo",
            model_version="v1",
            confidence_threshold=0.35,
            iou_threshold=0.45,
            device="cpu",
            inference_ms=12.5,
        ),
        detections=[DetectionEvidence(
            evidence_id="ev-fire-001",
            category="fire",
            label="fire",
            confidence=0.9,
            bbox=BoundingBox(x1=10, y1=10, x2=60, y2=70),
            area_ratio=0.3,
            source="test",
        )],
    )
    knowledge = [KnowledgeEvidence(
        rule_id="VG-FIRE-001",
        title="明火核查",
        matched_category="fire",
        basis="需现场复核",
        recommended_actions=["通知安全人员"],
        source="内部规则",
        source_title="规则集",
        source_section="明火",
        source_version="draft",
        citation_id="VG-FIRE-001",
        retrieval_score=1.0,
    )]
    risk = RiskAssessment(
        overall_level="high",
        summary="发现候选风险",
        items=[RiskItem(
            category="fire",
            level="high",
            reason="视觉证据",
            evidence_ids=["ev-fire-001"],
            recommended_actions=["通知安全人员"],
        )],
    )
    return vision, knowledge, risk


def test_compatible_llm_accepts_only_existing_evidence_references():
    async def handler(request: httpx.Request):
        body = {
            "explanation": "依据视觉证据和内部规则，需要现场复核。",
            "evidence_ids": ["ev-fire-001"],
            "knowledge_ids": ["VG-FIRE-001"],
            "uncertainties": ["单帧图像有限"],
            "follow_up_questions": ["是否有其他视角？"],
            "safety_boundary": "不得替代现场判断。",
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body, ensure_ascii=False)}}]})

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        reasoner = CompatibleLLMReasoner(
            base_url="https://llm.test/v1", api_key="test", model="test-model", client=client
        )
        vision, knowledge, risk = context()
        result = await reasoner.explain("分析风险", vision, knowledge, risk)
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.used_llm is True
    assert result.evidence_ids == ["ev-fire-001"]


def test_compatible_llm_rejects_fabricated_evidence_reference():
    async def handler(request: httpx.Request):
        body = {"explanation": "虚构引用", "evidence_ids": ["ev-missing"], "knowledge_ids": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        reasoner = CompatibleLLMReasoner(
            base_url="https://llm.test/v1", api_key="test", model="test-model", client=client
        )
        vision, knowledge, risk = context()
        try:
            await reasoner.explain("分析风险", vision, knowledge, risk)
        finally:
            await client.aclose()

    try:
        asyncio.run(run())
        raise AssertionError("Expected ReasoningValidationError")
    except ReasoningValidationError:
        pass


class FailingReasoner:
    name = "failing-llm"
    model = "broken"

    async def explain(self, task, vision, knowledge, risk):
        raise RuntimeError("unavailable")


def test_reasoning_failure_falls_back_and_is_recorded():
    vision, knowledge, risk = context()
    reasoner = FallbackReasoner(FailingReasoner(), DeterministicReasoner())
    result = asyncio.run(reasoner.explain("分析风险", vision, knowledge, risk))
    assert result.used_llm is False
    assert result.fallback_used is True
    assert "RuntimeError" in result.fallback_reason
