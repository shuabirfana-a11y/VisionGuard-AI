from io import BytesIO
import asyncio

import httpx
from PIL import Image

from app.core.agent import VisionGuardAgent
from app.services.knowledge import SafetyKnowledgeService
from app.services.report import SafetyReportService
from app.services.reasoning.deterministic import DeterministicReasoner
from app.services.risk import RiskAnalysisService
from app.services.vision.demo import DemoColorDetector
from app.main import app


def image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (120, 80), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_demo_detector_returns_structured_fire_evidence():
    result = asyncio.run(DemoColorDetector().detect(image_bytes((230, 90, 25)), "fire.png"))
    assert result.detector == "demo-color-heuristic-v1"
    assert result.inference.model_version == "1.0-demo"
    assert result.inference.inference_ms >= 0
    assert result.detections[0].category == "fire"
    assert result.detections[0].bbox.x2 == 120


def test_agent_runs_complete_tool_chain():
    agent = VisionGuardAgent(
        detector=DemoColorDetector(),
        knowledge=SafetyKnowledgeService(),
        risk=RiskAnalysisService(),
        report=SafetyReportService(),
        reasoner=DeterministicReasoner(),
    )
    result = asyncio.run(agent.analyze(image_bytes((230, 90, 25)), "scene.png", "分析火灾风险"))
    assert result.risk.overall_level in {"high", "critical"}
    assert [step.tool for step in result.agent_trace] == [
        "vision.detect",
        "knowledge.retrieve",
        "risk.analyze",
        "reasoning.explain",
        "report.generate",
    ]
    assert result.report.disclaimer


def test_http_api_accepts_image_and_returns_agent_trace():
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/analyze",
                files={"image": ("scene.png", image_bytes((230, 90, 25)), "image/png")},
                data={"task": "检查图片中的工业安全风险"},
            )

    response = asyncio.run(request())
    assert response.status_code == 200
    body = response.json()
    assert body["vision"]["detections"][0]["category"] == "fire"
    assert body["vision"]["inference"]["confidence_threshold"] == 0.35
    assert body["reasoning"]["used_llm"] is False
    assert body["reasoning"]["evidence_ids"] == ["ev-fire-001"]
    assert len(body["agent_trace"]) == 5
