import asyncio

import httpx

from app.main import app
from app.services.demo_cases import render_demo_case
from app.services.store import AnalysisStore
from app.services.vision.demo import DemoColorDetector


def test_standard_demo_cases_exercise_fire_smoke_and_review_paths():
    detector = DemoColorDetector()
    fire = asyncio.run(detector.detect(render_demo_case("synthetic-fire"), "fire.png"))
    smoke = asyncio.run(detector.detect(render_demo_case("synthetic-smoke"), "smoke.png"))
    clear = asyncio.run(detector.detect(render_demo_case("synthetic-clear"), "clear.png"))
    assert "fire" in {item.category for item in fire.detections}
    assert "smoke" in {item.category for item in smoke.detections}
    assert clear.detections == []


def test_stage4_metrics_records_and_report_exports():
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            image = render_demo_case("synthetic-fire")
            analysis = await client.post(
                "/api/v1/analyze",
                files={"image": ("synthetic-fire.png", image, "image/png")},
                data={"task": "联系 13800138000 并检查合成案例"},
            )
            request_id = analysis.json()["request_id"]
            metrics = await client.get("/api/v1/metrics")
            records = await client.get("/api/v1/analyses")
            html = await client.get(f"/api/v1/analyses/{request_id}/report.html")
            json_report = await client.get(f"/api/v1/analyses/{request_id}/report.json")
            cases = await client.get("/api/v1/demo-cases")
            case_image = await client.get("/api/v1/demo-cases/synthetic-smoke/image")
            return analysis, metrics, records, html, json_report, cases, case_image

    analysis, metrics, records, html, json_report, cases, case_image = asyncio.run(run())
    assert analysis.status_code == 200
    assert metrics.json()["total_analyses"] >= 1
    assert records.json()[0]["task"] == "联系 [phone] 并检查合成案例"
    assert "VisionGuard AI 工业安全视觉风险辅助报告" in html.text
    assert "中华人民共和国消防法" in html.text
    assert "适用边界" in html.text
    assert "https://wb.flk.npc.gov.cn/" in html.text
    assert json_report.headers["content-type"].startswith("application/json")
    assert len(cases.json()) == 3
    assert case_image.headers["x-visionguard-synthetic"] == "true"


def test_store_redacts_credentials_and_identifiers():
    store = AnalysisStore()
    assert store._redact("a@example.com sk-123456789 13800138000") == "[email] [api-key] [phone]"


def test_follow_up_qa_reuses_existing_evidence_without_redetection():
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            analysis = await client.post(
                "/api/v1/analyze",
                files={
                    "image": (
                        "synthetic-fire.png",
                        render_demo_case("synthetic-fire"),
                        "image/png",
                    )
                },
                data={"task": "分析火灾风险"},
            )
            request_id = analysis.json()["request_id"]
            answer = await client.post(
                f"/api/v1/analyses/{request_id}/ask",
                json={"question": "为什么判断为高风险？"},
            )
            return request_id, answer

    request_id, answer = asyncio.run(run())
    assert answer.status_code == 200
    body = answer.json()
    assert body["request_id"] == request_id
    assert "判断依据" in body["answer"]
    assert body["reasoning"]["evidence_ids"] == ["ev-fire-001"]
    assert [step["tool"] for step in body["agent_trace"]] == [
        "context.retrieve",
        "reasoning.explain",
    ]
    assert "未重新执行视觉检测" in body["agent_trace"][0]["summary"]


def test_follow_up_qa_rejects_expired_or_invalid_requests():
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.post(
                "/api/v1/analyses/not-found/ask", json={"question": "为什么？"}
            )
            invalid = await client.post(
                "/api/v1/analyses/not-found/ask", json={"question": "?"}
            )
            return missing, invalid

    missing, invalid = asyncio.run(run())
    assert missing.status_code == 404
    assert invalid.status_code == 422
