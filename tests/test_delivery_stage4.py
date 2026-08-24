import asyncio
from pathlib import Path

import httpx

from app.main import app
from app.services.demo_cases import CASES, render_demo_case
from app.services.store import AnalysisStore
from app.services.vision.demo import DemoColorDetector


def test_competition_ui_exposes_core_ai_chain_and_human_review_boundary():
    root = Path(__file__).resolve().parents[1]
    html = (root / "app" / "static" / "index.html").read_text(encoding="utf-8")
    script = (root / "app" / "static" / "app.js").read_text(encoding="utf-8")
    styles = (root / "app" / "static" / "styles.css").read_text(encoding="utf-8")
    assert "专业视觉取证" in html
    assert "Agent 任务编排" in html
    assert "安全知识增强" in html
    assert "需人工复核" in html
    assert "riskCard.dataset.level" in script
    assert "align-items: start" in styles


def test_standard_demo_cases_exercise_fire_smoke_and_review_paths():
    detector = DemoColorDetector()
    clear = asyncio.run(detector.detect(render_demo_case("synthetic-clear"), "clear.png"))
    assert render_demo_case("verified-fire").startswith(b"\xff\xd8")
    assert render_demo_case("verified-smoke").startswith(b"\xff\xd8")
    assert clear.detections == []
    assert CASES["verified-fire"].expected_signal == "fire"
    assert CASES["verified-smoke"].expected_signal == "smoke"
    assert CASES["verified-smoke"].license == "CC-BY-4.0"


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
            case_image = await client.get("/api/v1/demo-cases/verified-smoke/image")
            return analysis, metrics, records, html, json_report, cases, case_image

    analysis, metrics, records, html, json_report, cases, case_image = asyncio.run(run())
    assert analysis.status_code == 200
    assert metrics.json()["total_analyses"] >= 1
    assert records.json()[0]["task"] == "联系 [phone] 并检查合成案例"
    assert "VisionGuard AI 工业安全视觉风险辅助报告" in html.text
    assert "中华人民共和国消防法" in html.text
    assert "适用边界" in html.text
    assert "https://wb.flk.npc.gov.cn/" in html.text
    assert "Agent审计轨迹" in html.text
    assert "模型阈值" in html.text
    assert "回退状态" in html.text
    assert "步骤耗时" in html.text
    assert "结论须由现场安全人员复核" in html.text
    assert "可引用视觉证据" in html.text
    assert json_report.headers["content-type"].startswith("application/json")
    assert len(cases.json()) == 3
    assert case_image.headers["x-visionguard-synthetic"] == "false"
    assert case_image.headers["content-type"] == "image/jpeg"
    assert case_image.headers["x-visionguard-case"] == "verified-smoke"
    assert cases.json()[1]["source_note"].startswith("IFireSmoke")


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
    assert "ev-fire-001" in body["agent_trace"][0]["references"]
    assert body["agent_trace"][1]["duration_ms"] >= 0


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
