import asyncio
from hashlib import sha256

import httpx

from app.main import app, analysis_store
from app.services.demo_cases import render_demo_case
from app.services.store import AnalysisStore


def test_sessions_isolate_history_metrics_reports_and_followup():
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as alice, httpx.AsyncClient(transport=transport, base_url="http://test") as bob:
            created = await alice.post("/api/v1/analyze", files={"image": ("scene.png", render_demo_case("synthetic-fire"), "image/png")})
            assert created.status_code == 200
            request_id = created.json()["request_id"]
            cookie = created.headers["set-cookie"].lower()
            assert "httponly" in cookie and "samesite=strict" in cookie
            assert analysis_store._owners[request_id] == sha256(alice.cookies["vg_session"].encode()).hexdigest()
            for suffix in ["", "/report.html", "/report.json"]:
                path = f"/api/v1/analyses/{request_id}{suffix}"
                assert (await alice.get(path)).status_code == 200
                assert (await bob.get(path)).status_code == 404
            assert (await bob.post(f"/api/v1/analyses/{request_id}/ask", json={"question": "如何复核？"})).status_code == 404
            assert (await alice.post(f"/api/v1/analyses/{request_id}/ask", json={"question": "如何复核？"})).status_code == 200
            assert len((await alice.get("/api/v1/analyses")).json()) == 1
            assert (await bob.get("/api/v1/analyses")).json() == []
            assert (await alice.get("/api/v1/metrics")).json()["total_analyses"] == 1
            assert (await bob.get("/api/v1/metrics")).json()["total_analyses"] == 0
            alice.cookies.clear()
            assert (await alice.get(f"/api/v1/analyses/{request_id}")).status_code == 404

    asyncio.run(run())


def test_https_cookie_is_secure_and_malformed_cookie_is_replaced():
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
            response = await client.get("/api/v1/health", headers={"Cookie": "vg_session=invalid"})
            assert "Secure" in response.headers["set-cookie"]
            assert len(client.cookies["vg_session"]) == 43
            second = await client.get("/api/v1/health")
            assert "set-cookie" not in second.headers

    asyncio.run(run())


def test_eviction_removes_ownership_and_payload_together():
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/v1/analyze", files={"image": ("scene.png", render_demo_case("synthetic-clear"), "image/png")})

    from app.schemas import AnalysisResponse
    result = AnalysisResponse.model_validate(asyncio.run(run()).json())
    store = AnalysisStore(max_records=1)
    store.record(result, 1, owner="a")
    replacement = result.model_copy(update={"request_id": "next"})
    store.record(replacement, 2, owner="b")
    assert result.request_id not in store._owners
    assert store.get(result.request_id, owner="a") is None
    assert store.metrics(owner="a").total_analyses == 0
    assert store.get("next", owner="b") is not None


def test_task_length_is_enforced_server_side():
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/v1/analyze", data={"task": "x" * 1001}, files={"image": ("scene.png", render_demo_case("synthetic-clear"), "image/png")})

    assert asyncio.run(run()).status_code == 422
