import asyncio
from pathlib import Path
import tomllib

import httpx
import pytest

from app import __version__
from app.main import app, agent
from app.services.demo_cases import render_demo_case
from app.services.exports import report_html


@pytest.fixture
def result():
    return asyncio.run(agent.analyze(render_demo_case("synthetic-fire"), "image.png", "检查风险"))


def test_report_contains_traceable_references_and_image_identity(result):
    html = report_html(result)
    assert 'id="visual-0"' in html and 'href="#visual-0"' in html
    assert 'id="knowledge-0"' in html and 'href="#knowledge-0"' in html
    assert result.vision.input_image.sha256 in html
    assert 'src="/static/report.js"' in html
    assert "onclick=" not in html
    assert "不能作为专业模型效果证明" in html


def test_report_escapes_user_content_and_blocks_non_http_source_links(result):
    result.task = '<img src=x onerror="alert(1)">'
    result.knowledge[0].source_url = "javascript:alert(1)"
    result.knowledge[0].basis = "<script>alert(1)</script>"
    html = report_html(result)
    assert '<img src=x' not in html
    assert "&lt;img" in html
    assert "javascript:" not in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_package_and_api_versions_match():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/api/v1/health")
    assert config["project"]["version"] == app.version == __version__ == asyncio.run(request()).json()["version"]
