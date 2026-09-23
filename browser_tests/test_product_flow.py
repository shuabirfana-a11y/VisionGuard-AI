"""Chromium workflow checks. Demo images are fixtures, not accuracy evidence."""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw
from playwright.sync_api import expect, sync_playwright
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "browser-artifacts"


@pytest.fixture(scope="session")
def server_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, VISION_BACKEND="demo", REASONING_BACKEND="deterministic",
               FIRE_CLASSIFIER_ENABLED="false", DEPLOYMENT_PROFILE="public-demo")
    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail("Test server exited; see browser-artifacts/server.log")
                try:
                    with urllib.request.urlopen(f"{url}/api/v1/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                pytest.fail("Test server did not start")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture
def page(server_url, request):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1366, "height": 900}, accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(server_url)
        try:
            yield page
        finally:
            page.screenshot(path=str(ARTIFACTS / f"{request.node.name}.png"), full_page=True)
            context.close()
            browser.close()
            assert not errors, f"Uncaught browser errors: {errors}"


def image_payload(name="scene.png", color=(250, 80, 10), size=(1280, 720)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return {"name": name, "mimeType": "image/png", "buffer": output.getvalue()}


def select_image(page, name="scene.png", color=(250, 80, 10)):
    page.locator("#image").set_input_files(image_payload(name, color))
    expect(page.locator("#analyze-button")).to_be_enabled()
    expect(page.locator("#image-feedback")).to_contain_text(name)


def analyze(page):
    page.locator("#analyze-button").click()
    expect(page.locator("#status")).to_have_text("分析完成")
    expect(page.locator("#result")).to_be_visible()


def defer_fetch(page, endpoint):
    # Ignore AbortSignal deliberately to prove state guards reject late data.
    page.evaluate("""endpoint => {
      const original = window.fetch;
      window.__deferredStarted = false;
      window.fetch = (url, options) => {
        if (String(url).endsWith(endpoint)) {
          window.__deferredStarted = true;
          return new Promise(resolve => { window.__releaseResponse = data =>
            resolve(new Response(JSON.stringify(data), {
              status: 200, headers: {'Content-Type': 'application/json'}
            }));
          });
        }
        return original(url, options);
      };
    }""", endpoint)


def release_fetch(page, data):
    page.evaluate("data => window.__releaseResponse(data)", data)
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


def test_empty_page_is_actionable_and_technical_details_are_collapsed(page):
    expect(page.get_by_role("heading", name="现场风险核查")).to_be_visible()
    expect(page.locator("#analyze-button")).to_be_disabled()
    expect(page.locator("#runtime-notice")).to_contain_text("演示视觉模式")
    assert not page.locator(".operations-panel").evaluate("(node) => node.open")
    assert not page.locator("#case-library").evaluate("(node) => node.open")


def test_analysis_followup_and_full_resolution_export(page, tmp_path):
    select_image(page)
    analyze(page)
    expect(page.locator("#result-notice")).to_contain_text("不能作为专业模型效果证明")
    expect(page.locator("#actions li").first).to_be_visible()
    assert not page.locator(".trace-details").evaluate("(node) => node.open")
    page.locator("#qa-question").fill("为什么判断有风险？")
    page.locator("#qa-form button").click()
    expect(page.locator("#qa-answer")).to_contain_text("视觉引用")
    with page.expect_download() as download:
        page.locator("#evidence-download").click()
    target = tmp_path / "evidence.png"
    download.value.save_as(target)
    with Image.open(target) as exported:
        assert exported.size == (1280, 720)
    report_url = page.locator("#html-report").get_attribute("href")
    assert page.request.get(page.url.rstrip("/") + report_url).status == 200


def test_switch_image_clears_reports_questions_and_boxes(page):
    select_image(page)
    analyze(page)
    page.locator("#qa-question").fill("旧图片问题")
    select_image(page, "new.png", (20, 30, 40))
    expect(page.locator("#result")).to_be_hidden()
    expect(page.locator("#qa-question")).to_have_value("")
    assert page.locator("#html-report").get_attribute("href") is None
    expect(page.locator("#evidence-download")).to_be_disabled()
    analyze(page)
    expect(page.locator("#result-context")).to_contain_text("核查")


def test_late_analysis_cannot_replace_new_image(page):
    select_image(page)
    analyze(page)
    previous = json.loads(page.locator("#json").text_content())
    defer_fetch(page, "/api/v1/analyze")
    page.locator("#analyze-button").click()
    page.wait_for_function("window.__deferredStarted")
    select_image(page, "different.png", (20, 30, 40))
    release_fetch(page, previous)
    expect(page.locator("#result")).to_be_hidden()
    expect(page.locator("#status")).to_have_text("待分析")
    assert page.locator("#html-report").get_attribute("href") is None


def test_late_followup_cannot_attach_to_new_image(page):
    select_image(page)
    analyze(page)
    defer_fetch(page, "/ask")
    page.locator("#qa-question").fill("解释旧图片")
    page.locator("#qa-form button").click()
    page.wait_for_function("window.__deferredStarted")
    select_image(page, "different.png")
    release_fetch(page, {"answer": "旧图片迟到回答", "reasoning": {"evidence_ids": [], "knowledge_ids": []}})
    expect(page.locator("#qa-answer")).to_be_hidden()
    expect(page.locator("#qa-answer")).to_have_text("")


def test_cancel_clears_pending_result_and_ignores_late_response(page):
    select_image(page)
    defer_fetch(page, "/api/v1/analyze")
    page.locator("#analyze-button").click()
    page.wait_for_function("window.__deferredStarted")
    page.locator("#cancel-analysis").click()
    release_fetch(page, {})
    expect(page.locator("#empty")).to_contain_text("已停止等待")
    expect(page.locator("#analyze-button")).to_be_enabled()
    expect(page.locator("#result")).to_be_hidden()


def test_busy_service_allows_retry(page):
    select_image(page)
    def unavailable(route):
        route.fulfill(status=503, content_type="application/json",
                      body=json.dumps({"detail": "当前分析任务较多，请稍后重试"}))
    page.route("**/api/v1/analyze", unavailable)
    page.locator("#analyze-button").click()
    expect(page.locator("#empty")).to_contain_text("当前分析任务较多")
    expect(page.locator("#analyze-button")).to_be_enabled()
    page.unroute("**/api/v1/analyze", unavailable)
    analyze(page)


def test_corrupt_image_does_not_leave_old_report_available(page):
    select_image(page)
    analyze(page)
    page.locator("#image").set_input_files({"name": "broken.png", "mimeType": "image/png", "buffer": b"not an image"})
    expect(page.locator("#image-feedback")).to_contain_text("无法读取")
    expect(page.locator("#analyze-button")).to_be_disabled()
    expect(page.locator("#result")).to_be_hidden()
    assert page.locator("#html-report").get_attribute("href") is None


def test_editing_task_invalidates_previous_result(page):
    select_image(page)
    analyze(page)
    page.locator("#task").fill("重点核查烟雾")
    expect(page.locator("#result")).to_be_hidden()
    expect(page.locator("#empty")).to_contain_text("核查要求已更改")
    analyze(page)
    expect(page.locator("#result-context")).to_have_text("本次任务：重点核查烟雾")


def test_sample_loads_and_runs_without_preview_race(page):
    page.locator("#case-library summary").click()
    page.locator('[data-case-id="synthetic-clear"]').click()
    expect(page.locator("#status")).to_have_text("分析完成")
    expect(page.locator("#evidence-canvas")).to_be_visible()
    expect(page.locator("#result")).to_be_visible()


def test_mobile_page_has_no_horizontal_overflow(page):
    page.set_viewport_size({"width": 390, "height": 844})
    select_image(page)
    analyze(page)
    page.locator(".trace-details summary").first.click()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    expect(page.locator("#evidence-download")).to_be_visible()


def test_evidence_links_support_keyboard_and_do_not_change_export(page, tmp_path):
    select_image(page)
    analyze(page)
    with page.expect_download() as before:
        page.locator("#evidence-download").click()
    before.value.save_as(tmp_path / "all.png")
    reference = page.locator('#reasoning [data-reference="ev-fire-001"]')
    reference.focus()
    page.keyboard.press("Enter")
    expect(page.locator("#evidence-focus-status")).to_contain_text("已定位 ev-fire-001")
    expect(page.locator("#visual-0")).to_have_class("evidence is-selected")
    expect(page.locator("#evidence-canvas")).to_be_focused()
    with page.expect_download() as selected:
        page.locator("#evidence-download").click()
    selected.value.save_as(tmp_path / "selected.png")
    assert (tmp_path / "all.png").read_bytes() == (tmp_path / "selected.png").read_bytes()
    page.locator("#clear-evidence-focus").click()
    expect(page.locator("#visual-0")).to_have_class("evidence")
    select_image(page, "new.png", (20, 30, 40))
    expect(page.locator("#evidence-focus-status")).to_be_hidden()
    expect(page.locator("#clear-evidence-focus")).to_be_hidden()


def test_knowledge_reference_opens_the_right_source_on_mobile(page):
    page.set_viewport_size({"width": 390, "height": 844})
    select_image(page)
    analyze(page)
    page.locator('#reasoning [data-reference="LAW-FIRE-044"]').click()
    card = page.locator("#knowledge-0")
    expect(card).to_be_visible()
    expect(card).to_be_focused()
    expect(card).to_contain_text("适用边界")
    assert card.locator("p").text_content().strip()
    assert card.locator(".source-link").get_attribute("href").startswith("https://")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_canvas_click_selects_corresponding_evidence_card(page):
    select_image(page)
    analyze(page)
    canvas = page.locator("#evidence-canvas")
    canvas.click(position={"x": 40, "y": 40})
    expect(page.locator("#visual-0")).to_be_focused()
    expect(page.locator("#evidence-focus-status")).to_contain_text("ev-fire-001")


def test_phone_photo_preview_detection_and_export_share_orientation(page, tmp_path):
    picture = Image.new("RGB", (120, 80), (10, 25, 40))
    ImageDraw.Draw(picture).rectangle((10, 20, 49, 59), fill=(240, 90, 20))
    exif = Image.Exif()
    exif[274] = 6
    output = io.BytesIO()
    picture.save(output, format="JPEG", quality=95, exif=exif)
    page.locator("#image").set_input_files({"name": "phone.jpg", "mimeType": "image/jpeg", "buffer": output.getvalue()})
    expect(page.locator("#image-feedback")).to_contain_text("80 × 120")
    analyze(page)
    body = json.loads(page.locator("#json").text_content())
    assert (body["vision"]["image_width"], body["vision"]["image_height"]) == (80, 120)
    assert body["vision"]["input_image"]["orientation_corrected"]
    assert 18 <= body["vision"]["detections"][0]["bbox"]["x1"] <= 22
    with page.expect_download() as downloaded:
        page.locator("#evidence-download").click()
    downloaded.value.save_as(tmp_path / "oriented.png")
    with Image.open(tmp_path / "oriented.png") as exported:
        assert exported.size == (80, 120)
    page.locator(".trace-details summary").first.click()
    expect(page.locator("#model-meta")).to_contain_text("已校正拍摄方向")


def test_transparent_image_preview_uses_same_white_pixels_as_model(page, tmp_path):
    output = io.BytesIO()
    Image.new("RGBA", (1280, 720), (240, 90, 20, 0)).save(output, format="PNG")
    page.locator("#image").set_input_files({"name": "transparent.png", "mimeType": "image/png", "buffer": output.getvalue()})
    analyze(page)
    body = json.loads(page.locator("#json").text_content())
    assert body["vision"]["detections"] == []
    assert body["vision"]["input_image"]["transparency_composited"]
    assert page.locator("#evidence-canvas").evaluate("c => Array.from(c.getContext('2d').getImageData(0,0,1,1).data)") == [255, 255, 255, 255]
    with page.expect_download() as downloaded:
        page.locator("#evidence-download").click()
    downloaded.value.save_as(tmp_path / "white.png")
    with Image.open(tmp_path / "white.png") as exported:
        assert exported.getpixel((0, 0))[:3] == (255, 255, 255)


def test_report_print_and_reference_links_work_under_security_policy(page):
    select_image(page)
    analyze(page)
    page.context.add_init_script("window.__cspErrors = []; document.addEventListener('securitypolicyviolation', e => window.__cspErrors.push(e.violatedDirective));")
    with page.expect_popup() as popup:
        page.locator("#html-report").click()
    report = popup.value
    report.wait_for_load_state()
    expect(report.locator("body")).to_contain_text("不能作为专业模型效果证明")
    expect(report.locator("body")).to_contain_text("输入图片SHA-256")
    report.evaluate("window.print = () => { window.__printed = true; }")
    report.locator("#print-report").click()
    assert report.evaluate("window.__printed === true")
    report.locator('a[href="#visual-0"]').first.click()
    assert report.url.endswith("#visual-0")
    report.locator('a[href="#knowledge-0"]').first.click()
    assert report.url.endswith("#knowledge-0")
    assert report.evaluate("window.__cspErrors") == []
    report.screenshot(path=str(ARTIFACTS / "report-evidence-and-print.png"), full_page=True)
    report.close()


def test_separate_browser_cannot_read_another_sessions_report(page):
    select_image(page)
    analyze(page)
    report_path = page.locator("#html-report").get_attribute("href")
    assert "vg_session" not in page.evaluate("document.cookie")
    other = page.context.browser.new_context()
    try:
        assert other.request.get(page.url.rstrip("/") + report_path).status == 404
        assert other.request.get(page.url.rstrip("/") + "/api/v1/analyses").json() == []
    finally:
        other.close()


def test_overlong_image_side_is_blocked_before_analysis(page):
    page.locator("#image").set_input_files(image_payload(size=(12001, 2)))
    expect(page.locator("#image-feedback")).to_contain_text("单边不超过 12000")
    expect(page.locator("#analyze-button")).to_be_disabled()
