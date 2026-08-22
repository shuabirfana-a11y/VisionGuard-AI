import asyncio
from hashlib import sha256
from io import BytesIO

from PIL import Image

from app.services.vision.demo import DemoColorDetector
from app.services.vision.fallback import FallbackVisionDetector
from app.services.vision import build_detector
from app.services.vision.yolo import YoloVisionDetector
from app.config import Settings
import pytest


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (200, 100), color=(25, 30, 35)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = class_id
        self.conf = confidence
        self.xyxy = [xyxy]


class FakeResult:
    names = {0: "fire", 1: "smoke", 2: "person"}
    boxes = [
        FakeBox(0, 0.91, [20, 10, 100, 80]),
        FakeBox(1, 0.30, [30, 20, 90, 70]),
        FakeBox(2, 0.99, [0, 0, 20, 40]),
    ]


class FakeModel:
    def __init__(self):
        self.arguments = None

    def predict(self, **kwargs):
        self.arguments = kwargs
        return [FakeResult()]


def test_yolo_adapter_records_model_threshold_time_and_boxes():
    model = FakeModel()
    detector = YoloVisionDetector(
        "competition-fire-smoke.pt",
        model=model,
        model_version="stage2-test",
        confidence_threshold=0.40,
        iou_threshold=0.50,
        device="cpu",
    )
    result = asyncio.run(detector.detect(image_bytes(), "scene.png"))

    assert [item.category for item in result.detections] == ["fire"]
    assert result.detections[0].bbox.model_dump() == {"x1": 20, "y1": 10, "x2": 100, "y2": 80}
    assert result.inference.backend == "yolo"
    assert result.inference.model_version == "stage2-test"
    assert result.inference.confidence_threshold == 0.40
    assert result.inference.iou_threshold == 0.50
    assert result.inference.inference_ms >= 0
    assert model.arguments["conf"] == 0.40
    assert model.arguments["device"] == "cpu"


class FailingDetector:
    name = "failing-yolo"
    model_version = "broken"

    async def detect(self, image_bytes: bytes, file_name: str):
        raise RuntimeError("inference unavailable")


def test_demo_fallback_is_explicitly_recorded():
    detector = FallbackVisionDetector(FailingDetector(), DemoColorDetector())
    buffer = BytesIO()
    Image.new("RGB", (100, 100), color=(230, 90, 25)).save(buffer, format="PNG")
    result = asyncio.run(detector.detect(buffer.getvalue(), "scene.png"))

    assert result.inference.backend == "demo"
    assert result.inference.fallback_used is True
    assert "RuntimeError" in result.inference.fallback_reason
    assert result.limitations[0].startswith("已启用Demo回退")


def test_yolo_configuration_without_weights_builds_demo_fallback():
    detector = build_detector(Settings(vision_backend="yolo", yolo_model_path=""))
    result = asyncio.run(detector.detect(image_bytes(), "scene.png"))
    assert result.inference.backend == "demo"
    assert result.inference.fallback_used is True
    assert "YOLO_MODEL_PATH" in result.inference.fallback_reason


def test_yolo_rejects_unexpected_model_hash_before_inference(tmp_path):
    model_path = tmp_path / "candidate.pt"
    model_path.write_bytes(b"verified-model-placeholder")
    actual = sha256(model_path.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="SHA-256校验失败") as exc_info:
        YoloVisionDetector(
            str(model_path),
            model=FakeModel(),
            expected_sha256="0" * 64,
        )

    assert actual in str(exc_info.value)
