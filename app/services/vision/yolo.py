from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.schemas import BoundingBox, DetectionEvidence, InferenceMetadata, VisionResult
from app.services.vision.demo import InvalidImageError


def _scalar(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _vector(value: Any) -> list[float]:
    raw = value.tolist() if hasattr(value, "tolist") else value
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return [float(item) for item in raw]


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YoloVisionDetector:
    name = "ultralytics-yolo"

    def __init__(
        self,
        model_path: str,
        *,
        model_version: str = "",
        expected_sha256: str = "",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        model: Any | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("YOLO 阈值必须位于 0 到 1 之间")
        self.path = Path(model_path)
        if model is None and not self.path.is_file():
            raise RuntimeError(f"YOLO 模型文件不存在: {self.path}")
        full_digest = _file_digest(self.path) if self.path.is_file() else None
        normalized_expected = expected_sha256.strip().lower()
        if normalized_expected and full_digest != normalized_expected:
            raise RuntimeError(
                f"YOLO 模型SHA-256校验失败: expected={normalized_expected}, actual={full_digest}"
            )
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("尚未安装 ultralytics，请安装项目的 yolo 可选依赖") from exc
            model = YOLO(str(self.path))
        self.model = model
        self.model_version = model_version or self.path.stem or "configured-model"
        self.model_digest = full_digest[:12] if full_digest else None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    async def start(self) -> None:
        """Warm model kernels before the first judge-facing request."""
        predict_args: dict[str, Any] = {
            "source": Image.new("RGB", (640, 640), color=(24, 31, 40)),
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.device != "auto":
            predict_args["device"] = self.device
        for _ in range(2):
            self.model.predict(**predict_args)

    async def detect(self, image_bytes: bytes, file_name: str) -> VisionResult:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("上传文件不是可解析的图片") from exc
        width, height = image.size
        predict_args: dict[str, Any] = {
            "source": image,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.device != "auto":
            predict_args["device"] = self.device
        started = perf_counter()
        result = self.model.predict(**predict_args)[0]
        inference_ms = (perf_counter() - started) * 1000
        resolved_device = str(getattr(self.model, "device", self.device))

        detections: list[DetectionEvidence] = []
        aliases = {"fire": "fire", "flame": "fire", "smoke": "smoke"}
        for box in result.boxes:
            confidence = _scalar(box.conf)
            if confidence < self.confidence_threshold:
                continue
            class_id = int(_scalar(box.cls))
            display_label = str(result.names[class_id])
            raw_label = display_label.lower()
            category = next((mapped for key, mapped in aliases.items() if key in raw_label), None)
            if category not in {"fire", "smoke"}:
                continue
            x1, y1, x2, y2 = [round(value) for value in _vector(box.xyxy)]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = (x2 - x1) * (y2 - y1) / max(1, width * height)
            detections.append(DetectionEvidence(
                evidence_id=f"ev-{category}-{len(detections) + 1:03d}",
                category=category,
                label=display_label,
                confidence=round(confidence, 4),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                area_ratio=round(area_ratio, 4),
                source=f"{self.name}:{self.model_version}",
            ))
        return VisionResult(
            image_width=width,
            image_height=height,
            detector=self.name,
            inference=InferenceMetadata(
                backend="yolo",
                model_name=self.name,
                model_version=self.model_version,
                model_digest=self.model_digest,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                device=resolved_device,
                inference_ms=round(inference_ms, 2),
            ),
            detections=detections,
            limitations=["检测结果适用边界与性能指标必须依据当前模型版本的独立验证结果填写。"],
        )
