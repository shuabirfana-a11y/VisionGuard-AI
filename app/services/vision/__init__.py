from app.config import Settings
from app.services.vision.base import VisionDetector
from app.services.vision.demo import DemoColorDetector
from app.services.vision.fallback import FallbackVisionDetector
from app.services.vision.yolo import YoloVisionDetector


def build_detector(settings: Settings) -> VisionDetector:
    if settings.vision_backend == "yolo":
        fallback = DemoColorDetector()
        try:
            if not settings.yolo_model_path:
                raise RuntimeError("未设置 YOLO_MODEL_PATH")
            primary = YoloVisionDetector(
                settings.yolo_model_path,
                model_version=settings.yolo_model_version,
                confidence_threshold=settings.yolo_confidence_threshold,
                iou_threshold=settings.yolo_iou_threshold,
                device=settings.yolo_device,
            )
        except RuntimeError as exc:
            if not settings.vision_fallback_enabled:
                raise
            return FallbackVisionDetector(None, fallback, str(exc))
        if settings.vision_fallback_enabled:
            return FallbackVisionDetector(primary, fallback)
        return primary
    if settings.vision_backend != "demo":
        raise RuntimeError(f"不支持的 VISION_BACKEND: {settings.vision_backend}")
    return DemoColorDetector()


__all__ = ["VisionDetector", "build_detector"]
