from app.config import Settings
from app.services.vision.base import VisionDetector
from app.services.vision.demo import DemoColorDetector
from app.services.vision.fallback import FallbackVisionDetector
from app.services.vision.firebench import FireBenchClassifier, UnavailableFireClassifier
from app.services.vision.fusion import MultimodelVisionDetector
from app.services.vision.yolo import YoloVisionDetector


def build_detector(settings: Settings) -> VisionDetector:
    detector: VisionDetector
    if settings.vision_backend == "yolo":
        fallback = DemoColorDetector()
        try:
            if not settings.yolo_model_path:
                raise RuntimeError("未设置 YOLO_MODEL_PATH")
            primary = YoloVisionDetector(
                settings.yolo_model_path,
                model_version=settings.yolo_model_version,
                expected_sha256=settings.yolo_expected_sha256,
                confidence_threshold=settings.yolo_confidence_threshold,
                iou_threshold=settings.yolo_iou_threshold,
                device=settings.yolo_device,
            )
        except RuntimeError as exc:
            if not settings.vision_fallback_enabled:
                raise
            detector = FallbackVisionDetector(None, fallback, str(exc))
        else:
            detector = (
                FallbackVisionDetector(primary, fallback)
                if settings.vision_fallback_enabled
                else primary
            )
    elif settings.vision_backend == "demo":
        detector = DemoColorDetector()
    else:
        raise RuntimeError(f"不支持的 VISION_BACKEND: {settings.vision_backend}")

    if not settings.fire_classifier_enabled:
        return detector
    try:
        if not settings.fire_classifier_root:
            raise RuntimeError("未设置 FIRE_CLASSIFIER_ROOT")
        classifier = FireBenchClassifier(
            settings.fire_classifier_root,
            bundle=settings.fire_classifier_bundle or None,
            python_executable=settings.fire_classifier_python or None,
            device=settings.fire_classifier_device,
            mode=settings.fire_classifier_mode,
            worker_count=settings.fire_classifier_worker_count,
            timeout_seconds=settings.fire_classifier_timeout_seconds,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        if not settings.fire_classifier_fallback_enabled:
            raise
        classifier = UnavailableFireClassifier(str(exc))
    return MultimodelVisionDetector(detector, classifier)


__all__ = ["VisionDetector", "build_detector"]
