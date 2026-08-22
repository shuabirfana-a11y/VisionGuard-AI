from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    vision_backend: str = os.getenv("VISION_BACKEND", "demo").strip().lower()
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "").strip()
    yolo_model_version: str = os.getenv("YOLO_MODEL_VERSION", "").strip()
    yolo_expected_sha256: str = os.getenv("YOLO_EXPECTED_SHA256", "").strip().lower()
    yolo_confidence_threshold: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
    yolo_iou_threshold: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    yolo_device: str = os.getenv("YOLO_DEVICE", "auto").strip().lower()
    vision_fallback_enabled: bool = os.getenv("VISION_FALLBACK_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    fire_classifier_enabled: bool = os.getenv("FIRE_CLASSIFIER_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    fire_classifier_root: str = os.getenv("FIRE_CLASSIFIER_ROOT", "").strip()
    fire_classifier_bundle: str = os.getenv("FIRE_CLASSIFIER_BUNDLE", "").strip()
    fire_classifier_python: str = os.getenv("FIRE_CLASSIFIER_PYTHON", "").strip()
    fire_classifier_device: str = os.getenv("FIRE_CLASSIFIER_DEVICE", "auto").strip().lower()
    fire_classifier_mode: str = os.getenv("FIRE_CLASSIFIER_MODE", "persistent").strip().lower()
    fire_classifier_worker_count: int = int(os.getenv("FIRE_CLASSIFIER_WORKER_COUNT", "1"))
    fire_classifier_eager_start: bool = os.getenv(
        "FIRE_CLASSIFIER_EAGER_START", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}
    fire_classifier_timeout_seconds: float = float(os.getenv("FIRE_CLASSIFIER_TIMEOUT_SECONDS", "180"))
    fire_classifier_fallback_enabled: bool = os.getenv(
        "FIRE_CLASSIFIER_FALLBACK_ENABLED", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}
    reasoning_backend: str = os.getenv("REASONING_BACKEND", "deterministic").strip().lower()
    llm_base_url: str = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    llm_api_key: str = os.getenv("LLM_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "").strip()
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    reasoning_fallback_enabled: bool = os.getenv("REASONING_FALLBACK_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))


settings = Settings()
