from app.schemas import InferenceMetadata, VisionResult
from app.services.vision.base import VisionDetector
from app.services.vision.demo import InvalidImageError


class FallbackVisionDetector:
    """Runs the professional detector first and explicitly records any fallback."""

    name = "professional-with-demo-fallback"

    def __init__(
        self,
        primary: VisionDetector | None,
        fallback: VisionDetector,
        startup_reason: str | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.startup_reason = startup_reason
        self.model_version = primary.model_version if primary else fallback.model_version

    async def detect(self, image_bytes: bytes, file_name: str) -> VisionResult:
        if self.primary is not None:
            try:
                return await self.primary.detect(image_bytes, file_name)
            except InvalidImageError:
                raise
            except Exception as exc:
                reason = f"专业视觉后端推理失败：{type(exc).__name__}"
        else:
            reason = self.startup_reason or "专业视觉后端未配置"

        result = await self.fallback.detect(image_bytes, file_name)
        result.inference = InferenceMetadata(
            **result.inference.model_dump(exclude={"fallback_used", "fallback_reason"}),
            fallback_used=True,
            fallback_reason=reason,
        )
        result.limitations.insert(0, f"已启用Demo回退：{reason}")
        return result
