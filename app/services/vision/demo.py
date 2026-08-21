from io import BytesIO
from time import perf_counter

from PIL import Image, UnidentifiedImageError

from app.schemas import BoundingBox, DetectionEvidence, InferenceMetadata, VisionResult


class InvalidImageError(ValueError):
    pass


class DemoColorDetector:
    """Transparent demo-only heuristic for exercising the full product flow."""

    name = "demo-color-heuristic-v1"
    model_version = "1.0-demo"
    confidence_threshold = 0.35

    async def detect(self, image_bytes: bytes, file_name: str) -> VisionResult:
        started = perf_counter()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("上传文件不是可解析的图片") from exc

        original_width, original_height = image.size
        if original_width < 2 or original_height < 2:
            raise InvalidImageError("图片尺寸过小")

        sample = image.copy()
        sample.thumbnail((640, 640))
        width, height = sample.size
        pixels = sample.load()

        fire_points: list[tuple[int, int]] = []
        smoke_points: list[tuple[int, int]] = []
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if r >= 175 and 45 <= g <= 190 and b <= 115 and r >= g * 1.12:
                    fire_points.append((x, y))
                chroma = max(r, g, b) - min(r, g, b)
                luminance = (r + g + b) / 3
                if chroma <= 18 and 75 <= luminance <= 205:
                    smoke_points.append((x, y))

        detections: list[DetectionEvidence] = []
        detections.extend(self._evidence("fire", "疑似火焰色区域", fire_points, width, height, original_width, original_height))
        detections.extend(self._evidence("smoke", "疑似烟雾灰度区域", smoke_points, width, height, original_width, original_height))
        return VisionResult(
            image_width=original_width,
            image_height=original_height,
            detector=self.name,
            inference=InferenceMetadata(
                backend="demo",
                model_name=self.name,
                model_version=self.model_version,
                confidence_threshold=self.confidence_threshold,
                device="cpu",
                inference_ms=round((perf_counter() - started) * 1000, 2),
            ),
            detections=detections,
            limitations=[
                "当前默认后端为演示级颜色启发式，不是训练后的YOLO模型。",
                "结果仅用于验证工具调用和报告链路，不可用于真实工业安全决策。",
            ],
        )

    def _evidence(
        self,
        category: str,
        label: str,
        points: list[tuple[int, int]],
        width: int,
        height: int,
        original_width: int,
        original_height: int,
    ) -> list[DetectionEvidence]:
        ratio = len(points) / (width * height)
        threshold = 0.006 if category == "fire" else 0.04
        if ratio < threshold:
            return []
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        scale_x = original_width / width
        scale_y = original_height / height
        bbox = BoundingBox(
            x1=round(min(xs) * scale_x),
            y1=round(min(ys) * scale_y),
            x2=min(original_width, round((max(xs) + 1) * scale_x)),
            y2=min(original_height, round((max(ys) + 1) * scale_y)),
        )
        confidence = min(0.85, 0.35 + ratio * (6 if category == "fire" else 2.5))
        return [DetectionEvidence(
            evidence_id=f"ev-{category}-001",
            category=category,
            label=label,
            confidence=round(confidence, 3),
            bbox=bbox,
            area_ratio=round(ratio, 4),
            source=self.name,
        )]
