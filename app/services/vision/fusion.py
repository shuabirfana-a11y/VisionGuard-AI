from __future__ import annotations

import asyncio

from app.schemas import VisionFusionResult, VisionResult
from app.services.vision.base import FireClassifier, VisionDetector
from app.services.vision.firebench import UnavailableFireClassifier


def fuse_fire_evidence(result: VisionResult) -> VisionFusionResult:
    fire_count = sum(item.category == "fire" for item in result.detections)
    classification = result.fire_classification
    if classification is None or not classification.available:
        return VisionFusionResult(
            status="classifier_unavailable",
            summary="高精度火情确认分支不可用，当前结论仅依据目标检测证据。",
            detector_fire_count=fire_count,
            classifier_fire=None,
        )
    if fire_count and classification.prediction:
        return VisionFusionResult(
            status="confirmed",
            summary="目标检测定位到火焰，整图分类模型同时确认存在可见火焰。",
            detector_fire_count=fire_count,
            classifier_fire=True,
        )
    if not fire_count and classification.prediction:
        return VisionFusionResult(
            status="classifier_only",
            summary="整图分类模型判断存在可见火焰，但目标检测未能定位；可能涉及远距离或极小火焰，必须复核原图。",
            detector_fire_count=0,
            classifier_fire=True,
        )
    if fire_count and not classification.prediction:
        return VisionFusionResult(
            status="detector_only",
            summary="目标检测给出火焰候选框，但整图分类模型未确认火情，存在模型证据冲突。",
            detector_fire_count=fire_count,
            classifier_fire=False,
        )
    return VisionFusionResult(
        status="no_fire_evidence",
        summary="两个视觉分支均未形成可见火焰证据；该结果不等同于现场绝对安全。",
        detector_fire_count=0,
        classifier_fire=False,
    )


class MultimodelVisionDetector:
    name = "multimodel-fire-evidence"

    def __init__(self, detector: VisionDetector, classifier: FireClassifier) -> None:
        self.detector = detector
        self.classifier = classifier
        self.model_version = f"{detector.model_version}+{classifier.model_version}"

    async def detect(self, image_bytes: bytes, file_name: str) -> VisionResult:
        # The two models inspect the same immutable image and have no dependency on
        # each other. Starting both immediately removes avoidable serial latency,
        # while detector failures still cancel the optional confirmation branch.
        detector_task = asyncio.create_task(self.detector.detect(image_bytes, file_name))
        classifier_task = asyncio.create_task(
            self._classify_safely(image_bytes, file_name)
        )
        try:
            result = await detector_task
        except BaseException:
            classifier_task.cancel()
            await asyncio.gather(classifier_task, return_exceptions=True)
            raise
        classification = await classifier_task
        result.fire_classification = classification
        result.fusion = fuse_fire_evidence(result)
        result.limitations.extend(classification.limitations)
        if not classification.available and classification.error:
            result.limitations.append(f"火情确认分支错误：{classification.error}")
        if result.fusion.status in {"classifier_only", "detector_only"}:
            result.limitations.append("视觉分支证据不一致，禁止输出无条件确定性结论。")
        return result

    async def _classify_safely(
        self, image_bytes: bytes, file_name: str
    ):
        try:
            return await self.classifier.classify(image_bytes, file_name)
        except Exception as exc:
            return await UnavailableFireClassifier(
                f"{type(exc).__name__}: {exc}"
            ).classify(image_bytes, file_name)

    async def close(self) -> None:
        close = getattr(self.classifier, "close", None)
        if close is not None:
            await close()

    async def start(self) -> None:
        detector_start = getattr(self.detector, "start", None)
        classifier_start = getattr(self.classifier, "start", None)
        starters = []
        if detector_start is not None:
            starters.append(detector_start())
        if classifier_start is not None:
            starters.append(classifier_start())
        if starters:
            await asyncio.gather(*starters)
