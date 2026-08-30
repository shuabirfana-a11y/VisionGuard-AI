from typing import Protocol

from app.schemas import FireClassificationEvidence, VisionResult


class VisionDetector(Protocol):
    name: str
    model_version: str

    async def detect(self, image_bytes: bytes, file_name: str) -> VisionResult:
        ...


class FireClassifier(Protocol):
    name: str
    model_version: str

    async def classify(
        self, image_bytes: bytes, file_name: str
    ) -> FireClassificationEvidence:
        ...
