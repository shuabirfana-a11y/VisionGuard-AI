from typing import Protocol

from app.schemas import VisionResult


class VisionDetector(Protocol):
    name: str
    model_version: str

    async def detect(self, image_bytes: bytes, file_name: str) -> VisionResult:
        ...
