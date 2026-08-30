from __future__ import annotations

import argparse
import asyncio
import json
from hashlib import sha256
from pathlib import Path

from app.services.vision.yolo import YoloVisionDetector


EXPECTED_SHA256 = "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


async def verify(model_path: Path, image_path: Path) -> dict[str, object]:
    actual_digest = digest(model_path)
    if actual_digest != EXPECTED_SHA256:
        raise RuntimeError(f"model SHA-256 mismatch: {actual_digest}")
    detector = YoloVisionDetector(
        str(model_path),
        model_version="fire-smoke-yolov8n-upstream-v1.0.0",
        confidence_threshold=0.25,
        iou_threshold=0.30,
        device="cpu",
    )
    result = await detector.detect(image_path.read_bytes(), image_path.name)
    return {
        "model_sha256": actual_digest,
        "image": image_path.name,
        "inference": result.inference.model_dump(),
        "detections": [item.model_dump() for item in result.detections],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the configured fire/smoke YOLO model")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/third_party/yolov8-fire-smoke-v1.0.0/baseline_best.pt"),
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(verify(args.model, args.image)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
