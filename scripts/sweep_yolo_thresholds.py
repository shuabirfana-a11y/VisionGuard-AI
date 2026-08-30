from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation import detection_metrics
from app.services.vision.yolo import YoloVisionDetector
from scripts.evaluate_system import load_manifest


THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


async def collect(
    rows,
    model_path: Path,
    *,
    model_version: str,
    expected_sha256: str,
    iou_threshold: float,
    device: str,
):
    detector = YoloVisionDetector(
        str(model_path),
        model_version=model_version,
        expected_sha256=expected_sha256,
        confidence_threshold=min(THRESHOLDS),
        iou_threshold=iou_threshold,
        device=device,
    )
    records = []
    for row in rows:
        result = await detector.detect(row["image"].read_bytes(), row["image"].name)
        records.append({
            "ground_truth": row["ground_truth"] or [],
            "detections": [item.model_dump() for item in result.detections],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "third_party" / "yolov8-fire-smoke-v1.0.0" / "baseline_best.pt",
    )
    parser.add_argument("--model-version", default="")
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = load_manifest(args.manifest.resolve())
    raw_records = asyncio.run(collect(
        rows,
        args.model.resolve(),
        model_version=args.model_version or args.model.stem,
        expected_sha256=args.expected_sha256,
        iou_threshold=args.iou_threshold,
        device=args.device,
    ))
    sweep = []
    for threshold in THRESHOLDS:
        filtered = [{
            "ground_truth": record["ground_truth"],
            "detections": [
                item for item in record["detections"]
                if float(item["confidence"]) >= threshold
            ],
        } for record in raw_records]
        sweep.append({"threshold": threshold, "metrics": detection_metrics(filtered)})
    selected = max(
        sweep,
        key=lambda item: (
            item["metrics"]["box_level"]["overall"]["f1"],
            item["metrics"]["image_level"]["fire"]["recall"],
        ),
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(rows),
        "model": {
            "path": str(args.model.resolve()),
            "version": args.model_version or args.model.stem,
            "expected_sha256": args.expected_sha256 or None,
            "iou_threshold": args.iou_threshold,
            "device": args.device,
        },
        "selection_rule": "最大化框级总体F1；并列时优先火焰图片级Recall。",
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected["metrics"],
        "sweep": sweep,
        "claim_boundary": "该阈值由当前外部固定验证集选择，必须在新的未见测试集上复核后才能作为最终泛化指标。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
