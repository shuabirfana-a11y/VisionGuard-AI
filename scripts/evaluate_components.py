from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation import binary_metrics, detection_metrics, percentile
from app.services.vision.yolo import YoloVisionDetector
from scripts.evaluate_system import load_manifest


MODEL_SHA256 = "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"


async def evaluate_yolo(rows: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    detector = YoloVisionDetector(
        str(model_path),
        model_version="fire-smoke-yolov8n-upstream-v1.0.0",
        expected_sha256=MODEL_SHA256,
        confidence_threshold=0.25,
        iou_threshold=0.30,
        device="cpu",
    )
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        result = await detector.detect(row["image"].read_bytes(), row["image"].name)
        records.append({
            "ground_truth": row["ground_truth"] or [],
            "detections": [item.model_dump() for item in result.detections],
        })
        latencies.append(result.inference.inference_ms)
    return {
        "samples": len(records),
        "metrics": detection_metrics(records),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p95": percentile(latencies, 0.95),
            "maximum": max(latencies),
        },
    }


def evaluate_classifier(
    rows: list[dict[str, Any]], runtime_root: Path, python_executable: Path
) -> dict[str, Any]:
    batch_root = PROJECT_ROOT / "evaluation" / "data" / "classifier_batch"
    if batch_root.exists():
        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True)
    truth_by_name: dict[str, int] = {}
    for index, row in enumerate(rows):
        name = f"sample_{index:03d}{row['image'].suffix.lower()}"
        shutil.copy2(row["image"], batch_root / name)
        truth_by_name[name] = int(row["label"])
    output = PROJECT_ROOT / "evaluation" / "component_classifier_predictions.json"
    scores = PROJECT_ROOT / "evaluation" / "component_classifier_scores.csv"
    command = [
        str(python_executable),
        str(runtime_root / "scripts" / "predict.py"),
        "--images", str(batch_root),
        "--bundle", str(runtime_root / "artifacts" / "final_ensemble"),
        "--output", str(output),
        "--scores-csv", str(scores),
        "--expected-count", str(len(rows)),
        "--device", "cpu",
        "--batch-size", "8",
        "--workers", "0",
    ]
    started = perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    elapsed_ms = (perf_counter() - started) * 1000
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    with scores.open("r", encoding="utf-8-sig", newline="") as stream:
        score_rows = list(csv.DictReader(stream))
    if len(score_rows) != len(rows):
        raise RuntimeError(f"classifier returned {len(score_rows)} rows, expected {len(rows)}")
    predictions: dict[str, int] = {}
    probabilities: dict[str, float] = {}
    for row in score_rows:
        name = Path(row.get("filename") or row.get("file") or row.get("image") or "").name
        predictions[name] = int(row["prediction"])
        probabilities[name] = float(row["fused_score"])
    missing = sorted(set(truth_by_name) - set(predictions))
    if missing:
        raise RuntimeError(f"classifier scores missing files: {missing[:3]}")
    metrics = binary_metrics(
        [truth_by_name[name] for name in sorted(truth_by_name)],
        [predictions[name] for name in sorted(truth_by_name)],
    )
    return {
        "samples": len(rows),
        "metrics": metrics,
        "elapsed_ms": elapsed_ms,
        "throughput_images_per_second": len(rows) / (elapsed_ms / 1000),
        "score_range": {"minimum": min(probabilities.values()), "maximum": max(probabilities.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "third_party" / "yolov8-fire-smoke-v1.0.0" / "baseline_best.pt",
    )
    parser.add_argument(
        "--classifier-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "fire_highscore_submission",
    )
    parser.add_argument(
        "--classifier-python",
        type=Path,
        default=PROJECT_ROOT / ".venv-firebench" / "Scripts" / "python.exe",
    )
    args = parser.parse_args()
    rows = load_manifest(args.manifest.resolve())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "selection_boundary": "固定跨来源清单；未依据模型结果选样。",
        "yolo": asyncio.run(evaluate_yolo(rows, args.model.resolve())),
        "fire_classifier": evaluate_classifier(
            rows, args.classifier_root.resolve(), args.classifier_python.resolve()
        ),
        "claim_boundary": (
            "YOLO声明训练源为D-Fire，本清单来自IFireSmoke与Pyro-SDIS。"
            "分类模型训练图片来源未完整披露，因此仅称外部固定集验证。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
