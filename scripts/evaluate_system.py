#!/usr/bin/env python3
"""Evaluate a running VisionGuard API without inventing unavailable results."""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation import summarize_runs
from app.services.validation import assess_analysis_contract


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"测试清单不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            raw_image = (row.get("image") or "").strip()
            if not raw_image:
                raise ValueError(f"测试清单第{line_number}行缺少image")
            image = Path(raw_image)
            if not image.is_absolute():
                image = (path.parent / image).resolve()
            label_text = (row.get("label") or "").strip()
            if label_text not in {"", "0", "1"}:
                raise ValueError(f"测试清单第{line_number}行label必须为空、0或1")
            ground_truth_text = (row.get("ground_truth_json") or "").strip()
            ground_truth = json.loads(ground_truth_text) if ground_truth_text else None
            if ground_truth is not None and not isinstance(ground_truth, list):
                raise ValueError(f"测试清单第{line_number}行ground_truth_json必须是JSON数组")
            rows.append(
                {
                    "image": image,
                    "label": int(label_text) if label_text else None,
                    "task": (row.get("task") or "识别图片中的工业安全风险并给出核查建议").strip(),
                    "ground_truth": ground_truth,
                    "source": (row.get("source") or "").strip(),
                    "license": (row.get("license") or "").strip(),
                    "group": (row.get("group") or "").strip(),
                }
            )
    if not rows:
        raise ValueError("测试清单不能为空")
    return rows


async def evaluate(
    base_url: str, rows: list[dict[str, Any]], timeout: float, concurrency: int
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        # Establish a shared cookie before concurrent requests start, so reports
        # and metrics belong to one evaluation session rather than racing cookies.
        session_response = await client.get("/api/v1/health")
        session_response.raise_for_status()
        async def evaluate_one(row: dict[str, Any]) -> dict[str, Any]:
            image: Path = row["image"]
            record: dict[str, Any] = {
                "image": str(image),
                "label": row["label"],
                "source": row["source"],
                "license": row["license"],
                "group": row["group"],
                "ground_truth_available": row["ground_truth"] is not None,
                "ground_truth": row["ground_truth"] or [],
                "success": False,
            }
            if not image.is_file():
                record["error"] = "图片不存在"
                return record
            async with semaphore:
                started = perf_counter()
                try:
                    content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
                    response = await client.post(
                        "/api/v1/analyze",
                        files={"image": (image.name, image.read_bytes(), content_type)},
                        data={"task": row["task"]},
                    )
                    record["total_ms"] = round((perf_counter() - started) * 1000, 2)
                    response.raise_for_status()
                    body = response.json()
                    classifier = body["vision"].get("fire_classification") or {}
                    reasoning = body.get("reasoning") or {}
                    contract = assess_analysis_contract(body)
                    record.update(
                        {
                            "success": True,
                            "prediction": (
                                int(classifier["prediction"])
                                if classifier.get("available") and classifier.get("prediction") is not None
                                else None
                            ),
                            "probability": classifier.get("probability"),
                            "classifier_available": bool(classifier.get("available")),
                            "classifier_inference_ms": classifier.get("inference_ms"),
                            "fusion_status": (body["vision"].get("fusion") or {}).get("status"),
                            "risk_level": body["risk"]["overall_level"],
                            "detections": body["vision"].get("detections") or [],
                            "vision_fallback": bool(body["vision"]["inference"].get("fallback_used")),
                            **contract,
                            "reasoning_used_llm": bool(reasoning.get("used_llm")),
                            "reasoning_fallback": bool(reasoning.get("fallback_used")),
                            "reasoning_fallback_reason": reasoning.get("fallback_reason"),
                            "reasoning_explanation": reasoning.get("explanation"),
                            "reasoning_evidence_ids": reasoning.get("evidence_ids") or [],
                            "reasoning_knowledge_ids": reasoning.get("knowledge_ids") or [],
                            "reasoning_safety_boundary": reasoning.get("safety_boundary"),
                            "inference_metadata": body["vision"]["inference"],
                            "input_image": body["vision"].get("input_image"),
                        }
                    )
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
            return record

        return await asyncio.gather(*(evaluate_one(row) for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout必须大于0")
    if args.concurrency <= 0:
        raise ValueError("concurrency必须大于0")
    rows = load_manifest(args.manifest.resolve())
    records = asyncio.run(evaluate(args.base_url, rows, args.timeout, args.concurrency))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "manifest": str(args.manifest.resolve()),
        "concurrency": args.concurrency,
        "contract_version": "trace-and-reference-v2",
        "metric_definitions": {
            "agent_success_rate": "All six required steps completed exactly once and in order; not a task-understanding score.",
            "grounded_reasoning_rate": "Required references are valid and explanation/boundary are nonempty; not semantic correctness or expert approval.",
            "latency_ms": "Successful requests only; excludes time waiting for the client concurrency semaphore.",
        },
        "summary": summarize_runs(records),
        "records": records,
        "claim_boundary": "本报告仅适用于清单、模型版本、阈值、硬件和运行配置对应的本次测试。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["failed_samples"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
