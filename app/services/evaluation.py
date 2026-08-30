from __future__ import annotations

from collections import Counter
import math
from statistics import mean, median
from typing import Any, Iterable


def binary_metrics(truth: Iterable[int], prediction: Iterable[int]) -> dict[str, Any]:
    labels = list(truth)
    predictions = list(prediction)
    if len(labels) != len(predictions):
        raise ValueError("真实标签与预测数量不一致")
    if not labels:
        raise ValueError("至少需要一条有标签且推理成功的样本")
    if any(value not in {0, 1} for value in labels + predictions):
        raise ValueError("二分类标签必须为0或1")
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(labels, predictions))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(labels, predictions))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(labels, predictions))
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("百分位数至少需要一个样本")
    if not 0 <= quantile <= 1:
        raise ValueError("分位点必须位于0到1之间")
    position = (len(samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)


def box_iou(left: dict[str, float], right: dict[str, float]) -> float:
    intersection_width = max(0.0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"]))
    intersection_height = max(0.0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left["x2"] - left["x1"]) * max(0.0, left["y2"] - left["y1"])
    right_area = max(0.0, right["x2"] - right["x1"]) * max(0.0, right["y2"] - right["y1"])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def detection_metrics(
    records: list[dict[str, Any]], iou_threshold: float = 0.5
) -> dict[str, Any]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU阈值必须位于0到1之间")
    categories = ("fire", "smoke")
    counts = {category: {"tp": 0, "fp": 0, "fn": 0} for category in categories}
    image_truth: dict[str, list[int]] = {category: [] for category in categories}
    image_prediction: dict[str, list[int]] = {category: [] for category in categories}

    for record in records:
        expected = record.get("ground_truth") or []
        predicted = record.get("detections") or []
        for category in categories:
            truth_boxes = [item["bbox"] for item in expected if item.get("category") == category]
            predictions = sorted(
                [item for item in predicted if item.get("category") == category],
                key=lambda item: float(item.get("confidence", 0)),
                reverse=True,
            )
            matched: set[int] = set()
            for prediction in predictions:
                candidates = [
                    (box_iou(prediction["bbox"], box), index)
                    for index, box in enumerate(truth_boxes)
                    if index not in matched
                ]
                best_iou, best_index = max(candidates, default=(0.0, -1))
                if best_iou >= iou_threshold:
                    matched.add(best_index)
                    counts[category]["tp"] += 1
                else:
                    counts[category]["fp"] += 1
            counts[category]["fn"] += len(truth_boxes) - len(matched)
            image_truth[category].append(int(bool(truth_boxes)))
            image_prediction[category].append(int(bool(predictions)))

    def summarize_count(values: dict[str, int]) -> dict[str, Any]:
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**values, "precision": precision, "recall": recall, "f1": f1}

    overall_counts = {
        key: sum(counts[category][key] for category in categories)
        for key in ("tp", "fp", "fn")
    }
    return {
        "iou_threshold": iou_threshold,
        "box_level": {
            "overall": summarize_count(overall_counts),
            **{category: summarize_count(counts[category]) for category in categories},
        },
        "image_level": {
            category: binary_metrics(image_truth[category], image_prediction[category])
            for category in categories
        },
    }


def summarize_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [record for record in records if record.get("success")]
    failed = [record for record in records if not record.get("success")]
    latencies = [float(record["total_ms"]) for record in succeeded]
    labeled = [
        record
        for record in succeeded
        if record.get("label") in {0, 1} and record.get("prediction") in {0, 1}
    ]
    metrics = (
        binary_metrics(
            [int(record["label"]) for record in labeled],
            [int(record["prediction"]) for record in labeled],
        )
        if labeled
        else None
    )
    detection_records = [
        record
        for record in succeeded
        if record.get("ground_truth_available") is True
    ]
    return {
        "requested_samples": len(records),
        "successful_samples": len(succeeded),
        "failed_samples": len(failed),
        "request_success_rate": len(succeeded) / len(records) if records else 0.0,
        "agent_success_rate": (
            sum(bool(record.get("agent_success")) for record in succeeded) / len(succeeded)
            if succeeded
            else 0.0
        ),
        "llm_usage_rate": (
            sum(bool(record.get("reasoning_used_llm")) for record in succeeded) / len(succeeded)
            if succeeded
            else 0.0
        ),
        "reasoning_fallback_count": sum(
            bool(record.get("reasoning_fallback")) for record in succeeded
        ),
        "grounded_reasoning_rate": (
            sum(bool(record.get("reasoning_grounded")) for record in succeeded) / len(succeeded)
            if succeeded
            else 0.0
        ),
        "classifier_metrics": metrics,
        "detection_metrics": detection_metrics(detection_records) if detection_records else None,
        "latency_ms": (
            {
                "mean": mean(latencies),
                "median": median(latencies),
                "p95": percentile(latencies, 0.95),
                "maximum": max(latencies),
            }
            if latencies
            else None
        ),
        "fusion_status_counts": dict(Counter(record.get("fusion_status") for record in succeeded)),
        "risk_level_counts": dict(Counter(record.get("risk_level") for record in succeeded)),
        "vision_fallback_count": sum(bool(record.get("vision_fallback")) for record in succeeded),
        "classifier_unavailable_count": sum(
            not bool(record.get("classifier_available")) for record in succeeded
        ),
        "failures": [
            {"image": record.get("image"), "error": record.get("error")}
            for record in failed
        ],
    }
