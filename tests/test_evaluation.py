import pytest

from app.services.evaluation import (
    binary_metrics,
    box_iou,
    detection_metrics,
    percentile,
    summarize_runs,
)


def test_binary_metrics_are_computed_from_confusion_counts():
    result = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["tp"] == result["fp"] == result["fn"] == result["tn"] == 1
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_percentile_interpolates_and_validates_input():
    assert percentile([10, 20, 30], 0.95) == pytest.approx(29)
    with pytest.raises(ValueError):
        percentile([], 0.95)


def test_summary_keeps_system_and_classifier_failures_separate():
    records = [
        {
            "image": "a.png",
            "success": True,
            "label": 1,
            "prediction": 1,
            "total_ms": 100,
            "agent_success": True,
            "fusion_status": "confirmed",
            "risk_level": "critical",
            "vision_fallback": False,
            "classifier_available": True,
            "reasoning_used_llm": True,
            "reasoning_fallback": False,
            "reasoning_grounded": True,
        },
        {"image": "b.png", "success": False, "label": 0, "error": "timeout"},
    ]
    result = summarize_runs(records)
    assert result["request_success_rate"] == 0.5
    assert result["classifier_metrics"]["f1"] == 1.0
    assert result["failed_samples"] == 1
    assert result["llm_usage_rate"] == 1.0
    assert result["reasoning_fallback_count"] == 0
    assert result["grounded_reasoning_rate"] == 1.0
    assert result["failures"] == [{"image": "b.png", "error": "timeout"}]


def test_detection_metrics_match_boxes_by_class_and_iou():
    records = [{
        "ground_truth": [
            {"category": "fire", "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
            {"category": "smoke", "bbox": {"x1": 60, "y1": 10, "x2": 90, "y2": 50}},
        ],
        "detections": [
            {"category": "fire", "confidence": 0.9, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
            {"category": "fire", "confidence": 0.4, "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}},
        ],
    }]
    result = detection_metrics(records)
    assert result["box_level"]["overall"]["tp"] == 1
    assert result["box_level"]["overall"]["fp"] == 1
    assert result["box_level"]["overall"]["fn"] == 1
    assert result["image_level"]["fire"]["recall"] == 1
    assert result["image_level"]["smoke"]["recall"] == 0
    assert box_iou(records[0]["ground_truth"][0]["bbox"], records[0]["detections"][0]["bbox"]) == 1
