from copy import deepcopy
import pytest
from app.services.validation import EXPECTED_ANALYSIS_TOOLS, assess_analysis_contract


def valid_response():
    return {
        "agent_trace": [{"tool": tool, "status": "completed"} for tool in EXPECTED_ANALYSIS_TOOLS],
        "vision": {"detections": [{"evidence_id": "ev-fire-001"}]},
        "knowledge": [{"rule_id": "VG-FIRE-001"}],
        "reasoning": {
            "explanation": "存在疑似火焰线索，请现场核查。",
            "evidence_ids": ["ev-fire-001"],
            "knowledge_ids": ["VG-FIRE-001"],
            "safety_boundary": "须由现场人员复核。",
        },
    }


def test_complete_trace_and_references_pass():
    result = assess_analysis_contract(valid_response())
    assert result["agent_success"] is True
    assert result["reasoning_grounded"] is True
    assert result["reasoning_contract_errors"] == []


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "failed"])
def test_partial_or_invalid_trace_is_not_counted_as_agent_success(change):
    response = valid_response()
    if change == "missing":
        response["agent_trace"].pop()
    elif change == "duplicate":
        response["agent_trace"].append(deepcopy(response["agent_trace"][-1]))
    elif change == "reordered":
        response["agent_trace"][1:3] = reversed(response["agent_trace"][1:3])
    else:
        response["agent_trace"][2]["status"] = "failed"
    assert assess_analysis_contract(response)["agent_success"] is False


@pytest.mark.parametrize("field,value,error", [
    ("evidence_ids", [], "missing_visual_reference"),
    ("knowledge_ids", [], "missing_knowledge_reference"),
    ("evidence_ids", ["invented"], "unknown_visual_reference"),
    ("knowledge_ids", ["invented"], "unknown_knowledge_reference"),
    ("explanation", " ", "missing_explanation"),
    ("safety_boundary", " ", "missing_safety_boundary"),
])
def test_invalid_reasoning_contract_is_not_counted_as_grounded(field, value, error):
    response = valid_response()
    response["reasoning"][field] = value
    result = assess_analysis_contract(response)
    assert result["reasoning_grounded"] is False
    assert error in result["reasoning_contract_errors"]


def test_no_detection_allows_no_visual_reference():
    response = valid_response()
    response["vision"]["detections"] = []
    response["reasoning"]["evidence_ids"] = []
    response["reasoning"]["explanation"] = "未获得目标，不代表现场安全。"
    assert assess_analysis_contract(response)["reasoning_grounded"] is True


def test_available_classifier_reference_is_valid_but_unavailable_is_not():
    response = valid_response()
    response["vision"]["fire_classification"] = {"available": True, "evidence_id": "ev-classifier"}
    response["reasoning"]["evidence_ids"] = ["ev-classifier"]
    assert assess_analysis_contract(response)["reasoning_grounded"] is True
    response["vision"]["fire_classification"]["available"] = False
    assert assess_analysis_contract(response)["reasoning_grounded"] is False
