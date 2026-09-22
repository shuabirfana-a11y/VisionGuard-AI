"""Evaluation contracts check structure and references, not semantic correctness."""
from typing import Any

EXPECTED_ANALYSIS_TOOLS = (
    "agent.plan", "vision.detect", "knowledge.retrieve",
    "risk.analyze", "reasoning.explain", "report.generate",
)


def assess_analysis_contract(body: dict[str, Any]) -> dict[str, Any]:
    trace = body.get("agent_trace") or []
    actual_tools = [step.get("tool") for step in trace]
    agent_ok = actual_tools == list(EXPECTED_ANALYSIS_TOOLS) and all(
        step.get("status") == "completed" for step in trace
    )
    vision = body.get("vision") or {}
    classifier = vision.get("fire_classification") or {}
    available_evidence = {
        item["evidence_id"] for item in vision.get("detections") or []
        if item.get("evidence_id")
    }
    if classifier.get("available") and classifier.get("evidence_id"):
        available_evidence.add(classifier["evidence_id"])
    available_knowledge = {
        item["rule_id"] for item in body.get("knowledge") or [] if item.get("rule_id")
    }
    reasoning = body.get("reasoning") or {}
    evidence = set(reasoning.get("evidence_ids") or [])
    knowledge = set(reasoning.get("knowledge_ids") or [])
    errors = []
    if not evidence.issubset(available_evidence):
        errors.append("unknown_visual_reference")
    if available_evidence and not evidence:
        errors.append("missing_visual_reference")
    if not knowledge.issubset(available_knowledge):
        errors.append("unknown_knowledge_reference")
    if available_knowledge and not knowledge:
        errors.append("missing_knowledge_reference")
    if not str(reasoning.get("explanation") or "").strip():
        errors.append("missing_explanation")
    if not str(reasoning.get("safety_boundary") or "").strip():
        errors.append("missing_safety_boundary")
    return {
        "agent_success": agent_ok,
        "agent_tool_sequence": actual_tools,
        "reasoning_grounded": not errors,
        "reasoning_contract_errors": errors,
    }
