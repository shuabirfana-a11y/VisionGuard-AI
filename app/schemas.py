from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high", "critical", "unknown"]


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionEvidence(BaseModel):
    evidence_id: str
    category: str
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    area_ratio: float = Field(ge=0, le=1)
    source: str


class InferenceMetadata(BaseModel):
    backend: str
    model_name: str
    model_version: str
    model_digest: str | None = None
    confidence_threshold: float = Field(ge=0, le=1)
    iou_threshold: float | None = Field(default=None, ge=0, le=1)
    device: str
    inference_ms: float = Field(ge=0)
    fallback_used: bool = False
    fallback_reason: str | None = None


class VisionResult(BaseModel):
    image_width: int
    image_height: int
    detector: str
    inference: InferenceMetadata
    detections: list[DetectionEvidence]
    limitations: list[str] = Field(default_factory=list)


class KnowledgeEvidence(BaseModel):
    rule_id: str
    title: str
    matched_category: str
    basis: str
    recommended_actions: list[str]
    source: str
    source_title: str
    source_section: str
    source_version: str
    citation_id: str
    retrieval_score: float = Field(ge=0, le=1)


class RiskItem(BaseModel):
    category: str
    level: RiskLevel
    reason: str
    evidence_ids: list[str]
    recommended_actions: list[str]


class RiskAssessment(BaseModel):
    overall_level: RiskLevel
    summary: str
    items: list[RiskItem]
    requires_human_review: bool = True


class ReasoningResult(BaseModel):
    provider: str
    model: str
    used_llm: bool
    explanation: str
    evidence_ids: list[str]
    knowledge_ids: list[str]
    uncertainties: list[str]
    follow_up_questions: list[str]
    safety_boundary: str
    fallback_used: bool = False
    fallback_reason: str | None = None


class AgentTraceStep(BaseModel):
    tool: str
    status: Literal["completed", "skipped", "failed"]
    summary: str


class SafetyReport(BaseModel):
    title: str
    task: str
    conclusion: str
    evidence_summary: list[str]
    knowledge_basis: list[str]
    reasoning_summary: str
    uncertainties: list[str]
    actions: list[str]
    disclaimer: str


class AnalysisResponse(BaseModel):
    request_id: str
    task: str
    vision: VisionResult
    knowledge: list[KnowledgeEvidence]
    risk: RiskAssessment
    reasoning: ReasoningResult
    report: SafetyReport
    agent_trace: list[AgentTraceStep]


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    vision_backend: str
    capabilities: dict[str, Any]


class AnalysisRecord(BaseModel):
    request_id: str
    created_at: str
    task: str
    overall_level: RiskLevel
    detection_count: int
    vision_backend: str
    model_version: str
    inference_ms: float
    total_ms: float
    vision_fallback_used: bool
    reasoning_provider: str
    reasoning_fallback_used: bool


class MetricsResponse(BaseModel):
    total_analyses: int
    average_total_ms: float
    average_inference_ms: float
    vision_fallback_count: int
    reasoning_fallback_count: int
    risk_level_counts: dict[str, int]
    backend_counts: dict[str, int]


class DemoCaseInfo(BaseModel):
    case_id: str
    name: str
    description: str
    expected_signal: str
    synthetic: bool = True
