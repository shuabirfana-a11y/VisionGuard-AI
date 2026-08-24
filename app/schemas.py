from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high", "critical", "unknown"]
FusionStatus = Literal[
    "confirmed",
    "classifier_only",
    "detector_only",
    "no_fire_evidence",
    "classifier_unavailable",
]


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


class FireClassificationEvidence(BaseModel):
    evidence_id: str = "cls-fire-001"
    category: Literal["fire"] = "fire"
    label: str = "visible_fire"
    available: bool
    prediction: bool | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    source: str
    model_name: str
    model_version: str
    model_digest: str | None = None
    device: str
    inference_ms: float = Field(ge=0)
    limitations: list[str] = Field(default_factory=list)
    error: str | None = None


class VisionFusionResult(BaseModel):
    status: FusionStatus
    summary: str
    detector_fire_count: int = Field(ge=0)
    classifier_fire: bool | None = None
    requires_human_review: bool = True


class VisionResult(BaseModel):
    image_width: int
    image_height: int
    detector: str
    inference: InferenceMetadata
    detections: list[DetectionEvidence]
    fire_classification: FireClassificationEvidence | None = None
    fusion: VisionFusionResult | None = None
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
    source_url: str | None = None
    authority_level: Literal["law", "department_rule", "internal_method"] = "internal_method"
    applicability: str = ""
    citation_id: str
    retrieval_score: float = Field(ge=0, le=1)
    retrieval_method: str = "category-constrained-tfidf-rag-v1"
    matched_terms: list[str] = Field(default_factory=list)


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
    duration_ms: float = Field(default=0, ge=0)
    references: list[str] = Field(default_factory=list)


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


class FollowUpRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class FollowUpResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    reasoning: ReasoningResult
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
    source_note: str = "VisionGuard AI合成案例"
    license: str = "项目内生成"
