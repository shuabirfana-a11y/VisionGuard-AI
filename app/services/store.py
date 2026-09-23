from collections import Counter, deque
from datetime import datetime, timezone
import re

from app.schemas import AnalysisRecord, AnalysisResponse, MetricsResponse


class AnalysisStore:
    """Bounded in-memory store. Images and API credentials are never persisted."""

    def __init__(self, max_records: int = 100) -> None:
        self._records: deque[AnalysisRecord] = deque(maxlen=max_records)
        self._results: dict[str, AnalysisResponse] = {}
        self._owners: dict[str, str] = {}

    def record(self, result: AnalysisResponse, total_ms: float, *, owner: str) -> AnalysisRecord:
        if not owner:
            raise ValueError("Analysis owner is required")
        if len(self._records) == self._records.maxlen and self._records:
            oldest = self._records[0]
            self._results.pop(oldest.request_id, None)
            self._owners.pop(oldest.request_id, None)
        sanitized = result.model_copy(deep=True)
        sanitized.task = self._redact(result.task)[:300]
        sanitized.report.task = sanitized.task
        record = AnalysisRecord(
            request_id=sanitized.request_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            task=sanitized.task,
            overall_level=sanitized.risk.overall_level,
            detection_count=len(sanitized.vision.detections),
            vision_backend=sanitized.vision.inference.backend,
            model_version=sanitized.vision.inference.model_version,
            inference_ms=sanitized.vision.inference.inference_ms,
            total_ms=round(total_ms, 2),
            vision_fallback_used=sanitized.vision.inference.fallback_used,
            reasoning_provider=sanitized.reasoning.provider,
            reasoning_fallback_used=sanitized.reasoning.fallback_used,
        )
        self._records.append(record)
        self._results[sanitized.request_id] = sanitized
        self._owners[sanitized.request_id] = owner
        return record

    @staticmethod
    def _redact(value: str) -> str:
        value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", value)
        value = re.sub(r"(?<!\d)1\d{10}(?!\d)", "[phone]", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[api-key]", value)
        value = re.sub(r"(?<!\d)\d{15,18}[0-9Xx]?(?!\d)", "[identifier]", value)
        return value

    def list_records(self, *, owner: str) -> list[AnalysisRecord]:
        return [item for item in reversed(self._records) if self._owners.get(item.request_id) == owner]

    def get(self, request_id: str, *, owner: str) -> AnalysisResponse | None:
        if self._owners.get(request_id) != owner:
            return None
        return self._results.get(request_id)

    def metrics(self, *, owner: str) -> MetricsResponse:
        records = self.list_records(owner=owner)
        total = len(records)
        return MetricsResponse(
            total_analyses=total,
            average_total_ms=round(sum(item.total_ms for item in records) / total, 2) if total else 0,
            average_inference_ms=round(sum(item.inference_ms for item in records) / total, 2) if total else 0,
            vision_fallback_count=sum(item.vision_fallback_used for item in records),
            reasoning_fallback_count=sum(item.reasoning_fallback_used for item in records),
            risk_level_counts=dict(Counter(item.overall_level for item in records)),
            backend_counts=dict(Counter(item.vision_backend for item in records)),
        )
