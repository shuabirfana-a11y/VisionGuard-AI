from app.schemas import KnowledgeEvidence, ReasoningResult, RiskAssessment, SafetyReport, VisionResult


class SafetyReportService:
    async def generate(
        self,
        task: str,
        vision: VisionResult,
        knowledge: list[KnowledgeEvidence],
        risk: RiskAssessment,
        reasoning: ReasoningResult,
    ) -> SafetyReport:
        evidence = [
            f"{item.evidence_id}：{item.label}，置信度 {item.confidence:.2f}，位置 "
            f"({item.bbox.x1}, {item.bbox.y1})-({item.bbox.x2}, {item.bbox.y2})"
            for item in vision.detections
        ]
        if vision.fire_classification and vision.fire_classification.available:
            classification = vision.fire_classification
            evidence.append(
                f"{classification.evidence_id}：整图可见火焰判断 "
                f"{'阳性' if classification.prediction else '阴性'}，概率 {classification.probability:.2f}，"
                "该证据不提供检测框"
            )
        if not evidence:
            evidence = ["未获得明确风险目标证据。"]
        basis = [
            f"[{item.citation_id}] {item.source_title} / {item.source_section} / {item.source_version}；"
            f"适用边界：{item.applicability or '须结合现场适用条件复核'}"
            for item in knowledge
        ]
        actions: list[str] = []
        for item in risk.items:
            for action in item.recommended_actions:
                if action not in actions:
                    actions.append(action)
        if not actions:
            actions = ["保持现场巡检，并由安全人员结合原图和现场信息复核。"]
        return SafetyReport(
            title="VisionGuard AI 工业安全视觉风险辅助报告",
            task=task,
            conclusion=risk.summary,
            evidence_summary=evidence,
            knowledge_basis=basis,
            reasoning_summary=reasoning.explanation,
            uncertainties=reasoning.uncertainties,
            actions=actions,
            disclaimer="本报告为人工智能辅助分析结果，不替代现场核查、专业鉴定和企业应急处置制度。",
        )
