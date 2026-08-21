from app.schemas import KnowledgeEvidence, RiskAssessment, RiskItem, VisionResult


_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class RiskAnalysisService:
    async def analyze(
        self, vision: VisionResult, knowledge: list[KnowledgeEvidence]
    ) -> RiskAssessment:
        if not vision.detections:
            return RiskAssessment(
                overall_level="unknown",
                summary="未获得足以形成风险结论的视觉证据，不能据此判定现场安全。",
                items=[],
                requires_human_review=True,
            )

        knowledge_by_category = {item.matched_category: item for item in knowledge}
        items: list[RiskItem] = []
        for detection in vision.detections:
            if detection.category == "fire":
                level = "critical" if detection.confidence >= 0.75 else "high"
            elif detection.category == "smoke":
                level = "high" if detection.confidence >= 0.75 else "medium"
            else:
                level = "medium"
            matched = knowledge_by_category.get(detection.category)
            actions = matched.recommended_actions if matched else ["由现场安全人员复核目标及周边环境。"]
            items.append(RiskItem(
                category=detection.category,
                level=level,
                reason=(
                    f"视觉工具返回“{detection.label}”，置信度 {detection.confidence:.2f}，"
                    f"候选区域约占图像 {detection.area_ratio:.2%}。"
                ),
                evidence_ids=[detection.evidence_id],
                recommended_actions=actions,
            ))
        overall = max((item.level for item in items), key=lambda value: _ORDER[value])
        level_labels = {"critical": "极高", "high": "高", "medium": "中", "low": "低", "unknown": "待核查"}
        return RiskAssessment(
            overall_level=overall,
            summary=f"共发现 {len(items)} 类候选风险，最高风险等级为{level_labels[overall]}。",
            items=items,
            requires_human_review=True,
        )
