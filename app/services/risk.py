from app.schemas import KnowledgeEvidence, RiskAssessment, RiskItem, VisionResult


_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class RiskAnalysisService:
    async def analyze(
        self, vision: VisionResult, knowledge: list[KnowledgeEvidence]
    ) -> RiskAssessment:
        classifier = vision.fire_classification
        classifier_positive = bool(
            classifier and classifier.available and classifier.prediction
        )
        if not vision.detections and not classifier_positive:
            return RiskAssessment(
                overall_level="unknown",
                summary="未获得足以形成风险结论的视觉证据，不能据此判定现场安全。",
                items=[],
                requires_human_review=True,
            )

        knowledge_by_category: dict[str, KnowledgeEvidence] = {}
        for item in knowledge:
            knowledge_by_category.setdefault(item.matched_category, item)
        items: list[RiskItem] = []
        fire_item_index: int | None = None
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
            if detection.category == "fire" and fire_item_index is None:
                fire_item_index = len(items) - 1

        if classifier_positive and classifier is not None:
            matched = knowledge_by_category.get("fire")
            actions = matched.recommended_actions if matched else ["由现场安全人员复核原图及周边环境。"]
            if fire_item_index is None:
                items.append(RiskItem(
                    category="fire",
                    level="high",
                    reason=(
                        f"整图火情分类模型返回可见火焰概率 {classifier.probability:.2f}，"
                        "但目标检测未提供定位框，可能涉及远距离或极小火焰。"
                    ),
                    evidence_ids=[classifier.evidence_id],
                    recommended_actions=actions,
                ))
            else:
                fire_item = items[fire_item_index]
                fire_item.evidence_ids.append(classifier.evidence_id)
                fire_item.reason += (
                    f" 整图分类模型同时给出可见火焰概率 {classifier.probability:.2f}。"
                )
        elif (
            classifier is not None
            and classifier.available
            and classifier.prediction is False
            and fire_item_index is not None
        ):
            items[fire_item_index].evidence_ids.append(classifier.evidence_id)
            items[fire_item_index].reason += " 整图分类模型未确认火情，视觉证据存在冲突。"
        overall = max((item.level for item in items), key=lambda value: _ORDER[value])
        level_labels = {"critical": "极高", "high": "高", "medium": "中", "low": "低", "unknown": "待核查"}
        return RiskAssessment(
            overall_level=overall,
            summary=(
                f"共形成 {len(items)} 项候选风险，最高风险等级为{level_labels[overall]}。"
                + (f" {vision.fusion.summary}" if vision.fusion else "")
            ),
            items=items,
            requires_human_review=True,
        )
