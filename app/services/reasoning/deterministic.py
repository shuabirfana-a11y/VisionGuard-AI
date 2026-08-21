from app.schemas import KnowledgeEvidence, ReasoningResult, RiskAssessment, VisionResult


class DeterministicReasoner:
    name = "evidence-constrained-template"
    model = "rule-v1"

    async def explain(
        self,
        task: str,
        vision: VisionResult,
        knowledge: list[KnowledgeEvidence],
        risk: RiskAssessment,
    ) -> ReasoningResult:
        evidence_ids = [item.evidence_id for item in vision.detections]
        knowledge_ids = [item.rule_id for item in knowledge]
        if evidence_ids:
            category_labels = {"fire": "疑似火焰", "smoke": "疑似烟雾"}
            level_labels = {"critical": "极高", "high": "高", "medium": "中", "low": "低", "unknown": "待核查"}
            categories = "、".join(sorted({category_labels.get(item.category, item.category) for item in vision.detections}))
            explanation = (
                f"视觉工具提供了{categories}候选证据，风险模块依据证据置信度、区域占比和"
                f"{len(knowledge_ids)}条知识依据形成{level_labels[risk.overall_level]}风险辅助判断。"
            )
            uncertainties = list(vision.limitations)
        else:
            explanation = "视觉工具未返回明确风险目标，当前信息不足以形成安全结论。"
            uncertainties = ["单张图片可能受遮挡、画质、视角和模型能力限制。"]
        if vision.inference.fallback_used:
            uncertainties.insert(0, f"视觉环节发生Demo回退：{vision.inference.fallback_reason}")
        return ReasoningResult(
            provider=self.name,
            model=self.model,
            used_llm=False,
            explanation=explanation,
            evidence_ids=evidence_ids,
            knowledge_ids=knowledge_ids,
            uncertainties=uncertainties,
            follow_up_questions=["是否有相邻时段、其他视角或现场传感器信息可供复核？"],
            safety_boundary="结论仅用于风险辅助分析，必须由现场安全人员结合实际工况复核。",
        )
