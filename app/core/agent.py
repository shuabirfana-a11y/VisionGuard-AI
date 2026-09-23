import asyncio
from time import perf_counter
from uuid import uuid4

from app.core.tools import ToolDefinition, ToolRegistry
from app.schemas import AgentPlan, AgentTraceStep, AnalysisResponse, FollowUpResponse
from app.services.knowledge import SafetyKnowledgeService
from app.services.images import prepare_image
from app.services.report import SafetyReportService
from app.services.reasoning.base import Reasoner
from app.services.risk import RiskAnalysisService
from app.services.vision.base import VisionDetector


class VisionGuardAgent:
    """Organizes specialist tools and evidence-constrained reasoning into an auditable workflow."""

    def __init__(
        self,
        detector: VisionDetector,
        knowledge: SafetyKnowledgeService,
        risk: RiskAnalysisService,
        report: SafetyReportService,
        reasoner: Reasoner,
    ) -> None:
        self.detector = detector
        self.knowledge = knowledge
        self.risk = risk
        self.report = report
        self.reasoner = reasoner
        self.tools = ToolRegistry()
        self.tools.register(ToolDefinition("vision.detect", "提取结构化视觉风险证据", self._detect))
        self.tools.register(ToolDefinition("knowledge.retrieve", "检索与视觉证据相关的安全知识", self._retrieve))
        self.tools.register(ToolDefinition("risk.analyze", "基于证据和知识生成风险等级与建议", self._assess))
        self.tools.register(ToolDefinition("reasoning.explain", "基于有效证据生成受约束的风险解释", self._explain))
        self.tools.register(ToolDefinition("report.generate", "生成结构化安全辅助报告", self._report))

    async def _detect(self, image_bytes: bytes, file_name: str):
        prepared = await asyncio.to_thread(prepare_image, image_bytes)
        # All branches receive the same oriented pixels, without original EXIF or filename.
        result = await self.detector.detect(prepared.content, "normalized.png")
        result.input_image = prepared.metadata
        return result

    async def _retrieve(self, categories: list[str], task: str):
        return await self.knowledge.retrieve(categories, task)

    async def _assess(self, vision, knowledge):
        return await self.risk.analyze(vision, knowledge)

    async def _explain(self, task: str, vision, knowledge, risk):
        return await self.reasoner.explain(task, vision, knowledge, risk)

    async def _report(self, task: str, vision, knowledge, risk, reasoning):
        return await self.report.generate(task, vision, knowledge, risk, reasoning)

    async def analyze(self, image_bytes: bytes, file_name: str, task: str) -> AnalysisResponse:
        normalized_task = task.strip() or "识别图片中的工业安全风险并给出核查建议"
        trace: list[AgentTraceStep] = []

        step_started = perf_counter()
        plan = self._plan_task(normalized_task)
        trace.append(AgentTraceStep(
            tool="agent.plan",
            status="completed",
            summary=f"识别任务意图：{plan.intent}，规划 {len(plan.tool_sequence)} 个专业工具",
            duration_ms=round((perf_counter() - step_started) * 1000, 2),
            references=plan.tool_sequence,
        ))

        step_started = perf_counter()
        vision = await self.tools.invoke(
            "vision.detect", image_bytes=image_bytes, file_name=file_name
        )
        trace.append(AgentTraceStep(
            tool="vision.detect",
            status="completed",
            summary=(
                f"{vision.inference.backend}/{vision.inference.model_version} 在 "
                f"{vision.inference.inference_ms:.2f} ms 内获得 {len(vision.detections)} 条结构化视觉证据"
            ),
            duration_ms=round((perf_counter() - step_started) * 1000, 2),
            references=[item.evidence_id for item in vision.detections]
            + (
                [vision.fire_classification.evidence_id]
                if vision.fire_classification and vision.fire_classification.available
                else []
            ),
        ))

        categories = {item.category for item in vision.detections}
        if vision.fire_classification and vision.fire_classification.available and vision.fire_classification.prediction:
            categories.add("fire")
        step_started = perf_counter()
        knowledge = await self.tools.invoke(
            "knowledge.retrieve", categories=sorted(categories), task=normalized_task
        )
        trace.append(AgentTraceStep(
            tool="knowledge.retrieve",
            status="completed",
            summary=(
                f"通过 {self.knowledge.retrieval_method} 检索到 "
                f"{len(knowledge)} 条安全知识依据"
            ),
            duration_ms=round((perf_counter() - step_started) * 1000, 2),
            references=[item.citation_id for item in knowledge],
        ))

        step_started = perf_counter()
        risk = await self.tools.invoke("risk.analyze", vision=vision, knowledge=knowledge)
        trace.append(AgentTraceStep(
            tool="risk.analyze",
            status="completed",
            summary=f"形成风险等级：{risk.overall_level}",
            duration_ms=round((perf_counter() - step_started) * 1000, 2),
            references=list(dict.fromkeys(
                evidence_id
                for item in risk.items
                for evidence_id in item.evidence_ids
            )),
        ))

        step_started = perf_counter()
        reasoning = await self.tools.invoke(
            "reasoning.explain",
            task=normalized_task,
            vision=vision,
            knowledge=knowledge,
            risk=risk,
        )
        trace.append(AgentTraceStep(
            tool="reasoning.explain",
            status="completed",
            summary=(
                f"{reasoning.provider}/{reasoning.model} 生成证据约束解释"
                + ("，已启用回退" if reasoning.fallback_used else "")
            ),
            duration_ms=round((perf_counter() - step_started) * 1000, 2),
            references=list(dict.fromkeys(reasoning.evidence_ids + reasoning.knowledge_ids)),
        ))

        step_started = perf_counter()
        report = await self.tools.invoke(
            "report.generate",
            task=normalized_task,
            vision=vision,
            knowledge=knowledge,
            risk=risk,
            reasoning=reasoning,
        )
        trace.append(AgentTraceStep(
            tool="report.generate",
            status="completed",
            summary="生成结构化安全辅助报告",
            duration_ms=round((perf_counter() - step_started) * 1000, 2),
            references=list(dict.fromkeys(reasoning.evidence_ids + reasoning.knowledge_ids)),
        ))

        return AnalysisResponse(
            request_id=str(uuid4()),
            task=normalized_task,
            agent_plan=plan,
            vision=vision,
            knowledge=knowledge,
            risk=risk,
            reasoning=reasoning,
            report=report,
            agent_trace=trace,
        )

    @staticmethod
    def _plan_task(task: str) -> AgentPlan:
        intent_rules = [
            ("风险处置与报告", ("报告", "处置", "建议", "行动", "应急")),
            ("证据解释与依据核查", ("为什么", "原因", "解释", "依据", "法规")),
            ("风险识别与等级研判", ("风险", "危险", "等级", "隐患")),
            ("视觉目标定位", ("检测", "识别", "定位", "标框", "图片")),
        ]
        matched_terms: list[str] = []
        intent = "工业安全综合分析"
        for candidate, terms in intent_rules:
            hits = [term for term in terms if term in task]
            if hits:
                intent = candidate
                matched_terms.extend(hits)
                break
        requested_outputs = ["结构化视觉证据", "风险等级"]
        if any(term in task for term in ("为什么", "原因", "解释", "依据", "风险", "隐患")):
            requested_outputs.extend(["证据约束解释", "安全知识依据"])
        if any(term in task for term in ("建议", "行动", "处理", "处置", "应急", "核查")):
            requested_outputs.append("人工核查建议")
        if "报告" in task:
            requested_outputs.append("结构化安全报告")
        requested_outputs = list(dict.fromkeys(requested_outputs))
        tools = [
            "vision.detect",
            "knowledge.retrieve",
            "risk.analyze",
            "reasoning.explain",
            "report.generate",
        ]
        return AgentPlan(
            intent=intent,
            matched_terms=list(dict.fromkeys(matched_terms)),
            requested_outputs=requested_outputs,
            tool_sequence=tools,
            rationale="先获取可定位视觉证据，再检索适用知识，随后完成风险研判、受约束解释和报告组织。",
            safety_constraints=[
                "只允许引用本次视觉证据编号和检索到的知识编号",
                "未检出不等于安全，模型分歧必须显式提示",
                "结论仅用于辅助分析，必须由现场安全人员复核",
            ],
        )

    async def answer_follow_up(
        self, analysis: AnalysisResponse, question: str
    ) -> FollowUpResponse:
        normalized = question.strip()
        step_started = perf_counter()
        reasoning = await self.tools.invoke(
            "reasoning.explain",
            task=normalized,
            vision=analysis.vision,
            knowledge=analysis.knowledge,
            risk=analysis.risk,
        )
        answer = (
            reasoning.explanation
            if reasoning.used_llm
            else self._deterministic_follow_up(analysis, normalized)
        )
        return FollowUpResponse(
            request_id=analysis.request_id,
            question=normalized,
            answer=answer,
            reasoning=reasoning,
            agent_trace=[
                AgentTraceStep(
                    tool="context.retrieve",
                    status="completed",
                    summary=(
                        f"复用 {len(reasoning.evidence_ids)} 条视觉证据和 "
                        f"{len(reasoning.knowledge_ids)} 条知识依据，未重新执行视觉检测"
                    ),
                    references=list(dict.fromkeys(reasoning.evidence_ids + reasoning.knowledge_ids)),
                ),
                AgentTraceStep(
                    tool="reasoning.explain",
                    status="completed",
                    summary=f"{reasoning.provider}/{reasoning.model} 生成证据约束追问回答",
                    duration_ms=round((perf_counter() - step_started) * 1000, 2),
                    references=list(dict.fromkeys(reasoning.evidence_ids + reasoning.knowledge_ids)),
                ),
            ],
        )

    @staticmethod
    def _deterministic_follow_up(
        analysis: AnalysisResponse, question: str
    ) -> str:
        if any(word in question for word in ("确定", "一定", "安全吗", "误报", "可靠")):
            return (
                "不能仅凭本次单图分析作出确定性安全结论。"
                f"当前系统形成的最高风险等级为{analysis.risk.overall_level}，"
                "仍受图像视角、遮挡、模型适用范围和现场信息缺失影响，必须由现场安全人员复核。"
            )
        if any(word in question for word in ("怎么", "建议", "处理", "处置", "行动")):
            actions: list[str] = []
            for item in analysis.risk.items:
                for action in item.recommended_actions:
                    if action not in actions:
                        actions.append(action)
            if not actions:
                actions = ["结合现场巡检、传感器和其他视角继续核查。"]
            return "基于当前证据，建议按以下顺序核查：" + "；".join(actions[:4])
        evidence = "；".join(item.reason for item in analysis.risk.items)
        citations = "、".join(
            f"[{item.citation_id}]{item.source_title}{item.source_section}"
            for item in analysis.knowledge
        )
        if not evidence:
            evidence = "视觉工具未返回明确风险目标，未检出不等同于安全。"
        return f"判断依据包括：{evidence} 知识依据为：{citations or '系统方法边界'}。"
