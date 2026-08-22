from uuid import uuid4

from app.core.tools import ToolDefinition, ToolRegistry
from app.schemas import AgentTraceStep, AnalysisResponse, FollowUpResponse
from app.services.knowledge import SafetyKnowledgeService
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
        return await self.detector.detect(image_bytes, file_name)

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
        ))

        categories = {item.category for item in vision.detections}
        if vision.fire_classification and vision.fire_classification.available and vision.fire_classification.prediction:
            categories.add("fire")
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
        ))

        risk = await self.tools.invoke("risk.analyze", vision=vision, knowledge=knowledge)
        trace.append(AgentTraceStep(
            tool="risk.analyze",
            status="completed",
            summary=f"形成风险等级：{risk.overall_level}",
        ))

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
        ))

        report = await self.tools.invoke(
            "report.generate",
            task=normalized_task,
            vision=vision,
            knowledge=knowledge,
            risk=risk,
            reasoning=reasoning,
        )
        trace.append(AgentTraceStep(
            tool="report.generate", status="completed", summary="生成结构化安全辅助报告"
        ))

        return AnalysisResponse(
            request_id=str(uuid4()),
            task=normalized_task,
            vision=vision,
            knowledge=knowledge,
            risk=risk,
            reasoning=reasoning,
            report=report,
            agent_trace=trace,
        )

    async def answer_follow_up(
        self, analysis: AnalysisResponse, question: str
    ) -> FollowUpResponse:
        normalized = question.strip()
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
                ),
                AgentTraceStep(
                    tool="reasoning.explain",
                    status="completed",
                    summary=f"{reasoning.provider}/{reasoning.model} 生成证据约束追问回答",
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
