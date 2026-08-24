import hashlib
from pathlib import Path
from time import perf_counter

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.config import settings
from app.schemas import (
    AnalysisRecord,
    AnalysisResponse,
    DemoCaseInfo,
    FollowUpRequest,
    FollowUpResponse,
    HealthResponse,
    MetricsResponse,
    ReadinessCheck,
    ReadinessResponse,
)
from app.services.demo_cases import CASES, render_demo_case
from app.services.exports import report_html, report_json
from app.services.vision.demo import InvalidImageError


router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from app.main import agent, vision_runtime_status

    return HealthResponse(
        status="ok",
        project="VisionGuard AI",
        version="0.12.0",
        vision_backend=settings.vision_backend,
        capabilities={
            "image_upload": "available",
            "agent_tool_calling": "available",
            "structured_risk_report": "available",
            "traceable_knowledge_retrieval": "available",
            "evidence_grounded_follow_up": "available",
            "reasoning_backend": settings.reasoning_backend,
            "llm_reasoning": (
                "configured"
                if settings.reasoning_backend == "llm" and settings.llm_base_url and settings.llm_api_key and settings.llm_model
                else "deterministic_or_fallback"
            ),
            "llm_model": settings.llm_model or None,
            "llm_guardrails": [
                "structured_json",
                "evidence_id_allowlist",
                "rule_id_allowlist",
                "deterministic_fallback",
            ],
            "vector_rag": "category-constrained-tfidf-rag-v1",
            "yolo_model_configured": bool(settings.yolo_model_path),
            "fire_classifier_enabled": settings.fire_classifier_enabled,
            "fire_classifier_configured": bool(settings.fire_classifier_root),
            "fire_classifier_mode": settings.fire_classifier_mode,
            "fire_classifier_worker_count": settings.fire_classifier_worker_count,
            "fire_classifier_runtime": dict(vision_runtime_status),
            "vision_fallback_enabled": settings.vision_fallback_enabled,
            "registered_tools": agent.tools.descriptions(),
        },
    )


@router.get("/readiness", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    from app.main import agent, vision_runtime_status

    checks: list[ReadinessCheck] = []
    model_path = Path(settings.yolo_model_path) if settings.yolo_model_path else None
    if settings.vision_backend == "yolo" and model_path and model_path.is_file():
        digest_ok = True
        if settings.yolo_expected_sha256:
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            digest_ok = digest == settings.yolo_expected_sha256
        checks.append(ReadinessCheck(
            key="professional_vision",
            label="专业视觉模型",
            status="ready" if digest_ok else "error",
            detail="YOLO权重存在且校验通过" if digest_ok else "YOLO权重摘要与固定版本不一致",
        ))
    else:
        checks.append(ReadinessCheck(
            key="professional_vision",
            label="专业视觉模型",
            status="degraded",
            detail="当前使用Demo视觉回退，专业YOLO未就绪",
        ))

    classifier_state = str(vision_runtime_status.get("state", "unknown"))
    checks.append(ReadinessCheck(
        key="fire_classifier",
        label="火情复核模型",
        status="ready" if classifier_state == "ready" else "degraded",
        detail=(
            "多模型火情复核进程已预热"
            if classifier_state == "ready"
            else f"复核分支状态：{classifier_state}"
        ),
    ))

    llm_ready = False
    llm_detail = "未配置本地大模型，使用确定性推理"
    if settings.reasoning_backend == "llm" and settings.llm_base_url and settings.llm_model:
        health_root = settings.llm_base_url[:-3] if settings.llm_base_url.endswith("/v1") else settings.llm_base_url
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                response = await client.get(f"{health_root}/health")
                response.raise_for_status()
            llm_ready = True
            llm_detail = f"本地大模型服务可用：{settings.llm_model}"
        except (httpx.HTTPError, ValueError) as exc:
            llm_detail = f"大模型健康检查未通过：{type(exc).__name__}"
    checks.append(ReadinessCheck(
        key="local_llm",
        label="本地大模型",
        status="ready" if llm_ready else "degraded",
        detail=llm_detail,
    ))

    categories = {rule.get("category") for rule in agent.knowledge.rules}
    knowledge_ready = {"fire", "smoke", "no_detection"}.issubset(categories)
    checks.append(ReadinessCheck(
        key="knowledge_base",
        label="安全知识库",
        status="ready" if knowledge_ready else "error",
        detail=f"已载入{len(agent.knowledge.rules)}条版本化知识，覆盖{len(categories)}类风险边界",
    ))

    registered_tools = [item["name"] for item in agent.tools.descriptions()]
    expected_tools = agent._plan_task("工业安全综合分析").tool_sequence
    tools_ready = registered_tools == expected_tools
    checks.append(ReadinessCheck(
        key="agent_tools",
        label="Agent专业工具",
        status="ready" if tools_ready else "error",
        detail=f"已注册{len(registered_tools)}个工具，规划与注册表{'一致' if tools_ready else '不一致'}",
    ))

    demo_ready = all(render_demo_case(case_id) for case_id in CASES)
    checks.append(ReadinessCheck(
        key="demo_cases",
        label="验证案例",
        status="ready" if demo_ready else "error",
        detail=f"{len(CASES)}个可复现案例已就绪，并公开来源或生成边界",
    ))
    checks.append(ReadinessCheck(
        key="report_exports",
        label="报告导出",
        status="ready",
        detail="可打印HTML报告与结构化JSON导出接口已注册",
    ))

    statuses = {item.status for item in checks}
    overall = "not_ready" if "error" in statuses else "degraded" if "degraded" in statuses else "ready"
    return ReadinessResponse(status=overall, checks=checks)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(...),
    task: str = Form("识别图片中的工业安全风险并给出核查建议"),
) -> AnalysisResponse:
    from app.main import agent, analysis_store

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(status_code=415, detail="仅支持 JPEG、PNG 或 WebP 图片")
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传图片为空")
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"图片不能超过 {settings.max_upload_mb} MB")
    try:
        started = perf_counter()
        result = await agent.analyze(payload, image.filename or "upload", task)
        analysis_store.record(result, (perf_counter() - started) * 1000)
        return result
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/demo-cases", response_model=list[DemoCaseInfo])
async def demo_cases() -> list[DemoCaseInfo]:
    return list(CASES.values())


@router.get("/demo-cases/{case_id}/image")
async def demo_case_image(case_id: str) -> Response:
    try:
        payload = render_demo_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="演示案例不存在") from exc
    media_type = "image/jpeg" if payload.startswith(b"\xff\xd8") else "image/png"
    return Response(
        payload,
        media_type=media_type,
        headers={
            "X-VisionGuard-Synthetic": str(CASES[case_id].synthetic).lower(),
            "X-VisionGuard-Case": case_id,
            "Cache-Control": "no-store",
        },
    )


@router.get("/analyses", response_model=list[AnalysisRecord])
async def analyses() -> list[AnalysisRecord]:
    from app.main import analysis_store

    return analysis_store.list_records()


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    from app.main import analysis_store

    return analysis_store.metrics()


@router.get("/analyses/{request_id}", response_model=AnalysisResponse)
async def analysis_detail(request_id: str) -> AnalysisResponse:
    from app.main import analysis_store

    result = analysis_store.get(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="分析记录不存在或已过期")
    return result


@router.post("/analyses/{request_id}/ask", response_model=FollowUpResponse)
async def ask_follow_up(request_id: str, request: FollowUpRequest) -> FollowUpResponse:
    from app.main import agent, analysis_store

    result = analysis_store.get(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="分析记录不存在或已过期")
    return await agent.answer_follow_up(result, request.question)


@router.get("/analyses/{request_id}/report.html")
async def analysis_report_html(request_id: str) -> Response:
    from app.main import analysis_store

    result = analysis_store.get(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="分析记录不存在或已过期")
    return Response(
        report_html(result),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="visionguard-{request_id}.html"'},
    )


@router.get("/analyses/{request_id}/report.json")
async def analysis_report_json(request_id: str) -> Response:
    from app.main import analysis_store

    result = analysis_store.get(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="分析记录不存在或已过期")
    return Response(
        report_json(result),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="visionguard-{request_id}.json"'},
    )
