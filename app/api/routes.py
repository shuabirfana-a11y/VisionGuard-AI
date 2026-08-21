from time import perf_counter

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.config import settings
from app.schemas import AnalysisRecord, AnalysisResponse, DemoCaseInfo, HealthResponse, MetricsResponse
from app.services.demo_cases import CASES, render_demo_case
from app.services.exports import report_html, report_json
from app.services.vision.demo import InvalidImageError


router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from app.main import agent

    return HealthResponse(
        status="ok",
        project="VisionGuard AI",
        version="0.4.0",
        vision_backend=settings.vision_backend,
        capabilities={
            "image_upload": "available",
            "agent_tool_calling": "available",
            "structured_risk_report": "available",
            "reasoning_backend": settings.reasoning_backend,
            "llm_reasoning": (
                "configured"
                if settings.reasoning_backend == "llm" and settings.llm_base_url and settings.llm_api_key and settings.llm_model
                else "deterministic_or_fallback"
            ),
            "vector_rag": "planned",
            "yolo_model_configured": bool(settings.yolo_model_path),
            "vision_fallback_enabled": settings.vision_fallback_enabled,
            "registered_tools": agent.tools.descriptions(),
        },
    )


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
    return Response(
        payload,
        media_type="image/png",
        headers={"X-VisionGuard-Synthetic": "true", "Cache-Control": "no-store"},
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
