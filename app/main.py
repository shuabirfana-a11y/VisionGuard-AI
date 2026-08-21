from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.core.agent import VisionGuardAgent
from app.services.knowledge import SafetyKnowledgeService
from app.services.report import SafetyReportService
from app.services.reasoning import build_reasoner
from app.services.risk import RiskAnalysisService
from app.services.store import AnalysisStore
from app.services.vision import build_detector


BASE_DIR = Path(__file__).resolve().parent

agent = VisionGuardAgent(
    detector=build_detector(settings),
    knowledge=SafetyKnowledgeService(),
    risk=RiskAnalysisService(),
    report=SafetyReportService(),
    reasoner=build_reasoner(settings),
)
analysis_store = AnalysisStore(max_records=100)

app = FastAPI(
    title="VisionGuard AI",
    description="面向工业安全场景的多模态视觉风险智能体 MVP",
    version="0.4.0",
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")
