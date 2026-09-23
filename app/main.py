from contextlib import asynccontextmanager
from pathlib import Path
from hashlib import sha256
import re
import secrets

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app import __version__
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
vision_runtime_status = {
    "state": (
        "disabled"
        if not settings.fire_classifier_enabled
        else "lazy" if settings.fire_classifier_mode == "persistent" else "cli"
    ),
    "error": None,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    start = getattr(agent.detector, "start", None)
    if (
        settings.fire_classifier_eager_start
        and settings.fire_classifier_mode == "persistent"
        and start is not None
    ):
        vision_runtime_status["state"] = "warming"
        try:
            await start()
        except Exception as exc:
            vision_runtime_status["state"] = "error"
            vision_runtime_status["error"] = f"{type(exc).__name__}: {exc}"
        else:
            vision_runtime_status["state"] = "ready"
    yield
    close = getattr(agent.detector, "close", None)
    if close is not None:
        await close()


app = FastAPI(
    title="VisionGuard AI",
    description="面向工业安全场景的多模态视觉风险智能体 MVP",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    session = request.cookies.get("vg_session", "")
    new_session = re.fullmatch(r"[A-Za-z0-9_-]{43}", session) is None
    if new_session:
        session = secrets.token_urlsafe(32)
    # Store only a digest, never the browser's bearer cookie or uploaded image.
    request.state.analysis_owner = sha256(session.encode("ascii")).hexdigest()
    response = await call_next(request)
    if new_session:
        response.set_cookie(
            "vg_session", session, httponly=True, samesite="strict",
            secure=request.url.scheme == "https", max_age=8 * 60 * 60,
        )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'"
    )
    if request.url.path.startswith("/api/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")
