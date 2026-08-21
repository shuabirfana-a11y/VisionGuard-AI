from app.config import Settings
from app.services.reasoning.base import Reasoner
from app.services.reasoning.deterministic import DeterministicReasoner
from app.services.reasoning.fallback import FallbackReasoner
from app.services.reasoning.llm import CompatibleLLMReasoner


def build_reasoner(settings: Settings) -> Reasoner:
    deterministic = DeterministicReasoner()
    if settings.reasoning_backend == "deterministic":
        return deterministic
    if settings.reasoning_backend != "llm":
        raise RuntimeError(f"不支持的 REASONING_BACKEND: {settings.reasoning_backend}")
    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        if settings.reasoning_fallback_enabled:
            return FallbackReasoner(
                None,
                deterministic,
                "LLM_BASE_URL、LLM_API_KEY或LLM_MODEL未完整配置",
            )
        raise RuntimeError("LLM推理配置不完整")
    primary = CompatibleLLMReasoner(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    if settings.reasoning_fallback_enabled:
        return FallbackReasoner(primary, deterministic)
    return primary


__all__ = ["Reasoner", "build_reasoner"]

