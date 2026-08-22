import asyncio
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

from PIL import Image
import pytest

from app.schemas import FireClassificationEvidence
from app.services.knowledge import SafetyKnowledgeService
from app.services.risk import RiskAnalysisService
from app.services.vision.demo import DemoColorDetector
from app.services.vision.firebench import CommandResult, FireBenchClassifier
from app.services.vision.fusion import MultimodelVisionDetector, fuse_fire_evidence


def image_bytes(color=(25, 30, 35)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (160, 100), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def fake_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "fire_highscore_submission"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "predict.py").write_text("# fake", encoding="utf-8")
    bundle = root / "artifacts" / "final_ensemble"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({
            "fusion": {
                "active_threshold_profile": "main_lofo_median",
                "decision_threshold": 0.5,
                "threshold_profiles": {
                    "main_lofo_median": {"threshold": 0.5069862008853248}
                },
            }
        }),
        encoding="utf-8",
    )
    (bundle / "bundle.sha256").write_text("b" * 64 + "  files\n", encoding="utf-8")
    return root


def test_firebench_adapter_reads_frozen_fused_score(tmp_path):
    runtime = fake_runtime(tmp_path)

    async def runner(command: list[str], timeout: float) -> CommandResult:
        scores = Path(command[command.index("--scores-csv") + 1])
        output = Path(command[command.index("--output") + 1])
        scores.write_text(
            "filename,branch_a,branch_b,branch_c,fused_score,prediction\n"
            "input.png,0.8,0.9,0.7,0.9321,1\n",
            encoding="utf-8",
        )
        output.write_text('{"input.png": 1}', encoding="utf-8")
        return CommandResult(0, "ok", "")

    classifier = FireBenchClassifier(runtime, runner=runner, device="cpu")
    result = asyncio.run(classifier.classify(image_bytes(), "scene.png"))
    assert result.available is True
    assert result.prediction is True
    assert result.probability == 0.9321
    assert result.threshold == 0.5069862008853248
    assert result.model_digest == "bbbbbbbbbbbb"


def test_firebench_adapter_rejects_unknown_execution_mode(tmp_path):
    runtime = fake_runtime(tmp_path)
    with pytest.raises(ValueError, match="FIRE_CLASSIFIER_MODE"):
        FireBenchClassifier(runtime, mode="unknown")


def test_firebench_adapter_accepts_directml_device(tmp_path):
    runtime = fake_runtime(tmp_path)
    classifier = FireBenchClassifier(runtime, device="dml")
    assert classifier.device == "dml"


def test_firebench_adapter_validates_worker_pool_configuration(tmp_path):
    runtime = fake_runtime(tmp_path)
    with pytest.raises(ValueError, match="WORKER_COUNT"):
        FireBenchClassifier(runtime, mode="persistent", worker_count=0)
    with pytest.raises(ValueError, match="persistent"):
        FireBenchClassifier(runtime, mode="cli", worker_count=2)


class FixedClassifier:
    name = "fixed-classifier"
    model_version = "test"

    def __init__(self, prediction: bool, probability: float) -> None:
        self.prediction = prediction
        self.probability = probability

    async def classify(self, image_bytes: bytes, file_name: str):
        return FireClassificationEvidence(
            available=True,
            prediction=self.prediction,
            probability=self.probability,
            threshold=0.5,
            source="test",
            model_name=self.name,
            model_version=self.model_version,
            device="cpu",
            inference_ms=1,
        )


def test_classifier_only_fire_becomes_unlocalized_risk_evidence():
    detector = MultimodelVisionDetector(
        DemoColorDetector(), FixedClassifier(True, 0.93)
    )
    vision = asyncio.run(detector.detect(image_bytes(), "scene.png"))
    assert vision.detections == []
    assert vision.fusion.status == "classifier_only"

    knowledge_service = SafetyKnowledgeService()
    knowledge = asyncio.run(knowledge_service.retrieve(["fire"], "检查火灾风险"))
    risk = asyncio.run(RiskAnalysisService().analyze(vision, knowledge))
    assert risk.overall_level == "high"
    assert risk.items[0].evidence_ids == ["cls-fire-001"]
    assert "未提供定位框" in risk.items[0].reason


def test_detector_and_classifier_agreement_and_conflict_are_explicit():
    confirmed = asyncio.run(MultimodelVisionDetector(
        DemoColorDetector(), FixedClassifier(True, 0.96)
    ).detect(image_bytes((230, 90, 25)), "fire.png"))
    assert confirmed.fusion.status == "confirmed"

    conflict = asyncio.run(MultimodelVisionDetector(
        DemoColorDetector(), FixedClassifier(False, 0.12)
    ).detect(image_bytes((230, 90, 25)), "fire.png"))
    assert conflict.fusion.status == "detector_only"
    assert any("证据不一致" in item for item in conflict.limitations)


def test_fusion_never_treats_dual_negative_as_proof_of_safety():
    result = asyncio.run(MultimodelVisionDetector(
        DemoColorDetector(), FixedClassifier(False, 0.08)
    ).detect(image_bytes(), "clear.png"))
    fusion = fuse_fire_evidence(result)
    assert fusion.status == "no_fire_evidence"
    assert "不等同于现场绝对安全" in fusion.summary


def test_detector_and_classifier_run_in_parallel():
    class SlowDetector:
        name = "slow-detector"
        model_version = "test"

        async def detect(self, payload: bytes, file_name: str):
            await asyncio.sleep(0.08)
            return await DemoColorDetector().detect(payload, file_name)

    class SlowClassifier(FixedClassifier):
        async def classify(self, payload: bytes, file_name: str):
            await asyncio.sleep(0.08)
            return await super().classify(payload, file_name)

    detector = MultimodelVisionDetector(
        SlowDetector(), SlowClassifier(False, 0.08)
    )
    started = perf_counter()
    asyncio.run(detector.detect(image_bytes(), "scene.png"))
    elapsed = perf_counter() - started

    assert elapsed < 0.14
