from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Awaitable, Callable

from app.schemas import FireClassificationEvidence


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], float], Awaitable[CommandResult]]


async def _run_command(command: list[str], timeout_seconds: float) -> CommandResult:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise RuntimeError(f"火情分类推理超过 {timeout_seconds:g} 秒")
    return CommandResult(
        returncode=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def _bundle_digest(bundle: Path) -> str | None:
    digest_path = bundle / "bundle.sha256"
    if not digest_path.is_file():
        return None
    value = digest_path.read_text(encoding="utf-8").split()[0]
    return value[:12] if len(value) == 64 else None


class FireBenchWorkerClient:
    """JSON-lines client for the isolated, model-resident inference worker."""

    def __init__(
        self,
        *,
        python_executable: str,
        runtime_root: Path,
        bundle: Path,
        device: str,
        timeout_seconds: float,
    ) -> None:
        self.command = [
            python_executable,
            str(Path(__file__).resolve().parents[3] / "scripts" / "firebench_worker.py"),
            "--runtime-root",
            str(runtime_root),
            "--bundle",
            str(bundle),
            "--device",
            device,
        ]
        self.timeout_seconds = timeout_seconds
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.request_id = 0

    async def _read_json(self) -> dict:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("火情分类常驻进程尚未启动")
        raw = await asyncio.wait_for(
            self.process.stdout.readline(), timeout=self.timeout_seconds
        )
        if not raw:
            detail = ""
            if self.process.stderr is not None:
                stderr = await self.process.stderr.read()
                detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"火情分类常驻进程意外退出: {detail or '无错误信息'}")
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("火情分类常驻进程返回了非法响应") from exc

    async def _start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        ready = await self._read_json()
        if ready.get("status") != "ready":
            detail = ready.get("error", ready)
            await self.close()
            raise RuntimeError(f"火情分类常驻进程启动失败: {detail}")

    async def start(self) -> None:
        async with self.lock:
            await self._start()

    async def infer(self, image_path: Path) -> dict:
        async with self.lock:
            await self._start()
            if self.process is None or self.process.stdin is None:
                raise RuntimeError("火情分类常驻进程不可写")
            self.request_id += 1
            request = {"id": self.request_id, "image_path": str(image_path)}
            self.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            await self.process.stdin.drain()
            response = await self._read_json()
            if response.get("id") != self.request_id:
                raise RuntimeError("火情分类常驻进程响应编号不匹配")
            if response.get("status") != "ok":
                raise RuntimeError(str(response.get("error", "常驻推理失败")))
            return response

    async def close(self) -> None:
        process, self.process = self.process, None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.write(b'{"command":"shutdown"}\n')
            try:
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


class FireBenchWorkerPool:
    """Bounded pool of isolated resident workers for concurrent requests."""

    def __init__(self, worker_count: int, **client_kwargs) -> None:
        self.clients = [FireBenchWorkerClient(**client_kwargs) for _ in range(worker_count)]
        self.available: asyncio.Queue[FireBenchWorkerClient] = asyncio.Queue()
        self.start_lock = asyncio.Lock()
        self.started = False

    async def start(self) -> None:
        async with self.start_lock:
            if self.started:
                return
            await asyncio.gather(*(client.start() for client in self.clients))
            for client in self.clients:
                self.available.put_nowait(client)
            self.started = True

    async def infer(self, image_path: Path) -> dict:
        await self.start()
        client = await self.available.get()
        try:
            return await client.infer(image_path)
        finally:
            self.available.put_nowait(client)

    async def close(self) -> None:
        async with self.start_lock:
            await asyncio.gather(*(client.close() for client in self.clients))
            while not self.available.empty():
                self.available.get_nowait()
            self.started = False


class FireBenchClassifier:
    """Adapter for the teammate's frozen three-branch image classifier.

    The external runtime remains independently deployable. VisionGuard invokes its
    deterministic CLI without importing or copying the 460 MB model bundle into Git.
    """

    name = "firebench-three-branch"

    def __init__(
        self,
        runtime_root: str | Path,
        *,
        bundle: str | Path | None = None,
        python_executable: str | Path | None = None,
        device: str = "auto",
        mode: str = "cli",
        worker_count: int = 1,
        timeout_seconds: float = 180,
        runner: CommandRunner | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.script = self.runtime_root / "scripts" / "predict.py"
        self.bundle = (
            Path(bundle).expanduser().resolve()
            if bundle
            else self.runtime_root / "artifacts" / "final_ensemble"
        )
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.runner = runner or _run_command
        if device not in {"auto", "cpu", "cuda", "mps", "dml"}:
            raise ValueError("FIRE_CLASSIFIER_DEVICE 必须为 auto、cpu、cuda、mps 或 dml")
        if timeout_seconds <= 0:
            raise ValueError("火情分类超时时间必须大于 0")
        if mode not in {"cli", "persistent"}:
            raise ValueError("FIRE_CLASSIFIER_MODE 必须为 cli 或 persistent")
        if not 1 <= worker_count <= 8:
            raise ValueError("FIRE_CLASSIFIER_WORKER_COUNT 必须位于1到8之间")
        if mode != "persistent" and worker_count != 1:
            raise ValueError("多工作进程仅适用于persistent模式")
        if not self.script.is_file():
            raise RuntimeError(f"队友分类器入口不存在: {self.script}")
        manifest_path = self.bundle / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"队友分类模型清单不存在: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fusion = manifest.get("fusion", {})
        profiles = fusion.get("threshold_profiles", {})
        profile_name = fusion.get("active_threshold_profile", "decision_threshold")
        profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
        self.threshold = float(profile.get("threshold", fusion.get("decision_threshold", 0.5)))
        self.model_digest = _bundle_digest(self.bundle)
        self.model_version = f"three-branch-{self.model_digest or 'configured'}"
        self.worker = (
            FireBenchWorkerPool(
                worker_count,
                python_executable=self.python_executable,
                runtime_root=self.runtime_root,
                bundle=self.bundle,
                device=self.device,
                timeout_seconds=self.timeout_seconds,
            )
            if mode == "persistent"
            else None
        )
        self.worker_count = worker_count if mode == "persistent" else 0

    async def classify(
        self, image_bytes: bytes, file_name: str
    ) -> FireClassificationEvidence:
        suffix = Path(file_name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            suffix = ".png"
        with TemporaryDirectory(prefix="visionguard-firebench-") as temp:
            root = Path(temp)
            image_dir = root / "images"
            image_dir.mkdir()
            image_path = image_dir / f"input{suffix}"
            image_path.write_bytes(image_bytes)
            if self.worker is not None:
                started = perf_counter()
                row = await self.worker.infer(image_path)
                inference_ms = (perf_counter() - started) * 1000
                probability = float(row["fused_score"])
                prediction = int(row["prediction"])
                return self._evidence(probability, prediction, inference_ms)
            output = root / "result.json"
            scores = root / "scores.csv"
            command = [
                self.python_executable,
                str(self.script),
                "--images",
                str(image_dir),
                "--bundle",
                str(self.bundle),
                "--output",
                str(output),
                "--scores-csv",
                str(scores),
                "--expected-count",
                "0",
                "--device",
                self.device,
            ]
            started = perf_counter()
            result = await self.runner(command, self.timeout_seconds)
            inference_ms = (perf_counter() - started) * 1000
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip().splitlines()
                message = detail[-1] if detail else "未知错误"
                raise RuntimeError(f"队友火情分类器执行失败: {message}")
            if not scores.is_file():
                raise RuntimeError("队友火情分类器未生成分数文件")
            with scores.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1:
                raise RuntimeError(f"队友火情分类器返回 {len(rows)} 行，预期为 1 行")
            row = rows[0]
            probability = float(row["fused_score"])
            prediction = int(row["prediction"])
            return self._evidence(probability, prediction, inference_ms)

    def _evidence(
        self, probability: float, prediction: int, inference_ms: float
    ) -> FireClassificationEvidence:
        if not 0 <= probability <= 1 or prediction not in {0, 1}:
            raise RuntimeError("队友火情分类器返回非法分数或标签")
        return FireClassificationEvidence(
            available=True,
            prediction=bool(prediction),
            probability=round(probability, 6),
            threshold=self.threshold,
            source=f"{self.name}:{self.model_version}",
            model_name=self.name,
            model_version=self.model_version,
            model_digest=self.model_digest,
            device=self.device,
            inference_ms=round(inference_ms, 2),
            limitations=["该分支提供整图可见火焰判断，不提供目标检测框，也不将烟雾单独判为火焰。"],
        )

    async def close(self) -> None:
        if self.worker is not None:
            await self.worker.close()

    async def start(self) -> None:
        if self.worker is not None:
            await self.worker.start()


class UnavailableFireClassifier:
    name = "fire-classifier-unavailable"
    model_version = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def start(self) -> None:
        raise RuntimeError(self.reason)

    async def classify(
        self, image_bytes: bytes, file_name: str
    ) -> FireClassificationEvidence:
        return FireClassificationEvidence(
            available=False,
            source=self.name,
            model_name=self.name,
            model_version=self.model_version,
            device="unavailable",
            inference_ms=0,
            limitations=["高精度火情分类分支不可用，结果仅依据目标检测证据。"],
            error=self.reason,
        )
