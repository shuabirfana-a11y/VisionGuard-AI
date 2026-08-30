#!/usr/bin/env python3
"""Isolated JSON-lines worker that keeps FireBench backbones resident in memory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResidentFireBench:
    def __init__(self, runtime_root: Path, bundle: Path, device: str) -> None:
        source_root = runtime_root / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        import timm
        import torch
        from safetensors.torch import load_file as load_safetensors

        from firebench.bundle import load_bundle, resolve_bundle_member
        from firebench.multiview import resolve_device

        self.torch = torch
        self.bundle = bundle
        self.manifest, self.model_payload, self.digest = load_bundle(bundle)
        self.threshold = self._resolve_threshold(self.manifest["fusion"])
        if device == "dml":
            try:
                import torch_directml
            except ImportError as exc:
                raise RuntimeError("请求DirectML但未安装torch-directml") from exc
            torch_device = torch_directml.device()
            self.device = "dml"
            self.execution_device = torch_device
        else:
            self.device = resolve_device(device)
            torch_device = torch.device(self.device)
            self.execution_device = torch_device
        backbone_by_id = {
            item["id"]: item for item in self.manifest.get("backbones", [])
        }
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for branch in self.manifest["branches"]:
            grouped[(branch["model"], int(branch["input_size"]))].append(branch)

        self.groups: list[dict[str, Any]] = []
        for (model_name, input_size), branches in grouped.items():
            backbone_ids = {branch.get("backbone_id") for branch in branches}
            if len(backbone_ids) != 1:
                raise RuntimeError(f"骨干引用不一致: {model_name}@{input_size}")
            record = backbone_by_id.get(next(iter(backbone_ids)))
            if record is None:
                raise FileNotFoundError(f"未登记骨干模型: {model_name}@{input_size}")
            weights = resolve_bundle_member(bundle, record["relative_path"])
            if sha256_file(weights) != record["sha256"]:
                raise RuntimeError(f"骨干模型校验失败: {weights.name}")
            model = timm.create_model(
                model_name, pretrained=False, num_classes=0, img_size=input_size
            )
            state = load_safetensors(str(weights), device="cpu")
            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"骨干权重与结构不匹配: {weights.name}")
            model.eval().to(torch_device)
            config = model.pretrained_cfg
            mean = tuple(float(value) for value in config["mean"])
            std = tuple(float(value) for value in config["std"])
            reference = branches[0]
            if list(mean) != reference["mean"] or list(std) != reference["std"]:
                raise RuntimeError(f"预处理参数不匹配: {model_name}")
            self.groups.append(
                {
                    "model": model,
                    "input_size": input_size,
                    "mean": mean,
                    "std": std,
                    "fill": tuple(round(value * 255) for value in mean),
                    "branches": branches,
                    "need_multiview": any(
                        branch["mode"] in {"aggregate_logistic", "view_mean_logistic"}
                        for branch in branches
                    ),
                    "need_patch_tokens": any(
                        branch["mode"] == "patch_global_blend" for branch in branches
                    ),
                }
            )

    @staticmethod
    def _resolve_threshold(fusion: dict[str, Any]) -> float:
        name = fusion.get("active_threshold_profile")
        profiles = fusion.get("threshold_profiles", {})
        value = (
            profiles[name]["threshold"]
            if name and name in profiles
            else fusion.get("decision_threshold")
        )
        threshold = float(value)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise RuntimeError("冻结阈值非法")
        return threshold

    def _extract(self, image_path: Path, group: dict[str, Any]):
        from PIL import Image, ImageOps
        import torch.nn.functional as functional
        from torchvision.transforms import functional as vision_functional

        from firebench.multiview import ExtractedFeatures, five_views

        model = group["model"]
        input_size = group["input_size"]
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            views = five_views(image) if group["need_multiview"] else [image]
            tensors = []
            for view in views:
                padded = ImageOps.pad(
                    view,
                    (input_size, input_size),
                    method=Image.Resampling.BICUBIC,
                    color=group["fill"],
                    centering=(0.5, 0.5),
                )
                tensor = vision_functional.pil_to_tensor(padded).float().div_(255.0)
                tensors.append(
                    vision_functional.normalize(tensor, group["mean"], group["std"])
                )
        batch = self.torch.stack(tensors).unsqueeze(0).to(self.execution_device)
        multiview = patch_tokens = patch_global = None
        # DirectML's PrivateUse1 backend cannot update version counters on
        # inference tensors in several ViT operators. no_grad keeps inference
        # deterministic while preserving DirectML compatibility.
        grad_context = (
            self.torch.no_grad()
            if self.device == "dml"
            else self.torch.inference_mode()
        )
        with grad_context:
            if group["need_multiview"]:
                output = model(batch.flatten(0, 1))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                multiview = (
                    functional.normalize(output.float(), dim=1)
                    .reshape(1, len(views), -1)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            if group["need_patch_tokens"]:
                features = model.forward_features(batch[:, 0])
                prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
                global_features = model.forward_head(features, pre_logits=True)
                patch_tokens = (
                    functional.normalize(features[:, prefix_tokens:, :].float(), dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                patch_global = (
                    functional.normalize(global_features.float(), dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        return ExtractedFeatures(
            multiview=multiview,
            patch_tokens=patch_tokens,
            patch_global=patch_global,
            metadata={},
        )

    def predict(self, image_path: Path) -> dict[str, Any]:
        from firebench.bundle import fuse_scores, predict_level1_branch

        started = perf_counter()
        scores: dict[str, float] = {}
        for group in self.groups:
            extracted = self._extract(image_path, group)
            for branch in group["branches"]:
                result = predict_level1_branch(
                    branch,
                    self.model_payload["branches"][branch["name"]],
                    extracted,
                    patch_image_batch_size=1,
                )
                scores[branch["name"]] = float(result[0])
        ordered = np.asarray(
            [[scores[branch["name"]] for branch in self.manifest["branches"]]],
            dtype=np.float64,
        )
        fused = float(fuse_scores(ordered, self.manifest["fusion"])[0])
        return {
            "fused_score": fused,
            "prediction": int(fused >= self.threshold),
            "branch_scores": scores,
            "worker_inference_ms": round((perf_counter() - started) * 1000, 2),
            "device": self.device,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps", "dml"), default="auto"
    )
    args = parser.parse_args()
    try:
        engine = ResidentFireBench(
            args.runtime_root.expanduser().resolve(),
            args.bundle.expanduser().resolve(),
            args.device,
        )
    except Exception as exc:
        emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    warmup_ms = 0.0
    if args.device == "dml":
        # DirectML compiles several ViT operators on the first inference. Do that
        # during application startup so the first judge-facing request is warm.
        from PIL import Image

        with TemporaryDirectory(prefix="visionguard-dml-warmup-") as temp:
            warmup_image = Path(temp) / "warmup.png"
            Image.new("RGB", (512, 512), color=(24, 31, 40)).save(warmup_image)
            started = perf_counter()
            # The first pass compiles operators; the next two stabilize cached
            # execution before concurrent requests are accepted.
            for _ in range(3):
                engine.predict(warmup_image)
            warmup_ms = (perf_counter() - started) * 1000
    emit(
        {
            "status": "ready",
            "model_digest": engine.digest[:12],
            "device": engine.device,
            "threshold": engine.threshold,
            "warmup_ms": round(warmup_ms, 2),
        }
    )
    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                return 0
            request_id = request["id"]
            result = engine.predict(Path(request["image_path"]))
            emit({"id": request_id, "status": "ok", **result})
        except Exception as exc:
            emit(
                {
                    "id": request.get("id"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
