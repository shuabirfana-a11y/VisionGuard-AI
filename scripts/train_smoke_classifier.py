from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Sample:
    image_name: str
    label: int
    group: str


@dataclass(frozen=True)
class FileSample:
    image_path: Path
    label: int
    group: str


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def group_name(path: str) -> str:
    return Path(path).stem.split(".rf.")[0]


def load_inventory(archive: Path) -> dict[str, list[Sample]]:
    inventory: dict[str, list[Sample]] = {"train": [], "valid": [], "test": []}
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        for label_name in sorted(names):
            parts = label_name.split("/")
            if (
                len(parts) < 4
                or parts[1] not in inventory
                or parts[2] != "labels"
                or not label_name.endswith(".txt")
            ):
                continue
            text = bundle.read(label_name).decode("utf-8").strip()
            classes = {
                int(line.split()[0]) for line in text.splitlines() if line.strip()
            }
            if not classes or not classes.issubset({0, 1}):
                continue
            image_name = label_name.replace("/labels/", "/images/").removesuffix(".txt") + ".jpg"
            if image_name not in names:
                continue
            inventory[parts[1]].append(
                Sample(
                    image_name=image_name,
                    label=int(1 in classes),
                    group=group_name(image_name),
                )
            )
    held_out_groups = {
        sample.group for split in ("valid", "test") for sample in inventory[split]
    }
    inventory["train"] = [
        sample for sample in inventory["train"] if sample.group not in held_out_groups
    ]
    return inventory


class DinoFeatureExtractor:
    def __init__(self, bundle: Path, device: str) -> None:
        import timm
        import torch
        from safetensors.torch import load_file

        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        record = next(
            item for item in manifest["backbones"]
            if item["architecture"] == "vit_small_patch14_dinov2"
        )
        weights = bundle / record["relative_path"]
        if digest(weights) != record["sha256"]:
            raise RuntimeError("DINOv2 backbone hash mismatch")
        self.input_size = int(record["input_size"])
        self.mean = tuple(float(value) for value in record["mean"])
        self.std = tuple(float(value) for value in record["std"])
        self.fill = tuple(round(value * 255) for value in self.mean)
        self.torch = torch
        if device == "dml":
            import torch_directml

            self.device = torch_directml.device()
            self.device_name = "dml"
        else:
            self.device = torch.device(device)
            self.device_name = device
        self.model = timm.create_model(
            record["model"], pretrained=False, num_classes=0, img_size=self.input_size
        )
        state = load_file(str(weights), device="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.backbone = {
            "id": record["id"],
            "model": record["model"],
            "input_size": self.input_size,
            "weights_sha256": record["sha256"],
        }

    def tensor(self, payload: bytes):
        from torchvision.transforms import functional as vision_functional

        with Image.open(BytesIO(payload)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = ImageOps.pad(
                image,
                (self.input_size, self.input_size),
                method=Image.Resampling.BICUBIC,
                color=self.fill,
                centering=(0.5, 0.5),
            )
            value = vision_functional.pil_to_tensor(image).float().div_(255.0)
        return vision_functional.normalize(value, self.mean, self.std)

    def extract(self, archive: Path, samples: list[Sample], batch_size: int) -> np.ndarray:
        import torch.nn.functional as functional

        outputs: list[np.ndarray] = []
        with ZipFile(archive) as bundle:
            for start in range(0, len(samples), batch_size):
                current = samples[start:start + batch_size]
                batch = self.torch.stack(
                    [self.tensor(bundle.read(sample.image_name)) for sample in current]
                ).to(self.device)
                context = (
                    self.torch.no_grad()
                    if self.device_name == "dml"
                    else self.torch.inference_mode()
                )
                with context:
                    features = self.model(batch)
                    if isinstance(features, (tuple, list)):
                        features = features[0]
                    features = functional.normalize(features.float(), dim=1)
                outputs.append(features.cpu().numpy().astype(np.float32))
                done = min(start + batch_size, len(samples))
                if done == len(samples) or done % 256 == 0:
                    print(f"features {done}/{len(samples)}", flush=True)
        return np.concatenate(outputs, axis=0)

    def extract_files(self, samples: list[FileSample], batch_size: int) -> np.ndarray:
        import torch.nn.functional as functional

        outputs: list[np.ndarray] = []
        for start in range(0, len(samples), batch_size):
            current = samples[start:start + batch_size]
            batch = self.torch.stack(
                [self.tensor(sample.image_path.read_bytes()) for sample in current]
            ).to(self.device)
            context = (
                self.torch.no_grad()
                if self.device_name == "dml"
                else self.torch.inference_mode()
            )
            with context:
                features = self.model(batch)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = functional.normalize(features.float(), dim=1)
            outputs.append(features.cpu().numpy().astype(np.float32))
            done = min(start + batch_size, len(samples))
            if done == len(samples) or done % 256 == 0:
                print(f"file features {done}/{len(samples)}", flush=True)
        return np.concatenate(outputs, axis=0)


def load_file_inventory(manifest: Path) -> dict[str, list[FileSample]]:
    inventory: dict[str, list[FileSample]] = {"train": [], "valid": [], "test": []}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in inventory:
                raise RuntimeError(f"未知Pyro数据分区: {split}")
            image_path = (manifest.parent / row["image"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            inventory[split].append(
                FileSample(
                    image_path=image_path,
                    label=int(row["label"]),
                    group=row["group"],
                )
            )
    return inventory


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
    }


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    candidates = np.linspace(0.05, 0.95, 181)
    records = [
        (float(threshold), metrics(labels, probabilities >= threshold))
        for threshold in candidates
    ]
    return max(
        records,
        key=lambda item: (
            item[1]["f1"], item[1]["accuracy"], item[1]["recall"], -abs(item[0] - 0.5)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--firebench-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pyro-manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "dml"), default="dml")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    archive = args.archive.resolve()
    bundle = args.firebench_bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory(archive)
    pyro_inventory = (
        load_file_inventory(args.pyro_manifest.resolve())
        if args.pyro_manifest
        else None
    )
    extractor = DinoFeatureExtractor(bundle, args.device)
    arrays: dict[str, np.ndarray] = {}
    started = perf_counter()
    for split in ("train", "valid", "test"):
        arrays[split] = extractor.extract(archive, inventory[split], args.batch_size)
    labels = {
        split: np.asarray([sample.label for sample in inventory[split]], dtype=np.int8)
        for split in inventory
    }
    pyro_arrays: dict[str, np.ndarray] = {}
    pyro_labels: dict[str, np.ndarray] = {}
    if pyro_inventory is not None:
        for split in ("train", "valid", "test"):
            pyro_arrays[split] = extractor.extract_files(
                pyro_inventory[split], args.batch_size
            )
            pyro_labels[split] = np.asarray(
                [sample.label for sample in pyro_inventory[split]], dtype=np.int8
            )
    elapsed_ms = (perf_counter() - started) * 1000

    fit_array = arrays["train"]
    fit_labels = labels["train"]
    valid_array = arrays["valid"]
    valid_labels = labels["valid"]
    if pyro_inventory is not None:
        fit_array = np.concatenate((fit_array, pyro_arrays["train"]), axis=0)
        fit_labels = np.concatenate((fit_labels, pyro_labels["train"]), axis=0)
        valid_array = np.concatenate((valid_array, pyro_arrays["valid"]), axis=0)
        valid_labels = np.concatenate((valid_labels, pyro_labels["valid"]), axis=0)

    candidates: list[dict] = []
    for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        scaler = StandardScaler().fit(fit_array)
        classifier = LogisticRegression(
            C=c_value, max_iter=2000, class_weight="balanced", random_state=20260822
        ).fit(scaler.transform(fit_array), fit_labels)
        valid_probability = classifier.predict_proba(
            scaler.transform(valid_array)
        )[:, 1]
        threshold, valid_metrics = choose_threshold(valid_labels, valid_probability)
        candidates.append({
            "c": c_value,
            "scaler": scaler,
            "classifier": classifier,
            "threshold": threshold,
            "valid_metrics": valid_metrics,
        })
    selected = max(
        candidates,
        key=lambda item: (
            item["valid_metrics"]["f1"],
            item["valid_metrics"]["accuracy"],
            item["valid_metrics"]["recall"],
            -item["c"],
        ),
    )
    test_probability = selected["classifier"].predict_proba(
        selected["scaler"].transform(arrays["test"])
    )[:, 1]
    test_metrics = metrics(labels["test"], test_probability >= selected["threshold"])
    source_test_metrics: dict[str, dict] = {"ifiresmoke": test_metrics}
    if pyro_inventory is not None:
        pyro_test_probability = selected["classifier"].predict_proba(
            selected["scaler"].transform(pyro_arrays["test"])
        )[:, 1]
        source_test_metrics["pyro_sdis"] = metrics(
            pyro_labels["test"], pyro_test_probability >= selected["threshold"]
        )
        combined_labels = np.concatenate((labels["test"], pyro_labels["test"]))
        combined_probability = np.concatenate((test_probability, pyro_test_probability))
        test_metrics = metrics(
            combined_labels, combined_probability >= selected["threshold"]
        )
    model_path = output / "smoke_classifier.joblib"
    joblib.dump({
        "schema_version": 1,
        "scaler": selected["scaler"],
        "classifier": selected["classifier"],
        "threshold": selected["threshold"],
        "backbone": extractor.backbone,
        "labels": {"negative": "no_smoke", "positive": "smoke"},
    }, model_path)
    report = {
        "schema_version": 1,
        "created_at": "2026-08-22",
        "archive_sha256": digest(archive),
        "source": "shahriar-5/IFireSmoke Indoor FS",
        "split_policy": "official train/valid/test; train groups overlapping valid or test removed",
        "device": args.device,
        "feature_extraction_ms": round(elapsed_ms, 2),
        "backbone": extractor.backbone,
        "inventory": {
            split: {
                "samples": len(inventory[split]),
                "groups": len({sample.group for sample in inventory[split]}),
                "smoke": int(labels[split].sum()),
                "no_smoke": int((labels[split] == 0).sum()),
            }
            for split in inventory
        },
        "pyro_inventory": (
            {
                split: {
                    "samples": len(pyro_inventory[split]),
                    "groups": len({sample.group for sample in pyro_inventory[split]}),
                    "smoke": int(pyro_labels[split].sum()),
                    "no_smoke": int((pyro_labels[split] == 0).sum()),
                }
                for split in pyro_inventory
            }
            if pyro_inventory is not None
            else None
        ),
        "selection": {
            "regularization_c": selected["c"],
            "threshold": selected["threshold"],
            "validation_metrics": selected["valid_metrics"],
        },
        "test_metrics": test_metrics,
        "source_test_metrics": source_test_metrics,
        "model_sha256": digest(model_path),
        "claim_boundary": (
            "Metrics apply to source-group-disjoint IFireSmoke and optional Pyro-SDIS splits. "
            "The fixed 20-image Pyro-SDIS evaluation remains excluded from training, validation, and threshold selection."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "bundle.sha256").write_text(
        f"{digest(model_path)}  smoke_classifier.joblib\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
