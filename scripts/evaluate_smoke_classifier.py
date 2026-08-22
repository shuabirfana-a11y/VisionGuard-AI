from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from train_smoke_classifier import DinoFeatureExtractor, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--firebench-bundle", type=Path, required=True)
    parser.add_argument("--source-contains", default="pyro-sdis")
    parser.add_argument("--device", choices=("cpu", "cuda", "dml"), default="dml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if args.source_contains in row["source"]:
                rows.append(row)
    if not rows:
        raise RuntimeError("评测清单中没有匹配的数据源")

    model = joblib.load(args.classifier.resolve())
    extractor = DinoFeatureExtractor(args.firebench_bundle.resolve(), args.device)
    tensors = [
        extractor.tensor((manifest.parent / row["image"]).read_bytes())
        for row in rows
    ]
    import torch.nn.functional as functional

    batch = extractor.torch.stack(tensors).to(extractor.device)
    context = (
        extractor.torch.no_grad()
        if extractor.device_name == "dml"
        else extractor.torch.inference_mode()
    )
    with context:
        features = extractor.model(batch)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = functional.normalize(features.float(), dim=1)
    array = features.cpu().numpy().astype(np.float32)
    probabilities = model["classifier"].predict_proba(
        model["scaler"].transform(array)
    )[:, 1]
    threshold = float(model["threshold"])
    labels = np.asarray([
        int(any(item.get("category") == "smoke" for item in json.loads(row["ground_truth_json"])))
        for row in rows
    ], dtype=np.int8)
    predictions = probabilities >= threshold
    report = {
        "schema_version": 1,
        "source_filter": args.source_contains,
        "threshold": threshold,
        "metrics": metrics(labels, predictions),
        "predictions": [
            {
                "image_path": row["image"],
                "ground_truth": bool(label),
                "prediction": bool(prediction),
                "probability": round(float(probability), 6),
            }
            for row, label, prediction, probability in zip(
                rows, labels, predictions, probabilities, strict=True
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
