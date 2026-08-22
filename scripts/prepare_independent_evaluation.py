from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from zipfile import ZipFile

import httpx
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
DATA_ROOT = EVALUATION_ROOT / "data"
CACHE_ROOT = EVALUATION_ROOT / "cache"
MANIFEST = EVALUATION_ROOT / "independent_manifest.csv"
FULL_CHAIN_MANIFEST = EVALUATION_ROOT / "full_chain_manifest.csv"
IFIRE_REVISION = "22b3c06db783d3e75c74234faa511020912eac64"
IFIRE_ZIP_SHA256 = "7b138ad1f61883feba0de40624bdc0fa52bcf58238bde2d2627d3b6fc9160b5b"
FIREVIEWER_REVISION = "85ad763e6275537386f7eefdae5e3a18a55f1c71"
SELECTION_SALT = "visionguard-independent-evaluation-v1"


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(name: str) -> str:
    return sha256(f"{SELECTION_SALT}|{name}".encode()).hexdigest()


def download(client: httpx.Client, url: str, destination: Path, expected_sha256: str = "") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (not expected_sha256 or file_digest(destination) == expected_sha256):
        return
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.write_bytes(response.content)
    actual = file_digest(temporary)
    if expected_sha256 and actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download hash mismatch for {destination.name}: {actual}")
    temporary.replace(destination)


def yolo_ground_truth(label_text: str, width: int, height: int) -> list[dict[str, object]]:
    names = {0: "fire", 1: "smoke"}
    result: list[dict[str, object]] = []
    for raw in label_text.splitlines():
        if not raw.strip():
            continue
        class_id, cx, cy, box_width, box_height = map(float, raw.split()[:5])
        category = names[int(class_id)]
        result.append({
            "category": category,
            "bbox": {
                "x1": round((cx - box_width / 2) * width, 3),
                "y1": round((cy - box_height / 2) * height, 3),
                "x2": round((cx + box_width / 2) * width, 3),
                "y2": round((cy + box_height / 2) * height, 3),
            },
        })
    return result


def prepare_ifiresmoke(client: httpx.Client) -> list[dict[str, object]]:
    archive = CACHE_ROOT / "ifiresmoke" / "Indoor_FS.zip"
    url = f"https://huggingface.co/datasets/shahriar-5/IFireSmoke/resolve/{IFIRE_REVISION}/{quote('Indoor FS.zip')}?download=true"
    download(client, url, archive, IFIRE_ZIP_SHA256)
    rows: list[dict[str, object]] = []
    with ZipFile(archive) as bundle:
        labels = [name for name in bundle.namelist() if name.startswith("Indoor FS/test/labels/") and name.endswith(".txt")]
        candidates: dict[str, list[tuple[str, str, str]]] = {"fire": [], "smoke": []}
        for label_name in labels:
            text = bundle.read(label_name).decode("utf-8").strip()
            classes = {int(line.split()[0]) for line in text.splitlines() if line.strip()}
            if classes not in ({0}, {1}):
                continue
            category = "fire" if classes == {0} else "smoke"
            stem = Path(label_name).stem
            group = stem.split(".rf.")[0]
            image_name = label_name.replace("/labels/", "/images/").removesuffix(".txt") + ".jpg"
            if image_name in bundle.namelist():
                candidates[category].append((image_name, label_name, group))
        for category in ("fire", "smoke"):
            selected: list[tuple[str, str, str]] = []
            seen_groups: set[str] = set()
            for item in sorted(candidates[category], key=lambda value: selection_key(value[0])):
                if item[2] in seen_groups:
                    continue
                selected.append(item)
                seen_groups.add(item[2])
                if len(selected) == 30:
                    break
            if len(selected) < 30:
                raise RuntimeError(f"not enough unique IFireSmoke {category} groups")
            for index, (image_name, label_name, group) in enumerate(selected, start=1):
                image_bytes = bundle.read(image_name)
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
                ground_truth = yolo_ground_truth(bundle.read(label_name).decode("utf-8"), width, height)
                relative = Path("data") / "ifiresmoke" / f"{category}_{index:03d}.jpg"
                destination = EVALUATION_ROOT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(image_bytes)
                rows.append({
                    "image": relative.as_posix(),
                    "label": 1 if category == "fire" else 0,
                    "task": "识别可见火焰和烟雾，输出定位证据、风险等级及复核建议",
                    "ground_truth_json": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
                    "source": f"shahriar-5/IFireSmoke@{IFIRE_REVISION}",
                    "license": "CC-BY-4.0 (Hub card; archive README states MIT)",
                    "group": f"ifiresmoke:{group}",
                    "source_sha256": file_digest(destination),
                    "content_sha256": file_digest(destination),
                })
    return rows


def prepare_pyro_sdis(client: httpx.Client) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    source_start = 18369
    for relative_offset in range(0, 5537, 400):
        url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=fireviewer%2Ffire-smoke-detection-corpus-v1"
            f"&config=default&split=test&offset={source_start + relative_offset}&length=30"
        )
        response = client.get(url)
        response.raise_for_status()
        for wrapper in response.json()["rows"]:
            row = wrapper["row"]
            if row["source_name"] == "pyro-sdis":
                candidates.append(row)
    selected: list[dict[str, object]] = []
    globally_seen_sequences: set[str] = set()
    for negative in (False, True):
        pool = [row for row in candidates if bool(row["negative"]) is negative]
        category_rows: list[dict[str, object]] = []
        for row in sorted(pool, key=lambda value: selection_key(str(value["sample_id"]))):
            sequence = str(row["sequence_id"])
            if sequence in globally_seen_sequences:
                continue
            globally_seen_sequences.add(sequence)
            category_rows.append(row)
            if len(category_rows) == 10:
                break
        if len(category_rows) < 10:
            raise RuntimeError(f"not enough unique Pyro-SDIS samples for negative={negative}")
        selected.extend(category_rows)

    rows: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        expected_sha = str(row["sha256"])
        relative = Path("data") / "pyro_sdis" / f"sample_{index:03d}.jpg"
        destination = EVALUATION_ROOT / relative
        download(client, str(row["image"]["src"]), destination)
        annotations = json.loads(str(row["annotations_json"]))
        ground_truth = [
            {
                "category": "smoke",
                "bbox": {
                    "x1": round(item["bbox_xywh"][0], 3),
                    "y1": round(item["bbox_xywh"][1], 3),
                    "x2": round(item["bbox_xywh"][0] + item["bbox_xywh"][2], 3),
                    "y2": round(item["bbox_xywh"][1] + item["bbox_xywh"][3], 3),
                },
            }
            for item in annotations
            if item.get("class_name") == "smoke_visible"
        ]
        rows.append({
            "image": relative.as_posix(),
            "label": 0,
            "task": "识别远距离烟雾并判断是否需要人工复核",
            "ground_truth_json": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
            "source": f"fireviewer/fire-smoke-detection-corpus-v1@{FIREVIEWER_REVISION}:pyro-sdis",
            "license": "Apache-2.0",
            "group": str(row["sequence_id"]),
            "source_sha256": expected_sha,
            "content_sha256": file_digest(destination),
        })
    return rows


def main() -> None:
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    headers = {"User-Agent": "VisionGuard-AI-independent-evaluation"}
    with httpx.Client(headers=headers, timeout=180, follow_redirects=True) as client:
        rows = prepare_ifiresmoke(client) + prepare_pyro_sdis(client)
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image",
                "label",
                "task",
                "ground_truth_json",
                "source",
                "license",
                "group",
                "source_sha256",
                "content_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    groups = {
        "ifire_fire": [row for row in rows if "IFireSmoke" in row["source"] and row["label"] == 1],
        "ifire_smoke": [row for row in rows if "IFireSmoke" in row["source"] and row["label"] == 0],
        "pyro_smoke": [
            row for row in rows
            if ":pyro-sdis" in row["source"] and json.loads(row["ground_truth_json"])
        ],
        "pyro_negative": [
            row for row in rows
            if ":pyro-sdis" in row["source"] and not json.loads(row["ground_truth_json"])
        ],
    }
    full_chain_rows = [row for group in groups.values() for row in group[:5]]
    with FULL_CHAIN_MANIFEST.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(full_chain_rows)
    print(json.dumps({
        "manifest": str(MANIFEST),
        "samples": len(rows),
        "fire_images": sum(row["label"] == 1 for row in rows),
        "non_fire_images": sum(row["label"] == 0 for row in rows),
        "selection_salt": SELECTION_SALT,
        "full_chain_samples": len(full_chain_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
