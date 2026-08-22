from __future__ import annotations

import argparse
import asyncio
import csv
from hashlib import sha256
import json
from pathlib import Path

import httpx


DATASET = "fireviewer/fire-smoke-detection-corpus-v1"
REVISION = "85ad763e6275537386f7eefdae5e3a18a55f1c71"
PYRO_START = 18369
# The source currently exposes 5,537 Pyro-SDIS rows, but the final partial page
# intermittently fails in the dataset preview service. The first 5,500 rows are
# sufficient for the deterministic balanced split and avoid depending on it.
PYRO_COUNT = 5500
SELECTION_SALT = "visionguard-pyro-smoke-training-v1"


def selection_key(value: str) -> str:
    return sha256(f"{SELECTION_SALT}|{value}".encode()).hexdigest()


def is_visible_smoke(row: dict) -> bool:
    annotations = json.loads(str(row["annotations_json"]))
    return any(item.get("class_name") == "smoke_visible" for item in annotations)


async def download_one(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, row: dict, destination: Path
) -> None:
    if destination.is_file():
        return
    async with semaphore:
        response = await client.get(str(row["image"]["src"]))
        response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    temporary.write_bytes(response.content)
    temporary.replace(destination)


async def fetch_rows(client: httpx.AsyncClient, offset: int, length: int) -> list[dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET.replace('/', '%2F')}"
        f"&config=default&split=test&offset={PYRO_START + offset}&length={length}"
    )
    for attempt in range(6):
        response = await client.get(url)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return [item["row"] for item in response.json()["rows"]]
        await asyncio.sleep(2 ** attempt)
    response.raise_for_status()
    raise RuntimeError("元数据接口重试耗尽")


def allocate(rows: list[dict], fixed_groups: set[str], per_class: dict[str, int]) -> list[dict]:
    eligible = [
        row for row in rows
        if row["source_name"] == "pyro-sdis"
        and str(row["sequence_id"]) not in fixed_groups
        and (bool(row["negative"]) or is_visible_smoke(row))
    ]
    groups: dict[str, list[dict]] = {}
    for row in eligible:
        groups.setdefault(str(row["sequence_id"]), []).append(row)
    splits = ("train", "valid", "test")
    ranges = {"train": (0, 70), "valid": (70, 85), "test": (85, 100)}
    selected: list[dict] = []
    for split in splits:
        target = per_class[split]
        counts = {0: 0, 1: 0}
        low, high = ranges[split]
        split_groups = [
            group for group in groups
            if low <= int(selection_key(group)[:8], 16) % 100 < high
        ]
        for group in sorted(split_groups, key=selection_key):
            if counts[0] >= target and counts[1] >= target:
                continue
            candidates = sorted(groups[group], key=lambda row: selection_key(str(row["sample_id"])))
            additions: list[dict] = []
            for row in candidates:
                label = int(not bool(row["negative"]) and is_visible_smoke(row))
                if counts[label] >= target:
                    continue
                # Cap each sequence so a single camera-day cannot dominate a split.
                if sum(int(item["label"]) == label for item in additions) >= 20:
                    continue
                additions.append({**row, "split_name": split, "label": label})
                counts[label] += 1
            if additions:
                selected.extend(additions)
        if counts != {0: target, 1: target}:
            raise RuntimeError(f"无法为{split}构造平衡样本: {counts}")
    return selected


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent-manifest", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=350)
    parser.add_argument("--valid-per-class", type=int, default=80)
    parser.add_argument("--test-per-class", type=int, default=150)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixed_groups: set[str] = set()
    with args.independent_manifest.resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if ":pyro-sdis" in row["source"]:
                fixed_groups.add(row["group"])

    rows: list[dict] = []
    metadata_cache = output / "metadata.json"
    headers = {"User-Agent": "VisionGuard-AI-smoke-training"}
    async with httpx.AsyncClient(headers=headers, timeout=180, follow_redirects=True) as client:
        if metadata_cache.is_file():
            rows = json.loads(metadata_cache.read_text(encoding="utf-8"))
            print(f"metadata cache {len(rows)}/{PYRO_COUNT}", flush=True)
        else:
            requests = [
                (offset, min(100, PYRO_COUNT - offset))
                for offset in range(0, PYRO_COUNT, 100)
            ]
            for start in range(0, len(requests), 3):
                current = requests[start:start + 3]
                batches = await asyncio.gather(
                    *(fetch_rows(client, offset, length) for offset, length in current)
                )
                for batch in batches:
                    rows.extend(batch)
                done = min(current[-1][0] + current[-1][1], PYRO_COUNT)
                print(f"metadata {done}/{PYRO_COUNT}", flush=True)
            metadata_cache.write_text(json.dumps(rows), encoding="utf-8")
        selected = allocate(
            rows,
            fixed_groups,
            {
                "train": args.train_per_class,
                "valid": args.valid_per_class,
                "test": args.test_per_class,
            },
        )
        semaphore = asyncio.Semaphore(16)
        tasks = []
        for index, row in enumerate(selected):
            destination = output / "images" / f"{row['sample_id'].split(':')[-1]}.jpg"
            row["local_path"] = destination.relative_to(output).as_posix()
            tasks.append(download_one(client, semaphore, row, destination))
        for start in range(0, len(tasks), 100):
            await asyncio.gather(*tasks[start:start + 100])
            print(f"images {min(start + 100, len(tasks))}/{len(tasks)}", flush=True)

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["split", "image", "label", "group", "sample_id", "source_sha256"],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "split": row["split_name"],
                "image": row["local_path"],
                "label": row["label"],
                "group": row["sequence_id"],
                "sample_id": row["sample_id"],
                "source_sha256": row["sha256"],
            })
    summary = {
        "schema_version": 1,
        "source": f"{DATASET}@{REVISION}:pyro-sdis",
        "selection_salt": SELECTION_SALT,
        "excluded_fixed_evaluation_groups": len(fixed_groups),
        "samples": len(selected),
        "splits": {
            split: {
                "samples": sum(row["split_name"] == split for row in selected),
                "smoke": sum(row["split_name"] == split and row["label"] == 1 for row in selected),
                "no_smoke": sum(row["split_name"] == split and row["label"] == 0 for row in selected),
                "groups": len({row["sequence_id"] for row in selected if row["split_name"] == split}),
            }
            for split in ("train", "valid", "test")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async())
