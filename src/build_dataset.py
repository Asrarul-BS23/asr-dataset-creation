"""Assemble transcript manifests into a Hugging Face `datasets.DatasetDict`.

Produces the standard ASR shape -- an `Audio` column plus a text column -- so the result
drops straight into a Whisper training script or `load_dataset`.

Splits are grouped by `video_id`, never random over segments. A random segment split
leaks the same speaker and recording condition into train and test and produces numbers
that collapse the moment you run on real audio.

Usage
-----
    python src/build_dataset.py
    python src/build_dataset.py --min-logprob -0.8 --exclude-flagged
    python src/build_dataset.py --push --hub-id myname/banglish-asr --private
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, Features, Value

from common import load_config, read_jsonl
from normalize import NORMALIZER_VERSION, codemix_stats

FEATURES = Features(
    {
        "id": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "text": Value("string"),
        "text_normalized": Value("string"),
        "duration": Value("float32"),
        # provenance
        "video_id": Value("string"),
        "channel": Value("string"),
        "channel_id": Value("string"),
        "title": Value("string"),
        "url": Value("string"),
        "license": Value("string"),
        "permission_ref": Value("string"),
        "upload_date": Value("string"),
        "start": Value("float32"),
        "end": Value("float32"),
        # segmentation quality
        "speech_ratio": Value("float32"),
        "pause_before": Value("float32"),
        "pause_after": Value("float32"),
        # pseudo-label confidence
        "avg_logprob": Value("float32"),
        "compression_ratio": Value("float32"),
        "asr_model": Value("string"),
        "tier": Value("string"),
        "flags": Value("string"),
        # code-mixing
        "cmi": Value("float32"),
        "spf": Value("float32"),
        "matrix_language": Value("string"),
        "latin_ratio": Value("float32"),
        "n_tokens": Value("int32"),
        "normalizer_version": Value("string"),
    }
)


def collect_rows(cfg: dict, args) -> list[dict]:
    manifests = Path(cfg["paths"]["manifests"])
    rows: list[dict] = []
    dropped = defaultdict(int)

    for path in sorted(manifests.glob("transcripts_*.jsonl")):
        for r in read_jsonl(path):
            if not r.get("text"):
                dropped["empty"] += 1
                continue
            if args.exclude_flagged and r.get("flags"):
                dropped["flagged"] += 1
                continue
            if args.min_logprob is not None:
                lp = r.get("avg_logprob")
                if lp is None or lp < args.min_logprob:
                    dropped["low_logprob"] += 1
                    continue
            if not Path(r["audio_path"]).exists():
                dropped["missing_audio"] += 1
                continue

            stats = codemix_stats(r["text"])
            rows.append(
                {
                    "id": r["id"],
                    "audio": r["audio_path"],
                    "text": r["text"],
                    "text_normalized": r.get("text_normalized", ""),
                    "duration": r["duration"],
                    "video_id": r["video_id"],
                    "channel": r.get("channel") or "",
                    "channel_id": r.get("channel_id") or "",
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "license": r.get("license") or "",
                    "permission_ref": r.get("permission_ref") or "",
                    "upload_date": r.get("upload_date") or "",
                    "start": r["start"],
                    "end": r["end"],
                    "speech_ratio": r.get("speech_ratio", 0.0),
                    "pause_before": r.get("pause_before") if r.get("pause_before") is not None else -1.0,
                    "pause_after": r.get("pause_after") if r.get("pause_after") is not None else -1.0,
                    "avg_logprob": r.get("avg_logprob") if r.get("avg_logprob") is not None else 0.0,
                    "compression_ratio": r.get("compression_ratio", 0.0),
                    "asr_model": r.get("asr_model") or "",
                    "tier": r.get("tier") or "auto_high",
                    "flags": ",".join(r.get("flags") or []),
                    "cmi": stats["cmi"],
                    "spf": stats["spf"],
                    "matrix_language": stats["matrix_language"],
                    "latin_ratio": stats["latin_ratio"],
                    "n_tokens": stats["n_tokens"],
                    "normalizer_version": NORMALIZER_VERSION,
                }
            )

    if dropped:
        print(f"dropped: {dict(dropped)}")
    return rows


def grouped_split(rows: list[dict], cfg: dict) -> DatasetDict:
    """Split by video so no recording appears in two splits."""
    dcfg = cfg["dataset"]
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)

    videos = sorted(by_video)
    random.Random(dcfg["seed"]).shuffle(videos)

    total = sum(r["duration"] for r in rows)
    want_test = total * dcfg["test_size"]
    want_dev = total * dcfg["dev_size"]

    test, dev, train = [], [], []
    acc_test = acc_dev = 0.0
    for vid in videos:
        chunk = by_video[vid]
        dur = sum(r["duration"] for r in chunk)
        if acc_test < want_test:
            test += chunk
            acc_test += dur
        elif acc_dev < want_dev:
            dev += chunk
            acc_dev += dur
        else:
            train += chunk

    out = {}
    for name, split in (("train", train), ("validation", dev), ("test", test)):
        if split:
            out[name] = Dataset.from_list(split, features=FEATURES)
    return DatasetDict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--exclude-flagged", action="store_true", help="drop repetition loops etc.")
    ap.add_argument("--min-logprob", type=float, default=None, help="e.g. -0.8")
    ap.add_argument("--out", default=None, help="override paths.dataset")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--hub-id", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    rows = collect_rows(cfg, args)
    if not rows:
        raise SystemExit("No usable rows. Run transcribe.py first, or relax the filters.")

    ds = grouped_split(rows, cfg)

    out_dir = args.out or cfg["paths"]["dataset"]
    ds.save_to_disk(out_dir)

    print(f"\nSaved to {out_dir}\n{ds}")
    for name, split in ds.items():
        hours = sum(split["duration"]) / 3600
        mean_cmi = sum(split["cmi"]) / len(split)
        print(f"  {name:<11} {len(split):>6} segs  {hours:6.2f} h  mean CMI {mean_cmi:5.1f}")

    if args.push or cfg["dataset"]["push_to_hub"]:
        hub_id = args.hub_id or cfg["dataset"]["hub_id"]
        if "YOUR_HF_USERNAME" in hub_id:
            raise SystemExit("Set dataset.hub_id in the config or pass --hub-id.")
        private = args.private or cfg["dataset"]["private"]
        print(f"\nPushing to {hub_id} (private={private})...")
        ds.push_to_hub(hub_id, private=private)
        print("Done. Write a dataset card -- see rnd-docs/07.")


if __name__ == "__main__":
    main()
