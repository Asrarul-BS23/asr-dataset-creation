"""Run the whole pipeline end to end: download -> segment -> transcribe -> dataset.

Each stage is idempotent and skips work already done, so re-running after an interrupt
is safe and cheap.

Usage
-----
    python src/run_pipeline.py --urls urls.txt
    python src/run_pipeline.py --urls urls.txt --skip download
    python src/run_pipeline.py --urls urls.txt --model-id myname/whisper-banglish --push
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(SRC / script)] + extra
    print(f"\n{'=' * 70}\n  {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"{script} failed with exit code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--urls", help="file with one URL / video ID / channel per line")
    ap.add_argument("--channel", action="append", default=[])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument(
        "--local-path",
        action="append",
        default=[],
        help="local audio/video file or directory to ingest instead of downloading",
    )
    ap.add_argument("--recursive", action="store_true", help="with --local-path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-id", default=None, help="Hub repo id or local directory path")
    ap.add_argument("--model-path", default=None, help="alias for --model-id (local disk)")
    ap.add_argument("--exclude-flagged", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--hub-id", default=None)
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["download", "segment", "transcribe", "dataset"],
    )
    args = ap.parse_args()

    common = ["--config", args.config] if args.config else []

    if "download" not in args.skip:
        # Local files and YouTube both land in data/raw + data/meta, so they can be
        # mixed freely in one corpus; every later stage reads only those directories.
        if args.local_path:
            extra = list(common)
            for p in args.local_path:
                extra += ["--path", p]
            if args.recursive:
                extra.append("--recursive")
            run("ingest_local.py", extra)

        extra = list(common)
        if args.urls:
            extra += ["--urls", args.urls]
        for c in args.channel:
            extra += ["--channel", c]
        for v in args.video:
            extra += ["--video", v]
        if args.limit:
            extra += ["--limit", str(args.limit)]
        if len(extra) > len(common):
            run("download.py", extra)
        elif not args.local_path:
            print("No sources given; skipping download.")

    if "segment" not in args.skip:
        run("segment.py", list(common))

    if "transcribe" not in args.skip:
        extra = list(common)
        if args.model_id or args.model_path:
            extra += ["--model-id", args.model_id or args.model_path]
        run("transcribe.py", extra)

    if "dataset" not in args.skip:
        extra = list(common)
        if args.exclude_flagged:
            extra.append("--exclude-flagged")
        if args.push:
            extra.append("--push")
        if args.hub_id:
            extra += ["--hub-id", args.hub_id]
        run("build_dataset.py", extra)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
