"""Ingest local audio or video files into the pipeline.

The YouTube path is not the only way in. `data/raw/` + `data/meta/` is the interface
every later stage reads, so anything landed there in the right shape flows through
segmentation, transcription and dataset building unchanged.

Accepts any format ffmpeg can decode -- mp4, mkv, mov, webm, mp3, m4a, wav, flac, ogg,
opus, aac, wma -- and normalises to 16 kHz mono FLAC.

Usage
-----
    python src/ingest_local.py --path recording.mp4
    python src/ingest_local.py --path ~/podcasts --recursive
    python src/ingest_local.py --path ~/audio --glob "*.mp3" --speaker-hint "guest_A"

Then carry on as normal -- these files are indistinguishable from downloaded ones:

    python src/segment.py
    python src/transcribe.py --model-path <your-model>
    python src/build_dataset.py

IDs are derived from the filename plus a short hash of the absolute path, so two files
called `interview.mp3` in different folders do not collide, and re-running is
idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from common import ensure_dirs, have_ffmpeg, load_config, write_json

MEDIA_EXTENSIONS = {
    # audio
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".alac",
    # video (audio track is extracted)
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg",
}

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def make_id(path: Path, prefix: str = "") -> str:
    """Readable, collision-free, stable across runs."""
    stem = _SAFE.sub("_", path.stem).strip("_")[:48] or "audio"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}{stem}_{digest}"


def probe_duration(path: Path) -> float:
    """Seconds, via ffprobe. Returns 0.0 if it cannot be determined."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True)
        return float(out.stdout.decode().strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0.0


def has_audio_stream(path: Path) -> bool:
    """A video with no audio track would otherwise produce an empty FLAC."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True)
        return b"audio" in out.stdout
    except (OSError, subprocess.CalledProcessError):
        return False


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect_files(paths: list[str], pattern: str | None, recursive: bool) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            globber = p.rglob if recursive else p.glob
            if pattern:
                found += sorted(globber(pattern))
            else:
                found += sorted(
                    f for f in globber("*") if f.suffix.lower() in MEDIA_EXTENSIONS
                )
        else:
            print(f"  ! not found: {p}", file=sys.stderr)

    # De-duplicate by resolved path, preserving order.
    seen: set[Path] = set()
    out = []
    for f in found:
        r = f.resolve()
        if r not in seen and f.suffix.lower() in MEDIA_EXTENSIONS:
            seen.add(r)
            out.append(f)
    return out


def ingest_one(src: Path, cfg: dict, prefix: str, extra_meta: dict, overwrite: bool) -> str:
    """Returns 'ok' | 'skipped' | 'no_audio' | 'error'."""
    acfg = cfg["audio"]
    vid = make_id(src, prefix)
    out_flac = Path(cfg["paths"]["raw"]) / f"{vid}.flac"
    out_meta = Path(cfg["paths"]["meta"]) / f"{vid}.json"

    if out_flac.exists() and out_meta.exists() and not overwrite:
        return "skipped"

    if not has_audio_stream(src):
        print(f"  ! no audio stream in {src.name}", file=sys.stderr)
        return "no_audio"

    out_flac.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", str(src),
        "-vn",                                  # drop any video stream
        "-ac", str(acfg["channels"]),
        "-ar", str(acfg["sample_rate"]),
        "-c:a", "flac",
        str(out_flac),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        print(
            f"  ! ffmpeg failed on {src.name}:\n"
            f"{proc.stderr.decode('utf-8', 'ignore')[-800:]}",
            file=sys.stderr,
        )
        return "error"

    write_json(
        {
            "video_id": vid,
            "source_type": "local",
            "url": None,
            "title": src.stem,
            "channel": extra_meta.get("channel") or "local",
            "channel_id": None,
            "original_path": str(src.resolve()),
            "original_format": src.suffix.lower().lstrip("."),
            "original_sha256": sha256(src),
            "duration": probe_duration(out_flac),
            # Local material is yours (or permissioned) -- record which, because the
            # dataset card needs it and retrofitting provenance is miserable.
            "license": extra_meta.get("license") or "unspecified",
            "permission_ref": extra_meta.get("permission_ref"),
            "speaker_hint": extra_meta.get("speaker_hint"),
            "upload_date": None,
            "audio_sha256": sha256(out_flac),
            "sample_rate": acfg["sample_rate"],
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
        out_meta,
    )
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--path", action="append", required=True,
        help="file or directory (repeatable)",
    )
    ap.add_argument("--glob", default=None, help='e.g. "*.mp3" (directories only)')
    ap.add_argument("--recursive", action="store_true", help="descend into subdirectories")
    ap.add_argument("--id-prefix", default="", help='e.g. "field_" to tag a batch')
    ap.add_argument("--channel", default=None, help="group label, stored as `channel`")
    ap.add_argument("--license", default=None, help='e.g. "own-recording", "cc-by"')
    ap.add_argument("--permission-ref", default=None, help="consent form / email reference")
    ap.add_argument("--speaker-hint", default=None, help="free-text speaker note")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if not have_ffmpeg():
        sys.exit("ffmpeg not found on PATH. Install it before running this script.")

    files = collect_files(args.path, args.glob, args.recursive)
    if not files:
        sys.exit("No media files matched. Check --path, --glob and --recursive.")
    print(f"{len(files)} media file(s) to ingest")

    extra = {
        "channel": args.channel,
        "license": args.license,
        "permission_ref": args.permission_ref,
        "speaker_hint": args.speaker_hint,
    }

    tally = {"ok": 0, "skipped": 0, "no_audio": 0, "error": 0}
    for f in tqdm(files, desc="ingest"):
        tally[ingest_one(f, cfg, args.id_prefix, extra, args.overwrite)] += 1

    print(f"\n{tally}")

    total = 0.0
    for p in Path(cfg["paths"]["meta"]).glob("*.json"):
        with open(p, encoding="utf-8") as fh:
            total += json.load(fh).get("duration", 0) or 0
    print(f"Corpus now holds {total / 3600:.2f} hours of raw audio.")
    print("\nNext:  python src/segment.py")


if __name__ == "__main__":
    main()
