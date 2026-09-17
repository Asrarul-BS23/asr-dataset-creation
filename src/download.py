"""Download YouTube audio to 16 kHz mono FLAC, with provenance metadata.

Usage
-----
    python src/download.py --urls urls.txt
    python src/download.py --channel "https://www.youtube.com/@SomeChannel/videos"
    python src/download.py --video dQw4w9WgXcQ

`urls.txt` takes one video URL, video ID, or channel/playlist URL per line
(blank lines and `#` comments ignored).

Every download writes two files:
    data/raw/{video_id}.flac    16 kHz mono, immutable, never edited afterwards
    data/meta/{video_id}.json   title, channel, license, duration, sha256, fetch time

Re-running skips anything already present, so it is safe to interrupt.

NOTE ON LICENSING: set `download.license_filter: creativeCommon` in the config if you
intend to redistribute audio. See rnd-docs/03-youtube-sourcing.md before publishing
anything.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from common import ensure_dirs, have_ffmpeg, load_config, read_json, write_json


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def expand_sources(entries: list[str]) -> list[str]:
    """Turn channel/playlist URLs into flat lists of video IDs."""
    import yt_dlp

    video_ids: list[str] = []
    flat_opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True}

    for entry in entries:
        entry = entry.strip()
        if not entry or entry.startswith("#"):
            continue

        # A bare 11-character video ID.
        if len(entry) == 11 and "/" not in entry:
            video_ids.append(entry)
            continue

        if "watch?v=" in entry or "youtu.be/" in entry:
            vid = entry.split("watch?v=")[-1].split("youtu.be/")[-1]
            video_ids.append(vid.split("&")[0].split("?")[0])
            continue

        # Channel or playlist: enumerate it.
        with yt_dlp.YoutubeDL(flat_opts) as ydl:
            try:
                info = ydl.extract_info(entry, download=False)
            except Exception as exc:  # noqa: BLE001 - one bad channel must not kill the run
                print(f"  ! could not expand {entry}: {exc}", file=sys.stderr)
                continue
        for item in info.get("entries") or []:
            if not item:
                continue
            if item.get("_type") == "playlist":  # /videos pages nest one level
                for sub in item.get("entries") or []:
                    if sub and sub.get("id"):
                        video_ids.append(sub["id"])
            elif item.get("id"):
                video_ids.append(item["id"])

    # De-duplicate, preserving order.
    seen: set[str] = set()
    return [v for v in video_ids if not (v in seen or seen.add(v))]


def download_one(video_id: str, cfg: dict) -> str:
    """Returns one of: 'ok', 'skipped', 'filtered', 'error'."""
    import yt_dlp

    raw_dir = Path(cfg["paths"]["raw"])
    meta_dir = Path(cfg["paths"]["meta"])
    out_flac = raw_dir / f"{video_id}.flac"
    out_meta = meta_dir / f"{video_id}.json"

    if out_flac.exists() and out_meta.exists():
        return "skipped"

    dcfg = cfg["download"]
    acfg = cfg["audio"]

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(raw_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": acfg["codec"],
                "preferredquality": "0",
            }
        ],
        # Force the sample rate and channel count during the postprocessing pass so we
        # never keep a 48 kHz stereo intermediate around.
        "postprocessor_args": {
            "extractaudio": ["-ar", str(acfg["sample_rate"]), "-ac", str(acfg["channels"])]
        },
        "retries": 3,
        "ignoreerrors": False,
    }

    url = f"https://www.youtube.com/watch?v={video_id}"

    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as probe:
        try:
            info = probe.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! metadata failed for {video_id}: {exc}", file=sys.stderr)
            return "error"

    duration = info.get("duration") or 0
    if not (dcfg["min_duration"] <= duration <= dcfg["max_duration"]):
        return "filtered"
    if dcfg["skip_live"] and (info.get("is_live") or info.get("was_live")):
        return "filtered"
    lic = info.get("license") or "standard"
    if dcfg["license_filter"] != "any" and lic != dcfg["license_filter"]:
        return "filtered"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except Exception as exc:  # noqa: BLE001
            print(f"  ! download failed for {video_id}: {exc}", file=sys.stderr)
            return "error"

    if not out_flac.exists():
        print(f"  ! expected {out_flac} but it is missing", file=sys.stderr)
        return "error"

    write_json(
        {
            "video_id": video_id,
            "url": url,
            "title": info.get("title"),
            "channel": info.get("channel") or info.get("uploader"),
            "channel_id": info.get("channel_id"),
            "channel_url": info.get("channel_url"),
            "upload_date": info.get("upload_date"),
            "duration": duration,
            "license": lic,
            "view_count": info.get("view_count"),
            "categories": info.get("categories"),
            "tags": info.get("tags"),
            "has_manual_subs": bool(info.get("subtitles")),
            "audio_sha256": _sha256(out_flac),
            "sample_rate": acfg["sample_rate"],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            # Fill this in by hand when you have written permission from the channel.
            "permission_ref": None,
        },
        out_meta,
    )
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--urls", help="file with one URL / video ID / channel per line")
    ap.add_argument("--channel", action="append", default=[], help="channel or playlist URL")
    ap.add_argument("--video", action="append", default=[], help="video URL or ID")
    ap.add_argument("--limit", type=int, default=None, help="cap total videos this run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if not have_ffmpeg():
        sys.exit("ffmpeg not found on PATH. Install it before running this script.")

    entries: list[str] = list(args.channel) + list(args.video)
    if args.urls:
        entries += Path(args.urls).read_text(encoding="utf-8").splitlines()
    if not entries:
        sys.exit("Nothing to do. Pass --urls, --channel or --video.")

    print("Expanding sources...")
    video_ids = expand_sources(entries)
    per_channel_cap = cfg["download"]["max_videos_per_channel"]
    if args.limit:
        video_ids = video_ids[: args.limit]
    elif per_channel_cap:
        video_ids = video_ids[: per_channel_cap * max(1, len(entries))]
    print(f"{len(video_ids)} candidate videos")

    tally = {"ok": 0, "skipped": 0, "filtered": 0, "error": 0}
    for vid in tqdm(video_ids, desc="download"):
        status = download_one(vid, cfg)
        tally[status] += 1
        if status == "ok":
            time.sleep(cfg["download"]["sleep_interval"])

    print(f"\n{tally}")
    total_h = sum(
        read_json(p).get("duration", 0) for p in Path(cfg["paths"]["meta"]).glob("*.json")
    ) / 3600
    print(f"Corpus now holds {total_h:.1f} hours of raw audio.")


if __name__ == "__main__":
    main()
