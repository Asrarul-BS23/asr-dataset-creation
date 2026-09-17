"""Pause-aware segmentation: cut long audio into 15-25 s chunks at sentence-like breaks.

The idea
--------
A naive segmenter slices every N seconds and routinely cuts mid-word. This one:

1. Runs Silero VAD to find speech regions, so a cut is *only ever* placed inside a
   genuine silence -- never inside speech.
2. Treats every inter-speech gap as a candidate boundary, scored by how long the pause
   is. Long pauses correlate strongly with sentence and clause ends.
3. Chooses the globally best set of boundaries with dynamic programming, rather than
   greedily. Greedy segmentation makes a locally attractive cut and then gets forced
   into a terrible one two chunks later; DP trades a mediocre cut now for two good ones
   afterwards.

The DP minimises, over all valid partitions:

    sum over segments of [ w_target * ((dur - target)/target)^2       # length discipline
                        + w_silence * (dur - speech_dur)              # no dead air inside
                        - w_pause  * min(gap_after, sat)/sat          # cut at long pauses
                        + short_penalty if dur < min_duration ]

with segments longer than `max_duration` disallowed outright. Turn `w_pause` up if you
want cleaner sentence boundaries and care less about uniform length; turn `w_target` up
for the opposite.

Speech regions that are themselves longer than max_duration (someone talking for 40 s
without pausing) cannot be handled by boundary selection at all, so they are pre-split at
their lowest-energy frame inside the legal window.

Usage
-----
    python src/segment.py                     # everything in data/raw not yet segmented
    python src/segment.py --video dQw4w9WgXcQ
    python src/segment.py --overwrite
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    Span,
    ensure_dirs,
    load_audio,
    load_config,
    read_json,
    rms_envelope,
    save_audio,
    write_jsonl,
)

_VAD_MODEL = None


def get_vad():
    global _VAD_MODEL
    if _VAD_MODEL is not None:
        return _VAD_MODEL
    try:
        from silero_vad import load_silero_vad

        _VAD_MODEL = ("pip", load_silero_vad())
    except ImportError:
        import torch

        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        _VAD_MODEL = ("hub", model)
    return _VAD_MODEL


def detect_speech(wav: np.ndarray, sample_rate: int, scfg: dict) -> list[Span]:
    import torch

    kind, model = get_vad()
    tensor = torch.from_numpy(wav)

    if kind == "pip":
        from silero_vad import get_speech_timestamps
    else:
        from torch.hub import load as _load  # noqa: F401

        _, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        get_speech_timestamps = utils[0]

    stamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
        threshold=scfg["vad_threshold"],
        min_silence_duration_ms=scfg["min_silence_ms"],
        min_speech_duration_ms=scfg["min_speech_ms"],
    )
    return [Span(s["start"] / sample_rate, s["end"] / sample_rate) for s in stamps]


def presplit_long_regions(
    regions: list[Span], wav: np.ndarray, sample_rate: int, scfg: dict
) -> list[Span]:
    """Split any speech region longer than max_duration at its quietest interior frame.

    Continuous speech with no VAD-detectable pause still has energy minima -- breath
    points, plosive gaps. Cutting there is far less damaging than cutting at a fixed
    offset.
    """
    max_d = scfg["max_duration"]
    min_d = scfg["min_duration"]
    out: list[Span] = []

    for region in regions:
        cursor = region.start
        while region.end - cursor > max_d:
            lo, hi = cursor + min_d, cursor + max_d
            i0, i1 = int(lo * sample_rate), int(min(hi, region.end) * sample_rate)
            window = wav[i0:i1]
            if len(window) < sample_rate // 10:
                break
            env, hop = rms_envelope(window, sample_rate)
            # Smooth so we pick a genuine trough rather than a one-frame glitch.
            if len(env) >= 5:
                kernel = np.ones(5) / 5
                env = np.convolve(env, kernel, mode="same")
            cut = lo + (int(np.argmin(env)) * hop) / sample_rate
            out.append(Span(cursor, cut))
            cursor = cut
        out.append(Span(cursor, region.end))

    return [r for r in out if r.duration > 0.05]


def plan_segments(regions: list[Span], scfg: dict) -> list[tuple[int, int]]:
    """Dynamic-programming partition of speech regions into segments.

    Returns a list of (first_region_idx, last_region_idx) inclusive pairs.
    """
    n = len(regions)
    if n == 0:
        return []

    min_d = scfg["min_duration"]
    max_d = scfg["max_duration"]
    target = scfg["target_duration"]
    w_target = scfg["w_target"]
    w_pause = scfg["w_pause"]
    w_silence = scfg["w_silence"]
    sat = scfg["pause_saturation"]

    # Prefix sums of pure speech time, for the trapped-silence term.
    speech_prefix = np.zeros(n + 1)
    for i, r in enumerate(regions):
        speech_prefix[i + 1] = speech_prefix[i] + r.duration

    def gap_after(idx: int) -> float:
        if idx >= n - 1:
            return sat  # end of file: a free, maximally good boundary
        return max(0.0, regions[idx + 1].start - regions[idx].end)

    INF = float("inf")
    dp = [INF] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0

    for j in range(1, n + 1):
        last = j - 1
        for i in range(last, -1, -1):
            span = regions[last].end - regions[i].start
            if span > max_d:
                break  # every smaller i only makes the span longer
            if dp[i] == INF:
                continue

            speech_dur = speech_prefix[j] - speech_prefix[i]
            cost = w_target * ((span - target) / target) ** 2
            cost += w_silence * max(0.0, span - speech_dur)
            cost -= w_pause * min(gap_after(last), sat) / sat
            if span < min_d:
                # Allowed only as a tail; made expensive so the DP avoids it otherwise.
                cost += 25.0 * ((min_d - span) / min_d) ** 2 + (0.0 if j == n else 50.0)

            total = dp[i] + cost
            if total < dp[j]:
                dp[j] = total
                back[j] = i

    if dp[n] == INF:
        return []

    out: list[tuple[int, int]] = []
    j = n
    while j > 0:
        i = back[j]
        out.append((i, j - 1))
        j = i
    return list(reversed(out))


def segment_video(video_id: str, cfg: dict, overwrite: bool = False) -> list[dict]:
    scfg = cfg["segment"]
    sr = cfg["audio"]["sample_rate"]

    raw_path = Path(cfg["paths"]["raw"]) / f"{video_id}.flac"
    meta_path = Path(cfg["paths"]["meta"]) / f"{video_id}.json"
    out_dir = Path(cfg["paths"]["segments"]) / video_id
    manifest_path = Path(cfg["paths"]["manifests"]) / f"segments_{video_id}.jsonl"

    if manifest_path.exists() and not overwrite:
        return []

    meta = read_json(meta_path) if meta_path.exists() else {"video_id": video_id}
    wav = load_audio(raw_path, sr)
    total_dur = len(wav) / sr

    regions = detect_speech(wav, sr, scfg)
    if not regions:
        print(f"  ! no speech found in {video_id}")
        return []
    regions = presplit_long_regions(regions, wav, sr, scfg)
    plan = plan_segments(regions, scfg)

    pad = scfg["pad"]
    rows: list[dict] = []
    seq = 0

    for first, last in plan:
        start = max(0.0, regions[first].start - pad)
        end = min(total_dur, regions[last].end + pad)
        duration = end - start

        # Hard guarantee on the contract: every emitted segment is within [min, max].
        if duration < scfg["min_duration"] or duration > scfg["max_duration"]:
            continue

        speech_dur = sum(regions[k].duration for k in range(first, last + 1))
        speech_ratio = speech_dur / duration
        if speech_ratio < scfg["min_speech_ratio"]:
            continue

        seg_id = f"{video_id}_{seq:04d}"
        seg_path = out_dir / f"{seg_id}.flac"
        save_audio(wav[int(start * sr) : int(end * sr)], seg_path, sr)

        pause_before = (
            regions[first].start - regions[first - 1].end if first > 0 else math.inf
        )
        pause_after = (
            regions[last + 1].start - regions[last].end if last + 1 < len(regions) else math.inf
        )

        rows.append(
            {
                "id": seg_id,
                "video_id": video_id,
                "audio_path": str(seg_path),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "speech_ratio": round(speech_ratio, 3),
                "n_speech_regions": last - first + 1,
                "pause_before": None if math.isinf(pause_before) else round(pause_before, 3),
                "pause_after": None if math.isinf(pause_after) else round(pause_after, 3),
                "channel": meta.get("channel"),
                "channel_id": meta.get("channel_id"),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "license": meta.get("license"),
                "permission_ref": meta.get("permission_ref"),
                "upload_date": meta.get("upload_date"),
            }
        )
        seq += 1

    write_jsonl(rows, manifest_path)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", action="append", default=[], help="limit to these video IDs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_dir = Path(cfg["paths"]["raw"])
    video_ids = args.video or sorted(p.stem for p in raw_dir.glob("*.flac"))
    if not video_ids:
        raise SystemExit(f"No audio in {raw_dir}. Run download.py first.")

    total_segments = 0
    total_hours = 0.0
    durations: list[float] = []

    for vid in tqdm(video_ids, desc="segment"):
        try:
            rows = segment_video(vid, cfg, overwrite=args.overwrite)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {vid} failed: {exc}")
            continue
        total_segments += len(rows)
        durations += [r["duration"] for r in rows]
        total_hours += sum(r["duration"] for r in rows) / 3600

    if durations:
        arr = np.array(durations)
        print(
            f"\n{total_segments} segments, {total_hours:.2f} h\n"
            f"duration  mean={arr.mean():.1f}s  min={arr.min():.1f}s  "
            f"max={arr.max():.1f}s  median={np.median(arr):.1f}s"
        )
    else:
        print("\nNo new segments written (already segmented? use --overwrite).")


if __name__ == "__main__":
    main()
