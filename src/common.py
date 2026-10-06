"""Shared helpers: config loading, paths, JSONL I/O, audio I/O."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml


def use_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and crash on Bengali output."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


use_utf8_stdout()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Resolve every path relative to the repo root so scripts work from any cwd.
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for value in cfg["paths"].values():
        Path(value).mkdir(parents=True, exist_ok=True)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- audio


def have_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode any audio file to a mono float32 numpy array at `sample_rate`.

    Fast path: our own segments are already 16 kHz mono WAV/FLAC, which soundfile
    reads directly. That skips a subprocess spawn per segment (hundreds per run) and
    means transcription needs no ffmpeg at all -- useful on a machine that only has
    the segments, not the harvesting toolchain.

    Everything else falls through to ffmpeg, so we still do not care what container
    yt-dlp handed us.
    """
    path = Path(path)
    if path.suffix.lower() in (".wav", ".flac"):
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr == sample_rate:
                return np.ascontiguousarray(data, dtype=np.float32)
        except Exception:  # noqa: BLE001 - any problem: use ffmpeg instead
            pass

    cmd = [
        "ffmpeg", "-nostdin", "-threads", "1", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {path}:\n{proc.stderr.decode('utf-8', 'ignore')[-2000:]}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def save_audio(wav: np.ndarray, path: str | Path, sample_rate: int = 16000) -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate)


def rms_envelope(wav: np.ndarray, sample_rate: int, frame_ms: int = 20) -> tuple[np.ndarray, int]:
    """Frame-wise RMS. Returns (envelope, hop_samples)."""
    hop = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = max(1, len(wav) // hop)
    trimmed = wav[: n_frames * hop].reshape(n_frames, hop)
    return np.sqrt((trimmed.astype(np.float64) ** 2).mean(axis=1) + 1e-12), hop


@dataclass
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start
