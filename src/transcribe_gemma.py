"""Transcribe segments with Gemma 4's native audio input.

Separate from transcribe.py on purpose. Whisper is a dedicated ASR model with a fixed
decoding contract; Gemma is an instruction-following LLM that happens to accept audio.
The second needs prompt engineering, output cleaning and a different failure surface,
and mixing the two would make both harder to reason about.

Why bother, given Whisper exists: Gemma has no single language token gating the
utterance, so code-switching is just text to it -- and you can *tell it* the orthography
you want. That is Rule 1 of rnd-docs/04 stated directly to the model, which is not
expressible to Whisper at all.

AUDIO SUPPORT: only the E2B, E4B and 12B variants accept audio. The 26B A4B and 31B are
text+image only and will fail here. 12B is the largest audio-capable one.

Reads and writes exactly what transcribe.py does, so build_dataset.py is unchanged:
    data/manifests/segments_{video_id}.jsonl   ->  transcripts_{video_id}.jsonl

Usage
-----
    python src/transcribe_gemma.py --video 0Yj75u7CGWY --limit 50 --show 50 --overwrite
    python src/transcribe_gemma.py --model-id google/gemma-4-E4B-it
    python src/transcribe_gemma.py --prompt "Transcribe this audio verbatim."
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import ensure_dirs, load_audio, load_config, read_jsonl, write_jsonl
from normalize import normalize_text
from transcribe import compression_ratio, flag_segment

SR = 16000

# Variants that actually have an audio encoder. Checked by substring against the repo id.
AUDIO_CAPABLE = ("e2b", "e4b", "12b")

# The default instruction encodes the project's orthography contract (rnd-docs/04
# Rule 1). Keep it blunt and negative-constrained: instruction-tuned models love to
# add "Here is the transcription:" and a closing summary, and every such word is a
# false insertion against the reference.
DEFAULT_PROMPT = (
    "Transcribe this audio exactly as spoken.\n"
    "The speakers mix Bangla and English in the same sentence.\n"
    "Write Bangla words in Bengali script.\n"
    "Write English words in Latin script using standard English spelling -- "
    "do NOT transliterate English words into Bengali script.\n"
    "Transcribe verbatim, including filler words and false starts. "
    "Do not translate, summarise, correct or paraphrase anything.\n"
    "Output ONLY the transcription text. No preamble, no explanation, no quotation "
    "marks, no markdown."
)

# Preambles instruction-tuned models emit despite being told not to.
_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is)(?: the)?[^:\n]{0,40}:|"
    r"transcription:|transcript:|sure[,!.]?|okay[,!.]?|"
    r"the audio says:?|audio transcription:?)\s*",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_TRAILER = re.compile(
    r"\n\s*(note:|this (?:audio|transcript|recording)\b|the speaker\b|"
    r"\(.*(?:translation|note).*\)).*$",
    re.IGNORECASE | re.DOTALL,
)


def clean_output(text: str) -> str:
    """Strip the scaffolding an instruction-tuned model adds around a transcript."""
    text = text.strip()
    text = _FENCE.sub("", text).strip()
    # Preambles can stack ("Sure! Here is the transcription:").
    for _ in range(3):
        new = _PREAMBLE.sub("", text, count=1).strip()
        if new == text:
            break
        text = new
    text = _TRAILER.sub("", text).strip()
    # A whole-output quote wrapper, but not quotes that are part of the speech.
    if len(text) > 1 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    return text


class GemmaBackend:
    name = "gemma"

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        gcfg = cfg["gemma"]
        self.gcfg = gcfg
        model_id = gcfg["model_id"]
        device = gcfg["device"]

        if not any(v in model_id.lower() for v in AUDIO_CAPABLE):
            print(
                f"WARNING: {model_id} does not look like an audio-capable Gemma 4 "
                f"variant. Audio is supported on E2B, E4B and 12B only; 26B A4B and "
                f"31B are text+image. Continuing anyway -- expect it to fail."
            )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(
                "device: cuda requested but torch.cuda.is_available() is False. "
                "See requirements.txt for the right wheel index."
            )

        # Honour the configured dtype on CPU too. Forcing float32 doubles the memory
        # for no accuracy that matters here, and bfloat16 CPU inference works on recent
        # torch -- slow, but the alternative is often not fitting in RAM at all.
        # float16 on CPU is genuinely bad (little kernel coverage), so redirect it.
        want = gcfg["dtype"]
        if device == "cpu" and want == "float16":
            print("note: float16 is poorly supported on CPU; using bfloat16 instead.")
            want = "bfloat16"
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[want]

        print(f"Loading Gemma ({device}, {dtype}): {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        # The exact model class differs across Gemma generations and transformers
        # versions. Try the multimodal classes before the text-only fallback rather
        # than hard-coding one and breaking on the next release.
        model = None
        errors = []
        import transformers

        for cls_name in (
            "Gemma4ForConditionalGeneration",
            "AutoModelForImageTextToText",
            "Gemma3nForConditionalGeneration",
            "AutoModelForCausalLM",
        ):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(model_id, torch_dtype=dtype)
                print(f"  loaded via {cls_name}")
                break
            except Exception as exc:  # noqa: BLE001 - try the next candidate class
                errors.append(f"{cls_name}: {type(exc).__name__}: {exc}")
        if model is None:
            raise SystemExit(
                "Could not load the model with any known class:\n  "
                + "\n  ".join(errors)
            )

        self.model = model.to(device).eval()
        self.device = device
        self.dtype = dtype
        self.prompt = gcfg["prompt"] or DEFAULT_PROMPT
        print(f"Prompt:\n  {self.prompt.splitlines()[0]} ...")

    def _build_inputs(self, wav: np.ndarray):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def transcribe(self, wavs: list[np.ndarray]) -> list[dict]:
        torch = self.torch
        g = self.gcfg
        results = []

        # One at a time: multimodal chat templates batch awkwardly across processor
        # versions, and segments are short. Revisit if throughput is the bottleneck.
        for wav in wavs:
            inputs = self._build_inputs(wav)
            inputs = {
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            n_in = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=g["max_new_tokens"],
                    do_sample=False,          # greedy: transcription is not creative
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

            # Decode only the generated continuation, not the prompt echo.
            raw = self.processor.decode(out[0][n_in:], skip_special_tokens=True)
            text = clean_output(raw)

            results.append(
                {
                    "text": text,
                    "text_normalized": normalize_text(text),
                    # Gemma exposes no per-token ASR confidence comparable to Whisper's.
                    # Leave these null rather than inventing a number the gate would
                    # then treat as meaningful.
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "compression_ratio": round(compression_ratio(text), 3),
                    "raw_output": raw if raw != text else None,
                }
            )
        return results


def transcribe_video(
    video_id: str, cfg: dict, backend, overwrite: bool, limit: int | None = None
) -> int:
    gcfg = cfg["gemma"]
    seg_manifest = Path(cfg["paths"]["manifests"]) / f"segments_{video_id}.jsonl"
    out_manifest = Path(cfg["paths"]["manifests"]) / f"transcripts_{video_id}.jsonl"

    if not seg_manifest.exists():
        return 0
    if out_manifest.exists() and not overwrite:
        return 0

    segments = list(read_jsonl(seg_manifest))
    if limit:
        segments = segments[:limit]
    if not segments:
        return 0

    rows: list[dict] = []
    for seg in tqdm(segments, desc=video_id, leave=False):
        wav = load_audio(seg["audio_path"], cfg["audio"]["sample_rate"])
        pred = backend.transcribe([wav])[0]
        row = {**seg, **pred}
        row["asr_model"] = gcfg["model_id"]
        row["asr_backend"] = backend.name
        row["asr_prompt"] = backend.prompt
        # avg_logprob is absent, so the confidence gate runs on the remaining signals.
        row["flags"] = flag_segment(row, cfg["transcribe"])
        row["tier"] = "auto_low" if row["flags"] else "auto_high"
        rows.append(row)

    write_jsonl(rows, out_manifest)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-id", default=None, help="Hub repo id or local path")
    ap.add_argument("--prompt", default=None, help="override the transcription prompt")
    ap.add_argument(
        "--device", default=None, choices=["cuda", "cpu"],
        help="override gemma.device -- the config is shared across machines, so set "
             "this per run rather than committing a machine-specific value",
    )
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument(
        "--audio", action="append", default=[],
        help="transcribe these audio files directly and print the result, bypassing "
             "the manifests entirely (for smoke-testing the model on any machine)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.model_id:
        cfg["gemma"]["model_id"] = args.model_id
    if args.prompt:
        cfg["gemma"]["prompt"] = args.prompt
    if args.device:
        cfg["gemma"]["device"] = args.device
    if args.dtype:
        cfg["gemma"]["dtype"] = args.dtype

    manifests = Path(cfg["paths"]["manifests"])

    # Ad-hoc mode: no manifests, no pipeline state. The point is to prove the model
    # loads and the chat template works on this machine before trusting a real run.
    if args.audio:
        backend = GemmaBackend(cfg)
        for f in args.audio:
            p = Path(f).expanduser()
            if not p.exists():
                print(f"  ! not found: {p}")
                continue
            wav = load_audio(p, cfg["audio"]["sample_rate"])
            print(f"\n{p.name}  ({len(wav) / cfg['audio']['sample_rate']:.1f}s)")
            pred = backend.transcribe([wav])[0]
            print(f"  TEXT: {pred['text']}")
            if pred.get("raw_output"):
                print(f"  (cleaned from: {pred['raw_output'][:160]!r})")
        return

    video_ids = args.video or sorted(
        p.stem.replace("segments_", "") for p in manifests.glob("segments_*.jsonl")
    )
    if not video_ids:
        raise SystemExit("No segment manifests found. Run segment.py first.")

    backend = GemmaBackend(cfg)

    total = 0
    for vid in tqdm(video_ids, desc="transcribe"):
        try:
            total += transcribe_video(vid, cfg, backend, args.overwrite, args.limit)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"  ! {vid} failed: {exc}")
            traceback.print_exc()
            if args.fail_fast:
                raise

    print(f"\n{total} segments transcribed.")

    if args.show:
        from normalize import codemix_stats

        print(f"\n{'=' * 78}\n  first {args.show} transcripts\n{'=' * 78}")
        shown = 0
        for p in sorted(manifests.glob("transcripts_*.jsonl")):
            for row in read_jsonl(p):
                if shown >= args.show:
                    break
                cm = codemix_stats(row["text"])
                print(
                    f"\n[{shown:3}] {row['id']}  {row['duration']:.1f}s  "
                    f"cmi={cm['cmi']}  latin={cm['latin_ratio']}  tier={row['tier']}"
                    + (f"  flags={','.join(row['flags'])}" if row.get("flags") else "")
                )
                print(f"      {row['text']}")
                if row.get("raw_output"):
                    print(f"      (cleaned from: {row['raw_output'][:120]!r})")
                shown += 1
            if shown >= args.show:
                break


if __name__ == "__main__":
    main()
