"""Transcribe segments with your own Whisper model from the Hugging Face Hub.

Handles both repo shapes automatically:
  * a full fine-tune  -> loaded directly
  * a PEFT/LoRA adapter -> base model pulled from adapter_config.json, adapter merged in

Per segment it records the decoder's own quality signals (average token log-probability
and the gzip compression ratio of the text) so the confidence gate in
rnd-docs/05-pseudo-label-pipeline.md has something to work with. Do not skip these --
they are what lets you route segments to annotators by expected error rather than at
random.

Usage
-----
    python src/transcribe.py
    python src/transcribe.py --model-id myname/whisper-large-v3-banglish-lora
    python src/transcribe.py --video dQw4w9WgXcQ --overwrite
"""

from __future__ import annotations

import argparse
import platform
import zlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import ensure_dirs, load_audio, load_config, read_jsonl, write_jsonl
from normalize import normalize_text

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _wheel_index_url() -> str:
    """The right PyTorch wheel index for this machine.

    cuDNN, cuBLAS and the CUDA runtime are all vendored inside these wheels -- there is
    no separate cuDNN install step, and a system CUDA toolkit is not required.

      aarch64 -> cu130   DGX Spark (GB10, Blackwell sm_121) needs CUDA 13 wheels;
                         the cu124 x86 wheels have no sm_121 kernels and no arm64 build.
      x86_64  -> cu124   Colab and ordinary NVIDIA boxes.
    """
    if platform.machine().lower() in ("aarch64", "arm64"):
        return "https://download.pytorch.org/whl/cu130"
    return "https://download.pytorch.org/whl/cu124"


def resolve_model_source(model_id: str) -> tuple[bool, str | None, bool]:
    """Work out what `model_id` points at.

    Accepts a Hub repo id ("me/whisper-banglish") or a local directory
    ("D:/models/whisper-banglish", "./checkpoints/step-4000"). Returns
    (is_peft_adapter, base_model_id, is_local).

    A local path short-circuits entirely -- no network call is made, so this works
    offline and never mistakes a Windows path for a repo id.
    """
    import json

    local_dir = Path(model_id).expanduser()
    if local_dir.is_dir():
        adapter_cfg = local_dir / "adapter_config.json"
        if adapter_cfg.exists():
            base = json.loads(adapter_cfg.read_text(encoding="utf-8")).get(
                "base_model_name_or_path"
            )
            return True, base, True
        if not (local_dir / "config.json").exists():
            raise SystemExit(
                f"{local_dir} exists but holds neither config.json (full model) nor "
                f"adapter_config.json (LoRA adapter)."
            )
        return False, None, True

    # Looks like a path the user meant but mistyped -- fail loudly rather than
    # asking the Hub for a repo named "D:/models/...".
    if any(sep in model_id for sep in ("\\", "/")) and len(model_id.split("/")) != 2:
        raise SystemExit(f"Local model path not found: {model_id}")
    if Path(model_id).suffix or model_id.startswith("."):
        raise SystemExit(f"Local model path not found: {model_id}")

    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "adapter_config.json")
    except Exception:  # noqa: BLE001 - any failure just means "not an adapter repo"
        return False, None, False
    return True, json.loads(Path(path).read_text(encoding="utf-8")).get(
        "base_model_name_or_path"
    ), False


def load_model(cfg: dict):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    tcfg = cfg["transcribe"]
    model_id = tcfg["model_id"]
    device = tcfg["device"]

    if device.startswith("cuda"):
        # Fail loudly. A silent CPU fallback on large-v3 is ~50x slower and easy to miss
        # until you have burned a night on it.
        if not torch.cuda.is_available():
            raise SystemExit(
                "device: cuda requested but torch.cuda.is_available() is False.\n"
                f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
                f"machine {platform.machine()}\n"
                f"Install a CUDA build (cuDNN ships inside the wheel, nothing extra needed):\n"
                f"  pip install torch torchaudio --index-url {_wheel_index_url()}\n"
                "Or set transcribe.device: cpu to run without a GPU."
            )
        # cuDNN is bundled with the torch wheel; these just flip flags on it.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # fixed 30 s input shape -> stable kernels

        props = torch.cuda.get_device_properties(0)
        cudnn_v = torch.backends.cudnn.version()
        print(
            f"GPU: {props.name}  sm_{props.major}{props.minor}  "
            f"{props.total_memory / 1e9:.0f} GB\n"
            f"     torch {torch.__version__} / CUDA {torch.version.cuda} / "
            f"cuDNN {cudnn_v} / {platform.machine()}"
        )

    dtype = DTYPES[tcfg["dtype"]] if device != "cpu" else torch.float32
    if device.startswith("cuda") and dtype is torch.float16:
        if torch.cuda.get_device_properties(0).major >= 8:
            print("note: bfloat16 is the better choice on this GPU; set transcribe.dtype.")

    is_peft, adapter_base, is_local = resolve_model_source(model_id)
    base_id = tcfg.get("base_model_id") or adapter_base or model_id
    if is_peft and not base_id:
        raise SystemExit(
            f"{model_id} is a LoRA adapter but its adapter_config.json has no "
            f"base_model_name_or_path. Set transcribe.base_model_id (or pass "
            f"--base-model-id), e.g. openai/whisper-large-v3."
        )

    source = "local" if is_local else "hub"
    if is_peft:
        print(f"PEFT adapter detected ({source}).\n  base    : {base_id}\n  adapter : {model_id}")
        from peft import PeftModel

        model = WhisperForConditionalGeneration.from_pretrained(
            base_id, torch_dtype=dtype, attn_implementation="sdpa"
        )
        model = PeftModel.from_pretrained(model, model_id, torch_dtype=dtype)
        model = model.merge_and_unload()  # fold LoRA into the base weights for fast inference
    else:
        print(f"Loading full model ({source}): {model_id}")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        )

    # The processor usually lives with the fine-tune; fall back to the base repo.
    try:
        processor = WhisperProcessor.from_pretrained(model_id)
    except (OSError, ValueError):
        processor = WhisperProcessor.from_pretrained(base_id)

    model.to(device).eval()
    print(f"Model on {device} ({dtype}).")
    return model, processor, device, dtype


def compression_ratio(text: str) -> float:
    """Whisper's repetition detector. >2.4 means the decoder is looping."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


@torch.no_grad()
def transcribe_batch(batch_wavs, model, processor, device, dtype, tcfg) -> list[dict]:
    sr = 16000
    features = processor(
        batch_wavs, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
    )
    input_features = features.input_features.to(device, dtype=dtype)

    gen_kwargs = {
        "num_beams": tcfg["num_beams"],
        "language": tcfg["language"],
        "task": tcfg["task"],
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": 440,
    }
    if tcfg.get("no_repeat_ngram_size"):
        gen_kwargs["no_repeat_ngram_size"] = tcfg["no_repeat_ngram_size"]
    if tcfg.get("initial_prompt"):
        gen_kwargs["prompt_ids"] = processor.get_prompt_ids(
            tcfg["initial_prompt"], return_tensors="pt"
        ).to(device)

    out = model.generate(input_features, **gen_kwargs)

    # Average per-token log probability -- the primary confidence signal.
    try:
        transition = model.compute_transition_scores(
            out.sequences,
            out.scores,
            getattr(out, "beam_indices", None),
            normalize_logits=True,
        )
        transition = transition.float().cpu().numpy()
        avg_logprobs = []
        for row in transition:
            valid = row[np.isfinite(row)]
            avg_logprobs.append(float(valid.mean()) if valid.size else float("nan"))
    except Exception:  # noqa: BLE001 - scoring must never kill a transcription run
        avg_logprobs = [float("nan")] * out.sequences.shape[0]

    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)

    results = []
    for text, lp in zip(texts, avg_logprobs):
        text = text.strip()
        cr = compression_ratio(text)
        results.append(
            {
                "text": text,
                "text_normalized": normalize_text(text),
                "avg_logprob": None if np.isnan(lp) else round(lp, 4),
                "compression_ratio": round(cr, 3),
            }
        )
    return results


def flag_segment(row: dict, tcfg: dict) -> list[str]:
    """Cheap auto-filters. See rnd-docs/05 section 4."""
    flags = []
    if not row["text"]:
        flags.append("empty")
    if row["compression_ratio"] > tcfg["compression_ratio_threshold"]:
        flags.append("repetition_loop")
    if row["avg_logprob"] is not None and row["avg_logprob"] < tcfg["logprob_threshold"]:
        flags.append("low_confidence")
    words = row["text"].split()
    if words and row.get("duration") and len(words) / row["duration"] > 8:
        flags.append("impossible_rate")
    return flags


def transcribe_video(video_id: str, cfg: dict, model, processor, device, dtype, overwrite: bool):
    tcfg = cfg["transcribe"]
    seg_manifest = Path(cfg["paths"]["manifests"]) / f"segments_{video_id}.jsonl"
    out_manifest = Path(cfg["paths"]["manifests"]) / f"transcripts_{video_id}.jsonl"

    if not seg_manifest.exists():
        return 0
    if out_manifest.exists() and not overwrite:
        return 0

    segments = list(read_jsonl(seg_manifest))
    if not segments:
        return 0

    rows: list[dict] = []
    bs = tcfg["batch_size"]

    for i in range(0, len(segments), bs):
        chunk = segments[i : i + bs]
        wavs = [load_audio(s["audio_path"], cfg["audio"]["sample_rate"]) for s in chunk]
        preds = transcribe_batch(wavs, model, processor, device, dtype, tcfg)

        for seg, pred in zip(chunk, preds):
            row = {**seg, **pred}
            row["asr_model"] = tcfg["model_id"]
            row["asr_language"] = tcfg["language"]
            row["asr_num_beams"] = tcfg["num_beams"]
            row["flags"] = flag_segment(row, tcfg)
            row["tier"] = "auto_low" if row["flags"] else "auto_high"
            rows.append(row)

    write_jsonl(rows, out_manifest)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--model-id",
        default=None,
        help="Hub repo id OR a local directory path (full fine-tune or LoRA adapter)",
    )
    ap.add_argument(
        "--model-path",
        default=None,
        help="alias for --model-id, for when the model is on local disk",
    )
    ap.add_argument(
        "--base-model-id",
        default=None,
        help="base model for a LoRA adapter; also accepts a local path",
    )
    ap.add_argument("--language", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.model_id or args.model_path:
        cfg["transcribe"]["model_id"] = args.model_id or args.model_path
    if args.base_model_id:
        cfg["transcribe"]["base_model_id"] = args.base_model_id
    if args.language:
        cfg["transcribe"]["language"] = args.language
    if args.batch_size:
        cfg["transcribe"]["batch_size"] = args.batch_size

    if "YOUR_HF_USERNAME" in cfg["transcribe"]["model_id"]:
        raise SystemExit(
            "Set transcribe.model_id in configs/pipeline.yaml (or pass --model-id) "
            "to your own Hub repo."
        )

    manifests = Path(cfg["paths"]["manifests"])
    video_ids = args.video or sorted(
        p.stem.replace("segments_", "") for p in manifests.glob("segments_*.jsonl")
    )
    if not video_ids:
        raise SystemExit("No segment manifests found. Run segment.py first.")

    model, processor, device, dtype = load_model(cfg)

    total = 0
    for vid in tqdm(video_ids, desc="transcribe"):
        try:
            total += transcribe_video(
                vid, cfg, model, processor, device, dtype, args.overwrite
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {vid} failed: {exc}")

    print(f"\n{total} segments transcribed.")

    # Quick health report -- if auto_low dominates, fix decoding before annotating.
    tiers: dict[str, int] = {}
    for p in manifests.glob("transcripts_*.jsonl"):
        for row in read_jsonl(p):
            tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
    print(f"tiers: {tiers}")


if __name__ == "__main__":
    main()
