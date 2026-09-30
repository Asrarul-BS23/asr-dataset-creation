"""Transcribe segments with your own Whisper model.

Two backends, chosen automatically from what is actually in the model directory:

  * **CTranslate2 / faster-whisper** -- `model.bin` + `vocabulary.json`.
    Much faster for bulk pseudo-labelling, and it reports `avg_logprob` and
    `no_speech_prob` natively rather than having them reconstructed from logits.
  * **HF transformers** -- `config.json` + `model.safetensors`, or a PEFT/LoRA adapter
    (`adapter_config.json`), whose base model is resolved and merged in automatically.

Either backend accepts a Hub repo id or a local directory path.

Per segment it records the decoder's own quality signals so the confidence gate in
rnd-docs/05-pseudo-label-pipeline.md has something to work with. Do not skip these --
they are what lets you route segments to annotators by expected error rather than at
random.

Usage
-----
    python src/transcribe.py --model-path /home/me/outputs/faster-whisper-bangla-lora
    python src/transcribe.py --model-id myname/whisper-large-v3-banglish-lora
    python src/transcribe.py --backend hf --model-path ./checkpoints/checkpoint-4000
"""

from __future__ import annotations

import argparse
import platform
import traceback
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import ensure_dirs, load_audio, load_config, read_jsonl, write_jsonl
from normalize import normalize_text

SR = 16000


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


def compression_ratio(text: str) -> float:
    """Whisper's repetition detector. >2.4 means the decoder is looping."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


# --------------------------------------------------------------------- model resolution


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
                f"{local_dir} exists but holds neither config.json (full model / CT2) "
                f"nor adapter_config.json (LoRA adapter)."
            )
        return False, None, True

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


def detect_backend(model_id: str) -> str:
    """'ct2' or 'hf', from what the directory actually contains.

    A CTranslate2 export is `model.bin` plus a `vocabulary.*` file and a config.json
    that is NOT an HF model config. HF checkpoints carry model.safetensors (or the
    legacy pytorch_model.bin) instead.
    """
    d = Path(model_id).expanduser()
    if not d.is_dir():
        return "hf"  # Hub repo ids are assumed HF; pass --backend ct2 to override
    if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
        return "hf"
    if (d / "adapter_config.json").exists():
        return "hf"
    if (d / "model.bin").exists() and any(d.glob("vocabulary.*")):
        return "ct2"
    return "hf"


# ------------------------------------------------------------------------- CT2 backend


class CT2Backend:
    """faster-whisper / CTranslate2. The fast path for bulk pseudo-labelling."""

    name = "ct2"

    def __init__(self, cfg: dict):
        tcfg = cfg["transcribe"]
        self.tcfg = tcfg
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SystemExit(
                "This model is in CTranslate2 format but faster-whisper is not installed.\n"
                "  pip install faster-whisper\n"
                f"If the CUDA build will not install on {platform.machine()}, point\n"
                "--model-path at the pre-conversion HF checkpoint instead, or pass\n"
                "--backend hf."
            ) from None

        # CT2 names its precisions differently from torch.
        compute_type = {
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
        }.get(tcfg["dtype"], "float16")
        device = "cuda" if tcfg["device"].startswith("cuda") else "cpu"
        if device == "cpu" and compute_type in ("float16", "bfloat16"):
            compute_type = "int8"

        print(f"Loading CTranslate2 model ({device}, {compute_type}): {tcfg['model_id']}")
        try:
            self.model = WhisperModel(
                str(Path(tcfg["model_id"]).expanduser()),
                device=device,
                compute_type=compute_type,
            )
        except (ValueError, RuntimeError) as exc:
            msg = str(exc)

            # PyPI ships CPU-only CTranslate2 wheels for aarch64. There is no CUDA build
            # to pip install, so on DGX Spark this is a dead end unless you build CT2
            # from source -- and the HF backend is the far cheaper answer.
            if "not compiled with CUDA" in msg:
                raise SystemExit(
                    "CTranslate2 has no CUDA support in this install "
                    f"({platform.machine()}: PyPI ships CPU-only wheels for ARM).\n"
                    "\nPick one:\n"
                    "  1. RECOMMENDED -- use the pre-conversion HF checkpoint or LoRA\n"
                    "     adapter from the same training run, which runs on the GPU:\n"
                    "       ls <your-training-outputs-dir>\n"
                    "       python src/transcribe.py --model-path <hf-checkpoint>\n"
                    "  2. Run this CT2 model on CPU (slow -- large-v3 at roughly\n"
                    "     realtime, so hours per hour of audio):\n"
                    "       set transcribe.device: cpu in configs/pipeline.yaml\n"
                    "  3. Build CTranslate2 from source with CUDA for aarch64\n"
                    "     (hours of work; only worth it for sustained bulk inference)."
                ) from None

            if compute_type == "bfloat16":
                print(f"  bfloat16 unsupported here ({exc}); falling back to float16.")
                self.model = WhisperModel(
                    str(Path(tcfg["model_id"]).expanduser()),
                    device=device,
                    compute_type="float16",
                )
            else:
                raise

    def transcribe(self, wavs: list[np.ndarray]) -> list[dict]:
        t = self.tcfg
        results = []
        for wav in wavs:
            segments, _info = self.model.transcribe(
                wav,
                language=t["language"],
                task=t["task"],
                beam_size=t["num_beams"],
                condition_on_previous_text=t["condition_on_prev_text"],
                compression_ratio_threshold=t["compression_ratio_threshold"],
                log_prob_threshold=t["logprob_threshold"],
                no_speech_threshold=0.6,
                initial_prompt=t.get("initial_prompt"),
                vad_filter=False,  # already segmented upstream by segment.py
            )
            segs = list(segments)  # generator: decoding happens here
            text = " ".join(s.text.strip() for s in segs).strip()

            if segs:
                # Duration-weight the per-chunk scores rather than taking a flat mean,
                # so a 0.5 s tail chunk cannot dominate a 20 s segment's confidence.
                weights = np.array([max(s.end - s.start, 1e-3) for s in segs])
                weights /= weights.sum()
                avg_lp = float(np.sum(weights * np.array([s.avg_logprob for s in segs])))
                no_speech = float(max(s.no_speech_prob for s in segs))
            else:
                avg_lp, no_speech = float("nan"), 1.0

            results.append(
                {
                    "text": text,
                    "text_normalized": normalize_text(text),
                    "avg_logprob": None if np.isnan(avg_lp) else round(avg_lp, 4),
                    "no_speech_prob": round(no_speech, 4),
                    "compression_ratio": round(compression_ratio(text), 3),
                }
            )
        return results


# -------------------------------------------------------------------------- HF backend


class HFBackend:
    """transformers. Needed for PEFT/LoRA adapters and unconverted checkpoints."""

    name = "hf"

    def __init__(self, cfg: dict):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        tcfg = cfg["transcribe"]
        self.tcfg = tcfg
        model_id = tcfg["model_id"]
        device = tcfg["device"]

        if device.startswith("cuda"):
            # Fail loudly. A silent CPU fallback on large-v3 is ~50x slower and easy to
            # miss until you have burned a night on it.
            if not torch.cuda.is_available():
                raise SystemExit(
                    "device: cuda requested but torch.cuda.is_available() is False.\n"
                    f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
                    f"machine {platform.machine()}\n"
                    "Install a CUDA build (cuDNN ships inside the wheel):\n"
                    f"  pip install torch torchaudio --index-url {_wheel_index_url()}\n"
                    "Or set transcribe.device: cpu."
                )
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # fixed 30 s input -> stable kernels
            props = torch.cuda.get_device_properties(0)
            print(
                f"GPU: {props.name}  sm_{props.major}{props.minor}  "
                f"{props.total_memory / 1e9:.0f} GB\n"
                f"     torch {torch.__version__} / CUDA {torch.version.cuda} / "
                f"cuDNN {torch.backends.cudnn.version()} / {platform.machine()}"
            )

        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtypes[tcfg["dtype"]] if device != "cpu" else torch.float32
        if device.startswith("cuda") and dtype is torch.float16:
            if torch.cuda.get_device_properties(0).major >= 8:
                print("note: bfloat16 is the better choice on this GPU.")

        is_peft, adapter_base, is_local = resolve_model_source(model_id)
        base_id = tcfg.get("base_model_id") or adapter_base or model_id
        if is_peft and not base_id:
            raise SystemExit(
                f"{model_id} is a LoRA adapter but adapter_config.json has no "
                "base_model_name_or_path. Pass --base-model-id."
            )

        source = "local" if is_local else "hub"
        if is_peft:
            print(f"PEFT adapter ({source}).\n  base    : {base_id}\n  adapter : {model_id}")
            from peft import PeftModel

            model = WhisperForConditionalGeneration.from_pretrained(
                base_id, torch_dtype=dtype, attn_implementation="sdpa"
            )
            model = PeftModel.from_pretrained(model, model_id, torch_dtype=dtype)
            model = model.merge_and_unload()  # fold LoRA in for inference speed
        else:
            print(f"Loading HF model ({source}): {model_id}")
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, attn_implementation="sdpa"
            )

        try:
            processor = WhisperProcessor.from_pretrained(model_id)
        except (OSError, ValueError):
            processor = WhisperProcessor.from_pretrained(base_id)

        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.dtype = dtype

        # Fine-tuned Whisper checkpoints routinely carry a stale `forced_decoder_ids`
        # in their generation_config, e.g. [[1, None], [2, 50360]] -- note the None in
        # the language slot. transformers warns that it will ignore this in favour of
        # the explicit language=/task= we pass, but the None still reaches index
        # arithmetic and surfaces as a CUDA device-side assert in a scatter kernel,
        # with no usable Python traceback. We always pass language and task
        # explicitly, so this field is pure liability -- clear it.
        for holder, label in ((model.generation_config, "generation_config"),
                              (model.config, "config")):
            stale = getattr(holder, "forced_decoder_ids", None)
            if stale is not None:
                print(f"clearing stale {label}.forced_decoder_ids: {stale}")
                holder.forced_decoder_ids = None

        # A tokenizer/model vocab mismatch is a common cause of a CUDA device-side
        # assert in a scatter kernel: suppress_tokens or a forced language token lands
        # outside the embedding, and the failure surfaces asynchronously with a
        # useless stack trace. Check it here, where the error can still be readable.
        n_embed = model.get_input_embeddings().weight.shape[0]
        n_tok = len(processor.tokenizer)
        gcfg = model.generation_config
        worst = max(
            list(getattr(gcfg, "suppress_tokens", None) or [0])
            + list(getattr(gcfg, "begin_suppress_tokens", None) or [0])
        )
        print(f"vocab: embeddings={n_embed}  tokenizer={n_tok}  max_suppress_id={worst}")
        if worst >= n_embed:
            raise SystemExit(
                f"generation_config suppresses token id {worst} but the model has only "
                f"{n_embed} embeddings. The processor and the checkpoint disagree -- "
                "pass --base-model-id to force a matching base."
            )
        if n_tok > n_embed:
            print(
                f"  WARNING: tokenizer has {n_tok} tokens vs {n_embed} embeddings; "
                "the processor may not match this checkpoint."
            )
        print(f"Model on {device} ({dtype}).")

    def transcribe(self, wavs: list[np.ndarray]) -> list[dict]:
        torch = self.torch
        t = self.tcfg
        with torch.no_grad():
            features = self.processor(
                wavs, sampling_rate=SR, return_tensors="pt", return_attention_mask=True
            )
            input_features = features.input_features.to(self.device, dtype=self.dtype)
            # We asked the processor for an attention mask, so actually use it. Without
            # it transformers warns on batched input and falls back to guessing from the
            # pad token, which for Whisper is the same as eos.
            attention_mask = getattr(features, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # No max_new_tokens: Whisper's decoder has only 448 positions, and
            # max_new_tokens is added ON TOP of the forced decoder tokens. Setting it
            # near the limit can overrun the position embedding, which surfaces as a
            # CUDA device-side assert in a scatter kernel rather than a clear error.
            # The model's own generation_config already caps this correctly.
            gen_kwargs = {
                "num_beams": t["num_beams"],
                "language": t["language"],
                "task": t["task"],
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if t.get("no_repeat_ngram_size"):
                gen_kwargs["no_repeat_ngram_size"] = t["no_repeat_ngram_size"]
            if t.get("initial_prompt"):
                gen_kwargs["prompt_ids"] = self.processor.get_prompt_ids(
                    t["initial_prompt"], return_tensors="pt"
                ).to(self.device)

            out = self.model.generate(input_features, **gen_kwargs)

            try:
                transition = self.model.compute_transition_scores(
                    out.sequences,
                    out.scores,
                    getattr(out, "beam_indices", None),
                    normalize_logits=True,
                ).float().cpu().numpy()
                avg_logprobs = []
                for row in transition:
                    valid = row[np.isfinite(row)]
                    avg_logprobs.append(float(valid.mean()) if valid.size else float("nan"))
            except Exception:  # noqa: BLE001 - scoring must never kill a run
                avg_logprobs = [float("nan")] * out.sequences.shape[0]

            texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)

        results = []
        for text, lp in zip(texts, avg_logprobs):
            text = text.strip()
            results.append(
                {
                    "text": text,
                    "text_normalized": normalize_text(text),
                    "avg_logprob": None if np.isnan(lp) else round(lp, 4),
                    "no_speech_prob": None,  # not exposed by generate()
                    "compression_ratio": round(compression_ratio(text), 3),
                }
            )
        return results


def load_backend(cfg: dict, forced: str | None = None):
    backend = forced or detect_backend(cfg["transcribe"]["model_id"])
    return CT2Backend(cfg) if backend == "ct2" else HFBackend(cfg)


# ------------------------------------------------------------------------------- driver


def flag_segment(row: dict, tcfg: dict) -> list[str]:
    """Cheap auto-filters. See rnd-docs/05 section 4."""
    flags = []
    if not row["text"]:
        flags.append("empty")
    if row["compression_ratio"] > tcfg["compression_ratio_threshold"]:
        flags.append("repetition_loop")
    if row["avg_logprob"] is not None and row["avg_logprob"] < tcfg["logprob_threshold"]:
        flags.append("low_confidence")
    if row.get("no_speech_prob") is not None and row["no_speech_prob"] > 0.6 and row["text"]:
        flags.append("likely_hallucination")
    words = row["text"].split()
    if words and row.get("duration") and len(words) / row["duration"] > 8:
        flags.append("impossible_rate")
    return flags


def transcribe_video(video_id: str, cfg: dict, backend, overwrite: bool) -> int:
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
    bs = tcfg["batch_size"] if backend.name == "hf" else 1

    for i in range(0, len(segments), bs):
        chunk = segments[i : i + bs]
        wavs = [load_audio(s["audio_path"], cfg["audio"]["sample_rate"]) for s in chunk]
        preds = backend.transcribe(wavs)

        for seg, pred in zip(chunk, preds):
            row = {**seg, **pred}
            row["asr_model"] = tcfg["model_id"]
            row["asr_backend"] = backend.name
            row["asr_language"] = tcfg["language"]
            row["asr_num_beams"] = tcfg["num_beams"]
            row["flags"] = flag_segment(row, tcfg)
            row["tier"] = "auto_low" if row["flags"] else "auto_high"
            rows.append(row)

    write_jsonl(rows, out_manifest)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--model-id", default=None, help="Hub repo id OR local directory path"
    )
    ap.add_argument("--model-path", default=None, help="alias for --model-id")
    ap.add_argument("--base-model-id", default=None, help="base for a LoRA adapter")
    ap.add_argument(
        "--backend",
        choices=["auto", "hf", "ct2"],
        default="auto",
        help="auto-detected from the model directory; override if detection is wrong",
    )
    ap.add_argument("--language", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="re-raise on the first failure instead of continuing to the next video",
    )
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
            "Set transcribe.model_id in configs/pipeline.yaml (or pass --model-path)."
        )

    manifests = Path(cfg["paths"]["manifests"])
    video_ids = args.video or sorted(
        p.stem.replace("segments_", "") for p in manifests.glob("segments_*.jsonl")
    )
    if not video_ids:
        raise SystemExit("No segment manifests found. Run segment.py first.")

    backend = load_backend(cfg, None if args.backend == "auto" else args.backend)

    total = 0
    for vid in tqdm(video_ids, desc="transcribe"):
        try:
            total += transcribe_video(vid, cfg, backend, args.overwrite)
        except Exception as exc:  # noqa: BLE001
            # Print the stack, not just the message. A bare message discards exactly
            # the information needed to debug a failure inside transformers.
            print(f"  ! {vid} failed: {exc}")
            traceback.print_exc()
            if args.fail_fast:
                raise

    print(f"\n{total} segments transcribed.")

    tiers: dict[str, int] = {}
    for p in manifests.glob("transcripts_*.jsonl"):
        for row in read_jsonl(p):
            tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
    print(f"tiers: {tiers}")


if __name__ == "__main__":
    main()
