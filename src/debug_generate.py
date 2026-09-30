"""Minimal, isolated repro for the Whisper generate() CUDA assert.

Deliberately does NOT use the pipeline's backend classes. It loads the model, takes
ONE segment, and calls generate() with a single named set of options -- so whatever it
reports is about transformers/torch/the checkpoint, not about transcribe.py.

Each variant must run in a FRESH PROCESS: a CUDA device-side assert poisons the context,
so every call after the first failure fails regardless of cause. `--all` shells out per
variant to guarantee that.

Usage
-----
    python src/debug_generate.py --model-path <adapter> --all

    # or one at a time
    python src/debug_generate.py --model-path <adapter> --variant cpu_minimal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_audio, load_config, read_jsonl

# name -> generate() kwargs on top of the bare minimum
VARIANTS = {
    # CPU first: CPU raises a real Python IndexError naming the bad index, where CUDA
    # only gives an async assert. If CPU passes and CUDA fails, it is a kernel problem.
    "cpu_minimal":      dict(device="cpu",  beams=1, lang=False, scores=False),
    "cpu_lang":         dict(device="cpu",  beams=1, lang=True,  scores=False),
    "cpu_beam":         dict(device="cpu",  beams=5, lang=True,  scores=False),
    "cuda_minimal":     dict(device="cuda", beams=1, lang=False, scores=False),
    "cuda_lang":        dict(device="cuda", beams=1, lang=True,  scores=False),
    "cuda_beam":        dict(device="cuda", beams=5, lang=True,  scores=False),
    "cuda_beam_scores": dict(device="cuda", beams=5, lang=True,  scores=True),
}


def first_segment(cfg: dict) -> str:
    manifests = Path(cfg["paths"]["manifests"])
    for p in sorted(manifests.glob("segments_*.jsonl")):
        for row in read_jsonl(p):
            return row["audio_path"]
    raise SystemExit("No segments found. Run segment.py first.")


def run_variant(model_path: str, name: str, cfg: dict) -> None:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    v = VARIANTS[name]
    device = v["device"]
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    adapter_cfg = Path(model_path).expanduser() / "adapter_config.json"
    if adapter_cfg.exists():
        import json

        from peft import PeftModel

        base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, model_path, torch_dtype=dtype)
        model = model.merge_and_unload()
        processor = WhisperProcessor.from_pretrained(base)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
        processor = WhisperProcessor.from_pretrained(model_path)

    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.to(device).eval()

    wav = load_audio(first_segment(cfg), 16000)
    feats = processor(wav, sampling_rate=16000, return_tensors="pt")
    x = feats.input_features.to(device, dtype=dtype)

    kwargs = {"num_beams": v["beams"]}
    if v["lang"]:
        kwargs["language"] = "bn"
        kwargs["task"] = "transcribe"
    if v["scores"]:
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True

    print(f"  generate(**{kwargs})")
    with torch.no_grad():
        out = model.generate(x, **kwargs)
    seq = out.sequences if v["scores"] else out

    # Force the CPU sync here so an async CUDA assert surfaces inside this variant.
    ids = seq.cpu().tolist()[0]
    text = processor.batch_decode(seq, skip_special_tokens=True)[0]

    print(f"  first 12 token ids: {ids[:12]}")
    print(f"  min id={min(ids)}  max id={max(ids)}  vocab={model.config.vocab_size}")
    print(f"  TEXT: {text[:300]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--all", action="store_true", help="run every variant in its own process")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.all:
        results = {}
        for name in VARIANTS:
            print(f"\n{'=' * 66}\n  {name}\n{'=' * 66}")
            proc = subprocess.run(
                [sys.executable, __file__, "--model-path", args.model_path,
                 "--variant", name] + (["--config", args.config] if args.config else []),
                capture_output=True, text=True,
            )
            # Keep only the useful lines; the assert spam is one failure repeated.
            for line in (proc.stdout + proc.stderr).splitlines():
                if "ScatterGatherKernel.cu" not in line:
                    print(line)
            results[name] = "PASS" if proc.returncode == 0 else "FAIL"

        print(f"\n{'=' * 66}\n  SUMMARY\n{'=' * 66}")
        for name, status in results.items():
            print(f"  {status:5}  {name}")
        return

    if not args.variant:
        raise SystemExit("Pass --variant <name> or --all")
    run_variant(args.model_path, args.variant, cfg)
    print("  OK")


if __name__ == "__main__":
    main()
