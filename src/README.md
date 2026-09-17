# Pipeline

`download` → `segment` → `transcribe` → `build_dataset`

Every stage writes to disk and skips work already done, so you can interrupt and re-run
anything without losing progress.

## Setup

```powershell
pip install -r requirements.txt
winget install Gyan.FFmpeg     # ffmpeg must be on PATH
```

Then edit `configs/pipeline.yaml` — at minimum `transcribe.model_id` and
`dataset.hub_id`.

## Run it

```powershell
# one shot
python src/run_pipeline.py --urls urls.txt --model-id myname/whisper-large-v3-banglish-lora

# or stage by stage
python src/download.py      --urls urls.txt
python src/segment.py
python src/transcribe.py    --model-id myname/whisper-large-v3-banglish-lora
python src/build_dataset.py --exclude-flagged --push --hub-id myname/banglish-asr --private
```

`urls.txt` takes one entry per line — a video URL, a bare 11-char video ID, or a channel
`/videos` URL, which gets expanded automatically. `#` comments are ignored.

Start with two or three videos and inspect the output before scaling up.

## What each stage produces

```
data/raw/{video_id}.flac              16 kHz mono, immutable
data/meta/{video_id}.json             title, channel, license, sha256, fetch time
data/segments/{video_id}/*.flac       15–25 s chunks
data/manifests/segments_*.jsonl       segment boundaries + provenance
data/manifests/transcripts_*.jsonl    + text, avg_logprob, compression_ratio, tier
data/hf_dataset/                      DatasetDict, save_to_disk format
```

`data/raw/` is never modified after download — every later stage is a pure function of
it plus code, so a segmentation bug never costs you a re-download.

## The segmenter

This is the part worth understanding, in `segment.py`.

Silero VAD finds speech regions, so a cut can **only** land in a genuine silence — never
mid-word. Each inter-speech gap becomes a candidate boundary scored by pause length,
since long pauses track sentence and clause ends closely.

Boundaries are then chosen by **dynamic programming** over the whole file rather than
greedily. A greedy segmenter takes a locally attractive cut and gets forced into a bad
one two chunks later; DP accepts a mediocre cut now to buy two good ones after it. The
cost per segment:

| Term | Effect |
|---|---|
| `w_target` | Pulls duration toward `target_duration` (20 s) |
| `w_pause` | Rewards cutting at long pauses — **raise this for cleaner sentence boundaries** |
| `w_silence` | Penalises dead air trapped inside a segment |
| short penalty | Makes sub-15 s segments expensive except as a file tail |

Segments over `max_duration` are disallowed outright, so the 15–25 s contract is a hard
guarantee — anything that can't satisfy it is dropped rather than emitted out of range.

One case boundary selection can't fix: a speech region *itself* longer than 25 s, i.e.
someone talking 40 s without pausing. Those are pre-split at their lowest-energy interior
frame (breath points, plosive gaps), which is much less damaging than a fixed-offset cut.

Tuning: widen `min_silence_ms` (250 → 400) for fewer, more confident boundaries; raise
`w_pause` to 5–6 if you'd rather have clean sentence ends than uniform lengths.

## The transcriber

`model_id` accepts **either a Hub repo id or a local directory path**, and in both cases
figures out for itself whether it's a full fine-tune (`config.json`) or a PEFT/LoRA
adapter (`adapter_config.json`). For an adapter it resolves the base model, loads it, and
merges the LoRA weights in for inference speed. Your MediBeng + IndicVoices adapter works
as-is — just point `model_id` at it.

```powershell
# from the Hub
python src/transcribe.py --model-id myname/whisper-large-v3-banglish-lora

# from local disk (--model-path is an alias for --model-id)
python src/transcribe.py --model-path "D:/models/whisper-banglish-lora"
python src/transcribe.py --model-path ./checkpoints/checkpoint-4000

# local adapter + local base, fully offline
python src/transcribe.py --model-path D:/models/my-lora --base-model-id D:/models/whisper-large-v3
```

A local path short-circuits before any network call, so offline runs work. If the
adapter's `adapter_config.json` has no `base_model_name_or_path`, pass `--base-model-id`.
A path that doesn't exist fails immediately rather than being sent to the Hub as a repo
id.

Each segment records `avg_logprob` (mean per-token log probability, from
`compute_transition_scores`) and `compression_ratio` (gzip-based repetition detector,
>2.4 means the decoder is looping). These feed the confidence gate and set `tier` to
`auto_high` or `auto_low`.

Check the tier split printed at the end. **If `auto_low` dominates, fix decoding before
spending any annotation budget** — correcting bad pseudo-labels is slower than
transcribing from scratch.

## Dataset shape

Standard HF ASR format — `Audio` column plus text, so it loads directly into a Whisper
training script:

```python
from datasets import load_from_disk
ds = load_from_disk("data/hf_dataset")
ds["train"][0]["audio"]["array"]   # numpy, 16 kHz
ds["train"][0]["text"]
```

Alongside the audio and text, each row carries full provenance (URL, channel, license,
timestamps), segmentation quality (`speech_ratio`, surrounding pause lengths), pseudo-label
confidence (`avg_logprob`, `compression_ratio`, `tier`), and code-mixing measures (`cmi`,
`spf`, `matrix_language`, `latin_ratio`) so you can slice WER by code-mix intensity later.

**Splits are grouped by `video_id`.** Never split randomly over segments — that leaks the
same speaker and recording condition into train and test, and the numbers collapse on real
audio.

## Caveats

- `tier` is `auto_high`/`auto_low` from cheap filters only. Nothing here is human-verified —
  route it through correction before treating it as ground truth.
- Speaker-level split isn't implemented. Video-level grouping is a good proxy, but the same
  podcast guest appearing on two channels will still leak. Verify with speaker embeddings
  before publishing eval numbers.
- Before publishing audio, check `license` per row and read `rnd-docs/03-youtube-sourcing.md`.
