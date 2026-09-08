# Road-side video text reader

Point it at a video; it tells you what text is in it, where, and in which frames.

```bash
pip install -r requirements.txt
python detect_text_video.py --video road.mp4 --render annotated.mp4 --out results.json
```

```
stage 1  : 90 text regions detected
stage 2  : 86 readings kept (conf >= 0.3)
tracking : 3 distinct pieces of text
           #1   'EXIT 24'    frames 0-87  agreement 0.77
           #2   'PHARMACY'   frames 0-87  agreement 0.95
           #3   'MAIN ST'    frames 0-87  agreement 0.84
```

Uses **pretrained** detection and recognition weights, downloaded on first run.
Nothing to train.

> **Not the thesis system.** This is a standalone baseline built on off-the-shelf
> pretrained OCR. It shares no code with `vtspot/`, which trains from random
> initialisation — see [what this is and is not](#what-this-is-and-is-not).

---

## How it works

Each processed frame goes through two explicit stages:

```
frame ──> stage 1: DETECT ────> polygons ──> stage 2: RECOGNISE ──> text + confidence
          "where is text?"                   "what does it say?"
                                                       │
                                                       v
                                        stage 3: TRACK across frames
                                        one sign = one result, voted
```

**Stage 1 — detect.** A text detector returns a polygon per text region. Nothing
is read yet.

**Stage 2 — recognise.** Each detected region is read.

Two ways to run those, and they are not equal. `--mode pipeline` (the default)
hands the frame to the engine's end-to-end call, which does both stages and its
own merging and contrast retries in between. `--mode two_stage` crops each
detected polygon and reads it alone. Measured on a 1280×720 clip with four small
tilted signs:

| mode | read exactly | confidences |
|---|---|---|
| `pipeline` | **3 / 4** | 0.95–1.00 |
| `two_stage` | 2 / 4 | 0.52–1.00 |

`two_stage` also produced `[PHARMACY]` — its crop pulled in the sign's border,
which the recogniser read as brackets. So use `pipeline` for results, and
`two_stage` when you need to see *which* stage is failing.

**Stage 3 — track and vote.** Boxes that overlap between frames keep the same
id, so one signboard is reported once instead of once per frame. Across a
track's frames, the reading with the highest summed confidence wins — a word
misread in three frames and read correctly in twenty comes out correct, which no
single frame guarantees.

Knowing which stage failed still matters — a missing word is a detection
failure, a garbled word is a recognition failure, and the fixes are unrelated.
`--stage detect` draws stage-1 boxes and skips reading entirely, so one run
tells you which of the two you are looking at.

## Output

`--render` writes an annotated video: coloured polygon per text region, label
`#id TEXT confidence`, and a frame counter. Colour is keyed to track id, so an
id switch is visible at a glance.

`--out` writes JSON with both views of the same run:

```jsonc
{
  "tracks": [                         // one entry per distinct piece of text
    {"track_id": 2, "text": "PHARMACY", "text_confidence": 0.95,
     "start_frame": 0, "end_frame": 87, "num_frames": 30,
     "frames": {"0": [[x,y],[x,y],[x,y],[x,y]], "3": [...]}}
  ],
  "frames": [                         // and the raw per-frame detections
    {"frame": 0, "detections": [{"poly": [...], "text": "PHARMACY",
                                 "confidence": 0.91, "track_id": 2}]}
  ]
}
```

`text_confidence` is **agreement**, not the recogniser's score: the share of the
track's total confidence held by the winning reading. 1.00 means every frame
read it the same way; 0.40 means the frames disagreed and the transcription is
not to be trusted.

## When it reads nothing

Run the diagnostic first. It samples a few frames, tries each setting on all of
them, and reports what each found — which beats guessing, because the causes
need opposite fixes:

```bash
python diagnose.py --video road.mp4 --dump looked_at/
```

```
setting                regions  read  mean conf   sample
--------------------------------------------------------------------
default                     16    15       0.97   '24', 'MAIN STREET', '40'
small-text                  16    16       0.91   'EXIT 24', 'PHARMACY'   <-- best
tiny-text                   15    15       0.84   'EXIT 24', 'PHARMACY'
two_stage/small-text        16    16       0.75   '24]', 'EXIT ='
```

It prints the command to run with the winning setting. `--dump` writes one
annotated image per setting, so you can see what it found rather than trust the
count.

Reading the rows yourself:

| Symptom | Cause | Fix |
|---|---|---|
| regions high, read low | detection fine, text too degraded to resolve | `--mag 3`, or a better source |
| regions low everywhere | text below `min_size`, or fainter than `low_text` | `--preset tiny-text` |
| phrases split (`SPEED`, `40`) | detector did not link the words | `--merge-words` |
| nothing at any setting | the detail is not in the video | crop to the region, or re-shoot |

## Presets

| Preset | For |
|---|---|
| `default` | text that fills a decent part of the frame |
| `small-text` | distant road signage — **start here for dashcam video** |
| `small-text-merged` | same, plus glue split phrases back together |
| `tiny-text` | when `small-text` still misses things; slow |
| `fast` | long clips, big text only |

## Options

| Flag | Default | What it does |
|---|---|---|
| `--video` | — | input video (required) |
| `--out` | — | write results JSON |
| `--render` | — | write annotated video |
| `--frames-dir` | — | also save annotated frames as JPGs |
| `--engine` | `easyocr` | `easyocr` or `paddleocr` |
| `--langs` | `en` | comma-separated, e.g. `en,hi` |
| `--cpu` | off | force CPU |
| `--every` | `1` | run OCR on every Nth frame |
| `--max-frames` | `0` | stop after N frames (0 = all) |
| `--resize` | `0` | resize long side before OCR (0 = native) |
| `--preset` | `default` | see the table above |
| `--mode` | `pipeline` | `two_stage` splits detection from recognition |
| `--mag` | preset | upscale before detection — the knob for small text |
| `--merge-words` | off | glue split phrases together |
| `--low-text` | preset | lower = fainter strokes count as text |
| `--link` | preset | lower = neighbouring words merge |
| `--min-conf` | `0.30` | drop readings below this |
| `--min-chars` | `2` | drop readings shorter than this |
| `--no-track` | off | per-frame results only, no ids, no voting |
| `--iou` | `0.30` | tracker IoU threshold |
| `--max-age` | `12` | frames a track survives with no detection |
| `--min-hits` | `2` | frames a track needs to be reported |
| `--stage` | `both` | `detect` = stage 1 only |

## Speed

Recognition dominates, and it runs once per detected box — so cost scales with
how much text is on screen, not just resolution.

Measured on a 960×540 clip with 3 signs per frame, CPU only: **~2 fps**. A CUDA
GPU is roughly 10–20× that.

Three knobs, in the order worth trying:

- **`--every 3`** — process one frame in three. The output video still has every
  frame (skipped frames reuse the previous result), and at 25–30 fps the reuse is
  invisible. Roughly 3× throughput, and tracking still works because `--max-age`
  covers the gap.
- **`--resize 960`** — smaller frames, faster detection. Costs you the smallest
  text, so lower it only until distant signs start dropping out.
- **drop `--cpu`** — the single biggest win if a CUDA GPU is available.

## Troubleshooting

**Nothing detected.** Run `diagnose.py` — see [above](#when-it-reads-nothing).

**One sign gets several ids.** Detection is dropping frames. Raise `--max-age`
(the track then survives longer without a detection), or lower `--iou` to `0.2`
if the box jumps around between frames.

**Garbage readings on road texture.** Raise `--min-conf` to `0.5` and
`--min-chars` to `3`. Real signage clears both easily; noise usually does not.

**Phrases split into pieces.** `--merge-words`. It over-merges, though:
grouping is geometric, so two different signs that drift close together are
joined as readily as two halves of one phrase. Check the tracks table.

**Slow.** See [Speed](#speed) above.

**First run hangs for a minute.** It is downloading weights (~100 MB), once.
They cache in `~/.EasyOCR`.

## What this is and is not

| | This tool | `vtspot/` (the thesis) |
|---|---|---|
| Weights | pretrained, downloaded | trained from random init |
| Training | none | 3-stage curriculum |
| Detection + recognition | two separate pretrained models | one shared feature map |
| Tracking | IoU geometry only | learnt association embedding + LST matcher |
| Runs today | yes | after training |

Presenting it as a **baseline** is honest and useful — it is what the thesis
system is measured against. Presenting it as the thesis system is not: the code
here is a wrapper around someone else's pretrained models, and the point of
`vtspot/` is that it uses none.
