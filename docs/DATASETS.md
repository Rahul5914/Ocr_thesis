# Datasets

Two things drive the choice here that a normal (pretrained) project can ignore:

1. **Stage 1 replaces ImageNet.** Without a pretrained backbone, low-level
   features come from synthetic stills and large static text corpora. Under-feed
   this stage and nothing downstream recovers. The brief listed only MLT-19 for
   static data; that is nowhere near enough.
2. **Annotated video is scarce.** Every public video-text benchmark combined is
   on the order of tens of thousands of annotated frames. It is a *fine-tuning*
   resource, not a training set. This is why the synthetic video generator
   exists.

---

## Start here — no downloads needed

`vtspot/data/synth_static.py` and `vtspot/data/synth_video.py` generate training
data procedurally: ~17 still images/s and ~34 video frames/s on one CPU core,
with exact polygon and track-id ground truth, curved text, motion blur and
occlusion. Fonts are discovered from the system.

```bash
python tools/train.py --config configs/stage1_static.yaml   # nothing to download
python tools/train.py --config configs/stage2_video.yaml --init checkpoints/stage1_static/last.pt
```

Quality improves noticeably if you point `background_dir` at a folder of real
photographs — any unlabelled image collection will do, since only the pasted text
is labelled.

---

## Stage 1 — static text (learn strokes, glyphs, detection)

| Dataset | Size | Annotation | Why | Access |
|---|---|---|---|---|
| **SynthText** | 800k images, 8M words | word + char boxes | The standard synthetic pretraining corpus. Closest public analogue to what stage 1 generates | [VGG](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) — free |
| **MJSynth / Synth90k** | 9M word crops | transcription only | Recognition head only (no detection boxes). Very effective for CTC | [VGG](https://www.robots.ox.ac.uk/~vgg/data/text/) — free |
| **ICDAR2019-MLT** | 10k train / 10k test | word quads, 10 languages | Multilingual baseline; the brief's one static pick | [RRC](https://rrc.cvc.uab.es/?ch=15) — registration |
| **COCO-Text** | 63k images, 145k instances | word boxes | Real, uncurated, hard | [RRC](https://rrc.cvc.uab.es/?ch=5) — free |
| **TextOCR** | 28k images, 900k words | polygons | Dense real annotation, very high quality | [textvqa.org](https://textvqa.org/textocr/) — free |
| **Total-Text** | 1,555 images | polygons | **Curved text** — needed to exercise PolyAlign | [GitHub](https://github.com/cs-chan/Total-Text-Dataset) — free |
| **CTW1500** | 1,500 images | 14-point polygons | **Curved, line-level**; matches this repo's control-point convention exactly | [GitHub](https://github.com/Yuliang-Liu/Curve-Text-Detector) — free |
| **ICDAR2015 (static)** | 1,500 images | word quads | Incidental scene text; the classic detection benchmark | [RRC](https://rrc.cvc.uab.es/?ch=4) — registration |

**Minimum viable stage 1:** the built-in generator + SynthText.
**Recommended:** add TextOCR (real, dense) and CTW1500 (curved).

---

## Stage 2/3 — video text

| Dataset | Size | Annotation | Distinguishing feature | Access |
|---|---|---|---|---|
| **ICDAR2015 Video** | 49 videos (25 train / 24 test) | word quads + track ids | The standard benchmark. Report on this | [RRC ch.3](https://rrc.cvc.uab.es/?ch=3) — registration |
| **ICDAR2013 Video** | 28 videos | word quads + track ids | Easier, more stable camera. Good first target | [RRC](https://rrc.cvc.uab.es/?ch=3) — registration |
| **DSText V2** | 140 videos, 62.1k frames | word quads + track ids | **~24 instances/frame** (vs 5.55 for ICDAR15), heavily small text, 12 scenarios | [RRC ch.22](https://rrc.cvc.uab.es/?ch=22) / [Zenodo](https://zenodo.org/records/10010840) |
| **BOVText** | 2,021 videos, 1.75M frames | **textline** quads + track ids | Largest by far; bilingual EN/ZH; 32 open-world scenarios. Note line-level ≠ word-level | [GitHub](https://github.com/weijiawu/BOVText-Benchmark) |
| **RoadText-1K** | 1,000 videos (10 s @ 30fps) | axis-aligned boxes + track ids | Largest *real* annotated video corpus; driving scenes. **Missing from the brief** | [CVIT](https://cvit.iiit.ac.in/research/projects/cvit-projects/roadtext-1k) |
| **ArTVideo** | 60 videos, 13k frames, 170k instances | 14-pt polygons + masks | **>30% curved text** — the only video benchmark that tests curved tracking | [GoMatching++](https://github.com/Hxyz-123/GoMatching) |

### Practical notes

- **Report on ICDAR2015-video.** It is what the literature compares on. Add
  DSText V2 for density and ArTVideo for curvature.
- **BOVText is line-level.** Word-level and line-level annotations are not
  interchangeable; mixing them without care produces a detector that cannot
  decide what an instance is. Train on one convention, or add a
  line/word indicator.
- **RoadText-1K is axis-aligned only.** Excellent for scale and tracking,
  teaches nothing about orientation. Do not train on it alone.
- **Licensing:** RRC datasets need a (free) account and are research-use.
  RoadText-1K derives from BDD100K — check the BDD100K terms. Verify each
  licence yourself before publishing; several are non-commercial.

---

## Preparing a dataset

Every benchmark ships its own format. `tools/prepare_dataset.py` converts to the
unified schema in `vtspot/data/schema.py` and **validates the result**:

```bash
python tools/prepare_dataset.py --dataset icdar15_video \
    --videos  raw/icdar15/videos \
    --annotations raw/icdar15/gt \
    --out data/prepared/icdar15_video_train
```

The validation output is the point. It prints instances parsed, tracks found,
mean instances per frame, and flags polygons outside the frame — then compares
your density against the published figure:

```
mean instances/frame: 5.3 (published ~5.5) -> plausible
```

If that line says `SUSPICIOUS`, the parser matched the wrong format variant.
Every number you produce afterwards would be meaningless, and nothing else in
the pipeline can detect it. **Do not skip this check.** The usual causes are
0-vs-1-based frame numbering (shifts every annotation by one frame, quietly
halving IoU on fast-moving text) and a coordinate order mismatch.

Supported: `icdar13_video`, `icdar15_video`, `dstext`, `bovtext`, `artvideo`,
`roadtext1k`. The JSON converter auto-detects key-name and frame-offset variants;
add a new one in `vtspot/data/converters/`.

Already-extracted frames instead of video files:

```bash
python tools/prepare_dataset.py --dataset dstext \
    --frames raw/dstext/frames --annotations raw/dstext/gt \
    --out data/prepared/dstext_train
```

---

## Suggested recipe

| Stage | Data | Epochs | LR | Notes |
|---|---|---|---|---|
| 1 | Built-in synth stills (+ SynthText, TextOCR, CTW1500) | 20 | 1e-3 | Longest stage. This is your ImageNet |
| 2 | Built-in synth video (+ real video, unlabelled backgrounds) | 15 | 5e-4 | Tracking loss switches on |
| 3 | ICDAR15-video train (+ DSText, ArTVideo) | 30 | 1e-4 | Small corpus, overfits fast |

Evaluate on the held-out test splits with `tools/evaluate.py`, which reports
tracking-mode and spotting-mode metrics side by side.
