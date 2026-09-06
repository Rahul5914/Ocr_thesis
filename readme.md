# Video Text Spotting from scratch

A complete video text spotting system — detection, tracking and recognition —
trained **entirely from random initialisation**. No ImageNet backbone, no
pretrained language model, no pretrained detector.

Built to a research brief on video text spotting; `docs/RESEARCH_REVIEW.md`
verifies that brief against the primary sources and documents the corrections
that this implementation makes.

---

## What it does

Video in, text trajectories out: for every piece of text in the video, where it
is in each frame, what it says, and a stable identity across the whole sequence
even through occlusion.

```bash
python tools/predict_video.py --checkpoint checkpoints/stage3/last.pt \
                              --video clip.mp4 --out results.json --render annotated.mp4
```

```json
{"track_id": 7, "text": "PHARMACY", "text_confidence": 0.91,
 "start_frame": 12, "end_frame": 84, "frames": {"12": [[x,y], ...], ...}}
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                                   # 100 tests, ~15 s
python tools/train.py --config configs/smoke.yaml --device cpu   # ~1 min, verifies the pipeline
python tools/train.py --config configs/stage1_static.yaml        # real training
```

**No dataset download is needed to start.** Training data is generated
procedurally — ~17 still images/s and ~34 video frames/s on one CPU core, with
exact polygon and track-id ground truth, curved text, motion blur and occlusion.

## Architecture

```
frames ─> backbone (GroupNorm ResNet) ─> FPN ─┬─> DB head      ─> polygons
          12.2M params total                  │
                                              ├─> PolyAlign ─┬─> CTC head   ─> transcription
                                              │              └─> embed head ─> association
                                              └───────────────────────────────────────┐
                                                                                      v
                                            Long-Short-Term matcher + trajectory-level voting
```

One shared feature map feeds all three heads — the recognition and association
gradients flow back into the same features the detector uses.

| Component | Choice | Why (from-scratch specific) |
|---|---|---|
| Backbone | ResNet-style, **GroupNorm** | Batch-independent; video clip training means 2–4 clips/step. `norm: none` switches to a correct 4-rule Fixup |
| Detection | **Differentiable Binarization** | Supervises *every pixel every step*. DETR-style Hungarian matching supervises one query per instance and needs a pretrained backbone to converge |
| RoI | **PolyAlign** | Follows the text polygon, so curved text works. Pure `grid_sample` — nothing to compile |
| Recognition | **CTC** + training-only attention branch (GTC) | CTC speed at inference; attention shapes the shared features during training |
| Tracking | Contrastive embedding + LST matcher | Motion gate handles repeated text that appearance alone cannot separate |
| Transcription | Trajectory-level voting | Recovers words no single frame reads correctly — zero extra parameters |

## Key corrections to the source research

Full detail in [`docs/RESEARCH_REVIEW.md`](docs/RESEARCH_REVIEW.md). The four
that would have broken the build:

1. **DETR-style set prediction does not converge from scratch.** Every paper
   recommending it starts from ImageNet weights. Replaced with a dense DB head.
2. **A rotated-RoI head cannot read curved text.** Measured: 0.49 text signal vs
   **0.86** for PolyAlign on a curved band.
3. **A fixed unclip ratio under-expands long thin text** — which is what text
   mostly is. Adaptive unclipping: mean IoU 0.62 → **0.89**, and 7/7 instances
   clear the 0.5 matching threshold instead of 4/7.
4. **Spotting-mode metrics were missing.** On a case with perfect boxes and one
   word misread: tracking MOTA 1.00, **spotting MOTA 0.00**. `tools/evaluate.py`
   always prints both.

Plus: the He/Xavier formulas conflate variance with standard deviation; Fixup is
described as 1 of its 4 required rules; and the charset — which decides the size
of the output layer — is never specified.

## Layout

```
vtspot/
  models/      backbone, FPN, DB head, PolyAlign, CTC+attention, embedding, init
  losses/      DB, CTC, attention guidance, contrastive, multi-task weighting
  data/        synthetic stills, synthetic video, targets, transforms, converters
  tracking/    LST matcher, online tracker, CTC decode, trajectory aggregation
  eval/        MOTA / MOTP / IDF1 / HOTA in tracking and spotting modes
  engine/      trainer, schedulers
  predictor.py end-to-end video inference
tools/         train, evaluate, predict_video, prepare_dataset
configs/       stage1_static, stage2_video, stage3_finetune, smoke
docs/          RESEARCH_REVIEW, DATASETS, TRAINING_GUIDE
tests/         100 tests
```

## Documentation

- [`docs/RESEARCH_REVIEW.md`](docs/RESEARCH_REVIEW.md) — the brief verified,
  with corrections and measurements
- [`docs/DATASETS.md`](docs/DATASETS.md) — which datasets, why, how to get and
  prepare them
- [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) — the curriculum, reading
  loss curves, hardware, ablations worth running

## Status

The pipeline is complete and verified end to end: training, inference,
evaluation, and 100 passing tests including a from-scratch overfit test that
confirms all three heads learn. **No benchmark run has been performed** — that
needs GPU hours and the licensed datasets. Published numbers should come from
your own run of `tools/evaluate.py`; the measurements quoted above are unit-level
and reproducible from the test suite.
