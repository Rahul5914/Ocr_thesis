# Training guide

## The curriculum, and why it is not optional

With no pretrained weights, the three tasks cannot be learnt simultaneously from
random initialisation — the association head has nothing to associate until the
detector works, and the recogniser has nothing to read. The stages exist to give
each task a foundation before the next depends on it.

```
stage 1  synthetic stills   ->  strokes, glyphs, text/background separation
stage 2  synthetic video    ->  association, motion blur, object permanence
stage 3  real annotated video -> domain adaptation
```

Each stage initialises from the previous checkpoint:

```bash
python tools/train.py --config configs/stage1_static.yaml
python tools/train.py --config configs/stage2_video.yaml    --init checkpoints/stage1_static/last.pt
python tools/train.py --config configs/stage3_finetune.yaml --init checkpoints/stage2_video/last.pt
```

Verify the plumbing first (about a minute on CPU):

```bash
python tools/train.py --config configs/smoke.yaml --device cpu
python -m pytest tests/ -q
```

## What each stage is for

**Stage 1 — synthetic stills.** The longest stage; it is standing in for
ImageNet pretraining. Clips are length 1, so the tracking loss is inactive
(`ContrastiveTrackLoss` returns zero without cross-frame positives) and its
weight is 0 in the config. Treat 100k–300k images as a floor, not a target.

**Stage 2 — synthetic video.** The tracking loss switches on. `clip_len >= 3` is
required for the contrastive loss to have meaningful long-range positives at all.
Occluders in the generator create the disappear/reappear events that the
long-term matcher exists for; without them it never trains.

**Stage 3 — real video.** Low LR (1e-4), and watch for overfitting —
ICDAR15-video has 25 training videos. This stage adapts, it does not teach. It is
also where learnt loss weighting (`weighting.type: uncertainty`) becomes safe,
because all three losses are already descending.

## Reading the loss curves

| Symptom | Likely cause |
|---|---|
| `loss_prob` flat near 0.69 | Learning rate too low, or warmup too long |
| `loss_binary` pinned at 1.0 early | Normal. The DB binary map is inert until P crosses T; it engages once BCE has raised P. `db_k_warmup_steps` softens this |
| `loss_rec` stuck ~3.6 | CTC collapsing to all-blank. Check `inst_has_text` is non-empty and crops are not blank |
| `loss_track` pinned at `log(k)` | The SupCon floor for k positives per anchor — that is convergence, not failure |
| `grad_norm` > 100 after warmup | Lower the LR; clipping is masking a real instability |
| Loss goes NaN | The step is skipped and logged. Repeated NaNs mean LR too high, or AMP overflow — try `amp: false` to confirm |

Grad norms of 50–80 on the *first* few steps are expected from random
initialisation. They should fall below ~5 within a few hundred steps.

## Hardware

The default config (48-wide backbone, 640×640 crops, 12.2M parameters) fits a
16 GB GPU at batch 2 × 4 frames. To scale down:

| Constraint | Change |
|---|---|
| Less memory | `backbone_width: 32`, `crop_size: [512, 512]`, `clip_len: 2` |
| Faster iteration | `backbone_layers: [1,1,1,1]`, `fpn_channels: 128` |
| More capacity | `backbone_width: 64`, `backbone_layers: [3,4,6,3]` |

`amp: true` roughly halves activation memory on CUDA. It is ignored on CPU.

## Configuration reference

```yaml
model:
  backbone_width: 48        # stage channels widen 1x/2x/4x/8x from here
  norm: gn                  # gn (default) | bn | none (Fixup, no normalisation)
  fpn_channels: 256
  roi_h: 8                  # PolyAlign crop height
  roi_w: 32                 # crop width == CTC sequence length; bounds word length
  use_attention_branch: true  # GTC guidance during training; dropped at inference
  roi_jitter: 0.05          # polygon noise, bridges the GT-crop/predicted-crop gap
  db_k: 50.0                # DB steepness
  db_k_start: 2.0           # annealed from here; set == db_k for plain DBNet

train:
  warmup_steps: 2000        # not optional from random init
  grad_clip: 1.0
  ema_decay: 0.999          # averaged weights beat the last iterate here
  weighting:
    type: fixed             # use `uncertainty` only once losses are descending
    weights: {loss_det: 1.0, loss_rec: 1.0, loss_track: 0.5}
```

## Evaluation

```bash
python tools/evaluate.py --checkpoint checkpoints/stage3_finetune/last.pt \
                         --data data/prepared/icdar15_video_test
```

Prints **both** tracking-mode (IoU only) and spotting-mode (IoU + correct
transcription) MOTA / MOTP / IDF1 / HOTA. Quote the spotting numbers against
video-text-*spotting* literature and the tracking numbers against video-text-
*tracking* literature — the gap between them is large, and conflating the two is
the most common way these benchmarks get misreported.

## Inference

```bash
python tools/predict_video.py --checkpoint checkpoints/stage3_finetune/last.pt \
                              --video clip.mp4 --out results.json --render annotated.mp4
```

`results.json` holds one entry per trajectory: track id, aggregated
transcription and confidence, frame span, and the polygon in every frame.

## Things worth trying for a thesis

The scaffolding for each of these is already in place:

- **Fixup vs GroupNorm** (`norm: none` vs `norm: gn`) — a clean ablation on
  normalisation-free from-scratch training.
- **Adaptive vs fixed unclip** (`decode_db_polygons(adaptive_unclip=...)`) —
  quantify the effect on long thin instances at IoU 0.5.
- **GTC guidance on/off** (`use_attention_branch`) — does the training-only
  attention branch help CTC, and by how much?
- **Trajectory aggregation on/off** — the gap between per-frame and
  trajectory-level transcription accuracy.
- **Contrastive negative policy** (`hard_negative_weight` vs `ambiguous_weight`)
  — push identical-text instances apart, or let the motion prior decide?
- **Recognition rescoring** (`use_recognition_rescore`) — how much recall does
  the free rescoring signal recover on degraded frames?
