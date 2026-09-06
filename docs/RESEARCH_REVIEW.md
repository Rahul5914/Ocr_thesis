# Review of the Video Text Spotting research brief

Everything checkable in the brief was checked against the primary sources. The
survey is **substantially accurate** — the architectural taxonomy, the reasons
Transformer sequence modelling displaced heuristic matching, the CTC/attention
trade-off, and the metric critique are all correct and well-framed. The
citations are real; nothing is hallucinated.

What follows is split into three parts: numbers that need correcting, claims
that are right but incomplete in ways that matter, and **design decisions in the
brief that would have caused the from-scratch build to fail**. The third section
is the important one.

---

## Part 1 — Factual corrections

### 1.1 The initialisation formulas conflate variance with standard deviation ❌

The brief writes:

> $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

and gives Xavier as `N(0, sqrt(2/(n_in+n_out)))` in the comparison table.

In the notation `N(μ, σ²)` the second argument is the **variance**. He
initialisation specifies

$$\mathrm{Var}(W) = \frac{2}{n_{in}} \quad\Longleftrightarrow\quad \sigma = \sqrt{\frac{2}{n_{in}}}$$

so writing `N(0, sqrt(2/n_in))` describes a distribution whose *variance* is
`sqrt(2/n_in)`. For a 3×3×256 convolution (`n_in` = 2304) the intended σ is
0.029; the literal reading gives σ = 0.17, a **6× too-wide** distribution — and
6× per layer compounds catastrophically with depth. The same conflation appears
in the Xavier row.

This is a notation slip, not a conceptual error, but it is the kind that
survives into code. `vtspot/models/init.py` states it explicitly and
`tests/test_init.py::test_kaiming_std_matches_formula` asserts the realised σ
against the formula.

### 1.2 The Xavier-vs-He argument is correct, and now has numbers ✅

The brief's claim that Xavier on a deep ReLU network causes signal collapse is
right. Measured here on a 20-layer ReLU stack (`tests/test_init.py`):

| Init | `Var(layer 19) / Var(layer 0)`, median over 12 seeds | Range |
|---|---|---|
| He (Kaiming) | 2.1 × 10⁻¹ | [0.033, 0.503] |
| Xavier | 1.9 × 10⁻⁷ | [3.1 × 10⁻⁸, 1.2 × 10⁻⁵] |

Roughly **six orders of magnitude** of separation. The brief understated its own
case.

### 1.3 Fixup is described as one rule; it is four ❌

The brief gives only the scaling factor `L^(-1/(2m-2))`, which is correct but is
**step 2 of 4**. Applied alone, Fixup diverges. The full method (Zhang et al.,
ICLR 2019) is:

1. Initialise the classifier **and the last layer of every residual branch to
   zero**, so each branch starts as the identity.
2. Initialise everything else with He, scaling weights inside residual branches
   by `L^(-1/(2m-2))`.
3. Insert a **scalar bias** before every convolution, linear and element-wise op.
4. Insert a **scalar multiplier** before every convolution and linear.

Steps 1, 3 and 4 are missing from the brief. `FixupBasicBlock` in
`vtspot/models/backbone.py` implements all four; `tests/test_init.py` asserts the
zeroed branch outputs and the scaling factor numerically.

One further omission: Fixup removes BatchNorm's *regularisation* effect as well
as its normalisation, so it needs stronger augmentation (the paper uses mixup) to
match BN accuracy. It was also designed and tuned for SGD, not AdamW.

**Recommendation:** GroupNorm is the better default here and is what ships as
`norm: gn`. It is batch-size independent (essential — video clip training means
2–4 clips per step), needs no extra care, and costs nothing at inference. Fixup
is available as `norm: none` if you want to study it, which for a thesis is a
legitimate reason to keep it.

### 1.4 Numbers that drifted

| Claim in brief | Source says | Impact |
|---|---|---|
| CoText: **71.7%** IDF1 at **32.3** FPS | **72.0%** IDF1 at **41.0** FPS on ICDAR2015-video | Minor; cite correctly |
| DSText: "more than **15** instances per frame" | Average text density is **24** per frame, across **12** scenarios (vs 5.55 for ICDAR15) | Understates the benchmark's difficulty |
| DSText: 100 videos | 100 in the ICDAR-2023 competition; **DSText V2 has 140 videos / 62.1k frames** | Use V2 |

Verified as stated: TransDETR's "up to 11.0% improvement" and 72.8% IDF1;
VimTS's 5.5% MOTA gain from image-level data only, and VTD-368k via CoDeF;
ArTVideo's 60 videos with >30% curved text (13k frames, 170k instances);
BOVText's 2,021 videos / 1.75M frames / 32 scenarios; ICDAR13-video's 28 videos
and ICDAR15-video's 49. TraRA and VLSpotter are real papers, described
accurately.

---

## Part 2 — Right but incomplete

### 2.1 "From scratch" needs a definition, and the brief has the right one implicitly

The constraint is *no externally pretrained weights* — no ImageNet backbone, no
pretrained LM. It is **not** "no pretraining at all". Supervised pretraining on
data you generate yourself is still from scratch, and the brief's synthetic
curriculum is exactly that. Worth stating outright, because it reframes the
synthetic stage from a nice-to-have into **the thing that replaces ImageNet**.
Under-invest there and nothing downstream works. That is why
`vtspot/data/synth_static.py` and `synth_video.py` are the largest data modules
here rather than an afterthought.

### 2.2 Deformable attention is proposed for the wrong bottleneck ⚠️

The brief recommends deformable attention to solve "quadratic memory explosion"
from feeding 36 frames to a Transformer. Deformable attention does reduce
attention cost from O(HW·HW) to O(HW·K) — that part is right. But:

- The dominant memory cost in video text spotting is **activations across the
  clip** (frames × resolution × channels), which deformable attention does not
  touch. The effective fixes are gradient checkpointing, short clips (2–5
  frames) plus a memory bank rather than 36-frame windows, and AMP.
- Practically, deformable attention means compiling `MSDeformAttn` CUDA kernels.
  Making that a hard dependency of a from-scratch thesis project is a poor
  trade: it is a common source of build failures across CUDA versions, and it is
  only needed if you commit to a DETR-style decoder in the first place.

The design here sidesteps both: a fully-convolutional detector with a small
attention module operating only on ~32-token per-instance sequences. Nothing to
compile, no quadratic term.

### 2.3 Contrastive tracking has a failure mode the brief does not mention ⚠️

The brief recommends contrastive loss on the track head (correct — it is what
CoText does). What it omits: **scene text repeats constantly**. The same shop
sign appears twice, a price appears twice on a menu, "EXIT" appears on every
door. Two instances can be *pixel-identical*, so an appearance-only embedding is
being asked to separate pairs that carry no distinguishing appearance signal. It
will either fail on them or distort the embedding space trying.

The mitigations implemented here:
- a hard **motion gate** in the matcher, so appearance never has to break a tie
  that geometry already settles (`ShortTermMatcher`);
- optional re-weighting of same-transcription negatives in the contrastive loss
  (`ContrastiveTrackLoss(hard_negative_weight=...)`), exposing both policies —
  push them apart, or accept the ambiguity and let motion decide.

`tests/test_tracking.py::test_motion_gate_breaks_ties_between_identical_appearances`
covers this directly.

### 2.4 Trajectory-level aggregation is right; the proposed mechanism is not available to you ⚠️

The brief cites TraRA, which reaches trajectory-level consistency by adapting a
**9B-parameter vision-language model** with LoRA. That is a pretrained
large model — using it contradicts the from-scratch constraint outright.

The good news is that most of the benefit does not need it. Per-frame
transcriptions of one trajectory can be fused by confidence-weighted voting plus
per-character majority vote over near-identical readings — **zero extra
parameters, no extra training**. `aggregate_trajectory_text` implements this, and
`tests/test_tracking.py::test_aggregation_recovers_a_word_no_single_frame_got_right`
shows it recovering a word that *no individual frame* read correctly.

### 2.5 The rescoring mechanism can be free

The brief correctly identifies GoMatching++'s rescoring head as the fix for the
image→video domain gap (detector confidence is mis-calibrated on degraded video
frames, so a fixed threshold loses recall). GoMatching++ trains a head for this.

There is a signal already in the model: **a real text region decodes to a
confident string; a background false positive decodes to blanks or garbage.**
Combining detector score and recogniser confidence as a geometric mean gives a
rescoring signal for free (`VideoTextTracker._combined_score`). A trained head
is likely better; this costs nothing and needs no extra labels.

### 2.6 Uncertainty weighting needs a clamp

Kendall's homoscedastic weighting is a good suggestion, with one caveat the brief
omits: nothing stops the optimiser from raising `log σ²` without bound for a task
whose loss stays high — which **switches that task off**. Tracking, the hardest
of the three, is exactly the one it will pick. You discover this at the end of a
long run when IDF1 is zero. `UncertaintyWeighting` clamps `s ∈ [-3, 3]`
(`tests/test_losses.py::test_uncertainty_weight_is_clamped`), and the default for
stages 1–2 is fixed weights, because learnt weighting is only interpretable once
each loss is already descending.

### 2.7 Missing from the training-stability section

The brief covers warmup and cosine decay (both correct and essential). Three
additions that matter as much from random init:

- **Pre-LN, not post-LN, transformer blocks.** Post-LN needs warmup to survive at
  all and still diverges more readily from random init. This is a very common
  cause of "my transformer NaN'd in epoch 1".
- **Gradient clipping** (1.0 here) — measured grad norms hit 79 on the first
  steps of a from-scratch run.
- **No weight decay on norms, biases and positional embeddings.**

### 2.8 Missing from the metrics section

The brief's metric analysis is good — the MOTA-favours-detection critique is
correct, and confirmed here: a synthetic identity switch moves MOTA from 1.00 to
0.95 while IDF1 drops to 0.75 and AssA to 0.75
(`tests/test_metrics.py::test_mota_is_insensitive_to_fragmentation_but_idf1_is_not`).

What is missing is the distinction that decides whether your headline number
means anything: **for the end-to-end *spotting* task, a match requires the
transcription to be correct, not just the IoU.** Same metric code, one extra
predicate. Measured on a case where every box is perfect and one word of two is
misread:

| Mode | MOTA | IDF1 | HOTA |
|---|---|---|---|
| Tracking (IoU only) | 1.00 | 1.00 | 1.00 |
| Spotting (IoU + transcription) | 0.00 | 0.50 | 0.58 |

Reporting the first as a spotting result overstates it enormously. `evaluate()`
takes `spotting=True/False` and `tools/evaluate.py` **always prints both**, so
the gap is visible rather than assumed.

---

## Part 3 — Design decisions that would have broken the build

These are the loopholes. Each would have cost weeks.

### 3.1 A DETR-style set-prediction head, trained from scratch, is a trap 🔴

**This is the most consequential correction.**

The brief's architecture section recommends the TransDETR/VimTS family: text
queries, a Transformer decoder, Hungarian matching. That is the right reading of
the literature — those *are* the state of the art. But every one of them starts
from an **ImageNet-pretrained ResNet-50**, and several from COCO-pretrained
Deformable-DETR weights. The brief adopts their architecture while removing the
initialisation they depend on.

Why it fails without pretraining:

- Hungarian matching supervises **one query per ground-truth instance per step**.
  That is an extremely sparse gradient signal.
- Early in training the matching is unstable — which query owns which instance
  flips between epochs, so the supervision is not merely sparse but
  *inconsistent*.
- DETR needs ~500 epochs on COCO **with** a pretrained backbone. Removing the
  backbone initialisation makes it worse, not better.

**The correction: use a dense head.** Differentiable Binarization (DBNet)
supervises **every pixel at every step**. With no pretrained features to lean on,
that difference dominates everything else. Arbitrary shapes come free because the
output is a mask — which also solves §3.2.

This is not a downgrade for the task. DB-based spotters are competitive on scene
text, and for a thesis the tractability difference is the whole project.
Empirically here: detection loss 3.06 → 0.39 within 12 epochs on CPU.

### 3.2 A rotated-RoI recognition head cannot read curved text 🔴

The brief specifies "Affine transformations utilizing predicted translation and
rotation matrices" and a "rotated RoI module" for the recognition head — and,
correctly, notes elsewhere that curved text is a requirement (>30% of ArTVideo).
**These two requirements are incompatible.** An affine/rotated-rect crop of
curved text necessarily includes background between the baseline and the box.

Measured on a sine-curved text band:

| Crop method | Text signal recovered |
|---|---|
| Axis-aligned box | 0.49 |
| **PolyAlign (polygon-following)** | **0.86** |

Half the crop is background under the rectangular assumption — and that
background goes straight into the CTC head.

**The correction: PolyAlign** (`vtspot/models/roi.py`). Each instance is
described by *K* top and *K* bottom control points; sampling at `(u,v)` gives
`p = (1-v)·top(u) + v·bottom(u)`, which is one `grid_sample` call. A rotated
rectangle is the degenerate case `K=2`, so one code path serves straight,
rotated and curved text. It is differentiable end-to-end and needs no compiled
extension.

### 3.3 The unclip ratio is aspect-ratio dependent, and text is thin 🔴

Not in the brief, but it would have bitten silently. DB post-processing shrinks
polygons for training and re-expands ("unclips") predictions at inference. The
standard implementation uses one global `unclip_ratio = 1.5`. The exact inverse
is **not** a function of the shrink ratio alone — it depends on the instance's
own area and perimeter:

| Box | Exact unclip ratio (shrink = 0.4) |
|---|---|
| 60 × 55 | 1.45 |
| 100 × 20 | 2.50 |
| 300 × 12 | **4.88** |

A fixed 1.5 therefore **systematically under-expands long thin instances** —
which is what text mostly is. Recovering source polygons from their shrunk form:

| Method | Mean IoU | Instances clearing IoU ≥ 0.5 |
|---|---|---|
| Fixed ratio 1.5 | 0.618 | 4 / 7 |
| **Adaptive (ours)** | **0.893** | **7 / 7** |

For the elongated cases the fixed ratio lands at IoU 0.32–0.45 — *below the
matching threshold*, so a correct detection is scored as a false positive **and**
a false negative. `adaptive_unclip_distance` solves the offset analytically per
instance; `decode_db_polygons(adaptive_unclip=False)` restores standard
behaviour for comparison.

### 3.4 The charset is never specified, and it changes the architecture 🔴

The brief recommends BOVText (bilingual, English + Chinese) without addressing
what that does to the recognition head. A Chinese charset means **thousands** of
CTC classes instead of 37. From random initialisation, with no pretrained
embeddings, every class must be learnt from your own data — and the long tail of
Chinese characters appears a handful of times in any realistic corpus.

Decide this before training, because it determines the output layer, the synth
generator's font requirements, and how much data you need. `vtspot/utils/charset.py`
makes it explicit: `alnum` (36 classes, what ICDAR video spotting is actually
scored on) is the default; `ascii94` and a file-based charset are available. The
BOVText converter marks non-Latin instances as **ignore** by default, so they
suppress false positives without demanding classes the model cannot learn
(`--keep-non-latin` overrides).

### 3.5 Training the recogniser on predicted boxes ⚠️

Implied by "end-to-end". Early in a from-scratch run the detector's polygons are
noise, and cropping from noise teaches the recogniser to read noise. Here the
recognition and tracking heads always crop from **ground-truth** polygons, with
`roi_jitter` adding controlled noise so they are not brittle to a detector that
is merely good rather than perfect.

### 3.6 The dataset list is missing the static corpora ⚠️

The brief lists MLT-19 as the only static dataset. Stage 1 needs far more than
that. See `docs/DATASETS.md` — SynthText (800k), MJSynth (9M crops), COCO-Text,
TextOCR, Total-Text and CTW1500 (curved), plus RoadText-1K (1,000 videos, the
largest real annotated video corpus) which the brief omits entirely.

---

## Summary

| | Brief | Status |
|---|---|---|
| Landscape survey and taxonomy | Accurate | ✅ |
| Citations | All real, mostly precise | ✅ (3 numeric fixes) |
| He vs Xavier reasoning | Correct | ✅ (notation fixed) |
| Fixup | 1 of 4 rules | ❌ completed |
| Synthetic curriculum | Correct and central | ✅ |
| Warmup + cosine | Correct | ✅ (3 additions) |
| Metric critique | Correct | ✅ (spotting mode added) |
| DETR-style head from scratch | Would not converge | 🔴 replaced with dense DB head |
| Rotated RoI for curved text | Self-contradictory | 🔴 replaced with PolyAlign |
| Unclip ratio | Not addressed | 🔴 made adaptive |
| Charset | Not addressed | 🔴 made explicit |
| Deformable attention | Right idea, wrong bottleneck | ⚠️ avoided |
| Contrastive tracking | Missing the repeated-text failure | ⚠️ motion gate added |
| TraRA-style aggregation | Needs a 9B VLM | ⚠️ zero-parameter substitute |

The research was a sound basis. The corrections in Part 3 are what separate a
design that reads well from one that trains.
