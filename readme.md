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

Runs on any 16 GB GPU. For a lab server over SSH (recommended — no session
limit) use `configs/a4000_*.yaml`; for Colab or Kaggle open
`notebooks/train_colab_kaggle.ipynb`. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

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

## Full walkthrough: training on a college GPU server

End to end — SSH access, datasets, training, and getting results back into this
repository. Written for a single 16 GB GPU (RTX A4000 or similar) reached over
SSH. Platform comparison and memory tables are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Step 0 — Survey the server

```bash
ssh you@server.college.edu

nvidia-smi                    # GPU model, free VRAM, who else is using it
nproc                         # CPU cores -- this caps dataloader throughput
df -h ~ /scratch /tmp         # where you actually have space
python3 --version             # need >= 3.9
command -v sbatch && echo "SLURM cluster -- see Step 5b"
```

Three things to write down, because they determine your config:

- **Free VRAM**, not total. If others are on the card, `batch_size: 12` will
  OOM — use 8, or 4 if it's busy.
- **Core count.** Synthesis is CPU-bound (~17 images/s/core), so on an 8-core
  box the dataloader, not the GPU, is your bottleneck. Set
  `workers ≈ min(cores - 2, 8)`.
- **Where you have quota.** Home directories are often capped at 10–20 GB.
  Datasets and checkpoints belong on `/scratch` or equivalent, *not* in the repo.

### Step 1 — Let the server talk to GitHub

You need this to push results back. Generate a key **on the server** (never copy
your laptop's private key onto a shared machine):

```bash
ssh-keygen -t ed25519 -C "college-server" -f ~/.ssh/id_ed25519_github
cat ~/.ssh/id_ed25519_github.pub
```

Add that public key at **GitHub → Settings → SSH and GPG keys → New SSH key**.
Then tell SSH to use it for GitHub:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

ssh -T git@github.com        # expect: "Hi <user>! You've successfully authenticated"
```

> On a **shared** server, prefer a repo-scoped [deploy key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)
> (with write access) over an account key — it limits the blast radius if the
> machine is compromised. If SSH is blocked by a firewall, use HTTPS with a
> fine-grained personal access token instead.

### Step 2 — Clone and install

```bash
cd /scratch/$USER          # or wherever you have space
git clone -b claude/video-text-spotting-model-3ms4y1 \
    git@github.com:Rahul5914/Ocr_thesis.git
cd Ocr_thesis

python3 -m venv .venv && source .venv/bin/activate    # or conda create -n vtspot python=3.11
pip install --upgrade pip
pip install -r requirements.txt
```

Set your identity for commits made on the server:

```bash
git config user.name  "Rahul Kumar"
git config user.email "115581302+Rahul5914@users.noreply.github.com"
```

**Verify before you commit GPU days to it:**

```bash
python -m pytest tests/ -q                                        # ~15 s, 100 tests
python -c "from vtspot.data.synth_static import discover_fonts; print(len(discover_fonts()), 'fonts')"
python tools/train.py --config configs/smoke.yaml --device cpu    # ~1 min
```

No `sudo` needed. If the font count is under ~10, drop `.ttf` files into
`~/.fonts` — it's on the search path and needs no root. More fonts is one of the
cheapest ways to improve recognition generalisation.

### Step 3 — Datasets

**You can skip this entirely to start.** Stages 1 and 2 generate their data
procedurally — no download, no licence, no waiting. Go straight to Step 4 and
add real data later.

For stage 3 you need a real benchmark. **ICDAR2015-video** is the one to report
on ([rrc.cvc.uab.es](https://rrc.cvc.uab.es/?ch=3), free account required).
Download on your laptop, then transfer:

```bash
# from your laptop -- rsync resumes if the connection drops, scp does not
rsync -avP ~/Downloads/icdar15_video/ \
    you@server.college.edu:/scratch/$USER/raw/icdar15/
```

Convert to the unified schema:

```bash
python tools/prepare_dataset.py --dataset icdar15_video \
    --videos      /scratch/$USER/raw/icdar15/videos \
    --annotations /scratch/$USER/raw/icdar15/gt \
    --out         /scratch/$USER/data/icdar15_video_train
```

**Read the last line of that output.** It compares your parsed text density
against the published figure:

```
mean instances/frame: 5.3 (published ~5.5) -> plausible
```

If it says `SUSPICIOUS`, the parser matched the wrong format variant and every
number you produce afterwards is meaningless — nothing downstream can detect it.
The usual causes are 0-vs-1-based frame numbering and a coordinate-order
mismatch. See [`docs/DATASETS.md`](docs/DATASETS.md) for the other benchmarks.

### Step 4 — Train

**Run inside `tmux`.** An SSH disconnect kills a foreground process, and stage 1
runs for days.

```bash
tmux new -s vtspot
source .venv/bin/activate

# Stage 1 -- synthetic stills. This is what replaces ImageNet pretraining.
python tools/train.py --config configs/a4000_stage1.yaml
```

Detach with `Ctrl-b` then `d`. Reattach any time with `tmux attach -t vtspot`.

```bash
# Stage 2 -- synthetic video; the tracking loss switches on here
python tools/train.py --config configs/a4000_stage2.yaml \
    --init checkpoints/a4000_stage1/last.pt

# Stage 3 -- real video fine-tune (needs Step 3)
python tools/train.py --config configs/stage3_finetune.yaml \
    --init checkpoints/a4000_stage2/last.pt \
    --ckpt-dir checkpoints/stage3
```

Adjust for a shared card: `--batch-size 8 --workers 6`.

**Interrupted?** Every stage resumes:

```bash
python tools/train.py --config configs/a4000_stage1.yaml \
    --resume checkpoints/a4000_stage1/last.pt
```

**Monitoring**, from a second SSH session:

```bash
watch -n 5 nvidia-smi                                    # utilisation and VRAM
python -c "import json; h=json.load(open('checkpoints/a4000_stage1/history.json'));
[print('ep%3d det=%.3f rec=%.3f grad=%.1f' % (e['epoch'],e['loss_det'],e['loss_rec'],e['grad_norm'])) for e in h[-10:]]"
```

`loss_det` should fall steadily. `loss_binary` pinned at 1.0 early is normal.
`grad_norm` of 50–80 on the first steps is expected from random init and should
drop below ~5 within a few hundred steps. If GPU utilisation sits low, you are
dataloader-bound — raise `workers`. Full table in
[`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md).

Rough wall-clock: stage 1 **1–3 days**, stage 2 **12–24 h**, stage 3 **1–3 h**.
Stage 1 dominates because it is doing ImageNet's job. You can cut it short and
still get a working model; the quality ceiling drops with it.

### Step 5a — Evaluate

```bash
python tools/evaluate.py \
    --checkpoint checkpoints/stage3/last.pt \
    --data /scratch/$USER/data/icdar15_video_test \
    --out results.json
```

Prints **tracking mode** (IoU only) and **spotting mode** (IoU + correct
transcription) side by side. Quote the spotting numbers against video-text-
*spotting* literature — the gap between the two is large, and conflating them is
the most common way these benchmarks get misreported.

### Step 5b — If the server uses SLURM

Don't run training on the login node; submit a job:

```bash
sbatch scripts/train_slurm.sh              # stage 1
STAGE=2 sbatch scripts/train_slurm.sh      # stage 2, initialised from stage 1

squeue -u $USER
tail -f logs/vtspot-*.out
```

Edit `--partition` and `--gres` in that script to match your cluster — ask your
admin. It resumes from `last.pt` automatically, so a wall-clock limit shorter
than the run just means resubmitting.

### Step 6 — Get results back into this repository

**Do not commit checkpoints.** They are 195 MB each; `.gitignore` blocks `*.pt`
deliberately. Git is the wrong transport for weights.

Collect the small artefacts — config, loss history, evaluation, provenance:

```bash
./scripts/collect_results.sh a4000_stage1 checkpoints/a4000_stage1
./scripts/collect_results.sh stage3_icdar15 checkpoints/stage3 results.json
```

That writes `experiments/<name>/` containing `SUMMARY.md`, `config.yaml`,
`history.json` and `eval.json`. Commit and push from the server:

```bash
git add experiments/
git commit -m "Add stage 1-3 training results from college server (A4000)"
git push origin claude/video-text-spotting-model-3ms4y1
```

**Shipping the weights themselves** — pick one:

| Method | Good for | Limit |
|---|---|---|
| `scp` to your laptop | your own backup | none |
| **GitHub Release** | sharing one final model | 2 GB/file |
| Git LFS | versioned weights | free tier is 1 GB — a single 195 MB checkpoint eats it fast |

```bash
# to your laptop
scp you@server:/scratch/$USER/Ocr_thesis/checkpoints/stage3/last.pt ./

# or attach to a GitHub Release (from the laptop, once gh is authenticated)
gh release create v0.1-stage3 last.pt --notes "Stage 3, ICDAR15-video fine-tuned"
```

### Step 7 — Merge back into `main`

Once results are pushed and you're happy with them:

```bash
# keep the branch current with main first
git fetch origin
git merge origin/main            # resolve any conflicts, then re-run: pytest tests/ -q
git push origin claude/video-text-spotting-model-3ms4y1
```

Then open a pull request on GitHub from
`claude/video-text-spotting-model-3ms4y1` into `main`, or merge locally:

```bash
git checkout main
git merge --no-ff claude/video-text-spotting-model-3ms4y1
git push origin main
```

`--no-ff` keeps the branch's history as a distinct unit, which is what you want
when the branch represents a body of work rather than a single fix.

**Before merging, from a clean checkout:** `python -m pytest tests/ -q` should
report 100 passed.

---

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
configs/       stage{1,2,3}, plus a4000_* (16 GB server) and colab_* variants
notebooks/     train_colab_kaggle.ipynb
scripts/       train_slurm.sh, collect_results.sh
experiments/   committed run results (configs, loss history, metrics)
docs/          RESEARCH_REVIEW, DATASETS, TRAINING_GUIDE, DEPLOYMENT
tests/         100 tests
```

## Documentation

- [`docs/RESEARCH_REVIEW.md`](docs/RESEARCH_REVIEW.md) — the brief verified,
  with corrections and measurements
- [`docs/DATASETS.md`](docs/DATASETS.md) — which datasets, why, how to get and
  prepare them
- [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) — the curriculum, reading
  loss curves, hardware, ablations worth running
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — SSH server, SLURM, Colab and
  Kaggle, with measured memory figures

## Status

The pipeline is complete and verified end to end: training, inference,
evaluation, and 100 passing tests including a from-scratch overfit test that
confirms all three heads learn. **No benchmark run has been performed** — that
needs GPU hours and the licensed datasets. Published numbers should come from
your own run of `tools/evaluate.py`; the measurements quoted above are unit-level
and reproducible from the test suite.
