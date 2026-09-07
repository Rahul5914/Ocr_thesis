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
python -m pytest tests/ -q                                   # 103 tests, ~20 s
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

## Full walkthrough: training locally on an RTX A4000

For a machine you sit in front of (or remote-desktop into). If you are reaching
a *shared* server over SSH instead, skip to
[the server walkthrough](#full-walkthrough-training-on-a-remote-gpu-server) —
the difference is real: on your own machine you control the drivers, nothing
kills your session at hour 12, and you can watch rendered output directly.

Works on **Linux, Windows and WSL2**. Windows-specific steps are marked.

### Step 0 — Make the GPU visible

The one genuine prerequisite. Everything else is pip.

```bash
nvidia-smi
```

You want to see `NVIDIA RTX A4000` and `16376MiB`. If the command is missing:

- **Linux:** install the proprietary driver (`sudo ubuntu-drivers install` on
  Ubuntu, or your distro's `nvidia-driver` package), then reboot.
- **Windows:** install the **NVIDIA Studio Driver** for the A4000 from
  nvidia.com. Studio over Game Ready — it is the validated branch for compute
  workloads.

**You do not need to install the CUDA Toolkit.** PyTorch wheels bundle their own
CUDA runtime; a recent driver is all that is required. Installing a mismatched
system toolkit is a common way to break an otherwise working setup.

### Step 1 — Python and PyTorch

Use Python **3.10–3.12**.

```bash
git clone https://github.com/Rahul5914/Ocr_thesis.git
cd Ocr_thesis
```

**Linux / WSL2 / macOS:**
```bash
python3 -m venv .venv && source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # see note below
.venv\Scripts\Activate.ps1
```

> **`Activate.ps1 cannot be loaded because running scripts is disabled`**
>
> Windows blocks PowerShell scripts by default; a fresh install hits this every
> time. Neither fix below needs administrator rights.
>
> - `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` — applies to
>   that terminal only and reverts when you close it. Narrowest change; you
>   repeat it each new terminal.
> - `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` —
>   persists for your account. Microsoft's recommended setting: local scripts
>   run, downloaded ones still require a signature.
>
> **Or skip activation entirely.** The virtualenv's interpreter works when
> called directly, no policy change involved:
>
> ```powershell
> .venv\Scripts\python.exe -m pip install -r requirements.txt
> .venv\Scripts\python.exe tools\train.py --config configs\a4000_stage1.yaml
> ```
>
> From `cmd.exe` rather than PowerShell, `.venv\Scripts\activate.bat` is a
> batch file and is not affected by the policy at all.

Then install PyTorch. **On Windows the default PyPI wheel is CPU-only** — it is
about 124 MB, where a CUDA build is 2–3 GB — so it installs cleanly and then
trains at roughly 1% of the speed while merely looking "slow". Get the right
wheel from PyTorch's own index:

1. Open **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)**
   and pick Stable / your OS / Pip / Python / CUDA. It prints the exact command.
2. Run it, then `pip install -r requirements.txt`.

Do not copy a `--index-url .../whl/cuXXX` from a blog post or from this file:
the CUDA version moves with each PyTorch release (2.14 bundles CUDA 13.0), and a
stale index either has no matching wheel or silently gives you an old PyTorch.
The selector is the only source that stays correct.

```bash
pip install --upgrade pip
# ... the command the selector gave you ...
pip install -r requirements.txt
```

**Then verify — this is the step that matters:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and `NVIDIA RTX A4000`. **If it prints `False`, stop and fix it
before going further.**

- Installed the plain PyPI wheel on Windows → reinstall from the selector's
  index (`pip uninstall torch torchvision` first).
- Installed a CUDA wheel and still `False` → your **driver is older than the
  CUDA build**. Either update the driver, or pick an older CUDA version in the
  selector; `nvidia-smi` shows the maximum CUDA version your driver supports in
  its top-right corner.

**You do not need the CUDA Toolkit installed separately** — the wheel bundles
its own runtime. Only the driver has to be new enough.

### Step 2 — Verify the install

```bash
python -m pytest tests/ -q                                       # ~20 s, 103 tests
python -c "from vtspot.data.synth_static import discover_fonts; print(len(discover_fonts()), 'fonts')"
python tools/train.py --config configs/smoke.yaml --device cpu   # ~1 min
```

Fonts: Windows contributes its `C:\Windows\Fonts` collection automatically, and
matplotlib's bundled 40 faces are a fallback everywhere. More fonts is one of
the cheapest ways to improve recognition generalisation — on Linux,
`sudo apt install fonts-dejavu fonts-liberation fonts-freefont-ttf`.

### Step 3 — Datasets (optional to start)

**Stages 1 and 2 need no downloads.** Data is generated procedurally. Go to
Step 4 and add real data later.

For stage 3, get **ICDAR2015-video** ([rrc.cvc.uab.es](https://rrc.cvc.uab.es/?ch=3),
free account). Since you are local, just download into a folder and convert:

```bash
python tools/prepare_dataset.py --dataset icdar15_video \
    --videos      D:/datasets/icdar15/videos \
    --annotations D:/datasets/icdar15/gt \
    --out         D:/datasets/prepared/icdar15_video_train
```

**Read the last line of the output.** It compares parsed text density against
the published figure:

```
mean instances/frame: 5.3 (published ~5.5) -> plausible
```

`SUSPICIOUS` means the parser matched the wrong format variant, and every number
you produce afterwards is meaningless. Nothing downstream can detect it.

> **Windows note:** if you point `--frames` at already-extracted frames, Windows
> refuses to create the symlink used to keep the prepared folder self-contained
> (it needs Administrator or Developer Mode). This is handled: the source path
> is recorded in the annotation instead and the loader reads from there. You do
> not need to run as admin.

### Step 4 — Train

**Windows: use the helper script.** It picks the config and — the reason it
exists — **resumes from `last.pt` automatically** when one is present. A closed
terminal or a reboot then costs only the steps since the last checkpoint, not
the run. It calls `.venv\Scripts\python.exe` directly, so no `Activate.ps1` and
no execution-policy change are involved.

```powershell
.\scripts\train.ps1                 # stage 1, resumes if a checkpoint exists
.\scripts\train.ps1 -Stage 2        # stage 2, initialised from stage 1
.\scripts\train.ps1 -Workers 8      # override the worker count
.\scripts\train.ps1 -Fresh          # ignore the checkpoint and start over
```

By hand, on any platform:

```bash
python tools/train.py --config configs/a4000_stage1_fast.yaml
# after any interruption:
python tools/train.py --config configs/a4000_stage1_fast.yaml \
    --resume checkpoints/a4000_stage1/last.pt
```

**Check the speed before committing days to it.** `a4000_stage1.yaml` measured
~230 h on a real A4000; `a4000_stage1_fast.yaml` targets ~1 day. If throughput
looks wrong, measure rather than guess:

```bash
python tools/benchmark.py --config configs/a4000_stage1_fast.yaml
```

It times the dataloader and the GPU separately and names the bottleneck. The two
fixes are opposites — more workers help only if the GPU is starved, a smaller
crop or model only if it is not.

Then:

```bash
# Stage 2 -- synthetic video; the tracking loss switches on here
python tools/train.py --config configs/a4000_stage2.yaml \
    --init checkpoints/a4000_stage1/last.pt

# Stage 3 -- real video fine-tune (needs Step 3)
python tools/train.py --config configs/stage3_finetune.yaml \
    --init checkpoints/a4000_stage2/last.pt --ckpt-dir checkpoints/stage3
```

The configs are already sized for a 16 GB A4000 and enable two Ampere features:
**bf16 autocast** (fp32's exponent range, so no loss scaling and no `inf` at step
300 — worth having when early from-scratch gradient norms hit 50–80) and **TF32**
convolutions (~1.5–2× for immaterial precision loss).

Since the GPU is all yours, `batch_size: 12` (~9 GB of 16) is safe. Set
`workers` to match your **CPU** cores, not the GPU — synthesis is CPU-bound at
~17 images/s/core, so on a 6-core desktop the dataloader, not the A4000, is your
bottleneck. Check with `nproc` (Linux) or `echo %NUMBER_OF_PROCESSORS%`
(Windows), then `--workers` to about `cores - 2`.

Stage 1 runs **1–3 days**. Keep the machine awake:

- **Windows:** Settings → System → Power → *Screen and sleep* → **Never** sleep.
  A desktop that sleeps mid-run costs you the elapsed hours.
- **Linux:** `systemd-inhibit --what=sleep python tools/train.py ...`, or run
  inside `tmux` so a closed terminal does not kill it.

**Anything interrupts it?** Every stage resumes:

```bash
python tools/train.py --config configs/a4000_stage1.yaml \
    --resume checkpoints/a4000_stage1/last.pt
```

**Monitoring**, in a second terminal:

```bash
nvidia-smi -l 5              # utilisation and VRAM; low util = raise --workers
python -c "import json; h=json.load(open('checkpoints/a4000_stage1/history.json')); [print('ep%3d det=%.3f rec=%.3f grad=%.1f' % (e['epoch'],e['loss_det'],e['loss_rec'],e['grad_norm'])) for e in h[-10:]]"
```

`loss_det` should fall steadily; `loss_binary` pinned at 1.0 early is normal;
`grad_norm` of 50–80 on the first steps is expected from random init and should
fall below ~5 within a few hundred steps. Full table in
[`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md).

### Step 5 — Run it on a video and watch the result

The advantage of being local — you can just open the output:

```bash
python tools/predict_video.py \
    --checkpoint checkpoints/a4000_stage2/last.pt \
    --video my_clip.mp4 --out results.json --render annotated.mp4
```

`annotated.mp4` has boxes, track ids and transcriptions burned in. Evaluate
against a benchmark with:

```bash
python tools/evaluate.py --checkpoint checkpoints/stage3/last.pt \
    --data D:/datasets/prepared/icdar15_video_test --out results.json
```

It prints **tracking mode** (IoU only) and **spotting mode** (IoU + correct
transcription). Quote the spotting numbers against video-text-*spotting*
literature — conflating the two is the most common way these benchmarks get
misreported.

### Step 6 — Push results back to GitHub

Authenticate once (HTTPS with a personal access token is simplest on a personal
machine; `gh auth login` handles it, or use an SSH key as in the server
walkthrough):

```bash
git config user.name  "Rahul Kumar"
git config user.email "115581302+Rahul5914@users.noreply.github.com"

git checkout -b results/a4000-local
./scripts/collect_results.sh a4000_stage1 checkpoints/a4000_stage1
./scripts/collect_results.sh stage3_icdar15 checkpoints/stage3 results.json
git add experiments/ && git commit -m "Add A4000 training results"
git push -u origin results/a4000-local
```

**On Windows**, `collect_results.sh` needs a bash shell — use Git Bash (ships
with Git for Windows) or WSL2. Or copy the four files by hand: `config.yaml` and
`history.json` from the checkpoint directory, plus your `results.json`.

**Do not commit checkpoints.** They are 195 MB each and `.gitignore` blocks
`*.pt` deliberately. Share weights via a GitHub Release (2 GB/file) instead:

```bash
gh release create v0.1-stage3 checkpoints/stage3/last.pt \
    --notes "Stage 3, ICDAR15-video fine-tuned, RTX A4000"
```

**Do you need a pull request?** Only if someone reviews your work. Working
solo on your own repository, push straight to `main` — that is what the commands
above do. A PR is a review checkpoint, not a requirement for saving code.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | CPU-only wheel (the Windows PyPI default), or a driver older than the wheel's CUDA version. Reinstall from [the PyTorch selector](https://pytorch.org/get-started/locally/); check `nvidia-smi` for your driver's max CUDA |
| `CUDA out of memory` | Lower `crop_size` first, then `backbone_width`, then `batch_size`. **Keep `clip_len >= 3`** — it is load-bearing for the tracking loss |
| GPU utilisation < 40% | Dataloader-bound. Raise `workers`; if already at core count, lower `crop_size` |
| Training crawls, GPU idle | Almost always the CPU-only wheel — recheck Step 1's verify command |
| `no TrueType fonts found` | Should not happen (matplotlib fallback). If it does, drop `.ttf` files into `~/.fonts` |
| Windows: `Activate.ps1 cannot be loaded` | PowerShell execution policy. `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or call `.venv\Scripts\python.exe` directly. No admin needed — see Step 1 |
| Windows: `symlink` / WinError 1314 | Already handled — see the note in Step 3 |
| Windows: DataLoader hangs at start | Set `workers: 0` in the config to confirm, then raise gradually; Windows spawns rather than forks, so worker startup is much slower |

---

## Full walkthrough: training on a remote GPU server

For a **shared machine reached over SSH** (a college cluster or lab box). If the
GPU is in a computer you have direct access to, use the
[local walkthrough](#full-walkthrough-training-locally-on-an-rtx-a4000) above
instead. Platform comparison and memory tables are in
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
git clone git@github.com:Rahul5914/Ocr_thesis.git
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
python -m pytest tests/ -q                                        # ~20 s, 103 tests
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
git push origin main
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

**Working solo?** Commit on `main` and push. There is nothing to merge and no
PR to open:

```bash
git add experiments/
git commit -m "Add A4000 training results"
git push origin main
```

**Used a branch** (worth it when a run might not pan out, so `main` stays
clean)? Merge it when you are happy with the results:

```bash
git fetch origin
git checkout <your-branch>
git merge origin/main         # resolve conflicts, then re-run: pytest tests/ -q
git checkout main
git merge --no-ff <your-branch>
git push origin main
```

A pull request is only needed when someone else reviews the change. `--no-ff`
keeps the branch's commits visible as one unit rather than flattening them.

`--no-ff` keeps the branch's history as a distinct unit, which is what you want
when the branch represents a body of work rather than a single fix.

**Before merging, from a clean checkout:** `python -m pytest tests/ -q` should
report 103 passed.

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
tests/         103 tests
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
evaluation, and 103 passing tests including a from-scratch overfit test that
confirms all three heads learn. **No benchmark run has been performed** — that
needs GPU hours and the licensed datasets. Published numbers should come from
your own run of `tools/evaluate.py`; the measurements quoted above are unit-level
and reproducible from the test suite.
