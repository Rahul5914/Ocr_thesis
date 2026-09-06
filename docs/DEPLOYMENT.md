# Where to run this

The pipeline is plain PyTorch with no compiled extensions, so it runs anywhere
PyTorch does. What differs between platforms is **how long a run can survive**,
which matters because stage 1 is long by design.

| Platform | GPU | Session limit | Verdict |
|---|---|---|---|
| **College/lab server over SSH** | whatever it has | none | **Best.** Full curriculum in one go |
| Kaggle | P100 16 GB / 2×T4 | 12 h, 30 h/week | Good. Needs resume between sessions |
| Colab free | T4 16 GB | ~12 h, idle disconnects | Workable. Needs resume + Drive |
| Colab Pro | L4 / A100 | longer | Good |

Memory, measured (peak RSS, CPU fp32 — an upper bound; AMP on GPU uses roughly
half):

| Config | Params | Peak |
|---|---|---|
| `stage1_static` / `a4000_stage1` (w48, 640) | 12.2M | 8.8 GB |
| `stage2_video` / `a4000_stage2` (w48, 640, clip 4) | 12.2M | 8.8 GB |
| `colab_stage1/2` (w32, 512) | 8.6M | 5.9 GB |
| w32, 448, clip 3 | 8.6M | 3.8 GB |

Any 16 GB card runs the full-size configs with AMP. If you hit OOM, lower
`crop_size` first, then `backbone_width`, then `batch_size` — **keep
`clip_len >= 3`**, which is load-bearing for the tracking loss.

---

## A server over SSH (recommended)

This is the right home for the job: no session cap, persistent disk, and you can
walk away from it.

### Setup without root

University servers rarely give you `sudo`. Nothing here needs it — the synthetic
generator falls back to the ~40 TrueType fonts bundled with matplotlib, which is
present in every scientific Python install.

```bash
git clone -b claude/video-text-spotting-model-3ms4y1 \
    https://github.com/Rahul5914/Ocr_thesis.git && cd Ocr_thesis

python -m venv .venv && source .venv/bin/activate     # or: conda create -n vtspot python=3.11
pip install -r requirements.txt

python -c "from vtspot.data.synth_static import discover_fonts; print(len(discover_fonts()), 'fonts')"
python -m pytest tests/ -q                             # ~15 s, verifies the install
```

If that font count is under ~10, drop extra `.ttf` files into `~/.fonts` — it is
on the search path, needs no root, and font diversity is one of the cheapest
ways to improve recognition generalisation.

### Run it so it survives your SSH session

An SSH disconnect kills your foreground process. Use `tmux`:

```bash
tmux new -s vtspot
python tools/train.py --config configs/a4000_stage1.yaml
# detach: Ctrl-b then d      reattach later: tmux attach -t vtspot
```

Or `nohup` if `tmux` is unavailable:

```bash
nohup python tools/train.py --config configs/a4000_stage1.yaml > stage1.log 2>&1 &
tail -f stage1.log
```

### The full curriculum

```bash
python tools/train.py --config configs/a4000_stage1.yaml
python tools/train.py --config configs/a4000_stage2.yaml --init checkpoints/a4000_stage1/last.pt
python tools/train.py --config configs/stage3_finetune.yaml --init checkpoints/a4000_stage2/last.pt
```

### RTX A4000 notes

16 GB Ampere, ~19 TFLOPS fp32. `configs/a4000_*.yaml` are sized for it and use
two Ampere features the generic configs do not:

- **bf16 autocast** (`amp_dtype: bf16`). Same speed as fp16 with fp32's exponent
  range, so no loss scaling and no `inf` at step 300 — worth having when early
  from-scratch gradients are large (measured grad norms of 50–80 on the first
  steps). Falls back to fp16 automatically if the GPU lacks bf16.
- **TF32** matmul/conv, on by default in `tools/train.py` (`--no-tf32` disables).
  Roughly 1.5–2× on convolutions for an immaterial precision loss.

`batch_size: 12` uses about 9 GB of the 16. **If the card is shared with other
students, drop to 8** and check with `nvidia-smi` before starting — a 140 W
A4000 in a shared box is easy to oversubscribe.

`workers: 6` should match available **CPU** cores, not the GPU. Synthesis is
CPU-bound (~17 images/s/core), so on a machine with few cores the dataloader,
not the A4000, becomes the bottleneck. Check with `nproc` and set it to
`min(cores - 2, 8)`.

### If the server uses SLURM

```bash
sbatch scripts/train_slurm.sh
squeue -u $USER          # check status
tail -f logs/vtspot-*.out
```

Edit the `--partition` and `--gres` lines in that script to match your cluster —
ask whoever administers it for the right values. The script requeues itself on
timeout and resumes from `last.pt`, so a 24-hour wall-clock limit does not cost
you the run.

---

## Colab / Kaggle

Open `notebooks/train_colab_kaggle.ipynb`. It handles Drive mounting, the
platform differences, and the resume loop.

The one thing to internalise: **sessions die, so checkpoint often and resume.**
`configs/colab_*.yaml` set `steps_per_epoch` to a few hundred so a checkpoint
lands every few minutes, and `keep_last_n: 3` prunes old ones — each is 195 MB,
and 20 unpruned epochs would be 3.9 GB, over a free Drive's comfort zone.

```bash
# re-run this after every disconnect; it picks up where it stopped
python tools/train.py --config configs/colab_stage1.yaml --resume $CKPT/stage1/last.pt
```

**Kaggle:** enable *Settings → Accelerator → GPU* **and** *Settings → Internet →
On* (off by default; `pip install` fails silently-ish without it). Write
checkpoints to `/kaggle/working` (20 GB, persists per notebook).

**Colab:** mount Drive first — `/content` is wiped when the session ends.

---

## Expected wall-clock

Rough, for the A4000 configs. Synthesis is CPU-bound, so these move a lot with
core count.

| Stage | Steps | Order of magnitude |
|---|---|---|
| 1 — synthetic stills | ~600k images | 1–3 days |
| 2 — synthetic video | ~80k clips | 12–24 h |
| 3 — real video fine-tune | small corpus | 1–3 h |

Stage 1 dominates because it is doing the job ImageNet pretraining would
otherwise have done. You can cut it short and still get a working model — the
quality ceiling just drops with it.

Sanity-check before committing days of GPU time:

```bash
python tools/train.py --config configs/smoke.yaml --device cpu   # ~1 min
python -m pytest tests/ -q                                        # ~15 s
```
