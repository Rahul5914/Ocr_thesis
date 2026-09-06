#!/bin/bash
# Gather one training run's *small* artefacts into experiments/<name>/ so they
# can be committed and pushed back from the server.
#
#   ./scripts/collect_results.sh a4000_stage1 checkpoints/a4000_stage1
#   ./scripts/collect_results.sh stage3_icdar15 checkpoints/stage3 results.json
#
# Collected: the config actually used, the loss history, evaluation output, and
# a provenance summary (commit, host, GPU, final losses).
#
# Deliberately NOT collected: *.pt checkpoints. They are ~195 MB each and git is
# the wrong transport for them -- see the README for how to ship weights.

set -euo pipefail

NAME="${1:?usage: collect_results.sh <run-name> <checkpoint-dir> [eval-results.json]}"
CKPT_DIR="${2:?usage: collect_results.sh <run-name> <checkpoint-dir> [eval-results.json]}"
EVAL_JSON="${3:-}"

OUT="experiments/${NAME}"
mkdir -p "${OUT}"

[ -f "${CKPT_DIR}/config.yaml" ]  && cp "${CKPT_DIR}/config.yaml"  "${OUT}/config.yaml"
[ -f "${CKPT_DIR}/history.json" ] && cp "${CKPT_DIR}/history.json" "${OUT}/history.json"
[ -n "${EVAL_JSON}" ] && [ -f "${EVAL_JSON}" ] && cp "${EVAL_JSON}" "${OUT}/eval.json"

{
  echo "# Run: ${NAME}"
  echo
  echo "| field | value |"
  echo "|---|---|"
  echo "| date | $(date -u '+%Y-%m-%d %H:%M UTC') |"
  echo "| commit | $(git rev-parse --short HEAD 2>/dev/null || echo unknown) |"
  echo "| host | $(hostname) |"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "| gpu | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) |"
  fi
  echo "| checkpoint dir | \`${CKPT_DIR}\` |"

  if [ -f "${OUT}/history.json" ]; then
    python3 - "${OUT}/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
if h:
    last = h[-1]
    print(f"| epochs completed | {len(h)} |")
    for k in ("loss_total", "loss_det", "loss_rec", "loss_track"):
        if k in last:
            print(f"| final {k} | {last[k]:.4f} |")
    secs = sum(e.get("seconds", 0) for e in h)
    print(f"| wall clock | {secs/3600:.1f} h |")
PY
  fi

  if [ -f "${OUT}/eval.json" ]; then
    echo
    echo "## Evaluation"
    python3 - "${OUT}/eval.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for mode in ("tracking", "spotting"):
    if mode in d:
        m = d[mode]
        print(f"\n**{mode}** — "
              + ", ".join(f"{k.upper()}={m[k]}" for k in
                          ("mota", "idf1", "hota") if k in m))
PY
    echo
    echo "> Quote the **spotting** numbers against video-text-*spotting*"
    echo "> literature; tracking-mode numbers are not comparable to them."
  fi
} > "${OUT}/SUMMARY.md"

echo "collected -> ${OUT}"
ls -la "${OUT}"
echo
echo "next:  git add ${OUT} && git commit -m 'Add results for ${NAME}' && git push"
