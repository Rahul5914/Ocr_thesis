# Experiment results

One directory per training run, created by `scripts/collect_results.sh`:

```bash
./scripts/collect_results.sh a4000_stage1 checkpoints/a4000_stage1
./scripts/collect_results.sh stage3_icdar15 checkpoints/stage3 results.json
```

Each contains:

| file | what |
|---|---|
| `SUMMARY.md` | provenance — commit, host, GPU, epochs, final losses, metrics |
| `config.yaml` | the config actually used, copied from the checkpoint dir |
| `history.json` | per-epoch losses, learning rate, gradient norms, wall clock |
| `eval.json` | tracking-mode and spotting-mode metrics, if evaluated |

**Weights are not stored here.** A checkpoint is ~195 MB, and `.gitignore`
blocks `*.pt` on purpose — git is the wrong transport for them. Ship weights via
`scp` or a GitHub Release; see the walkthrough in the top-level README.

Committing these small artefacts is what makes a run reproducible: the config
pins every hyperparameter, and `SUMMARY.md` pins the commit it ran against.
