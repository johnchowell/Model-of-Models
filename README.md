# Model of Models (MoM)

A controlled cross-domain study of **amortized weight generation**: instead of
specializing a model to a task by attending to examples (in-context) or taking
gradient steps (MAML/fine-tuning), a hypernetwork *emits the weights of a small
task-specific specialist* in a single forward pass. This repository contains the
code, paper, and results.

**Thesis.** Amortized weight-emission dominates within a trained low-dimensional task
family, at far lower test-time cost than gradient adaptation or in-context attention,
but it cannot match in-context attention for high-dimensional sequence modeling, and
that gap grows with model scale.

## Headline results

| Task | Result |
|---|---|
| Sinusoid (few-shot regression) | ~380x lower MSE than MAML at 0 test-time gradient steps |
| Shape generation | noise-floor accuracy, 26-840x lower inference cost than autoregression |
| Blood-test (HCV) few-shot | ties SOTA TabPFN (0.966 vs 0.967) with a reusable specialist |
| Gene-expression (TCGA) few-shot | saturated tie (all methods >=0.99) |
| Meta-RL (HalfCheetah-Vel) | statistical tie with PEARL-style latent; more consistent across seeds |
| enwik8 LM | captures only 16% to 11% of the in-context gain; the gap widens with scale |
| ARC | 11.5% on seen rules, 0% on unseen-rule induction (a clean negative) |
| Composition | emitted specialists are algebraically composable in weight space |

## Repository layout

```
paper/                 LaTeX source, figures, and compiled PDF
exp*.py                experiments (numbered roughly in order of the study)
results/               result JSONs for every experiment
data/README.md         dataset download instructions (data itself is not vendored)
requirements.txt       Python dependencies
```

Key experiments: `exp14` sinusoid, `exp11` shapes, `exp15b` enwik8, `exp16` ARC,
`exp17` meta-RL, `exp18` blood, `exp19` genes, `exp20` ablations, `exp22`/`exp22b`
composition, `exp23` scaling. `exp1`–`exp10` are the appendix negative-result arc
(training without global backpropagation).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # PyTorch: install the build matching your CUDA driver
```

See `data/README.md` for obtaining the datasets, then run any experiment, e.g.
`python exp14_sinusoid.py`.

## License

See [LICENSE](LICENSE).
