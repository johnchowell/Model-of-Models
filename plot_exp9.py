import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from exp1_synthgrad import CLASS_NAMES, DEV, make_batch
from exp8_ar import ARNet, rollout

torch.manual_seed(7)
e8 = json.load(open("exp8_results.json"))
e9 = {**json.load(open("exp9_results_v1_v2.json")), **json.load(open("exp9_results_v3_v4.json"))}

nets = {}
sd8 = torch.load("exp8_nets.pt", map_location=DEV)
for name, sd in [("backprop", sd8["backprop"]), ("greedy-local", torch.load("exp9_v2.pt", map_location=DEV))]:
    nets[name] = ARNet().to(DEV)
    nets[name].load_state_dict(sd)
    nets[name].eval()

fig = plt.figure(figsize=(19, 8.5))
gs = fig.add_gridspec(3, 9, width_ratios=[2.2] + [1] * 8, height_ratios=[1, 1, 1])

axl = fig.add_subplot(gs[:, 0])
axl.plot(e8["backprop"]["step"], e8["backprop"]["mse"], c="tab:blue", label="backprop")
axl.plot(e8["self"]["step"], e8["self"]["mse"], c="tab:red", label="grad predictors + aux")
axl.plot(e9["v1"]["step"], e9["v1"]["mse"], c="tab:orange", alpha=0.7, label="error-cond predictors")
axl.plot(e9["v2"]["step"], e9["v2"]["mse"], c="tab:green", lw=2, label="greedy local (aux only)")
axl.axhline(e8["baseline"], ls=":", c="gray", label="copy-last baseline")
axl.set(xlabel="step", ylabel="teacher-forced MSE (log)", yscale="log",
        title="AR next-point prediction\n(all non-blue: 0% global backprop)")
axl.legend(fontsize=8); axl.grid(alpha=0.3)

SEED = 8
x, y = make_batch(8, 32)
outs = {m: rollout(n, x, SEED).cpu() for m, n in nets.items()}
xc = x.cpu()
rows = [("ground truth", xc, "k"), ("backprop rollout", outs["backprop"], "tab:blue"),
        ("greedy-local rollout\n(no backprop)", outs["greedy-local"], "tab:green")]
for r, (title, pts, c) in enumerate(rows):
    for i in range(8):
        ax = fig.add_subplot(gs[r, i + 1])
        ax.plot(xc[i, :, 0], xc[i, :, 1], "-", c="lightgray", lw=1)
        ax.plot(pts[i, :SEED, 0], pts[i, :SEED, 1], "-", c="gray", lw=1.5)
        ax.plot(pts[i, SEED:, 0], pts[i, SEED:, 1], "-", c=c, lw=1.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if r == 0:
            ax.set_title(CLASS_NAMES[y[i]], fontsize=8)
        if i == 0:
            ax.set_ylabel(title, fontsize=8)

fig.suptitle("Exp 9/10 — greedy local learning (per-stage aux heads, zero backprop) nearly matches backprop on generation")
fig.tight_layout()
fig.savefig("exp9_final.png", dpi=130)
print("saved exp9_final.png")
