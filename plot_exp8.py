import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from exp1_synthgrad import CLASS_NAMES, DEV, make_batch
from exp8_ar import ARNet, rollout

torch.manual_seed(7)
res = json.load(open("exp8_results.json"))
sd = torch.load("exp8_nets.pt", map_location=DEV)
nets = {}
for m in ["backprop", "self"]:
    nets[m] = ARNet().to(DEV)
    nets[m].load_state_dict(sd[m])
    nets[m].eval()

SEED = 8
x, y = make_batch(8, 32)
outs = {m: rollout(n, x, SEED).cpu() for m, n in nets.items()}
xc = x.cpu()

fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(3, 9, width_ratios=[1.6] + [1] * 8)

axl = fig.add_subplot(gs[:, 0])
axl.plot(res["backprop"]["step"], res["backprop"]["mse"], c="tab:blue", label="backprop")
axl.plot(res["self"]["step"], res["self"]["mse"], c="tab:green", label="self-trained (0% backprop)")
axl.axhline(res["baseline"], ls=":", c="gray", label="copy-last baseline")
axl.set(xlabel="step", ylabel="teacher-forced MSE", title="Next-point prediction", yscale="log")
axl.legend(fontsize=8); axl.grid(alpha=0.3)

rows = [("ground truth", xc, "k"), ("backprop rollout", outs["backprop"], "tab:blue"),
        ("self-trained rollout", outs["self"], "tab:green")]
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

fig.suptitle("Exp 8 — autoregressive shape generation; gray = seed points, color = generated")
fig.tight_layout()
fig.savefig("exp8_rollouts.png", dpi=130)
print("saved exp8_rollouts.png")
