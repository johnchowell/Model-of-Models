import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("exp1_results.json"))
modes = {"backprop": ("tab:blue", "backprop (every step)"),
         "dni": ("tab:orange", "predicted grads, refresh 1/10"),
         "dni-pure": ("tab:red", "predicted grads only after warmup")}

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

for m, (c, lab) in modes.items():
    h = res[m]
    ax[0].plot(h["step"], h["acc"], color=c, label=lab)
    ax[1].plot(h["t"], h["acc"], color=c, label=lab)
ax[0].set(xlabel="step", ylabel="eval accuracy", title="Accuracy vs steps")
ax[0].axhline(1 / 8, ls=":", c="gray"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set(xlabel="wall-clock (s)", ylabel="eval accuracy", title="Accuracy vs time")
ax[1].grid(alpha=0.3)

for m in ["dni", "dni-pure"]:
    h = res[m]
    ax[2].plot(h["step"], h["cos"], color=modes[m][0], label=modes[m][1])
ax[2].set(xlabel="step", ylabel="cosine(predicted, true grad)",
          title="Gradient-prediction quality")
ax[2].axhline(0, ls=":", c="gray"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

b = res["bench"]
names = ["forward_only", "synthetic_step", "backprop_step"]
labels = ["forward\nonly", "train w/ predicted\ngrads (no backprop)", "train w/ full\nbackprop"]
x = range(len(names))
ms = [b[n]["ms"] for n in names]
mb = [b[n]["peak_mb"] for n in names]
ax[3].bar([i - 0.2 for i in x], ms, 0.4, color="tab:purple", label="ms / step")
ax3b = ax[3].twinx()
ax3b.bar([i + 0.2 for i in x], mb, 0.4, color="tab:green", label="peak MB")
ax[3].set_xticks(list(x)); ax[3].set_xticklabels(labels, fontsize=8)
ax[3].set_ylabel("ms / step", color="tab:purple")
ax3b.set_ylabel("peak memory (MB)", color="tab:green")
ax[3].set_title("Scaled model (8L, d=256, T=256): time & memory")
for i, (a, c) in enumerate(zip(ms, mb)):
    ax[3].text(i - 0.2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
    ax3b.text(i + 0.2, c, f"{c:.0f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Exp 1 — transformer on vector shapes, trained by its own predicted per-token gradients")
fig.tight_layout()
fig.savefig("exp1_results.png", dpi=130)
print("saved exp1_results.png")
