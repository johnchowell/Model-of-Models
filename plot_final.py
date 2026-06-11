import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

e5 = json.load(open("exp5_results.json"))
e6 = json.load(open("exp6_results.json"))
fb = json.load(open("final_bench.json"))

fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))

ax[0].plot(e5["bp-10k"]["step"], e5["bp-10k"]["acc"], "k--", label="backprop")
for s in [1, 2, 3]:
    h = e5[f"boot-10k-s{s}"]
    ax[0].plot(h["step"], h["acc"], alpha=0.8, label=f"self-trained seed {s}")
ax[0].set(xlabel="step", ylabel="eval accuracy", title="Pure self-training (0% backprop), 2 layers")
ax[0].axhline(1 / 8, ls=":", c="gray"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

names = ["plain bootstrap\n(exp5)", "+clip only", "+aux every 2", "progressive", "+aux every stage"]
vals4 = [max(e5["boot-L4"]["acc"]), max(e6["v0-control"]["acc"]),
         max(e6["v1b-aux2"]["acc"]), max(e6["v2-progressive"]["acc"]),
         max(e6["v1a-aux1"]["acc"])]
bars = ax[1].bar(range(5), vals4, color=["tab:red", "tab:red", "tab:orange", "tab:orange", "tab:green"])
ax[1].axhline(1.0, ls="--", c="k", lw=0.8); ax[1].axhline(1 / 8, ls=":", c="gray")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(names, fontsize=7.5)
ax[1].set(ylabel="best accuracy", title="4-layer rescue (all 0% backprop)")
for b, v in zip(bars, vals4):
    ax[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
h6 = e6["winner-6L"]
ax[1].text(0.02, 0.93, f"winner at 6 layers: {max(h6['acc']):.1%}", transform=ax[1].transAxes,
           fontsize=9, color="tab:green")

ax[2].plot(fb["depth"], fb["bp_mb"], "o-", c="tab:red", label="backprop step")
ax[2].plot(fb["depth"], fb["boot_mb"], "o-", c="tab:green", label="self-train step")
ax[2].plot(fb["depth"], fb["infer_mb"], "o--", c="gray", label="inference (no_grad)")
ax[2].set(xlabel="transformer layers", ylabel="peak memory (MB)",
          title="Training memory vs depth (d=256, T=256, B=64)")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

ax[3].plot(fb["depth"], fb["bp_ms"], "o-", c="tab:red", label="backprop step")
ax[3].plot(fb["depth"], fb["boot_ms"], "o-", c="tab:green", label="self-train step")
ax[3].plot(fb["depth"], fb["infer_ms"], "o--", c="gray", label="inference (no_grad)")
ax[3].set(xlabel="transformer layers", ylabel="ms / step", title="Step time vs depth")
ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)

fig.suptitle("Transformer on vector shapes trained by its own predicted per-token gradients — zero global backprop")
fig.tight_layout()
fig.savefig("final_results.png", dpi=130)
print("saved final_results.png")
