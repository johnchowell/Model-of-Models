"""Generate paper figures: mechanism ablation, scaling curve. Run from paper/."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

ab = json.load(open("../exp20_results.json"))
sc = json.load(open("../exp23_results.json"))

# ---- Fig: mechanism ablation (two panels) ----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 3.6))

# panel A: matched vs shuffled vs swap (log scale), direct variant
d = ab["direct"]
cats = ["matched\ncontext", "shuffled\ncontext", "specialist\nswap (B on A)"]
vals = [d["matched"], d["shuffled"], d["swap_other"]]
bars = axa.bar(cats, vals, color=["tab:green", "tab:red", "tab:red"])
axa.axhline(ab["target_variance"], ls="--", c="gray", lw=1)
axa.text(2.4, ab["target_variance"] * 1.1, "target variance\n(predict nothing)",
         fontsize=8, color="gray", ha="right")
axa.set_yscale("log")
axa.set_ylabel("MSE (log scale)")
axa.set_title("(a) Specialist is task-conditioned")
for bar, v in zip(bars, vals):
    axa.text(bar.get_x() + bar.get_width() / 2, v * 1.3, f"{v:.4f}",
             ha="center", fontsize=8)

# panel B: context-size sweep, both variants
for mode, c in [("direct", "tab:purple"), ("film", "tab:green")]:
    ks = sorted(int(k) for k in ab[mode]["context_sweep"])
    vs = [ab[mode]["context_sweep"][str(k)] for k in ks]
    axb.plot(ks, vs, "o-", c=c, label=f"MoM-{mode}")
axb.axvline(2, ls=":", c="gray", lw=1)
axb.text(2.1, 0.5, "2 latent\nparams", fontsize=8, color="gray")
axb.set_yscale("log")
axb.set_xlabel("context size $K$ (examples)")
axb.set_ylabel("query MSE (log scale)")
axb.set_title("(b) Accuracy scales with information given")
axb.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig_ablation.pdf")
print("saved fig_ablation.pdf")

# ---- Fig: scaling curve ----
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
P = [r["params"] / 1e6 for r in sc]
zs = [r["zero_shot"] for r in sc]
ic = [r["in_context"] for r in sc]
hl = [r["hyper_lora"] for r in sc]
cap = [r["pct_capture"] for r in sc]

ax1.plot(P, zs, "o-", c="gray", label="zero-shot")
ax1.plot(P, ic, "o-", c="tab:blue", label="in-context")
ax1.plot(P, hl, "o-", c="tab:purple", label="MoM hyper-LoRA")
ax1.set_xscale("log")
ax1.set_xlabel("base model params (M)")
ax1.set_ylabel("bits / char")
ax1.set_title("(a) enwik8 conditioning vs scale")
ax1.set_xticks(P); ax1.set_xticklabels([f"{p:.0f}" if p >= 1 else f"{p:.1f}" for p in P])
ax1.legend(frameon=False)

# capture % excluding degenerate 1M point
ax2.plot(P[1:], cap[1:], "s-", c="tab:purple", ms=8)
ax2.scatter([P[0]], [cap[0]], facecolors="none", edgecolors="tab:red", s=70, zorder=3)
ax2.annotate("degenerate\n(ICL gain ~0)", (P[0], cap[0]), fontsize=8, color="tab:red",
             xytext=(P[0] * 1.5, cap[0] - 40), arrowprops=dict(arrowstyle="->", color="tab:red"))
ax2.set_xscale("log")
ax2.set_xlabel("base model params (M)")
ax2.set_ylabel("\\% of in-context gain captured")
ax2.set_title("(b) Gap to in-context widens with scale")
ax2.set_xticks(P); ax2.set_xticklabels([f"{p:.0f}" if p >= 1 else f"{p:.1f}" for p in P])
for p, cc in zip(P[1:], cap[1:]):
    ax2.text(p, cc + 4, f"{cc:.0f}\\%", ha="center", fontsize=9)
fig2.tight_layout()
fig2.savefig("fig_scaling.pdf")
print("saved fig_scaling.pdf")
