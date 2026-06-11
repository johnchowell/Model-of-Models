import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from exp1_synthgrad import CLASS_NAMES, DEV
from exp8_ar import ARNet, rollout
from exp11_hypernet import CTX, FilmSpecialist, HyperNet, batch_with_t

torch.manual_seed(11)
res = json.load(open("exp11_results.json"))

nets = {}
for m in ["direct", "film"]:
    nets[m] = HyperNet(m).to(DEV)
    nets[m].load_state_dict(torch.load(f"exp11_{m}.pt", map_location=DEV))
    nets[m].eval()
arnet = ARNet().to(DEV)
arnet.load_state_dict(torch.load("exp8_nets.pt", map_location=DEV)["backprop"])
arnet.eval()

pts, t, y = batch_with_t(8)
with torch.no_grad():
    gen = {m: n.generate(pts[:, :CTX], t[:, :CTX], t).cpu() for m, n in nets.items()}
    gen["ar"] = rollout(arnet, pts, CTX).cpu()
ptsc = pts.cpu()

fig = plt.figure(figsize=(19, 9))
gs = fig.add_gridspec(4, 9, width_ratios=[2.2] + [1] * 8)

axl = fig.add_subplot(gs[:2, 0])
for m, c in [("direct", "tab:purple"), ("film", "tab:green")]:
    h = res[m]
    axl.plot(h["step"], h["cont"], c=c, label=f"hyper-{m} (continuation)")
axl.axhline(res["ar_cont_mse"], ls="--", c="tab:blue", label="AR transformer rollout")
axl.axhline(2 * 0.03**2, ls=":", c="gray", label="noise floor")
axl.set(xlabel="step", ylabel="MSE on unseen curve part", yscale="log",
        title="Continuation accuracy")
axl.legend(fontsize=7); axl.grid(alpha=0.3)

axb = fig.add_subplot(gs[2:, 0])
b = res["bench"]
names = ["AR rollout\n(204k params)", "hyper+spec\n(1 enc pass)",
         f"specialist alone\n({b['params']['specialist_direct']} params)",
         f"film specialist\n({b['params']['specialist_film_mods']} mods)"]
vals = [b["ar_rollout_ms"], b["hyper_direct_ms"], b["spec_direct_ms"], b["spec_film_ms"]]
bars = axb.bar(range(4), vals, color=["tab:blue", "tab:purple", "tab:purple", "tab:green"])
for i, v in enumerate(vals):
    axb.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
axb.set_xticks(range(4)); axb.set_xticklabels(names, fontsize=7)
axb.set(ylabel="ms / batch of 256 shapes", title="Generation cost", yscale="log")

rows = [("ground truth", ptsc, "k"), ("AR transformer", gen["ar"], "tab:blue"),
        ("specialist (direct)", gen["direct"], "tab:purple"),
        ("specialist (film)", gen["film"], "tab:green")]
for r, (title, p, c) in enumerate(rows):
    for i in range(8):
        ax = fig.add_subplot(gs[r, i + 1])
        ax.plot(ptsc[i, :, 0], ptsc[i, :, 1], "-", c="lightgray", lw=1)
        ax.scatter(ptsc[i, :CTX, 0], ptsc[i, :CTX, 1], s=4, c="gray")
        ax.plot(p[i, CTX:, 0], p[i, CTX:, 1], "-", c=c, lw=1.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if r == 0:
            ax.set_title(CLASS_NAMES[y[i]], fontsize=8)
        if i == 0:
            ax.set_ylabel(title, fontsize=8)

fig.suptitle("Exp 11 — the model outputs a specialist network per shape: "
             "gray dots = 12 context points, color = the generated continuation")
fig.tight_layout()
fig.savefig("exp11_results.png", dpi=130)
print("saved exp11_results.png")
