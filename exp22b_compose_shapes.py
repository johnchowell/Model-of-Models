import json

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import exp11_hypernet as e

e.DEV = "cpu"
import exp1_synthgrad as base
base.DEV = "cpu"
_mb = base.make_batch
base.make_batch = lambda B, T, device="cpu", noise=0.03: _mb(B, T, device="cpu", noise=noise)
e.make_batch = base.make_batch
torch.manual_seed(1)

def load():
    net = e.HyperNet("direct").to("cpu")
    net.load_state_dict(torch.load("exp11_direct.pt", map_location="cpu"))
    net.eval()
    return net

@torch.no_grad()
def quant(net, ntask=1500, B=300, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    t = torch.linspace(0, 1, e.T).expand(B, e.T)
    res = {a: 0.0 for a in alphas}
    ab = 0.0
    n = 0
    for _ in range(ntask // B):
        pa, _, _ = e.batch_with_t(B)
        pb, _, _ = e.batch_with_t(B)
        ta = net.make_specialist(pa[:, :e.CTX], t[:, :e.CTX])
        tb = net.make_specialist(pb[:, :e.CTX], t[:, :e.CTX])
        for al in alphas:
            th = (1 - al) * ta + al * tb
            pred = e.spec_apply_direct(th, t)
            tgt = (1 - al) * pa + al * pb
            res[al] += F.mse_loss(pred, tgt, reduction="sum").item()
        ab += F.mse_loss(pa, pb, reduction="sum").item()
        n += B * e.T * 2
    return {a: v / n for a, v in res.items()}, ab / n

@torch.no_grad()
def morph_figure(net):

    torch.manual_seed(3)
    while True:
        p, _, y = e.batch_with_t(64)
        idx0 = (y == 0).nonzero()
        idx1 = (y == 1).nonzero()
        if len(idx0) and len(idx1):
            break
    ia, ib = idx0[0].item(), idx1[0].item()
    t = torch.linspace(0, 1, e.T).unsqueeze(0)
    ta = net.make_specialist(p[ia:ia + 1, :e.CTX], t[:, :e.CTX])
    tb = net.make_specialist(p[ib:ib + 1, :e.CTX], t[:, :e.CTX])
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(1, len(alphas), figsize=(3 * len(alphas), 3))
    for k, al in enumerate(alphas):
        th = (1 - al) * ta + al * tb
        c = e.spec_apply_direct(th, t)[0].cpu()
        tgt = ((1 - al) * p[ia:ia + 1] + al * p[ib:ib + 1])[0].cpu()
        ax[k].plot(tgt[:, 0], tgt[:, 1], "--", c="lightgray", lw=2, label="morph target")
        ax[k].plot(c[:, 0], c[:, 1], "-", c="tab:purple", lw=1.8, label="weight-blend")
        ax[k].set_title(f"a={al}"); ax[k].set_aspect("equal")
        ax[k].set_xticks([]); ax[k].set_yticks([])
    ax[0].legend(fontsize=7)
    fig.suptitle("Specialist weight interpolation: (1-a)*theta_circle + a*theta_square")
    fig.tight_layout(); fig.savefig("exp22b_morph.png", dpi=130)
    print("saved exp22b_morph.png")

if __name__ == "__main__":
    net = load()
    comp, ab = quant(net)
    print(f"A-vs-B shape distance (MSE) = {ab:.3f}")
    for a, v in comp.items():
        print(f"  a={a:.2f}: weight-blend vs morph MSE {v:.4f}")
    mid = comp[0.5]
    ratio = ab / mid
    print(f"midpoint {mid:.4f} = {ratio:.0f}x below A-B distance "
          f"({'COMPOSES on non-closed family' if mid < 0.2 * ab else 'partial/no compose'})")
    morph_figure(net)
    json.dump({"interp": {str(k): v for k, v in comp.items()}, "ab_distance": ab,
               "midpoint_ratio": ratio}, open("exp22b_results.json", "w"))
