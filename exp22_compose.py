import json

import numpy as np
import torch
import torch.nn.functional as F

import exp14_sinusoid as e

e.DEV = "cpu"
_orig = e.sample_tasks
e.sample_tasks = lambda B, n, device="cpu": _orig(B, n, device="cpu")
torch.manual_seed(0)

def load():
    net = e.MoM("direct").to("cpu")
    net.load_state_dict(torch.load("exp14_mom_direct.pt", map_location="cpu"))
    net.eval()
    return net

@torch.no_grad()
def compose(net, ntask=2000, B=400, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    xq = torch.linspace(-5, 5, e.NTEST).expand(B, e.NTEST)
    res = {a: 0.0 for a in alphas}
    blend_vs_self = 0.0
    n = 0
    for _ in range(ntask // B):
        xa, ya = e.sample_tasks(B, e.K + e.NTEST)
        xb, yb = e.sample_tasks(B, e.K + e.NTEST)
        ta = net.emit(xa[:, :e.K], ya[:, :e.K])
        tb = net.emit(xb[:, :e.K], yb[:, :e.K])

        ya_q = a_curve(xa, ya, xq)
        yb_q = a_curve(xb, yb, xq)
        for al in alphas:
            th = (1 - al) * ta + al * tb
            pred = e.spec_apply(th, xq)
            tgt = (1 - al) * ya_q + al * yb_q
            res[al] += F.mse_loss(pred, tgt, reduction="sum").item()

        blend_vs_self += F.mse_loss(ya_q, yb_q, reduction="sum").item()
        n += B * e.NTEST
    return {a: v / n for a, v in res.items()}, blend_vs_self / n

def a_curve(x, y, xq):

    xall = x
    yall = y
    s, c = torch.sin(xall), torch.cos(xall)

    Xm = torch.stack([s, c], -1)
    sol = torch.linalg.lstsq(Xm, yall.unsqueeze(-1)).solution.squeeze(-1)
    a, b = sol[:, :1], sol[:, 1:]
    return a * torch.sin(xq) + b * torch.cos(xq)

if __name__ == "__main__":
    net = load()
    comp, ab_dist = compose(net)
    rng = np.random.RandomState(0)

    print(f"A-vs-B target distance (MSE) = {ab_dist:.3f}  (the scale to beat)")
    print("interpolation a -> MSE(specialist((1-a)tA+a tB), (1-a)yA+a yB):")
    for a, v in comp.items():
        print(f"  a={a:.2f}: {v:.4f}")
    mid = comp[0.5]
    verdict = ("COMPOSES: midpoint weights track blended curve "
               f"({mid:.4f} << A-B dist {ab_dist:.3f})") if mid < 0.2 * ab_dist else \
              (f"does NOT compose cleanly (mid {mid:.4f} vs A-B dist {ab_dist:.3f})")
    print(verdict)
    json.dump({"interp": {str(k): v for k, v in comp.items()},
               "ab_distance": ab_dist, "verdict": verdict},
              open("exp22_results.json", "w"))
