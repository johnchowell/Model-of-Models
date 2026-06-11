import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import exp14_sinusoid as e

e.DEV = "cpu"
_orig = e.sample_tasks
e.sample_tasks = lambda B, n, device="cpu": _orig(B, n, device="cpu")
torch.manual_seed(0)

def load(mode):
    net = e.MoM(mode).to("cpu")
    net.load_state_dict(torch.load(f"exp14_mom_{mode}.pt", map_location="cpu"))
    net.eval()
    return net

@torch.no_grad()
def matched_vs_shuffled(net, ntask=4000, B=500):
    m_err = s_err = n = 0
    for _ in range(ntask // B):
        x, y = e.sample_tasks(B, e.K + e.NTEST)
        perm = torch.randperm(B)

        pm = net.predict(x[:, :e.K], y[:, :e.K], x[:, e.K:])
        m_err += F.mse_loss(pm, y[:, e.K:], reduction="sum").item()

        ps = net.predict(x[perm, :e.K], y[perm, :e.K], x[:, e.K:])
        s_err += F.mse_loss(ps, y[:, e.K:], reduction="sum").item()
        n += B * e.NTEST
    return m_err / n, s_err / n

@torch.no_grad()
def specialist_swap(net, ntask=4000, B=500):
    aa = ab = n = 0
    for _ in range(ntask // B):
        xa, ya = e.sample_tasks(B, e.K + e.NTEST)
        xb, yb = e.sample_tasks(B, e.K + e.NTEST)

        pa = net.predict(xa[:, :e.K], ya[:, :e.K], xa[:, e.K:])
        aa += F.mse_loss(pa, ya[:, e.K:], reduction="sum").item()

        pb = net.predict(xb[:, :e.K], yb[:, :e.K], xa[:, e.K:])
        ab += F.mse_loss(pb, ya[:, e.K:], reduction="sum").item()
        n += B * e.NTEST
    return aa / n, ab / n

@torch.no_grad()
def context_sweep(net, Ks=(1, 2, 3, 5, 10, 20), ntask=2000, B=500):
    out = {}
    for K in Ks:
        old = e.K
        e.K = K
        err = n = 0
        for _ in range(ntask // B):
            x, y = e.sample_tasks(B, K + e.NTEST)
            p = net.predict(x[:, :K], y[:, :K], x[:, K:])
            err += F.mse_loss(p, y[:, K:], reduction="sum").item()
            n += B * e.NTEST
        out[K] = err / n
        e.K = old
    return out

if __name__ == "__main__":
    res = {}

    x, y = e.sample_tasks(2000, e.K + e.NTEST)
    var = y[:, e.K:].var().item()
    res["target_variance"] = var
    print(f"target variance (prior/chance MSE ~ {var:.3f})", flush=True)
    for mode in ["direct", "film"]:
        net = load(mode)
        m, s = matched_vs_shuffled(net)
        aa, ab = specialist_swap(net)
        sweep = context_sweep(net)
        res[mode] = {"matched": m, "shuffled": s, "swap_self": aa, "swap_other": ab,
                     "context_sweep": sweep}
        print(f"\n[{mode}]")
        print(f"  matched ctx {m:.4f}  |  shuffled ctx {s:.4f}  "
              f"(ratio {s / m:.0f}x, vs variance {var:.3f})")
        print(f"  swap: A-spec/A-tgt {aa:.4f}  |  B-spec/A-tgt {ab:.4f}  "
              f"(ratio {ab / aa:.0f}x)")
        print(f"  context-size sweep MSE: " +
              "  ".join(f"K={k}:{v:.4f}" for k, v in sweep.items()))
    with open("exp20_results.json", "w") as f:
        json.dump(res, f)
