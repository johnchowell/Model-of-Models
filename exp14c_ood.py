import json
import math

import torch
import torch.nn.functional as F

import exp14_sinusoid as e

torch.manual_seed(0)

def sample_tasks_range(B, n, lo, hi, device=e.DEV):
    A = torch.rand(B, 1, device=device) * (hi - lo) + lo
    ph = torch.rand(B, 1, device=device) * math.pi
    x = torch.rand(B, n, device=device) * 10 - 5
    return x, A * torch.sin(x - ph)

@torch.no_grad()
def eval_mom_range(net, lo, hi, ntask=2000, B=500):
    tot, n = 0.0, 0
    for _ in range(ntask // B):
        x, y = sample_tasks_range(B, e.K + e.NTEST, lo, hi)
        pred = net.predict(x[:, :e.K], y[:, :e.K], x[:, e.K:])
        tot += F.mse_loss(pred, y[:, e.K:], reduction="sum").item()
        n += B * e.NTEST
    return tot / n

def eval_maml_range(net, adapted, eval_steps, lo, hi, ntask=2000):
    from torch.func import functional_call, vmap

    def task_mse(params, cx, cy, qx, qy):
        p = adapted(params, cx, cy, eval_steps)
        return F.mse_loss(functional_call(net, p, (qx.unsqueeze(0),)).squeeze(0), qy)

    x, y = sample_tasks_range(ntask, e.K + e.NTEST, lo, hi)
    params = {k: v.detach() for k, v in net.named_parameters()}
    return vmap(task_mse, in_dims=(None, 0, 0, 0, 0))(
        params, x[:, :e.K], y[:, :e.K], x[:, e.K:], y[:, e.K:]).mean().item()

if __name__ == "__main__":
    moms = {}
    for mode in ["direct", "film"]:
        moms[mode] = e.MoM(mode).to(e.DEV)
        moms[mode].load_state_dict(torch.load(f"exp14_mom_{mode}.pt", map_location=e.DEV))
        moms[mode].eval()
    maml, adapted, _ = e.maml_train(seed=1)
    out = {}
    for name, (lo, hi) in [("in-dist A:[0.1,5]", (0.1, 5.0)), ("near-OOD A:[5,6]", (5.0, 6.0)),
                           ("far-OOD A:[6,7]", (6.0, 7.0))]:
        row = {f"mom_{m}": eval_mom_range(n, lo, hi) for m, n in moms.items()}
        row["maml_5step"] = eval_maml_range(maml, adapted, 5, lo, hi)
        row["maml_20step"] = eval_maml_range(maml, adapted, 20, lo, hi)
        out[name] = row
        print(f"{name:20} " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
    with open("exp14c_results.json", "w") as f:
        json.dump(out, f)
