import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, GradPredictor, Net, evaluate, make_batch
from exp3_normdni import true_step

torch.manual_seed(0)

def boot_step(net, sgs, opt, sgopt, rms, x, y, beta=0.95):
    opt.zero_grad(set_to_none=True)
    sgopt.zero_grad(set_to_none=True)
    h, hs = x, []
    in_grads = [None] * len(net.stages)
    for j, (s, sg) in enumerate(zip(net.stages, sgs)):
        h_in = h.detach().requires_grad_(j > 0)
        h_out = s(h_in)
        with torch.no_grad():
            g = sg(h_out, y) * (rms[j] if rms[j] is not None else 1.0)
        h_out.backward(g)
        if j > 0:
            in_grads[j - 1] = h_in.grad.detach()
        hs.append(h_out.detach())
        h = hs[-1]

    h_top = h.requires_grad_(True)
    loss = F.cross_entropy(net.head(h_top.mean(1)), y)
    loss.backward()
    in_grads[-1] = h_top.grad.detach()

    sg_loss = 0.0
    for j, (sg, h_j, tgt) in enumerate(zip(sgs, hs, in_grads)):
        r = tgt.pow(2).mean().sqrt().clamp_min(1e-12)
        rms[j] = r if rms[j] is None else beta * rms[j] + (1 - beta) * r
        sg_loss = sg_loss + F.mse_loss(sg(h_j, y), tgt / rms[j])
    sg_loss.backward()
    opt.step()
    sgopt.step()
    return loss.item()

def run(mode, steps=4000, warmup=600, refresh=5, B=128, T=32, lr=1e-3, seed=1):
    torch.manual_seed(seed)
    net = Net(T=T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(64) for _ in net.stages]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
    rms = [None] * len(sgs)
    hist = {"step": [], "acc": [], "cos": [], "t": []}
    t0, last_cos, n_true = time.perf_counter(), [float("nan")], 0
    for i in range(steps):
        x, y = make_batch(B, T)
        use_true = (mode == "backprop") or i < warmup or \
                   (refresh > 0 and i % refresh == 0)
        if use_true:
            _, last_cos = true_step(net, sgs, opt, sgopt, rms, x, y)
            n_true += 1
        else:
            boot_step(net, sgs, opt, sgopt, rms, x, y)
        if (i + 1) % 100 == 0:
            hist["step"].append(i + 1)
            hist["acc"].append(evaluate(net))
            hist["cos"].append(sum(last_cos) / len(last_cos))
            hist["t"].append(time.perf_counter() - t0)
    name = mode if mode == "backprop" else f"boot refresh={refresh if refresh > 0 else 'never'}"
    print(f"[{name:22}] final acc {hist['acc'][-1]:.3f}  grad-cos {hist['cos'][-1]:.3f}  "
          f"backprop on {n_true / steps:.0%} of steps")
    return hist

if __name__ == "__main__":
    out = {"backprop": run("backprop")}
    for r in [10, 50, 0]:
        out[f"boot-r{r}"] = run("boot", refresh=r)
    out["boot-nowarm"] = run("boot", warmup=0, refresh=0)
    with open("exp4_results.json", "w") as f:
        json.dump(out, f)
