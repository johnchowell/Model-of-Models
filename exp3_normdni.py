import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, GradPredictor, Net, evaluate, make_batch

torch.manual_seed(0)

def fwd_with_grads(net, x, y):
    hs, h = [], x
    for s in net.stages:
        h = s(h); h.retain_grad(); hs.append(h)
    loss = F.cross_entropy(net.head(h.mean(1)), y)
    loss.backward()
    return hs, [h.grad.detach() for h in hs], loss

def true_step(net, sgs, opt, sgopt, rms, x, y, beta=0.95):
    opt.zero_grad(set_to_none=True)
    hs, gs, loss = fwd_with_grads(net, x, y)
    opt.step()
    sgopt.zero_grad(set_to_none=True)
    sg_loss, coss = 0.0, []
    for j, (sg, h, g) in enumerate(zip(sgs, hs, gs)):
        r = g.pow(2).mean().sqrt()
        rms[j] = r if rms[j] is None else beta * rms[j] + (1 - beta) * r
        pred = sg(h.detach(), y)
        sg_loss = sg_loss + F.mse_loss(pred, g / rms[j])
        coss.append(F.cosine_similarity(pred.flatten(1), g.flatten(1), -1).mean())
    sg_loss.backward()
    sgopt.step()
    return loss.item(), [c.item() for c in coss]

def synth_step(net, sgs, opt, rms, x, y):
    opt.zero_grad(set_to_none=True)
    h = x
    for j, (s, sg) in enumerate(zip(net.stages, sgs)):
        h = s(h)
        with torch.no_grad():
            g = sg(h, y) * rms[j]
        h.backward(g)
        h = h.detach()
    h.requires_grad_(True)
    loss = F.cross_entropy(net.head(h.mean(1)), y)
    loss.backward()
    opt.step()
    return loss.item()

def phase_a():
    print("== Phase A: how predictable are grads at each training stage? ==")
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ckpt in [0, 100, 300, 1000]:

        trained = getattr(phase_a, "trained", 0)
        for _ in range(ckpt - trained):
            x, y = make_batch(128, 32)
            opt.zero_grad(); F.cross_entropy(net(x), y).backward(); opt.step()
        phase_a.trained = ckpt

        sgs = nn.ModuleList([GradPredictor(64) for _ in net.stages]).to(DEV)
        sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
        rms = [None] * len(sgs)
        for _ in range(1200):
            x, y = make_batch(128, 32)
            hs, gs, _ = fwd_with_grads(net, x, y)
            net.zero_grad(set_to_none=True)
            sgopt.zero_grad(set_to_none=True)
            l = 0.0
            for j, (sg, h, g) in enumerate(zip(sgs, hs, gs)):
                r = g.pow(2).mean().sqrt()
                rms[j] = r if rms[j] is None else 0.95 * rms[j] + 0.05 * r
                l = l + F.mse_loss(sg(h.detach(), y), g / rms[j])
            l.backward(); sgopt.step()
        with torch.no_grad():
            pass
        x, y = make_batch(256, 32)
        hs, gs, loss = fwd_with_grads(net, x, y)
        net.zero_grad(set_to_none=True)
        with torch.no_grad():
            cs = [F.cosine_similarity(sg(h.detach(), y).flatten(1), g.flatten(1), -1).mean().item()
                  for sg, h, g in zip(sgs, hs, gs)]
        print(f"  net@{ckpt:4d} steps (loss {loss.item():.3f}): cosine per layer "
              + "  ".join(f"{c:.3f}" for c in cs))

def run(mode, steps=4000, warmup=600, refresh=5, B=128, T=32, lr=1e-3, seed=1):
    torch.manual_seed(seed)
    net = Net(T=T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(64) for _ in net.stages]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
    rms = [None] * len(sgs)
    hist = {"step": [], "acc": [], "cos": [], "t": []}
    t0, last_cos = time.perf_counter(), [float("nan")]
    n_true = 0
    for i in range(steps):
        x, y = make_batch(B, T)
        use_true = (mode == "backprop") or i < warmup or \
                   (mode == "dni" and i % refresh == 0)
        if use_true:
            _, last_cos = true_step(net, sgs, opt, sgopt, rms, x, y)
            n_true += 1
        else:
            synth_step(net, sgs, opt, rms, x, y)
        if (i + 1) % 100 == 0:
            hist["step"].append(i + 1)
            hist["acc"].append(evaluate(net))
            hist["cos"].append(sum(last_cos) / len(last_cos))
            hist["t"].append(time.perf_counter() - t0)
    print(f"[{mode} refresh={refresh}] final acc {hist['acc'][-1]:.3f}  "
          f"grad-cos {hist['cos'][-1]:.3f}  backprop on {n_true / steps:.0%} of steps")
    return hist

if __name__ == "__main__":
    phase_a()
    print("\n== Phase B: self-trained loop with normalized predicted grads ==")
    out = {}
    out["backprop"] = run("backprop")
    for r in [2, 5, 10]:
        out[f"dni-r{r}"] = run("dni", refresh=r)
    out["dni-pure"] = run("dni", refresh=10**9)
    with open("exp3_results.json", "w") as f:
        json.dump(out, f)
