import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, NCLASS, Net, evaluate, make_batch

torch.manual_seed(0)

def greedy_step(net, auxs, opt, x, y, clip=1.0, aux_w=0.3):
    opt.zero_grad(set_to_none=True)
    h = x
    for s, aux in zip(net.stages, auxs):
        h_out = s(h.detach())
        (F.cross_entropy(aux(h_out.mean(1)), y) * aux_w).backward()
        h = h_out.detach()
    h.requires_grad_(True)
    loss = F.cross_entropy(net.head(h.mean(1)), y)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    return loss.item()

def run(layers, steps=6000, B=128, T=32, seed=1):
    torch.manual_seed(seed)
    net = Net(d=64, layers=layers, T=T).to(DEV)
    auxs = [nn.Linear(64, NCLASS).to(DEV) for _ in net.stages]
    ps = list(net.parameters()) + [p for a in auxs for p in a.parameters()]
    opt = torch.optim.Adam(ps, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    accs = []
    for i in range(steps):
        x, y = make_batch(B, T)
        greedy_step(net, auxs, opt, x, y)
        sched.step()
        if (i + 1) % 200 == 0:
            accs.append(evaluate(net))
    print(f"[aux-only {layers}L] final acc {accs[-1]:.3f}  best {max(accs):.3f}  "
          f"(exp6 SG+aux 4L: 0.999, 6L: 0.983)")
    return accs

if __name__ == "__main__":
    out = {f"L{L}": run(L) for L in [4, 6]}
    with open("exp10_results.json", "w") as f:
        json.dump(out, f)
