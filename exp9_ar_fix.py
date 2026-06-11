import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, make_batch
from exp8_ar import ARNet, CondGradPredictor, ar_loss, bp_step, eval_mse, targets_of, upd_rms

torch.manual_seed(0)

def self_step(net, sgs, auxs, opt, sgopt, rms, x, clip=1.0, aux_w=0.3,
              err_cond=True, use_sg=True):
    c0 = targets_of(x)
    opt.zero_grad(set_to_none=True)
    sgopt.zero_grad(set_to_none=True)
    h, conds = x, []
    for j, (s, sg) in enumerate(zip(net.stages, sgs)):
        h_in = h.detach().requires_grad_(j > 0)
        h_out = s(h_in)
        aux_mod = auxs[j] if auxs[j] is not None else net.head
        with torch.no_grad():
            err = aux_mod(h_out) - c0 if err_cond else torch.zeros_like(c0)
            c = torch.cat([c0, err], -1)
            conds.append(c)
            g = sg(h_out, c) * (rms[j] if rms[j] is not None else 1.0) if use_sg else None
        if auxs[j] is not None:
            aux_loss = ar_loss(auxs[j](h_out), x) * aux_w
            if use_sg:
                torch.autograd.backward([h_out, aux_loss], [g, torch.ones_like(aux_loss)])
            else:
                aux_loss.backward()
        elif use_sg:
            h_out.backward(g)
        if j > 0 and use_sg and h_in.grad is not None:
            tgt = h_in.grad.detach()
            upd_rms(rms, j - 1, tgt)
            F.mse_loss(sgs[j - 1](h_in.detach(), conds[j - 1]), tgt / rms[j - 1]).backward()
        h = h_out.detach()
    h.requires_grad_(True)
    loss = ar_loss(net.head(h), x)
    loss.backward()
    if use_sg:
        tgt = h.grad.detach()
        upd_rms(rms, len(sgs) - 1, tgt)
        F.mse_loss(sgs[-1](h.detach(), conds[-1]), tgt / rms[-1]).backward()
    nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    sgopt.step()
    return loss.item()

VARIANTS = {
    "v1": dict(err_cond=True, use_sg=True, hidden=128, aux_w=0.3),
    "v2": dict(err_cond=False, use_sg=False, hidden=128, aux_w=0.3),
    "v3": dict(err_cond=True, use_sg=True, hidden=256, aux_w=0.3),
    "v4": dict(err_cond=True, use_sg=True, hidden=128, aux_w=1.0),
}

def run(name, layers=4, steps=8000, B=128, T=32, seed=1):
    cfg = VARIANTS[name]
    torch.manual_seed(seed)
    net = ARNet(layers=layers, T=T).to(DEV)
    sgs = nn.ModuleList([CondGradPredictor(64, cond=4, hidden=cfg["hidden"])
                         for _ in net.stages]).to(DEV)

    auxs = [nn.Linear(64, 2).to(DEV) for _ in net.stages[:-1]] + \
           [None if cfg["use_sg"] else nn.Linear(64, 2).to(DEV)]
    ps = list(net.parameters()) + [p for a in auxs if a is not None for p in a.parameters()]
    opt = torch.optim.Adam(ps, lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    rms = [None] * len(sgs)
    hist = {"step": [], "mse": []}
    for i in range(steps):
        x, _ = make_batch(B, T)
        self_step(net, sgs, auxs, opt, sgopt, rms, x,
                  aux_w=cfg["aux_w"], err_cond=cfg["err_cond"], use_sg=cfg["use_sg"])
        sched.step()
        if (i + 1) % 200 == 0:
            m, base = eval_mse(net)
            hist["step"].append(i + 1)
            hist["mse"].append(m)
    print(f"[{name}] final MSE {hist['mse'][-1]:.5f}  best {min(hist['mse']):.5f}  "
          f"(baseline {base:.5f}, exp8 backprop 0.00223, exp8 self 0.02576)")
    torch.save(net.state_dict(), f"exp9_{name}.pt")
    return hist

if __name__ == "__main__":
    out = {v: run(v) for v in sys.argv[1:]}
    with open(f"exp9_results_{'_'.join(sys.argv[1:])}.json", "w") as f:
        json.dump(out, f)
