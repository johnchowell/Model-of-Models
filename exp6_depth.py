import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, NCLASS, Block, GradPredictor, Net, evaluate, make_batch

torch.manual_seed(0)

def boot_step(net, sgs, auxs, opt, sgopt, rms, x, y, clip=1.0, aux_w=0.3):
    opt.zero_grad(set_to_none=True)
    sgopt.zero_grad(set_to_none=True)
    h, hs = x, []
    in_grads = [None] * len(net.stages)
    for j, (s, sg) in enumerate(zip(net.stages, sgs)):
        h_in = h.detach().requires_grad_(j > 0)
        h_out = s(h_in)
        with torch.no_grad():
            g = sg(h_out, y) * (rms[j] if rms[j] is not None else 1.0)
        if auxs[j] is not None:
            aux_loss = F.cross_entropy(auxs[j](h_out.mean(1)), y) * aux_w
            torch.autograd.backward([h_out, aux_loss], [g, torch.ones_like(aux_loss)])
        else:
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
        rms[j] = r if rms[j] is None else 0.95 * rms[j] + 0.05 * r
        sg_loss = sg_loss + F.mse_loss(sg(h_j, y), tgt / rms[j])
    sg_loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    sgopt.step()
    return loss.item()

def make_auxs(n_stages, every):
    auxs = [None] * n_stages
    if every:
        j = n_stages - 1 - every
        while j >= 0:
            auxs[j] = nn.Linear(64, NCLASS).to(DEV)
            j -= every
    return auxs

def opt_params(net, auxs):
    ps = list(net.parameters())
    for a in auxs:
        if a is not None:
            ps += list(a.parameters())
    return ps

def run(name, layers=4, steps=6000, aux_every=0, B=128, T=32, seed=1):
    torch.manual_seed(seed)
    net = Net(d=64, layers=layers, T=T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(64) for _ in net.stages]).to(DEV)
    auxs = make_auxs(len(net.stages), aux_every)
    opt = torch.optim.Adam(opt_params(net, auxs), lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    rms = [None] * len(sgs)
    hist = {"step": [], "acc": []}
    for i in range(steps):
        x, y = make_batch(B, T)
        boot_step(net, sgs, auxs, opt, sgopt, rms, x, y)
        sched.step()
        if (i + 1) % 200 == 0:
            hist["step"].append(i + 1)
            hist["acc"].append(evaluate(net))
    print(f"[{name:18}] final acc {hist['acc'][-1]:.3f}  best {max(hist['acc']):.3f}")
    return hist

def run_progressive(name, total_layers=4, steps_per_phase=2000, B=128, T=32, seed=1,
                    aux_every=0):
    torch.manual_seed(seed)
    net = Net(d=64, layers=2, T=T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(64) for _ in net.stages]).to(DEV)
    rms = [None] * len(sgs)
    hist = {"step": [], "acc": []}
    done = 0
    for phase in range(total_layers - 1):
        if phase > 0:
            net.stages.append(Block(64, 4).to(DEV))
            sgs.append(GradPredictor(64).to(DEV))
            rms.append(None)
        auxs = make_auxs(len(net.stages), aux_every)
        opt = torch.optim.Adam(opt_params(net, auxs), lr=1e-3)
        sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_per_phase, eta_min=1e-4)
        for i in range(steps_per_phase):
            x, y = make_batch(B, T)
            boot_step(net, sgs, auxs, opt, sgopt, rms, x, y)
            sched.step()
            done += 1
            if done % 200 == 0:
                hist["step"].append(done)
                hist["acc"].append(evaluate(net))
    print(f"[{name:18}] final acc {hist['acc'][-1]:.3f}  best {max(hist['acc']):.3f}  "
          f"({len(net.stages) - 1} layers)")
    return hist

if __name__ == "__main__":
    out = {}
    print("== 4-layer rescue, 6k steps, 0% backprop ==")
    out["v0-control"] = run("v0 control", aux_every=0)
    out["v1a-aux1"] = run("v1a aux every 1", aux_every=1)
    out["v1b-aux2"] = run("v1b aux every 2", aux_every=2)
    out["v2-progressive"] = run_progressive("v2 progressive")
    print("== stability check: 2-layer, 10k steps, with fixes ==")
    out["stab-2L"] = run("2L + clip/decay", layers=2, steps=10000)
    with open("exp6_results.json", "w") as f:
        json.dump(out, f)

    best = max((k for k in out if k.startswith("v")), key=lambda k: max(out[k]["acc"]))
    print(f"== winner ({best}) at 6 layers ==")
    if best == "v2-progressive":
        out["winner-6L"] = run_progressive("winner 6L", total_layers=6, steps_per_phase=1500)
    else:
        ae = {"v0-control": 0, "v1a-aux1": 1, "v1b-aux2": 2}[best]
        out["winner-6L"] = run("winner 6L", layers=6, steps=9000, aux_every=ae)
    with open("exp6_results.json", "w") as f:
        json.dump(out, f)
