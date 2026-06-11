import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, Block, Embed, make_batch

torch.manual_seed(0)

class CausalBlock(Block):
    def forward(self, h):
        T = h.shape[1]
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        a = self.n1(h)
        h = h + self.attn(a, a, a, attn_mask=m, need_weights=False)[0]
        return h + self.mlp(self.n2(h))

class ARNet(nn.Module):
    def __init__(self, d=64, heads=4, layers=4, T=32):
        super().__init__()
        self.stages = nn.ModuleList([Embed(d, T)] + [CausalBlock(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        h = x
        for s in self.stages:
            h = s(h)
        return self.head(h)

class CondGradPredictor(nn.Module):
    def __init__(self, d, cond=2, hidden=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d + cond, hidden), nn.GELU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, h, c):
        return self.f(torch.cat([h, c], -1))

def ar_loss(pred, x):
    return F.mse_loss(pred[:, :-1], x[:, 1:])

def targets_of(x):

    return torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], 1)

def upd_rms(rms, j, tgt, beta=0.95):
    r = tgt.pow(2).mean().sqrt().clamp_min(1e-12)
    rms[j] = r if rms[j] is None else beta * rms[j] + (1 - beta) * r

def bp_step(net, opt, x):
    opt.zero_grad(set_to_none=True)
    loss = ar_loss(net(x), x)
    loss.backward()
    opt.step()
    return loss.item()

def self_step(net, sgs, auxs, opt, sgopt, rms, x, clip=1.0, aux_w=0.3):
    c = targets_of(x)
    opt.zero_grad(set_to_none=True)
    sgopt.zero_grad(set_to_none=True)
    h = x
    for j, (s, sg) in enumerate(zip(net.stages, sgs)):
        h_in = h.detach().requires_grad_(j > 0)
        h_out = s(h_in)
        with torch.no_grad():
            g = sg(h_out, c) * (rms[j] if rms[j] is not None else 1.0)
        if auxs[j] is not None:
            aux_loss = ar_loss(auxs[j](h_out), x) * aux_w
            torch.autograd.backward([h_out, aux_loss], [g, torch.ones_like(aux_loss)])
        else:
            h_out.backward(g)
        if j > 0:
            tgt = h_in.grad.detach()
            upd_rms(rms, j - 1, tgt)
            F.mse_loss(sgs[j - 1](h_in.detach(), c), tgt / rms[j - 1]).backward()
        h = h_out.detach()
    h.requires_grad_(True)
    loss = ar_loss(net.head(h), x)
    loss.backward()
    tgt = h.grad.detach()
    upd_rms(rms, len(sgs) - 1, tgt)
    F.mse_loss(sgs[-1](h.detach(), c), tgt / rms[-1]).backward()
    nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    sgopt.step()
    return loss.item()

@torch.no_grad()
def eval_mse(net, nbatch=10, B=256, T=32):
    m, base = 0.0, 0.0
    for _ in range(nbatch):
        x, _ = make_batch(B, T)
        m += ar_loss(net(x), x).item()
        base += F.mse_loss(x[:, :-1], x[:, 1:]).item()
    return m / nbatch, base / nbatch

@torch.no_grad()
def rollout(net, x, seed_len=8):
    h = x[:, :seed_len].clone()
    T = x.shape[1]
    for _ in range(T - seed_len):
        pad = torch.zeros(h.shape[0], T - h.shape[1], 2, device=h.device)
        pred = net(torch.cat([h, pad], 1))
        h = torch.cat([h, pred[:, h.shape[1] - 1: h.shape[1]]], 1)
    return h

def run(mode, layers=4, steps=8000, B=128, T=32, seed=1):
    torch.manual_seed(seed)
    net = ARNet(layers=layers, T=T).to(DEV)
    sgs = nn.ModuleList([CondGradPredictor(64) for _ in net.stages]).to(DEV)
    auxs = [nn.Linear(64, 2).to(DEV) for _ in net.stages[:-1]] + [None]
    ps = list(net.parameters()) + [p for a in auxs if a is not None for p in a.parameters()]
    opt = torch.optim.Adam(ps, lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    rms = [None] * len(sgs)
    hist = {"step": [], "mse": []}
    for i in range(steps):
        x, _ = make_batch(B, T)
        if mode == "backprop":
            bp_step(net, opt, x)
        else:
            self_step(net, sgs, auxs, opt, sgopt, rms, x)
        sched.step()
        if (i + 1) % 200 == 0:
            m, base = eval_mse(net)
            hist["step"].append(i + 1)
            hist["mse"].append(m)
    print(f"[{mode:9}] final MSE {hist['mse'][-1]:.5f}  best {min(hist['mse']):.5f}  "
          f"(copy-last baseline {base:.5f})")
    return net, hist, base

if __name__ == "__main__":
    out = {}
    nets = {}
    for mode in ["backprop", "self"]:
        nets[mode], out[mode], out["baseline"] = run(mode)
    with open("exp8_results.json", "w") as f:
        json.dump(out, f)
    torch.save({m: n.state_dict() for m, n in nets.items()}, "exp8_nets.pt")
