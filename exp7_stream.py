import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, NCLASS, GradPredictor, Net, evaluate, make_batch
from exp3_normdni import true_step
from exp6_depth import boot_step, make_auxs, opt_params

torch.manual_seed(0)

def upd_rms(rms, j, tgt, beta=0.95):
    r = tgt.pow(2).mean().sqrt().clamp_min(1e-12)
    rms[j] = r if rms[j] is None else beta * rms[j] + (1 - beta) * r

def stream_step(net, sgs, auxs, opt, sgopt, rms, x, y, clip=1.0, aux_w=0.3):
    opt.zero_grad(set_to_none=True)
    sgopt.zero_grad(set_to_none=True)
    h = x
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
            tgt = h_in.grad.detach()
            upd_rms(rms, j - 1, tgt)
            F.mse_loss(sgs[j - 1](h_in.detach(), y), tgt / rms[j - 1]).backward()
        h = h_out.detach()
    h.requires_grad_(True)
    loss = F.cross_entropy(net.head(h.mean(1)), y)
    loss.backward()
    tgt = h.grad.detach()
    upd_rms(rms, len(sgs) - 1, tgt)
    F.mse_loss(sgs[-1](h.detach(), y), tgt / rms[-1]).backward()
    nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    sgopt.step()
    return loss.item()

def run_parity(layers=6, steps=9000, B=128, T=32, seed=1):
    torch.manual_seed(seed)
    net = Net(d=64, layers=layers, T=T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(64) for _ in net.stages]).to(DEV)
    auxs = make_auxs(len(net.stages), 1)
    opt = torch.optim.Adam(opt_params(net, auxs), lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    rms = [None] * len(sgs)
    accs = []
    for i in range(steps):
        x, y = make_batch(B, T)
        stream_step(net, sgs, auxs, opt, sgopt, rms, x, y)
        sched.step()
        if (i + 1) % 300 == 0:
            accs.append(evaluate(net))
    print(f"[stream parity {layers}L] final acc {accs[-1]:.3f}  best {max(accs):.3f}")
    return accs

def bench(iters=15):
    res = {"depth": [], "bp_ms": [], "bp_mb": [], "boot_ms": [], "boot_mb": [],
           "stream_ms": [], "stream_mb": []}
    B, T, d = 64, 256, 256
    x, y = make_batch(B, T)

    def measure(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t = time.perf_counter()
            fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
        return sorted(ts)[len(ts) // 2], torch.cuda.max_memory_allocated() / 2**20

    for L in [8, 24]:
        net = Net(d, 8, L, T).to(DEV)
        sgs = nn.ModuleList([GradPredictor(d) for _ in net.stages]).to(DEV)
        auxs = [nn.Linear(d, NCLASS).to(DEV) for _ in net.stages[:-1]] + [None]
        opt = torch.optim.Adam(opt_params(net, auxs), lr=1e-3)
        sgopt = torch.optim.Adam(sgs.parameters(), lr=1e-3)
        rms = [None] * len(sgs)
        bms, bmb = measure(lambda: true_step(net, sgs, opt, sgopt, rms, x, y))
        oms, omb = measure(lambda: boot_step(net, sgs, auxs, opt, sgopt, rms, x, y))
        sms, smb = measure(lambda: stream_step(net, sgs, auxs, opt, sgopt, rms, x, y))
        res["depth"].append(L)
        res["bp_ms"].append(bms); res["bp_mb"].append(bmb)
        res["boot_ms"].append(oms); res["boot_mb"].append(omb)
        res["stream_ms"].append(sms); res["stream_mb"].append(smb)
        print(f"L={L:2d}  backprop {bms:5.0f}ms/{bmb:5.0f}MB   "
              f"exp6 boot {oms:5.0f}ms/{omb:5.0f}MB   "
              f"streamed {sms:5.0f}ms/{smb:5.0f}MB   mem {bmb / smb:.1f}x vs bp")
        del net, sgs, auxs, opt, sgopt
        torch.cuda.empty_cache()
    return res

if __name__ == "__main__":
    out = {"parity": run_parity()}
    out["bench"] = bench()
    with open("exp7_results.json", "w") as f:
        json.dump(out, f)
