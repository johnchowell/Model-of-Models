import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, GradPredictor, Net, evaluate, make_batch
from exp3_normdni import true_step
from exp4_bootstrap import boot_step

torch.manual_seed(0)

def run(mode, steps, layers=2, d=64, B=128, T=32, seed=1, log=None):
    torch.manual_seed(seed)
    net = Net(d=d, layers=layers, T=T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(d) for _ in net.stages]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=3e-3)
    rms = [None] * len(sgs)
    hist = {"step": [], "acc": [], "t": []}
    t0 = time.perf_counter()
    for i in range(steps):
        x, y = make_batch(B, T)
        if mode == "backprop":
            true_step(net, sgs, opt, sgopt, rms, x, y)
        else:
            boot_step(net, sgs, opt, sgopt, rms, x, y)
        if (i + 1) % 200 == 0:
            hist["step"].append(i + 1)
            hist["acc"].append(evaluate(net))
            hist["t"].append(time.perf_counter() - t0)
    if log:
        print(f"[{log}] final acc {hist['acc'][-1]:.3f}  best {max(hist['acc']):.3f}")
    return hist

def bench(d=256, heads=8, layers=8, T=256, B=64, iters=30):
    net = Net(d, heads, layers, T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(d) for _ in net.stages]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=1e-3)
    rms = [None] * len(sgs)
    x, y = make_batch(B, T)
    res = {}
    for name, fn in [
        ("forward_only", lambda: net(x)),
        ("backprop_step", lambda: true_step(net, sgs, opt, sgopt, rms, x, y)),
        ("boot_step", lambda: boot_step(net, sgs, opt, sgopt, rms, x, y)),
    ]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t = time.perf_counter()
            fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
        res[name] = {"ms": sorted(ts)[len(ts) // 2],
                     "peak_mb": torch.cuda.max_memory_allocated() / 2**20}
        print(f"[bench {name:>14}] {res[name]['ms']:7.1f} ms   peak {res[name]['peak_mb']:7.0f} MB")
    return res

if __name__ == "__main__":
    out = {}
    print("== A: pure self-training vs backprop, 10k steps ==")
    out["bp-10k"] = run("backprop", 10000, log="backprop  seed1")
    for s in [1, 2, 3]:
        out[f"boot-10k-s{s}"] = run("boot", 10000, seed=s, log=f"pure boot seed{s}")
    print("== B: depth scaling, 6k steps ==")
    for L in [2, 4, 6]:
        out[f"bp-L{L}"] = run("backprop", 6000, layers=L, log=f"backprop  {L} layers")
        out[f"boot-L{L}"] = run("boot", 6000, layers=L, log=f"pure boot {L} layers")
    print("== C: speed/memory at scale (8L, d=256, T=256, B=64) ==")
    out["bench"] = bench()
    with open("exp5_results.json", "w") as f:
        json.dump(out, f)
