import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, NCLASS, GradPredictor, Net, make_batch
from exp3_normdni import true_step
from exp6_depth import boot_step, opt_params

torch.manual_seed(0)

def measure(fn, iters=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[len(ts) // 2], torch.cuda.max_memory_allocated() / 2**20

def setup(layers, d=256, heads=8, T=256):
    net = Net(d, heads, layers, T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(d) for _ in net.stages]).to(DEV)
    auxs = [nn.Linear(d, NCLASS).to(DEV) for _ in net.stages[:-1]] + [None]
    opt = torch.optim.Adam(opt_params(net, auxs), lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=1e-3)
    return net, sgs, auxs, opt, sgopt, [None] * len(sgs)

if __name__ == "__main__":
    res = {"depth": [], "bp_mb": [], "boot_mb": [], "bp_ms": [], "boot_ms": [],
           "infer_ms": [], "infer_mb": []}
    B, T = 64, 256
    x, y = make_batch(B, T)
    for L in [4, 8, 16, 24]:
        net, sgs, auxs, opt, sgopt, rms = setup(L)
        with torch.no_grad():
            ims, imb = measure(lambda: net(x))
        bms, bmb = measure(lambda: true_step(net, sgs, opt, sgopt, rms, x, y))
        sms, smb = measure(lambda: boot_step(net, sgs, auxs, opt, sgopt, rms, x, y))
        res["depth"].append(L)
        res["infer_ms"].append(ims); res["infer_mb"].append(imb)
        res["bp_ms"].append(bms); res["bp_mb"].append(bmb)
        res["boot_ms"].append(sms); res["boot_mb"].append(smb)
        print(f"L={L:2d}  infer {ims:6.0f}ms/{imb:5.0f}MB   "
              f"backprop {bms:6.0f}ms/{bmb:5.0f}MB   "
              f"self-train {sms:6.0f}ms/{smb:5.0f}MB   mem ratio {bmb / smb:.1f}x")
        del net, sgs, auxs, opt, sgopt
        torch.cuda.empty_cache()
    with open("final_bench.json", "w") as f:
        json.dump(res, f)
