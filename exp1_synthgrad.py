import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda"
NCLASS = 8
CLASS_NAMES = ["circle", "square", "triangle", "line", "sine", "spiral", "zigzag", "rose"]

def poly_walk(t, V, closed=True):

    k = V.shape[0]
    n = k if closed else k - 1
    s = t * n
    i = s.floor().long().clamp(max=n - 1)
    f = (s - i.float()).unsqueeze(-1)
    V2 = torch.cat([V, V[:1]], 0) if closed else V
    return V2[i] * (1 - f) + V2[i + 1] * f

def make_batch(B, T, device=DEV, noise=0.03):
    t = torch.linspace(0, 1, T, device=device).expand(B, T)
    two_pi = 2 * math.pi
    sq = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float, device=device)
    tri = torch.tensor([[0, 1], [-0.87, -0.5], [0.87, -0.5]], dtype=torch.float, device=device)
    zz = torch.tensor([[-1, -0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0.5], [1, -0.5]],
                      dtype=torch.float, device=device)
    shapes = torch.stack([
        torch.stack([torch.cos(two_pi * t), torch.sin(two_pi * t)], -1),
        poly_walk(t, sq),
        poly_walk(t, tri),
        torch.stack([2 * t - 1, torch.zeros_like(t)], -1),
        torch.stack([2 * t - 1, 0.6 * torch.sin(two_pi * 2 * t)], -1),
        torch.stack([t * torch.cos(2 * two_pi * t), t * torch.sin(2 * two_pi * t)], -1),
        poly_walk(t, zz, closed=False),
        torch.stack([(0.5 + 0.5 * torch.cos(5 * two_pi * t)) * torch.cos(two_pi * t),
                     (0.5 + 0.5 * torch.cos(5 * two_pi * t)) * torch.sin(two_pi * t)], -1),
    ])
    y = torch.randint(0, NCLASS, (B,), device=device)
    pts = shapes.gather(0, y.view(1, B, 1, 1).expand(1, B, T, 2)).squeeze(0)
    th = torch.rand(B, device=device) * two_pi
    c, s = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    scale = 0.5 + torch.rand(B, 1, 1, device=device)
    shift = (torch.rand(B, 1, 2, device=device) - 0.5) * 0.5
    pts = torch.einsum("btd,bde->bte", pts, R) * scale + shift
    return pts + noise * torch.randn_like(pts), y

class Embed(nn.Module):
    def __init__(self, d, T):
        super().__init__()
        self.proj = nn.Linear(2, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)

    def forward(self, x):
        return self.proj(x) + self.pos

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        a = self.n1(h)
        h = h + self.attn(a, a, a, need_weights=False)[0]
        return h + self.mlp(self.n2(h))

class GradPredictor(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d + NCLASS, hidden), nn.GELU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, h, y):
        lab = F.one_hot(y, NCLASS).float().unsqueeze(1).expand(-1, h.shape[1], -1)
        return self.f(torch.cat([h, lab], -1))

class Net(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, T=32):
        super().__init__()
        self.stages = nn.ModuleList([Embed(d, T)] + [Block(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, NCLASS)

    def forward(self, x):
        h = x
        for s in self.stages:
            h = s(h)
        return self.head(h.mean(1))

def true_step(net, sgs, opt, sgopt, x, y, train_sg=True):
    opt.zero_grad(set_to_none=True)
    hs, h = [], x
    for s in net.stages:
        h = s(h)
        h.retain_grad()
        hs.append(h)
    loss = F.cross_entropy(net.head(h.mean(1)), y)
    loss.backward()
    opt.step()
    cos = float("nan")
    if train_sg:
        sgopt.zero_grad(set_to_none=True)
        sg_loss, coss = 0.0, []
        for sg, h in zip(sgs, hs):
            pred, tgt = sg(h.detach(), y), h.grad.detach()
            sg_loss = sg_loss + F.mse_loss(pred, tgt)
            coss.append(F.cosine_similarity(pred.flatten(1), tgt.flatten(1), -1).mean())
        sg_loss.backward()
        sgopt.step()
        cos = torch.stack(coss).mean().item()
    return loss.item(), cos

def synth_step(net, sgs, opt, x, y):
    opt.zero_grad(set_to_none=True)
    h = x
    for s, sg in zip(net.stages, sgs):
        h = s(h)
        with torch.no_grad():
            g = sg(h, y)
        h.backward(g)
        h = h.detach()
    h.requires_grad_(True)
    loss = F.cross_entropy(net.head(h.mean(1)), y)
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def evaluate(net, nbatch=10, B=256, T=32):
    acc = 0.0
    for _ in range(nbatch):
        x, y = make_batch(B, T)
        acc += (net(x).argmax(-1) == y).float().mean().item()
    return acc / nbatch

def run(mode, steps=3000, warmup=400, refresh=10, B=128, T=32, lr=1e-3, seed=1):
    torch.manual_seed(seed)
    net = Net(T=T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(64) for _ in net.stages]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=lr)
    hist = {"step": [], "acc": [], "cos": [], "t": []}
    t0, last_cos = time.perf_counter(), float("nan")
    for i in range(steps):
        x, y = make_batch(B, T)
        if mode == "backprop":
            use_true = True
        elif mode == "dni":
            use_true = i < warmup or i % refresh == 0
        elif mode == "dni-pure":
            use_true = i < warmup
        if use_true:
            _, c = true_step(net, sgs, opt, sgopt, x, y, train_sg=(mode != "backprop"))
            if c == c:
                last_cos = c
        else:
            synth_step(net, sgs, opt, x, y)
        if (i + 1) % 100 == 0:
            hist["step"].append(i + 1)
            hist["acc"].append(evaluate(net))
            hist["cos"].append(last_cos)
            hist["t"].append(time.perf_counter() - t0)
    frac_bp = 1.0 if mode == "backprop" else (warmup + (steps - warmup) / refresh) / steps \
        if mode == "dni" else warmup / steps
    print(f"[{mode}] final acc {hist['acc'][-1]:.3f}  grad-cos {last_cos:.3f}  "
          f"backprop on {frac_bp:.0%} of steps  ({hist['t'][-1]:.0f}s)")
    return hist

def benchmark(d=256, heads=8, layers=8, T=256, B=64, iters=30):
    net = Net(d, heads, layers, T).to(DEV)
    sgs = nn.ModuleList([GradPredictor(d) for _ in net.stages]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sgopt = torch.optim.Adam(sgs.parameters(), lr=1e-3)
    res = {}

    def measure(fn, name):
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

    x, y = make_batch(B, T)
    measure(lambda: net(x), "forward_only")
    measure(lambda: true_step(net, sgs, opt, sgopt, x, y, train_sg=False), "backprop_step")
    measure(lambda: synth_step(net, sgs, opt, x, y), "synthetic_step")
    for k, v in res.items():
        print(f"[bench {k:>14}] {v['ms']:7.1f} ms   peak {v['peak_mb']:7.0f} MB")
    return res

if __name__ == "__main__":
    print("== learning curves (tiny model: 2 layers, d=64, T=32) ==")
    out = {m: run(m) for m in ["backprop", "dni", "dni-pure"]}
    print("\n== speed/memory benchmark (8 layers, d=256, T=256, B=64) ==")
    out["bench"] = benchmark()
    with open("exp1_results.json", "w") as f:
        json.dump(out, f)
