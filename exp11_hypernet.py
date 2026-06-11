import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, Block, make_batch

torch.manual_seed(0)
CTX, T, KF, H = 12, 32, 8, 32
NFEAT = 2 * KF + 1

def fourier(t):
    k = torch.arange(1, KF + 1, device=t.device).float() * 2 * math.pi
    a = t.unsqueeze(-1) * k
    return torch.cat([torch.sin(a), torch.cos(a), t.unsqueeze(-1) * 2 - 1], -1)

SHAPES = [(NFEAT, H), (H,), (H, H), (H,), (H, 2), (2,)]
SIZES = [math.prod(s) for s in SHAPES]
NPARAM = sum(SIZES)

def init_theta0():
    parts = []
    for s in SHAPES:
        if len(s) == 2:
            parts.append(torch.randn(s).flatten() / math.sqrt(s[0]))
        else:
            parts.append(torch.zeros(s))
    return torch.cat(parts)

def spec_apply_direct(theta, t):
    chunks = list(theta.split(SIZES, -1))
    W1, b1, W2, b2, W3, b3 = [c.view(c.shape[0], *s) for c, s in zip(chunks, SHAPES)]
    h = fourier(t)
    h = F.gelu(torch.einsum("btk,bkh->bth", h, W1) + b1.unsqueeze(1))
    h = F.gelu(torch.einsum("btk,bkh->bth", h, W2) + b2.unsqueeze(1))
    return torch.einsum("btk,bkh->bth", h, W3) + b3.unsqueeze(1)

class FilmSpecialist(nn.Module):
    NMOD = 2 * (H + H + 2)

    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(NFEAT, H), nn.Linear(H, H), nn.Linear(H, 2)

    def forward(self, mod, t):
        s1, m1, s2, m2, s3, m3 = mod.split([H, H, H, H, 2, 2], -1)
        u = lambda v: v.unsqueeze(1)
        h = F.gelu(self.l1(fourier(t)) * (1 + u(s1)) + u(m1))
        h = F.gelu(self.l2(h) * (1 + u(s2)) + u(m2))
        return self.l3(h) * (1 + u(s3)) + u(m3)

class HyperNet(nn.Module):
    def __init__(self, mode, d=128, layers=3):
        super().__init__()
        self.mode = mode
        self.inp = nn.Linear(3, d)
        self.blocks = nn.ModuleList([Block(d, 8) for _ in range(layers)])
        n_out = NPARAM if mode == "direct" else FilmSpecialist.NMOD
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, n_out))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        if mode == "direct":
            self.theta0 = nn.Parameter(init_theta0())
        else:
            self.spec = FilmSpecialist()

    def make_specialist(self, ctx_pts, ctx_t):
        z = self.inp(torch.cat([ctx_pts, ctx_t.unsqueeze(-1)], -1))
        for b in self.blocks:
            z = b(z)
        out = self.head(z.mean(1))
        return out + self.theta0 if self.mode == "direct" else out

    def generate(self, ctx_pts, ctx_t, t):
        w = self.make_specialist(ctx_pts, ctx_t)
        return spec_apply_direct(w, t) if self.mode == "direct" else self.spec(w, t)

def batch_with_t(B):
    pts, y = make_batch(B, T)
    t = torch.linspace(0, 1, T, device=DEV).expand(B, T)
    return pts, t, y

@torch.no_grad()
def evaluate(net, nbatch=10, B=512):
    full, cont = 0.0, 0.0
    for _ in range(nbatch):
        pts, t, _ = batch_with_t(B)
        pred = net.generate(pts[:, :CTX], t[:, :CTX], t)
        full += F.mse_loss(pred, pts).item()
        cont += F.mse_loss(pred[:, CTX:], pts[:, CTX:]).item()
    return full / nbatch, cont / nbatch

def train(mode, steps=8000, B=256, seed=1):
    torch.manual_seed(seed)
    net = HyperNet(mode).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    hist = {"step": [], "full": [], "cont": []}
    for i in range(steps):
        pts, t, _ = batch_with_t(B)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(net.generate(pts[:, :CTX], t[:, :CTX], t), pts)
        loss.backward()
        opt.step()
        sched.step()
        if (i + 1) % 200 == 0:
            f, c = evaluate(net)
            hist["step"].append(i + 1); hist["full"].append(f); hist["cont"].append(c)
    print(f"[hyper-{mode:6}] full MSE {hist['full'][-1]:.5f}  "
          f"continuation MSE {hist['cont'][-1]:.5f}  (noise floor ~{2 * 0.03**2:.5f})")
    torch.save(net.state_dict(), f"exp11_{mode}.pt")
    return hist

def ar_baseline():
    from exp8_ar import ARNet, rollout
    net = ARNet().to(DEV)
    net.load_state_dict(torch.load("exp8_nets.pt", map_location=DEV)["backprop"])
    net.eval()
    cont = 0.0
    with torch.no_grad():
        for _ in range(10):
            pts, _, _ = batch_with_t(512)
            gen = rollout(net, pts, seed_len=CTX)
            cont += F.mse_loss(gen[:, CTX:], pts[:, CTX:]).item()
    print(f"[AR transformer] continuation MSE {cont / 10:.5f} (rollout)")
    return cont / 10, net

def bench(hnets, arnet, B=256, iters=20):
    pts, t, _ = batch_with_t(B)
    from exp8_ar import rollout
    res = {}

    def measure(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        return sorted(ts)[len(ts) // 2]

    with torch.no_grad():
        res["ar_rollout_ms"] = measure(lambda: rollout(arnet, pts, CTX))
        for m, n in hnets.items():
            res[f"hyper_{m}_ms"] = measure(lambda: n.generate(pts[:, :CTX], t[:, :CTX], t))
            res[f"spec_{m}_ms"] = None

        for m, n in hnets.items():
            w = n.make_specialist(pts[:, :CTX], t[:, :CTX])
            f = (lambda: spec_apply_direct(w, t)) if m == "direct" else (lambda: n.spec(w, t))
            res[f"spec_{m}_ms"] = measure(f)
    ar_params = sum(p.numel() for p in arnet.parameters())
    res["params"] = {"ar_transformer": ar_params, "specialist_direct": NPARAM,
                     "specialist_film_mods": FilmSpecialist.NMOD}
    for k, v in res.items():
        print(k, v)
    return res

if __name__ == "__main__":
    out = {m: train(m) for m in ["direct", "film"]}
    out["ar_cont_mse"], arnet = ar_baseline()
    hnets = {}
    for m in ["direct", "film"]:
        hnets[m] = HyperNet(m).to(DEV)
        hnets[m].load_state_dict(torch.load(f"exp11_{m}.pt", map_location=DEV))
        hnets[m].eval()
    out["bench"] = bench(hnets, arnet)
    with open("exp11_results.json", "w") as f:
        json.dump(out, f)
