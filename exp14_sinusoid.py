import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
K, NTEST, H = 10, 100, 40
torch.manual_seed(0)

def sample_tasks(B, n, device=DEV):
    A = torch.rand(B, 1, device=device) * 4.9 + 0.1
    ph = torch.rand(B, 1, device=device) * math.pi
    x = torch.rand(B, n, device=device) * 10 - 5
    return x, A * torch.sin(x - ph)

SHAPES = [(1, H), (H,), (H, H), (H,), (H, 1), (1,)]
SIZES = [math.prod(s) for s in SHAPES]
NPARAM = sum(SIZES)

def theta0():
    parts = []
    for s in SHAPES:
        parts.append((torch.randn(s) / math.sqrt(s[0])).flatten() if len(s) == 2
                     else torch.zeros(s))
    return torch.cat(parts)

def spec_apply(theta, x):
    cs = list(theta.split(SIZES, -1))
    W1, b1, W2, b2, W3, b3 = [c.view(c.shape[0], *s) for c, s in zip(cs, SHAPES)]
    h = x.unsqueeze(-1)
    h = F.relu(torch.einsum("btk,bkh->bth", h, W1) + b1.unsqueeze(1))
    h = F.relu(torch.einsum("btk,bkh->bth", h, W2) + b2.unsqueeze(1))
    return (torch.einsum("btk,bkh->bth", h, W3) + b3.unsqueeze(1)).squeeze(-1)

class FilmSpec(nn.Module):
    NMOD = 2 * (H + H + 1)

    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(1, H), nn.Linear(H, H), nn.Linear(H, 1)

    def forward(self, mod, x):
        s1, m1, s2, m2, s3, m3 = mod.split([H, H, H, H, 1, 1], -1)
        u = lambda v: v.unsqueeze(1)
        h = F.relu(self.l1(x.unsqueeze(-1)) * (1 + u(s1)) + u(m1))
        h = F.relu(self.l2(h) * (1 + u(s2)) + u(m2))
        return (self.l3(h) * (1 + u(s3)) + u(m3)).squeeze(-1)

class MoM(nn.Module):
    def __init__(self, mode, d=128):
        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.post = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        n_out = NPARAM if mode == "direct" else FilmSpec.NMOD
        self.head = nn.Linear(d, n_out)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        if mode == "direct":
            self.t0 = nn.Parameter(theta0())
        else:
            self.spec = FilmSpec()

    def emit(self, cx, cy):
        z = self.enc(torch.stack([cx, cy], -1)).mean(1)
        out = self.head(self.post(z))
        return out + self.t0 if self.mode == "direct" else out

    def predict(self, cx, cy, x):
        w = self.emit(cx, cy)
        return spec_apply(w, x) if self.mode == "direct" else self.spec(w, x)

def train_mom(mode, steps=30000, B=256, lr=1e-3, seed=1):
    torch.manual_seed(seed)
    net = MoM(mode).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    for i in range(steps):
        x, y = sample_tasks(B, K + NTEST)
        opt.zero_grad(set_to_none=True)
        pred = net.predict(x[:, :K], y[:, :K], x[:, K:])
        F.mse_loss(pred, y[:, K:]).backward()
        opt.step()
        sched.step()
    return net

@torch.no_grad()
def eval_mom(net, ntask=2000, B=500):
    tot, n = 0.0, 0
    for _ in range(ntask // B):
        x, y = sample_tasks(B, K + NTEST)
        pred = net.predict(x[:, :K], y[:, :K], x[:, K:])
        tot += F.mse_loss(pred, y[:, K:], reduction="sum").item()
        n += B * NTEST
    return tot / n

from torch.func import functional_call, grad, vmap

class PlainSpec(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(1, H), nn.Linear(H, H), nn.Linear(H, 1)

    def forward(self, x):
        h = F.relu(self.l1(x.unsqueeze(-1)))
        h = F.relu(self.l2(h))
        return self.l3(h).squeeze(-1)

def maml_train(steps=70000, B=25, inner_lr=0.01, outer_lr=1e-3, inner_steps=1, seed=1):
    torch.manual_seed(seed)
    net = PlainSpec().to(DEV)
    names = list(dict(net.named_parameters()).keys())
    opt = torch.optim.Adam(net.parameters(), lr=outer_lr)

    def task_loss(params, x, y):
        return F.mse_loss(functional_call(net, params, (x.unsqueeze(0),)).squeeze(0), y)

    def adapted(params, cx, cy, nsteps):
        for _ in range(nsteps):
            g = grad(task_loss)(params, cx, cy)
            params = {k: params[k] - inner_lr * g[k] for k in names}
        return params

    def outer_loss(params, cx, cy, qx, qy):
        return task_loss(adapted(params, cx, cy, inner_steps), qx, qy)

    batched = vmap(outer_loss, in_dims=(None, 0, 0, 0, 0))
    t0 = time.perf_counter()
    for i in range(steps):
        x, y = sample_tasks(B, K + K)
        opt.zero_grad(set_to_none=True)
        params = dict(net.named_parameters())
        batched(params, x[:, :K], y[:, :K], x[:, K:], y[:, K:]).mean().backward()
        opt.step()
        if (i + 1) % 10000 == 0:
            print(f"  maml step {i + 1} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return net, adapted, names

@torch.no_grad()
def _noop():
    pass

def maml_eval(net, adapted, eval_steps, ntask=2000):
    def task_mse(params, cx, cy, qx, qy):
        p = adapted(params, cx, cy, eval_steps)
        return F.mse_loss(functional_call(net, p, (qx.unsqueeze(0),)).squeeze(0), qy)

    x, y = sample_tasks(ntask, K + NTEST)
    params = {k: v.detach() for k, v in net.named_parameters()}
    mses = vmap(task_mse, in_dims=(None, 0, 0, 0, 0))(
        params, x[:, :K], y[:, :K], x[:, K:], y[:, K:])
    return mses.mean().item()

if __name__ == "__main__":
    out = {"published": {"MAML (CAVIA paper)": "0.23-0.29", "CAVIA": "0.19-0.21"}}
    for mode in ["direct", "film"]:
        t0 = time.perf_counter()
        net = train_mom(mode)
        mse = eval_mom(net)
        out[f"mom_{mode}"] = mse
        print(f"[MoM {mode:6}] 10-shot MSE {mse:.4f}  (0 grad steps, "
              f"{time.perf_counter() - t0:.0f}s train)", flush=True)
        torch.save(net.state_dict(), f"exp14_mom_{mode}.pt")
    print("training our MAML baseline (2nd order)...", flush=True)
    net, adapted, _ = maml_train()
    for es in [1, 5, 10]:
        m = maml_eval(net, adapted, es)
        out[f"maml_{es}step"] = m
        print(f"[our MAML] 10-shot MSE {m:.4f}  ({es} grad steps)", flush=True)
    with open("exp14_results.json", "w") as f:
        json.dump(out, f)
