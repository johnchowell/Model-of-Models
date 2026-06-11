import json
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
EP_LEN, ZDIM, CTXN, H = 200, 16, 64, 256
torch.manual_seed(0)
np.random.seed(0)

class CheetahVel:
    def __init__(self, v):
        self.env = gym.make("HalfCheetah-v5")
        self.v = v

    def reset(self, seed=None):
        s, _ = self.env.reset(seed=seed)
        return s

    def step(self, a):
        s, _, te, tr, info = self.env.step(a)
        r = -abs(info["x_velocity"] - self.v) - 0.05 * float(np.square(a).sum())
        return s, r, te or tr

SDIM, ADIM = 17, 6

class Buffer:
    def __init__(self, cap=40000):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, SDIM), np.float32)
        self.a = np.zeros((cap, ADIM), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.s2 = np.zeros((cap, SDIM), np.float32)

    def add(self, s, a, r, s2):
        j = self.i % self.cap
        self.s[j], self.a[j], self.r[j], self.s2[j] = s, a, r, s2
        self.i += 1
        self.n = min(self.n + 1, self.cap)

    def sample(self, B):
        idx = np.random.randint(0, self.n, B)
        return (torch.from_numpy(self.s[idx]).to(DEV), torch.from_numpy(self.a[idx]).to(DEV),
                torch.from_numpy(self.r[idx]).to(DEV), torch.from_numpy(self.s2[idx]).to(DEV))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(SDIM + ADIM + 1 + SDIM, 128), nn.ReLU(),
                               nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ZDIM))

    def forward(self, s, a, r, s2):
        return self.f(torch.cat([s, a, r, s2], -1)).mean(1)

class Policy(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        ind = SDIM + (ZDIM if mode == "latent" else 0)
        self.l1, self.l2 = nn.Linear(ind, H), nn.Linear(H, H)
        self.mu, self.logstd = nn.Linear(H, ADIM), nn.Linear(H, ADIM)
        if mode == "mom":
            self.film = nn.Linear(ZDIM, 4 * H)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, s, z):
        if self.mode == "latent":
            h = torch.cat([s, z], -1)
            h = F.relu(self.l1(h))
            h = F.relu(self.l2(h))
        else:
            s1, m1, s2_, m2 = self.film(z).split(H, -1)
            h = F.relu(self.l1(s) * (1 + s1) + m1)
            h = F.relu(self.l2(h) * (1 + s2_) + m2)
        mu, ls = self.mu(h), self.logstd(h).clamp(-10, 2)
        return mu, ls

    def sample(self, s, z):
        mu, ls = self(s, z)
        std = ls.exp()
        e = torch.randn_like(mu)
        pre = mu + e * std
        a = torch.tanh(pre)
        logp = (-0.5 * (e ** 2 + 2 * ls + np.log(2 * np.pi))).sum(-1, keepdim=True) \
            - torch.log(1 - a ** 2 + 1e-6).sum(-1, keepdim=True)
        return a, logp

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(SDIM + ADIM + ZDIM, H), nn.ReLU(),
                               nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, s, a, z):
        return self.f(torch.cat([s, a, z], -1))

def rollout(env, pol, enc, ctx, explore=True):
    with torch.no_grad():
        if ctx is None or len(ctx[0]) == 0:
            z = torch.zeros(1, ZDIM, device=DEV)
        else:
            s, a, r, s2 = [torch.from_numpy(np.array(x, np.float32)).unsqueeze(0).to(DEV)
                           for x in ctx]
            if r.dim() == 2:
                r = r.unsqueeze(-1)
            z = enc(s, a, r, s2)
    s = env.reset()
    traj, R = [], 0.0
    for _ in range(EP_LEN):
        st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            if explore:
                a, _ = pol.sample(st, z)
            else:
                a = torch.tanh(pol(st, z)[0])
        a = a.squeeze(0).cpu().numpy()
        s2, r, done = env.step(a)
        traj.append((s, a, r, s2))
        R += r
        s = s2
        if done:
            break
    return traj, R

def train(mode, iters=300, ntrain=100, meta_b=8, B=256, updates=1000, lr=3e-4,
          gamma=0.99, tau=5e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_v = np.random.uniform(0, 3, ntrain)
    test_v = np.random.uniform(0, 3, 30)
    envs = {i: CheetahVel(v) for i, v in enumerate(train_v)}
    bufs = {i: Buffer() for i in range(ntrain)}
    enc = Encoder().to(DEV)
    pol = Policy(mode).to(DEV)
    q1, q2, q1t, q2t = Q().to(DEV), Q().to(DEV), Q().to(DEV), Q().to(DEV)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    qopt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters())
                            + list(enc.parameters()), lr=lr)
    popt = torch.optim.Adam(pol.parameters(), lr=lr)
    log_alpha = torch.zeros(1, device=DEV, requires_grad=True)
    aopt = torch.optim.Adam([log_alpha], lr=lr)
    tgt_ent = -ADIM
    t0 = time.perf_counter()
    hist = {"iter": [], "prior": [], "post": []}

    def ctx_batch(i, n=CTXN):
        b = bufs[i]
        idx = np.random.randint(0, b.n, n)
        return (torch.from_numpy(b.s[idx]).to(DEV), torch.from_numpy(b.a[idx]).to(DEV),
                torch.from_numpy(b.r[idx]).to(DEV), torch.from_numpy(b.s2[idx]).to(DEV))

    for it in range(iters):

        for i in np.random.choice(ntrain, 5, replace=False):
            ctx = None if bufs[i].n < CTXN else \
                [x.cpu().numpy() for x in ctx_batch(i)]
            traj, _ = rollout(envs[i], pol, enc, ctx)
            for s, a, r, s2 in traj:
                bufs[i].add(s, a, r, s2)
        ready = [i for i in range(ntrain) if bufs[i].n >= max(B, CTXN)]
        if len(ready) < meta_b:
            continue
        for _ in range(updates):
            tasks = np.random.choice(ready, meta_b, replace=False)
            alpha = log_alpha.exp().detach()

            qopt.zero_grad(set_to_none=True)
            qls = 0.0
            for i in tasks:
                cs, ca, cr, cs2 = ctx_batch(i)
                z = enc(cs.unsqueeze(0), ca.unsqueeze(0), cr.unsqueeze(0), cs2.unsqueeze(0))
                s, a, r, s2 = bufs[i].sample(B)
                zB = z.expand(B, -1)
                with torch.no_grad():
                    a2, lp2 = pol.sample(s2, zB)
                    qt = torch.min(q1t(s2, a2, zB), q2t(s2, a2, zB)) - alpha * lp2
                    y = r + gamma * qt
                qls = qls + F.mse_loss(q1(s, a, zB), y) + F.mse_loss(q2(s, a, zB), y)
            qls.backward()
            qopt.step()

            popt.zero_grad(set_to_none=True)
            pls, lps = 0.0, []
            for i in tasks:
                with torch.no_grad():
                    cs, ca, cr, cs2 = ctx_batch(i)
                    z = enc(cs.unsqueeze(0), ca.unsqueeze(0), cr.unsqueeze(0),
                            cs2.unsqueeze(0))
                s, _, _, _ = bufs[i].sample(B)
                zB = z.expand(B, -1)
                anew, lp = pol.sample(s, zB)
                qmin = torch.min(q1(s, anew, zB), q2(s, anew, zB))
                pls = pls + (alpha * lp - qmin).mean()
                lps.append(lp.detach().mean())
            pls.backward()
            popt.step()
            aopt.zero_grad(set_to_none=True)
            (-log_alpha.exp() * (torch.stack(lps).mean() + tgt_ent)).backward()
            aopt.step()
            with torch.no_grad():
                for qa, qb in [(q1, q1t), (q2, q2t)]:
                    for p, pt in zip(qa.parameters(), qb.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
        if (it + 1) % 10 == 0:
            pr, po = [], []
            for v in test_v[:10]:
                te = CheetahVel(v)
                t1, R1 = rollout(te, pol, enc, None, explore=False)
                ctx = list(zip(*t1))
                _, R2 = rollout(te, pol, enc, [np.array(x) for x in ctx], explore=False)
                pr.append(R1); po.append(R2)
            hist["iter"].append(it + 1)
            hist["prior"].append(float(np.mean(pr)))
            hist["post"].append(float(np.mean(po)))
            print(f"[{mode}] iter {it + 1}  prior {np.mean(pr):.0f}  "
                  f"post-adapt {np.mean(po):.0f}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
            torch.save({"enc": enc.state_dict(), "pol": pol.state_dict()},
                       f"metarl_{mode}.pt")

    pr, po = [], []
    for v in test_v:
        te = CheetahVel(v)
        t1, R1 = rollout(te, pol, enc, None, explore=False)
        ctx = [np.array(x) for x in zip(*t1)]
        _, R2 = rollout(te, pol, enc, ctx, explore=False)
        pr.append(R1); po.append(R2)
    out = {"hist": hist, "final_prior": float(np.mean(pr)),
           "final_post": float(np.mean(po))}
    print(f"[{mode}] FINAL  prior {out['final_prior']:.0f}  "
          f"post-adapt {out['final_post']:.0f}", flush=True)
    with open(f"exp17_results_{mode}.json", "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else "mom")
