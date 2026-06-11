import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
NGENE, NCLASS, KSHOT = 1000, 5, 5

def load():
    z = np.load("data/rnaseq/prepared.npz", allow_pickle=True)
    X, y = z["X"], z["y"]
    top = np.argsort(X.var(0))[::-1][:NGENE]
    return X[:, top], y, z["classes"]

def part_a(X, y):
    print(f"== Part A: standard references ==  n={len(y)} genes={X.shape[1]}", flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {}
    for name, mk in [("logreg", lambda: LogisticRegression(max_iter=2000, C=0.1)),
                     ("rf", lambda: RandomForestClassifier(300, random_state=0))]:
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = mk().fit(sc.transform(X[tr]), y[tr])
            p = m.predict(sc.transform(X[te]))
            accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
        out[name] = {"acc": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}
        print(f"  {name:8} acc {np.mean(accs):.3f}  macro-F1 {np.mean(f1s):.3f}", flush=True)
    return out

class GeneFewShot:
    def __init__(self, X, y, tr_idx):
        self.sc = StandardScaler().fit(X[tr_idx])
        self.X = self.sc.transform(X).astype(np.float32)
        self.y = y
        self.by = {c: tr_idx[y[tr_idx] == c] for c in range(NCLASS)}

    def set_pools(self, idx):
        self.by = {c: idx[self.y[idx] == c] for c in range(NCLASS)}

    def task(self, B, nq=5, device=DEV):
        sx = np.empty((B, NCLASS * KSHOT, NGENE), np.float32)
        sy = np.empty((B, NCLASS * KSHOT), np.int64)
        qx = np.empty((B, NCLASS * nq, NGENE), np.float32)
        qy = np.empty((B, NCLASS * nq), np.int64)
        for b in range(B):
            si, qi, syl, qyl = [], [], [], []
            for c in range(NCLASS):
                pool = self.by[c]
                pick = np.random.choice(pool, KSHOT + nq, replace=len(pool) < KSHOT + nq)
                si += list(pick[:KSHOT]); qi += list(pick[KSHOT:])
                syl += [c] * KSHOT; qyl += [c] * nq
            sx[b], sy[b] = self.X[si], syl
            qx[b], qy[b] = self.X[qi], qyl
        t = lambda a: torch.from_numpy(a).to(device)
        return t(sx), t(sy), t(qx), t(qy)

class GeneNet(nn.Module):
    def __init__(self, mode, d=256, H=128):
        super().__init__()
        self.mode, self.H = mode, H
        self.row = nn.Sequential(nn.Linear(NGENE, d), nn.GELU(), nn.Linear(d, d))
        self.lab = nn.Embedding(NCLASS, d)
        self.enc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        if mode == "icl":
            self.q = nn.Linear(NGENE, d)
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NCLASS))
        else:
            self.l1 = nn.Linear(NGENE, H)
            self.head = nn.Linear(H, NCLASS)
            self.film = nn.Linear(d, 2 * H)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, sx, sy, qx):
        srow = self.row(sx) + self.lab(sy)
        z = self.enc(srow.mean(1))
        if self.mode == "icl":
            q = self.q(qx)
            a = self.attn(q, srow, srow, need_weights=False)[0]
            return self.out(q + a)
        s, m = self.film(z).split(self.H, -1)
        h = F.relu(self.l1(qx) * (1 + s.unsqueeze(1)) + m.unsqueeze(1))
        return self.head(h)

def train_net(mode, data, steps=3000, B=32, lr=1e-3, seed=1):
    torch.manual_seed(seed)
    net = GeneNet(mode).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    for i in range(steps):
        sx, sy, qx, qy = data.task(B)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(net(sx, sy, qx).flatten(0, 1), qy.flatten()).backward()
        opt.step(); sched.step()
    return net

@torch.no_grad()
def eval_net(net, data, ntask=300, nq=5):
    acc = []
    for _ in range(ntask):
        sx, sy, qx, qy = data.task(1, nq=nq)
        acc.append((net(sx, sy, qx).argmax(-1) == qy).float().mean().item())
    return float(np.mean(acc))

def pertask_baselines(data, ntask=300, nq=5):
    log, nc = [], []
    for _ in range(ntask):
        sx, sy, qx, qy = data.task(1, nq=nq, device="cpu")
        sx, sy, qx, qy = [a.numpy()[0] for a in (sx, sy, qx, qy)]
        try:
            m = LogisticRegression(max_iter=300, C=0.1).fit(sx, sy)
            log.append(accuracy_score(qy, m.predict(qx)))
        except Exception:
            log.append(1.0 / NCLASS)
        cent = np.stack([sx[sy == c].mean(0) for c in range(NCLASS)])
        d = ((qx[:, None] - cent[None]) ** 2).sum(-1)
        nc.append(accuracy_score(qy, d.argmin(1)))
    return float(np.mean(log)), float(np.mean(nc))

def part_b(X, y):
    print(f"== Part B: 5-way {KSHOT}-shot quadrant ==", flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = {m: [] for m in ["mom", "icl", "pertask_logfit", "nearest_centroid"]}
    for tr, te in skf.split(X, y):
        dtr = GeneFewShot(X, y, tr)
        dte = GeneFewShot(X, y, tr); dte.set_pools(te)
        if min(len(v) for v in dte.by.values()) < 2:
            continue
        for mode in ["mom", "icl"]:
            rows[mode].append(eval_net(train_net(mode, dtr), dte))
        lg, nc = pertask_baselines(dte)
        rows["pertask_logfit"].append(lg); rows["nearest_centroid"].append(nc)
    out = {}
    for m, v in rows.items():
        v = np.array(v)
        out[m] = {"acc": float(v.mean()), "acc_std": float(v.std())}
        print(f"  {m:16} acc {v.mean():.3f}+/-{v.std():.3f}", flush=True)
    return out

if __name__ == "__main__":
    X, y, classes = load()
    print("classes:", list(classes), flush=True)
    out = {"part_a": part_a(X, y), "part_b": part_b(X, y)}
    with open("exp19_results.json", "w") as f:
        json.dump(out, f)
