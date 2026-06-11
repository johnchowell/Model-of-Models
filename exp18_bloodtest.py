import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

def load():
    df = pd.read_csv("data/hcv/hcvdat0.csv", index_col=0)
    df = df[~df["Category"].str.startswith("0s")]
    y = (~df["Category"].str.startswith("0=")).astype(int).values
    df["Sex"] = (df["Sex"] == "m").astype(float)
    X = df.drop(columns=["Category"]).astype(float).values
    feat = list(df.drop(columns=["Category"]).columns)
    return X, y, feat

def part_a(X, y):
    print(f"== Part A: standard tabular benchmark ==  n={len(y)}  disease={y.sum()}",
          flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    models = {
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": lambda: RandomForestClassifier(300, class_weight="balanced", random_state=0),
    }
    out = {}
    for name, mk in models.items():
        aucs, bas = [], []
        for tr, te in skf.split(X, y):
            pipe_imp = SimpleImputer().fit(X[tr])
            sc = StandardScaler().fit(pipe_imp.transform(X[tr]))
            Xtr = sc.transform(pipe_imp.transform(X[tr]))
            Xte = sc.transform(pipe_imp.transform(X[te]))
            m = mk().fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            bas.append(balanced_accuracy_score(y[te], p > 0.5))
        out[name] = {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                     "bal_acc": float(np.mean(bas))}
        print(f"  {name:8} ROC-AUC {np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
              f"bal-acc {np.mean(bas):.3f}", flush=True)
    return out

NFEAT = 12
KSHOT = 16

class FewShotData:
    def __init__(self, X, y, idx):
        imp = SimpleImputer().fit(X[idx])
        self.sc = StandardScaler().fit(imp.transform(X[idx]))
        self.imp = imp
        self.X = self.sc.transform(imp.transform(X))
        self.y = y
        self.pos = idx[y[idx] == 1]
        self.neg = idx[y[idx] == 0]

    def task(self, B, nq=16, device=DEV):
        k = KSHOT // 2
        sx = np.empty((B, KSHOT, NFEAT), np.float32)
        sy = np.empty((B, KSHOT), np.float32)
        qx = np.empty((B, nq, NFEAT), np.float32)
        qy = np.empty((B, nq), np.float32)
        for b in range(B):
            sp = np.random.choice(self.pos, k, replace=len(self.pos) < k)
            sn = np.random.choice(self.neg, k, replace=False)
            qp = np.random.choice(self.pos, nq // 2, replace=True)
            qn = np.random.choice(self.neg, nq // 2, replace=False)
            si = np.concatenate([sp, sn]); qi = np.concatenate([qp, qn])
            sx[b], sy[b] = self.X[si], self.y[si]
            qx[b], qy[b] = self.X[qi], self.y[qi]
        t = lambda a: torch.from_numpy(a).to(device)
        return t(sx), t(sy), t(qx), t(qy)

class FewShotNet(nn.Module):
    def __init__(self, mode, d=128, H=64):
        super().__init__()
        self.mode, self.H = mode, H
        self.row = nn.Sequential(nn.Linear(NFEAT + 1, d), nn.GELU(), nn.Linear(d, d))
        self.enc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        if mode == "icl":
            self.q = nn.Linear(NFEAT, d)
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        else:
            self.l1 = nn.Linear(NFEAT, H)
            self.l2 = nn.Linear(H, 1)
            self.film = nn.Linear(d, 2 * H)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, sx, sy, qx):
        z = self.enc(self.row(torch.cat([sx, sy.unsqueeze(-1)], -1)).mean(1))
        if self.mode == "icl":
            srow = self.row(torch.cat([sx, sy.unsqueeze(-1)], -1))
            q = self.q(qx)
            a = self.attn(q, srow, srow, need_weights=False)[0]
            return self.out(q + a).squeeze(-1)
        s, m = self.film(z).split(self.H, -1)
        h = F.relu(self.l1(qx) * (1 + s.unsqueeze(1)) + m.unsqueeze(1))
        return self.l2(h).squeeze(-1)

def train_fewshot(mode, data, steps=4000, B=64, lr=1e-3, seed=1):
    torch.manual_seed(seed)
    net = FewShotNet(mode).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    for i in range(steps):
        sx, sy, qx, qy = data.task(B)
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(net(sx, sy, qx), qy).backward()
        opt.step(); sched.step()
    return net

@torch.no_grad()
def eval_fewshot(net, data, ntask=400, nq=32):
    ys, ps = [], []
    for _ in range(ntask):
        sx, sy, qx, qy = data.task(1, nq=nq)
        ps.append(torch.sigmoid(net(sx, sy, qx)).flatten().cpu().numpy())
        ys.append(qy.flatten().cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return roc_auc_score(y, p), balanced_accuracy_score(y, p > 0.5)

def pertask_logfit(data, ntask=400, nq=32):
    ys, ps = [], []
    for _ in range(ntask):
        sx, sy, qx, qy = data.task(1, nq=nq, device="cpu")
        sx, sy, qx, qy = [a.numpy()[0] for a in (sx, sy, qx, qy)]
        try:
            m = LogisticRegression(max_iter=500).fit(sx, sy)
            ps.append(m.predict_proba(qx)[:, 1])
        except Exception:
            ps.append(np.full(len(qy), sy.mean()))
        ys.append(qy)
    y, p = np.concatenate(ys), np.concatenate(ps)
    return roc_auc_score(y, p), balanced_accuracy_score(y, p > 0.5)

def part_b(X, y):
    print("== Part B: few-shot MoM quadrant (16-shot, balanced support) ==", flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = {m: [] for m in ["mom", "icl", "pertask_logfit"]}
    for tr, te in skf.split(X, y):
        dtr = FewShotData(X, y, tr)

        dte = FewShotData(X, y, tr)
        dte.pos = te[y[te] == 1]; dte.neg = te[y[te] == 0]
        if len(dte.pos) < 2:
            continue
        for mode in ["mom", "icl"]:
            net = train_fewshot(mode, dtr)
            rows[mode].append(eval_fewshot(net, dte))
        rows["pertask_logfit"].append(pertask_logfit(dte))
    out = {}
    for m, vals in rows.items():
        au = np.array([v[0] for v in vals]); ba = np.array([v[1] for v in vals])
        out[m] = {"auc": float(au.mean()), "auc_std": float(au.std()),
                  "bal_acc": float(ba.mean())}
        print(f"  {m:16} ROC-AUC {au.mean():.3f}+/-{au.std():.3f}  "
              f"bal-acc {ba.mean():.3f}", flush=True)
    return out

if __name__ == "__main__":
    X, y, feat = load()
    print("features:", feat, flush=True)
    out = {"part_a": part_a(X, y), "part_b": part_b(X, y)}
    with open("exp18_results.json", "w") as f:
        json.dump(out, f)
