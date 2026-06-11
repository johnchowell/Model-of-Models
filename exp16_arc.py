import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
S, NCOL, NPAIR = 12, 11, 3
PAD = 10
torch.manual_seed(0)

class ArcGen:
    def __init__(self, root="data/arc_gen", val_frac=0.1, seed=0):
        self.tasks = {}
        for f in sorted(glob.glob(f"{root}/*.npz")):
            z = np.load(f)
            inp, out = z["inputs"], z["outputs"]
            self.tasks[os.path.basename(f)[:-4]] = (inp, out)
        self.keys = sorted(self.tasks)
        rng = np.random.RandomState(seed)
        self.split = {k: max(4, int(len(self.tasks[k][0]) * (1 - val_frac)))
                      for k in self.keys}

    def batch(self, B, val=False, device=DEV):
        ctx = np.empty((B, NPAIR, 2, S, S), dtype=np.int64)
        q_in = np.empty((B, S, S), dtype=np.int64)
        q_out = np.empty((B, S, S), dtype=np.int64)
        for b in range(B):
            k = self.keys[np.random.randint(len(self.keys))]
            inp, out = self.tasks[k]
            cut = self.split[k]
            pool = np.arange(cut, len(inp)) if val else np.arange(cut)
            idx = np.random.choice(pool, NPAIR + 1, replace=len(pool) < NPAIR + 1)
            for j in range(NPAIR):
                ctx[b, j, 0] = inp[idx[j]]
                ctx[b, j, 1] = out[idx[j]]
            q_in[b], q_out[b] = inp[idx[-1]], out[idx[-1]]
        t = lambda a: torch.from_numpy(np.where(a < 0, PAD, a)).to(device)
        return t(ctx), t(q_in), t(q_out)

def load_real_arc(split, max_s=S):
    items = []
    for f in sorted(glob.glob(f"data/arc/data/{split}/*.json")):
        t = json.load(open(f))
        grids = [np.array(p[k]) for p in t["train"] + t["test"] for k in ("input", "output")]
        if any(max(g.shape) > max_s for g in grids) or len(t["train"]) < NPAIR:
            continue

        def pad(g):
            g = np.array(g)
            return np.pad(g, ((0, max_s - g.shape[0]), (0, max_s - g.shape[1])),
                          constant_values=PAD)

        ctx = np.stack([[pad(t["train"][j]["input"]), pad(t["train"][j]["output"])]
                        for j in range(NPAIR)])
        for te in t["test"]:
            items.append((ctx, pad(te["input"]), pad(te["output"]),
                          os.path.basename(f)[:-5]))
    return items

class Blk(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.at = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, film=None):
        a = self.n1(x)
        x = x + self.at(a, a, a, need_weights=False)[0]
        h = self.n2(x)
        if film is not None:
            s, m = film
            h = h * (1 + s.unsqueeze(1)) + m.unsqueeze(1)
        return x + self.mlp(h)

class GridEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.col = nn.Embedding(NCOL, d)
        self.pr = nn.Parameter(torch.randn(1, S, 1, d) * 0.02)
        self.pc = nn.Parameter(torch.randn(1, 1, S, d) * 0.02)
        self.role = nn.Embedding(3, d)

    def forward(self, g, role):
        e = self.col(g) + self.pr + self.pc + self.role.weight[role]
        return e.flatten(1, 2)

class ArcMoM(nn.Module):
    def __init__(self, mode="mom", d=256, heads=8, enc_l=4, dec_l=6):
        super().__init__()
        self.mode, self.d, self.dec_l = mode, d, dec_l
        self.ge = GridEmbed(d)
        self.enc = nn.ModuleList([Blk(d, heads) for _ in range(enc_l)])
        self.dec = nn.ModuleList([Blk(d, heads) for _ in range(dec_l)])
        self.canvas = nn.Parameter(torch.randn(1, S * S, d) * 0.02)
        self.head = nn.Linear(d, NCOL)
        if mode == "mom":
            self.film = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * dec_l * d))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)

    def encode_task(self, ctx):
        B = ctx.shape[0]
        toks = torch.cat([self.ge(ctx[:, j, 0], 0) for j in range(NPAIR)] +
                         [self.ge(ctx[:, j, 1], 1) for j in range(NPAIR)], 1)
        for b in self.enc:
            toks = b(toks)
        return toks.mean(1), toks

    def forward(self, ctx, q_in, film_override=None):
        B = q_in.shape[0]
        z, ctx_toks = self.encode_task(ctx)
        x = torch.cat([self.ge(q_in, 2), self.canvas.expand(B, -1, -1)], 1)
        if self.mode == "icl":
            x = torch.cat([ctx_toks, x], 1)
            films = [None] * self.dec_l
        else:
            f = film_override if film_override is not None else self.film(z)
            films = f.view(B, self.dec_l, 2, self.d).permute(1, 2, 0, 3)
        for b, fl in zip(self.dec, films):
            x = b(x, None if fl is None else (fl[0], fl[1]))
        return self.head(x[:, -S * S:]).view(B, S, S, NCOL)

def exact_match(logits, tgt):
    return (logits.argmax(-1) == tgt).flatten(1).all(1).float()

def evaluate_real(net, items, B=64):
    em, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            ctx = torch.from_numpy(np.stack([c for c, _, _, _ in chunk])).long().to(DEV)
            qi = torch.from_numpy(np.stack([q for _, q, _, _ in chunk])).long().to(DEV)
            qo = torch.from_numpy(np.stack([q for _, _, q, _ in chunk])).long().to(DEV)
            em += exact_match(net(ctx, qi), qo).sum().item()
            n += len(chunk)
    return em / n, n

def train(mode, steps=60000, B=64, lr=3e-4, seed=1):
    torch.manual_seed(seed)
    gen = ArcGen()
    print(f"{len(gen.keys)} generated rule-sets loaded", flush=True)
    net = ArcMoM(mode).to(DEV)
    print("params:", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.03)
    t0 = time.perf_counter()
    for i in range(steps):
        ctx, qi, qo = gen.batch(B)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(ctx, qi).flatten(0, 2), qo.flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if (i + 1) % 2000 == 0:
            with torch.no_grad():
                ctx, qi, qo = gen.batch(256, val=True)
                em = exact_match(net(ctx, qi), qo).mean().item()
            print(f"[{mode}] step {i + 1}  loss {loss.item():.3f}  "
                  f"gen-val exact {em:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            torch.save(net.state_dict(), f"arc_{mode}_ckpt.pt")
    torch.save(net.state_dict(), f"arc_{mode}.pt")
    return net, gen

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "mom"
    bsz = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    net, gen = train(mode, B=bsz)
    res = {}
    with torch.no_grad():
        ctx, qi, qo = gen.batch(2000, val=True)
        res["generated_heldout_exact"] = exact_match(net(ctx, qi), qo).mean().item()
    for split in ["training", "evaluation"]:
        items = load_real_arc(split)
        em, n = evaluate_real(net, items)
        res[f"real_{split}_exact"] = em
        res[f"real_{split}_n"] = n
        print(f"[{mode}] real ARC {split}: exact {em:.3f} over {n} test grids", flush=True)
    print(json.dumps(res))
    with open(f"exp16_results_{mode}.json", "w") as f:
        json.dump(res, f)
