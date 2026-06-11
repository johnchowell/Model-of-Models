import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp12_lm import DEV, LM, LMBlock, load_corpus, lora_delta

torch.manual_seed(0)
CTX, TGT, RANK, ALPHA = 512, 256, 4, 8.0
D, NL, DFF = 256, 6, 1024

PER_BLOCK = [(RANK, D), (D, RANK), (RANK, DFF), (D, RANK)]
SIZES = [math.prod(s) for s in PER_BLOCK] * NL
NPARAM = sum(SIZES)
SCALE = ALPHA / RANK

def theta0():
    parts = []
    for _ in range(NL):
        for shape, is_a in zip(PER_BLOCK, [True, False, True, False]):
            parts.append((torch.randn(shape) / RANK).flatten() if is_a
                         else torch.zeros(shape).flatten())
    return torch.cat(parts)

def unpack(theta):
    B = theta.shape[0]
    chunks = theta.split(SIZES, -1)
    lora, i = [], 0
    for _ in range(NL):
        Ao = chunks[i].view(B, *PER_BLOCK[0]); Bo = chunks[i + 1].view(B, *PER_BLOCK[1])
        Af = chunks[i + 2].view(B, *PER_BLOCK[2]); Bf = chunks[i + 3].view(B, *PER_BLOCK[3])
        lora.append(((Ao, Bo), (Af, Bf)))
        i += 4
    return lora

class HyperLoRA(nn.Module):
    def __init__(self, d=256, heads=8, layers=4):
        super().__init__()
        self.emb = nn.Embedding(256, d)
        self.pos = nn.Parameter(torch.randn(1, CTX, d) * 0.02)
        self.blocks = nn.ModuleList([LMBlock(d, heads) for _ in range(layers)])
        self.nf = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, NPARAM))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.t0 = nn.Parameter(theta0())

    def forward(self, ctx):
        x = self.emb(ctx) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x).mean(1)) + self.t0

DOC = 4096

def sample_doc_batch(data, B, device=DEV):
    srcs = list(data.values())
    w = torch.tensor([len(s) for s in srcs], dtype=torch.float)
    ctx = torch.empty(B, CTX, dtype=torch.long)
    tgt = torch.empty(B, TGT + 1, dtype=torch.long)
    si = torch.multinomial(w, B, replacement=True)
    for i, j in enumerate(si):
        s = srcs[j]
        do = torch.randint(0, len(s) - DOC, (1,)).item()
        co = torch.randint(0, DOC - CTX, (1,)).item()
        while True:
            to = torch.randint(0, DOC - TGT - 1, (1,)).item()
            if to + TGT + 1 <= co or to >= co + CTX:
                break
        ctx[i] = s[do + co: do + co + CTX]
        tgt[i] = s[do + to: do + to + TGT + 1]
    return ctx.to(device), tgt.to(device)

def chunk_bpc(logits, tgt):
    return F.cross_entropy(logits.flatten(0, 1), tgt.flatten()) / math.log(2)

def train(steps=4000, B=32, lr=3e-4, seed=1):
    torch.manual_seed(seed)
    base = LM(T=CTX + TGT).to(DEV)
    base.load_state_dict(torch.load("base_lm.pt", map_location=DEV))
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    hyper = HyperLoRA().to(DEV)
    train_d, val_d = load_corpus()
    opt = torch.optim.AdamW(hyper.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    hist = {"step": [], "val": []}
    t0 = time.perf_counter()
    for i in range(steps):
        ctx, tgt = sample_doc_batch(train_d, B)
        opt.zero_grad(set_to_none=True)
        lora = unpack(hyper(ctx))
        loss = chunk_bpc(base(tgt[:, :-1], lora=lora, scale=SCALE), tgt[:, 1:])
        loss.backward()
        nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)
        opt.step()
        sched.step()
        if (i + 1) % 250 == 0:
            v = eval_hyper(base, hyper, val_d, nbatch=8)
            hist["step"].append(i + 1); hist["val"].append(v)
            print(f"step {i + 1}  train bpc {loss.item():.3f}  val bpc {v:.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    torch.save(hyper.state_dict(), "hyper_lora.pt")
    return base, hyper, hist

@torch.no_grad()
def eval_hyper(base, hyper, data, nbatch=8, B=32):
    tot = 0.0
    for _ in range(nbatch):
        ctx, tgt = sample_doc_batch(data, B)
        lora = unpack(hyper(ctx))
        tot += chunk_bpc(base(tgt[:, :-1], lora=lora, scale=SCALE), tgt[:, 1:]).item()
    return tot / nbatch

@torch.no_grad()
def eval_baselines(base, data, nbatch=8, B=32):
    zs, ic = 0.0, 0.0
    for _ in range(nbatch):
        ctx, tgt = sample_doc_batch(data, B)
        zs += chunk_bpc(base(tgt[:, :-1]), tgt[:, 1:]).item()
        seq = torch.cat([ctx, tgt], 1)
        logits = base(seq[:, :-1])
        ic += chunk_bpc(logits[:, CTX:], tgt[:, 1:]).item()
    return zs / nbatch, ic / nbatch

def tuned_lora_ceiling(base, data, ndocs=8, steps=100, lr=1e-2):
    tot = 0.0
    for k in range(ndocs):
        torch.manual_seed(100 + k)
        ctx, tgt = sample_doc_batch(data, 1)
        th = theta0().to(DEV).unsqueeze(0).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = chunk_bpc(base(ctx[:, :-1], lora=unpack(th), scale=SCALE), ctx[:, 1:])
            loss.backward()
            opt.step()
        with torch.no_grad():
            tot += chunk_bpc(base(tgt[:, :-1], lora=unpack(th), scale=SCALE),
                             tgt[:, 1:]).item()
    return tot / ndocs

if __name__ == "__main__":
    print(f"LoRA params emitted per doc: {NPARAM}")
    base, hyper, hist = train()
    _, val_d = load_corpus()
    zs, ic = eval_baselines(base, val_d, nbatch=16)
    hl = eval_hyper(base, hyper, val_d, nbatch=16)
    tl = tuned_lora_ceiling(base, val_d)
    print(f"\nheld-out bits/char:  zero-shot {zs:.3f}   in-context {ic:.3f}   "
          f"hyper-LoRA {hl:.3f}   tuned-LoRA(100 Adam steps) {tl:.3f}")
    with open("exp13_results.json", "w") as f:
        json.dump({"hist": hist, "zero_shot": zs, "in_context": ic,
                   "hyper_lora": hl, "tuned_lora": tl, "nparam": NPARAM}, f)
