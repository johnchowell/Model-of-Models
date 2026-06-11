import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import exp13_hyperlora as h13
from exp12_lm import DEV, LM
from exp13_hyperlora import (CTX, SCALE, TGT, HyperLoRA, chunk_bpc, theta0, unpack)
from exp15_enwik8 import load_enwik8

torch.manual_seed(0)

def sample_contig(data, B, device=DEV):
    srcs = list(data.values())
    w = torch.tensor([len(s) for s in srcs], dtype=torch.float)
    ctx = torch.empty(B, CTX, dtype=torch.long)
    tgt = torch.empty(B, TGT + 1, dtype=torch.long)
    si = torch.multinomial(w, B, replacement=True)
    for i, j in enumerate(si):
        s = srcs[j]
        o = torch.randint(0, len(s) - CTX - TGT - 1, (1,)).item()
        ctx[i] = s[o: o + CTX]
        tgt[i] = s[o + CTX - 1: o + CTX + TGT]
    return ctx.to(device), tgt.to(device)

def train_hyper(base, train_d, val_d, steps=4000, B=32, lr=3e-4):
    hyper = HyperLoRA().to(DEV)
    opt = torch.optim.AdamW(hyper.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0 = time.perf_counter()
    for i in range(steps):
        ctx, tgt = sample_contig(train_d, B)
        opt.zero_grad(set_to_none=True)
        loss = chunk_bpc(base(tgt[:, :-1], lora=unpack(hyper(ctx)), scale=SCALE), tgt[:, 1:])
        loss.backward()
        nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)
        opt.step()
        sched.step()
        if (i + 1) % 1000 == 0:
            print(f"hyper {i + 1}  train bpc {loss.item():.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    torch.save(hyper.state_dict(), "enwik8_hyper_contig.pt")
    return hyper

@torch.no_grad()
def eval_all(base, hyper, data, nbatch=16, B=32):
    zs = ic = hl = 0.0
    for _ in range(nbatch):
        ctx, tgt = sample_contig(data, B)
        zs += chunk_bpc(base(tgt[:, :-1]), tgt[:, 1:]).item()
        seq = torch.cat([ctx, tgt[:, 1:]], 1)
        logits = base(seq[:, :-1])
        ic += chunk_bpc(logits[:, CTX - 1:], tgt[:, 1:]).item()
        hl += chunk_bpc(base(tgt[:, :-1], lora=unpack(hyper(ctx)), scale=SCALE),
                        tgt[:, 1:]).item()
    return zs / nbatch, ic / nbatch, hl / nbatch

def tuned_ceiling(base, data, ndocs=16, steps=20, lr=1e-3):
    tot = 0.0
    for k in range(ndocs):
        torch.manual_seed(200 + k)
        ctx, tgt = sample_contig(data, 1)
        th = theta0().to(DEV).unsqueeze(0).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            chunk_bpc(base(ctx[:, -TGT - 1:-1], lora=unpack(th), scale=SCALE),
                      ctx[:, -TGT:]).backward()
            opt.step()
        with torch.no_grad():
            tot += chunk_bpc(base(tgt[:, :-1], lora=unpack(th), scale=SCALE),
                             tgt[:, 1:]).item()
    return tot / ndocs

if __name__ == "__main__":
    train_d, val_d, test_d = load_enwik8()
    base = LM(T=CTX + TGT).to(DEV)
    base.load_state_dict(torch.load("enwik8_lm.pt", map_location=DEV))
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    print("== training hypernet (contiguous protocol) ==", flush=True)
    hyper = train_hyper(base, train_d, val_d)
    print("== final eval on TEST ==", flush=True)
    zs, ic, hl = eval_all(base, hyper, test_d)
    tl = tuned_ceiling(base, test_d)
    cap = (zs - hl) / (zs - ic) * 100 if zs > ic else float("nan")
    print(f"enwik8 test bpc:  zero-shot {zs:.3f}   in-context(contig) {ic:.3f}   "
          f"hyper-LoRA {hl:.3f}   tuned-LoRA(20@1e-3) {tl:.3f}")
    print(f"hyper-LoRA captures {cap:.0f}% of the in-context gain")
    with open("exp15b_results.json", "w") as f:
        json.dump({"zero_shot": zs, "in_context_contig": ic, "hyper_lora": hl,
                   "tuned_lora": tl, "pct_gain_captured": cap}, f)
