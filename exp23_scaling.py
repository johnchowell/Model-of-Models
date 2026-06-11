import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp12_lm import DEV, LM, sample_lm_batch
import exp13_hyperlora as h13
from exp13_hyperlora import CTX, SCALE, TGT, chunk_bpc, theta0, unpack
from exp15_enwik8 import load_enwik8
from exp15b_contig import sample_contig

CONFIGS = [(128, 4), (256, 6), (384, 8)]

def build_lm(d, layers, T=CTX + TGT):
    return LM(d=d, heads=max(4, d // 32), layers=layers, T=T).to(DEV)

def pretrain(model, train_d, val_d, steps, B=32, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    for i in range(steps):
        xb = sample_lm_batch(train_d, B, CTX + TGT)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb[:, :-1]).flatten(0, 1), xb[:, 1:].flatten()).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    return model

class HyperLoRA(nn.Module):
    def __init__(self, base_d, base_layers, enc_d=256, enc_l=4):
        super().__init__()
        from exp12_lm import LMBlock
        self.bd, self.bl = base_d, base_layers
        self.dff = 4 * base_d
        self.per = [(h13.RANK, base_d), (base_d, h13.RANK),
                    (h13.RANK, self.dff), (base_d, h13.RANK)]
        self.sizes = [a * b for a, b in self.per] * base_layers
        self.nparam = sum(self.sizes)
        self.emb = nn.Embedding(256, enc_d)
        self.pos = nn.Parameter(torch.randn(1, CTX, enc_d) * 0.02)
        self.blocks = nn.ModuleList([LMBlock(enc_d, 8) for _ in range(enc_l)])
        self.nf = nn.LayerNorm(enc_d)
        self.head = nn.Sequential(nn.Linear(enc_d, 512), nn.GELU(), nn.Linear(512, self.nparam))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.t0 = nn.Parameter(self._theta0())

    def _theta0(self):
        parts = []
        for _ in range(self.bl):
            for shape, is_a in zip(self.per, [True, False, True, False]):
                parts.append((torch.randn(shape) / h13.RANK).flatten() if is_a
                             else torch.zeros(shape).flatten())
        return torch.cat(parts)

    def unpack(self, theta):
        B = theta.shape[0]
        ch = theta.split(self.sizes, -1)
        lora, i = [], 0
        for _ in range(self.bl):
            Ao = ch[i].view(B, *self.per[0]); Bo = ch[i + 1].view(B, *self.per[1])
            Af = ch[i + 2].view(B, *self.per[2]); Bf = ch[i + 3].view(B, *self.per[3])
            lora.append(((Ao, Bo), (Af, Bf))); i += 4
        return lora

    def forward(self, ctx):
        x = self.emb(ctx) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x).mean(1)) + self.t0

def train_hyper(base, hyper, train_d, steps, B=32, lr=3e-4):
    opt = torch.optim.AdamW(hyper.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    for i in range(steps):
        ctx, tgt = sample_contig(train_d, B)
        opt.zero_grad(set_to_none=True)
        chunk_bpc(base(tgt[:, :-1], lora=hyper.unpack(hyper(ctx)), scale=SCALE),
                  tgt[:, 1:]).backward()
        nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)
        opt.step(); sched.step()
    return hyper

@torch.no_grad()
def evaluate(base, hyper, data, nbatch=16, B=32):
    zs = ic = hl = 0.0
    for _ in range(nbatch):
        ctx, tgt = sample_contig(data, B)
        zs += chunk_bpc(base(tgt[:, :-1]), tgt[:, 1:]).item()
        seq = torch.cat([ctx, tgt[:, 1:]], 1)
        ic += chunk_bpc(base(seq[:, :-1])[:, CTX - 1:], tgt[:, 1:]).item()
        hl += chunk_bpc(base(tgt[:, :-1], lora=hyper.unpack(hyper(ctx)), scale=SCALE),
                        tgt[:, 1:]).item()
    return zs / nbatch, ic / nbatch, hl / nbatch

if __name__ == "__main__":
    train_d, val_d, test_d = load_enwik8()
    out = []
    for d, L in CONFIGS:
        t0 = time.perf_counter()
        base = build_lm(d, L)
        nparam = sum(p.numel() for p in base.parameters())
        pretrain(base, train_d, val_d, steps=6000)
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
        hyper = HyperLoRA(d, L).to(DEV)
        train_hyper(base, hyper, train_d, steps=3000)
        zs, ic, hl = evaluate(base, hyper, test_d)
        cap = (zs - hl) / (zs - ic) * 100 if zs > ic else float("nan")
        row = {"d": d, "layers": L, "params": nparam, "zero_shot": zs,
               "in_context": ic, "hyper_lora": hl, "pct_capture": cap}
        out.append(row)
        print(f"base {nparam/1e6:.1f}M (d{d} L{L}): zs {zs:.3f} ic {ic:.3f} "
              f"hl {hl:.3f} -> capture {cap:.0f}%  ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        json.dump(out, open("exp23_results.json", "w"))
    print("SCALING:", " ".join(f"{r['params']/1e6:.1f}M:{r['pct_capture']:.0f}%" for r in out))
