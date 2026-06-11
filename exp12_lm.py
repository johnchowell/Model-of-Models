import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
VOCAB = 256
SOURCES = ["shakespeare", "alice", "python", "c_headers"]

def load_corpus(val_frac=0.1):
    train, val = {}, {}
    for s in SOURCES:
        b = torch.frombuffer(open(f"data/{s}.txt", "rb").read(), dtype=torch.uint8).long()
        cut = int(len(b) * (1 - val_frac))
        train[s], val[s] = b[:cut], b[cut:]
    return train, val

def sample_lm_batch(data, B, T, device=DEV):
    srcs = list(data.values())
    w = torch.tensor([len(s) for s in srcs], dtype=torch.float)
    out = torch.empty(B, T + 1, dtype=torch.long)
    si = torch.multinomial(w, B, replacement=True)
    for i, j in enumerate(si):
        s = srcs[j]
        o = torch.randint(0, len(s) - T - 1, (1,)).item()
        out[i] = s[o:o + T + 1]
    return out.to(device)

def lora_delta(x, A, B, scale):

    return torch.einsum("btr,bdr->btd", torch.einsum("btd,brd->btr", x, A), B) * scale

class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, lo=None, scale=1.0):
        Bsz, T, d = x.shape
        q, k, v = self.qkv(x).split(d, -1)
        sh = (Bsz, T, self.h, self.dh)
        q, k, v = [z.view(sh).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(Bsz, T, d)
        out = self.o(y)
        if lo is not None:
            out = out + lora_delta(y, lo[0], lo[1], scale)
        return out

class LMBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, lo=None, lf=None, scale=1.0):
        x = x + self.attn(self.n1(x), lo, scale)
        h = F.gelu(self.fc1(self.n2(x)))
        m = self.fc2(h)
        if lf is not None:
            m = m + lora_delta(h, lf[0], lf[1], scale)
        return x + m

class LM(nn.Module):
    def __init__(self, d=256, heads=8, layers=6, T=256):
        super().__init__()
        self.T = T
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        self.blocks = nn.ModuleList([LMBlock(d, heads) for _ in range(layers)])
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, idx, lora=None, scale=1.0):

        x = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for j, b in enumerate(self.blocks):
            lo, lf = lora[j] if lora is not None else (None, None)
            x = b(x, lo, lf, scale)
        return self.head(self.nf(x))

def bpc(model, data, nbatch=20, B=64, T=256):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(nbatch):
            xb = sample_lm_batch(data, B, T)
            loss = F.cross_entropy(model(xb[:, :-1]).flatten(0, 1), xb[:, 1:].flatten())
            tot += loss.item()
    model.train()
    return tot / nbatch / math.log(2)

def pretrain(steps=6000, B=32, T=768, lr=3e-4):

    torch.manual_seed(0)
    train, val = load_corpus()
    model = LM(T=T).to(DEV)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0 = time.perf_counter()
    for i in range(steps):
        xb = sample_lm_batch(train, B, T)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb[:, :-1]).flatten(0, 1), xb[:, 1:].flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (i + 1) % 500 == 0:
            v = bpc(model, val)
            print(f"step {i + 1}  train CE {loss.item():.3f}  val bpc {v:.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            torch.save(model.state_dict(), "base_lm_ckpt.pt")
    torch.save(model.state_dict(), "base_lm.pt")
    per = {s: bpc(model, {s: val[s]}) for s in SOURCES}
    print("final val bpc per source:", {k: round(v, 3) for k, v in per.items()})
    with open("exp12_results.json", "w") as f:
        json.dump(per, f)

if __name__ == "__main__":
    pretrain()
