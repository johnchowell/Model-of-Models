import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import exp13_hyperlora as h13
from exp12_lm import DEV, LM, sample_lm_batch
from exp13_hyperlora import (CTX, SCALE, TGT, HyperLoRA, chunk_bpc, eval_baselines,
                             eval_hyper, sample_doc_batch, tuned_lora_ceiling, unpack)

torch.manual_seed(0)

def load_enwik8():
    b = torch.frombuffer(bytearray(open("data/enwik8", "rb").read()), dtype=torch.uint8).long()
    n = len(b)
    return ({"train": b[:int(0.9 * n)]}, {"val": b[int(0.9 * n):int(0.95 * n)]},
            {"test": b[int(0.95 * n):]})

def pretrain(train_d, val_d, steps=8000, B=32, T=768, lr=3e-4):
    model = LM(T=T).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0 = time.perf_counter()
    for i in range(steps):
        xb = sample_lm_batch(train_d, B, T)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb[:, :-1]).flatten(0, 1), xb[:, 1:].flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (i + 1) % 500 == 0:
            with torch.no_grad():
                vb = sample_lm_batch(val_d, 32, 256)
                v = F.cross_entropy(model(vb[:, :-1]).flatten(0, 1),
                                    vb[:, 1:].flatten()).item() / math.log(2)
            print(f"pretrain {i + 1}  val bpc {v:.3f}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
            torch.save(model.state_dict(), "enwik8_lm_ckpt.pt")
    torch.save(model.state_dict(), "enwik8_lm.pt")
    return model

def train_hyper(base, train_d, val_d, steps=4000, B=32, lr=3e-4):
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    hyper = HyperLoRA().to(DEV)
    opt = torch.optim.AdamW(hyper.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
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
        if (i + 1) % 500 == 0:
            v = eval_hyper(base, hyper, val_d, nbatch=8)
            print(f"hyper {i + 1}  train bpc {loss.item():.3f}  val bpc {v:.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            torch.save(hyper.state_dict(), "enwik8_hyper_ckpt.pt")
    torch.save(hyper.state_dict(), "enwik8_hyper.pt")
    return hyper

if __name__ == "__main__":
    train_d, val_d, test_d = load_enwik8()
    print("== pretraining base on enwik8 ==", flush=True)
    base = pretrain(train_d, val_d)
    print("== training hyper-LoRA ==", flush=True)
    hyper = train_hyper(base, train_d, val_d)
    print("== final 4-way eval on TEST split ==", flush=True)
    zs, ic = eval_baselines(base, test_d, nbatch=16)
    hl = eval_hyper(base, hyper, test_d, nbatch=16)
    tl = tuned_lora_ceiling(base, test_d)
    print(f"enwik8 test bits/char:  zero-shot {zs:.3f}   in-context {ic:.3f}   "
          f"hyper-LoRA {hl:.3f}   tuned-LoRA {tl:.3f}")
    with open("exp15_results.json", "w") as f:
        json.dump({"zero_shot": zs, "in_context": ic, "hyper_lora": hl,
                   "tuned_lora": tl}, f)
