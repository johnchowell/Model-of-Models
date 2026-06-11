import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_synthgrad import DEV, NCLASS, GradPredictor, Net, make_batch

torch.manual_seed(0)

class CtxGradPredictor(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * d + NCLASS, hidden), nn.GELU(),
                               nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, h, y):
        lab = F.one_hot(y, NCLASS).float().unsqueeze(1).expand(-1, h.shape[1], -1)
        ctx = h.mean(1, keepdim=True).expand_as(h)
        return self.f(torch.cat([h, ctx, lab], -1))

def true_grads(net, x, y):
    hs, h = [], x
    for s in net.stages:
        h = s(h); h.retain_grad(); hs.append(h)
    loss = F.cross_entropy(net.head(h.mean(1)), y)
    loss.backward()
    return [h.detach() for h in hs], [h.grad.detach() for h in hs], loss.item()

def fit_predictors(net, make_sg, steps=2000, B=128, T=32, lr=3e-3, normalize=True):
    sgs = nn.ModuleList([make_sg() for _ in net.stages]).to(DEV)
    opt = torch.optim.Adam(sgs.parameters(), lr=lr)
    rms = [None] * len(sgs)
    for i in range(steps):
        x, y = make_batch(B, T)
        hs, gs, _ = true_grads(net, x, y)
        net.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for j, (sg, h, g) in enumerate(zip(sgs, hs, gs)):
            r = g.pow(2).mean().sqrt()
            rms[j] = r if rms[j] is None else 0.99 * rms[j] + 0.01 * r
            tgt = g / rms[j] if normalize else g
            loss = loss + F.mse_loss(sg(h, y), tgt)
        loss.backward()
        opt.step()

    coss = [[] for _ in sgs]
    for _ in range(20):
        x, y = make_batch(B, T)
        hs, gs, _ = true_grads(net, x, y)
        net.zero_grad(set_to_none=True)
        with torch.no_grad():
            for j, (sg, h, g) in enumerate(zip(sgs, hs, gs)):
                tgt = g / rms[j] if normalize else g
                coss[j].append(F.cosine_similarity(sg(h, y).flatten(1),
                                                   tgt.flatten(1), -1).mean())
    return [torch.stack(c).mean().item() for c in coss], [r.item() for r in rms]

if __name__ == "__main__":

    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for i in range(1500):
        x, y = make_batch(128, 32)
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
    for p in net.parameters():
        p.requires_grad_(True)

    x, y = make_batch(128, 32)
    _, gs, loss = true_grads(net, x, y)
    net.zero_grad(set_to_none=True)
    print(f"frozen net loss {loss:.3f}; per-layer grad RMS:",
          [f"{g.pow(2).mean().sqrt().item():.2e}" for g in gs])

    for name, mk, norm in [
        ("exp1 predictor, raw targets", lambda: GradPredictor(64), False),
        ("exp1 predictor, normalized", lambda: GradPredictor(64), True),
        ("ctx predictor, normalized", lambda: CtxGradPredictor(64), True),
    ]:
        cos, rms = fit_predictors(net, mk, normalize=norm)
        print(f"[{name:32}] per-layer cosine: " + "  ".join(f"{c:.3f}" for c in cos))
