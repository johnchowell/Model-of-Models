import json

import torch
import torch.nn.functional as F

import exp11_hypernet as e

e.CTX = 16
torch.manual_seed(1)

def main(steps=20000, B=256):
    net = e.HyperNet("film", d=192, layers=4).to(e.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=5e-5)
    hist = {"step": [], "cont": []}
    for i in range(steps):
        pts, t, _ = e.batch_with_t(B)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(net.generate(pts[:, :e.CTX], t[:, :e.CTX], t), pts)
        loss.backward()
        opt.step()
        sched.step()
        if (i + 1) % 500 == 0:
            _, c = e.evaluate(net)
            hist["step"].append(i + 1); hist["cont"].append(c)
    print(f"[film-big ctx16] continuation MSE {hist['cont'][-1]:.5f}  "
          f"best {min(hist['cont']):.5f}")
    torch.save(net.state_dict(), "exp11b_film_big.pt")

    ar_cont, _ = e.ar_baseline()
    with open("exp11b_results.json", "w") as f:
        json.dump({"film_big": hist, "ar_cont_mse": ar_cont}, f)

if __name__ == "__main__":
    main()
