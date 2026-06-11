import json

import torch

from exp14_sinusoid import eval_mom, maml_eval, maml_train, train_mom

if __name__ == "__main__":
    out = {}
    for mode in ["direct", "film"]:
        ms = []
        for seed in [1, 2, 3, 4, 5]:
            net = train_mom(mode, seed=seed)
            ms.append(eval_mom(net))
            print(f"  mom-{mode} seed {seed}: {ms[-1]:.5f}", flush=True)
        t = torch.tensor(ms)
        out[f"mom_{mode}"] = {"mean": t.mean().item(), "std": t.std().item(), "all": ms}
        print(f"[MoM {mode}] {t.mean():.5f} +/- {t.std():.5f}")
    for seed in [1, 2, 3]:
        net, adapted, _ = maml_train(seed=seed)
        for es in [5]:
            out.setdefault("maml_5step", []).append(maml_eval(net, adapted, es))
            print(f"  maml seed {seed}: {out['maml_5step'][-1]:.4f}", flush=True)
    t = torch.tensor(out["maml_5step"])
    print(f"[MAML 5-step] {t.mean():.4f} +/- {t.std():.4f}")
    with open("exp14b_results.json", "w") as f:
        json.dump(out, f)
