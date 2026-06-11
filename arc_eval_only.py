import json, sys, torch
import exp16_arc as a
mode = sys.argv[1]
net = a.ArcMoM(mode).to(a.DEV)
net.load_state_dict(torch.load(f"arc_{mode}.pt", map_location=a.DEV))
net.eval()
gen = a.ArcGen()
res = {}
with torch.no_grad():
    ctx, qi, qo = gen.batch(1000, val=True)
    res["generated_heldout_exact"] = a.exact_match(net(ctx, qi), qo).mean().item()
for split in ["training", "evaluation"]:
    items = a.load_real_arc(split)
    em, n = a.evaluate_real(net, items, B=16)
    res[f"real_{split}_exact"] = em; res[f"real_{split}_n"] = n
    print(f"[{mode}] real ARC {split}: exact {em:.4f} over {n} grids", flush=True)
print(f"[{mode}] gen-heldout exact {res['generated_heldout_exact']:.3f}", flush=True)
json.dump(res, open(f"exp16_results_{mode}.json","w"))
