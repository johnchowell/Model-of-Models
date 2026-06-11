import json
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

import exp18_bloodtest as blood
import exp19_genes as genes

DEV = "cpu"

def blood_tabpfn(ntask=400, nq=32):
    X, y, _ = blood.load()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs, bas, times = [], [], []
    for tr, te in skf.split(X, y):
        d = blood.FewShotData(X, y, tr)
        d.pos = te[y[te] == 1]; d.neg = te[y[te] == 0]
        if len(d.pos) < 2:
            continue
        ys, ps, t0 = [], [], time.perf_counter()
        for _ in range(ntask):
            sx, sy, qx, qy = d.task(1, nq=nq, device="cpu")
            sx, sy, qx, qy = [a.numpy()[0] for a in (sx, sy, qx, qy)]
            clf = TabPFNClassifier(device=DEV)
            clf.fit(sx, sy.astype(int))
            ps.append(clf.predict_proba(qx)[:, 1]); ys.append(qy)
        times.append((time.perf_counter() - t0) / ntask * 1000)
        y_, p_ = np.concatenate(ys), np.concatenate(ps)
        aucs.append(roc_auc_score(y_, p_)); bas.append(balanced_accuracy_score(y_, p_ > 0.5))
    return {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "bal_acc": float(np.mean(bas)), "ms_per_task": float(np.mean(times))}

def gene_tabpfn(ntask=200, nq=5, npca=20):
    X, y, _ = genes.load()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    accs, times = [], []
    for tr, te in skf.split(X, y):
        d = genes.GeneFewShot(X, y, tr); d.set_pools(te)
        if min(len(v) for v in d.by.values()) < 2:
            continue
        acc, t0 = [], time.perf_counter()
        for _ in range(ntask):
            sx, sy, qx, qy = d.task(1, nq=nq, device="cpu")
            sx, sy, qx, qy = [a.numpy()[0] for a in (sx, sy, qx, qy)]
            pca = PCA(npca).fit(sx)
            clf = TabPFNClassifier(device=DEV)
            clf.fit(pca.transform(sx), sy.astype(int))
            acc.append(accuracy_score(qy, clf.predict(pca.transform(qx))))
        times.append((time.perf_counter() - t0) / ntask * 1000)
        accs.append(np.mean(acc))
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "ms_per_task": float(np.mean(times)), "note": f"PCA->{npca} (TabPFN feat limit)"}

if __name__ == "__main__":
    out = {}
    print("== TabPFN on blood (16-shot) ==", flush=True)
    out["blood"] = blood_tabpfn()
    print(f"  AUC {out['blood']['auc']:.3f}+/-{out['blood']['auc_std']:.3f}  "
          f"bal-acc {out['blood']['bal_acc']:.3f}  {out['blood']['ms_per_task']:.0f} ms/task",
          flush=True)
    print("== TabPFN on genes (5-shot, PCA-50) ==", flush=True)
    out["gene"] = gene_tabpfn()
    print(f"  acc {out['gene']['acc']:.3f}+/-{out['gene']['acc_std']:.3f}  "
          f"{out['gene']['ms_per_task']:.0f} ms/task", flush=True)
    with open("exp21_results.json", "w") as f:
        json.dump(out, f)
