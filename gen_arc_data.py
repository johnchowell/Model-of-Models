import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "data/re-arc")
import generators

MAXS, PER_TASK, TRIES = 12, 400, 4000

def gen_task(key):
    g = getattr(generators, f"generate_{key}")
    exs = []
    rng_failures = 0
    for i in range(TRIES):
        if len(exs) >= PER_TASK:
            break
        try:
            ex = g(0, 1)
            inp, out = np.array(ex["input"], dtype=np.int8), np.array(ex["output"], dtype=np.int8)
            if max(inp.shape + out.shape) <= MAXS:
                exs.append((inp, out))
        except Exception:
            rng_failures += 1
            if rng_failures > 500:
                break
    return key, exs

if __name__ == "__main__":
    keys = sorted(json.load(open("data/arc_keys.json")) if os.path.exists("data/arc_keys.json")
                  else [n[len("generate_"):] for n in dir(generators) if n.startswith("generate_")])
    os.makedirs("data/arc_gen", exist_ok=True)
    with Pool(28) as p:
        for j, (key, exs) in enumerate(p.imap_unordered(gen_task, keys)):
            if len(exs) >= 20:
                np.savez_compressed(
                    f"data/arc_gen/{key}.npz",
                    inputs=np.array([np.pad(i, ((0, MAXS - i.shape[0]), (0, MAXS - i.shape[1])),
                                            constant_values=-1) for i, _ in exs]),
                    outputs=np.array([np.pad(o, ((0, MAXS - o.shape[0]), (0, MAXS - o.shape[1])),
                                             constant_values=-1) for _, o in exs]))
            if (j + 1) % 50 == 0:
                print(f"{j + 1}/400 tasks", flush=True)
    n = len(os.listdir("data/arc_gen"))
    print(f"done: {n} tasks with >=20 small examples")
