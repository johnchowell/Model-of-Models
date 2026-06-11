# Datasets

Not vendored here (size / licensing). Obtain and place under `data/`.

- **enwik8**: `http://mattmahoney.net/dc/enwik8.zip` to `data/enwik8`
- **UCI HCV** (blood panel): UCI dataset 571 to `data/hcv/hcvdat0.csv`
- **TCGA PAN-CAN RNA-seq**: UCI dataset 401; run the prepare step in
  `exp19_genes.py` to produce `data/rnaseq/prepared.npz`
- **ARC-AGI**: `github.com/fchollet/ARC-AGI` to `data/arc`
- **RE-ARC**: `github.com/michaelhodel/re-arc` to `data/re-arc`; run
  `gen_arc_data.py` to produce `data/arc_gen/`
- **HalfCheetah-Vel**: provided by `gymnasium[mujoco]` (no download)
