# AttenMLP — Experiment Instructions (ICAC 2026 camera-ready)

This pack contains four scripts that produce everything the reviewers asked for:

| File | Reviewer point it answers | What it does |
|---|---|---|
| `inspect_arch.py` | R2 — reproducibility (centroids, K, dims, heads, fusion) | Prints the **real** architecture configuration of your model |
| `run_modelnet40_seeds.sh` | R1 — statistical significance / CIs | Trains AttenMLP (and DualMLP) over several seeds on ModelNet40 |
| `run_scanobjectnn_seeds.sh` | R3 + external — harder dataset | Trains AttenMLP (and DualMLP) on ScanObjectNN |
| `aggregate_results.py` | R1 — mean±std, 95% CI, significance | Parses all runs, computes statistics + paired t-test, prints a LaTeX snippet |

Your setup (32 GB RAM, RTX 4080, Ubuntu 24.04) is more than enough. Base code: the `snehaputul/DualMLP` repo you used for the paper.

Nothing below invents results. You run the experiments; the scripts collect the real numbers; send them back and I drop them into the paper.

---

## 0. One-time setup

```bash
# your AttenMLP working copy (the DualMLP fork you built the paper on)
cd /path/to/DualMLP

# environment (use the one you already trained with; otherwise:)
conda env create -f environment.yml   # or your existing env
conda activate <your-env>

# stats dependency for the aggregator
pip install scipy
```

Copy the four files from this pack into the repo:

```bash
# inspector + ModelNet40 runner + aggregator
cp inspect_arch.py run_modelnet40_seeds.sh aggregate_results.py  classification_ModelNet40/
# ScanObjectNN runner + aggregator
cp run_scanobjectnn_seeds.sh aggregate_results.py               classification_ScanObjectNN/
```

---

## 1. Get the architecture numbers (Reviewer #2) — ~1 minute

This is the task that replaces guessing. It reads the true configuration out of the model you build.

```bash
cd classification_ModelNet40

# IMPORTANT: pass the SAME architecture args you used for the paper run.
# If unsure, open  checkpoints/<any-old-run>/args.txt  — it lists them exactly.
python inspect_arch.py \
    --model AttenMLP \
    --dual_net True --last_layer_concat add \
    --num_points_low 1024 --neighbours_low 32 \
    --num_points_high 2048 --neighbours_high 32 \
    --num_channel 32
```

It prints, for each pathway: input points, embedding dim, **centroids per stage (M)**, **K-neighbours per stage**, feature width per stage, classifier head, **attention heads**, and total parameters.

➡ **Copy the whole printout back to me.** That fills the reproducibility table directly. (If the attention head count is not auto-detected — e.g. your attention class has an unusual name — just tell me the number of heads you used; the paper says 8 is optimal.)

---

## 2. ModelNet40 multi-seed (Reviewer #1) — main run

Open `run_modelnet40_seeds.sh` and **edit `attenmlp_cmd()`** so it is *exactly* the command that produced your 93.85 % result, keeping `$SEED` where the seed goes. Do the same sanity-check on `dualmlp_cmd()` (the baseline). Both are pre-filled with batch 16 / lr 0.025→0.005 / 100 epochs to match Table I — adjust the architecture args to yours.

```bash
cd classification_ModelNet40
bash run_modelnet40_seeds.sh 2>&1 | tee modelnet_sweep.log
```

This runs 5 AttenMLP seeds, then 5 DualMLP-baseline seeds (set `RUN_BASELINE="false"` at the top if you only want AttenMLP). Runs land in `checkpoints/AttenMLP-seed-<n>/` and `checkpoints/pointMLP-dualbase-seed-<n>/`.

**Rough time on an RTX 4080:** ~1.5–3 h per run at 100 epochs → about 15–30 h for all 10. You can stop after 3 AttenMLP seeds if time is tight (3 is the minimum for a CI); the baseline is only needed for the significance test.

---

## 3. ScanObjectNN (Reviewer #3 + external) — harder benchmark

```bash
cd classification_ScanObjectNN
# data downloads automatically on first run into ./data/h5_files/
bash run_scanobjectnn_seeds.sh 2>&1 | tee scanobjectnn_sweep.log
```

Uses the hardest **PB_T50_RS** split (the repo's default). 3 seeds for AttenMLP and 3 for the baseline.
**Rough time:** ~2–4 h per run at 200 epochs → about 12–24 h for all 6.

If the auto-download is slow, grab `h5_files.zip` and unzip it so this file exists:
`classification_ScanObjectNN/data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5`

---

## 4. Aggregate everything

```bash
# ModelNet40
cd classification_ModelNet40
python aggregate_results.py \
    --name "AttenMLP" --glob "checkpoints/AttenMLP-seed-*" \
    --baseline-name "DualMLP" --baseline-glob "checkpoints/pointMLP-dualbase-seed-*"

# ScanObjectNN
cd ../classification_ScanObjectNN
python aggregate_results.py \
    --name "AttenMLP (ScanObjectNN)" --glob "checkpoints/AttenMLP-sonn-seed-*" \
    --baseline-name "DualMLP" --baseline-glob "checkpoints/pointMLP-sonn-dualbase-seed-*"
```

Each prints mean ± std, a 95 % CI, a paired t-test (with p-value and Cohen's d), and a ready-to-paste LaTeX row.

---

## What to send me

1. The **`inspect_arch.py` printout** (Step 1).
2. The **two `aggregate_results.py` printouts** (Step 4) — ModelNet40 and ScanObjectNN.

With those three blocks I will:
- fill the reproducibility / configuration table (R2),
- add the mean ± std and the significance statement to the results (R1),
- add a ScanObjectNN results table and update the conclusion (R3 / external),
- and finalise the camera-ready PDF + the point-by-point response-to-reviewers document.

---

### Minimum viable version (if you are short on time)

- **Must do:** Step 1 (1 min) + Step 2 with `SEEDS=(0 1 2)` and `RUN_BASELINE="false"` → gives reproducibility table + a 3-seed CI. That alone addresses R2 and most of R1.
- **Strongly recommended:** add the baseline in Step 2 (paired t-test → a real significance claim) and Step 3 (ScanObjectNN → second dataset for R3).
