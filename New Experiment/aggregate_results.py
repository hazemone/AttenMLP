#!/usr/bin/env python3
# =====================================================================
#  aggregate_results.py  --  multi-seed statistics for AttenMLP
# ---------------------------------------------------------------------
#  Parses the out.txt files produced by your training runs, extracts
#  the final OA (Best Test acc) and mAcc (Best Test acc_B) of each run,
#  and reports:
#     * mean +/- standard deviation across seeds
#     * 95% confidence interval (Student-t)
#     * (optional) a paired t-test of AttenMLP vs a baseline, matched
#       by seed, with Cohen's d effect size
#  It also prints a ready-to-paste LaTeX snippet.
#
#  This is what satisfies Reviewer #1's request for statistical
#  significance / confidence intervals.
#
#  USAGE
#  -----
#  Single method (mean +/- std + 95% CI):
#     python aggregate_results.py \
#         --name "AttenMLP" --glob "checkpoints/AttenMLP-seed*"
#
#  Two methods, paired test (AttenMLP vs DualMLP over the SAME seeds):
#     python aggregate_results.py \
#         --name "AttenMLP" --glob "checkpoints/AttenMLP-seed*" \
#         --baseline-name "DualMLP" --baseline-glob "checkpoints/DualMLP-seed*"
#
#  Notes:
#   * Point --glob at the run directories (each must contain out.txt).
#   * The seed is parsed from the trailing  -<number>  of the dir name
#     (that is how main.py names checkpoints), so runs are matched by
#     seed for the paired test.
# =====================================================================

import argparse
import glob
import math
import os
import re
import statistics
import sys

OA_RE   = re.compile(r"\+\+\s*Best Train acc:\s*[0-9.]+\s*\|\s*Best Test acc:\s*([0-9.]+)\s*\+\+")
MACC_RE = re.compile(r"\+\+\s*Best Train acc_B:\s*[0-9.]+\s*\|\s*Best Test acc_B:\s*([0-9.]+)\s*\+\+")
SEED_RE = re.compile(r"-(\d+)\s*$")

try:
    from scipy import stats as _scs
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# t critical values (two-sided, 95%) for small dof, fallback when no scipy
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        14: 2.145, 19: 2.093, 29: 2.045}


def t_crit_95(dof):
    if HAVE_SCIPY:
        return float(_scs.t.ppf(0.975, dof))
    if dof in _T95:
        return _T95[dof]
    keys = sorted(_T95)
    return _T95[min(keys, key=lambda k: abs(k - dof))]


def parse_run(run_dir):
    out = os.path.join(run_dir, "out.txt")
    if not os.path.isfile(out):
        # also accept log.txt as fallback
        out = os.path.join(run_dir, "log.txt")
        if not os.path.isfile(out):
            return None
    text = open(out, "r", errors="ignore").read()
    oa = OA_RE.findall(text)
    macc = MACC_RE.findall(text)
    if not oa or not macc:
        return None
    seed = None
    m = SEED_RE.search(os.path.basename(run_dir.rstrip("/")))
    if m:
        seed = int(m.group(1))
    return {"dir": run_dir, "seed": seed,
            "OA": float(oa[-1]), "mAcc": float(macc[-1])}


def collect(pattern):
    runs = []
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        r = parse_run(d)
        if r:
            runs.append(r)
    return runs


def summarise(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = t_crit_95(n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
    return n, mean, sd, half


def report_metric(label, runs, key):
    vals = [r[key] for r in runs]
    n, mean, sd, half = summarise(vals)
    seeds = [r["seed"] for r in runs]
    print("  %-5s : %.2f +/- %.2f   (95%% CI +/- %.2f, n=%d)   values=%s seeds=%s"
          % (label, mean, sd, half, n,
             ["%.2f" % v for v in vals], seeds))
    return mean, sd, half, n


def paired_test(a_runs, b_runs, key):
    a = {r["seed"]: r[key] for r in a_runs if r["seed"] is not None}
    b = {r["seed"]: r[key] for r in b_runs if r["seed"] is not None}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    da = [a[s] for s in common]
    db = [b[s] for s in common]
    diffs = [x - y for x, y in zip(da, db)]
    mean_d = statistics.mean(diffs)
    sd_d = statistics.stdev(diffs)
    n = len(diffs)
    if sd_d == 0:
        return {"seeds": common, "mean_diff": mean_d, "t": float("inf"),
                "p": 0.0, "cohen_d": float("inf"), "n": n}
    t = mean_d / (sd_d / math.sqrt(n))
    cohen_d = mean_d / sd_d  # paired Cohen's d
    if HAVE_SCIPY:
        tt = _scs.ttest_rel(da, db)
        t, p = float(tt.statistic), float(tt.pvalue)
    else:
        p = None
    return {"seeds": common, "mean_diff": mean_d, "t": t,
            "p": p, "cohen_d": cohen_d, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="AttenMLP")
    ap.add_argument("--glob", required=True, help="glob of run dirs, e.g. 'checkpoints/AttenMLP-seed*'")
    ap.add_argument("--baseline-name", default=None)
    ap.add_argument("--baseline-glob", default=None)
    args = ap.parse_args()

    runs = collect(args.glob)
    if not runs:
        print("No completed runs matched %r (need out.txt with final '++ Best Test acc' lines)."
              % args.glob)
        sys.exit(1)

    print("=" * 70)
    print(" Method: %s   (%d runs)" % (args.name, len(runs)))
    print("=" * 70)
    a_oa = report_metric("OA", runs, "OA")
    a_macc = report_metric("mAcc", runs, "mAcc")

    base_runs = None
    if args.baseline_glob:
        base_runs = collect(args.baseline_glob)
        if base_runs:
            print("\n" + "=" * 70)
            print(" Baseline: %s   (%d runs)" % (args.baseline_name or "Baseline", len(base_runs)))
            print("=" * 70)
            report_metric("OA", base_runs, "OA")
            report_metric("mAcc", base_runs, "mAcc")
        else:
            print("\n[warn] baseline glob %r matched no completed runs." % args.baseline_glob)

    # ---- LaTeX snippet ----------------------------------------------
    print("\n" + "-" * 70)
    print(" LaTeX (paste into the paper)")
    print("-" * 70)
    print("%% mean $\\pm$ std over %d seeds" % a_oa[3])
    print("\\textbf{%s (Ours)} & \\textbf{%.2f} $\\pm$ %.2f & \\textbf{%.2f} $\\pm$ %.2f \\\\"
          % (args.name, a_macc[0], a_macc[1], a_oa[0], a_oa[1]))
    print("%% columns above: mAcc(%%)  OA(%%)")

    # ---- significance ------------------------------------------------
    if base_runs:
        print("\n" + "-" * 70)
        print(" Paired t-test (matched by seed): %s vs %s"
              % (args.name, args.baseline_name or "Baseline"))
        print("-" * 70)
        for key, lab in (("OA", "OA"), ("mAcc", "mAcc")):
            res = paired_test(runs, base_runs, key)
            if res is None:
                print("  %-5s : not enough matched seeds for a paired test." % lab)
                continue
            pstr = ("p=%.4f" % res["p"]) if res["p"] is not None else "p=N/A (install scipy)"
            sig = ""
            if res["p"] is not None:
                sig = "  [significant at 0.05]" if res["p"] < 0.05 else "  [n.s. at 0.05]"
            print("  %-5s : mean diff=%+.2f  t(%d)=%.3f  %s  Cohen's d=%.2f%s"
                  % (lab, res["mean_diff"], res["n"] - 1, res["t"], pstr, res["cohen_d"], sig))
        if not HAVE_SCIPY:
            print("\n  [note] scipy not found -> p-values omitted. Run: pip install scipy")

    print("\nDone. Send this whole block back and I will insert the numbers")
    print("and the significance statement into the paper.")


if __name__ == "__main__":
    main()
