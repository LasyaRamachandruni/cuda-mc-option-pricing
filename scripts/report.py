#!/usr/bin/env python3
"""Turn a benchmark CSV (from ./mc or ./mc_cpu) into a markdown table and a chart.

Usage: python3 scripts/report.py results/results.csv
Writes results/<name>.md next to the CSV, and results/<name>.png if matplotlib
is installed.
"""
import csv
import math
import os
import sys


def num(x):
    try:
        v = float(x)
    except ValueError:
        return math.nan
    return v


def ms(v):
    if math.isnan(v):
        return "–"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms"


def speedup(v):
    return "–" if math.isnan(v) else f"{v:.1f}x"


def main(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no rows")
    r0 = rows[0]
    has_gpu = r0["gpu_name"] != "none"
    threads = r0["cpu_threads"]
    steps = r0["steps"]

    lines = []
    label = f", {r0['label']}" if r0["label"] else ""
    lines.append(f"**Hardware{label}:** CPU {r0['cpu_model']}, {threads} threads visible to OpenMP"
                 + (f"; GPU {r0['gpu_name']}" if has_gpu else "; no GPU") + ".  ")
    lines.append(f"**Workload:** {steps} time steps per path; times are the fastest of "
                 "several runs. Prices are mean ± 1 standard error; 95% CI = ± 1.96 SE.")
    lines.append("")
    head = ["Paths", "CPU scalar 1T", "CPU SIMD 1T", f"CPU SIMD+OpenMP ({threads}T)"]
    if has_gpu:
        head += ["GPU (CUDA)", "GPU vs scalar 1T", f"GPU vs SIMD+OpenMP"]
    else:
        head += ["SIMD+OpenMP vs scalar 1T"]
    head += [f"European call (BS {num(r0['bs']):.4f})", "EU distance from BS (SEs)", "Asian call"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for r in rows:
        cells = [f"{int(r['paths']):,}", ms(num(r["scalar_1t_ms"])), ms(num(r["simd_1t_ms"])),
                 ms(num(r["simd_omp_ms"]))]
        if has_gpu:
            cells += [ms(num(r["gpu_total_ms"])), speedup(num(r["gpu_vs_scalar_1t"])),
                      speedup(num(r["gpu_vs_simd_omp"]))]
        else:
            cells += [speedup(num(r["simd_omp_vs_scalar"]))]
        cells += [f"{num(r['eu_simd']):.4f} ± {num(r['eu_se']):.4f}", f"{num(r['eu_z_bs']):.2f}",
                  f"{num(r['asian_simd']):.4f} ± {num(r['asian_se']):.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"

    base = os.path.splitext(path)[0]
    with open(base + ".md", "w") as f:
        f.write(table)
    print(table)
    print(f"wrote {base}.md")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart")
        return

    paths = [int(r["paths"]) for r in rows]
    fig, ax = plt.subplots(1, 2 if has_gpu else 1, figsize=(11 if has_gpu else 6, 4))
    ax0 = ax[0] if has_gpu else ax
    for col, name in [("scalar_1t_ms", "CPU scalar, 1 thread"), ("simd_1t_ms", "CPU SIMD, 1 thread"),
                      ("simd_omp_ms", f"CPU SIMD + OpenMP, {threads} threads")] + (
                         [("gpu_total_ms", f"GPU ({r0['gpu_name']})")] if has_gpu else []):
        ax0.loglog(paths, [num(r[col]) for r in rows], "o-", label=name)
    ax0.set_xlabel(f"paths ({steps} steps each)")
    ax0.set_ylabel("time (ms)")
    ax0.set_title("Runtime")
    ax0.legend()
    if has_gpu:
        ax[1].semilogx(paths, [num(r["gpu_vs_scalar_1t"]) for r in rows], "o-",
                       label="vs CPU scalar, 1 thread")
        ax[1].semilogx(paths, [num(r["gpu_vs_simd_omp"]) for r in rows], "o-",
                       label=f"vs CPU SIMD + OpenMP, {threads} threads")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("paths")
        ax[1].set_ylabel("GPU speedup (x)")
        ax[1].set_title("GPU speedup")
        ax[1].legend()
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=150)
    print(f"wrote {base}.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
