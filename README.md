# CUDA Monte Carlo Option Pricing

[![CPU test](https://github.com/LasyaRamachandruni/cuda-mc-option-pricing/actions/workflows/cpu-test.yml/badge.svg)](https://github.com/LasyaRamachandruni/cuda-mc-option-pricing/actions/workflows/cpu-test.yml)

Prices options by simulating stock-price paths with geometric Brownian motion, on the CPU and on the GPU (CUDA), and benchmarks them against each other.

- **European call:** checked against the Black-Scholes closed form (the result should land within ~3 standard errors).
- **Arithmetic-average Asian call:** path-dependent, with no closed form, so every path simulates all 252 daily steps. Checked against an independent control-variate reference in the test.

## Implementations

| | What it is | Where |
|---|---|---|
| CPU scalar, 1 thread | one path at a time; the original baseline, kept as a reference | `mc_core.h` |
| CPU SIMD, 1 thread | 32 paths stepped together so the inner loop vectorizes (AVX2/AVX-512, glibc libmvec) | `cpu_simd.cpp` |
| CPU SIMD + OpenMP | the SIMD version on all cores | `cpu_simd.cpp` |
| GPU | CUDA kernel, one path per thread, shared-memory reduction | `mc_option.cu` |

GPU speedups are reported against **both** the scalar single-thread baseline and the SIMD + OpenMP baseline. The second one is the fair comparison.

## How it works

- **Same paths everywhere.** Each random draw is a hash of `(seed, path, step)` (splitmix64), turned into normals with Box-Muller. There's no RNG state, so path *i* is identical in all four implementations, and their prices must agree to float rounding. The benchmark checks this on every run (it exits non-zero if any pair differs by more than 0.1 standard errors).
- **SIMD CPU baseline.** The time-step loop is a recurrence and can't vectorize, so `cpu_simd.cpp` vectorizes across paths instead: each OpenMP thread takes a block of 32 paths and advances them one step at a time. Built with `-O3 -march=native -ffast-math -fno-associative-math` (fast-math only in this file, so GCC calls the vector `expf/logf/sinf/cosf`). Without `-fno-associative-math`, the AVX2 build had a small systematic bias (+6e-5 relative); the test now catches that.
- **GPU kernel.** A grid-stride loop has each thread simulate many paths and keep running sums of the payoff and payoff². A shared-memory tree reduction gives one partial result per block, and the host adds up those few hundred partials. The grid size comes from the occupancy API (`SMs × active blocks per SM × 4`).
- **Timing.** Each timing is the fastest of several runs (`--reps N` sets the minimum). GPU time uses CUDA events and covers the kernel plus the copy back, after a warm-up launch. CPU time uses `std::chrono`, after the OpenMP thread pool has been warmed up.
- **Accuracy.** Paths are simulated in float and payoffs summed in double. Every price is reported with its standard error (SE); the 95% confidence interval is ± 1.96 SE.

## Results

**GPU numbers are pending a re-run against the multi-core SIMD baseline.** The earlier T4 run compared the GPU only to the scalar CPU code, which overstated the speedup, so those numbers have been withdrawn from this README. (The raw CSV is kept in `results/archive/` for reference only.)

### CPU only (measured)

2-vCPU sandbox, Intel Xeon @ 2.10 GHz (AVX-512 capable; GCC used 256-bit AVX2 vectors), no GPU, 252 steps per path, fastest of 3 runs. From `results/cpu_sandbox.csv`:

| Paths | CPU scalar 1T | CPU SIMD 1T | CPU SIMD + OpenMP (2T) | SIMD + OpenMP vs scalar | European call ± SE (BS 10.4506) | Asian call ± SE |
|---|---|---|---|---|---|---|
| 65,536 | 269.9 ms | 36.9 ms | 19.2 ms | 14.1x | 10.3635 ± 0.0572 | 5.7579 ± 0.0312 |
| 262,144 | 1.06 s | 159.6 ms | 78.1 ms | 13.5x | 10.4462 ± 0.0287 | 5.7923 ± 0.0156 |
| 1,048,576 | 4.49 s | 651.8 ms | 365.2 ms | 12.3x | 10.4550 ± 0.0144 | 5.7816 ± 0.0078 |
| 4,194,304 | 18.27 s | 2.58 s | 1.35 s | 13.5x | 10.4512 ± 0.0072 | 5.7801 ± 0.0039 |

- Vectorizing alone is ~7x faster than the scalar code on one thread; OpenMP on 2 vCPUs adds ~1.9x.
- At 4.2M paths, the European price is 0.09 SE from Black-Scholes (95% CI 10.4371 – 10.4653, which contains 10.4506).
- The scalar and SIMD prices agree to within 0.001 SE at every size (same paths).
- This was measured on a shared sandbox, so timings vary from run to run. They are only comparable with each other, not with any GPU number from a different machine.

### GPU (to be measured)

Steps to produce the GPU table (Colab, free T4):

1. Open `run_benchmark_colab.ipynb` in Colab (File → Open notebook → GitHub → this repo).
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. Runtime → **Run all**. The notebook clones the repo, runs `make mc test_cpu`, runs `./test_cpu`, then `./mc --reps 3 --label colab --csv results/results.csv`, then `python3 scripts/report.py results/results.csv`.
4. The last cell downloads `colab_results.zip` with `results.csv` (including CPU model, thread count and GPU name), `results.md` (the table) and `results.png`. Commit those three files to `results/` and paste `results.md` here.

Note that Colab's 2 vCPUs are usually two hyperthreads of one physical core, so even the SIMD + OpenMP baseline there is a small CPU. The speedup vs SIMD + OpenMP is the number to quote, together with the core count. A GPU vs a full many-core server CPU would be a smaller ratio again.

## Run it

**CPU only (any Linux machine with g++ ≥ 9):**

```bash
make            # builds mc_cpu (benchmark) and test_cpu
make test       # correctness test, about 5 s
make bench-cpu  # CPU benchmark -> results/cpu_only.csv
python3 scripts/report.py results/cpu_only.csv   # markdown table + chart
```

**With an NVIDIA GPU and the CUDA toolkit:**

```bash
make mc test_cpu
./test_cpu
mkdir -p results && ./mc --reps 3 --csv results/results.csv
python3 scripts/report.py results/results.csv
```

Flags: `--steps N`, `--min-paths N`, `--max-paths N`, `--max-scalar N` (skip the slow single-thread runs above N paths), `--reps N`, `--seed N`, `--label TEXT`, `--csv FILE`.

**Profiling (optional):** the notebook has cells for both; on another machine:

```bash
nsys profile --stats=true -o results/mc_nsys ./mc --max-paths 1048576 --max-scalar 0 --csv /tmp/p.csv
ncu --kernel-name mc_kernel --launch-skip 1 --launch-count 1 --section SpeedOfLight --section Occupancy \
    ./mc --min-paths 1048576 --max-paths 1048576 --max-scalar 0 --csv /tmp/p.csv
```

## Tests

`make test` (also run by GitHub Actions on every push) checks, without a GPU:

- European call from the scalar and SIMD + OpenMP pricers is within 3 SE of Black-Scholes.
- Scalar and SIMD prices on the same paths agree to < 0.01 SE.
- The arithmetic Asian price is within 3 combined SE of an independent reference. The reference is a double-precision simulation with a different RNG (`std::mt19937_64` streams) that uses the discretely monitored geometric Asian call (closed form, 5.5655) as a control variate, giving 5.7820 ± 0.0004.
- Geometric Asian < arithmetic Asian < European.

The GPU kernel calls the same `simulate_path()` as the scalar CPU code, and the benchmark compares GPU against CPU prices on every run.

Parameters: S0 = 100, K = 100, r = 5%, σ = 20%, T = 1 year, 252 steps per path, seed 42. Black-Scholes European call = 10.4506.
