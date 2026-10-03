# CUDA Monte Carlo Option Pricing

Prices options by simulating stock-price paths with geometric Brownian motion, on the CPU (C++, single-threaded and OpenMP) and on the GPU (CUDA), and benchmarks the speedup.

- **European call:** checked against the Black-Scholes closed form (the result should land within ~3 standard errors).
- **Arithmetic-average Asian call:** path-dependent, with no closed form, which is why every path simulates all 252 daily steps.

## How it works

- **Same code on CPU and GPU.** `simulate_path()` is a `__host__ __device__` function, so both sides run identical math.
- **Counter-based RNG.** Each random draw is a hash of `(seed, path, step)` (splitmix64), turned into normals with Box-Muller. There's no RNG state to store or seed per thread, and path *i* is the same on CPU and GPU. GPU prices therefore match CPU prices to float rounding, which the program checks.
- **GPU kernel.** A grid-stride loop has each thread simulate many paths and keep running sums of the payoff and payoff². A shared-memory tree reduction then gives one partial result per block, and the host adds up those few hundred partials. The grid is sized from the occupancy API (`SMs × active blocks per SM × 4`).
- **Timing.** CUDA events time the kernel plus the copy back. The GPU is warmed up first, so context creation isn't counted. CPU runs are timed with `std::chrono`.
- **Accuracy.** Payoffs are summed in double precision and paths are simulated in float. The program reports the standard error and the distance from Black-Scholes in standard errors.

## Run it

**No GPU? Use Google Colab (free):** open `run_benchmark_colab.ipynb` in Colab, choose Runtime → Change runtime type → T4 GPU, then Run all. It builds the code, runs the benchmark, draws `speedup.png` and prints a results table for this README.

**With an NVIDIA GPU and the CUDA toolkit:**

```bash
make            # nvcc -O3 -arch=native -Xcompiler -fopenmp -o mc mc_option.cu
./mc            # 65K to 4M paths, 252 steps each; writes results.csv
```

**CPU only (correctness check):** `make mc_cpu && ./mc_cpu --max-paths 1048576`

Flags: `--steps N`, `--max-paths N`, `--max-single-thread N` (skip slow single-thread runs above N paths), `--seed N`, `--csv FILE`.

## Results

Measured on Google Colab: **NVIDIA Tesla T4** GPU vs the runtime's **2 CPU threads**, 252 steps per path.

| Paths | CPU 1 thread | CPU OpenMP | GPU | Speedup vs 1T | Speedup vs OpenMP | EU call (BS 10.4506) |
|---|---|---|---|---|---|---|
| 65,536 | 355 ms | 255 ms | 0.6 ms | 644x | 463x | 10.3635 ± 0.0572 |
| 262,144 | 1429 ms | 1041 ms | 2.2 ms | 644x | 469x | 10.4462 ± 0.0287 |
| 1,048,576 | 5632 ms | 5242 ms | 8.5 ms | 659x | 613x | 10.4550 ± 0.0144 |
| 4,194,304 | 23692 ms | 18151 ms | 33.8 ms | 701x | 537x | 10.4512 ± 0.0072 |

At 4.2M paths (1.06B simulated steps), the GPU takes 33.8 ms against 23.7 s on one CPU thread. The European price lands 0.1 standard errors from Black-Scholes.

Caveat: the CPU baseline is scalar C++ on Colab's 2 vCPUs (typically two hyperthreads of one physical core), which is why OpenMP only gains ~1.3×. A many-core, SIMD-vectorized CPU baseline would shrink the speedup considerably.

![speedup](speedup.png)

Parameters: S0 = 100, K = 100, r = 5%, σ = 20%, T = 1 year, 252 steps per path. Black-Scholes European call = 10.4506.