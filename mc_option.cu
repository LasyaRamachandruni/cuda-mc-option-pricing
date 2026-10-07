// Monte Carlo option pricing: CPU (scalar and SIMD+OpenMP) vs GPU (CUDA).
//
// Prices a European call (checked against the Black-Scholes closed form) and an
// arithmetic-average Asian call (path-dependent, no closed form) by simulating
// geometric Brownian motion paths with `steps` time steps each.
//
// Three implementations, all simulating the same paths (counter-based RNG, see
// mc_core.h), so their prices should agree to float rounding:
//   scalar 1T : one CPU thread, one path at a time (the original reference)
//   SIMD 1T   : one CPU thread, SIMD across paths (cpu_simd.cpp)
//   SIMD+OMP  : all CPU cores via OpenMP, SIMD across paths
//   GPU       : CUDA kernel, one path per thread in a grid-stride loop
// GPU speedups are reported against both scalar 1T and SIMD+OMP.
//
// Build: see Makefile (`make mc` for GPU, `make mc_cpu` for CPU only).
// Run:   ./mc [--steps 252] [--min-paths 65536] [--max-paths 4194304]
//             [--max-scalar N] [--reps 1] [--seed 42] [--label TEXT]
//             [--csv results/results.csv]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mc_core.h"

#if defined(__CUDACC__) && !defined(CPU_ONLY)
#include <cuda_runtime.h>
#define HAS_GPU 1
#else
#define HAS_GPU 0
#endif

// ------------------------------------------------------------------- GPU
#if HAS_GPU
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

constexpr int kThreads = 256;

// Each thread simulates paths in a grid-stride loop and keeps running sums;
// each block then reduces its threads' sums in shared memory and writes one
// partial result. The host adds up the per-block partials (a few hundred).
__global__ void mc_kernel(Params p, uint64_t n_paths, Sums* partial) {
    __shared__ double sh[4][kThreads];
    double eu = 0, eu2 = 0, as = 0, as2 = 0;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_paths; i += stride) {
        float e, a;
        simulate_path(p, i, e, a);
        eu += e; eu2 += (double)e * e;
        as += a; as2 += (double)a * a;
    }
    int t = threadIdx.x;
    sh[0][t] = eu; sh[1][t] = eu2; sh[2][t] = as; sh[3][t] = as2;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off)
            for (int k = 0; k < 4; ++k) sh[k][t] += sh[k][t + off];
        __syncthreads();
    }
    if (t == 0) {
        Sums s;
        s.eu = sh[0][0]; s.eu2 = sh[1][0]; s.as = sh[2][0]; s.as2 = sh[3][0];
        partial[blockIdx.x] = s;
    }
}

struct GpuResult { Sums sums; float kernel_ms; float total_ms; };

GpuResult gpu_price(const Params& p, uint64_t n_paths, int n_blocks) {
    Sums* d_partial;
    std::vector<Sums> h_partial(n_blocks);
    CUDA_CHECK(cudaMalloc(&d_partial, n_blocks * sizeof(Sums)));
    cudaEvent_t start, mid, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&mid));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    mc_kernel<<<n_blocks, kThreads>>>(p, n_paths, d_partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(mid));
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, n_blocks * sizeof(Sums),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    GpuResult res{};
    CUDA_CHECK(cudaEventElapsedTime(&res.kernel_ms, start, mid));
    CUDA_CHECK(cudaEventElapsedTime(&res.total_ms, start, stop));
    for (const Sums& s : h_partial) {
        res.sums.eu += s.eu; res.sums.eu2 += s.eu2;
        res.sums.as += s.as; res.sums.as2 += s.as2;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(mid));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_partial));
    return res;
}
#endif

// ------------------------------------------------------------------ helpers
// Runs f() at least `min_reps` times and returns the fastest time in ms. Short
// runs are repeated more (up to 5 times, or until ~0.5 s in total) so timer
// noise doesn't dominate. On shared/noisy machines, raise --reps.
int g_min_reps = 1;

template <class F>
double best_ms(F&& f) {
    double best = INFINITY, total = 0;
    for (int r = 0; r < g_min_reps || (r < 5 && total < 500.0); ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, ms);
        total += ms;
    }
    return best;
}

std::string cpu_model() {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return "unknown";
    char line[512];
    std::string model = "unknown";
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "model name", 10) == 0) {
            const char* c = strchr(line, ':');
            if (c) {
                model = c + 1;
                while (!model.empty() && model.front() == ' ') model.erase(0, 1);
                while (!model.empty() && (model.back() == '\n' || model.back() == ' '))
                    model.pop_back();
            }
            break;
        }
    }
    fclose(f);
    return model;
}

// Keep free-text CSV fields from breaking the columns.
std::string csv_field(std::string s) {
    for (char& c : s)
        if (c == ',' || c == '"' || c == '\n') c = ' ';
    return s;
}

// --------------------------------------------------------------------- main
int main(int argc, char** argv) {
    Params p;
    uint64_t min_paths = 1ULL << 16;   // 65,536
    uint64_t max_paths = 1ULL << 22;   // 4,194,304
    uint64_t max_scalar = 1ULL << 22;  // skip the slow single-thread runs above this
    std::string csv_path = "results/results.csv";
    std::string label = "";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); return 1; }
        const char* v = argv[++i];
        if (a == "--steps") p.steps = std::atoi(v);
        else if (a == "--min-paths") min_paths = std::strtoull(v, nullptr, 10);
        else if (a == "--max-paths") max_paths = std::strtoull(v, nullptr, 10);
        else if (a == "--max-scalar") max_scalar = std::strtoull(v, nullptr, 10);
        else if (a == "--seed") p.seed = std::strtoull(v, nullptr, 10);
        else if (a == "--csv") csv_path = v;
        else if (a == "--label") label = v;
        else if (a == "--reps") g_min_reps = std::max(1, std::atoi(v));
        else { fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
    }
    if (p.steps < 1 || min_paths < 2 || max_paths < min_paths) {
        fprintf(stderr, "bad arguments\n");
        return 1;
    }

    int cpu_threads = 1;
#ifdef _OPENMP
    cpu_threads = omp_get_max_threads();
#endif
    const double bs = black_scholes_call(p);
    const std::string cpu = cpu_model();
    std::string gpu = "none";

    printf("Monte Carlo option pricing, GBM\n");
    printf("S0=%.0f K=%.0f r=%.2f sigma=%.2f T=%.1f, %d steps/path, seed %llu\n", p.S0, p.K,
           p.r, p.sigma, p.T, p.steps, (unsigned long long)p.seed);
    printf("Black-Scholes European call = %.4f\n", bs);
    printf("CPU: %s, %d OpenMP threads\n", cpu.c_str(), cpu_threads);

    int n_blocks = 0;
    (void)n_blocks;
#if HAS_GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, mc_kernel,
                                                             kThreads, 0));
    n_blocks = prop.multiProcessorCount * blocks_per_sm * 4;
    gpu = prop.name;
    printf("GPU: %s (sm_%d%d, %d SMs), %d blocks x %d threads\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, n_blocks, kThreads);
    gpu_price(p, 1 << 16, n_blocks);  // warm-up: context creation, module load
#else
    printf("GPU: not built (CPU-only binary)\n");
#endif
    cpu_price_simd(p, 1 << 14, 0);  // warm-up: start the OpenMP thread pool
    printf("\n");

    FILE* csv = fopen(csv_path.c_str(), "w");
    if (!csv)
        fprintf(stderr, "warning: can't write %s (does the directory exist?)\n",
                csv_path.c_str());
    else
        fprintf(csv,
                "label,cpu_model,cpu_threads,gpu_name,paths,steps,"
                "scalar_1t_ms,simd_1t_ms,simd_omp_ms,gpu_kernel_ms,gpu_total_ms,"
                "simd_omp_vs_scalar,gpu_vs_scalar_1t,gpu_vs_simd_omp,"
                "bs,eu_scalar,eu_simd,eu_gpu,eu_se,eu_z_bs,"
                "asian_scalar,asian_simd,asian_gpu,asian_se,max_gap_se\n");

    printf("%9s %10s %10s %10s %9s | %9s %9s | %18s %6s | %17s\n", "paths", "scalar ms",
           "SIMD 1T ms", "SIMD+OMP", "GPU ms", "GPU/scal", "GPU/OMP", "EU call +- SE",
           "z(BS)", "Asian +- SE");

    bool ok = true;
    for (uint64_t n = min_paths; n <= max_paths; n <<= 2) {
        const bool run_scalar = n <= max_scalar;
        Sums ss{}, s1{}, so{};
        double t_scalar = NAN, t_simd1 = NAN;
        if (run_scalar) {
            t_scalar = best_ms([&] { ss = cpu_price_scalar(p, n); });
            t_simd1 = best_ms([&] { s1 = cpu_price_simd(p, n, 1); });
        }
        double t_omp = best_ms([&] { so = cpu_price_simd(p, n, 0); });

        double eu, eu_se, as, as_se;
        mean_se(so.eu, so.eu2, n, eu, eu_se);
        mean_se(so.as, so.as2, n, as, as_se);
        double eu_s = run_scalar ? ss.eu / n : NAN, as_s = run_scalar ? ss.as / n : NAN;

        double eu_g = NAN, as_g = NAN, g_kernel = NAN, g_total = NAN;
#if HAS_GPU
        g_total = INFINITY;
        for (int r = 0; r < 5; ++r) {  // CUDA event times, fastest of 5
            GpuResult g = gpu_price(p, n, n_blocks);
            if (g.total_ms < g_total) { g_total = g.total_ms; g_kernel = g.kernel_ms; }
            eu_g = g.sums.eu / n;
            as_g = g.sums.as / n;
        }
#endif
        // Every implementation simulates the same paths, so the only differences
        // are float rounding. Largest gap to the SIMD+OMP price, in SE units:
        double gap = 0;
        auto check = [&](double x, double ref, double se) {
            if (!std::isnan(x)) gap = std::max(gap, std::fabs(x - ref) / se);
        };
        check(eu_s, eu, eu_se); check(eu_g, eu, eu_se);
        check(as_s, as, as_se); check(as_g, as, as_se);
        if (run_scalar) { check(s1.eu / n, eu, eu_se); check(s1.as / n, as, as_se); }
        double z_bs = std::fabs(eu - bs) / eu_se;
        if (z_bs > 4.0 || gap > 0.1) ok = false;

        double sp_omp = t_scalar / t_omp, sp_g1 = t_scalar / g_total, sp_go = t_omp / g_total;
        printf("%9llu %10.1f %10.1f %10.1f %9.2f | %8.1fx %8.1fx | %9.4f +- %.4f %6.2f | "
               "%8.4f +- %.4f\n",
               (unsigned long long)n, t_scalar, t_simd1, t_omp, g_total, sp_g1, sp_go, eu,
               eu_se, z_bs, as, as_se);
        if (csv)
            fprintf(csv,
                    "%s,%s,%d,%s,%llu,%d,%.3f,%.3f,%.3f,%.4f,%.4f,%.2f,%.2f,%.2f,"
                    "%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                    csv_field(label).c_str(), csv_field(cpu).c_str(), cpu_threads,
                    csv_field(gpu).c_str(), (unsigned long long)n, p.steps, t_scalar, t_simd1,
                    t_omp, g_kernel, g_total, sp_omp, sp_g1, sp_go, bs, eu_s, eu, eu_g, eu_se,
                    z_bs, as_s, as, as_g, as_se, gap);
    }
    if (csv) {
        fclose(csv);
        printf("\nWrote %s\n", csv_path.c_str());
    }
    printf("Prices are from SIMD+OMP (%d threads). z(BS) = |EU - Black-Scholes| / SE, under ~3\n"
           "is expected. Times are the fastest of several runs; GPU ms = kernel + copy back.\n",
           cpu_threads);
    printf("Scalar / SIMD / GPU simulate the same paths and must agree to < 0.1 SE: %s\n",
           ok ? "OK" : "MISMATCH");
    return ok ? 0 : 2;
}
