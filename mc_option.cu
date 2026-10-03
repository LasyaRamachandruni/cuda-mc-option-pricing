// Monte Carlo option pricing: CPU (C++, single-thread + OpenMP) vs GPU (CUDA).
//
// Prices a European call (checked against the Black-Scholes closed form) and an
// arithmetic-average Asian call (path-dependent, no closed form) by simulating
// geometric Brownian motion paths with `steps` time steps each.
//
// CPU and GPU run the *same* simulation code and the same counter-based random
// number generator, so path i is identical on both. Any price difference is
// float rounding only, which makes the GPU result easy to verify.
//
// Build (GPU):      nvcc -O3 -arch=native -Xcompiler -fopenmp -o mc mc_option.cu
// Build (CPU only): g++ -O3 -fopenmp -x c++ -DCPU_ONLY -o mc_cpu mc_option.cu
// Run:              ./mc [--steps 252] [--max-paths 4194304] [--csv results.csv]

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

#if defined(__CUDACC__) && !defined(CPU_ONLY)
#include <cuda_runtime.h>
#define HD __host__ __device__
#define HAS_GPU 1
#else
#define HD
#define HAS_GPU 0
#endif

// ---------------------------------------------------------------- parameters
struct Params {
    float S0 = 100.0f;     // spot
    float K = 100.0f;      // strike
    float r = 0.05f;       // risk-free rate
    float sigma = 0.20f;   // volatility
    float T = 1.0f;        // maturity (years)
    int steps = 252;       // time steps per path (daily for 1 year)
    uint64_t seed = 42;
};

// Per-path payoffs summed over many paths (double to keep the sums accurate).
struct Sums {
    double eu = 0, eu2 = 0;  // European call payoff and payoff^2
    double as = 0, as2 = 0;  // Asian call payoff and payoff^2
};

// ---------------------------------------------- counter-based random numbers
// splitmix64 finalizer: a fast, well-mixed 64-bit hash. Hashing (seed, path,
// step) gives each draw its own independent value with no RNG state, so the
// CPU and GPU produce exactly the same stream for every path.
HD inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Two independent standard normals via Box-Muller from one 64-bit hash.
HD inline void normal_pair(uint64_t seed, uint64_t path, uint32_t k, float& z0, float& z1) {
    uint64_t h = mix64(mix64(seed ^ (path * 0xD1B54A32D192ED03ULL)) + k);
    // Uniforms in (0, 1]; never 0, so log() is safe.
    float u1 = ((uint32_t)(h >> 32) + 1.0f) * (1.0f / 4294967296.0f);
    float u2 = ((uint32_t)h) * (1.0f / 4294967296.0f);
    float rad = sqrtf(-2.0f * logf(u1));
    float ang = 6.283185307f * u2;
    z0 = rad * cosf(ang);
    z1 = rad * sinf(ang);
}

// ------------------------------------------------------ one simulated path
// Simulates log S under GBM: log S += (r - sigma^2/2) dt + sigma sqrt(dt) Z.
// Returns the discounted European and Asian call payoffs for this path.
HD inline void simulate_path(const Params& p, uint64_t path, float& eu, float& as) {
    const float dt = p.T / p.steps;
    const float drift = (p.r - 0.5f * p.sigma * p.sigma) * dt;
    const float vol = p.sigma * sqrtf(dt);
    const float disc = expf(-p.r * p.T);

    float logS = logf(p.S0);
    float sumS = 0.0f;
    for (int s = 0; s < p.steps; s += 2) {
        float z0, z1;
        normal_pair(p.seed, path, (uint32_t)(s >> 1), z0, z1);
        logS += drift + vol * z0;
        sumS += expf(logS);
        if (s + 1 < p.steps) {
            logS += drift + vol * z1;
            sumS += expf(logS);
        }
    }
    float ST = expf(logS);
    float avg = sumS / p.steps;
    eu = disc * fmaxf(ST - p.K, 0.0f);
    as = disc * fmaxf(avg - p.K, 0.0f);
}

// ------------------------------------------------------------------- CPU
Sums cpu_price(const Params& p, uint64_t n_paths, bool parallel) {
    double eu = 0, eu2 = 0, as = 0, as2 = 0;
#pragma omp parallel for reduction(+ : eu, eu2, as, as2) schedule(static) if (parallel)
    for (long long i = 0; i < (long long)n_paths; ++i) {
        float e, a;
        simulate_path(p, (uint64_t)i, e, a);
        eu += e; eu2 += (double)e * e;
        as += a; as2 += (double)a * a;
    }
    return {eu, eu2, as, as2};
}

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
    if (t == 0) partial[blockIdx.x] = {sh[0][0], sh[1][0], sh[2][0], sh[3][0]};
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
    cudaEventDestroy(start); cudaEventDestroy(mid); cudaEventDestroy(stop);
    cudaFree(d_partial);
    return res;
}
#endif

// ------------------------------------------------------------------ helpers
double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double black_scholes_call(const Params& p) {
    double S = p.S0, K = p.K, r = p.r, v = p.sigma, T = p.T;
    double d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
    double d2 = d1 - v * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// Mean and standard error of the mean from a sum and sum of squares.
void mean_se(double sum, double sum2, uint64_t n, double& mean, double& se) {
    mean = sum / n;
    double var = (sum2 / n - mean * mean) * n / (n - 1.0);
    se = std::sqrt(var > 0 ? var / n : 0.0);
}

template <class F>
double time_ms(F&& f) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// --------------------------------------------------------------------- main
int main(int argc, char** argv) {
    Params p;
    uint64_t max_paths = 1ULL << 22;  // 4,194,304
    uint64_t max_single_thread = 1ULL << 22;
    std::string csv_path = "results.csv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : (char*)"0"; };
        if (a == "--steps") p.steps = std::atoi(next());
        else if (a == "--max-paths") max_paths = std::strtoull(next(), nullptr, 10);
        else if (a == "--max-single-thread") max_single_thread = std::strtoull(next(), nullptr, 10);
        else if (a == "--seed") p.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--csv") csv_path = next();
        else { fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
    }

    int cpu_threads = 1;
#ifdef _OPENMP
    cpu_threads = omp_get_max_threads();
#endif
    const double bs = black_scholes_call(p);

    printf("Monte Carlo option pricing, GBM\n");
    printf("S0=%.0f K=%.0f r=%.2f sigma=%.2f T=%.1f, %d steps/path\n", p.S0, p.K, p.r,
           p.sigma, p.T, p.steps);
    printf("Black-Scholes European call = %.4f\n", bs);
    printf("CPU threads (OpenMP): %d\n", cpu_threads);

    int n_blocks = 0;
    (void)n_blocks;
#if HAS_GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, mc_kernel,
                                                             kThreads, 0));
    n_blocks = prop.multiProcessorCount * blocks_per_sm * 4;
    printf("GPU: %s (sm_%d%d, %d SMs), %d blocks x %d threads\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, n_blocks, kThreads);
    gpu_price(p, 1 << 16, n_blocks);  // warm-up: context creation, module load
#else
    printf("GPU: not built (CPU_ONLY)\n");
#endif
    printf("\n");

    FILE* csv = fopen(csv_path.c_str(), "w");
    if (csv)
        fprintf(csv, "paths,steps,cpu_1t_ms,cpu_omp_ms,cpu_threads,gpu_kernel_ms,gpu_total_ms,"
                     "speedup_vs_1t,speedup_vs_omp,eu_gpu,eu_se,eu_cpu,bs,asian_gpu,asian_cpu\n");

    printf("%10s %11s %11s %10s %9s %9s %16s %9s %9s\n", "paths", "CPU 1T ms", "CPU OMP ms",
           "GPU ms", "x vs 1T", "x vs OMP", "EU call +- SE", "|err|/SE", "Asian");

    for (uint64_t n = 1ULL << 16; n <= max_paths; n <<= 2) {
        Sums s1{}, so{};
        double t1 = NAN, to;
        if (n <= max_single_thread) t1 = time_ms([&] { s1 = cpu_price(p, n, false); });
        to = time_ms([&] { so = cpu_price(p, n, true); });

        double eu, se, as, as_se;
        Sums ref = so;
        double g_kernel = NAN, g_total = NAN;
#if HAS_GPU
        GpuResult g = gpu_price(p, n, n_blocks);
        g_kernel = g.kernel_ms;
        g_total = g.total_ms;
        ref = g.sums;
#endif
        mean_se(ref.eu, ref.eu2, n, eu, se);
        mean_se(ref.as, ref.as2, n, as, as_se);
        double eu_cpu = so.eu / n, as_cpu = so.as / n;
        double sp1 = t1 / g_total, spo = to / g_total;

        printf("%10llu %11.1f %11.1f %10.2f %9.1f %9.1f %9.4f+-%.4f %9.2f %9.4f\n",
               (unsigned long long)n, t1, to, g_total, sp1, spo, eu, se, std::fabs(eu - bs) / se,
               as);
        if (csv)
            fprintf(csv, "%llu,%d,%.3f,%.3f,%d,%.3f,%.3f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    (unsigned long long)n, p.steps, t1, to, cpu_threads, g_kernel, g_total, sp1,
                    spo, eu, se, eu_cpu, bs, as, as_cpu);

        // CPU and GPU simulate identical paths, so their prices should agree to
        // float rounding. Flag anything bigger as a bug.
        if (HAS_GPU && std::fabs(eu - eu_cpu) > 1e-3 * (1.0 + std::fabs(eu_cpu)))
            printf("  WARNING: GPU and CPU European prices differ (%.6f vs %.6f)\n", eu, eu_cpu);
    }
    if (csv) {
        fclose(csv);
        printf("\nWrote %s\n", csv_path.c_str());
    }
    printf("|err|/SE = distance from Black-Scholes in standard errors (under ~3 is expected).\n");
    printf("GPU ms includes the kernel and copying the per-block results back.\n");
    return 0;
}
