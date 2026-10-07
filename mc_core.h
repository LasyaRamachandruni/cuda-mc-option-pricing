// Shared pieces of the Monte Carlo pricer: parameters, the counter-based RNG,
// the per-path simulation, and the scalar CPU reference. Included by the CUDA
// benchmark (mc_option.cu), the SIMD/OpenMP CPU baseline (cpu_simd.cpp) and the
// CPU correctness test (test_cpu.cpp).
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD
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
// step) gives each draw its own independent value with no RNG state, so every
// implementation (scalar CPU, SIMD CPU, GPU) produces the same stream for path i.
HD inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Per-path key; constant over the path, so callers compute it once.
HD inline uint64_t path_key(uint64_t seed, uint64_t path) {
    return mix64(seed ^ (path * 0xD1B54A32D192ED03ULL));
}

// Two independent standard normals via Box-Muller from one 64-bit hash.
HD inline void normal_pair(uint64_t key, uint32_t k, float& z0, float& z1) {
    uint64_t h = mix64(key + k);
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
// Returns the discounted European and arithmetic-average Asian call payoffs.
HD inline void simulate_path(const Params& p, uint64_t path, float& eu, float& as) {
    const float dt = p.T / p.steps;
    const float drift = (p.r - 0.5f * p.sigma * p.sigma) * dt;
    const float vol = p.sigma * sqrtf(dt);
    const float disc = expf(-p.r * p.T);
    const uint64_t key = path_key(p.seed, path);

    float logS = logf(p.S0);
    float sumS = 0.0f;
    for (int s = 0; s < p.steps; s += 2) {
        float z0, z1;
        normal_pair(key, (uint32_t)(s >> 1), z0, z1);
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

// ------------------------------------------------------------- CPU versions
// Scalar reference: one thread, one path at a time. This is the original
// baseline; kept so results stay comparable with earlier runs.
inline Sums cpu_price_scalar(const Params& p, uint64_t n_paths) {
    Sums s;
    for (uint64_t i = 0; i < n_paths; ++i) {
        float e, a;
        simulate_path(p, i, e, a);
        s.eu += e; s.eu2 += (double)e * e;
        s.as += a; s.as2 += (double)a * a;
    }
    return s;
}

// Optimized CPU baseline (cpu_simd.cpp): OpenMP across `threads` threads
// (0 = all available), with each thread stepping a block of paths in lockstep
// so the inner loop vectorizes. Simulates the same paths as the scalar version.
Sums cpu_price_simd(const Params& p, uint64_t n_paths, int threads);

// ------------------------------------------------------------------ helpers
inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline double black_scholes_call(const Params& p) {
    double S = p.S0, K = p.K, r = p.r, v = p.sigma, T = p.T;
    double d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
    double d2 = d1 - v * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// Mean and standard error of the mean from a sum and sum of squares.
inline void mean_se(double sum, double sum2, uint64_t n, double& mean, double& se) {
    mean = sum / n;
    double var = (sum2 / n - mean * mean) * n / (n - 1.0);
    se = std::sqrt(var > 0 ? var / n : 0.0);
}
