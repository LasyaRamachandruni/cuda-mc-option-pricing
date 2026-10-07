// Optimized CPU baseline: OpenMP over all cores + SIMD across paths.
//
// The scalar version simulates one path at a time, and the time-step loop is a
// recurrence (logS depends on the previous step), so it can't vectorize. Here
// each thread takes a block of kLanes paths and advances them together one time
// step at a time; the inner loop over lanes has no dependencies and maps onto
// AVX2/AVX-512 registers, with expf/logf/sinf/cosf going to glibc's vector math
// library (libmvec).
//
// Build with: -O3 -march=native -ffast-math -fopenmp
// -ffast-math is needed for GCC to call the libmvec vector functions. It's kept
// to this file so the scalar reference is compiled with normal IEEE semantics.
//
// Paths use the same counter-based RNG as the scalar and GPU code, so path i is
// the same everywhere and prices agree to float rounding.

#include "mc_core.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr int kLanes = 32;  // paths per block; a few SIMD registers' worth

void simulate_block(const Params& p, uint64_t first_path, float* eu, float* as) {
    const float dt = p.T / p.steps;
    const float drift = (p.r - 0.5f * p.sigma * p.sigma) * dt;
    const float vol = p.sigma * sqrtf(dt);
    const float disc = expf(-p.r * p.T);
    const float logS0 = logf(p.S0);
    const float inv_steps = 1.0f / p.steps;

    alignas(64) uint64_t key[kLanes];
    alignas(64) float logS[kLanes];
    alignas(64) float sumS[kLanes];

#pragma omp simd aligned(key, logS, sumS : 64)
    for (int l = 0; l < kLanes; ++l) {
        key[l] = path_key(p.seed, first_path + l);
        logS[l] = logS0;
        sumS[l] = 0.0f;
    }

    const int pairs = p.steps / 2;
    for (int k = 0; k < pairs; ++k) {
#pragma omp simd aligned(key, logS, sumS : 64)
        for (int l = 0; l < kLanes; ++l) {
            float z0, z1;
            normal_pair(key[l], (uint32_t)k, z0, z1);
            float x = logS[l] + (drift + vol * z0);
            float s = sumS[l] + expf(x);
            x += drift + vol * z1;
            s += expf(x);
            logS[l] = x;
            sumS[l] = s;
        }
    }
    if (p.steps & 1) {  // odd step count: one last step using z0 of the next pair
#pragma omp simd aligned(key, logS, sumS : 64)
        for (int l = 0; l < kLanes; ++l) {
            float z0, z1;
            normal_pair(key[l], (uint32_t)pairs, z0, z1);
            logS[l] += drift + vol * z0;
            sumS[l] += expf(logS[l]);
        }
    }

#pragma omp simd aligned(logS, sumS : 64)
    for (int l = 0; l < kLanes; ++l) {
        eu[l] = disc * fmaxf(expf(logS[l]) - p.K, 0.0f);
        as[l] = disc * fmaxf(sumS[l] * inv_steps - p.K, 0.0f);
    }
}
}  // namespace

Sums cpu_price_simd(const Params& p, uint64_t n_paths, int threads) {
#ifdef _OPENMP
    if (threads <= 0) threads = omp_get_max_threads();
#else
    threads = 1;
#endif
    const long long n_blocks = (long long)((n_paths + kLanes - 1) / kLanes);
    double seu = 0, seu2 = 0, sas = 0, sas2 = 0;

#pragma omp parallel for num_threads(threads) schedule(static) \
    reduction(+ : seu, seu2, sas, sas2)
    for (long long b = 0; b < n_blocks; ++b) {
        alignas(64) float eu[kLanes];
        alignas(64) float as[kLanes];
        const uint64_t first = (uint64_t)b * kLanes;
        simulate_block(p, first, eu, as);
        // The last block may run past n_paths; drop those lanes.
        const int valid = (int)((n_paths - first < (uint64_t)kLanes) ? n_paths - first : kLanes);
        for (int l = 0; l < valid; ++l) {
            seu += eu[l]; seu2 += (double)eu[l] * eu[l];
            sas += as[l]; sas2 += (double)as[l] * as[l];
        }
    }
    Sums s;
    s.eu = seu; s.eu2 = seu2; s.as = sas; s.as2 = sas2;
    return s;
}
