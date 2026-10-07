// CPU correctness test (no GPU needed). Run with `make test`.
//
// 1. European call from the scalar and SIMD+OpenMP pricers is within 3 standard
//    errors of the Black-Scholes closed form.
// 2. Scalar and SIMD simulate the same paths, so they must agree to < 0.1 SE.
//    They aren't bit-identical (float rounding over 252 steps, and the SIMD
//    file uses -ffast-math vector exp/log), but a real bug in the vectorized
//    loop would show up as a gap of many SEs.
// 3. Asian arithmetic call vs an independent reference: a double-precision
//    simulation with a different RNG (std::mt19937_64 streams) that uses the
//    discretely monitored geometric Asian call (closed form) as a control
//    variate. That reference has a much smaller SE, and its own geometric
//    estimate is checked against the closed form too.
// 4. Sanity bounds: geometric Asian <= arithmetic Asian <= European.
//
// The GPU runs the same simulate_path() as the scalar CPU code (mc_core.h),
// so checks 1-3 cover the math the kernel uses; the benchmark itself checks
// the GPU against the CPU on every run.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mc_core.h"

static int g_failures = 0;

static void check(bool ok, const char* what, double z, double limit) {
    printf("  [%s] %-58s %7.3f (limit %.3g)\n", ok ? "PASS" : "FAIL", what, z, limit);
    if (!ok) ++g_failures;
}

// Discretely monitored geometric-average Asian call, averaging S(t_1..t_n)
// with t_i = i*T/n (same convention as simulate_path: S0 not included).
// log G is normal with mean m and variance v below.
static double geometric_asian_call(const Params& p) {
    const double n = p.steps, dt = (double)p.T / p.steps;
    const double sig = p.sigma, nu = p.r - 0.5 * sig * sig;
    const double m = std::log((double)p.S0) + nu * dt * (n + 1) / 2;
    const double v = sig * sig * dt * (n + 1) * (2 * n + 1) / (6 * n);
    const double sd = std::sqrt(v);
    const double d2 = (m - std::log((double)p.K)) / sd, d1 = d2 + sd;
    return std::exp(-p.r * p.T) * (std::exp(m + 0.5 * v) * norm_cdf(d1) - p.K * norm_cdf(d2));
}

struct Reference {
    double asian, asian_se;      // control-variate arithmetic Asian estimate
    double geo_mc, geo_se;       // plain MC geometric Asian (to check vs closed form)
    double corr;                 // correlation of arithmetic and geometric payoffs
};

// Independent double-precision simulation. Paths are split into fixed chunks,
// each with its own mt19937_64 stream, so results don't depend on thread count.
static Reference asian_reference(const Params& p, long long n_paths, uint64_t seed) {
    const int chunks = 64;
    const double dt = (double)p.T / p.steps;
    const double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
    const double vol = p.sigma * std::sqrt(dt);
    const double disc = std::exp(-p.r * p.T);
    const double geo_exact = geometric_asian_call(p);

    // Sums of A, G, A^2, G^2, A*G over all paths.
    double sa = 0, sg = 0, saa = 0, sgg = 0, sag = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : sa, sg, saa, sgg, sag)
    for (int c = 0; c < chunks; ++c) {
        std::seed_seq ss{seed, (uint64_t)c};
        std::mt19937_64 rng(ss);
        std::normal_distribution<double> N01;
        const long long lo = n_paths * c / chunks, hi = n_paths * (c + 1) / chunks;
        for (long long i = lo; i < hi; ++i) {
            double x = std::log((double)p.S0), sum = 0, sum_log = 0;
            for (int s = 0; s < p.steps; ++s) {
                x += drift + vol * N01(rng);
                sum += std::exp(x);
                sum_log += x;
            }
            double a = disc * std::fmax(sum / p.steps - p.K, 0.0);
            double g = disc * std::fmax(std::exp(sum_log / p.steps) - p.K, 0.0);
            sa += a; sg += g; saa += a * a; sgg += g * g; sag += a * g;
        }
    }
    const double n = (double)n_paths;
    const double ma = sa / n, mg = sg / n;
    const double va = (saa / n - ma * ma) * n / (n - 1);
    const double vg = (sgg / n - mg * mg) * n / (n - 1);
    const double cag = (sag / n - ma * mg) * n / (n - 1);
    const double b = cag / vg;  // optimal control-variate coefficient
    Reference r;
    r.asian = ma - b * (mg - geo_exact);
    r.asian_se = std::sqrt(std::fmax(va - b * cag, 0.0) / n);  // var(A - bG) = va - cag^2/vg
    r.geo_mc = mg;
    r.geo_se = std::sqrt(vg / n);
    r.corr = cag / std::sqrt(va * vg);
    return r;
}

int main() {
    Params p;  // S0=100 K=100 r=5% sigma=20% T=1, 252 steps, seed 42
    const double bs = black_scholes_call(p);
    const double geo = geometric_asian_call(p);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("CPU correctness test: S0=%.0f K=%.0f r=%.2f sigma=%.2f T=%.1f, %d steps, %d threads\n",
           p.S0, p.K, p.r, p.sigma, p.T, p.steps, threads);
    printf("Black-Scholes European call      = %.6f\n", bs);
    printf("Geometric Asian call (closed form) = %.6f\n\n", geo);

    // --- 1 + 2: European vs Black-Scholes, scalar vs SIMD on the same paths
    const uint64_t n_scalar = 1 << 18, n_simd = 1 << 20;
    Sums ss = cpu_price_scalar(p, n_scalar);
    Sums sv_small = cpu_price_simd(p, n_scalar, 0);
    Sums sv = cpu_price_simd(p, n_simd, 0);

    double eu_s, se_s, as_s, ase_s, eu_v, se_v, as_v, ase_v, eu_vs, se_vs, as_vs, ase_vs;
    mean_se(ss.eu, ss.eu2, n_scalar, eu_s, se_s);
    mean_se(ss.as, ss.as2, n_scalar, as_s, ase_s);
    mean_se(sv_small.eu, sv_small.eu2, n_scalar, eu_vs, se_vs);
    mean_se(sv_small.as, sv_small.as2, n_scalar, as_vs, ase_vs);
    mean_se(sv.eu, sv.eu2, n_simd, eu_v, se_v);
    mean_se(sv.as, sv.as2, n_simd, as_v, ase_v);

    printf("European call\n");
    printf("  scalar    %8llu paths: %.4f +- %.4f\n", (unsigned long long)n_scalar, eu_s, se_s);
    printf("  SIMD+OMP  %8llu paths: %.4f +- %.4f\n", (unsigned long long)n_simd, eu_v, se_v);
    check(std::fabs(eu_s - bs) / se_s < 3, "scalar European vs Black-Scholes (|diff|/SE)",
          std::fabs(eu_s - bs) / se_s, 3);
    check(std::fabs(eu_v - bs) / se_v < 3, "SIMD+OMP European vs Black-Scholes (|diff|/SE)",
          std::fabs(eu_v - bs) / se_v, 3);
    check(std::fabs(eu_s - eu_vs) / se_s < 0.1, "scalar vs SIMD European, same paths (|diff|/SE)",
          std::fabs(eu_s - eu_vs) / se_s, 0.1);
    check(std::fabs(as_s - as_vs) / ase_s < 0.1, "scalar vs SIMD Asian, same paths (|diff|/SE)",
          std::fabs(as_s - as_vs) / ase_s, 0.1);

    // --- 3: Asian vs independent control-variate reference
    const long long n_ref = 1 << 18;
    Reference ref = asian_reference(p, n_ref, 20261007);
    printf("\nArithmetic Asian call\n");
    printf("  scalar    %8llu paths: %.4f +- %.4f\n", (unsigned long long)n_scalar, as_s, ase_s);
    printf("  SIMD+OMP  %8llu paths: %.4f +- %.4f\n", (unsigned long long)n_simd, as_v, ase_v);
    printf("  reference %8lld paths: %.4f +- %.4f  (mt19937_64, double, geometric control "
           "variate, corr %.4f)\n", n_ref, ref.asian, ref.asian_se, ref.corr);
    printf("  reference geometric MC: %.4f +- %.4f (closed form %.4f)\n", ref.geo_mc, ref.geo_se,
           geo);
    check(std::fabs(ref.geo_mc - geo) / ref.geo_se < 3,
          "reference geometric MC vs closed form (|diff|/SE)",
          std::fabs(ref.geo_mc - geo) / ref.geo_se, 3);
    const double zs = std::fabs(as_s - ref.asian) / std::hypot(ase_s, ref.asian_se);
    const double zv = std::fabs(as_v - ref.asian) / std::hypot(ase_v, ref.asian_se);
    check(zs < 3, "scalar Asian vs reference (|diff|/combined SE)", zs, 3);
    check(zv < 3, "SIMD+OMP Asian vs reference (|diff|/combined SE)", zv, 3);

    // --- 4: ordering. Arithmetic >= geometric pathwise (AM-GM); averaging
    // lowers the effective volatility, so the Asian is cheaper than the European.
    check(geo < ref.asian && ref.asian < bs, "geometric < arithmetic Asian < European (bool)",
          geo < ref.asian && ref.asian < bs, 1);

    printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures,
           g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
