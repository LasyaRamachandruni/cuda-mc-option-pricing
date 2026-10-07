**Hardware, 2-vCPU sandbox (no GPU):** CPU Intel(R) Xeon(R) Processor @ 2.10GHz, 2 threads visible to OpenMP; no GPU.  
**Workload:** 252 time steps per path; times are the fastest of several runs. Prices are mean ± 1 standard error; 95% CI = ± 1.96 SE.

| Paths | CPU scalar 1T | CPU SIMD 1T | CPU SIMD+OpenMP (2T) | SIMD+OpenMP vs scalar 1T | European call (BS 10.4506) | EU distance from BS (SEs) | Asian call |
|---|---|---|---|---|---|---|---|
| 65,536 | 269.9 ms | 36.9 ms | 19.2 ms | 14.1x | 10.3635 ± 0.0572 | 1.52 | 5.7579 ± 0.0312 |
| 262,144 | 1.06 s | 159.6 ms | 78.1 ms | 13.5x | 10.4462 ± 0.0287 | 0.15 | 5.7923 ± 0.0156 |
| 1,048,576 | 4.49 s | 651.8 ms | 365.2 ms | 12.3x | 10.4550 ± 0.0144 | 0.31 | 5.7816 ± 0.0078 |
| 4,194,304 | 18.27 s | 2.58 s | 1.35 s | 13.5x | 10.4512 ± 0.0072 | 0.09 | 5.7801 ± 0.0039 |
