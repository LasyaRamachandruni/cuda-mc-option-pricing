**Hardware, colab:** CPU Intel(R) Xeon(R) CPU @ 2.00GHz, 2 threads visible to OpenMP; GPU Tesla T4.  
**Workload:** 252 time steps per path; times are the fastest of several runs. Prices are mean ± 1 standard error; 95% CI = ± 1.96 SE.

| Paths | CPU scalar 1T | CPU SIMD 1T | CPU SIMD+OpenMP (2T) | GPU (CUDA) | GPU vs scalar 1T | GPU vs SIMD+OpenMP | European call (BS 10.4506) | EU distance from BS (SEs) | Asian call |
|---|---|---|---|---|---|---|---|---|---|
| 65,536 | 331.8 ms | 60.0 ms | 42.3 ms | 0.64 ms | 518.4x | 66.1x | 10.3635 ± 0.0572 | 1.52 | 5.7579 ± 0.0312 |
| 262,144 | 1.36 s | 241.0 ms | 167.4 ms | 2.18 ms | 624.1x | 76.8x | 10.4462 ± 0.0287 | 0.15 | 5.7923 ± 0.0156 |
| 1,048,576 | 5.45 s | 1.01 s | 668.0 ms | 8.52 ms | 640.0x | 78.4x | 10.4550 ± 0.0144 | 0.31 | 5.7816 ± 0.0078 |
| 4,194,304 | 22.96 s | 3.96 s | 2.75 s | 33.8 ms | 679.5x | 81.4x | 10.4512 ± 0.0072 | 0.09 | 5.7801 ± 0.0039 |
