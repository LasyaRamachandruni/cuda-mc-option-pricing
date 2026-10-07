CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
# The SIMD baseline needs -march=native for AVX2/AVX-512 and -ffast-math so GCC
# uses glibc's vector math (libmvec). -fno-associative-math stops GCC from
# reordering float adds: with it on, the AVX2 build drifted to a small but
# systematic +6e-5 price bias vs the scalar and GPU code. Only cpu_simd.cpp
# gets these flags; the scalar reference keeps normal IEEE math.
SIMDFLAGS ?= -O3 -std=c++17 -fopenmp -march=native -ffast-math -fno-associative-math
NVFLAGS  ?= -O3 -std=c++17 -arch=native -Xcompiler -fopenmp

.PHONY: all test bench-cpu run clean

all: mc_cpu test_cpu

cpu_simd.o: cpu_simd.cpp mc_core.h
	$(CXX) $(SIMDFLAGS) -c cpu_simd.cpp -o $@

# GPU build (needs the CUDA toolkit and an NVIDIA GPU)
mc: mc_option.cu cpu_simd.o mc_core.h
	$(NVCC) $(NVFLAGS) mc_option.cu cpu_simd.o -o $@ -lgomp -lm

# CPU-only build of the same benchmark (any machine with g++)
mc_cpu: mc_option.cu cpu_simd.o mc_core.h
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -x c++ mc_option.cu -x none cpu_simd.o -o $@ -lm

# CPU correctness test: European vs Black-Scholes, Asian vs an independent
# control-variate reference. No GPU needed; this is what CI runs.
test_cpu: test_cpu.cpp cpu_simd.o mc_core.h
	$(CXX) $(CXXFLAGS) test_cpu.cpp cpu_simd.o -o $@ -lm

test: test_cpu
	./test_cpu

bench-cpu: mc_cpu
	mkdir -p results
	./mc_cpu --csv results/cpu_only.csv

run: mc
	mkdir -p results
	./mc --csv results/results.csv

clean:
	rm -f mc mc_cpu test_cpu *.o
