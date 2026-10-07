CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
# The SIMD baseline needs -march=native for AVX2/AVX-512 and -ffast-math so GCC
# uses glibc's vector math (libmvec). Only cpu_simd.cpp gets these flags; the
# scalar reference keeps normal IEEE math.
SIMDFLAGS ?= -O3 -std=c++17 -fopenmp -march=native -ffast-math
NVFLAGS  ?= -O3 -std=c++17 -arch=native -Xcompiler -fopenmp

.PHONY: all bench-cpu run clean

all: mc_cpu

cpu_simd.o: cpu_simd.cpp mc_core.h
	$(CXX) $(SIMDFLAGS) -c cpu_simd.cpp -o $@

# GPU build (needs the CUDA toolkit and an NVIDIA GPU)
mc: mc_option.cu cpu_simd.o mc_core.h
	$(NVCC) $(NVFLAGS) mc_option.cu cpu_simd.o -o $@ -lgomp -lm

# CPU-only build of the same benchmark (any machine with g++)
mc_cpu: mc_option.cu cpu_simd.o mc_core.h
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -x c++ mc_option.cu -x none cpu_simd.o -o $@ -lm

bench-cpu: mc_cpu
	mkdir -p results
	./mc_cpu --csv results/cpu_only.csv

run: mc
	mkdir -p results
	./mc --csv results/results.csv

clean:
	rm -f mc mc_cpu test_cpu *.o
