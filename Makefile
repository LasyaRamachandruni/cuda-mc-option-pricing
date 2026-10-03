# GPU build (needs the CUDA toolkit and an NVIDIA GPU)
mc: mc_option.cu
	nvcc -O3 -arch=native -Xcompiler -fopenmp -o mc mc_option.cu

# CPU-only build (any machine with g++), for checking correctness
mc_cpu: mc_option.cu
	g++ -O3 -fopenmp -x c++ -DCPU_ONLY -o mc_cpu mc_option.cu

run: mc
	./mc

clean:
	rm -f mc mc_cpu results.csv
