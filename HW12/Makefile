# Warnings
WFLAGS	:= -Wall -Wextra

# Optimization and architecture
OPT		:= -O3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

.DEFAULT_GOAL := all

EXEC := problem1 problem2

all : Makefile $(EXEC)

problem1: problem1.cpp
	mpicxx $(CXXSTD) $(WFLAGS) $(OPT) -fopenmp -o $@ $<

problem2: problem2.cpp problem2.cu
	nvcc -c -Xcompiler "$(CXXSTD) $(WFLAGS) $(OPT)" -o problem2_cuda.o problem2.cu
	mpicxx -c $(CXXSTD) $(WFLAGS) $(OPT) -fopenmp -o problem2.o problem2.cpp
	mpicxx problem2_cuda.o problem2.o -L$(CUDA_LIB_PATH) -lcudart -o problem2

.PHONY: clean
clean:
	@ rm -f $(EXEC) *.o
