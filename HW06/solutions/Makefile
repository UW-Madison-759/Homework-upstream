# Warnings
WFLAGS := -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT  := -O3

# OpenMP
OPENMP := -fopenmp

# Language standard
CXXSTD	:= -std=c++14

# Linker options
LDFLAGS := $(OPENMP)

# Names of executables to create
EXEC := problem1A problem1B

.DEFAULT_GOAL := all
all : Makefile $(EXEC) problem2 problem3

%.o : %.cpp Makefile
	@ echo Compiling $<...
	@ $(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(CXXFLAGS) $(OPENMP) -c $< -o $@

$(EXEC) : % : %.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

problem2: problem2.cu
	@ echo Building $@...
	@ @ nvcc $< -o $@

problem3: problem3.cu
	@ echo Building $@...
	@ @ nvcc $< -o $@

.PHONY: clean
clean:
	@ rm -f *.o $(EXEC) problem2 problem3
