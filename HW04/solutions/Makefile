# Warnings
WFLAGS := -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT := -O3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

# OpenMP flags
OMP := -fopenmp

# Linker options
LDFLAGS := $(OMP)

# Names of executables to create
EXEC := $(foreach f,1 2,problem$f)

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address $(OMP)
debug : $(EXEC)

all : Makefile $(EXEC)

%.o : %.c Makefile
	@ echo Compiling $<...
	@ $(CC) $(CCSTD) $(WFLAGS) -Wno-sign-conversion -Wno-unused-result $(OPT) $(CFLAGS) $(OMP) -c $< -o $@

%.o : %.cpp Makefile
	@ echo Compiling $<...
	@ $(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(CXXFLAGS) $(OMP) -c $< -o $@

$(EXEC) : % : %.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	@ rm -f *.o $(EXEC) *.in

.PHONY: dist
dist:
	@ tar -zcf hw03.tar.gz *.cpp *.hpp Makefile
