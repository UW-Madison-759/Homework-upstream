# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

# Linker options
LDFLAGS := 

# Names of executables to create
EXEC	:= problem1.exe problem2.exe problem3.exe problem3b.exe

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : $(EXEC)

all : Makefile $(EXEC)

%.o : %.c Makefile
	@ echo Compiling $<...
	@ $(CC) $(CCSTD) $(WFLAGS) $(OPT) $(CFLAGS) -c $< -o $@

%.o : %.cpp Makefile
	@ echo Compiling $<...
	@ $(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(CXXFLAGS) -c $< -o $@

problem1.exe : problem1.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

problem2.exe : problem2.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

problem3.exe : problem3.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

problem3b.exe : problem3b.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	@ rm -f *.o $(EXEC)
