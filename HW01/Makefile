# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := 

# Names of executables to create
EXEC	:= problem1.exe problem2.exe

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : Makefile $(EXEC)

%.o : %.c Makefile
	@ echo Compiling $<...
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -c $< -o $@

%.o : %.cpp Makefile
	@ echo Compiling $<...
	$(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(ARCH) $(CXXFLAGS) -c $< -o $@

# Here is an example of how to create a target to produce an executable
# from a source file called 'problem1.cpp'. If you are using C, then
# you will need to change $(CXX) to $(CC).
problem1.exe :  problem1.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	@ rm -f *.o *.exe
