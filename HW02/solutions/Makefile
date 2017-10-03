# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := 

# Names of executables to create
EXEC := $(foreach f,1a 1b 1c 2a 2b 3a 3b 3c,problem$f.exe)

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : Makefile $(EXEC)

%.o : %.c Makefile
	@ echo Compiling $<...
	@ $(CC) $(CCSTD) $(WFLAGS) -Wno-sign-conversion -Wno-unused-result $(OPT) $(ARCH) $(CFLAGS) -c $< -o $@

%.o : %.cpp Makefile
	@ echo Compiling $<...
	@ $(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(ARCH) $(CXXFLAGS) -c $< -o $@

# This is a static pattern rule. Combined with the definition of EXEC above,
# it greatly simplifies the Makefile
#
# See the docs for details:
# https://www.gnu.org/software/make/manual/html_node/Static-Usage.html#Static-Usage
$(EXEC) : %.exe : %.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	@ rm -f *.o $(EXEC) *.out *.in
