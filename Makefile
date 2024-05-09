CUDA_TOOLKIT := /usr/local/cuda-12.0
INC          := -I$(CUDA_TOOLKIT)/include -I./
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcusparse

all: spmm_blockedell_example

spmm_blockedell_example: spmm_blockedell_example.cpp
	nvcc -std=c++11 $(INC) spmm_blockedell_example.cpp mmio.c smsh.c -o spmm_blockedell_example $(LIBS)
	gcc mmio.h

clean:
	rm -f spmm_blockedell_example

test:
	@echo "\n==== SpMM BLOCKED ELL Test ====\n"
	./spmm_blockedell_example

.PHONY: clean all test
