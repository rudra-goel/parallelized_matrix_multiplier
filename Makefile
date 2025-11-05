CFLAGS= -O3 -Wextra
NVCC=nvcc
NVCC_FLAGS= 
DEPS=

matrix: matrix.cu $(DEPS)
	$(NVCC) -o $@ $< $(NVCC_FLAGS) $(addprefix -Xcompiler ,$(CFLAGS))

clean:
	rm -rf matrix

test-cpu: matrix
	./matrix 0

test-gpu-naive: matrix
	./matrix 1

test-gpu-tiled: matrix
	./matrix 2

test: test-cpu test-gpu-naive test-gpu-tiled

q5: q5.cu util.h
	$(NVCC) -o q5 q5.cu $(NVCC_FLAGS)


.PHONY: clean test-cpu test-gpu-naive test-gpu-tiled test