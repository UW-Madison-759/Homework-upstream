#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sum(int *x) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	x[index] = blockIdx.x + threadIdx.x;
}

int main() {	
	const int N = 16;
	int x[N];
	int *dArray;
	cudaMalloc((void**) &dArray, sizeof(int) * N);
	
	sum<<<2,8>>>(dArray);
	cudaMemcpy(x, dArray, sizeof(int) * N, cudaMemcpyDeviceToHost);	
	
	for(int i=0; i<N; i++)
		printf("%d\n", x[i]);
	
	cudaFree(dArray);
	return 0;
}
