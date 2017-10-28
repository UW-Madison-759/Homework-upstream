#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void myKernel(double* dA, double* dB, double* dC, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > size)
		return;
	dC[index] = dA[index] + dB[index];
}

double* read_array(const char* filename, int len) {
	double *x = (double*) malloc(len * sizeof(double));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%lf", &x[i]);
	}
	fclose(fp);
	return x;
}

int main(int argc, char *argv[]) {
	if (argc != 3) {
		printf("Invalid argument Usage: ./problem3 N M");
		return -1;
	}

	const int N = atoi(argv[1]);
	const int M = atoi(argv[2]);

	//defining variables for timing
	cudaEvent_t startEvent_inc, stopEvent_inc, startEvent_exc, stopEvent_exc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventCreate(&startEvent_exc);
	cudaEventCreate(&stopEvent_exc);
	float elapsedTime_inc, elapsedTime_exc;

	double *hA = read_array("inputA.inp", N);
	double *hB = read_array("inputB.inp", N);
	double *hC = (double*) malloc(N * sizeof(double));
	double *refC = (double*) malloc(N * sizeof(double)); // Used to verify functional correctness

	for (int i = 0; i < N; i++)
		refC[i] = hA[i] + hB[i];

	cudaEventRecord(startEvent_inc, 0); // starting timing for inclusive

	double *dA, *dB, *dC;
	cudaMalloc((void**) &dA, sizeof(double) * N);
	cudaMalloc((void**) &dB, sizeof(double) * N);
	cudaMalloc((void**) &dC, sizeof(double) * N);

	cudaMemcpy(dA, hA, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double) * N, cudaMemcpyHostToDevice);

	cudaEventRecord(startEvent_exc, 0); // staring timing for exclusive

	myKernel<<<N/M+1,M>>>(dA,dB,dC,N);

	cudaEventRecord(stopEvent_exc, 0);  // ending timing for exclusive
	cudaEventSynchronize(stopEvent_exc);
	cudaEventElapsedTime(&elapsedTime_exc, startEvent_exc, stopEvent_exc);

	cudaMemcpy(hC, dC, sizeof(double) * N, cudaMemcpyDeviceToHost);

	cudaEventRecord(stopEvent_inc, 0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);

	printf("%d\n%d\n%.3f\n%.3f\n%.3g\n", N, M, elapsedTime_exc, elapsedTime_inc, hC[N - 1]);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	free(refC);
	free(hB);
	free(hA);

	return 0;
}
