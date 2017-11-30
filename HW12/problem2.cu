#include <stdio.h>
#include <cuda.h>

#define CUDA_CHECK(value, label) {              \
   cudaError_t c = (value);                     \
   if (c != cudaSuccess) {                      \
   fprintf(stderr,                              \
     "Error: '%s' at line %d in %s\n",          \
     cudaGetErrorString(c),__LINE__,__FILE__);  \
   goto label;                                  \
   } }

static __global__ void prefix_scan_device(float *in, float *out, int size) {
	// Do CUDA stuff
}

void prefix_scan(float *in, float *out, int size) {
	float *d_in=0, *d_out=0;
	CUDA_CHECK(cudaMalloc(&d_in, size * sizeof(float)), cuda_error)
	CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(float)), cuda_error)
	
	CUDA_CHECK(cudaMemcpy(d_in, in, size * sizeof(float), cudaMemcpyHostToDevice), cuda_error)
	prefix_scan_device<<<128, 1>>>(d_in, d_out, size);
	CUDA_CHECK(cudaMemcpy(out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost), cuda_error)

cuda_error:
	if(d_in) cudaFree(d_in);
	if(d_out) cudaFree(d_out);
}
