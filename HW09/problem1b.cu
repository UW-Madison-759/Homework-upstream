#include<iostream>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>

#define RADIUS 3

int checkResults(int startElem, int endElem, float* cudaRes, float* res)
{
    int nDiffs=0;
    const float smallVal = 0.0001f;
    for(int i=startElem; i<endElem; i++)
        if(fabs(cudaRes[i]-res[i])>smallVal)
            nDiffs++;
    return nDiffs;
}

void initializeWeights(float* weights, int rad)
{
    // for now hardcoded for RADIUS=3
    weights[0] = 0.50f;
    weights[1] = 0.75f;
    weights[2] = 1.25f;
    weights[3] = 2.00f;
    weights[4] = 1.25f;
    weights[5] = 0.75f;
    weights[6] = 0.50f;
}
void initializeArray(FILE* fp,float* arr, int nElements)
{
    for( int i=0; i<nElements; i++){
                fscanf(fp,"%f",&arr[i]);
                if(getc(fp) == EOF) rewind(fp);
    }
}

void applyStencil1D_SEQ(int sIdx, int eIdx, const float *weights, float *in, float *out) {
  
  for (int i = sIdx; i < eIdx; i++) {   
    out[i] = 0;
    //loop over all elements in the stencil
    for (int j = -RADIUS; j <= RADIUS; j++) {
      out[i] += weights[j + RADIUS] * in[i + j]; 
    }
    out[i] = out[i] / (2 * RADIUS + 1);
  }
}

__global__ void applyStencil1D(int sIdx, int eIdx, const float *weights, float *in, float *out) {
    int i = sIdx + blockIdx.x*blockDim.x + threadIdx.x;
    if( i < eIdx ) {
        float result = 0.f;
        result += weights[0]*in[i-3];
        result += weights[1]*in[i-2];
        result += weights[2]*in[i-1];
        result += weights[3]*in[i];
        result += weights[4]*in[i+1];
        result += weights[5]*in[i+2];
        result += weights[6]*in[i+3];
        result /=7.f;
        out[i] = result;
    }
}

int main(int argc, char* argv[]) {
  if(argc!=2){
        printf("Usage %s N\n",argv[0]);
	return 1;
  }
  int N=atoi(argv[1]);
  FILE *fp = fopen("problem1.inp","r");
  int size = N * sizeof(float); 
  int wsize = (2 * RADIUS + 1) * sizeof(float); 
  //allocate resources
  float *weights, *in, *cuda_out; 
  cudaMallocHost((void **)&weights, wsize);
  cudaMallocHost((void **)&in, size); 
  cudaMallocHost((void **)&cuda_out, size); 

  float *out     = (float *)malloc(size); 
  float time = 0.f;
  initializeWeights(weights, RADIUS);
  initializeArray(fp,in, N);
  float *d_weights;  cudaMalloc(&d_weights, wsize);
  float *d_in;       cudaMalloc(&d_in, size);
  float *d_out;      cudaMalloc(&d_out, size);
  
  cudaMemcpy(d_weights,weights,wsize,cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  applyStencil1D<<<(N+511)/512, 512>>>(RADIUS, N-RADIUS, d_weights, d_in, d_out);
  cudaMemcpy(cuda_out, d_out, size, cudaMemcpyDeviceToHost);

  applyStencil1D_SEQ(RADIUS, N-RADIUS, weights, in, out);
  int nDiffs = checkResults(RADIUS, N-RADIUS, cuda_out, out);
  if(nDiffs)printf("Test Failed\n"); // This should never print
  printf("%f\n%f\n",cuda_out[N-RADIUS-1],time);
  //free resources 
  cudaFreeHost(weights); cudaFreeHost(in); cudaFreeHost(cuda_out); 
  free(out);
  cudaFree(d_weights);  cudaFree(d_in);  cudaFree(d_out);
  return 0;
}
