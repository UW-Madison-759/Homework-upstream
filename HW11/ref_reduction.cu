// Reference Reduction scan - Author: Jeiru Hu 
#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define BLOCK_SIZE 1024


__device__ void warpreduce(volatile float *s_in, int threadId)
{
	s_in[threadId]+=s_in[threadId+32];
	if(threadId<16)s_in[threadId]+=s_in[threadId+16];
	if(threadId<8)s_in[threadId]+=s_in[threadId+8];
	if(threadId<4)s_in[threadId]+=s_in[threadId+4];
	if(threadId<2)s_in[threadId]+=s_in[threadId+2];
	if(threadId<1)s_in[threadId]+=s_in[threadId+1];




}

__global__ void reduction(float *g_data, float *d_out, int n)
{
	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	//int bSize = blockDim.x;
	__shared__ float s_in[1024];
    int startSize = BLOCK_SIZE;
    	
	if(threadId < n){
		s_in[threadId]=g_data[blockId*BLOCK_SIZE + threadId];
	}
	if(threadId >= n) {
		s_in[threadId] = 0.0;
	}
    //synchronize the threads to make sure all the data is loaded
    __syncthreads();
    
    for(unsigned int i=startSize/2;i>32;i>>=1){
        //add the second half of the data to the first half
        if(threadId < i){
            s_in[threadId] += s_in[threadId+i];
        }
        //synchronize the threads to make sure all the caculation is made
        __syncthreads();
    }
   if(threadId<32)
	warpreduce(s_in,threadId);
    //copy the result back
    if(threadId==0){
		d_out[blockId] = s_in[0];
    }
}

float reductionOnDevice(float *d_in, int num) {
	int blockx = (num + BLOCK_SIZE - 1)/BLOCK_SIZE;
	dim3 dimGrid(blockx, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	float *d_out;
	assert(cudaSuccess == cudaMalloc(&d_out, blockx*sizeof(float)));
	float *dd_out;
	float *ddd_out;
	float *ret = (float *)malloc(sizeof(float));
	
	int t = (blockx == 1) ? num:BLOCK_SIZE;
	reduction<<<dimGrid, dimBlock>>>(d_in, d_out, t);
	if(blockx == 1) {
		cudaMemcpy(ret, d_out, sizeof(float), cudaMemcpyDeviceToHost);
		return ret[0];
	}
	
	
	
	if(blockx > BLOCK_SIZE){
		//use several blocks in second level reduction
		int blockxx = (blockx + BLOCK_SIZE -1)/BLOCK_SIZE;
		dim3 dimGridd(blockxx, 1, 1);
		int tt;
		tt = (blockx == 1) ? num:BLOCK_SIZE;
		cudaMalloc(&dd_out, blockxx*sizeof(float));
		reduction<<<dimGridd, dimBlock>>>(d_out,dd_out,tt);
		//can use a single block
		cudaMalloc(&ddd_out, sizeof(float));
		reduction<<<1, dimBlock>>>(dd_out,ddd_out,blockxx);
		cudaMemcpy(ret, ddd_out, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(dd_out);
		cudaFree(ddd_out);
	} else {
		//can use a single block
		cudaMalloc(&ddd_out, sizeof(float));
		reduction<<<1, dimBlock>>>(d_out,ddd_out,blockx);
		cudaMemcpy(ret, ddd_out, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(ddd_out);
	}
	cudaFree(d_out);
	return ret[0];
}

float* read_array(const char* filename, int len) {
	float *x = (float*) malloc(len * sizeof(float));
	FILE *fp = fopen(filename, "r");
        for( int i=0; i<len; i++){
		int r=fscanf(fp,"%f",&x[i]);		
		if(r == EOF){
			rewind(fp);
		}
		x[i]-=5;
         }
	fclose(fp);
	return x;
}
/*
float* read_array(const char* filename, int len) {
	float *x = (float*) malloc(len * sizeof(float));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%f", &x[i]);
	}
	fclose(fp);
	return x;
}
*/
void computeSum( float* reference, float* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  unsigned int i;
  for( i = 0; i < len; ++i) 
  {
      total_sum += idata[i];
  }
  *reference = total_sum;
}

int main( int argc, char** argv) 
{
	if(argc != 2) {
		fprintf(stderr, "usage: ./problem2 N\n");
		exit(1);
	}
	int num_elements = atoi(argv[1]);

	float* h_data=read_array("problem1.inp",num_elements);
	
	float reference = 1.0f;  
	computeSum(&reference , h_data, num_elements);

	int size = num_elements*sizeof(float);
	float *d_in;
	assert(cudaSuccess == cudaMalloc((void**)&d_in, size));

	//start inclusive timing
	float time;
	cudaEvent_t startIn,stopIn;
	cudaEventCreate(&startIn);
	cudaEventCreate(&stopIn);
	cudaEventRecord(startIn, 0);	
	
	assert(cudaSuccess == cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice));
	//float result = computeOnDevice(h_data, num_elements);
	float result = reductionOnDevice(d_in, num_elements);
	
	//stop inclusive timing
	cudaEventRecord(stopIn, 0);     
	cudaEventSynchronize(stopIn);
	cudaEventElapsedTime(&time, startIn, stopIn);     
	cudaEventDestroy(startIn); 
	cudaEventDestroy(stopIn);

	// Run accuracy test
	float epsilon = 0.3f;
	unsigned int result_regtest = (abs(result - reference) <= epsilon);

	if(!result_regtest)	printf("Test failed device: %f  host: %f\n",result,reference);
	//print the outputs
	printf("%d\n%f\n%f\n",num_elements, result, time);
	//printf("%f\n", time);
	// cleanup memory
	cudaFree(d_in);  
	//cudaFree(d_out);
	free( h_data);
	return 0;
}
