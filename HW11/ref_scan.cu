// Reference Scan implementation - Author: Ananoymous student of ME759 Fall 2017
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>


int checkResults(float*res, float* cudaRes,int length)
{
	int nDiffs=0;
	const float smallVal = .3f; // Keeping this extra high as we have repetitive addition and sequence matters
	for(int i=0; i<length; i++)
		if(fabs(cudaRes[i]-res[i])>smallVal)
			{nDiffs++;
       //printf("%f %f\n",cudaRes[i],res[i]);
      }
	return nDiffs;
}

void initializeArray(FILE* fp,float* arr, int nElements)
{
	for( int i=0; i<nElements; i++){
		int r=fscanf(fp,"%f",&arr[i]);
		if(r == EOF){
			rewind(fp);
		}
		arr[i]-=5; // This is to make the data zero mean. Otherwise we reach large numbers and lose precision
	}
}

void inclusiveScan_SEQ(float *in, float *out,int length) {
	float sum=0.f;
	for (int i =0; i < length; i++) {
		sum+=in[i];
		out[i]=sum;
	}
}

__global__ void scan(float *g_odata, float *g_idata,int n){
  extern volatile __shared__ float temp[];
  int thid = threadIdx.x;
  int pout = 0,pin = 1;
  
  if(thid<n)
    temp[thid] = g_idata[thid];
  else
    temp[thid] = 0.0f;
  __syncthreads();
  
  for(int offset = 1;offset<n;offset <<=1)
  {
    pout = 1- pout;
    pin = 1 - pout;
    
    if(thid >= offset)
      temp[pout*n+thid] = temp[pin*n+thid]+temp[pin*n+thid-offset];
    else
      temp[pout*n+thid] = temp[pin*n+thid];
      
    __syncthreads();
  }
  if(thid<n)
    g_odata[thid] = temp[pout*n+thid];
}

__global__ void scanlarge(float *g_odata, float *g_idata,float *aux_data,int n,int arraysize){
  extern volatile __shared__ float temp[];
  int thid = threadIdx.x;
  int start = blockIdx.x*1024;
  int aux_in = blockIdx.x;
  int pout = 0,pin = 1;
  
  if(thid+start<arraysize)
    temp[thid] = g_idata[thid+start];
  else
    temp[thid] = 0.00;
  __syncthreads();
  
  for(int offset = 1;offset<n;offset <<=1)
  {
    pout = 1- pout;
    pin = 1 - pout;
    
    if(thid >= offset)
      temp[pout*n+thid] = temp[pin*n+thid]+temp[pin*n+thid-offset];
    else
      temp[pout*n+thid] = temp[pin*n+thid];
      
    __syncthreads();
  }
  if(thid+start<arraysize){
    g_odata[thid+start] = temp[pout*n+thid];}
  aux_data[aux_in] = temp[1023];
}

__global__ void addscan(float *g_odata,float *g_idata,float *aux_data,int arraysize){
  extern volatile __shared__ float temp[];
  int thid = threadIdx.x;
  int start = (blockIdx.x+1)*1024;
  int aux_in = blockIdx.x;
  
  if(thid+start<arraysize)
    temp[thid] = g_idata[thid+start];
  else
    temp[thid] = 0.00;
  __syncthreads();
  
  temp[thid]+=aux_data[aux_in];
  __syncthreads();  
  if(thid+start<arraysize)
    g_odata[thid+start]=temp[thid];
}


int main(int argc, char* argv[]) {
	if(argc!=2){
		printf("Usage %s N\n",argv[0]);
		return 1;
	}
	int N=atoi(argv[1]);
	FILE *fp = fopen("problem1.inp","r");
	int size = N * sizeof(float); 
	//allocate resources
  int threadsperblock,blocksPerGrid;
	float *in      = (float *)malloc(size);
	float *out     = (float *)malloc(size); 
	float *cuda_out= (float *)malloc(size);
	float time = 0.f;
	initializeArray(fp,in, N);
	// Your code here
 
  float *dout,*din;
  
  cudaMalloc((void**)&dout,size);
	cudaMalloc((void**)&din,size);
 
  cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
  cudaEventRecord(startEvent_inc,0); // starting timing for inclusive  
  
  cudaMemcpy(din,in,size,cudaMemcpyHostToDevice);
  cudaMemset(dout,0,size);
  
  if(N>1024)
  {
     threadsperblock = 1024;
     blocksPerGrid = (N+threadsperblock-1)/threadsperblock;
     float *aux;
     float *auxscan;
     int num = 1024;
     cudaMalloc((void**)&aux,sizeof(float)*blocksPerGrid);
     cudaMemset(aux,0,sizeof(float)*blocksPerGrid);
     cudaMalloc((void**)&auxscan,sizeof(float)*blocksPerGrid);
     cudaMemset(auxscan,0,sizeof(float)*blocksPerGrid);
     
     if(blocksPerGrid<=1024)
     {
       scanlarge<<<blocksPerGrid,threadsperblock,2048*sizeof(float)>>>(dout,din,aux,num,N);
       cudaDeviceSynchronize();
       // Scanning the auxilliary array
       scan<<<1,blocksPerGrid,2*blocksPerGrid*sizeof(float)>>>(auxscan,aux,blocksPerGrid);
       cudaDeviceSynchronize();
       // Adding the scanned block to get final result
       addscan<<<blocksPerGrid,threadsperblock,2048*sizeof(float)>>>(dout,dout,auxscan,N);
       cudaDeviceSynchronize();
       cudaMemcpy(cuda_out,dout,size,cudaMemcpyDeviceToHost);
     }
     else
     {
       float *auxblock;
       float *auxscanblock;
       float *auxout;
       volatile int blocksperGridaux = (blocksPerGrid+1023)/1024;
       //int blocksperGridaux = 2;
       cudaMalloc((void**)&auxblock,sizeof(float)*blocksperGridaux);
       cudaMemset(auxblock,0,sizeof(float)*blocksperGridaux);
       cudaMalloc((void**)&auxscanblock,sizeof(float)*blocksperGridaux);
       cudaMemset(auxscanblock,0,sizeof(float)*blocksperGridaux);
       cudaMalloc((void**)&auxout,sizeof(float)*blocksPerGrid);
       cudaMemset(auxout,0,sizeof(float)*blocksPerGrid);
       
       scanlarge<<<blocksPerGrid,threadsperblock,2048*sizeof(float)>>>(dout,din,aux,num,N); // We get the block sums here
       cudaDeviceSynchronize();
       
       
       scanlarge<<<blocksperGridaux,threadsperblock,2048*sizeof(float)>>>(auxout,aux,auxblock,num,blocksPerGrid); // Block sum array size is greater than 1024. So repeat the whole > 1024 process
       cudaDeviceSynchronize();
       scan<<<1,blocksperGridaux,2*blocksperGridaux*sizeof(float)>>>(auxscanblock,auxblock,blocksperGridaux); // Aux sum of sux array
       cudaDeviceSynchronize();
       addscan<<<blocksperGridaux-1,threadsperblock,2048*sizeof(float)>>>(auxout,auxout,auxscanblock,blocksPerGrid); // Fully scanned auxilliary array
       cudaDeviceSynchronize();
       
       
       addscan<<<blocksPerGrid,threadsperblock,2048*sizeof(float)>>>(dout,dout,auxout,N);
       cudaDeviceSynchronize();
       cudaMemcpy(cuda_out,dout,size,cudaMemcpyDeviceToHost);
     }
  }
  else{
     threadsperblock = N;
     blocksPerGrid = 1; 
     scan<<<blocksPerGrid,threadsperblock,2*size>>>(dout,din,N);
     cudaMemcpy(cuda_out,dout,size,cudaMemcpyDeviceToHost);
     } 
  cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
  cudaEventSynchronize(stopEvent_inc);   
	cudaEventElapsedTime(&time, startEvent_inc, stopEvent_inc);   
 
	inclusiveScan_SEQ(in, out,N);
	int nDiffs = checkResults(out, cuda_out,N);

	if(nDiffs)printf("Test Failed\n"); // This should never print
	printf("%d\n%f\n%f\n",N,cuda_out[N-1],time);
  //printf("%d\n",nDiffs);
  //printf("%f\n",out[N-1]);


	//free resources 
	free(in); free(out); free(cuda_out);
  cudaFree(din);cudaFree(dout);
  //cudaFree(dsize);
	return 0;
}
