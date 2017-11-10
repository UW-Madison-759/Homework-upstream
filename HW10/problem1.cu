#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <math.h>


int checkResults(float*res, float* cudaRes,int length)
{
	int nDiffs=0;
	const float smallVal = 0.01f; // Keeping this extra high as we have repetitive addition and sequence matters
	for(int i=0; i<length; i++)
		if(fabs(cudaRes[i]-res[i])>smallVal)
			nDiffs++;
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


int main(int argc, char* argv[]) {
	if(argc!=2){
		printf("Usage %s N\n",argv[0]);
		return 1;
	}
	int N=atoi(argv[1]);
	FILE *fp = fopen("problem1.inp","r");
	int size = N * sizeof(float); 
	//allocate resources
	float *in      = (float *)malloc(size);
	float *out     = (float *)malloc(size); 
	float *cuda_out= (float *)malloc(size);
	float time = 0.f;
	initializeArray(fp,in, N);
	// Your code here

	inclusiveScan_SEQ(in, out,N);
	int nDiffs = checkResults(out, cuda_out,N);

	if(nDiffs)printf("Test Failed\n"); // This should never print
	printf("%d\n%f\n%f\n",N,cuda_out[N-1],time);



	//free resources 
	free(in); free(out); free(cuda_out);
	return 0;
}
