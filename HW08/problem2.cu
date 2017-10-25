#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// includes, project

// includes, kernels

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

double* read_array(const char* filename, int len) {
	double *x = (double*) malloc(len * sizeof(double));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%lf", &x[i]);
	}
	fclose(fp);
	return x;
}

void computeOnDevice(double* hA,double* hB, double* hC, int nRows, int tileSize, float* incTime );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv) 
{
	if(argc!=2)
	{
		printf("Usage: ./problem2 N\n");
		return 0;
	}
	int nRows = 1024;
	int num_elements = nRows*nRows;
	int tileSize = atoi(argv[1]);  //change this for scaling analysis
	float incTime=0; // Time for GPU
	double* hA = read_array("inputA.inp",num_elements);
	double* hB = read_array("inputB.inp",num_elements);
	double* hC = (double*) malloc(num_elements * sizeof(double));

	// **===-------- Modify the body of this function -----------===**
	computeOnDevice( hA, hB,hC, nRows, tileSize, &incTime);
	// **===-----------------------------------------------------------===**


	printf("%f\n%f\n%d\n",hC[num_elements-1],incTime,tileSize);
	// cleanup memory
	free(hA);
	free(hB);
	free(hC);

	return 0;
}



void computeOnDevice(double* hA,double* hB, double* hC, int nRows, int TileSize, float* incTime)
{
	
	return;//Placeholder
}


