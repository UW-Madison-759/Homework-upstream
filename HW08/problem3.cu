#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>



double* read_array(const char* filename, int len) {
	double *x = (double*) malloc(len * sizeof(double));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%lf", &x[i]);
	}
	fclose(fp);
	return x;
}

void computeOnDevice(double* hA,double* hB, double* hC, int nRows,
	int nInnerDimension,int nCols, int tileSize, float* incTime );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv) 
{
	if(argc!=5)
        {
                printf("Usage: ./problem2 i j k N\n");
                return 0;
        }


	int nRows = atoi(argv[1]);
	int nInnerDimension = atoi(argv[2]);
	int nCols = atoi(argv[3]);
	int num_elementsA= nRows*nInnerDimension;
	int num_elementsB=nInnerDimension*nCols;
	int num_elementsC= nRows*nCols;
	int tileSize = atoi(argv[4]);  //change this for scaling analysis
	float incTime=0; // Time for GPU
	double* hA = read_array("problem3.inp",num_elementsA);
	double* hB = read_array("problem3.inp",num_elementsB);
	double* hC = (double*) malloc(num_elementsC * sizeof(double));

	// **===-------- Modify the body of this function -----------===**
	computeOnDevice( hA, hB,hC, nRows, nInnerDimension, nCols, tileSize, &incTime);
	// **===-----------------------------------------------------------===**


	printf("%f\n%f\n%d\n%d\n%d\n",hC[num_elementsC-1],incTime,tileSize,nRows,nCols);
	// cleanup memory
	free(hA);
	free(hB);
	free(hC);

	return 0;
}



void computeOnDevice(double* hA,double* hB, double* hC, int nRows, int nInnerDimension, int nCols, int TileSize, float* incTime)
{
	
	return;//Placeholder
}


