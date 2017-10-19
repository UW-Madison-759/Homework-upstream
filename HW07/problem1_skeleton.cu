#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


int* read_array(const char* filename, int len) {
	int *x = (int*) malloc(len * sizeof(int));
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < len; i++) {
		fscanf(fp, "%d", &x[i]);
	}
	fclose(fp);
	return x;
}

int main(int argc, char *argv[]) {
	if (argc != 1) {
		printf("Invalid argument Usage: ./problem1");
		return -1;
	}

	const int rowWidth=32;
        const int colWidth=16;	
	int *hA = read_array("inputA.inp",rowWidth*colWidth );
	int *hB = read_array("inputB.inp", rowWidth);
	int *hC = (int*) malloc(colWidth * sizeof(int));
	int *refC;
	// TODO - allocate host memory for refC (you have to figure out how much)
	// The skeleton currently segfaults because refC is accessed without allocation

	// TODO do a reference host implementation (Ch) here. ie populate answer in refC



	int *dA, *dB, *dC;
	// TODO allocate device memory for dA,dB and dC


	// TODO copy data from host to GPU 


	// TODO call your kernel

	// TODO copyback results
	float Error=0;

	for(int i=0;i<colWidth;i++)
		Error+=(hC[i]-refC[i])*(hC[i]-refC[i]);
	printf("%f\n%d",sqrt(Error),hC[colWidth-1]);

	free(refC);
	free(hB);
	free(hA);

	return 0;
}
