/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Matrix addition: C = alpha*A  + beta*B, where alpha and beta are two scalars.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
// includes, project
//#include <cutil.h>

// includes, kernels
#include "matrixadd_kernel.cu"
#include "matrixadd.h"
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold( float*, const float*, const float, const float*, const float, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps, float * error);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void MakeN(Matrix* M, Matrix* N);

void MatrixAddOnDevice(const Matrix M, const float alpha, const Matrix N, const float beta, Matrix P, float * inc, float * exc);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Matrices for the program
	Matrix  M;
	Matrix  N;
	Matrix  P;
	// Number of elements in the solution matrix
	//  Assuming square matrices, so the sizes of M, N and P are equal
	unsigned int size_elements = WP * HP;
	int errorM = 0;

	srand(2012);
	if(argc != 2)
	{
		printf("Error Usage ./problem2 u\n");
	}
	int u=atoi(argv[1]);
	char filename[100]="problem2.inp";

	// Check command line for input matrix files
	if(u==0) 
	{
		// No inputs provided
		// Allocate and initialize the matrices
		M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
		N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
		P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	}
	else
	{
		// Inputs provided
		// Allocate and read source matrices from disk
		M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
		N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);		
		P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
		errorM = ReadFile(&M, filename);
		MakeN(&M, &N);
		// check for read errors
		if(errorM != size_elements)
		{
			printf("Error reading input files %d\n", errorM);
			return 1;
		}
	}

	// alpha*M + beta*N on the device
	float alpha = 1.f;
	float beta  = 1.f;
	//time the operation
	float inclusiveTime, exclusiveTime,norm=0; 
	MatrixAddOnDevice(M, alpha, N, beta, P,&inclusiveTime,&exclusiveTime);

	// compute the matrix addition on the CPU for comparison
	Matrix reference = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	cudaError_t error;
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	computeGold(reference.elements, M.elements, alpha, N.elements, beta, HM, WM);

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// check if the device result is equivalent to the expected solution
	bool res = CompareResults(reference.elements, P.elements, 
			size_elements, 0.0001f,&norm);
	if(res==0)printf("Test failed\n"); // This should not be printed in the correct implementation
	printf("%f\n%f\n%f\n%f\n",sqrt(norm),msecTotal,inclusiveTime, exclusiveTime);	


	// Free host matrices
	free(M.elements);
	M.elements = NULL;
	free(N.elements);
	N.elements = NULL;
	free(P.elements);
	P.elements = NULL;
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! wrapper around the device implementation
////////////////////////////////////////////////////////////////////////////////
void MatrixAddOnDevice(const Matrix M, const float alpha, const Matrix N, const float beta, Matrix P,float * inc,float *exc)
	// ADD YOUR CODE HERE
{   
	cudaEvent_t startEvent_inc, stopEvent_inc, startEvent_exc, stopEvent_exc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventCreate(&startEvent_exc);
	cudaEventCreate(&stopEvent_exc);
	float  elapsedTime_inc, elapsedTime_exc;
	cudaEventRecord(startEvent_inc,0); // starting timing for inclusive
	//Allocate device matrices

	// copy matrices to device 
	cudaEventRecord(startEvent_exc,0); // staring timing for exclusive

	//launch kernel
	cudaEventRecord(stopEvent_exc,0);  // ending timing for exclusive
	cudaEventSynchronize(stopEvent_exc);   
	cudaEventElapsedTime(&elapsedTime_exc, startEvent_exc, stopEvent_exc);
	// Read P from the device


	cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);   
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);
	*inc = elapsedTime_inc;
	*exc = elapsedTime_exc;	
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
	cudaError_t error;
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	error = cudaMalloc((void**)&Mdevice.elements, size);
	if (error != cudaSuccess)
	{
		printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
	return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix AllocateMatrix(int height, int width, int init)
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;

	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	}
	return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
			cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
			cudaMemcpyDeviceToHost);
}
//compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps,float * error)
{
	for(unsigned int i = 0; i < elements; i++){
		float temp = sqrt((A[i]-B[i])*(A[i]-B[i]));
		*error+=temp;
		if(temp>eps){
			return false;
		} 
	}
	return true;
}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = MATRIX_SIZE*MATRIX_SIZE;
	std::ifstream ifile(file_name);

	for(unsigned int i = 0; i < data_read; i++){
		ifile>>M->elements[i];
	}
	ifile.close();
	return data_read;
}

// Read a 16x16 floating point matrix in from file
void MakeN(Matrix* M, Matrix* N)
{
	unsigned int data_read = MATRIX_SIZE*MATRIX_SIZE;

	for(unsigned int i = 0; i < data_read; i++){
		N->elements[i]=1.f/(0.2f+M->elements[i]);
	}
}


// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
	std::ofstream ofile(file_name);
	for(unsigned int i = 0; i < M.width*M.height; i++){
		ofile<<M.elements[i];
	}
	ofile.close();
}
