#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int* read_matrix(FILE *fp, int len) {
	int *x = (int *) malloc(sizeof(int) * len * len);
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			fscanf(fp, "%d", &x[i * len + j]);
		}
	}
	return x;
}

int main(int argc, char* argv[]) {
	/*
	 * 	The problem statement has only one argument, but passing the name via
	 * 	the command line is a more portable way of handling this. This program
	 * 	would then be run from the terminal or an sbatch script like so
	 *
	 * 		./problem2 1 9 2 $(uname -n)
	 *
	 * 	For now, we just have a placeholder for the server name.
	 */
	if (argc != 4) {
		fprintf(stderr, "Usage %s number_of_threads imageSize featureSize\n", argv[0]);
		return -1;
	}
	int N = atoi(argv[1]);
	int imageSize = atoi(argv[2]);
	int featureSize = atoi(argv[3]);
	const char server_name[] = "foo";

	if (featureSize > imageSize) {
		fprintf(stderr, "Error! Feature image cannot be larger\n");
		return -1;
	}

	FILE *fp = fopen("problem2.dat", "r");
	int* image = read_matrix(fp, imageSize);
	int* feature = read_matrix(fp, featureSize);
	int similarityMatrixSize = imageSize - featureSize + 1; // matrix containing correlation values
	int* values = (int *) malloc(sizeof(int) * similarityMatrixSize * similarityMatrixSize);
	fclose(fp);

	omp_set_num_threads(N);

	double start = omp_get_wtime();

	// Reverse the rows
	int* reversedImage = (int *) malloc(sizeof(int) * imageSize * imageSize);
	for (int i = 0; i < imageSize; i++)
		memcpy(reversedImage + i * imageSize, image + (imageSize - i - 1) * imageSize, imageSize * sizeof(int));

	// cross correlation
	int finalX=-1, finalY=-1, max = -featureSize * featureSize; // set max to minimum
	int distance = 2 * imageSize; // set distance to maximum
#pragma omp parallel for 
	for (int i = 0; i < similarityMatrixSize; i++) {
		for (int j = 0; j < similarityMatrixSize; j++) {
			for (int ii = 0; ii < featureSize; ii++) {
				for (int jj = 0; jj < featureSize; jj++) {
					values[i * (similarityMatrixSize) + j] += reversedImage[(i + ii) * imageSize + (j + jj)] * feature[ii * featureSize + jj];
				}
			}
		}
	}

	for (int i = 0; i < similarityMatrixSize * similarityMatrixSize; i++) // single loop maybe better for branch prediction
		if (values[i] > max) {
			max = values[i];
			finalX = i / similarityMatrixSize;
			finalY = i % similarityMatrixSize;
		} else if (values[i] == max && distance > (i)) {
			finalX = i / similarityMatrixSize;
			finalY = i % similarityMatrixSize;
			distance = i;
		}

	double stop = omp_get_wtime();

	printf("%d %d %d %lf %s %d %d %d\n", N, imageSize, featureSize, (stop - start) * 1000, server_name, finalX, finalY, max);

	free(image);
	free(feature);
	free(reversedImage);
	free(values);
	return 0;
}
