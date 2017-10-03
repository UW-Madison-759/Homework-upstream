#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* read_matrix(FILE *fp, int len) {
	int *x = (int *)malloc(sizeof(int) * len * len);
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			fscanf(fp, "%d ", &x[i * len + j]);
		}
	}
	return x;
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		fprintf(stderr, "Usage %s imageSize featureSize\n", argv[0]);
		return -1;
	}

	int imageSize = atoi(argv[1]);
	int featureSize = atoi(argv[2]);

	if (featureSize > imageSize) {
		printf("Error! Feature image cannot be larger\n");
	}

	FILE *fp = fopen("problem3.dat", "r");
	int* image = read_matrix(fp, imageSize);
	int* feature = read_matrix(fp, featureSize);
	fclose(fp);

	// Reverse the rows
	int* reversedImage = (int *) malloc(sizeof(int) * imageSize * imageSize);
	for (int i = 0; i < imageSize; i++)
		memcpy(reversedImage + i * imageSize, image + (imageSize - i - 1) * imageSize, imageSize * sizeof(int));

	// Write reversed image to file
	FILE *fp2 = fopen("problem3.out", "w");
	for (int i = 0; i < imageSize; i++) {
		for (int j = 0; j < imageSize; j++) {
			fprintf(fp2, "%d ", reversedImage[i * imageSize + j]);
		}
	}

	for (int i = 0; i < featureSize; i++) {
		for (int j = 0; j < featureSize; j++) {
			fprintf(fp2, "%d ", feature[i * featureSize + j]);
		}
		fprintf(fp2, "\n");
	}
	fclose(fp2);

	free(image);
	free(feature);
	free(reversedImage);

	return 0;
}
