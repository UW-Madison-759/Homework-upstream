#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	if (argc != 3) {
		printf("Usage %s imageSize featureSize\n", argv[0]);
		return -1;
	}

	int imageSize = atoi(argv[1]);
	int featureSize = atoi(argv[2]);

	if (featureSize > imageSize) {
		printf("Error! Feature image cannot be larger\n");
	}

	int featureStartIndex = imageSize / 2 - 1; // to put feature matrix in center
	if (featureStartIndex + featureSize >= imageSize) { // Making sure feature not out of bound
		featureStartIndex = imageSize - featureSize - 1;
		if (featureStartIndex < 0)
			featureStartIndex = 0; //corner case
	}

	int featureEndIndex = featureStartIndex + featureSize;

	FILE *fp = fopen("problem3.dat", "w");

	// writing image
	for (int i = 0; i < imageSize; i++) {
		for (int j = 0; j < imageSize; j++) {
			if (i >= featureStartIndex && i < featureEndIndex && j >= featureStartIndex && j < featureEndIndex)
				fprintf(fp, "%d ", 1);
			else if (j % 2 == 0 && i < imageSize / 2)
				fprintf(fp, "%d ", -1);
			else if (j % 2 == 1 && i < imageSize / 2)
				fprintf(fp, "%d ", 1);
			else if (j % 2 == 0 && i >= imageSize / 2)
				fprintf(fp, "%d ", 1);
			else if (j % 2 == 1 && i >= imageSize / 2)
				fprintf(fp, "%d ", -1);
		}
		fprintf(fp, "\n");
	}

	// writing feature
	for (int i = 0; i < featureSize; i++) {
		for (int j = 0; j < featureSize; j++) {
			fprintf(fp, "%d ", 1);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
	return 0;
}
