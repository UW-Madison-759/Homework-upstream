#include <stdio.h>
#include <string.h>

int main() {
	FILE *fp = fopen("problem1.txt", "r");
	char id[20] = {'\0'};
	fscanf(fp, "%20s", id);
	const size_t size = strlen(id);
	printf("Hello! I'm student %s.\n", id + (size - 4));
	fclose(fp);
}
