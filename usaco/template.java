/*
ID: libra_k1
LANG: C
TASK: template
*/

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

char* test_name = "template";

char in_file[50];
char out_file[50];
FILE *in, *out;

int N;
int M;

void create_filenames() {
    strcpy(in_file, test_name);
    strcat(in_file, ".in");
    strcpy(out_file, test_name);
    strcat(out_file, ".out");
}

void main(void) {

    create_filenames();
	in = fopen(in_file, "r");
	out = fopen(out_file, "w");

    fscanf(in, "%d", &N);
    fscanf(in, "%d", &M);

    fprintf(out, "NONE\n");

	exit(0);
}
