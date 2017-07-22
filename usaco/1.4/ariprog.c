/*
ID: libra_k1
LANG: C
TASK: ariprog
*/

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

char* test_name = "ariprog";

char in_file[50];
char out_file[50];
FILE *in, *out;

int N;
int M;

int count = 0;

void create_filenames() {
    strcpy(in_file, test_name);
    strcat(in_file, ".in");
    strcpy(out_file, test_name);
    strcat(out_file, ".out");
}

int min(int a, int b) {
    return a < b ? a : b;
}

int is_bisquare(int n) {
    int ceil = min(M, (int) sqrt(n));
    for (int i = ceil; i >= 0; i--) {
        int x = (int) sqrt(n - i * i);
        if (x > M) {
            break;
        }
        if (x * x + i * i == n) {
            return 1;
        }
    }
    return 0;
}

int is_bisquare_prog(int a, int b) {
    for (int i = 0; i < N; i++) {
        if (! is_bisquare(a + b * i)) {
            return 0;
        }
    }
    return 1;
}

void search() {
    int upper = 2 * M * M;
    for (int b = 1; b <= upper; b++) {
        for (int a = 0; a <= upper - (N - 1) * b; a++) {
            if (is_bisquare_prog(a, b)) {
                fprintf(out, "%d %d\n", a, b);
                count++;
            }
        }
    }
}

void main(void) {

    create_filenames();
	in = fopen(in_file, "r");
	out = fopen(out_file, "w");

    fscanf(in, "%d", &N);
    fscanf(in, "%d", &M);

    search();
    if (count == 0) {
        fprintf(out, "NONE\n");
    }

	exit(0);
}
