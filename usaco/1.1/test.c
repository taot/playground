/*
ID: your_id_here
LANG: C
TASK: test
*/
#include <stdlib.h>
#include <stdio.h>
int main () {
    FILE *fin  = fopen ("test.in", "r");
    FILE *fout = fopen ("test.out", "w");
    int a, b;
    fscanf (fin, "%d %d", &a, &b);	/* the two input integers */
    fprintf (fout, "%d\n", a+b);
    exit (0);
}
