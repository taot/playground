#include <stdio.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    short int v = -12345;
    unsigned short uv = (unsigned short) v;
    printf("v = %d, uv = %u\n", v, uv);

    // unsigned length = 0u;
    // unsigned l2 = length - 1;
    // printf("%u\n", l2);

    short x = SHRT_MIN;
    short x1 = -x;
    printf("%d\n", x1);

    return 0;
}
