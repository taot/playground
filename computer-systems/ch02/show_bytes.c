#include <stdio.h>

typedef unsigned char *byte_pointer;

void show_bytes(byte_pointer start, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) {
        printf(" %.2x", start[i]);
    }
    printf("\n");
}

void show_int(int x) {
    show_bytes((byte_pointer) &x, sizeof(int));
}

void show_short(short x) {
    show_bytes((byte_pointer) &x, sizeof(short));
}

void show_ushort(unsigned short x) {
    show_bytes((byte_pointer) &x, sizeof(unsigned short));
}

void show_float(float x) {
    show_bytes((byte_pointer) &x, sizeof(float));
}

void show_pointer(void *x) {
    show_bytes((byte_pointer) &x, sizeof(void *));
}

int main(int argc, char *argv[]) {
    // show_int(100);
    // show_float(5.8);
    // int x = 10;
    // show_pointer(&x);

    show_short(-12345);
    show_ushort(53191);

    return 0;
}
