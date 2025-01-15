#include "print.h"
#include <stdio.h>

int print(const char* str) {
    if (str == NULL) {
        fprintf(stderr, "Error: NULL string\n");
        return -1;
    }
    return puts(str);
}
