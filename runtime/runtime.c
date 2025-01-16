#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"

String* create_string(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src);
    String* s = (String*)malloc(sizeof(String));
    if (!s) return NULL;
    s->length = len;
    s->data = (char*)malloc(len + 1);
    if (!s->data) {
        free(s);
        return NULL;
    }
    strcpy(s->data, src);
    return s;
}

void free_string(String* s) {
    if (!s) return;
    if (s->data) free(s->data);
    free(s);
}

void print_i32(int val) {
    printf("%d\n", val);
}

void print_string(String* s) {
    if (!s) {
        printf("(null)\n");
        return;
    }
    printf("%s\n", s->data);
}
