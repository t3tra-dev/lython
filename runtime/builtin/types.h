#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>

typedef struct String {
    size_t length;
    char* data;
} String;

typedef struct PyList {
    int size;
    int capacity;
    void **items;
} PyList;

typedef struct PyDict {
    int size;
    int capacity;
    void **keys;
    void **values;
} PyDict;

#endif // TYPES_H
