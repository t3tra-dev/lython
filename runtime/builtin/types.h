#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>

// 文字列型
typedef struct String {
    size_t length;
    char* data;
} String;

// intをボックス化した PyInt 型
typedef struct PyInt {
    int value;
} PyInt;

// リスト構造
typedef struct PyList {
    int size;
    int capacity;
    void **items;
} PyList;

// 辞書構造
typedef struct PyDict {
    int size;
    int capacity;
    void **keys;
    void **values;
} PyDict;

#endif // TYPES_H
