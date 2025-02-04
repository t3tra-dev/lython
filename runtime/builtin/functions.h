#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stddef.h>
#include "types.h"

// 文字列操作
String* create_string(const char* src);
void free_string(String* s);
String* int2str(int value);
String* str2str(String* s);

// ハッシュ関数
unsigned int hash_object(void *key);

// リスト操作
PyList* PyList_New(int capacity);
int PyList_Append(PyList *list, void *item);
void* PyList_GetItem(PyList *list, int index);
void PyList_Free(PyList *list);

// 辞書操作
PyDict* PyDict_New(int capacity);
int PyDict_SetItem(PyDict *dict, void *key, void *value);
void* PyDict_GetItem(PyDict *dict, void *key);
void PyDict_Free(PyDict *dict);

// 出力関数
void print(String* s);

#endif // FUNCTIONS_H
