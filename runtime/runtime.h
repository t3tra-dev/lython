#ifndef STATIC_RUNTIME_H
#define STATIC_RUNTIME_H

#include <stddef.h>

typedef struct {
    size_t length;
    char* data;
} String;

// 文字列(ヒープ領域)作成
String* create_string(const char* src);

//文字列(ヒープ領域)を解放
void free_string(String* s);

// int2str : int => String*
String* int2str(int value);

// str2str : String* => String*
String* str2str(String* s);

// printの実装
void print(String* s);

#endif /* STATIC_RUNTIME_H */
