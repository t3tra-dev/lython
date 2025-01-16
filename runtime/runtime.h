#ifndef STATIC_RUNTIME_H
#define STATIC_RUNTIME_H

#include <stddef.h>

typedef struct {
    size_t length;
    char*  data;
} String;

// 文字列生成
String* create_string(const char* src);

// 文字列破棄
void free_string(String* s);

// 整数出力
void print_i32(int val);

// 文字列出力
void print_string(String* s);

#endif /* STATIC_RUNTIME_H */
