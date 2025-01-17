#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"

// 文字列(ヒープ領域)作成
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

//文字列(ヒープ領域)を解放
void free_string(String* s) {
    if (!s) return;
    if (s->data) free(s->data);
    free(s);
}

// int2str : int => String*
String* int2str(int value) {
    char buf[32];  // 十分なサイズのバッファ
    sprintf(buf, "%d", value); // 10進数変換
    return create_string(buf);
}

// str2str : String* => String*
String* str2str(String* s) {
    return s;
}

// printの実装
void print(String* s) {
    if (!s) {
        printf("(null)\n");
        return;
    }
    printf("%s\n", s->data);
}
