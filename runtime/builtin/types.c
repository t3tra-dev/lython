#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 文字列型 */
typedef struct {
    size_t length;
    char* data;
} String;

/* ===== PyList の実装 ===== */

typedef struct {
    int size;         // 要素数
    int capacity;     // 確保済み容量
    void **items;     // 要素（ポインタ）の配列
} PyList;

/* ===== PyDict の実装 ===== */

typedef struct {
    int size;         // 現在の要素数
    int capacity;     // ハッシュテーブル全体のサイズ
    void **keys;      // キーの配列（NULL は未使用スロット）
    void **values;    // 値の配列
} PyDict;
