#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "functions.h"

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

/* 仮のハッシュ関数（実際は各オブジェクトの hash 関数に委ねる） */
unsigned int hash_object(void *key) {
    /* ここでは key を整数（ポインタ値）とみなして hash を返す */
    return (unsigned int)((uintptr_t)key);
}

/* 新しいリストを作成．初期容量を指定（0 なら既定値） */
PyList* PyList_New(int capacity) {
    if (capacity <= 0) {
        capacity = 8;  // 小さい固定値
    }
    PyList *list = malloc(sizeof(PyList));
    if (!list) return NULL;
    list->size = 0;
    list->capacity = capacity;
    list->items = malloc(sizeof(void*) * capacity);
    if (!list->items) {
        free(list);
        return NULL;
    }
    return list;
}

/* リストに要素を追加. オーバーアロケーションにより再確保を最小限にする */
int PyList_Append(PyList *list, void *item) {
    if (list->size >= list->capacity) {
        int new_capacity = (list->capacity * 3) / 2 + 1; // 1.5倍+1 の拡大
        void **new_items = realloc(list->items, sizeof(void*) * new_capacity);
        if (!new_items) return -1;
        list->items = new_items;
        list->capacity = new_capacity;
    }
    list->items[list->size++] = item;
    return 0;
}

/* リストから index 番目の要素を返す（存在しなければ NULL） */
void* PyList_GetItem(PyList *list, int index) {
    if (index < 0 || index >= list->size)
        return NULL;
    return list->items[index];
}

/* リスト全体を解放 */
void PyList_Free(PyList *list) {
    if (list) {
        free(list->items);
        free(list);
    }
}

/* 新しい辞書を作成。capacity が小さすぎた場合は既定値に */
PyDict* PyDict_New(int capacity) {
    if (capacity < 8)
        capacity = 8;
    PyDict *dict = malloc(sizeof(PyDict));
    if (!dict) return NULL;
    dict->size = 0;
    dict->capacity = capacity;
    dict->keys = malloc(sizeof(void*) * capacity);
    dict->values = malloc(sizeof(void*) * capacity);
    if (!dict->keys || !dict->values) {
        free(dict->keys);
        free(dict->values);
        free(dict);
        return NULL;
    }
    for (int i = 0; i < capacity; i++) {
        dict->keys[i] = NULL;
        dict->values[i] = NULL;
    }
    return dict;
}

/* 内部: 再ハッシュ（resize）処理 */
static int PyDict_Resize(PyDict *dict, int new_capacity) {
    void **old_keys = dict->keys;
    void **old_values = dict->values;
    int old_capacity = dict->capacity;
    
    void **new_keys = malloc(sizeof(void*) * new_capacity);
    void **new_values = malloc(sizeof(void*) * new_capacity);
    if (!new_keys || !new_values) {
        free(new_keys);
        free(new_values);
        return -1;
    }
    for (int i = 0; i < new_capacity; i++) {
        new_keys[i] = NULL;
        new_values[i] = NULL;
    }
    // 再配置: すべての既存エントリを新テーブルへ再挿入
    for (int i = 0; i < old_capacity; i++) {
        if (old_keys[i] != NULL) {
            unsigned int h = hash_object(old_keys[i]);
            int j = h % new_capacity;
            while (new_keys[j] != NULL) {
                j = (j + 1) % new_capacity;
            }
            new_keys[j] = old_keys[i];
            new_values[j] = old_values[i];
        }
    }
    free(old_keys);
    free(old_values);
    dict->keys = new_keys;
    dict->values = new_values;
    dict->capacity = new_capacity;
    return 0;
}

/* 辞書にキー, 値ペアを設定(既存キーの場合は上書き) */
int PyDict_SetItem(PyDict *dict, void *key, void *value) {
    if (dict->size * 2 >= dict->capacity) {  // load factor 0.5
        if (PyDict_Resize(dict, dict->capacity * 2) < 0)
            return -1;
    }
    unsigned int h = hash_object(key);
    int i = h % dict->capacity;
    while (dict->keys[i] != NULL) {
        if (dict->keys[i] == key) {  // 単純なポインタ比較
            dict->values[i] = value;
            return 0;
        }
        i = (i + 1) % dict->capacity;
    }
    dict->keys[i] = key;
    dict->values[i] = value;
    dict->size++;
    return 0;
}

/* 辞書からキーに対応する値を取得（なければ NULL） */
void* PyDict_GetItem(PyDict *dict, void *key) {
    unsigned int h = hash_object(key);
    int i = h % dict->capacity;
    while (dict->keys[i] != NULL) {
        if (dict->keys[i] == key)
            return dict->values[i];
        i = (i + 1) % dict->capacity;
    }
    return NULL;
}

/* 辞書全体を解放 */
void PyDict_Free(PyDict *dict) {
    if (dict) {
        free(dict->keys);
        free(dict->values);
        free(dict);
    }
}

// printの実装
void print(String* s) {
    if (!s) {
        printf("(null)\n");
        return;
    }
    printf("%s\n", s->data);
}
