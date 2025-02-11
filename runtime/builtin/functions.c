#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>  // Boehm GC
#include "types.h"
#include "functions.h"

// 文字列(ヒープ領域)作成
String* create_string(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src);
    // GC_malloc を利用
    String* s = (String*)GC_malloc(sizeof(String));
    s->length = len;
    s->data = (char*)GC_malloc(len + 1);
    strcpy(s->data, src);
    return s;
}

// PyInt ボックス化
PyInt* PyInt_FromI32(int value) {
    PyInt* obj = (PyInt*)GC_malloc(sizeof(PyInt));
    obj->value = value;
    return obj;
}

int PyInt_AsI32(PyInt* obj) {
    if (!obj) return 0; // 仮
    return obj->value;
}

// int2str : int => String*
String* int2str(int value) {
    char buf[32];
    sprintf(buf, "%d", value);
    return create_string(buf);
}

// str2str : String* => String*
String* str2str(String* s) {
    return s;
}

/* 仮のハッシュ関数（実際は各オブジェクトの hash 関数に委ねる） */
unsigned int hash_object(void *key) {
    String *str = (String*)key;
    unsigned int hash = 5381;
    for (size_t i = 0; i < str->length; i++) {
        hash = ((hash << 5) + hash) + str->data[i];  // hash * 33 + c
    }
    return hash;
}

/* 新しいリストを作成．初期容量を指定（0 なら既定値） */
PyList* PyList_New(int capacity) {
    if (capacity <= 0) {
        capacity = 8;
    }
    PyList *list = GC_malloc(sizeof(PyList));
    list->size = 0;
    list->capacity = capacity;
    list->items = GC_malloc(sizeof(void*) * capacity);
    return list;
}

/* リストに要素を追加 */
int PyList_Append(PyList *list, void *item) {
    if (list->size >= list->capacity) {
        int new_capacity = (list->capacity * 3) / 2 + 1;
        void **new_items = GC_malloc(sizeof(void*) * new_capacity);
        // 古い要素をコピー (Boehm だと再alloc風にできないのでコピー)
        for (int i = 0; i < list->size; i++) {
            new_items[i] = list->items[i];
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }
    list->items[list->size++] = item;
    return 0;
}

void* PyList_GetItem(PyList *list, int index) {
    if (index < 0 || index >= list->size)
        return NULL;
    return list->items[index];
}

/* 新しい辞書を作成 */
PyDict* PyDict_New(int capacity) {
    if (capacity < 8) capacity = 8;
    PyDict *dict = GC_malloc(sizeof(PyDict));
    dict->size = 0;
    dict->capacity = capacity;
    dict->keys = GC_malloc(sizeof(void*) * capacity);
    dict->values = GC_malloc(sizeof(void*) * capacity);
    for (int i = 0; i < capacity; i++) {
        dict->keys[i] = NULL;
        dict->values[i] = NULL;
    }
    return dict;
}

/* 内部: 再ハッシュ(リサイズ) */
static int PyDict_Resize(PyDict *dict, int new_capacity) {
    void **old_keys = dict->keys;
    void **old_values = dict->values;
    int old_capacity = dict->capacity;

    void **new_keys = GC_malloc(sizeof(void*) * new_capacity);
    void **new_values = GC_malloc(sizeof(void*) * new_capacity);
    for (int i = 0; i < new_capacity; i++) {
        new_keys[i] = NULL;
        new_values[i] = NULL;
    }
    // 全エントリを再配置
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
    dict->keys = new_keys;
    dict->values = new_values;
    dict->capacity = new_capacity;
    return 0;
}

int PyDict_SetItem(PyDict *dict, void *key, void *value) {
    if (dict->size * 2 >= dict->capacity) {
        if (PyDict_Resize(dict, dict->capacity * 2) < 0)
            return -1;
    }
    unsigned int h = hash_object(key);
    int i = h % dict->capacity;
    while (dict->keys[i] != NULL) {
        String *key1 = (String*)dict->keys[i];
        String *key2 = (String*)key;
        if (key1->length == key2->length && 
            strcmp(key1->data, key2->data) == 0) {
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

void* PyDict_GetItem(PyDict *dict, void *key) {
    // キーが文字列リテラルの場合は一時的なString*を作成
    String *temp_key = NULL;
    if (((uintptr_t)key & 1) == 0) {  // ポインタがString*でない場合
        temp_key = create_string((const char*)key);
        key = temp_key;
    }
    
    unsigned int h = hash_object(key);
    int i = h % dict->capacity;
    while (dict->keys[i] != NULL) {
        String *key1 = (String*)dict->keys[i];
        String *key2 = (String*)key;
        if (key1->length == key2->length && 
            strcmp(key1->data, key2->data) == 0) {
            return dict->values[i];
        }
        i = (i + 1) % dict->capacity;
    }
    return NULL;
}

// printの実装
void print(String* s) {
    if (!s) {
        puts("(null)");
        return;
    }
    puts(s->data);
}
