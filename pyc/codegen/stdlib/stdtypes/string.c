#include "string.h"
#include "baseexception.h"
#include <stdlib.h>
#include <string.h>

// 文字列オブジェクト用の__str__メソッド
PyObject* string_str(PyObject* self) {
    return self;  // 文字列オブジェクトは自身を返す
}

// 文字列用のメソッドテーブル
PyMethodTable string_method_table = {
    .eq = default_eq,
    .ne = NULL,
    .lt = NULL,
    .le = NULL,
    .gt = NULL,
    .ge = NULL,
    .add = NULL,
    .sub = NULL,
    .mul = NULL,
    .div = NULL,
    .mod = NULL,
    .neg = NULL,
    .pos = NULL,
    .abs = NULL,
    .iter = NULL,
    .next = NULL,
    .str = string_str,  // 文字列用の__str__メソッドを設定
    .dealloc = default_dealloc
};

// PyString_Typeの定義
TypeObject PyString_Type = {
    {&PyType_Type, 1, &string_method_table},  // PyObject base
    "str",                                    // name
    sizeof(PyStringObject),                   // basicsize
    0,                                        // itemsize
    NULL,                                     // new
    &PyBaseException_Type                     // base_type
};

// 文字列生成関数
PyObject* PyString_FromString(const char* str) {
    if (!str) return NULL;
    
    // メモリ確保
    PyStringObject* obj = (PyStringObject*)PyObject_New(&PyString_Type);
    if (obj == NULL) return NULL;
    
    // 文字列の長さを計算
    size_t len = strlen(str);
    
    // 文字列用のメモリを確保
    obj->value = malloc(len + 1);
    if (obj->value == NULL) {
        free(obj);
        return NULL;
    }
    
    // 文字列をコピー
    strcpy(obj->value, str);
    obj->length = len;
    
    return (PyObject*)obj;
}
