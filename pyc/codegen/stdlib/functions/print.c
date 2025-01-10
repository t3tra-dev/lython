#include "print.h"
#include "../stdtypes/string.h"
#include <stdio.h>

// print関数の実装
int print(PyObject* obj) {
    if (!obj) return -1;
    
    // objを文字列に変換
    PyObject* str_obj = obj->methods->str(obj);
    if (!str_obj) return -1;
    
    // 文字列型チェック
    if (str_obj->ob_type != &PyString_Type) {
        fprintf(stderr, "TypeError: __str__ returned non-string\n");
        return -1;
    }
    
    // 文字列を出力
    PyStringObject* str = (PyStringObject*)str_obj;
    if (!str->value) return -1;
    
    return puts(str->value);
}
