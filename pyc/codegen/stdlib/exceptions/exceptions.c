#include "exceptions.h"
#include "../stdtypes/string.h"
#include "../stdtypes/baseexception.h"
#include <stdio.h>
#include <stdlib.h>

// 例外を発生させる関数
void raise_exception(PyBaseExceptionObject* exc) {
    // エラーメッセージを取得して出力 (簡易的な実装)
    PyObject* str_obj = exc->base.methods->str((PyObject*)exc);
    if (str_obj->ob_type != &PyString_Type) {
        fprintf(stderr, "TypeError: __str__ returned non-string\n");
        exit(1);
    }
    PyStringObject* str = (PyStringObject*)str_obj;
    fprintf(stderr, "Exception: %s\n", str->value);
    exit(1);
}
