#include "baseexception.h"
#include <stdlib.h>

// BaseException用のメソッドテーブル
PyMethodTable baseexception_method_table = {
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
    .str = default_str,
    .dealloc = default_dealloc
};

// PyBaseException_Typeの定義
TypeObject PyBaseException_Type = {
    {&PyType_Type, 1, &baseexception_method_table},  // PyObject base
    "BaseException",                                // name
    sizeof(PyBaseExceptionObject),                  // basicsize
    0,                                              // itemsize
    NULL,                                           // new
    NULL                                            // base_type
};

// PyBaseException_New関数の実装 (簡易版)
PyObject* PyBaseException_New(TypeObject* type, PyObject* args, PyObject* kwds) {
    PyBaseExceptionObject* obj = (PyBaseExceptionObject*)PyObject_New(type);
    if (obj == NULL) return NULL;

    obj->args = args;
    return (PyObject*)obj;
}
