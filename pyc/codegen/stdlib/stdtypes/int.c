#include "int.h"
#include "baseexception.h"
#include <stdlib.h>

// int型の加算関数
PyObject* int_add(PyObject* a, PyObject* b) {
    PyIntObject* int_a = (PyIntObject*)a;
    PyIntObject* int_b = (PyIntObject*)b;
    return PyInt_FromLong(int_a->value + int_b->value);
}

// int型用のメソッドテーブル
PyMethodTable int_method_table = {
    .eq = default_eq,
    .ne = NULL,
    .lt = NULL,
    .le = NULL,
    .gt = NULL,
    .ge = NULL,
    .add = int_add,
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

// Int型の型オブジェクトの定義
TypeObject PyInt_Type = {
    {&PyType_Type, 1, &int_method_table},  // PyObject base
    "int",                                  // name
    sizeof(PyIntObject),                    // basicsize
    0,                                      // itemsize
    NULL,                                   // new
    &PyBaseException_Type                   // base_type
};

// Int型のオブジェクト生成関数
PyObject* PyInt_FromLong(long value) {
    PyIntObject* obj = (PyIntObject*)PyObject_New(&PyInt_Type);
    if (obj == NULL) return NULL;

    obj->value = value;
    return (PyObject*)obj;
}
