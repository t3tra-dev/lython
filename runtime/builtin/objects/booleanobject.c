/**
 * runtime/builtin/objects/booleanobject.c
 * Lython真偽値オブジェクトの実装
 */

#include <stdio.h>
#include <gc.h>
#include "object.h"
#include "booleanobject.h"
#include "unicodeobject.h"  /* 文字列表現のため */

/* ブール型オブジェクト定義の前方宣言 */
static PyObject* bool_repr(PyObject *op);
static PyObject* bool_str(PyObject *op);
static Py_hash_t bool_hash(PyObject *op);

/* ブール型オブジェクト定義 */
PyTypeObject PyBool_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "bool",                      /* tp_name */
    sizeof(PyObject),            /* tp_basicsize */
    0,                           /* tp_itemsize */
    0,                           /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    bool_repr,                   /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    bool_hash,                   /* tp_hash */
    0,                           /* tp_call */
    bool_str,                    /* tp_str */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    0,                           /* tp_richcompare */
    0,                           /* tp_dict */
    0,                           /* tp_base */
    0,                           /* tp_bases */
    0,                           /* tp_new */
    0,                           /* tp_init */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
};

/**
 * ブール値の文字列表現を生成
 */
static PyObject* bool_repr(PyObject *op) {
    if (op == Py_True) {
        return PyUnicode_FromString("True");
    } else {
        return PyUnicode_FromString("False");
    }
}

/**
 * ブール値の文字列表現を生成 (str()と同じ)
 */
static PyObject* bool_str(PyObject *op) {
    return bool_repr(op);
}

/**
 * ブール値のハッシュ値を計算
 */
static Py_hash_t bool_hash(PyObject *op) {
    if (op == Py_True) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * ブール値サブシステムの初期化
 */
void _PyBool_Init(void) {
    /* 型オブジェクトの初期化 */
    PyType_Ready(&PyBool_Type);
    
    /* True/Falseオブジェクトの型を設定 */
    Py_TYPE(Py_True) = &PyBool_Type;
    Py_TYPE(Py_False) = &PyBool_Type;
}
