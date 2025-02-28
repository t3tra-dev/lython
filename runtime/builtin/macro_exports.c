#include <stddef.h>
#include "objects/object.h"

/* 参照カウント操作の実装 */
void Py_INCREF(PyObject *op) {
    if (op != NULL)
        op->ob_refcnt++;
}

void Py_DECREF(PyObject *op) {
    if (op != NULL && --op->ob_refcnt == 0) {
        if (op->ob_type && op->ob_type->tp_dealloc)
            op->ob_type->tp_dealloc(op);
    }
}

void Py_XINCREF(PyObject *op) {
    if (op != NULL)
        Py_INCREF(op);
}

void Py_XDECREF(PyObject *op) {
    if (op != NULL)
        Py_DECREF(op);
}

#define Py_None (&_Py_NoneStruct)
