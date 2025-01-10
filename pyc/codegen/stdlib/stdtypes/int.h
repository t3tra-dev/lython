#ifndef INT_H
#define INT_H

#include "../object.h"

// int型の加算関数
PyObject* int_add(PyObject* a, PyObject* b);

// PyIntObjectの宣言
typedef struct {
    PyObject base;
    long value;
} PyIntObject;

// 整数用のメソッドテーブル
extern PyMethodTable int_method_table;

// PyInt_Typeの宣言
extern TypeObject PyInt_Type;

// PyInt_FromLong関数の宣言
PyObject* PyInt_FromLong(long value);

#endif // INT_H
