#ifndef BASEEXCEPTION_H
#define BASEEXCEPTION_H

#include "../object.h"

// PyBaseExceptionObjectの宣言
typedef struct {
    PyObject base;
    PyObject* args;
} PyBaseExceptionObject;

// BaseException用のメソッドテーブル
extern PyMethodTable baseexception_method_table;

// PyBaseException_Typeの宣言
extern TypeObject PyBaseException_Type;

// PyBaseException_New関数の宣言
PyObject* PyBaseException_New(TypeObject* type, PyObject* args, PyObject* kwds);

#endif // BASEEXCEPTION_H
