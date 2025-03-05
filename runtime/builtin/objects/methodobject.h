/**
 * runtime/builtin/objects/methodobject.h
 * Lythonメソッドオブジェクトの定義
 */

#ifndef LYTHON_METHODOBJECT_H
#define LYTHON_METHODOBJECT_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* メソッドオブジェクト構造体 */
typedef struct {
    PyObject ob_base;
    PyObject *im_func;          /* 関数オブジェクト */
    PyObject *im_self;          /* バインドされているインスタンス */
    PyObject *im_weakreflist;   /* 弱参照リスト */
    void *vectorcall;           /* 高速呼び出し関数 */
} PyMethodObject;

/* メソッド型オブジェクト */
extern PyTypeObject PyMethod_Type;

/* マクロ */
#define PyMethod_Check(op) PyObject_TypeCheck(op, &PyMethod_Type)
#define PyMethod_GET_FUNCTION(m) (((PyMethodObject *)(m))->im_func)
#define PyMethod_GET_SELF(m) (((PyMethodObject *)(m))->im_self)

/* メソッドAPI */
PyObject* PyMethod_New(PyObject *func, PyObject *self);
PyObject* PyMethod_Function(PyObject *meth);
PyObject* PyMethod_Self(PyObject *meth);

/* 初期化関数 */
void _PyMethod_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_METHODOBJECT_H */
