/**
 * runtime/builtin/objects/functionobject.h
 * Lython関数オブジェクトの定義
 */

#ifndef LYTHON_FUNCTIONOBJECT_H
#define LYTHON_FUNCTIONOBJECT_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 関数オブジェクト構造体 */
typedef struct {
    PyObject ob_base;
    PyObject *func_name;        /* 関数名 */
    PyObject *func_code;        /* コードオブジェクトまたは関数ポインタ */
    PyObject *func_globals;     /* グローバル名前空間 */
    PyObject *func_defaults;    /* デフォルト引数 (タプル) */
    PyObject *func_doc;         /* ドキュメント文字列 */
    PyObject *func_dict;        /* 関数属性 */
    int func_is_native;         /* ネイティブC関数かどうか */
} PyFunctionObject;

/* ネイティブ関数typedef */
typedef PyObject* (*PyNativeFunction)(PyObject *self, PyObject *args);

/* 関数型オブジェクト */
extern PyTypeObject PyFunction_Type;

/* マクロ */
#define PyFunction_Check(op) PyObject_TypeCheck(op, &PyFunction_Type)
#define PyFunction_GET_CODE(func) (((PyFunctionObject *)(func))->func_code)

/* 関数API */
PyObject* PyFunction_New(PyObject *code, PyObject *globals);
PyObject* PyFunction_GetCode(PyObject *op);
PyObject* PyFunction_GetGlobals(PyObject *op);
PyObject* PyFunction_GetDefaults(PyObject *op);
int PyFunction_SetDefaults(PyObject *op, PyObject *defaults);

/* ネイティブ関数作成 */
PyObject* PyFunction_FromNative(const char *name, PyNativeFunction func, const char *doc);

/* 初期化関数 */
void _PyFunction_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_FUNCTIONOBJECT_H */
