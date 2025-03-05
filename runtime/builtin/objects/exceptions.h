/**
 * runtime/builtin/objects/exceptions.h
 * Lython例外オブジェクトの定義
 */

#ifndef LYTHON_EXCEPTIONS_H
#define LYTHON_EXCEPTIONS_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 例外型オブジェクト構造体 */
typedef struct {
    PyObject ob_base;
    PyObject *args;       /* 例外の引数 (タプル) */
    PyObject *msg;        /* 例外メッセージ (文字列) */
} PyBaseExceptionObject;

/* 基本例外型オブジェクト */
extern PyTypeObject PyBaseException_Type;
extern PyTypeObject PyException_Type;

/* 標準例外型オブジェクト */
extern PyObject *PyExc_BaseException;
extern PyObject *PyExc_Exception;
extern PyObject *PyExc_TypeError;
extern PyObject *PyExc_AttributeError;
extern PyObject *PyExc_ValueError;
extern PyObject *PyExc_IndexError;
extern PyObject *PyExc_KeyError;
extern PyObject *PyExc_RuntimeError;
extern PyObject *PyExc_NameError;
extern PyObject *PyExc_NotImplementedError;

/* 例外API */
void PyErr_SetString(PyObject *exception, const char *msg);
void PyErr_SetObject(PyObject *exception, PyObject *value);
PyObject* PyErr_Format(PyObject *exception, const char *format, ...);
PyObject* PyErr_Occurred(void);
void PyErr_Clear(void);
int PyErr_ExceptionMatches(PyObject *exc);
void PyErr_BadInternalCall(void);
PyObject* PyErr_NoMemory(void);

/* 例外作成・操作関数 */
PyObject* PyException_New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
PyObject* PyException_GetMessage(PyObject *exc);
PyObject* PyException_GetArgs(PyObject *exc);
int PyException_SetMessage(PyObject *exc, PyObject *msg);

/* 初期化関数 */
void _PyExc_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_EXCEPTIONS_H */
