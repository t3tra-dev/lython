/**
 * runtime/builtin/objects/classobject.h
 * Lythonクラスとインスタンスオブジェクトの定義
 */

#ifndef LYTHON_CLASSOBJECT_H
#define LYTHON_CLASSOBJECT_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* クラスオブジェクト構造体 */
typedef struct {
    PyObject ob_base;
    PyObject *cl_name;      /* クラス名 (文字列) */
    PyObject *cl_dict;      /* クラス辞書 (属性とメソッド) */
    PyObject *cl_bases;     /* 基底クラス (リスト) */
} PyClassObject;

/* インスタンスオブジェクト構造体 */
typedef struct {
    PyObject ob_base;
    PyObject *in_class;     /* クラスオブジェクト */
    PyObject *in_dict;      /* インスタンス辞書 (属性) */
} PyInstanceObject;

/* クラスとインスタンスの型オブジェクト */
extern PyTypeObject PyClass_Type;
extern PyTypeObject PyInstance_Type;

/* マクロ */
#define PyClass_Check(op) PyObject_TypeCheck(op, &PyClass_Type)
#define PyInstance_Check(op) PyObject_TypeCheck(op, &PyInstance_Type)

/* クラスAPI */
PyObject* PyClass_New(PyObject *name, PyObject *bases, PyObject *dict);
PyObject* PyInstance_New(PyObject *class, PyObject *args, PyObject *kwargs);
PyObject* PyInstance_NewRaw(PyObject *class);

/* 初期化関数 */
void _PyClass_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_CLASSOBJECT_H */
