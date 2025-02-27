/**
 * runtime/builtin/objects/listobject.h
 * Lythonリストオブジェクトの定義
 * CPython 3.xのリストオブジェクトに対応
 */

#ifndef LYTHON_LISTOBJECT_H
#define LYTHON_LISTOBJECT_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* リストオブジェクトの定義 */
typedef struct {
    PyObject ob_base;
    Py_ssize_t allocated;  /* 確保済みの要素数 */
    PyObject **ob_item;    /* 要素の配列 (item_0, item_1, ...) */
} PyListObject;

/* リストオブジェクト型の外部参照 */
extern PyTypeObject PyList_Type;

/* マクロ */
#define PyList_Check(op) PyObject_TypeCheck(op, &PyList_Type)
#define PyList_CheckExact(op) (Py_TYPE(op) == &PyList_Type)

/* 初期サイズとリサイズ定数 */
#define PyList_MINSIZE 8
#define PyList_MAXFREELIST 80

/* リスト作成・操作関数 */
PyObject* PyList_New(Py_ssize_t size);
Py_ssize_t PyList_Size(PyObject *list);
PyObject* PyList_GetItem(PyObject *list, Py_ssize_t index);
int PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item);
int PyList_Insert(PyObject *list, Py_ssize_t index, PyObject *item);
int PyList_Append(PyObject *list, PyObject *item);
PyObject* PyList_GetSlice(PyObject *list, Py_ssize_t low, Py_ssize_t high);
int PyList_SetSlice(PyObject *list, Py_ssize_t low, Py_ssize_t high, PyObject *itemlist);
int PyList_Sort(PyObject *list);
int PyList_Reverse(PyObject *list);
PyObject* PyList_AsTuple(PyObject *list);

/* 初期化関数 */
void _PyList_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_LISTOBJECT_H */
