/**
 * runtime/builtin/objects/dictobject.h
 * Lython辞書オブジェクトの定義
 * CPython 3.xの辞書オブジェクトに対応
 */

#ifndef LYTHON_DICTOBJECT_H
#define LYTHON_DICTOBJECT_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 辞書エントリ構造体 */
typedef struct {
    PyObject *key;
    PyObject *value;
    Py_hash_t hash;  /* キーのハッシュ値のキャッシュ */
} PyDictEntry;

/* 辞書オブジェクトの定義 */
typedef struct {
    PyObject ob_base;
    Py_ssize_t capacity;    /* 確保済みのエントリ数 */
    Py_ssize_t used;        /* 使用中のエントリ数 */
    PyDictEntry *entries;   /* エントリの配列 */
    Py_ssize_t mask;        /* capacity - 1, ハッシュマスク用 */
    Py_hash_t hash;         /* オブジェクトのハッシュ値 */
} PyDictObject;

/* 辞書オブジェクト型の外部参照 */
extern PyTypeObject PyDict_Type;

/* マクロ */
#define PyDict_Check(op) PyObject_TypeCheck(op, &PyDict_Type)
#define PyDict_CheckExact(op) (Py_TYPE(op) == &PyDict_Type)

/* 辞書の初期サイズ */
#define PyDict_MINSIZE 8

/* 辞書作成・操作関数 */
PyObject* PyDict_New(void);
PyObject* PyDict_GetItem(PyObject *dict, PyObject *key);
int PyDict_SetItem(PyObject *dict, PyObject *key, PyObject *value);
int PyDict_DelItem(PyObject *dict, PyObject *key);
void PyDict_Clear(PyObject *dict);
int PyDict_Next(PyObject *dict, Py_ssize_t *pos, PyObject **key, PyObject **value);
PyObject* PyDict_Keys(PyObject *dict);
PyObject* PyDict_Values(PyObject *dict);
PyObject* PyDict_Items(PyObject *dict);
Py_ssize_t PyDict_Size(PyObject *dict);
int PyDict_Update(PyObject *dict, PyObject *other);
int PyDict_Merge(PyObject *dict, PyObject *other, int override);

/* 文字列キーを使った操作 */
PyObject* PyDict_GetItemString(PyObject *dict, const char *key);
int PyDict_SetItemString(PyObject *dict, const char *key, PyObject *value);
int PyDict_DelItemString(PyObject *dict, const char *key);

/* 初期化関数 */
void _PyDict_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_DICTOBJECT_H */
