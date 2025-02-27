/**
 * runtime/builtin/objects/unicodeobject.h
 * Lython Unicode文字列オブジェクトの定義
 * CPython 3.xのUnicodeオブジェクトに対応
 */

#ifndef LYTHON_UNICODEOBJECT_H
#define LYTHON_UNICODEOBJECT_H

#include <stddef.h>
#include <stdint.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Unicode文字列オブジェクト */
typedef struct {
    PyObject ob_base;    /* PyObject共通ヘッダ */
    Py_ssize_t length;   /* 文字数 (コードポイント数) */
    Py_hash_t hash;      /* ハッシュキャッシュ (-1は未計算) */
    int kind;            /* 文字列表現の種類 (UTF-8のみサポート) */
    char *str;           /* 実際の文字列データ (NUL終端) */
} PyUnicodeObject;

/* Unicode種別定数 */
#define PyUnicode_WCHAR_KIND 0  /* UTF-8のみサポートするので種別は1つだけ */

/* Unicodeオブジェクト型の外部参照 */
extern PyTypeObject PyUnicode_Type;

/* マクロ */
#define PyUnicode_Check(op) PyObject_TypeCheck(op, &PyUnicode_Type)
#define PyUnicode_CheckExact(op) (Py_TYPE(op) == &PyUnicode_Type)

/* Unicode文字列操作関数 */
PyObject* PyUnicode_FromString(const char *u);
PyObject* PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size);
PyObject* PyUnicode_FromFormat(const char *format, ...);
PyObject* PyUnicode_Concat(PyObject *left, PyObject *right);
PyObject* PyUnicode_FromWideChar(const wchar_t *w, Py_ssize_t size);

/* 文字列の情報取得 */
Py_ssize_t PyUnicode_GetLength(PyObject *unicode);
const char* PyUnicode_AsUTF8(PyObject *unicode);
const char* PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size);

/* 比較操作 */
int PyUnicode_Compare(PyObject *left, PyObject *right);
int PyUnicode_CompareWithASCIIString(PyObject *left, const char *right);

/* ハッシュ計算 */
Py_hash_t PyUnicode_Hash(PyObject *unicode);

/* 初期化関数 */
void _PyUnicode_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_UNICODEOBJECT_H */
