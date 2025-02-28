/**
 * runtime/builtin/objects/booleanobject.h
 * Lython真偽値オブジェクトの定義
 * CPython 3.xのブールオブジェクトに対応
 */

#ifndef LYTHON_BOOLEANOBJECT_H
#define LYTHON_BOOLEANOBJECT_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 真偽値オブジェクト型の外部参照 */
extern PyTypeObject PyBool_Type;

/* マクロ */
#define PyBool_Check(op) PyObject_TypeCheck(op, &PyBool_Type)

/* bool値オブジェクト作成関数 (object.hで定義) */
/* PyObject* PyBool_FromLong(long v); */

/* 初期化関数 */
void _PyBool_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_BOOLEANOBJECT_H */
