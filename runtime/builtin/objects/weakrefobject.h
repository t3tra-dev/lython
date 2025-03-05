/**
 * runtime/builtin/objects/weakrefobject.h
 * Lython弱参照オブジェクトの定義（最小実装）
 */

#ifndef LYTHON_WEAKREFOBJECT_H
#define LYTHON_WEAKREFOBJECT_H

#include <stddef.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 弱参照リスト型（実際の詳細は省略） */
typedef struct _PyWeakrefBase PyWeakReference;

/* 弱参照API */
void PyObject_ClearWeakRefs(PyObject *object);

/* 初期化関数 */
void _PyWeakref_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_WEAKREFOBJECT_H */
