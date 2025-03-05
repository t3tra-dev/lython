/**
 * runtime/builtin/objects/weakrefobject.c
 * Lython弱参照オブジェクトの実装（最小実装）
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "weakrefobject.h"

/**
 * オブジェクトに関連付けられた弱参照をクリア
 * 現段階では最小実装として何もしない
 */
void PyObject_ClearWeakRefs(PyObject *object) {
    /* 現段階では実装省略 - 後に完全実装 */
    return;
}

/**
 * 弱参照システムの初期化
 */
void _PyWeakref_Init(void) {
    /* 現段階では実装省略 */
}
