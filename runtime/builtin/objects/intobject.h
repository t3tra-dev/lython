#ifndef LYTHON_INTOBJECT_H
#define LYTHON_INTOBJECT_H

#include <stddef.h>
#include <stdint.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 整数オブジェクトの定義 */
typedef struct {
    PyObject ob_base;
    int32_t ob_ival;  /* 32ビット整数値 */
} PyIntObject;

/* 整数オブジェクト型の外部参照 */
extern PyTypeObject PyInt_Type;

/* マクロ */
#define PyInt_Check(op) PyObject_TypeCheck(op, &PyInt_Type)
#define PyInt_CheckExact(op) (Py_TYPE(op) == &PyInt_Type)

/* 整数操作関数 */
PyObject* PyInt_FromI32(int32_t ival);
int32_t PyInt_AsI32(PyObject *op);

/* 初期化関数 */
void _PyInt_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_INTOBJECT_H */
