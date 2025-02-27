#ifndef LYTHON_FUNCTIONS_H
#define LYTHON_FUNCTIONS_H

#include <stddef.h>
#include "../builtin/objects/object.h"
#include "../builtin/objects/unicodeobject.h"
#include "../builtin/objects/intobject.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 基本的な出力関数 */
void print(PyObject *obj);

/* 変換関数 */
PyObject* int2str(int value);
PyObject* str2str(PyObject *str);
PyObject* create_string(const char *str);

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_FUNCTIONS_H */
