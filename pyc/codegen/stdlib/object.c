#include "object.h"
#include "stdtypes/string.h"

// メソッドの実装
int default_eq(PyObject* self, PyObject* other) {
    return self == other;
}

PyObject* default_str(PyObject* self) {
    char buf[128];
    snprintf(buf, sizeof(buf), "<object at %p>", (void*)self);
    return PyString_FromString(buf);
}

void default_dealloc(PyObject* self) {
    if (!self) return;
    
    // 参照カウントをデクリメント
    if (--self->ob_refcnt == 0) {
        // 文字列オブジェクトの場合、内部バッファを解放
        if (self->ob_type == &PyString_Type) {
            PyStringObject* str = (PyStringObject*)self;
            if (str->value) {
                free(str->value);
            }
        }
        free(self);
    }
}

// メソッドテーブルの定義
PyMethodTable default_method_table = {
    .eq = default_eq,
    .ne = NULL,
    .lt = NULL,
    .le = NULL,
    .gt = NULL,
    .ge = NULL,
    .add = NULL,
    .sub = NULL,
    .mul = NULL,
    .div = NULL,
    .mod = NULL,
    .neg = NULL,
    .pos = NULL,
    .abs = NULL,
    .iter = NULL,
    .next = NULL,
    .str = default_str,
    .dealloc = default_dealloc
};

// メタ型の定義（型の型）
TypeObject PyType_Type = {
    {NULL, 1, &default_method_table},  // PyObject base
    "type",                            // name
    sizeof(TypeObject),                // basicsize
    0,                                 // itemsize
    NULL,                              // new
    NULL                               // base_type
};

// オブジェクト生成のヘルパー関数
PyObject* PyObject_New(TypeObject* type) {
    PyObject* obj = malloc(type->basicsize);
    if (obj == NULL) return NULL;

    obj->ob_type = type;
    obj->ob_refcnt = 1;
    obj->methods = (type == &PyType_Type || type->base.methods == NULL) ? &default_method_table : type->base.methods;

    return obj;
}
