#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 前方宣言
typedef struct TypeObject TypeObject;
typedef struct PyObject PyObject;
typedef struct PyMethodTable PyMethodTable;
PyObject* PyString_FromString(const char* str);
PyObject* PyInt_FromLong(long value);
extern TypeObject PyInt_Type;  // PyInt_Typeの外部宣言
extern TypeObject PyString_Type;  // PyString_Typeの外部宣言

// すべてのPythonオブジェクトのメソッドテーブル
struct PyMethodTable {
    // 基本的な比較演算
    int (*eq)(PyObject*, PyObject*);         // __eq__
    int (*ne)(PyObject*, PyObject*);         // __ne__
    int (*lt)(PyObject*, PyObject*);         // __lt__
    int (*le)(PyObject*, PyObject*);         // __le__
    int (*gt)(PyObject*, PyObject*);         // __gt__
    int (*ge)(PyObject*, PyObject*);         // __ge__

    // 数値演算
    PyObject* (*add)(PyObject*, PyObject*);  // __add__
    PyObject* (*sub)(PyObject*, PyObject*);  // __sub__
    PyObject* (*mul)(PyObject*, PyObject*);  // __mul__
    PyObject* (*div)(PyObject*, PyObject*);  // __div__
    PyObject* (*mod)(PyObject*, PyObject*);  // __mod__
    
    // 右辺演算
    PyObject* (*radd)(PyObject*, PyObject*); // __radd__
    PyObject* (*rsub)(PyObject*, PyObject*); // __rsub__
    PyObject* (*rmul)(PyObject*, PyObject*); // __rmul__
    PyObject* (*rdiv)(PyObject*, PyObject*); // __rdiv__
    PyObject* (*rmod)(PyObject*, PyObject*); // __rmod__

    // 単項演算
    PyObject* (*neg)(PyObject*);             // __neg__
    PyObject* (*pos)(PyObject*);             // __pos__
    PyObject* (*abs)(PyObject*);             // __abs__
    
    // 型変換
    PyObject* (*str)(PyObject*);             // __str__
    PyObject* (*repr)(PyObject*);            // __repr__
    PyObject* (*to_int)(PyObject*);          // __int__
    PyObject* (*to_float)(PyObject*);        // __float__
    PyObject* (*bool)(PyObject*);            // __bool__
    
    // メモリ管理
    void (*dealloc)(PyObject*);              // __del__
    PyObject* (*copy)(PyObject*);            // __copy__

    // 属性アクセス
    PyObject* (*getattr)(PyObject*, char*);  // __getattr__
    int (*setattr)(PyObject*, char*, PyObject*); // __setattr__
    
    // イテレーション
    PyObject* (*iter)(PyObject*);            // __iter__
    PyObject* (*next)(PyObject*);            // __next__
    
    // コンテキストマネージャ
    PyObject* (*enter)(PyObject*);           // __enter__
    PyObject* (*exit)(PyObject*, PyObject*, PyObject*, PyObject*); // __exit__
};

typedef struct PyMethodTable PyMethodTable;

// 基底のPyObject構造体
struct PyObject {
    TypeObject* ob_type;
    size_t ob_refcnt;
    PyMethodTable* methods;
};

// 型オブジェクト
struct TypeObject {
    PyObject base;
    const char* name;
    size_t basicsize;
    size_t itemsize;
    PyObject* (*new)(TypeObject*);
    TypeObject* base_type;  // 親クラス
};

// Int型の実装
typedef struct {
    PyObject base;
    long value;
} PyIntObject;

// 文字列型の実装
typedef struct {
    PyObject base;
    char* value;
    size_t length;
} PyStringObject;

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
    if (--self->ob_refcnt == 0) {
        free(self);
    }
}

PyObject* int_add(PyObject* self, PyObject* other) {
    // 引数の型チェック
    if (self->ob_type != &PyInt_Type || other->ob_type != &PyInt_Type) {
        // 型が異なる場合はエラー処理を行うべきだが、ここではNULLを返す
        return NULL;
    }

    // 整数値を取得
    PyIntObject* int_self = (PyIntObject*)self;
    PyIntObject* int_other = (PyIntObject*)other;

    // オーバーフローチェックは省略
    long result = int_self->value + int_other->value;

    // 新しいPyIntObjectを作成して返す
    return PyInt_FromLong(result);
    return NULL;
}

// 文字列オブジェクト用の__str__メソッド
PyObject* string_str(PyObject* self) {
    return self;  // 文字列オブジェクトは自身を返す
}

// メソッドテーブルの定義
PyMethodTable default_method_table = {
    .eq = default_eq,      // eq は正しい型
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

// int型用のメソッドテーブル
PyMethodTable int_method_table = {
    .eq = default_eq,
    .ne = NULL,
    .lt = NULL,
    .le = NULL,
    .gt = NULL,
    .ge = NULL,
    .add = int_add,
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

// 文字列用のメソッドテーブル
PyMethodTable string_method_table = {
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
    .str = string_str,  // 文字列用の__str__メソッドを設定
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

// Int型の型オブジェクトの定義
TypeObject PyInt_Type = {
    {&PyType_Type, 1, &int_method_table},  // PyObject base
    "int",                                  // name
    sizeof(PyIntObject),                    // basicsize
    0,                                      // itemsize
    NULL,                                   // new
    NULL                                    // base_type
};

// PyString_Typeの定義
TypeObject PyString_Type = {
    {&PyType_Type, 1, &string_method_table},  // PyObject base
    "str",                                    // name
    sizeof(PyStringObject),                   // basicsize
    0,                                        // itemsize
    NULL,                                     // new
    NULL                                      // base_type
};

// オブジェクト生成のヘルパー関数
PyObject* PyObject_New(TypeObject* type) {
    PyObject* obj = malloc(type->basicsize);
    if (obj == NULL) return NULL;
    
    obj->ob_type = type;
    obj->ob_refcnt = 1;
    obj->methods = type == &PyType_Type ? &default_method_table : type->base.methods;
    
    return obj;
}

// Int型のオブジェクト生成関数
PyObject* PyInt_FromLong(long value) {
    PyIntObject* obj = (PyIntObject*)PyObject_New(&PyInt_Type);
    if (obj == NULL) return NULL;
    
    obj->value = value;
    return (PyObject*)obj;
}

// 文字列生成関数
PyObject* PyString_FromString(const char* str) {
    PyStringObject* obj = (PyStringObject*)PyObject_New(&PyString_Type);
    if (obj == NULL) return NULL;
    
    size_t len = strlen(str);
    obj->value = malloc(len + 1);
    if (obj->value == NULL) {
        free(obj);
        return NULL;
    }
    
    strcpy(obj->value, str);
    obj->length = len;
    return (PyObject*)obj;
    return NULL;
}

