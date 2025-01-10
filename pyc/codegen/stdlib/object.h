#ifndef OBJECT_H
#define OBJECT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 前方宣言
typedef struct TypeObject TypeObject;
typedef struct PyObject PyObject;
typedef struct PyMethodTable PyMethodTable;

// メソッドテーブルの定義
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

// メソッドの実装
int default_eq(PyObject* self, PyObject* other);
PyObject* default_str(PyObject* self);
void default_dealloc(PyObject* self);
PyObject* int_add(PyObject* self, PyObject* other);

// メソッドテーブルの定義
extern PyMethodTable default_method_table;

// メタ型の定義（型の型）
extern TypeObject PyType_Type;

// オブジェクト生成のヘルパー関数
PyObject* PyObject_New(TypeObject* type);

// PyStringObjectの宣言
typedef struct {
    PyObject base;
    char* value;
    size_t length;
} PyStringObject;

#endif // OBJECT_H
