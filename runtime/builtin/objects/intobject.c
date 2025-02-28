#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "intobject.h"
#include "unicodeobject.h"  /* 文字列表現のため */

/* 整数型オブジェクト定義の前方宣言 */
static PyObject* int_repr(PyObject *op);
static PyObject* int_str(PyObject *op);
static Py_hash_t int_hash(PyObject *op);
static PyObject* int_richcompare(PyObject *a, PyObject *b, int op);
static void int_dealloc(PyObject *op);

/* 整数型のシーケンスメソッド */
static PyNumberMethods int_as_number = {
    0,                          /* nb_add */
    0,                          /* nb_subtract */
    0,                          /* nb_multiply */
    0,                          /* nb_remainder */
    0,                          /* nb_divmod */
    0,                          /* nb_power */
    0,                          /* nb_negative */
    0,                          /* nb_positive */
    0,                          /* nb_absolute */
};

/* 整数型オブジェクト定義 */
PyTypeObject PyInt_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "int",                       /* tp_name */
    sizeof(PyIntObject),         /* tp_basicsize */
    0,                           /* tp_itemsize */
    int_dealloc,                 /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    int_repr,                    /* tp_repr */
    &int_as_number,              /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    int_hash,                    /* tp_hash */
    0,                           /* tp_call */
    int_str,                     /* tp_str */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    int_richcompare,             /* tp_richcompare */
    0,                           /* tp_dict */
    0,                           /* tp_base */
    0,                           /* tp_bases */
    0,                           /* tp_new */
    0,                           /* tp_init */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
};

/**
 * 整数オブジェクトのデストラクタ
 * (GCにより実際にはほとんど使われない)
 */
static void int_dealloc(PyObject *op) {
    /* 型を取り除いて循環参照を防ぐ */
    Py_TYPE(op) = NULL;
}

/**
 * 整数値の文字列表現を生成
 */
static PyObject* int_repr(PyObject *op) {
    PyIntObject *io = (PyIntObject *)op;
    char buffer[32];
    
    /* 整数値を文字列に変換 */
    snprintf(buffer, sizeof(buffer), "%d", io->ob_ival);
    
    /* 新しいUnicode文字列オブジェクトを作成 */
    return PyUnicode_FromString(buffer);
}

/**
 * 整数値の文字列表現を生成 (str()と同じ)
 */
static PyObject* int_str(PyObject *op) {
    return int_repr(op);
}

/**
 * 整数値のハッシュ値を計算
 */
static Py_hash_t int_hash(PyObject *op) {
    PyIntObject *io = (PyIntObject *)op;
    
    /* 整数値をそのままハッシュとして使用 */
    return (Py_hash_t)io->ob_ival;
}

/**
 * 整数値の比較
 */
static PyObject* int_richcompare(PyObject *a, PyObject *b, int op) {
    /* bが整数でなければNoneを返す（Python側で処理させる） */
    if (!PyInt_Check(b)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    PyIntObject *ia = (PyIntObject *)a;
    PyIntObject *ib = (PyIntObject *)b;
    int cmp;
    
    /* 値を比較 */
    if (ia->ob_ival < ib->ob_ival)
        cmp = -1;
    else if (ia->ob_ival > ib->ob_ival)
        cmp = 1;
    else
        cmp = 0;
    
    /* 比較演算子ごとに結果を評価 */
    int result;
    switch (op) {
        case Py_LT: result = cmp < 0; break;
        case Py_LE: result = cmp <= 0; break;
        case Py_EQ: result = cmp == 0; break;
        case Py_NE: result = cmp != 0; break;
        case Py_GT: result = cmp > 0; break;
        case Py_GE: result = cmp >= 0; break;
        default: return NULL; /* 不正な演算子 */
    }
    
    /* Bool値を返す */
    if (result) {
        Py_INCREF(Py_True);
        return Py_True;
    } else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

/**
 * 新しい整数オブジェクトを作成
 */
PyObject* PyInt_FromI32(int32_t ival) {
    PyIntObject *io = (PyIntObject *)GC_malloc(sizeof(PyIntObject));
    if (io == NULL) {
        return NULL;
    }
    
    /* 初期化 */
    Py_TYPE(io) = &PyInt_Type;
    Py_REFCNT(io) = 1;
    io->ob_ival = ival;
    
    return (PyObject *)io;
}

/**
 * 整数オブジェクトから符号付き32ビット整数値を取得
 */
int32_t PyInt_AsI32(PyObject *op) {
    if (op == NULL || !PyInt_Check(op)) {
        return -1;  /* エラー */
    }
    
    return ((PyIntObject *)op)->ob_ival;
}

/**
 * 整数サブシステムの初期化
 */
void _PyInt_Init(void) {
    /* 型オブジェクトの初期化 */
    PyType_Ready(&PyInt_Type);
}
