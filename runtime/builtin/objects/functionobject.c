/**
 * runtime/builtin/objects/functionobject.c
 * Lython関数オブジェクトの実装
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "functionobject.h"
#include "unicodeobject.h"
#include "dictobject.h"
#include "exceptions.h"
#include "methodobject.h"

/* 前方宣言 */
static void function_dealloc(PyFunctionObject *op);
static PyObject* function_call(PyObject *func, PyObject *args, PyObject *kwargs);
static PyObject* function_repr(PyObject *op);
static PyObject* function_str(PyObject *op);
static PyObject* function_get(PyObject *func, PyObject *instance, PyObject *owner);
static int function_traverse(PyFunctionObject *op, visitproc visit, void *arg);

/* 関数型オブジェクト定義 */
PyTypeObject PyFunction_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "function",                  /* tp_name */
    sizeof(PyFunctionObject),    /* tp_basicsize */
    0,                           /* tp_itemsize */
    (destructor)function_dealloc, /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    function_repr,               /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    function_call,               /* tp_call */
    function_str,                /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    0,                           /* tp_flags */
    0,                           /* tp_doc */
    (traverseproc)function_traverse, /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    0,                           /* tp_methods */
    0,                           /* tp_members */
    0,                           /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    function_get,                /* tp_descr_get */
};

/**
 * 関数のデストラクタ
 */
static void function_dealloc(PyFunctionObject *op) {
    Py_XDECREF(op->func_name);
    Py_XDECREF(op->func_code);
    Py_XDECREF(op->func_globals);
    Py_XDECREF(op->func_defaults);
    Py_XDECREF(op->func_doc);
    Py_XDECREF(op->func_dict);
    /* メモリはGCが管理しているので明示的解放は不要 */
}

/**
 * 関数呼び出し実装
 */
static PyObject* function_call(PyObject *func, PyObject *args, PyObject *kwargs) {
    PyFunctionObject *f = (PyFunctionObject *)func;
    
    /* ネイティブC関数かどうかチェック */
    if (f->func_is_native) {
        PyNativeFunction native_func = (PyNativeFunction)f->func_code;
        return native_func(NULL, args);
    }
    
    /* LLVM生成関数の呼び出し - ここでは未実装 */
    PyErr_SetString(PyExc_NotImplementedError, 
                   "Python関数呼び出しはまだ実装されていません");
    return NULL;
}

/**
 * 関数の表示
 */
static PyObject* function_repr(PyObject *op) {
    PyFunctionObject *f = (PyFunctionObject *)op;
    return PyUnicode_FromFormat("<function %U>", f->func_name);
}

/**
 * 関数の文字列表示
 */
static PyObject* function_str(PyObject *op) {
    return function_repr(op);
}

/**
 * 関数デスクリプタget - 関数をバインドメソッドに変換
 */
static PyObject* function_get(PyObject *func, PyObject *instance, PyObject *owner) {
    if (instance == NULL) {
        /* クラスからアクセスされた場合は関数そのものを返す */
        Py_INCREF(func);
        return func;
    } else {
        /* インスタンスからアクセスされた場合はバインドメソッドを返す */
        return PyMethod_New(func, instance);
    }
}

/**
 * 関数トラバース - ガベージコレクション用
 */
static int function_traverse(PyFunctionObject *op, visitproc visit, void *arg) {
    Py_VISIT(op->func_name);
    Py_VISIT(op->func_code);
    Py_VISIT(op->func_globals);
    Py_VISIT(op->func_defaults);
    Py_VISIT(op->func_doc);
    Py_VISIT(op->func_dict);
    return 0;
}

/**
 * 新しい関数オブジェクトを作成
 */
PyObject* PyFunction_New(PyObject *code, PyObject *globals) {
    PyFunctionObject *op = (PyFunctionObject *)GC_malloc(sizeof(PyFunctionObject));
    if (op == NULL) {
        return NULL;
    }
    
    Py_TYPE(op) = &PyFunction_Type;
    Py_REFCNT(op) = 1;
    
    op->func_code = Py_NewRef(code);
    op->func_globals = Py_NewRef(globals);
    op->func_name = PyUnicode_FromString("<unnamed>");
    op->func_defaults = NULL;
    op->func_doc = Py_NewRef(Py_None);
    op->func_dict = NULL;
    op->func_is_native = 0;
    
    return (PyObject *)op;
}

/**
 * ネイティブC関数から関数オブジェクトを作成
 */
PyObject* PyFunction_FromNative(const char *name, PyNativeFunction func, const char *doc) {
    PyFunctionObject *op = (PyFunctionObject *)GC_malloc(sizeof(PyFunctionObject));
    if (op == NULL) {
        return NULL;
    }
    
    Py_TYPE(op) = &PyFunction_Type;
    Py_REFCNT(op) = 1;
    
    op->func_name = PyUnicode_FromString(name);
    op->func_code = (PyObject *)func; /* 関数ポインタを直接格納 */
    op->func_globals = NULL;
    op->func_defaults = NULL;
    
    if (doc != NULL) {
        op->func_doc = PyUnicode_FromString(doc);
    } else {
        op->func_doc = Py_NewRef(Py_None);
    }
    
    op->func_dict = NULL;
    op->func_is_native = 1;
    
    return (PyObject *)op;
}

/**
 * 関数からコードオブジェクトを取得
 */
PyObject* PyFunction_GetCode(PyObject *op) {
    if (!PyFunction_Check(op)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    
    return PyFunction_GET_CODE(op);
}

/**
 * 関数からグローバル辞書を取得
 */
PyObject* PyFunction_GetGlobals(PyObject *op) {
    if (!PyFunction_Check(op)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    
    PyObject *globals = ((PyFunctionObject *)op)->func_globals;
    if (globals == NULL) {
        globals = Py_None;
    }
    
    return globals;
}

/**
 * 関数からデフォルト引数を取得
 */
PyObject* PyFunction_GetDefaults(PyObject *op) {
    if (!PyFunction_Check(op)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    
    PyObject *defaults = ((PyFunctionObject *)op)->func_defaults;
    if (defaults == NULL) {
        defaults = Py_None;
    }
    
    return defaults;
}

/**
 * 関数にデフォルト引数を設定
 */
int PyFunction_SetDefaults(PyObject *op, PyObject *defaults) {
    if (!PyFunction_Check(op)) {
        PyErr_BadInternalCall();
        return -1;
    }
    
    PyFunctionObject *func = (PyFunctionObject *)op;
    
    Py_XDECREF(func->func_defaults);
    func->func_defaults = Py_XNewRef(defaults);
    
    return 0;
}

/**
 * 関数システムの初期化
 */
void _PyFunction_Init(void) {
    PyType_Ready(&PyFunction_Type);
}
