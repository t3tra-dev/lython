/**
 * runtime/builtin/objects/exceptions.c
 * Lython例外オブジェクトの実装
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "exceptions.h"
#include "unicodeobject.h"
#include "listobject.h"
#include "dictobject.h"

/* スレッドごとの例外状態 (簡略化のため単一グローバル変数) */
static PyObject *_current_exception = NULL;

/* 前方宣言 */
static PyObject* exception_repr(PyObject *exc);
static PyObject* exception_str(PyObject *exc);
static void exception_dealloc(PyBaseExceptionObject *exc);
static int exception_traverse(PyBaseExceptionObject *exc, visitproc visit, void *arg);

/* ベース例外型 */
PyTypeObject PyBaseException_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "BaseException",             /* tp_name */
    sizeof(PyBaseExceptionObject), /* tp_basicsize */
    0,                           /* tp_itemsize */
    (destructor)exception_dealloc, /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    exception_repr,              /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    0,                           /* tp_call */
    exception_str,               /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    0,                           /* tp_flags */
    0,                           /* tp_doc */
    (traverseproc)exception_traverse, /* tp_traverse */
};

/* 標準例外型 */
PyTypeObject PyException_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "Exception",                 /* tp_name */
    sizeof(PyBaseExceptionObject), /* tp_basicsize */
    0,                           /* tp_itemsize */
    0,                           /* tp_dealloc - 継承 */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_repr - 継承 */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    0,                           /* tp_call */
    0,                           /* tp_str - 継承 */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    0,                           /* tp_flags */
    0,                           /* tp_doc */
    0,                           /* tp_traverse - 継承 */
};

/* グローバル例外オブジェクト */
PyObject *PyExc_BaseException;
PyObject *PyExc_Exception;
PyObject *PyExc_TypeError;
PyObject *PyExc_AttributeError;
PyObject *PyExc_ValueError;
PyObject *PyExc_IndexError;
PyObject *PyExc_KeyError;
PyObject *PyExc_RuntimeError;
PyObject *PyExc_NameError;
PyObject *PyExc_NotImplementedError;

/* 例外型辞書 */
static PyObject *_exception_types = NULL;

/**
 * 例外デストラクタ
 */
static void exception_dealloc(PyBaseExceptionObject *exc) {
    Py_XDECREF(exc->args);
    Py_XDECREF(exc->msg);
    /* メモリはGCが管理しているので明示的解放は不要 */
}

/**
 * 例外トラバース - ガベージコレクション用
 */
static int exception_traverse(PyBaseExceptionObject *exc, visitproc visit, void *arg) {
    Py_VISIT(exc->args);
    Py_VISIT(exc->msg);
    return 0;
}

/**
 * 例外の表示
 */
static PyObject* exception_repr(PyObject *obj) {
    PyBaseExceptionObject *exc = (PyBaseExceptionObject *)obj;
    PyObject *msg = exc->msg;
    
    if (msg == NULL || msg == Py_None) {
        return PyUnicode_FromFormat("<%s>", Py_TYPE(obj)->tp_name);
    } else {
        return PyUnicode_FromFormat("<%s: %S>", Py_TYPE(obj)->tp_name, msg);
    }
}

/**
 * 例外の文字列表示
 */
static PyObject* exception_str(PyObject *obj) {
    PyBaseExceptionObject *exc = (PyBaseExceptionObject *)obj;
    
    if (exc->msg != NULL && exc->msg != Py_None) {
        Py_INCREF(exc->msg);
        return exc->msg;
    }
    
    return PyUnicode_FromString("");
}

/**
 * 新しい例外型を作成
 */
static PyObject* create_exception_type(const char *name, PyTypeObject *base) {
    PyTypeObject *type = (PyTypeObject *)GC_malloc(sizeof(PyTypeObject));
    if (type == NULL) {
        return NULL;
    }
    
    /* ベース型のコピーを作成 */
    memcpy(type, base, sizeof(PyTypeObject));
    
    /* 型名を設定 */
    type->tp_name = name;
    
    /* ベース型を設定 */
    type->tp_base = base;
    
    /* 型初期化 */
    PyType_Ready(type);
    
    /* 例外インスタンスを作成 */
    PyObject *instance = PyException_New(type, NULL, NULL);
    if (instance == NULL) {
        return NULL;
    }
    
    /* 例外型辞書に登録 */
    if (_exception_types != NULL) {
        PyObject *name_obj = PyUnicode_FromString(name);
        if (name_obj != NULL) {
            PyDict_SetItem(_exception_types, name_obj, (PyObject *)type);
            Py_DECREF(name_obj);
        }
    }
    
    return instance;
}

/**
 * 新しい例外インスタンスを作成
 */
PyObject* PyException_New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyBaseExceptionObject *exc = (PyBaseExceptionObject *)GC_malloc(sizeof(PyBaseExceptionObject));
    if (exc == NULL) {
        return NULL;
    }
    
    Py_TYPE(exc) = type;
    Py_REFCNT(exc) = 1;
    
    if (args == NULL) {
        exc->args = PyList_New(0);
    } else {
        exc->args = Py_NewRef(args);
    }
    
    exc->msg = Py_None;
    Py_INCREF(Py_None);
    
    return (PyObject *)exc;
}

/**
 * 例外からメッセージを取得
 */
PyObject* PyException_GetMessage(PyObject *exc) {
    if (!PyObject_TypeCheck(exc, &PyBaseException_Type)) {
        PyErr_SetString(PyExc_TypeError, "not an exception object");
        return NULL;
    }
    
    PyObject *msg = ((PyBaseExceptionObject *)exc)->msg;
    if (msg == NULL) {
        msg = Py_None;
    }
    
    Py_INCREF(msg);
    return msg;
}

/**
 * 例外から引数を取得
 */
PyObject* PyException_GetArgs(PyObject *exc) {
    if (!PyObject_TypeCheck(exc, &PyBaseException_Type)) {
        PyErr_SetString(PyExc_TypeError, "not an exception object");
        return NULL;
    }
    
    PyObject *args = ((PyBaseExceptionObject *)exc)->args;
    if (args == NULL) {
        args = Py_None;
    }
    
    Py_INCREF(args);
    return args;
}

/**
 * 例外にメッセージを設定
 */
int PyException_SetMessage(PyObject *exc, PyObject *msg) {
    if (!PyObject_TypeCheck(exc, &PyBaseException_Type)) {
        PyErr_SetString(PyExc_TypeError, "not an exception object");
        return -1;
    }
    
    PyBaseExceptionObject *e = (PyBaseExceptionObject *)exc;
    Py_XDECREF(e->msg);
    
    if (msg == NULL) {
        e->msg = Py_None;
        Py_INCREF(Py_None);
    } else {
        e->msg = Py_NewRef(msg);
    }
    
    return 0;
}

/**
 * 例外を設定（文字列メッセージ付き）
 */
void PyErr_SetString(PyObject *exception, const char *msg) {
    PyObject *msg_obj = NULL;
    
    if (msg != NULL) {
        msg_obj = PyUnicode_FromString(msg);
    }
    
    PyErr_SetObject(exception, msg_obj);
    Py_XDECREF(msg_obj);
}

/**
 * 例外を設定（値付き）
 */
void PyErr_SetObject(PyObject *exception, PyObject *value) {
    if (exception == NULL) {
        exception = PyExc_RuntimeError;
    }
    
    PyBaseExceptionObject *exc = (PyBaseExceptionObject *)exception;
    
    /* 現在の例外を設定 */
    Py_XDECREF(_current_exception);
    _current_exception = Py_NewRef(exception);
    
    /* メッセージを設定 */
    if (value != NULL) {
        PyException_SetMessage((PyObject *)exc, value);
    }
}

/**
 * フォーマット文字列で例外を設定
 */
PyObject* PyErr_Format(PyObject *exception, const char *format, ...) {
    va_list va;
    PyObject *msg;
    
    va_start(va, format);
    char buffer[1024]; /* 実際はより堅牢な実装が必要 */
    vsnprintf(buffer, sizeof(buffer), format, va);
    va_end(va);
    
    msg = PyUnicode_FromString(buffer);
    PyErr_SetObject(exception, msg);
    Py_XDECREF(msg);
    
    return NULL; /* 常にNULLを返す (エラー表現のため) */
}

/**
 * 内部エラー (バグ) の例外を設定
 */
void PyErr_BadInternalCall(void) {
    PyErr_SetString(PyExc_RuntimeError, "bad internal call");
}

/**
 * メモリ不足例外を設定
 */
PyObject* PyErr_NoMemory(void) {
    PyErr_SetString(PyExc_RuntimeError, "out of memory");
    return NULL;
}

/**
 * 現在の例外を取得
 */
PyObject* PyErr_Occurred(void) {
    return _current_exception;
}

/**
 * 例外をクリア
 */
void PyErr_Clear(void) {
    Py_XDECREF(_current_exception);
    _current_exception = NULL;
}

/**
 * 例外型マッチングチェック
 */
int PyErr_ExceptionMatches(PyObject *exc) {
    PyObject *current = PyErr_Occurred();
    if (current == NULL || exc == NULL) {
        return 0;
    }
    
    /* 同一の例外型ならマッチ */
    if (current == exc) {
        return 1;
    }
    
    /* 型の継承関係をチェック (簡易版) */
    PyTypeObject *current_type = Py_TYPE(current);
    PyTypeObject *exc_type = (PyTypeObject *)exc;
    
    return PyType_IsSubtype(current_type, exc_type);
}

/**
 * 例外システムの初期化
 */
void _PyExc_Init(void) {
    /* 例外型辞書初期化 */
    _exception_types = PyDict_New();
    if (_exception_types == NULL) {
        return;
    }
    
    /* 型初期化 */
    PyType_Ready(&PyBaseException_Type);
    PyType_Ready(&PyException_Type);
    
    /* Exceptionの基底をBaseExceptionに設定 */
    PyException_Type.tp_base = &PyBaseException_Type;
    
    /* 標準例外型作成 */
    PyExc_BaseException = create_exception_type("BaseException", &PyBaseException_Type);
    PyExc_Exception = create_exception_type("Exception", &PyException_Type);
    PyExc_TypeError = create_exception_type("TypeError", &PyException_Type);
    PyExc_AttributeError = create_exception_type("AttributeError", &PyException_Type);
    PyExc_ValueError = create_exception_type("ValueError", &PyException_Type);
    PyExc_IndexError = create_exception_type("IndexError", &PyException_Type);
    PyExc_KeyError = create_exception_type("KeyError", &PyException_Type);
    PyExc_RuntimeError = create_exception_type("RuntimeError", &PyException_Type);
    PyExc_NameError = create_exception_type("NameError", &PyException_Type);
    PyExc_NotImplementedError = create_exception_type("NotImplementedError", &PyException_Type);
}
