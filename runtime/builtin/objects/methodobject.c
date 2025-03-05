/**
 * runtime/builtin/objects/methodobject.c
 * Lythonメソッドオブジェクトの実装
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "exceptions.h"
#include "methodobject.h"
#include "functionobject.h"
#include "unicodeobject.h"
#include "listobject.h"

/* 前方宣言 */
static void method_dealloc(PyMethodObject *m);
static PyObject* method_call(PyObject *meth, PyObject *args, PyObject *kwargs);
static PyObject* method_repr(PyObject *m);
static Py_hash_t method_hash(PyObject *m);
static PyObject* method_richcompare(PyObject *a, PyObject *b, int op);
static PyObject* method_descr_get(PyObject *meth, PyObject *obj, PyObject *type);
static int method_traverse(PyMethodObject *m, visitproc visit, void *arg);

/* メソッド型オブジェクト定義 */
PyTypeObject PyMethod_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "method",                    /* tp_name */
    sizeof(PyMethodObject),      /* tp_basicsize */
    0,                           /* tp_itemsize */
    (destructor)method_dealloc,  /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    method_repr,                 /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    method_hash,                 /* tp_hash */
    method_call,                 /* tp_call */
    0,                           /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
};

/**
 * メソッドのデストラクタ
 */
static void method_dealloc(PyMethodObject *m) {
    if (m->im_weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)m);
    }
    Py_DECREF(m->im_func);
    Py_XDECREF(m->im_self);
    /* メモリはGCが管理しているので明示的解放は不要 */
}

/**
 * メソッド呼び出し - 関数をself引数付きで呼び出す
 */
static PyObject* method_call(PyObject *meth, PyObject *args, PyObject *kwargs) {
    PyMethodObject *m = (PyMethodObject *)meth;
    PyObject *self = m->im_self;
    PyObject *func = m->im_func;
    
    /* selfを先頭に持つ新しい引数リストを作成 */
    PyObject *new_args = PyList_New(0);
    if (new_args == NULL) {
        return NULL;
    }
    
    /* selfを最初の引数として追加 */
    if (PyList_Append(new_args, self) < 0) {
        Py_DECREF(new_args);
        return NULL;
    }
    
    /* 残りの引数を追加 */
    if (args != NULL) {
        Py_ssize_t len = PyList_Size(args);
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject *item = PyList_GetItem(args, i);
            if (PyList_Append(new_args, item) < 0) {
                Py_DECREF(new_args);
                return NULL;
            }
        }
    }
    
    /* 更新された引数で関数を呼び出し */
    PyObject *result = PyObject_Call(func, new_args, kwargs);
    Py_DECREF(new_args);
    
    return result;
}

/**
 * メソッドの表示
 */
static PyObject* method_repr(PyObject *obj) {
    PyMethodObject *m = (PyMethodObject *)obj;
    PyObject *self = m->im_self;
    PyObject *func = m->im_func;
    
    /* 関数名を取得 */
    PyObject *func_name;
    if (PyObject_HasAttrString(func, "__name__")) {
        func_name = PyObject_GetAttrString(func, "__name__");
        if (!PyUnicode_Check(func_name)) {
            Py_DECREF(func_name);
            func_name = PyUnicode_FromString("?");
        }
    } else {
        func_name = PyUnicode_FromString("?");
    }
    
    /* メソッド表現をフォーマット */
    PyObject *result = PyUnicode_FromFormat("<bound method %U of %R>",
                                          func_name, self);
    Py_DECREF(func_name);
    
    return result;
}

/**
 * メソッドのハッシュ値計算
 */
static Py_hash_t method_hash(PyObject *obj) {
    PyMethodObject *m = (PyMethodObject *)obj;
    Py_hash_t x, y;
    
    x = PyObject_Hash(m->im_self);
    if (x == -1) {
        return -1;
    }
    
    y = PyObject_Hash(m->im_func);
    if (y == -1) {
        return -1;
    }
    
    /* ハッシュを組み合わせる */
    x ^= y;
    if (x == -1) {
        x = -2;
    }
    
    return x;
}

/**
 * メソッドの比較
 */
static PyObject* method_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyMethod_Check(a) || !PyMethod_Check(b)) {
        return Py_NewRef(Py_NotImplemented);
    }
    
    if (op != Py_EQ && op != Py_NE) {
        return Py_NewRef(Py_NotImplemented);
    }
    
    PyMethodObject *ma = (PyMethodObject *)a;
    PyMethodObject *mb = (PyMethodObject *)b;
    
    int eq = PyObject_RichCompareBool(ma->im_func, mb->im_func, Py_EQ);
    if (eq < 0) {
        return NULL;
    }
    
    if (eq) {
        eq = PyObject_RichCompareBool(ma->im_self, mb->im_self, Py_EQ);
        if (eq < 0) {
            return NULL;
        }
    }
    
    if (op == Py_EQ) {
        if (eq) {
            return Py_NewRef(Py_True);
        } else {
            return Py_NewRef(Py_False);
        }
    } else { /* op == Py_NE */
        if (eq) {
            return Py_NewRef(Py_False);
        } else {
            return Py_NewRef(Py_True);
        }
    }
}

/**
 * メソッドdescriptor get - 同じメソッドを返す
 */
static PyObject* method_descr_get(PyObject *meth, PyObject *obj, PyObject *type) {
    Py_INCREF(meth);
    return meth;
}

/**
 * メソッドのトラバース - ガベージコレクション用
 */
static int method_traverse(PyMethodObject *m, visitproc visit, void *arg) {
    Py_VISIT(m->im_func);
    Py_VISIT(m->im_self);
    return 0;
}

/**
 * 新しいメソッドオブジェクトを作成
 */
PyObject* PyMethod_New(PyObject *func, PyObject *self) {
    if (self == NULL) {
        PyErr_SetString(PyExc_TypeError, "PyMethod_New: self must not be NULL");
        return NULL;
    }
    
    PyMethodObject *m = (PyMethodObject *)GC_malloc(sizeof(PyMethodObject));
    if (m == NULL) {
        return NULL;
    }
    
    Py_TYPE(m) = &PyMethod_Type;
    Py_REFCNT(m) = 1;
    
    m->im_weakreflist = NULL;
    m->im_func = Py_NewRef(func);
    m->im_self = Py_NewRef(self);
    m->vectorcall = NULL;
    
    return (PyObject *)m;
}

/**
 * メソッドから関数を取得
 */
PyObject* PyMethod_Function(PyObject *meth) {
    if (!PyMethod_Check(meth)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    
    return PyMethod_GET_FUNCTION(meth);
}

/**
 * メソッドからselfを取得
 */
PyObject* PyMethod_Self(PyObject *meth) {
    if (!PyMethod_Check(meth)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    
    return PyMethod_GET_SELF(meth);
}

/**
 * メソッドシステムの初期化
 */
void _PyMethod_Init(void) {
    PyType_Ready(&PyMethod_Type);
}
