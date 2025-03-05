/**
 * runtime/builtin/objects/classobject.c
 * Lythonクラスとインスタンスオブジェクトの実装
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "classobject.h"
#include "unicodeobject.h"
#include "dictobject.h"
#include "exceptions.h"
#include "listobject.h"
#include "functionobject.h"
#include "methodobject.h"

/* 前方宣言 */
static PyObject* class_repr(PyObject *class);
static PyObject* class_call(PyObject *class, PyObject *args, PyObject *kwargs);
static PyObject* class_getattro(PyObject *class, PyObject *name);
static int class_setattro(PyObject *class, PyObject *name, PyObject *value);

static PyObject* instance_repr(PyObject *self);
static PyObject* instance_getattro(PyObject *self, PyObject *name);
static int instance_setattro(PyObject *self, PyObject *name, PyObject *value);

/* 足りない関数の宣言 */
PyObject* PyObject_GetAttr(PyObject *obj, PyObject *name) {
    if (obj->ob_type->tp_getattro != NULL) {
        return obj->ob_type->tp_getattro(obj, name);
    } else if (obj->ob_type->tp_getattr != NULL) {
        const char *name_str = PyUnicode_AsUTF8(name);
        if (name_str != NULL) {
            return obj->ob_type->tp_getattr(obj, (char *)name_str);
        }
    }
    
    PyErr_Format(PyExc_AttributeError,
                 "'%.50s' object has no attribute '%.400s'",
                 obj->ob_type->tp_name, PyUnicode_AsUTF8(name));
    return NULL;
}

PyObject* PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs) {
    /* 呼び出し可能オブジェクトのtp_call関数を使用 */
    if (callable->ob_type->tp_call != NULL) {
        return callable->ob_type->tp_call(callable, args, kwargs);
    }
    
    PyErr_Format(PyExc_TypeError, "'%.200s' object is not callable",
                 callable->ob_type->tp_name);
    return NULL;
}

/* クラス型オブジェクト定義 */
PyTypeObject PyClass_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "type",                      /* tp_name */
    sizeof(PyClassObject),       /* tp_basicsize */
    0,                           /* tp_itemsize */
    0,                           /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    class_repr,                  /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    class_call,                  /* tp_call */
    0,                           /* tp_str */
    class_getattro,              /* tp_getattro */
    class_setattro,              /* tp_setattro */
    0,                           /* tp_as_buffer */
    0,                           /* tp_flags */
    0,                           /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
};

/* インスタンス型オブジェクト定義 */
PyTypeObject PyInstance_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "object",                    /* tp_name */
    sizeof(PyInstanceObject),    /* tp_basicsize */
    0,                           /* tp_itemsize */
    0,                           /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    instance_repr,               /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    0,                           /* tp_call */
    0,                           /* tp_str */
    instance_getattro,           /* tp_getattro */
    instance_setattro,           /* tp_setattro */
    0,                           /* tp_as_buffer */
    0,                           /* tp_flags */
    0,                           /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
};

/* クラスメソッド */

/**
 * クラス表示（str/repr）
 */
static PyObject* class_repr(PyObject *obj) {
    PyClassObject *cls = (PyClassObject *)obj;
    const char *name = PyUnicode_AsUTF8(cls->cl_name);
    
    if (name == NULL) {
        name = "?";
    }
    
    return PyUnicode_FromFormat("<class '%s'>", name);
}

/**
 * クラスの呼び出し - 新しいインスタンスを作成
 */
static PyObject* class_call(PyObject *obj, PyObject *args, PyObject *kwargs) {
    return PyInstance_New(obj, args, kwargs);
}

/**
 * クラス属性の検索
 */
static PyObject* class_getattro(PyObject *obj, PyObject *name) {
    PyClassObject *cls = (PyClassObject *)obj;
    PyObject *attr;
    
    /* クラス辞書を検索 */
    attr = PyDict_GetItem(cls->cl_dict, name);
    if (attr != NULL) {
        Py_INCREF(attr);
        return attr;
    }
    
    /* 基底クラスの検索 - 現在は未実装 */
    
    /* 見つからない場合 */
    PyErr_Format(PyExc_AttributeError,
                 "type object '%.50s' has no attribute '%.400s'",
                 PyUnicode_AsUTF8(cls->cl_name),
                 PyUnicode_AsUTF8(name));
    return NULL;
}

/**
 * クラス属性の設定
 */
static int class_setattro(PyObject *obj, PyObject *name, PyObject *value) {
    PyClassObject *cls = (PyClassObject *)obj;
    
    /* クラス辞書に設定 */
    if (PyDict_SetItem(cls->cl_dict, name, value) < 0) {
        return -1;
    }
    
    return 0;
}

/* インスタンスメソッド */

/**
 * インスタンスの表示
 */
static PyObject* instance_repr(PyObject *obj) {
    PyInstanceObject *self = (PyInstanceObject *)obj;
    PyObject *class_name = ((PyClassObject *)self->in_class)->cl_name;
    const char *name = PyUnicode_AsUTF8(class_name);
    
    if (name == NULL) {
        name = "?";
    }
    
    return PyUnicode_FromFormat("<%s object at %p>", name, self);
}

/**
 * インスタンス属性の検索
 */
static PyObject* instance_getattro(PyObject *obj, PyObject *name) {
    PyInstanceObject *self = (PyInstanceObject *)obj;
    PyObject *attr;
    
    /* インスタンス辞書を検索 */
    attr = PyDict_GetItem(self->in_dict, name);
    if (attr != NULL) {
        Py_INCREF(attr);
        return attr;
    }
    
    /* クラスを検索 */
    PyClassObject *cls = (PyClassObject *)self->in_class;
    attr = PyDict_GetItem(cls->cl_dict, name);
    if (attr != NULL) {
        /* メソッドの場合、selfにバインド */
        if (PyFunction_Check(attr)) {
            PyObject *bound = PyMethod_New(attr, obj);
            return bound;
        }
        
        Py_INCREF(attr);
        return attr;
    }
    
    /* 見つからない場合 */
    PyErr_Format(PyExc_AttributeError,
                 "'%.50s' object has no attribute '%.400s'",
                 PyUnicode_AsUTF8(((PyClassObject *)self->in_class)->cl_name),
                 PyUnicode_AsUTF8(name));
    return NULL;
}

/**
 * インスタンス属性の設定
 */
static int instance_setattro(PyObject *obj, PyObject *name, PyObject *value) {
    PyInstanceObject *self = (PyInstanceObject *)obj;
    
    /* インスタンス辞書に設定 */
    if (PyDict_SetItem(self->in_dict, name, value) < 0) {
        return -1;
    }
    
    return 0;
}

/**
 * 新しいクラスオブジェクトを作成
 */
PyObject* PyClass_New(PyObject *name, PyObject *bases, PyObject *dict) {
    if (!PyUnicode_Check(name)) {
        PyErr_SetString(PyExc_TypeError, "class name must be a string");
        return NULL;
    }
    
    if (bases != NULL && !PyList_Check(bases)) {
        PyErr_SetString(PyExc_TypeError, "bases must be a list");
        return NULL;
    }
    
    if (dict != NULL && !PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "dict must be a dictionary");
        return NULL;
    }
    
    PyClassObject *cls = (PyClassObject *)GC_malloc(sizeof(PyClassObject));
    if (cls == NULL) {
        return NULL;
    }
    
    Py_TYPE(cls) = &PyClass_Type;
    Py_REFCNT(cls) = 1;
    
    cls->cl_name = Py_NewRef(name);
    
    if (dict == NULL) {
        cls->cl_dict = PyDict_New();
    } else {
        cls->cl_dict = Py_NewRef(dict);
    }
    
    if (bases == NULL) {
        cls->cl_bases = PyList_New(0);
    } else {
        cls->cl_bases = Py_NewRef(bases);
    }
    
    return (PyObject *)cls;
}

/**
 * 初期化せずに新しいインスタンスを作成
 */
PyObject* PyInstance_NewRaw(PyObject *class) {
    if (!PyClass_Check(class)) {
        PyErr_SetString(PyExc_TypeError, "PyInstance_NewRaw: class argument must be a class");
        return NULL;
    }
    
    PyInstanceObject *self = (PyInstanceObject *)GC_malloc(sizeof(PyInstanceObject));
    if (self == NULL) {
        return NULL;
    }
    
    Py_TYPE(self) = &PyInstance_Type;
    Py_REFCNT(self) = 1;
    
    self->in_class = Py_NewRef(class);
    self->in_dict = PyDict_New();
    
    return (PyObject *)self;
}

/**
 * 新しいインスタンスを作成し__init__を呼び出す
 */
PyObject* PyInstance_New(PyObject *class, PyObject *args, PyObject *kwargs) {
    PyObject *self = PyInstance_NewRaw(class);
    if (self == NULL) {
        return NULL;
    }
    
    /* __init__メソッドを検索 */
    PyObject *init_str = PyUnicode_FromString("__init__");
    if (init_str == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    
    PyObject *init = PyObject_GetAttr(class, init_str);
    Py_DECREF(init_str);
    
    if (init != NULL) {
        /* __init__(self, *args, **kwargs)を呼び出し */
        PyObject *init_args = PyList_New(0);
        if (init_args == NULL) {
            Py_DECREF(self);
            Py_DECREF(init);
            return NULL;
        }
        
        /* selfを最初の引数として追加 */
        if (PyList_Append(init_args, self) < 0) {
            Py_DECREF(self);
            Py_DECREF(init);
            Py_DECREF(init_args);
            return NULL;
        }
        
        /* 残りの引数を追加 */
        if (args != NULL) {
            Py_ssize_t len = PyList_Size(args);
            for (Py_ssize_t i = 0; i < len; i++) {
                PyObject *item = PyList_GetItem(args, i);
                if (PyList_Append(init_args, item) < 0) {
                    Py_DECREF(self);
                    Py_DECREF(init);
                    Py_DECREF(init_args);
                    return NULL;
                }
            }
        }
        
        /* __init__呼び出し */
        PyObject *result = PyObject_Call(init, init_args, kwargs);
        Py_DECREF(init);
        Py_DECREF(init_args);
        
        if (result == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        
        Py_DECREF(result);
    }
    
    return self;
}

/**
 * クラスシステムの初期化
 */
void _PyClass_Init(void) {
    PyType_Ready(&PyClass_Type);
    PyType_Ready(&PyInstance_Type);
}
