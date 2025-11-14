#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

struct PyObject;
struct PyFunctionObject;
struct PyUnicodeObject;
struct PyTupleObject;
struct PyDictObject;

// 参照カウントは適当でOK（MVP）。後でちゃんと実装する。
void __py_incref(PyObject*);
void __py_decref(PyObject*);

// 文字列
PyUnicodeObject* __py_str_from_utf8(const char* data, std::size_t len);

// タプル (positional args 用)
PyTupleObject* __py_tuple_new(std::size_t n);
void __py_tuple_setitem(PyTupleObject*, std::size_t, PyObject*);

// 辞書
PyObject* __py_dict_new();
PyObject* __py_dict_insert(PyObject* dict, PyObject* key, PyObject* value);

// None シングルトン
PyObject* __py_get_none();

// PyFunctionObject 関連：vectorcall で実行
PyFunctionObject* __py_get_builtin_print();
PyObject* __py_call_vectorcall(
    PyObject* callable,
    PyTupleObject* posargs,
    PyTupleObject* kwnames,
    PyTupleObject* kwvalues
);
PyObject* __py_call(
    PyObject* callable,
    PyTupleObject* posargs,
    PyObject* kwargs
);

} // extern "C"
