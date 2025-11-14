#include "lyrt_abi.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using Py_ssize_t = std::ptrdiff_t;

struct PyTypeObject;

struct PyObject {
  Py_ssize_t ob_refcnt;
  PyTypeObject *ob_type;
};

struct PyVarObject : PyObject {
  Py_ssize_t ob_size;
};

using vectorcallfunc = PyObject *(*)(PyObject *, PyObject *const *, size_t,
                                     PyObject *);

struct PyTypeObject {
  PyVarObject ob_base;
  const char *tp_name;
  Py_ssize_t tp_basicsize;
  Py_ssize_t tp_itemsize;
  Py_ssize_t tp_vectorcall_offset;
};

struct PyUnicodeObject {
  PyVarObject ob_base;
  std::int64_t hash;
  char *utf8_data;
  Py_ssize_t utf8_length;
};

struct PyTupleObject {
  PyVarObject ob_base;
  PyObject *ob_item[1];
};

struct PyFunctionObject {
  PyObject ob_base;
  PyObject *func_code;
  PyObject *func_globals;
  PyObject *func_defaults;
  PyObject *func_kwdefaults;
  PyObject *func_closure;
  PyObject *func_doc;
  PyObject *func_name;
  PyObject *func_dict;
  PyObject *func_weakreflist;
  PyObject *func_module;
  PyObject *func_annotations;
  PyObject *func_qualname;
  vectorcallfunc vectorcall;
};

struct PyDictEntry {
  PyObject *key;
  PyObject *value;
};

struct PyDictObject {
  PyObject ob_base;
  std::vector<PyDictEntry> entries;
};

namespace {

PyTypeObject makeType(const char *name, Py_ssize_t basicsize,
                      Py_ssize_t itemsize, Py_ssize_t vectorcallOffset = 0) {
  PyTypeObject type{};
  type.ob_base.ob_refcnt = 1;
  type.ob_base.ob_type = nullptr;
  type.ob_base.ob_size = 0;
  type.tp_name = name;
  type.tp_basicsize = basicsize;
  type.tp_itemsize = itemsize;
  type.tp_vectorcall_offset = vectorcallOffset;
  return type;
}

PyTypeObject &unicodeType() {
  static PyTypeObject type =
      makeType("str", sizeof(PyUnicodeObject), 0, /*vectorcallOffset=*/0);
  return type;
}

PyTypeObject &tupleType() {
  static PyTypeObject type =
      makeType("tuple", sizeof(PyTupleObject), sizeof(PyObject *), 0);
  return type;
}

PyTypeObject &functionType() {
  static PyTypeObject type =
      makeType("function", sizeof(PyFunctionObject), 0,
               static_cast<Py_ssize_t>(offsetof(PyFunctionObject, vectorcall)));
  return type;
}

PyTypeObject &noneType() {
  static PyTypeObject type =
      makeType("NoneType", sizeof(PyObject), 0, /*vectorcallOffset=*/0);
  return type;
}

PyTypeObject &dictType() {
  static PyTypeObject type =
      makeType("dict", sizeof(PyDictObject), 0, /*vectorcallOffset=*/0);
  return type;
}

PyObject *noneSingleton() {
  static PyObject none = [] {
    PyObject obj{};
    obj.ob_refcnt = 1;
    obj.ob_type = &noneType();
    return obj;
  }();
  return &none;
}

const char *asCString(PyObject *obj) {
  if (!obj || obj->ob_type != &unicodeType()) {
    return nullptr;
  }
  auto *uni = reinterpret_cast<PyUnicodeObject *>(obj);
  return uni->utf8_data;
}

void printTuple(PyTupleObject *tuple);

void printPyObject(PyObject *object) {
  if (object == nullptr) {
    std::cout << "<null>";
    return;
  }
  if (object == noneSingleton()) {
    std::cout << "None";
    return;
  }
  if (object->ob_type == &unicodeType()) {
    auto *str = reinterpret_cast<PyUnicodeObject *>(object);
    if (str->utf8_data) {
      std::cout.write(str->utf8_data, str->utf8_length);
    } else {
      std::cout << "<str>";
    }
    return;
  }
  if (object->ob_type == &tupleType()) {
    printTuple(reinterpret_cast<PyTupleObject *>(object));
    return;
  }
  if (object->ob_type == &dictType()) {
    auto *dict = reinterpret_cast<PyDictObject *>(object);
    std::cout << "{";
    for (size_t i = 0; i < dict->entries.size(); ++i) {
      printPyObject(dict->entries[i].key);
      std::cout << ": ";
      printPyObject(dict->entries[i].value);
      if (i + 1 < dict->entries.size())
        std::cout << ", ";
    }
    std::cout << "}";
    return;
  }
  const char *typeName = object->ob_type ? object->ob_type->tp_name : "object";
  std::cout << "<" << typeName << ">";
}

void printTuple(PyTupleObject *tuple) {
  if (!tuple) {
    std::cout << "()";
    return;
  }
  std::cout << "(";
  Py_ssize_t size = tuple->ob_base.ob_size;
  for (Py_ssize_t i = 0; i < size; ++i) {
    printPyObject(tuple->ob_item[i]);
    if (i + 1 < size) {
      std::cout << ", ";
    }
  }
  std::cout << ")";
}

PyTupleObject *allocateTuple(std::size_t n) {
  const std::size_t baseSize = sizeof(PyTupleObject);
  const std::size_t extra =
      n > 0 ? (n - 1) * sizeof(PyObject *) : 0; // ob_item already has 1 slot.
  void *raw = ::operator new(baseSize + extra);
  auto *tuple = reinterpret_cast<PyTupleObject *>(raw);
  tuple->ob_base.ob_refcnt = 1;
  tuple->ob_base.ob_type = &tupleType();
  tuple->ob_base.ob_size = static_cast<Py_ssize_t>(n);
  for (std::size_t i = 0; i < n; ++i) {
    tuple->ob_item[i] = nullptr;
  }
  return tuple;
}

PyDictObject *asDict(PyObject *object) {
  if (!object || object->ob_type != &dictType())
    return nullptr;
  return reinterpret_cast<PyDictObject *>(object);
}

PyObject *builtin_print_vectorcall(PyObject *, PyObject *const *args,
                                   size_t nargsf, PyObject *kwnamesObj) {
  std::size_t positionalCount = nargsf;
  auto *kwnames = reinterpret_cast<PyTupleObject *>(kwnamesObj);
  std::size_t keywordCount =
      kwnames ? static_cast<std::size_t>(kwnames->ob_base.ob_size) : 0;
  std::cout << "[print] ";
  for (std::size_t i = 0; i < positionalCount; ++i) {
    printPyObject(args[i]);
    if (i + 1 < positionalCount || keywordCount > 0) {
      std::cout << ' ';
    }
  }
  for (std::size_t i = 0; i < keywordCount; ++i) {
    const char *name = asCString(kwnames->ob_item[i]);
    if (name) {
      std::cout << name << "=";
    } else {
      std::cout << "kw" << i << "=";
    }
    printPyObject(args[positionalCount + i]);
    if (i + 1 < keywordCount) {
      std::cout << ' ';
    }
  }
  std::cout << std::endl;
  return noneSingleton();
}

PyFunctionObject *createBuiltinPrint() {
  void *raw = ::operator new(sizeof(PyFunctionObject));
  std::memset(raw, 0, sizeof(PyFunctionObject));
  auto *func = reinterpret_cast<PyFunctionObject *>(raw);
  func->ob_base.ob_refcnt = 1;
  func->ob_base.ob_type = &functionType();
  func->func_name =
      reinterpret_cast<PyObject *>(__py_str_from_utf8("print", 5));
  func->vectorcall = &builtin_print_vectorcall;
  return func;
}

vectorcallfunc loadVectorcall(PyObject *callable) {
  if (!callable || !callable->ob_type) {
    return nullptr;
  }
  PyTypeObject *type = callable->ob_type;
  if (type->tp_vectorcall_offset <= 0) {
    return nullptr;
  }
  auto *slot = reinterpret_cast<vectorcallfunc *>(
      reinterpret_cast<char *>(callable) + type->tp_vectorcall_offset);
  return slot ? *slot : nullptr;
}

} // namespace

extern "C" {

void __py_incref(PyObject *obj) {
  if (!obj) {
    return;
  }
  ++obj->ob_refcnt;
}

void __py_decref(PyObject *obj) {
  if (!obj) {
    return;
  }
  if (obj->ob_refcnt > 0) {
    --obj->ob_refcnt;
  }
}

PyUnicodeObject *__py_str_from_utf8(const char *data, std::size_t len) {
  if (!data) {
    return nullptr;
  }
  void *raw = ::operator new(sizeof(PyUnicodeObject));
  auto *str = reinterpret_cast<PyUnicodeObject *>(raw);
  str->ob_base.ob_refcnt = 1;
  str->ob_base.ob_type = &unicodeType();
  str->ob_base.ob_size = static_cast<Py_ssize_t>(len);
  str->utf8_length = static_cast<Py_ssize_t>(len);
  str->utf8_data = new char[len + 1];
  std::memcpy(str->utf8_data, data, len);
  str->utf8_data[len] = '\0';
  str->hash = 0;
  return str;
}

PyTupleObject *__py_tuple_new(std::size_t n) { return allocateTuple(n); }

void __py_tuple_setitem(PyTupleObject *tuple, std::size_t index,
                        PyObject *value) {
  if (!tuple) {
    return;
  }
  auto size = static_cast<std::size_t>(tuple->ob_base.ob_size);
  if (index >= size) {
    assert(false && "tuple index out of range");
    return;
  }
  tuple->ob_item[index] = value;
  __py_incref(value);
}

PyObject *__py_dict_new() {
  void *raw = ::operator new(sizeof(PyDictObject));
  auto *dict = reinterpret_cast<PyDictObject *>(raw);
  dict->ob_base.ob_refcnt = 1;
  dict->ob_base.ob_type = &dictType();
  dict->entries.clear();
  return reinterpret_cast<PyObject *>(dict);
}

PyObject *__py_dict_insert(PyObject *dictObj, PyObject *key, PyObject *value) {
  auto *dict = asDict(dictObj);
  if (!dict)
    return nullptr;

  auto it =
      std::find_if(dict->entries.begin(), dict->entries.end(),
                   [&](const PyDictEntry &entry) { return entry.key == key; });
  if (it == dict->entries.end()) {
    dict->entries.push_back({key, value});
    __py_incref(key);
    __py_incref(value);
  } else {
    it->value = value;
    __py_incref(value);
  }
  return dictObj;
}

PyObject *__py_get_none() { return noneSingleton(); }

PyFunctionObject *__py_get_builtin_print() {
  static PyFunctionObject *builtin = createBuiltinPrint();
  return builtin;
}

PyObject *__py_call_vectorcall(PyObject *callable, PyTupleObject *posargs,
                               PyTupleObject *kwnames,
                               PyTupleObject *kwvalues) {
  vectorcallfunc fn = loadVectorcall(callable);
  if (!fn) {
    return nullptr;
  }
  std::size_t positionalCount =
      posargs ? static_cast<std::size_t>(posargs->ob_base.ob_size) : 0;
  std::size_t keywordCount =
      kwnames ? static_cast<std::size_t>(kwnames->ob_base.ob_size) : 0;
  if (keywordCount > 0) {
    if (!kwvalues ||
        kwvalues->ob_base.ob_size != static_cast<Py_ssize_t>(keywordCount)) {
      return nullptr;
    }
  }
  std::vector<PyObject *> storage;
  storage.reserve(positionalCount + keywordCount);
  if (posargs) {
    for (std::size_t i = 0; i < positionalCount; ++i) {
      storage.push_back(posargs->ob_item[i]);
    }
  }
  if (kwvalues) {
    for (std::size_t i = 0; i < keywordCount; ++i) {
      storage.push_back(kwvalues->ob_item[i]);
    }
  }
  PyObject *kwnamesObj = reinterpret_cast<PyObject *>(kwnames);
  return fn(callable, storage.data(), positionalCount, kwnamesObj);
}

PyObject *__py_call(PyObject *callable, PyTupleObject *posargs,
                    PyObject *kwargs) {
  PyTupleObject *kwnames = nullptr;
  PyTupleObject *kwvalues = nullptr;
  if (!kwargs || kwargs == noneSingleton()) {
    kwnames = allocateTuple(0);
    kwvalues = allocateTuple(0);
  } else if (auto *dict = asDict(kwargs)) {
    std::size_t size = dict->entries.size();
    kwnames = allocateTuple(size);
    kwvalues = allocateTuple(size);
    for (std::size_t i = 0; i < size; ++i) {
      __py_tuple_setitem(kwnames, i, dict->entries[i].key);
      __py_tuple_setitem(kwvalues, i, dict->entries[i].value);
    }
  } else {
    return nullptr;
  }

  return __py_call_vectorcall(callable, posargs, kwnames, kwvalues);
}

} // extern "C"
