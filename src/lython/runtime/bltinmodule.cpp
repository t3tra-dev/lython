#include "lyrt.h"

#include "objects/dict.h"
#include "objects/function.h"
#include "objects/tuple.h"
#include "objects/unicode.h"

#include <cstring>
#include <iostream>

extern "C" LyObject *builtin_print_impl(LyObject *module,
                                        LyObject *const *objects,
                                        Ly_ssize_t objects_length,
                                        LyObject *sep, LyObject *end,
                                        LyObject *file, int flush);

namespace {

void printTuple(LyTupleObject *tuple);

void printObject(LyObject *object) {
  if (!object) {
    std::cout << "<null>";
    return;
  }
  if (object == Ly_GetNone()) {
    std::cout << "None";
    return;
  }
  if (object->ob_type == &LyUnicode_Type()) {
    auto *unicode = reinterpret_cast<LyUnicodeObject *>(object);
    if (unicode->utf8_data)
      std::cout.write(unicode->utf8_data, unicode->utf8_length);
    else
      std::cout << "<str>";
    return;
  }
  if (object->ob_type == &LyTuple_Type()) {
    printTuple(reinterpret_cast<LyTupleObject *>(object));
    return;
  }
  if (object->ob_type == &LyDict_Type()) {
    auto *dict = LyDict_Cast(object);
    std::cout << "{";
    for (std::size_t i = 0; i < dict->entries.size(); ++i) {
      printObject(dict->entries[i].key);
      std::cout << ": ";
      printObject(dict->entries[i].value);
      if (i + 1 < dict->entries.size())
        std::cout << ", ";
    }
    std::cout << "}";
    return;
  }
  const char *typeName = object->ob_type ? object->ob_type->tp_name : "object";
  std::cout << "<" << typeName << ">";
}

void printTuple(LyTupleObject *tuple) {
  if (!tuple) {
    std::cout << "()";
    return;
  }
  std::cout << "(";
  Ly_ssize_t size = tuple->ob_base.ob_size;
  for (Ly_ssize_t i = 0; i < size; ++i) {
    printObject(tuple->ob_item[i]);
    if (i + 1 < size)
      std::cout << ", ";
  }
  std::cout << ")";
}

LyObject *builtin_print_vectorcall(LyObject *callable, LyObject *const *args,
                                   std::size_t nargsf, LyObject *) {
  return builtin_print_impl(callable, args, static_cast<Ly_ssize_t>(nargsf),
                            nullptr, nullptr, nullptr, 0);
}

} // namespace

extern "C" {

LyObject *builtin_print_impl(LyObject *, LyObject *const *objects,
                             Ly_ssize_t objects_length, LyObject *, LyObject *,
                             LyObject *, int) {
  for (Ly_ssize_t i = 0; i < objects_length; ++i) {
    printObject(objects[i]);
    if (i + 1 < objects_length)
      std::cout << ' ';
  }
  std::cout << std::endl;
  return Ly_GetNone();
}

LyFunctionObject *Ly_GetBuiltinPrint() {
  static LyFunctionObject *builtin = [] {
    void *raw = ::operator new(sizeof(LyFunctionObject));
    std::memset(raw, 0, sizeof(LyFunctionObject));
    auto *func = reinterpret_cast<LyFunctionObject *>(raw);
    func->ob_base.ob_refcnt = 1;
    func->ob_base.ob_type = &LyFunction_Type();
    func->func_name =
        reinterpret_cast<LyObject *>(LyUnicode_FromUTF8("print", 5));
    func->vectorcall = &builtin_print_vectorcall;
    return func;
  }();
  return builtin;
}
}
