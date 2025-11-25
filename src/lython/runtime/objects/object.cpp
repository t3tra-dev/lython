#include "objects/object.h"
#include "objects/bool.h"
#include "objects/dict.h"
#include "objects/float.h"
#include "objects/function.h"
#include "objects/long.h"
#include "objects/tuple.h"
#include "objects/unicode.h"

#include <cstdio>
#include <cstring>

namespace {

LyTypeObject makeType(const char *name, Ly_ssize_t basicsize,
                      Ly_ssize_t itemsize, Ly_ssize_t vectorcallOffset = 0,
                      LyDeallocFunc dealloc = nullptr,
                      LyReprFunc repr = nullptr) {
  LyTypeObject type{};
  type.ob_base.ob_base.ob_refcnt = 1;
  type.ob_base.ob_base.ob_type = nullptr;
  type.ob_base.ob_size = 0;
  type.tp_name = name;
  type.tp_basicsize = basicsize;
  type.tp_itemsize = itemsize;
  type.tp_vectorcall_offset = vectorcallOffset;
  type.tp_dealloc = dealloc;
  type.tp_repr = repr;
  return type;
}

} // namespace

LyTypeObject &LyUnicode_Type() {
  static LyTypeObject type = makeType("str", sizeof(LyUnicodeObject), 0, 0,
                                      &LyUnicode_Dealloc, &LyUnicode_Repr);
  return type;
}

LyTypeObject &LyBool_Type() {
  // Bool is singleton (True/False), no deallocation needed
  static LyTypeObject type =
      makeType("bool", sizeof(LyBoolObject), 0, 0, nullptr, &LyBool_Repr);
  return type;
}

LyTypeObject &LyTuple_Type() {
  static LyTypeObject type =
      makeType("tuple", sizeof(LyTupleObject), sizeof(LyObject *), 0,
               &LyTuple_Dealloc, &LyTuple_Repr);
  return type;
}

LyTypeObject &LyFunction_Type() {
  static LyTypeObject type =
      makeType("function", sizeof(LyFunctionObject), 0,
               static_cast<Ly_ssize_t>(offsetof(LyFunctionObject, vectorcall)),
               &LyFunction_Dealloc, nullptr);
  return type;
}

LyTypeObject &LyNone_Type() {
  // None is singleton, no deallocation needed
  static LyTypeObject type =
      makeType("NoneType", sizeof(LyObject), 0, 0, nullptr, nullptr);
  return type;
}

LyTypeObject &LyDict_Type() {
  static LyTypeObject type = makeType("dict", sizeof(LyDictObject), 0, 0,
                                      &LyDict_Dealloc, &LyDict_Repr);
  return type;
}

LyTypeObject &LyLong_Type() {
  static LyTypeObject type = makeType("int", sizeof(LyLongObject), 0, 0,
                                      &LyLong_Dealloc, &LyLong_Repr);
  return type;
}

LyTypeObject &LyFloat_Type() {
  static LyTypeObject type = makeType("float", sizeof(LyFloatObject), 0, 0,
                                      &LyFloat_Dealloc, &LyFloat_Repr);
  return type;
}

extern "C" {

LyObject *Ly_GetNone() {
  static LyObject none = [] {
    LyObject obj{};
    obj.ob_refcnt = 1;
    obj.ob_type = &LyNone_Type();
    return obj;
  }();
  return &none;
}

void Ly_IncRef(LyObject *object) {
  if (!object)
    return;
  ++object->ob_refcnt;
}

void Ly_DecRef(LyObject *object) {
  if (!object)
    return;
  if (object->ob_refcnt <= 0)
    return;
  if (--object->ob_refcnt == 0) {
    LyTypeObject *type = object->ob_type;
    if (type && type->tp_dealloc)
      type->tp_dealloc(object);
  }
}
static thread_local std::size_t reprDepth = 0;

LyUnicodeObject *LyObject_Repr(LyObject *object) {
  auto makeFallback = [&](const char *fmt) {
    char buffer[128];
    const char *typeName = object && object->ob_type && object->ob_type->tp_name
                               ? object->ob_type->tp_name
                               : "object";
    std::snprintf(buffer, sizeof(buffer), fmt, typeName,
                  static_cast<void *>(object));
    return LyUnicode_FromUTF8(buffer, std::strlen(buffer));
  };

  if (!object)
    return LyUnicode_FromUTF8("<NULL>", 6);

  LyTypeObject *type = object->ob_type;
  if (!type || !type->tp_repr)
    return makeFallback("<%s object at %p>");

  constexpr std::size_t kMaxReprDepth = 64;
  if (reprDepth > kMaxReprDepth)
    return LyUnicode_FromUTF8("<...>", 5);

  ++reprDepth;
  LyUnicodeObject *result = type->tp_repr(object);
  --reprDepth;

  if (!result)
    return makeFallback("<%s object at %p>");

  if (result->ob_base.ob_base.ob_type != &LyUnicode_Type()) {
    Ly_DecRef(reinterpret_cast<LyObject *>(result));
    return makeFallback("<%s object at %p>");
  }
  return result;
}
}
