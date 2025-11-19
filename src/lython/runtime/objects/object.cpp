#include "objects/object.h"
#include "objects/dict.h"
#include "objects/function.h"
#include "objects/tuple.h"
#include "objects/unicode.h"

#include <cstddef>

namespace {

LyTypeObject makeType(const char *name, Ly_ssize_t basicsize,
                      Ly_ssize_t itemsize, Ly_ssize_t vectorcallOffset = 0) {
  LyTypeObject type{};
  type.ob_base.ob_base.ob_refcnt = 1;
  type.ob_base.ob_base.ob_type = nullptr;
  type.ob_base.ob_size = 0;
  type.tp_name = name;
  type.tp_basicsize = basicsize;
  type.tp_itemsize = itemsize;
  type.tp_vectorcall_offset = vectorcallOffset;
  return type;
}

} // namespace

LyTypeObject &LyUnicode_Type() {
  static LyTypeObject type = makeType("str", sizeof(LyUnicodeObject), 0);
  return type;
}

LyTypeObject &LyTuple_Type() {
  static LyTypeObject type =
      makeType("tuple", sizeof(LyTupleObject), sizeof(LyObject *));
  return type;
}

LyTypeObject &LyFunction_Type() {
  static LyTypeObject type =
      makeType("function", sizeof(LyFunctionObject), 0,
               static_cast<Ly_ssize_t>(offsetof(LyFunctionObject, vectorcall)));
  return type;
}

LyTypeObject &LyNone_Type() {
  static LyTypeObject type = makeType("NoneType", sizeof(LyObject), 0);
  return type;
}

LyTypeObject &LyDict_Type() {
  static LyTypeObject type = makeType("dict", sizeof(LyDictObject), 0);
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
  if (object->ob_refcnt > 0)
    --object->ob_refcnt;
}
}
