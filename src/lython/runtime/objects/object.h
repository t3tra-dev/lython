#pragma once

#include <cstddef>

using Ly_ssize_t = std::ptrdiff_t;

struct LyObject;
struct LyUnicodeObject;
struct LyTupleObject;
struct LyDictObject;
struct LyLongObject;
struct LyFloatObject;
struct LyBoolObject;

using LyReprFunc = LyUnicodeObject *(*)(LyObject *);
using LyDeallocFunc = void (*)(LyObject *);

struct LyTypeObject;

struct LyObject {
  Ly_ssize_t ob_refcnt;
  LyTypeObject *ob_type;
};

struct LyVarObject {
  LyObject ob_base;
  Ly_ssize_t ob_size;
};

struct LyTypeObject {
  LyVarObject ob_base;
  const char *tp_name;
  Ly_ssize_t tp_basicsize;
  Ly_ssize_t tp_itemsize;
  Ly_ssize_t tp_vectorcall_offset;
  LyDeallocFunc tp_dealloc;
  LyReprFunc tp_repr;
};

LyTypeObject &LyUnicode_Type();
LyTypeObject &LyBool_Type();
LyTypeObject &LyTuple_Type();
LyTypeObject &LyFunction_Type();
LyTypeObject &LyNone_Type();
LyTypeObject &LyDict_Type();
LyTypeObject &LyLong_Type();
LyTypeObject &LyFloat_Type();

extern "C" {

LyObject *Ly_GetNone();
void Ly_IncRef(LyObject *object);
void Ly_DecRef(LyObject *object);
LyUnicodeObject *LyObject_Repr(LyObject *object);
}
