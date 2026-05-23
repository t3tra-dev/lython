#include "objects/object.h"
#include "objects/bool.h"
#include "objects/float.h"
#include "objects/function.h"
#include "objects/long.h"
#include "objects/unicode.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

LyTypeObject makeType(const char *name, Ly_ssize_t basicsize,
                      Ly_ssize_t itemsize, Ly_ssize_t vectorcallOffset = 0,
                      LyDeallocFunc dealloc = nullptr,
                      LyReprFunc repr = nullptr) {
  LyTypeObject type{};
  type.ob_base.ob_base.ob_refcnt = kLyImmortalRefcount;
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

inline Ly_ssize_t atomicLoadRefCount(const Ly_ssize_t *ptr) {
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

inline void atomicIncRefCount(Ly_ssize_t *ptr) {
  __atomic_add_fetch(ptr, 1, __ATOMIC_RELAXED);
}

inline bool atomicTryDecRefCount(Ly_ssize_t *ptr, Ly_ssize_t &previous) {
  previous = atomicLoadRefCount(ptr);
  while (previous > 0) {
    Ly_ssize_t desired = previous - 1;
    if (__atomic_compare_exchange_n(ptr, &previous, desired,
                                    /*weak=*/false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE)) {
      return true;
    }
  }
  return false;
}

bool isImmortalObject(LyObject *object) {
  if (!object)
    return false;
  if (LyObject_HasImmortalRefcount(object))
    return true;
  if (object == Ly_GetNone())
    return true;
  if (object->ob_type == &LyBool_Type())
    return true;
  if (object->ob_type == &LyLong_Type() &&
      LyLong_IsSmallInt(reinterpret_cast<LyLongObject *>(object))) {
    return true;
  }
  return false;
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
    obj.ob_refcnt = kLyImmortalRefcount;
    obj.ob_type = &LyNone_Type();
    return obj;
  }();
  return &none;
}

void Ly_IncRef(LyObject *object) {
  if (!object)
    return;
  if (isImmortalObject(object))
    return;
  atomicIncRefCount(&object->ob_refcnt);
}

void Ly_DecRef(LyObject *object) {
  if (!object)
    return;
  if (isImmortalObject(object))
    return;
  Ly_ssize_t previous = 0;
  if (!atomicTryDecRefCount(&object->ob_refcnt, previous)) {
    std::fprintf(stderr,
                 "fatal: Ly_DecRef observed non-positive refcount (%td) for "
                 "object %p\n",
                 static_cast<std::ptrdiff_t>(previous),
                 static_cast<void *>(object));
    std::abort();
  }
  if (previous == 1) {
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

bool LyObject_EqBool(LyObject *lhs, LyObject *rhs) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;

  if (lhs->ob_type == &LyUnicode_Type() && rhs->ob_type == &LyUnicode_Type()) {
    auto *left = reinterpret_cast<LyUnicodeObject *>(lhs);
    auto *right = reinterpret_cast<LyUnicodeObject *>(rhs);
    if (left->utf8_length != right->utf8_length)
      return false;
    return std::memcmp(left->utf8_data, right->utf8_data,
                       static_cast<std::size_t>(left->utf8_length)) == 0;
  }

  if (lhs->ob_type == &LyLong_Type() && rhs->ob_type == &LyLong_Type()) {
    return LyLong_AsI64(lhs) == LyLong_AsI64(rhs);
  }

  if (lhs->ob_type == &LyFloat_Type() && rhs->ob_type == &LyFloat_Type()) {
    return LyFloat_AsDouble(lhs) == LyFloat_AsDouble(rhs);
  }

  if (lhs->ob_type == &LyBool_Type() && rhs->ob_type == &LyBool_Type()) {
    return LyBool_AsBool(lhs) == LyBool_AsBool(rhs);
  }

  return false;
}

LyUnicodeObject *LyClass_ReprNamed(const char *name, const void *object) {
  char buffer[160];
  const char *className = name ? name : "class";
  std::snprintf(buffer, sizeof(buffer), "<%s object at %p>", className, object);
  return LyUnicode_FromUTF8(buffer, std::strlen(buffer));
}

void *LyMem_Alloc(std::size_t size) { return ::operator new(size); }

void LyMem_Free(void *ptr) { ::operator delete(ptr); }
}
