#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

using Ly_ssize_t = std::ptrdiff_t;
using LyMutex = std::mutex;
inline constexpr Ly_ssize_t kLyImmortalRefcount =
    std::numeric_limits<Ly_ssize_t>::max();

struct LyObject;
struct LyUnicodeObject;
struct LyLongObject;
struct LyFloatObject;
struct LyBoolObject;
struct LyExceptionObject;

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

inline bool LyObject_HasImmortalRefcount(const LyObject *object) {
  if (!object)
    return false;
  Ly_ssize_t refcount = __atomic_load_n(&object->ob_refcnt, __ATOMIC_ACQUIRE);
  return refcount == kLyImmortalRefcount;
}

LyTypeObject &LyUnicode_Type();
LyTypeObject &LyBool_Type();
LyTypeObject &LyFunction_Type();
LyTypeObject &LyNone_Type();
LyTypeObject &LyLong_Type();
LyTypeObject &LyFloat_Type();
LyTypeObject &LyException_Type();

extern "C" {

LyObject *Ly_GetNone();
void Ly_IncRef(LyObject *object);
void Ly_DecRef(LyObject *object);
LyUnicodeObject *LyObject_Repr(LyObject *object);
bool LyObject_EqBool(LyObject *lhs, LyObject *rhs);
LyUnicodeObject *LyClass_ReprNamed(const char *name, const void *object);
void *LyMem_Alloc(std::size_t size);
void LyMem_Free(void *ptr);
}
