#include "objects/float.h"
#include "objects/unicode.h"

#include <cstdio>
#include <cstring>

namespace {

LyFloatObject *allocateFloat() {
  auto *obj = new LyFloatObject();
  obj->ob_refcnt = 1;
  obj->ob_type = &LyFloat_Type();
  obj->value = 0.0;
  return obj;
}

} // namespace

extern "C" {

LyFloatObject *LyFloat_FromDouble(double value) {
  auto *obj = allocateFloat();
  obj->value = value;
  return obj;
}

LyFloatObject *LyFloat_Add(const LyFloatObject *lhs, const LyFloatObject *rhs) {
  return LyFloat_FromDouble(lhs->value + rhs->value);
}

LyFloatObject *LyFloat_Sub(const LyFloatObject *lhs, const LyFloatObject *rhs) {
  return LyFloat_FromDouble(lhs->value - rhs->value);
}
}

void LyFloat_Dealloc(LyObject *object) {
  if (!object)
    return;
  delete reinterpret_cast<LyFloatObject *>(object);
}

LyUnicodeObject *LyFloat_Repr(LyObject *object) {
  auto *floatObj = reinterpret_cast<LyFloatObject *>(object);
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.12g", floatObj->value);
  return LyUnicode_FromUTF8(buffer, std::strlen(buffer));
}
