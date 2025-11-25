#include "objects/bool.h"
#include "objects/unicode.h"

namespace {

LyBoolObject *trueSingleton() {
  static LyBoolObject trueObj = [] {
    LyBoolObject obj{};
    obj.ob_refcnt = 1;
    obj.ob_type = &LyBool_Type();
    obj.value = true;
    return obj;
  }();
  return &trueObj;
}

LyBoolObject *falseSingleton() {
  static LyBoolObject falseObj = [] {
    LyBoolObject obj{};
    obj.ob_refcnt = 1;
    obj.ob_type = &LyBool_Type();
    obj.value = false;
    return obj;
  }();
  return &falseObj;
}

} // namespace

extern "C" {

LyBoolObject *LyBool_FromBool(bool value) {
  return value ? trueSingleton() : falseSingleton();
}

bool LyBool_AsBool(LyObject *object) {
  if (!object || object->ob_type != &LyBool_Type())
    return false;
  return reinterpret_cast<LyBoolObject *>(object)->value;
}

LyUnicodeObject *LyBool_Repr(LyObject *object) {
  if (!object || object->ob_type != &LyBool_Type())
    return LyUnicode_FromUTF8("<bool>", 6);
  bool value = reinterpret_cast<LyBoolObject *>(object)->value;
  if (value)
    return LyUnicode_FromUTF8("True", 4);
  return LyUnicode_FromUTF8("False", 5);
}
}
