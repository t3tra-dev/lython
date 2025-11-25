#pragma once

#include "objects/object.h"

struct LyBoolObject : LyObject {
  bool value;
};

extern "C" {

LyBoolObject *LyBool_FromBool(bool value);
LyUnicodeObject *LyBool_Repr(LyObject *object);
bool LyBool_AsBool(LyObject *object);
}
