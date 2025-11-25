#pragma once

#include "objects/object.h"

struct LyFloatObject : LyObject {
  double value = 0.0;
};

extern "C" {

LyFloatObject *LyFloat_FromDouble(double value);
LyFloatObject *LyFloat_Add(const LyFloatObject *lhs, const LyFloatObject *rhs);
LyFloatObject *LyFloat_Sub(const LyFloatObject *lhs, const LyFloatObject *rhs);
}

void LyFloat_Dealloc(LyObject *object);
LyUnicodeObject *LyFloat_Repr(LyObject *object);
