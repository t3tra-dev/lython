#pragma once

#include <cstdint>

#include "objects/dict.h"
#include "objects/float.h"
#include "objects/function.h"
#include "objects/long.h"
#include "objects/object.h"
#include "objects/tuple.h"
#include "objects/unicode.h"

extern "C" {

LyUnicodeObject *LyUnicode_FromUTF8(const char *data, std::size_t len);

LyTupleObject *LyTuple_New(std::size_t size);
void LyTuple_SetItem(LyTupleObject *tuple, std::size_t index, LyObject *value);

LyObject *LyDict_New();
LyObject *LyDict_Insert(LyObject *dict, LyObject *key, LyObject *value);

LyObject *Ly_GetNone();
LyFunctionObject *Ly_GetBuiltinPrint();

LyLongObject *LyLong_FromI64(std::int64_t value);
LyFloatObject *LyFloat_FromDouble(double value);
LyObject *LyNumber_Add(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Sub(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Le(LyObject *lhs, LyObject *rhs);
bool LyBool_AsBool(LyObject *object);

LyObject *Ly_CallVectorcall(LyObject *callable, LyTupleObject *posargs,
                            LyTupleObject *kwnames, LyTupleObject *kwvalues);
LyObject *Ly_Call(LyObject *callable, LyTupleObject *posargs, LyObject *kwargs);

} // extern "C"
