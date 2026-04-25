#pragma once

#include "objects/object.h"

#include <cstddef>
#include <cstdint>

using LyVectorcallFunc = LyObject *(*)(LyObject *, LyObject *const *,
                                       std::size_t, LyObject *);

struct LyFunctionObject {
  LyObject ob_base;
  LyObject *func_name;
  LyVectorcallFunc vectorcall;
};

void LyFunction_Dealloc(LyObject *object);
LyVectorcallFunc LyFunction_LoadVectorcall(LyObject *callable);
