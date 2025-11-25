#pragma once

#include "objects/object.h"

#include <cstddef>

using LyVectorcallFunc = LyObject *(*)(LyObject *, LyObject *const *,
                                       std::size_t, LyObject *);

struct LyFunctionObject {
  LyObject ob_base;
  LyObject *func_code;
  LyObject *func_globals;
  LyObject *func_defaults;
  LyObject *func_kwdefaults;
  LyObject *func_closure;
  LyObject *func_doc;
  LyObject *func_name;
  LyObject *func_dict;
  LyObject *func_weakreflist;
  LyObject *func_module;
  LyObject *func_annotations;
  LyObject *func_qualname;
  LyVectorcallFunc vectorcall;
};

void LyFunction_Dealloc(LyObject *object);
LyVectorcallFunc LyFunction_LoadVectorcall(LyObject *callable);
