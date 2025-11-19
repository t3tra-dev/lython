#pragma once

#include "objects/object.h"

struct LyTupleObject {
  LyVarObject ob_base;
  LyObject *ob_item[1];
};

extern "C" {

LyTupleObject *LyTuple_New(std::size_t size);
void LyTuple_SetItem(LyTupleObject *tuple, std::size_t index, LyObject *value);
}
