#pragma once

#include "objects/object.h"

#include <vector>

struct LyDictEntry {
  LyObject *key;
  LyObject *value;
};

struct LyDictObject {
  LyObject ob_base;
  std::vector<LyDictEntry> entries;
};

LyDictObject *LyDict_Cast(LyObject *object);

extern "C" {

LyObject *LyDict_New();
LyObject *LyDict_Insert(LyObject *dict, LyObject *key, LyObject *value);
}

void LyDict_Dealloc(LyObject *object);
LyUnicodeObject *LyDict_Repr(LyObject *object);
