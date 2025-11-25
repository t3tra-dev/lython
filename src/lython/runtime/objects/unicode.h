#pragma once

#include "objects/object.h"

#include <cstddef>
#include <cstdint>

struct LyUnicodeObject {
  LyVarObject ob_base;
  std::int64_t hash;
  char *utf8_data;
  Ly_ssize_t utf8_length;
};

const char *LyUnicode_AsUTF8(LyObject *object);

extern "C" {

LyUnicodeObject *LyUnicode_FromUTF8(const char *data, std::size_t len);
}

void LyUnicode_Dealloc(LyObject *object);
LyUnicodeObject *LyUnicode_Repr(LyObject *object);
