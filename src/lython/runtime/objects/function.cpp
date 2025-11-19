#include "objects/function.h"

LyVectorcallFunc LyFunction_LoadVectorcall(LyObject *callable) {
  if (!callable || !callable->ob_type)
    return nullptr;
  LyTypeObject *type = callable->ob_type;
  if (type->tp_vectorcall_offset <= 0)
    return nullptr;
  auto *slot = reinterpret_cast<LyVectorcallFunc *>(
      reinterpret_cast<char *>(callable) + type->tp_vectorcall_offset);
  return slot ? *slot : nullptr;
}
