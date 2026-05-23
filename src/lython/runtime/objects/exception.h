#pragma once

#include "objects/object.h"
#include "objects/unicode.h"

struct LyExceptionObject {
  LyObject ob_base;
  LyObject *type;
  LyUnicodeObject *message;
  LyObject *args;
  LyObject *cause;
  LyObject *context;
  LyObject *traceback;
  LyObject *location;
  LyObject *extras;
};

LyTypeObject &LyException_Type();

extern "C" {
LyExceptionObject *LyException_New(LyObject *type, LyUnicodeObject *message,
                                   LyObject *args, LyObject *cause,
                                   LyObject *context, LyObject *traceback,
                                   LyObject *location, LyObject *extras);
void LyException_SetCurrent(LyObject *exception);
LyObject *LyException_GetCurrent();
void LyException_Clear();
void LyEH_Throw(LyObject *exception);
LyObject *LyEH_Capture(void *exc);
void LyEH_SetJump(void *env);
void LyEH_ClearJump();
}

void LyException_Dealloc(LyObject *object);
LyUnicodeObject *LyException_Repr(LyObject *object);
