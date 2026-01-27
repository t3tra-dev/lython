#pragma once

#include "objects/object.h"

#include <cstdint>

struct LyTracebackFrame {
  const char *filename;
  const char *funcname;
  std::int32_t line;
  std::int32_t col;
  LyTracebackFrame *prev;
};

extern "C" {
void LyTraceback_Push(const char *filename, const char *funcname, std::int32_t line,
                      std::int32_t col);
void LyTraceback_Pop();
void LyTraceback_Clear();
void LyTraceback_Print(LyObject *exception);
std::int32_t LyEH_ReportUnhandled(LyObject *exception);
}
