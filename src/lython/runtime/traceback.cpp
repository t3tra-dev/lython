#include "traceback.h"

#include "objects/exception.h"
#include "objects/unicode.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

thread_local LyTracebackFrame *g_traceback_top = nullptr;

std::string readSourceLine(const char *filename, std::int32_t line) {
  if (!filename || line <= 0)
    return {};
  std::FILE *fp = std::fopen(filename, "r");
  if (!fp)
    return {};
  std::string result;
  std::int32_t current = 1;
  char buffer[512];
  while (std::fgets(buffer, sizeof(buffer), fp)) {
    if (current == line) {
      result = buffer;
      break;
    }
    ++current;
  }
  std::fclose(fp);
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
    result.pop_back();
  return result;
}

const char *safeStr(const char *value, const char *fallback) {
  return value ? value : fallback;
}

} // namespace

extern "C" {

void LyTraceback_Push(const char *filename, const char *funcname,
                      std::int32_t line, std::int32_t col) {
  auto *frame = new LyTracebackFrame();
  frame->filename = filename;
  frame->funcname = funcname;
  frame->line = line;
  frame->col = col;
  frame->prev = g_traceback_top;
  g_traceback_top = frame;
}

void LyTraceback_Pop() {
  if (!g_traceback_top)
    return;
  auto *top = g_traceback_top;
  g_traceback_top = top->prev;
  delete top;
}

void LyTraceback_Clear() {
  while (g_traceback_top)
    LyTraceback_Pop();
}

void LyTraceback_Print(LyObject *exception) {
  std::fprintf(stderr, "Traceback (most recent call last):\n");

  // Collect frames to print from oldest to newest.
  std::vector<LyTracebackFrame *> frames;
  for (auto *frame = g_traceback_top; frame; frame = frame->prev)
    frames.push_back(frame);
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    LyTracebackFrame *frame = *it;
    const char *filename = safeStr(frame->filename, "<unknown>");
    const char *funcname = safeStr(frame->funcname, "<unknown>");
    std::fprintf(stderr, "  File \"%s\", line %d, in %s\n", filename,
                 static_cast<int>(frame->line), funcname);
    std::string source = readSourceLine(filename, frame->line);
    if (!source.empty())
      std::fprintf(stderr, "    %s\n", source.c_str());
  }

  LyObject *exc = exception ? exception : LyException_GetCurrent();
  if (!exc) {
    std::fprintf(stderr, "Exception: <unknown>\n");
    return;
  }
  const char *typeName = exc->ob_type ? exc->ob_type->tp_name : "Exception";
  auto *excObj = reinterpret_cast<LyExceptionObject *>(exc);
  const char *msg =
      excObj->message ? LyUnicode_AsUTF8(reinterpret_cast<LyObject *>(excObj->message))
                      : "";
  if (msg && msg[0])
    std::fprintf(stderr, "%s: %s\n", typeName, msg);
  else
    std::fprintf(stderr, "%s\n", typeName);
}

std::int32_t LyEH_ReportUnhandled(LyObject *exception) {
  LyTraceback_Print(exception);
  LyException_Clear();
  LyTraceback_Clear();
  return 1;
}

} // extern "C"
