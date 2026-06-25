#include "traceback.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct LyTracebackFrame {
  std::string filename;
  std::string funcname;
  std::int32_t line;
  std::int32_t col;
  std::int32_t endLine;
  std::int32_t endCol;
  bool highlight;
};

thread_local std::vector<LyTracebackFrame> g_traceback_stack;

std::string copyMemRefString(const LyI8Descriptor &descriptor) {
  if (!lython::abi::memref::readable(descriptor) || descriptor.size == 0)
    return {};
  std::string result;
  result.reserve(static_cast<std::size_t>(descriptor.size));
  for (std::int64_t i = 0; i < descriptor.size; ++i) {
    std::int64_t index = lython::abi::memref::index(descriptor, i);
    result.push_back(static_cast<char>(descriptor.aligned[index]));
  }
  return result;
}

std::string readSourceLine(const std::string &filename, std::int32_t line) {
  if (filename.empty() || line <= 0)
    return {};
  std::FILE *fp = std::fopen(filename.c_str(), "r");
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

const char *safeStr(const std::string &value, const char *defaultValue) {
  return value.empty() ? defaultValue : value.c_str();
}

const char *exceptionClassName(std::int64_t classId) {
  switch (classId) {
  case 5:
    return "BaseException";
  case 50:
    return "Exception";
  case 51:
    return "RuntimeError";
  case 52:
    return "TypeError";
  case 53:
    return "ValueError";
  case 54:
    return "KeyError";
  case 55:
    return "IndexError";
  case 56:
    return "AssertionError";
  case 57:
    return "StopIteration";
  case 58:
    return "StopAsyncIteration";
  default:
    return "Exception";
  }
}

std::int64_t exceptionClassId(const LyExceptionPartsDescriptor &descriptor) {
  constexpr std::int64_t kExceptionHeaderSlots = 3;
  constexpr std::int64_t kClassSlot = 2;
  if (!lython::abi::memref::root(descriptor.header, kExceptionHeaderSlots))
    return 0;
  return descriptor.header.aligned[kClassSlot];
}

void printExceptionSummary(std::int64_t classId,
                           const LyI8Descriptor &payload) {
  const char *className = exceptionClassName(classId);
  if (!lython::abi::memref::rootDynamic(payload)) {
    std::fprintf(stderr, "%s: <invalid>\n", className);
    return;
  }

  if (payload.size == 0) {
    std::fprintf(stderr, "%s\n", className);
    return;
  }

  if (!payload.aligned) {
    std::fprintf(stderr, "%s: <unknown>\n", className);
    return;
  }

  std::string message = copyMemRefString(payload);
  std::fprintf(stderr, "%s: %.*s\n", className,
               static_cast<int>(message.size()), message.data());
}

std::size_t markerStartColumn(const std::string &source, std::int32_t col) {
  if (col > 0 && static_cast<std::size_t>(col) < source.size())
    return static_cast<std::size_t>(col);
  for (std::size_t index = 0; index < source.size(); ++index)
    if (source[index] != ' ' && source[index] != '\t')
      return index;
  return 0;
}

void printSourceMarker(const std::string &source, std::int32_t col,
                       std::int32_t endCol) {
  if (source.empty())
    return;
  std::size_t start = markerStartColumn(source, col);
  if (start >= source.size())
    return;

  std::fprintf(stderr, "    ");
  for (std::size_t index = 0; index < start; ++index)
    std::fputc(source[index] == '\t' ? '\t' : ' ', stderr);
  std::size_t end = source.size();
  if (endCol > col && endCol > 0)
    end =
        std::min<std::size_t>(static_cast<std::size_t>(endCol), source.size());
  if (end <= start)
    end = std::min<std::size_t>(start + 1, source.size());
  std::size_t count = end - start;
  if (count <= 2) {
    for (std::size_t index = 0; index < count; ++index)
      std::fputc('^', stderr);
  } else {
    for (std::size_t index = 0; index + 2 < count; ++index)
      std::fputc('~', stderr);
    std::fprintf(stderr, "^^");
  }
  std::fprintf(stderr, "\n");
}

std::size_t leadingWhitespace(const std::string &source) {
  std::size_t index = 0;
  while (index < source.size() &&
         (source[index] == ' ' || source[index] == '\t'))
    ++index;
  return index;
}

} // namespace

extern "C" {

void LyTraceback_Push(std::uint8_t *file_allocated, std::uint8_t *file_aligned,
                      std::int64_t file_offset, std::int64_t file_size,
                      std::int64_t file_stride, std::uint8_t *func_allocated,
                      std::uint8_t *func_aligned, std::int64_t func_offset,
                      std::int64_t func_size, std::int64_t func_stride,
                      std::int32_t line, std::int32_t col) {
  LyI8Descriptor file = lython::abi::memref::i8(
      file_allocated, file_aligned, file_offset, file_size, file_stride);
  LyI8Descriptor function = lython::abi::memref::i8(
      func_allocated, func_aligned, func_offset, func_size, func_stride);
  g_traceback_stack.push_back({copyMemRefString(file),
                               copyMemRefString(function), line, col, 0, 0,
                               false});
}

void LyTraceback_PushCString(const char *file, const char *func,
                             std::int32_t line, std::int32_t col) {
  LyTraceback_PushCStringRange(file, func, line, col, line, 0);
}

void LyTraceback_PushCStringRange(const char *file, const char *func,
                                  std::int32_t line, std::int32_t col,
                                  std::int32_t end_line, std::int32_t end_col) {
  g_traceback_stack.push_back(
      {file ? file : "", func ? func : "", line, col, end_line, end_col, true});
}

void LyTraceback_Pop() {
  if (g_traceback_stack.empty())
    return;
  g_traceback_stack.pop_back();
}

void LyTraceback_Clear() { g_traceback_stack.clear(); }

void LyTraceback_PrintMessage(std::int64_t class_id,
                              std::uint8_t *payload_allocated,
                              std::uint8_t *payload_aligned,
                              std::int64_t payload_offset,
                              std::int64_t payload_size,
                              std::int64_t payload_stride) {
  std::cout.flush();
  std::fflush(stdout);

  std::fprintf(stderr, "Traceback (most recent call last):\n");

  for (auto it = g_traceback_stack.rbegin(); it != g_traceback_stack.rend();
       ++it) {
    const LyTracebackFrame &frame = *it;
    const char *filename = safeStr(frame.filename, "<unknown>");
    const char *funcname = safeStr(frame.funcname, "<unknown>");
    std::fprintf(stderr, "  File \"%s\", line %d, in %s\n", filename,
                 static_cast<int>(frame.line), funcname);
    std::string source = readSourceLine(frame.filename, frame.line);
    if (!source.empty()) {
      std::size_t stripped = leadingWhitespace(source);
      std::string displaySource = source.substr(stripped);
      std::int32_t displayCol = frame.col;
      if (displayCol >= static_cast<std::int32_t>(stripped))
        displayCol -= static_cast<std::int32_t>(stripped);
      else
        displayCol = 0;
      std::int32_t displayEndCol = 0;
      if (frame.endLine == frame.line && frame.endCol > 0) {
        displayEndCol = frame.endCol;
        if (displayEndCol >= static_cast<std::int32_t>(stripped))
          displayEndCol -= static_cast<std::int32_t>(stripped);
        else
          displayEndCol = 0;
      }
      std::fprintf(stderr, "    %s\n", displaySource.c_str());
      if (frame.highlight)
        printSourceMarker(displaySource, displayCol, displayEndCol);
    }
  }

  LyI8Descriptor payload =
      lython::abi::memref::i8(payload_allocated, payload_aligned,
                              payload_offset, payload_size, payload_stride);
  printExceptionSummary(class_id, payload);
}

int LyRunPythonMain(void (*entry)()) {
  if (!entry)
    return 1;
  try {
    entry();
    return 0;
  } catch (...) {
    LyExceptionPartsDescriptor descriptor;
    if (!LyEH_TakeCurrentDescriptor(&descriptor)) {
      std::fprintf(
          stderr, "error: uncaught native exception during Python execution\n");
      LyTraceback_Clear();
      return 1;
    }

    LyTraceback_PrintMessage(
        exceptionClassId(descriptor), descriptor.message_bytes.allocated,
        descriptor.message_bytes.aligned, descriptor.message_bytes.offset,
        descriptor.message_bytes.size, descriptor.message_bytes.stride);
    LyTraceback_Clear();
    return 1;
  }
}

} // extern "C"
