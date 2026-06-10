#include "traceback.h"

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
  case 1:
    return "BaseException";
  case 2:
    return "Exception";
  case 3:
    return "RuntimeError";
  case 4:
    return "TypeError";
  case 5:
    return "ValueError";
  case 6:
    return "KeyError";
  case 7:
    return "IndexError";
  case 8:
    return "AssertionError";
  default:
    return "Exception";
  }
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
  g_traceback_stack.push_back(
      {copyMemRefString(file), copyMemRefString(function), line, col});
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

  for (const LyTracebackFrame &frame : g_traceback_stack) {
    const char *filename = safeStr(frame.filename, "<unknown>");
    const char *funcname = safeStr(frame.funcname, "<unknown>");
    std::fprintf(stderr, "  File \"%s\", line %d, in %s\n", filename,
                 static_cast<int>(frame.line), funcname);
    std::string source = readSourceLine(frame.filename, frame.line);
    if (!source.empty())
      std::fprintf(stderr, "    %s\n", source.c_str());
  }

  LyI8Descriptor payload =
      lython::abi::memref::i8(payload_allocated, payload_aligned,
                              payload_offset, payload_size, payload_stride);
  printExceptionSummary(class_id, payload);
}

} // extern "C"
