#include "lyrt.h"

#include <cstdlib>
#include <iostream>

extern "C" {

static void printBytes(std::uint8_t *allocated, std::uint8_t *aligned,
                       std::int64_t offset, std::int64_t size,
                       std::int64_t stride, std::int64_t length) {
  LyI8Descriptor bytes =
      lython::abi::memref::i8(allocated, aligned, offset, size, stride);
  if (length < 0)
    std::abort();
  if (length > 0) {
    if (!lython::abi::memref::readable(bytes) || length > bytes.size)
      std::abort();
    if (bytes.stride == 1) {
      const char *start =
          reinterpret_cast<const char *>(bytes.aligned + bytes.offset);
      std::cout.write(start, length);
    } else {
      for (std::int64_t i = 0; i < length; ++i)
        std::cout.put(static_cast<char>(
            bytes.aligned[lython::abi::memref::index(bytes, i)]));
    }
  }
}

void LyHost_Print(std::uint8_t *allocated, std::uint8_t *aligned,
                  std::int64_t offset, std::int64_t size, std::int64_t stride,
                  std::int64_t length) {
  printBytes(allocated, aligned, offset, size, stride, length);
}

void LyHost_PrintLine(std::uint8_t *allocated, std::uint8_t *aligned,
                      std::int64_t offset, std::int64_t size,
                      std::int64_t stride, std::int64_t length) {
  printBytes(allocated, aligned, offset, size, stride, length);
  std::cout << std::endl;
}
}
