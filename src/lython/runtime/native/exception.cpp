#include "abi.h"

#include <cstdlib>

namespace {

constexpr std::int64_t kObjectHeaderSlots = 2;
constexpr std::int64_t kExceptionHeaderSlots = 3;

enum class CurrentExceptionKind {
  None,
  MemRefParts,
};

struct LyEHSignal {};

struct CurrentException {
  CurrentExceptionKind kind = CurrentExceptionKind::None;
  LyExceptionPartsDescriptor parts;
};

thread_local CurrentException g_current_exception;

void releaseCurrentException() {
  if (g_current_exception.kind == CurrentExceptionKind::None)
    return;
  std::abort();
}

void validateObjectHeaderDescriptor(const LyI64Descriptor &header) {
  if (!lython::abi::memref::root(header, kObjectHeaderSlots))
    std::abort();
}

void validateExceptionHeaderDescriptor(const LyI64Descriptor &header) {
  if (!lython::abi::memref::root(header, kExceptionHeaderSlots))
    std::abort();
}

void validateBytesDescriptor(const LyI8Descriptor &bytes) {
  if (!lython::abi::memref::rootDynamic(bytes))
    std::abort();
}

LyExceptionPartsDescriptor makePartsDescriptor(
    std::int64_t *header_allocated, std::int64_t *header_aligned,
    std::int64_t header_offset, std::int64_t header_size,
    std::int64_t header_stride, std::int64_t *message_header_allocated,
    std::int64_t *message_header_aligned, std::int64_t message_header_offset,
    std::int64_t message_header_size, std::int64_t message_header_stride,
    std::uint8_t *message_bytes_allocated, std::uint8_t *message_bytes_aligned,
    std::int64_t message_bytes_offset, std::int64_t message_bytes_size,
    std::int64_t message_bytes_stride) {
  LyExceptionPartsDescriptor descriptor{
      lython::abi::memref::i64(header_allocated, header_aligned, header_offset,
                               header_size, header_stride),
      lython::abi::memref::i64(message_header_allocated, message_header_aligned,
                               message_header_offset, message_header_size,
                               message_header_stride),
      lython::abi::memref::i8(message_bytes_allocated, message_bytes_aligned,
                              message_bytes_offset, message_bytes_size,
                              message_bytes_stride),
  };
  validateExceptionHeaderDescriptor(descriptor.header);
  validateObjectHeaderDescriptor(descriptor.message_header);
  validateBytesDescriptor(descriptor.message_bytes);
  return descriptor;
}

void setCurrentExceptionParts(LyExceptionPartsDescriptor descriptor) {
  releaseCurrentException();
  g_current_exception.parts = descriptor;
  g_current_exception.kind = CurrentExceptionKind::MemRefParts;
}

bool takeCurrentPartsDescriptor(LyExceptionPartsDescriptor *descriptor) {
  if (!descriptor ||
      g_current_exception.kind != CurrentExceptionKind::MemRefParts)
    return false;

  *descriptor = g_current_exception.parts;
  g_current_exception = {};
  return true;
}

} // namespace

extern "C" {

void LyEH_ThrowException(
    std::int64_t *header_allocated, std::int64_t *header_aligned,
    std::int64_t header_offset, std::int64_t header_size,
    std::int64_t header_stride, std::int64_t *message_header_allocated,
    std::int64_t *message_header_aligned, std::int64_t message_header_offset,
    std::int64_t message_header_size, std::int64_t message_header_stride,
    std::uint8_t *message_bytes_allocated, std::uint8_t *message_bytes_aligned,
    std::int64_t message_bytes_offset, std::int64_t message_bytes_size,
    std::int64_t message_bytes_stride) {
  setCurrentExceptionParts(makePartsDescriptor(
      header_allocated, header_aligned, header_offset, header_size,
      header_stride, message_header_allocated, message_header_aligned,
      message_header_offset, message_header_size, message_header_stride,
      message_bytes_allocated, message_bytes_aligned, message_bytes_offset,
      message_bytes_size, message_bytes_stride));
  throw LyEHSignal{};
}

void LyEH_RethrowCurrent() {
  if (g_current_exception.kind == CurrentExceptionKind::None)
    std::abort();
  throw LyEHSignal{};
}

bool LyEH_TakeCurrentDescriptor(LyExceptionPartsDescriptor *descriptor) {
  return takeCurrentPartsDescriptor(descriptor);
}
}
