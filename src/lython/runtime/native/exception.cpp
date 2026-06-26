#include "abi.h"
#include "traceback.h"

#include <cstdlib>

extern "C" void *__cxa_begin_catch(void *);
extern "C" void __cxa_end_catch();

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
  bool nativeCatchActive = false;
  LyExceptionPartsDescriptor parts;
};

thread_local CurrentException g_current_exception;

void validateExceptionHeaderDescriptor(const LyI64Descriptor &header);

std::int64_t exceptionBaseClassId(std::int64_t classId) {
  switch (classId) {
  case 50:     // Exception
    return 5;  // BaseException
  case 62:     // asyncio.CancelledError
    return 5;  // BaseException
  case 51:     // RuntimeError
  case 52:     // TypeError
  case 53:     // ValueError
  case 56:     // AssertionError
  case 57:     // StopIteration
  case 58:     // StopAsyncIteration
  case 59:     // ArithmeticError
  case 60:     // LookupError
    return 50; // Exception
  case 54:     // KeyError
  case 55:     // IndexError
    return 60; // LookupError
  case 61:     // ZeroDivisionError
    return 59; // ArithmeticError
  default:
    return 0;
  }
}

std::int64_t exceptionClassId(const LyExceptionPartsDescriptor &descriptor) {
  constexpr std::int64_t kClassSlot = 2;
  validateExceptionHeaderDescriptor(descriptor.header);
  return descriptor.header.aligned[kClassSlot];
}

bool classIdMatches(std::int64_t exceptionClassId,
                    std::int64_t handlerClassId) {
  while (exceptionClassId != 0) {
    if (exceptionClassId == handlerClassId)
      return true;
    exceptionClassId = exceptionBaseClassId(exceptionClassId);
  }
  return false;
}

void releaseCurrentException() {
  if (g_current_exception.kind == CurrentExceptionKind::None)
    return;
  std::abort();
}

void endNativeCatchIfActive() {
  if (!g_current_exception.nativeCatchActive)
    return;
  __cxa_end_catch();
  g_current_exception.nativeCatchActive = false;
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

  endNativeCatchIfActive();
  *descriptor = g_current_exception.parts;
  g_current_exception = {};
  return true;
}

std::int64_t currentExceptionClassId() {
  if (g_current_exception.kind != CurrentExceptionKind::MemRefParts)
    return 0;
  return exceptionClassId(g_current_exception.parts);
}

void discardCurrentException() {
  endNativeCatchIfActive();
  g_current_exception = {};
  LyTraceback_Clear();
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

void LyEH_BeginCatch(void *exception_object) {
  if (!exception_object ||
      g_current_exception.kind != CurrentExceptionKind::MemRefParts)
    std::abort();
  __cxa_begin_catch(exception_object);
  g_current_exception.nativeCatchActive = true;
}

bool LyEH_ClassIdMatches(std::int64_t exception_class_id,
                         std::int64_t handler_class_id) {
  return classIdMatches(exception_class_id, handler_class_id);
}

std::int64_t LyEH_CurrentExceptionClassId() {
  return currentExceptionClassId();
}

bool LyEH_CurrentExceptionMatches(std::int64_t handler_class_id) {
  return classIdMatches(currentExceptionClassId(), handler_class_id);
}

bool LyEH_DiscardCurrentExceptionIfMatches(std::int64_t handler_class_id) {
  if (!LyEH_CurrentExceptionMatches(handler_class_id))
    return false;
  discardCurrentException();
  return true;
}

void LyEH_DiscardCurrentException() { discardCurrentException(); }

void LyEH_RethrowCurrent() {
  if (g_current_exception.kind == CurrentExceptionKind::None)
    std::abort();
  endNativeCatchIfActive();
  throw LyEHSignal{};
}

bool LyEH_TakeCurrentDescriptor(LyExceptionPartsDescriptor *descriptor) {
  return takeCurrentPartsDescriptor(descriptor);
}
}
