#pragma once

#include <cstdint>

struct LyI64Descriptor {
  std::int64_t *allocated = nullptr;
  std::int64_t *aligned = nullptr;
  std::int64_t offset = 0;
  std::int64_t size = 0;
  std::int64_t stride = 1;
};

struct LyI8Descriptor {
  std::uint8_t *allocated = nullptr;
  std::uint8_t *aligned = nullptr;
  std::int64_t offset = 0;
  std::int64_t size = 0;
  std::int64_t stride = 1;
};

struct LyExceptionPartsDescriptor {
  LyI64Descriptor header;
  LyI64Descriptor message_header;
  LyI8Descriptor message_bytes;
};

namespace lython::abi::memref {

inline LyI64Descriptor i64(std::int64_t *allocated, std::int64_t *aligned,
                           std::int64_t offset, std::int64_t size,
                           std::int64_t stride) {
  return {allocated, aligned, offset, size, stride};
}

inline LyI8Descriptor i8(std::uint8_t *allocated, std::uint8_t *aligned,
                         std::int64_t offset, std::int64_t size,
                         std::int64_t stride) {
  return {allocated, aligned, offset, size, stride};
}

inline bool readable(const LyI8Descriptor &descriptor) {
  return descriptor.aligned && descriptor.offset >= 0 && descriptor.size >= 0 &&
         descriptor.stride > 0;
}

inline bool root(const LyI64Descriptor &descriptor, std::int64_t size) {
  return descriptor.allocated && descriptor.aligned &&
         descriptor.allocated == descriptor.aligned && descriptor.offset == 0 &&
         descriptor.size == size && descriptor.stride == 1;
}

inline bool root(const LyI8Descriptor &descriptor, std::int64_t size) {
  return descriptor.allocated && descriptor.aligned &&
         descriptor.allocated == descriptor.aligned && descriptor.offset == 0 &&
         descriptor.size == size && descriptor.stride == 1;
}

inline bool rootDynamic(const LyI8Descriptor &descriptor) {
  if (descriptor.offset != 0 || descriptor.size < 0 || descriptor.stride != 1 ||
      descriptor.allocated != descriptor.aligned)
    return false;
  return descriptor.size == 0 || descriptor.allocated;
}

inline std::int64_t index(const LyI8Descriptor &descriptor,
                          std::int64_t logicalIndex) {
  return descriptor.offset + logicalIndex * descriptor.stride;
}

} // namespace lython::abi::memref

extern "C" {
void LyEH_ThrowException(
    std::int64_t *header_allocated, std::int64_t *header_aligned,
    std::int64_t header_offset, std::int64_t header_size,
    std::int64_t header_stride, std::int64_t *message_header_allocated,
    std::int64_t *message_header_aligned, std::int64_t message_header_offset,
    std::int64_t message_header_size, std::int64_t message_header_stride,
    std::uint8_t *message_bytes_allocated, std::uint8_t *message_bytes_aligned,
    std::int64_t message_bytes_offset, std::int64_t message_bytes_size,
    std::int64_t message_bytes_stride);
void LyEH_BeginCatch(void *exception_object);
bool LyEH_ClassIdMatches(std::int64_t exception_class_id,
                         std::int64_t handler_class_id);
std::int64_t LyEH_CurrentExceptionClassId();
bool LyEH_CurrentExceptionMatches(std::int64_t handler_class_id);
bool LyEH_DiscardCurrentExceptionIfMatches(std::int64_t handler_class_id);
void LyEH_DiscardCurrentException();
void LyEH_RethrowCurrent();
bool LyEH_TakeCurrentDescriptor(LyExceptionPartsDescriptor *descriptor);
}
