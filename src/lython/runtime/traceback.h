#pragma once

#include "abi.h"

#include <cstdint>

extern "C" {
void LyTraceback_Push(std::uint8_t *file_allocated, std::uint8_t *file_aligned,
                      std::int64_t file_offset, std::int64_t file_size,
                      std::int64_t file_stride, std::uint8_t *func_allocated,
                      std::uint8_t *func_aligned, std::int64_t func_offset,
                      std::int64_t func_size, std::int64_t func_stride,
                      std::int32_t line, std::int32_t col);
void LyTraceback_Pop();
void LyTraceback_Clear();
void LyTraceback_PrintMessage(std::uint8_t *payload_allocated,
                              std::uint8_t *payload_aligned,
                              std::int64_t payload_offset,
                              std::int64_t payload_size,
                              std::int64_t payload_stride);
}
