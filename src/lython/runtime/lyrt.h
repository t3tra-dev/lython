#pragma once

#include <cstdint>

#include "abi.h"
#include "traceback.h"

extern "C" {
void LyHost_Print(std::uint8_t *allocated, std::uint8_t *aligned,
                  std::int64_t offset, std::int64_t size, std::int64_t stride,
                  std::int64_t length);
void LyHost_PrintLine(std::uint8_t *allocated, std::uint8_t *aligned,
                      std::int64_t offset, std::int64_t size,
                      std::int64_t stride, std::int64_t length);
double LyFloat_Round(double value, std::int64_t ndigits);
std::int64_t LyFloat_RoundToI64(double value);
std::int64_t LyInt_Round(std::int64_t value, std::int64_t ndigits);
void LyRt_InstallStackGuard();
} // extern "C"
