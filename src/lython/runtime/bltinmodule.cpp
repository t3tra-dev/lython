#include "lyrt.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

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

double LyFloat_Round(double value, std::int64_t ndigits) {
  if (!std::isfinite(value))
    std::abort();
  if (ndigits > 308 || ndigits < -308)
    return value;
  std::uint64_t magnitude = ndigits < 0
                                ? static_cast<std::uint64_t>(-(ndigits + 1)) + 1
                                : static_cast<std::uint64_t>(ndigits);
  double scale = std::pow(10.0, static_cast<double>(magnitude));
  if (!std::isfinite(scale) || scale == 0.0)
    std::abort();
  if (ndigits >= 0)
    return std::nearbyint(value * scale) / scale;
  return std::nearbyint(value / scale) * scale;
}

std::int64_t LyFloat_RoundToI64(double value) {
  if (!std::isfinite(value))
    std::abort();
  double rounded = std::nearbyint(value);
  if (rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max()))
    std::abort();
  return static_cast<std::int64_t>(rounded);
}

std::int64_t LyInt_Round(std::int64_t value, std::int64_t ndigits) {
  if (ndigits >= 0)
    return value;

  std::uint64_t magnitude = static_cast<std::uint64_t>(-(ndigits + 1)) + 1;
  if (magnitude > 19)
    return 0;
  std::uint64_t scale = 1;
  for (std::uint64_t i = 0; i < magnitude; ++i) {
    if (scale > std::numeric_limits<std::uint64_t>::max() / 10)
      return 0;
    scale *= 10;
  }

  std::uint64_t absValue = value >= 0
                               ? static_cast<std::uint64_t>(value)
                               : static_cast<std::uint64_t>(-(value + 1)) + 1;
  if (scale == 0)
    std::abort();

  std::uint64_t quotient = absValue / scale;
  std::uint64_t remainder = absValue % scale;
  std::uint64_t roundedQuotient = quotient;
  if (remainder > scale - remainder) {
    ++roundedQuotient;
  } else if (remainder == scale - remainder && (quotient & 1) != 0) {
    ++roundedQuotient;
  }

  if (roundedQuotient != 0 &&
      scale >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
              roundedQuotient) {
    if (!(value < 0 && roundedQuotient == 1 &&
          scale == (static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) +
                    1))) {
      std::abort();
    }
  }

  std::uint64_t roundedAbs = roundedQuotient * scale;
  if (value >= 0) {
    if (roundedAbs >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      std::abort();
    return static_cast<std::int64_t>(roundedAbs);
  }
  if (roundedAbs ==
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1)
    return std::numeric_limits<std::int64_t>::min();
  if (roundedAbs >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    std::abort();
  return -static_cast<std::int64_t>(roundedAbs);
}
}
