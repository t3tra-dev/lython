#pragma once

#include "objects/object.h"

#include <cstdint>
#include <cstddef>

// 30-bit digit (same as CPython)
using digit = std::uint32_t;
using sdigit = std::int32_t;
using twodigits = std::uint64_t;
using stwodigits = std::int64_t;

constexpr int kDigitBits = 30;
constexpr digit kDigitBase = 1u << kDigitBits;
constexpr digit kDigitMask = kDigitBase - 1;

// Tag encoding (CPython-style):
// - Bits 0-1: sign (0 = positive, 1 = zero, 2 = negative)
// - Bit 2: reserved (immortality in CPython, unused here for now)
// - Bits 3+: number of digits
constexpr int kNonSizeBits = 3;

// TAG_FROM_SIGN_AND_SIZE: sign is -1, 0, or 1
// (1 - sign) maps: -1 -> 2 (negative), 0 -> 1 (zero), 1 -> 0 (positive)
inline uintptr_t tagFromSignAndSize(int sign, Ly_ssize_t size) {
  return static_cast<uintptr_t>(1 - sign) |
         (static_cast<uintptr_t>(size) << kNonSizeBits);
}

// CPython-compatible structure with flexible array member
struct LyLongObject {
  // Base object fields
  Ly_ssize_t ob_refcnt;
  LyTypeObject *ob_type;
  // Long-specific fields
  uintptr_t lv_tag;      // sign + digit count encoded
  digit ob_digit[];      // Flexible array member - digits stored inline

  // Get number of digits
  Ly_ssize_t digitCount() const {
    return static_cast<Ly_ssize_t>(lv_tag >> kNonSizeBits);
  }

  // Get sign from tag: 0 = positive, 1 = zero, 2 = negative
  int signTag() const { return static_cast<int>(lv_tag & 0x3); }

  // Get sign as -1, 0, or 1
  int sign() const { return 1 - signTag(); }

  // Check if zero
  bool isZero() const { return signTag() == 1; }

  // Check if negative
  bool isNegative() const { return signTag() == 2; }

  // Check if compact (0 or 1 digits, fits in 30-bit signed)
  bool isCompact() const { return lv_tag < (2u << kNonSizeBits); }

  // Get digit at index (no bounds checking for speed)
  digit getDigit(Ly_ssize_t i) const { return ob_digit[i]; }

  // Set digit at index
  void setDigit(Ly_ssize_t i, digit d) { ob_digit[i] = d; }
};

// Check if two LyLongObjects are both compact
inline bool LyLong_BothAreCompact(const LyLongObject *a,
                                   const LyLongObject *b) {
  return (a->lv_tag | b->lv_tag) < (2u << kNonSizeBits);
}

// Get compact value as stwodigits (only valid if isCompact())
// For 0 digits: returns 0
// For 1 digit: returns signed value
inline stwodigits medium_value(const LyLongObject *op) {
  Ly_ssize_t ndigits = op->digitCount();
  if (ndigits == 0) {
    return 0;
  }
  stwodigits val = static_cast<stwodigits>(op->ob_digit[0]);
  return op->isNegative() ? -val : val;
}

// Check if a stwodigits value fits in medium representation (single digit)
// Medium int range: [-kDigitMask, kDigitMask] = [-2^30+1, 2^30-1]
inline bool is_medium_int(stwodigits x) {
  // CPython's trick: (x + MASK) < (MASK + BASE) checks -MASK <= x <= MASK
  twodigits x_plus_mask = static_cast<twodigits>(x) + kDigitMask;
  return x_plus_mask < (static_cast<twodigits>(kDigitMask) + kDigitBase);
}

// Small integer cache range (same as CPython: -5 to 256)
constexpr std::int64_t kSmallIntMin = -5;
constexpr std::int64_t kSmallIntMax = 256;
constexpr std::size_t kSmallIntCacheSize = kSmallIntMax - kSmallIntMin + 1;

// Check if value is in small int cache range
inline bool IS_SMALL_INT(stwodigits x) {
  return x >= kSmallIntMin && x <= kSmallIntMax;
}

// Initialize small integer cache (call once at startup)
void LyLong_InitSmallIntCache();

// Get cached small integer (must be in range)
LyLongObject *LyLong_GetSmallInt(sdigit value);

// Check if a value is a cached small integer (immortal)
bool LyLong_IsSmallInt(const LyLongObject *obj);

// Check if value fits in compact representation
bool LyLong_IsCompact(const LyLongObject *obj);

// Allocate a new LyLongObject with given number of digits
LyLongObject *LyLong_New(Ly_ssize_t ndigits);

// Set sign and digit count
inline void LyLong_SetSignAndDigitCount(LyLongObject *op, int sign,
                                         Ly_ssize_t size) {
  op->lv_tag = tagFromSignAndSize(sign, size);
}

extern "C" {

LyLongObject *LyLong_FromI64(std::int64_t value);
LyLongObject *LyLong_FromSTwoDigits(stwodigits value);
LyLongObject *LyLong_Add_Slow(const LyLongObject *lhs, const LyLongObject *rhs);
LyLongObject *LyLong_Sub_Slow(const LyLongObject *lhs, const LyLongObject *rhs);
int LyLong_Compare(const LyLongObject *lhs, const LyLongObject *rhs);
double LyLong_AsDouble(const LyLongObject *value);
LyObject *LyNumber_Add(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Sub(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Le(LyObject *lhs, LyObject *rhs);
// Exported wrappers for JIT/AOT linking (also available as inline in header)
LyLongObject *LyLong_Add(const LyLongObject *a, const LyLongObject *b);
LyLongObject *LyLong_Sub(const LyLongObject *a, const LyLongObject *b);
}


// Inline fast-path for compact integer comparison (<=)
__attribute__((always_inline))
inline int LyLong_Compare_Inline(const LyLongObject *a, const LyLongObject *b) {
  if (LyLong_BothAreCompact(a, b)) {
    stwodigits va = medium_value(a);
    stwodigits vb = medium_value(b);
    return (va > vb) - (va < vb);
  }
  return LyLong_Compare(a, b);
}

void LyLong_Dealloc(LyObject *object);
LyUnicodeObject *LyLong_Repr(LyObject *object);
