#include "objects/long.h"
#include "objects/bool.h"
#include "objects/float.h"
#include "objects/unicode.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Small integer cache - preallocated immortal integers
// Each entry needs space for the object header + 1 digit
struct SmallIntStorage {
  Ly_ssize_t ob_refcnt;
  LyTypeObject *ob_type;
  uintptr_t lv_tag;
  digit ob_digit[1];
};
static SmallIntStorage smallIntCache[kSmallIntCacheSize];
static bool smallIntCacheInitialized = false;

// Freelist for single-digit integers (most common allocation)
// Use a union to store next pointer in the freed object
struct FreelistNode {
  Ly_ssize_t ob_refcnt;  // Not used in freelist
  LyTypeObject *ob_type; // Keep type pointer for safety
  uintptr_t lv_tag;      // Not used in freelist
  FreelistNode *next;    // Next pointer (overlays ob_digit)
};

constexpr std::size_t kFreelistMaxSize = 4096;
static FreelistNode *singleDigitFreelist = nullptr;
static std::size_t singleDigitFreelistSize = 0;

// Pop from single-digit freelist (returns nullptr if empty)
static LyLongObject *freelistPop() {
  if (singleDigitFreelist == nullptr) {
    return nullptr;
  }
  FreelistNode *node = singleDigitFreelist;
  singleDigitFreelist = node->next;
  --singleDigitFreelistSize;
  return reinterpret_cast<LyLongObject *>(node);
}

// Push to single-digit freelist (returns false if full)
static bool freelistPush(LyLongObject *obj) {
  if (singleDigitFreelistSize >= kFreelistMaxSize) {
    return false;
  }
  auto *node = reinterpret_cast<FreelistNode *>(obj);
  node->next = singleDigitFreelist;
  singleDigitFreelist = node;
  ++singleDigitFreelistSize;
  return true;
}

// Convert stwodigits to string for debugging
std::string longToString(const LyLongObject *value) {
  if (value->isZero()) {
    return "0";
  }

  Ly_ssize_t ndigits = value->digitCount();

  // For compact values, use simple conversion
  if (value->isCompact()) {
    stwodigits val = medium_value(value);
    return std::to_string(val);
  }

  // For larger values, use the digit-by-digit conversion
  std::vector<digit> temp(ndigits);
  for (Ly_ssize_t i = 0; i < ndigits; ++i) {
    temp[i] = value->getDigit(i);
  }

  std::vector<std::uint32_t> decimalChunks;
  constexpr std::uint32_t chunkBase = 1000000000u;

  auto isZero = [](const std::vector<digit> &digits) {
    return std::all_of(digits.begin(), digits.end(),
                       [](digit d) { return d == 0; });
  };

  while (!isZero(temp)) {
    std::uint64_t remainder = 0;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(temp.size()) - 1;
         i >= 0; --i) {
      std::uint64_t current = (remainder << kDigitBits) + temp[i];
      temp[i] = static_cast<digit>(current / chunkBase);
      remainder = current % chunkBase;
    }
    decimalChunks.push_back(static_cast<std::uint32_t>(remainder));
  }

  std::string result;
  if (value->isNegative()) {
    result.push_back('-');
  }
  auto rit = decimalChunks.rbegin();
  if (rit != decimalChunks.rend()) {
    result += std::to_string(*rit++);
  }
  for (; rit != decimalChunks.rend(); ++rit) {
    std::string chunk = std::to_string(*rit);
    result.append(9 - chunk.size(), '0');
    result += chunk;
  }
  return result;
}

} // namespace

// Allocate a new LyLongObject with given number of digits
LyLongObject *LyLong_New(Ly_ssize_t ndigits) {
  // Ensure at least 1 digit for consistent behavior
  Ly_ssize_t alloc_digits = ndigits > 0 ? ndigits : 1;

  LyLongObject *obj = nullptr;

  // Try freelist for single-digit allocations
  if (alloc_digits == 1) {
    obj = freelistPop();
  }

  // Fall back to malloc if freelist is empty or larger allocation needed
  if (!obj) {
    std::size_t size = offsetof(LyLongObject, ob_digit) + alloc_digits * sizeof(digit);
    obj = static_cast<LyLongObject *>(std::malloc(size));
    if (!obj) {
      return nullptr;
    }
    obj->ob_type = &LyLong_Type();
  }

  obj->ob_refcnt = 1;
  obj->lv_tag = tagFromSignAndSize(ndigits != 0 ? 1 : 0, ndigits);
  obj->ob_digit[0] = 0; // Initialize first digit
  return obj;
}

void LyLong_InitSmallIntCache() {
  if (smallIntCacheInitialized) {
    return;
  }

  for (std::int64_t i = kSmallIntMin; i <= kSmallIntMax; ++i) {
    std::size_t idx = static_cast<std::size_t>(i - kSmallIntMin);
    SmallIntStorage &storage = smallIntCache[idx];

    // Immortal object - very high refcount
    storage.ob_refcnt = 1000000;
    storage.ob_type = &LyLong_Type();

    if (i == 0) {
      // Zero: sign tag = 1 (zero), 0 digits
      storage.lv_tag = tagFromSignAndSize(0, 0);
      storage.ob_digit[0] = 0;
    } else if (i > 0) {
      // Positive: sign tag = 0 (positive), 1 digit
      storage.lv_tag = tagFromSignAndSize(1, 1);
      storage.ob_digit[0] = static_cast<digit>(i);
    } else {
      // Negative: sign tag = 2 (negative), 1 digit
      storage.lv_tag = tagFromSignAndSize(-1, 1);
      storage.ob_digit[0] = static_cast<digit>(-i);
    }
  }
  smallIntCacheInitialized = true;
}

LyLongObject *LyLong_GetSmallInt(sdigit value) {
  if (!smallIntCacheInitialized) {
    LyLong_InitSmallIntCache();
  }
  return reinterpret_cast<LyLongObject *>(
      &smallIntCache[value - kSmallIntMin]);
}

bool LyLong_IsSmallInt(const LyLongObject *obj) {
  if (!obj || !smallIntCacheInitialized) {
    return false;
  }
  // Check if pointer falls within the cache array
  const auto *ptr = reinterpret_cast<const SmallIntStorage *>(obj);
  return ptr >= &smallIntCache[0] &&
         ptr < &smallIntCache[kSmallIntCacheSize];
}

bool LyLong_IsCompact(const LyLongObject *obj) {
  return obj && obj->isCompact();
}

extern "C" {

// Create LyLongObject from stwodigits (used for arithmetic results)
LyLongObject *LyLong_FromSTwoDigits(stwodigits value) {
  // Small int cache check
  if (IS_SMALL_INT(value)) {
    return LyLong_GetSmallInt(static_cast<sdigit>(value));
  }

  // Medium int (fits in single digit)
  if (is_medium_int(value)) {
    LyLongObject *obj = LyLong_New(1);
    if (!obj) {
      return nullptr;
    }
    int sign = value < 0 ? -1 : 1;
    digit abs_val = value < 0 ? static_cast<digit>(-value)
                              : static_cast<digit>(value);
    LyLong_SetSignAndDigitCount(obj, sign, 1);
    obj->ob_digit[0] = abs_val;
    return obj;
  }

  // Large int (requires 2 digits)
  LyLongObject *obj = LyLong_New(2);
  if (!obj) {
    return nullptr;
  }

  twodigits abs_val;
  int sign;
  if (value < 0) {
    sign = -1;
    // Handle INT64_MIN carefully
    abs_val = static_cast<twodigits>(-(value + 1)) + 1;
  } else {
    sign = 1;
    abs_val = static_cast<twodigits>(value);
  }

  LyLong_SetSignAndDigitCount(obj, sign, 2);
  obj->ob_digit[0] = static_cast<digit>(abs_val & kDigitMask);
  obj->ob_digit[1] = static_cast<digit>(abs_val >> kDigitBits);
  return obj;
}

LyLongObject *LyLong_FromI64(std::int64_t value) {
  return LyLong_FromSTwoDigits(static_cast<stwodigits>(value));
}

// Compare magnitudes of two LyLongObjects (ignoring sign)
static int compareMagnitude(const LyLongObject *lhs, const LyLongObject *rhs) {
  Ly_ssize_t lsize = lhs->digitCount();
  Ly_ssize_t rsize = rhs->digitCount();

  if (lsize != rsize) {
    return lsize < rsize ? -1 : 1;
  }

  for (Ly_ssize_t i = lsize - 1; i >= 0; --i) {
    digit ld = lhs->getDigit(i);
    digit rd = rhs->getDigit(i);
    if (ld != rd) {
      return ld < rd ? -1 : 1;
    }
  }
  return 0;
}

// Normalize: remove leading zeros from a long object (CPython-style)
static LyLongObject *long_normalize(LyLongObject *v) {
  Ly_ssize_t j = v->digitCount();
  Ly_ssize_t i = j;

  while (i > 0 && v->ob_digit[i - 1] == 0) {
    --i;
  }
  if (i != j) {
    if (i == 0) {
      LyLong_SetSignAndDigitCount(v, 0, 0);
    } else {
      LyLong_SetSignAndDigitCount(v, v->isNegative() ? -1 : 1, i);
    }
  }
  return v;
}

// Flip the sign of a long object in place (CPython-style)
static void LyLong_FlipSign(LyLongObject *v) {
  if (!v->isZero()) {
    // Toggle sign: if positive (tag bits 0), make negative (tag bits 2)
    // if negative (tag bits 2), make positive (tag bits 0)
    int newSign = v->isNegative() ? 1 : -1;
    LyLong_SetSignAndDigitCount(v, newSign, v->digitCount());
  }
}

// Add the absolute values of two integers (CPython x_add style)
static LyLongObject *x_add(const LyLongObject *a, const LyLongObject *b) {
  Ly_ssize_t size_a = a->digitCount();
  Ly_ssize_t size_b = b->digitCount();

  // Ensure a is the larger of the two
  if (size_a < size_b) {
    std::swap(a, b);
    std::swap(size_a, size_b);
  }

  LyLongObject *z = LyLong_New(size_a + 1);
  if (z == nullptr) {
    return nullptr;
  }

  digit carry = 0;
  Ly_ssize_t i;
  for (i = 0; i < size_b; ++i) {
    carry += a->ob_digit[i] + b->ob_digit[i];
    z->ob_digit[i] = carry & kDigitMask;
    carry >>= kDigitBits;
  }
  for (; i < size_a; ++i) {
    carry += a->ob_digit[i];
    z->ob_digit[i] = carry & kDigitMask;
    carry >>= kDigitBits;
  }
  z->ob_digit[i] = carry;
  return long_normalize(z);
}

// Subtract the absolute values of two integers (CPython x_sub style)
static LyLongObject *x_sub(const LyLongObject *a, const LyLongObject *b) {
  Ly_ssize_t size_a = a->digitCount();
  Ly_ssize_t size_b = b->digitCount();
  int sign = 1;

  // Ensure a is the larger of the two
  if (size_a < size_b) {
    sign = -1;
    std::swap(a, b);
    std::swap(size_a, size_b);
  } else if (size_a == size_b) {
    // Find highest digit where a and b differ
    Ly_ssize_t i = size_a;
    while (--i >= 0 && a->ob_digit[i] == b->ob_digit[i])
      ;
    if (i < 0) {
      return LyLong_GetSmallInt(0);
    }
    if (a->ob_digit[i] < b->ob_digit[i]) {
      sign = -1;
      std::swap(a, b);
    }
    size_a = size_b = i + 1;
  }

  LyLongObject *z = LyLong_New(size_a);
  if (z == nullptr) {
    return nullptr;
  }

  digit borrow = 0;
  for (Ly_ssize_t i = 0; i < size_b; ++i) {
    // Using unsigned arithmetic for borrow detection
    borrow = a->ob_digit[i] - b->ob_digit[i] - borrow;
    z->ob_digit[i] = borrow & kDigitMask;
    borrow >>= kDigitBits;
    borrow &= 1; // Keep only one sign bit
  }
  for (Ly_ssize_t i = size_b; i < size_a; ++i) {
    borrow = a->ob_digit[i] - borrow;
    z->ob_digit[i] = borrow & kDigitMask;
    borrow >>= kDigitBits;
    borrow &= 1;
  }

  LyLongObject *result = long_normalize(z);
  if (sign < 0 && !result->isZero()) {
    LyLong_FlipSign(result);
  }
  return result;
}

// Slow path for long_add (called from inline fast-path in header)
LyLongObject *LyLong_Add_Slow(const LyLongObject *a, const LyLongObject *b) {
  LyLongObject *z;
  if (a->isNegative()) {
    if (b->isNegative()) {
      // (-a) + (-b) = -(a + b)
      z = x_add(a, b);
      if (z != nullptr && !z->isZero()) {
        LyLong_FlipSign(z);
      }
    } else {
      // (-a) + b = b - a
      z = x_sub(b, a);
    }
  } else {
    if (b->isNegative()) {
      // a + (-b) = a - b
      z = x_sub(a, b);
    } else {
      // a + b
      z = x_add(a, b);
    }
  }
  return z;
}

// Slow path for long_sub (called from inline fast-path in header)
LyLongObject *LyLong_Sub_Slow(const LyLongObject *a, const LyLongObject *b) {
  LyLongObject *z;
  if (a->isNegative()) {
    if (b->isNegative()) {
      // (-a) - (-b) = b - a
      z = x_sub(b, a);
    } else {
      // (-a) - b = -(a + b)
      z = x_add(a, b);
      if (z != nullptr && !z->isZero()) {
        LyLong_FlipSign(z);
      }
    }
  } else {
    if (b->isNegative()) {
      // a - (-b) = a + b
      z = x_add(a, b);
    } else {
      // a - b
      z = x_sub(a, b);
    }
  }
  return z;
}

int LyLong_Compare(const LyLongObject *lhs, const LyLongObject *rhs) {
  // Fast path: both compact
  if (LyLong_BothAreCompact(lhs, rhs)) {
    stwodigits lval = medium_value(lhs);
    stwodigits rval = medium_value(rhs);
    if (lval < rval) return -1;
    if (lval > rval) return 1;
    return 0;
  }

  // Different signs
  int lsign = lhs->sign();
  int rsign = rhs->sign();
  if (lsign != rsign) {
    return lsign < rsign ? -1 : 1;
  }

  // Same sign: compare magnitudes
  int cmp = compareMagnitude(lhs, rhs);
  if (cmp == 0) return 0;
  // If negative, reverse comparison
  return lsign < 0 ? -cmp : cmp;
}

double LyLong_AsDouble(const LyLongObject *value) {
  if (value->isCompact()) {
    return static_cast<double>(medium_value(value));
  }

  double result = 0.0;
  Ly_ssize_t ndigits = value->digitCount();
  for (Ly_ssize_t i = ndigits - 1; i >= 0; --i) {
    result *= static_cast<double>(kDigitBase);
    result += static_cast<double>(value->getDigit(i));
  }
  return value->isNegative() ? -result : result;
}

LyObject *LyNumber_Add(LyObject *lhs, LyObject *rhs) {
  if (!lhs || !rhs) {
    return nullptr;
  }

  auto *longType = &LyLong_Type();
  auto *floatType = &LyFloat_Type();

  // int + int
  if (lhs->ob_type == longType && rhs->ob_type == longType) {
    auto *result = LyLong_Add(reinterpret_cast<LyLongObject *>(lhs),
                              reinterpret_cast<LyLongObject *>(rhs));
    return reinterpret_cast<LyObject *>(result);
  }

  // float + any or any + float
  if (lhs->ob_type == floatType || rhs->ob_type == floatType) {
    double left = (lhs->ob_type == floatType)
                      ? reinterpret_cast<LyFloatObject *>(lhs)->value
                      : LyLong_AsDouble(reinterpret_cast<LyLongObject *>(lhs));
    double right = (rhs->ob_type == floatType)
                       ? reinterpret_cast<LyFloatObject *>(rhs)->value
                       : LyLong_AsDouble(reinterpret_cast<LyLongObject *>(rhs));
    return reinterpret_cast<LyObject *>(LyFloat_FromDouble(left + right));
  }

  return nullptr;
}

LyObject *LyNumber_Sub(LyObject *lhs, LyObject *rhs) {
  if (!lhs || !rhs) {
    return nullptr;
  }

  auto *longType = &LyLong_Type();
  auto *floatType = &LyFloat_Type();

  // int - int
  if (lhs->ob_type == longType && rhs->ob_type == longType) {
    auto *result = LyLong_Sub(reinterpret_cast<LyLongObject *>(lhs),
                              reinterpret_cast<LyLongObject *>(rhs));
    return reinterpret_cast<LyObject *>(result);
  }

  // float - any or any - float
  if (lhs->ob_type == floatType || rhs->ob_type == floatType) {
    double left = (lhs->ob_type == floatType)
                      ? reinterpret_cast<LyFloatObject *>(lhs)->value
                      : LyLong_AsDouble(reinterpret_cast<LyLongObject *>(lhs));
    double right = (rhs->ob_type == floatType)
                       ? reinterpret_cast<LyFloatObject *>(rhs)->value
                       : LyLong_AsDouble(reinterpret_cast<LyLongObject *>(rhs));
    return reinterpret_cast<LyObject *>(LyFloat_FromDouble(left - right));
  }

  return nullptr;
}

LyObject *LyNumber_Le(LyObject *lhs, LyObject *rhs) {
  if (!lhs || !rhs) {
    return nullptr;
  }

  auto *longType = &LyLong_Type();
  auto *floatType = &LyFloat_Type();

  // int <= int
  if (lhs->ob_type == longType && rhs->ob_type == longType) {
    int cmp = LyLong_Compare(reinterpret_cast<LyLongObject *>(lhs),
                             reinterpret_cast<LyLongObject *>(rhs));
    return reinterpret_cast<LyObject *>(LyBool_FromBool(cmp <= 0));
  }

  // float <= any or any <= float
  if (lhs->ob_type == floatType || rhs->ob_type == floatType) {
    double left = (lhs->ob_type == floatType)
                      ? reinterpret_cast<LyFloatObject *>(lhs)->value
                      : LyLong_AsDouble(reinterpret_cast<LyLongObject *>(lhs));
    double right = (rhs->ob_type == floatType)
                       ? reinterpret_cast<LyFloatObject *>(rhs)->value
                       : LyLong_AsDouble(reinterpret_cast<LyLongObject *>(rhs));
    return reinterpret_cast<LyObject *>(LyBool_FromBool(left <= right));
  }

  return nullptr;
}

// Exported wrapper for inline LyLong_Add (for JIT/AOT linking)
LyLongObject *LyLong_Add(const LyLongObject *a, const LyLongObject *b) {
  if (LyLong_BothAreCompact(a, b)) {
    stwodigits z = medium_value(a) + medium_value(b);
    return LyLong_FromSTwoDigits(z);
  }
  return LyLong_Add_Slow(a, b);
}

// Exported wrapper for inline LyLong_Sub (for JIT/AOT linking)
LyLongObject *LyLong_Sub(const LyLongObject *a, const LyLongObject *b) {
  if (LyLong_BothAreCompact(a, b)) {
    stwodigits z = medium_value(a) - medium_value(b);
    return LyLong_FromSTwoDigits(z);
  }
  return LyLong_Sub_Slow(a, b);
}

} // extern "C"

void LyLong_Dealloc(LyObject *object) {
  if (!object) {
    return;
  }
  // Don't free cached small integers (they're immortal)
  auto *longObj = reinterpret_cast<LyLongObject *>(object);
  if (LyLong_IsSmallInt(longObj)) {
    return;
  }

  // Try to add single-digit objects to freelist
  if (longObj->digitCount() <= 1) {
    if (freelistPush(longObj)) {
      return; // Successfully added to freelist
    }
  }

  // Fall back to free
  std::free(longObj);
}

LyUnicodeObject *LyLong_Repr(LyObject *object) {
  auto *longObj = reinterpret_cast<LyLongObject *>(object);
  std::string text = longToString(longObj);
  return LyUnicode_FromUTF8(text.c_str(), text.size());
}
