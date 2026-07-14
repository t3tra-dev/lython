#pragma once

// Physical layout of a payload box slot: 16 i64 words per element. Words
// [4, 9) hold the pointer word of each physical memref (position i at
// kPointerWordBase + i), words [9, 14) the matching size words. The runtime
// support module (RuntimeSupportBuilder) and every lower* TU that probes or
// rebuilds boxed payloads must agree on these offsets; they are defined only
// here.

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include <cstdint>

namespace py::lowering::box_abi {

inline constexpr std::int64_t kWordsPerBox = 16;
inline constexpr std::int64_t kPointerWordBase = 4;
inline constexpr std::int64_t kSizeWordBase = 9;

inline mlir::MemRefType boxWordsType(mlir::Builder &builder) {
  return mlir::MemRefType::get({kWordsPerBox}, builder.getI64Type());
}

} // namespace py::lowering::box_abi
