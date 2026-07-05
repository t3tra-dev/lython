#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::native {

inline constexpr llvm::StringLiteral kTargetTripleAttr{"ly.target.triple"};
inline constexpr llvm::StringLiteral kTargetPointerWidthAttr{
    "ly.target.pointer_width"};
inline constexpr llvm::StringLiteral kTargetCLongWidthAttr{
    "ly.target.c_long_width"};

inline constexpr llvm::StringLiteral kNativeSymbolAttr{"ly.native.symbol"};
inline constexpr llvm::StringLiteral kNativeArgTypesAttr{"ly.native.argtypes"};
inline constexpr llvm::StringLiteral kNativeResultTypeAttr{
    "ly.native.result_type"};
inline constexpr llvm::StringLiteral kNativeABIAttr{"ly.native.abi"};
inline constexpr llvm::StringLiteral kNativeProcessLibraryAttr{
    "ly.native.process_library"};
inline constexpr llvm::StringLiteral kNativeTargetTripleAttr{
    "ly.native.target.triple"};
inline constexpr llvm::StringLiteral kNativeTargetPointerWidthAttr{
    "ly.native.target.pointer_width"};
inline constexpr llvm::StringLiteral kNativeTargetCLongWidthAttr{
    "ly.native.target.c_long_width"};

struct TargetPlatformFacts {
  std::string triple;
  std::uint64_t pointerWidth = 0;
  std::uint64_t cLongWidth = 0;

  std::uint64_t pointerBytes() const { return pointerWidth / 8; }
  std::uint64_t cLongBytes() const { return cLongWidth / 8; }
};

enum class NativeABIKind {
  SignedInteger,
  UnsignedInteger,
  Floating,
  Pointer,
};

struct NativeABIType {
  std::uint64_t size = 0;
  std::uint64_t align = 0;
  NativeABIKind kind = NativeABIKind::SignedInteger;
};

std::optional<TargetPlatformFacts>
readTargetPlatformFacts(mlir::ModuleOp module);
std::uint64_t expectedPointerWidth(llvm::StringRef triple);
std::uint64_t expectedCLongWidth(llvm::StringRef triple,
                                 std::uint64_t pointerWidth);
bool isSupportedNativeTarget(llvm::StringRef triple);
mlir::LogicalResult verifyTargetPlatformFacts(mlir::ModuleOp module);

llvm::StringRef stripCtypesModule(llvm::StringRef contract);
std::optional<NativeABIType>
ctypesScalarLayout(llvm::StringRef contract,
                   const std::optional<TargetPlatformFacts> &facts);
bool isIntegerABI(const NativeABIType &type);
bool isFloatingABI(const NativeABIType &type);
bool isPointerABI(const NativeABIType &type);

} // namespace py::native
