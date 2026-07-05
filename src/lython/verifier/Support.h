#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>

namespace py::verifier {

class VerificationResult {
public:
  void fail() { result = mlir::failure(); }

  void check(mlir::LogicalResult candidate) {
    if (mlir::failed(candidate))
      fail();
  }

  bool failed() const { return mlir::failed(result); }
  bool succeeded() const { return mlir::succeeded(result); }
  mlir::LogicalResult get() const { return result; }

private:
  mlir::LogicalResult result = mlir::success();
};

inline bool isIntegerType(mlir::Type type, unsigned width) {
  auto integer = mlir::dyn_cast_if_present<mlir::IntegerType>(type);
  return integer && integer.getWidth() == width;
}

inline mlir::FailureOr<llvm::StringRef>
readRequiredStringAttr(mlir::Operation *op, llvm::StringRef name,
                       llvm::StringRef subject) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(name);
  if (!attr)
    return op->emitError() << subject << " is missing string attribute "
                           << name;
  return attr.getValue();
}

inline mlir::FailureOr<bool>
readRequiredBoolAttr(mlir::Operation *op, llvm::StringRef name,
                     llvm::StringRef subject) {
  auto attr = op->getAttrOfType<mlir::BoolAttr>(name);
  if (!attr)
    return op->emitError() << subject << " is missing bool attribute " << name;
  return attr.getValue();
}

inline mlir::FailureOr<std::uint64_t>
readRequiredUnsignedIntegerAttr(mlir::Operation *op, llvm::StringRef name,
                                llvm::StringRef subject) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return op->emitError() << subject << " is missing integer attribute "
                           << name;
  if (attr.getInt() < 0)
    return op->emitError() << subject << " attribute " << name
                           << " must be non-negative";
  return static_cast<std::uint64_t>(attr.getInt());
}

inline mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>>
readOptionalStringArrayAttr(mlir::Operation *op, llvm::StringRef name) {
  llvm::SmallVector<llvm::StringRef, 8> values;
  auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!attr)
    return values;

  for (mlir::Attribute element : attr) {
    auto string = mlir::dyn_cast<mlir::StringAttr>(element);
    if (!string)
      return op->emitOpError() << name
                               << " entries must be string attributes";
    values.push_back(string.getValue());
  }
  return values;
}

inline mlir::FailureOr<llvm::SmallVector<std::string, 4>>
readRequiredStringArrayAttr(mlir::Operation *op, llvm::StringRef name,
                            llvm::StringRef subject) {
  auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!attr)
    return op->emitError() << subject << " is missing array attribute "
                           << name;

  llvm::SmallVector<std::string, 4> values;
  values.reserve(attr.size());
  for (auto [index, raw] : llvm::enumerate(attr)) {
    auto string = mlir::dyn_cast<mlir::StringAttr>(raw);
    if (!string)
      return op->emitError() << name << " element " << index
                             << " must be a string";
    values.push_back(string.getValue().str());
  }
  return values;
}

template <typename OpT, typename VerifyFn>
mlir::LogicalResult walkVerify(mlir::ModuleOp module, VerifyFn &&verify) {
  VerificationResult verified;
  module.walk([&](OpT op) -> mlir::WalkResult {
    if (verified.failed())
      return mlir::WalkResult::interrupt();
    verified.check(verify(op));
    return verified.failed() ? mlir::WalkResult::interrupt()
                             : mlir::WalkResult::advance();
  });
  return verified.get();
}

template <typename VerifyFn>
mlir::LogicalResult walkVerifyOperations(mlir::ModuleOp module,
                                         VerifyFn &&verify) {
  return walkVerify<mlir::Operation *>(module, std::forward<VerifyFn>(verify));
}

} // namespace py::verifier
