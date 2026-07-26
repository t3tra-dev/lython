#include "Runtime/Core/Lowerer.h"

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace py::lowering {
namespace {

class RuntimeLoweringPass
    : public mlir::PassWrapper<RuntimeLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLoweringPass)

  llvm::StringRef getArgument() const final {
    return "lython-runtime-lowering";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    // Inline descriptor assembly (payload box word → memref view) emits
    // llvm-dialect ops directly; the boxed-method dispatch hooks emit ub.poison
    // for their miss arm.
    registry.insert<mlir::LLVM::LLVMDialect, mlir::ub::UBDialect>();
  }
  llvm::StringRef getDescription() const final {
    return "lower resolved Py dialect operations to the runtime ABI";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(requireResolvedInput(module))) {
      signalPassFailure();
      return;
    }

    if (mlir::failed(RuntimeBundleLowerer(module).lowerModule()))
      signalPassFailure();
  }

private:
  // Why not narrow this to identity-shaped casts, so a manifest body could
  // assemble a memref descriptor inline the way memrefFromBoxWords does? Two
  // reasons, and neither is a cost worth paying: an identity-shaped cast is
  // this pass's OWN owned-local-object marker vocabulary, and any 1-to-1 cast
  // is absorbed unconditionally into the ownership alias relation
  // (isOwnershipIdentityOp, common/Ownership.cpp), so admitting author-written
  // casts here would put values into that relation before the pass that owns
  // the vocabulary has run. And the capability is not actually withheld:
  // __ly_global_view_i8/i32/i64/f64 hand a manifest body the same descriptor
  // through a call (buildGlobalViewFunction, lowering/Common/
  // RuntimeSupportBuilder.cpp), which is how __ly_exc_payload_view already
  // rebuilds an exception member's triple from raw box words.
  mlir::LogicalResult requireResolvedInput(mlir::ModuleOp module) {
    mlir::LogicalResult result = mlir::success();
    module.walk([&](mlir::UnrealizedConversionCastOp op) {
      op.emitError()
          << "lowering requires fully resolved Python IR; "
             "builtin.unrealized_conversion_cast is resolution evidence, not "
             "a runtime ABI value. To rebuild a memref view over a payload "
             "a raw pointer word addresses, call __ly_global_view_i8/i32/i64/"
             "f64(pointer, size) and narrow the result with memref.cast";
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    });
    return result;
  }
};

} // namespace
} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass() {
  return std::make_unique<lowering::RuntimeLoweringPass>();
}

} // namespace py
