// This pass verifies that @native functions (marked with 'native' attribute)
// do not use any py.* types. This enforces the separation between
// Primitive World (P) and Object World (O) as specified in the modal logic
// framework.
//
// Theoretical basis:
//   - @native functions operate entirely in the Primitive World
//   - py.* types belong to the Object World (with GC)
//   - Mixing worlds violates the modal separation principle

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

struct NativeVerificationPass
    : public mlir::PassWrapper<NativeVerificationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeVerificationPass)

  llvm::StringRef getArgument() const override {
    return "verify-native-functions";
  }

  llvm::StringRef getDescription() const override {
    return "Verify that @native functions do not use py.* types";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::LogicalResult result = mlir::success();

    module.walk([&](mlir::func::FuncOp funcOp) {
      if (!funcOp->hasAttr("native"))
        return;

      llvm::StringRef funcName = funcOp.getSymName();

      auto checkType = [&](mlir::Type type, mlir::Operation *op,
                           llvm::StringRef context) -> mlir::LogicalResult {
        if (!isPyType(type))
          return mlir::success();

        op->emitError() << "native function '" << funcName
                        << "' must not use py.* types in " << context
                        << ": found " << type
                        << "\n  Note: @native functions operate in the "
                           "Primitive World and cannot use Python objects";
        return mlir::failure();
      };

      funcOp.walk([&](mlir::Operation *op) {
        if (mlir::isa<mlir::func::FuncOp>(op))
          return;

        for (mlir::Value operand : op->getOperands()) {
          if (mlir::failed(checkType(operand.getType(), op, "operand")))
            result = mlir::failure();
        }

        for (mlir::Type resultType : op->getResultTypes()) {
          if (mlir::failed(checkType(resultType, op, "result")))
            result = mlir::failure();
        }
      });
    });

    if (mlir::failed(result))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass() {
  return std::make_unique<NativeVerificationPass>();
}

} // namespace py
