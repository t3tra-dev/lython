// This pass verifies that @native functions (marked with 'native' attribute)
// do not use any py.* types. This enforces the separation between
// Primitive World (P) and Object World (O) as specified in the modal logic
// framework (see note_modal.md).
//
// Theoretical basis:
//   - @native functions operate entirely in the Primitive World
//   - py.* types belong to the Object World (with GC)
//   - Mixing worlds violates the modal separation principle

#include "RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

struct NativeVerificationPass
    : public PassWrapper<NativeVerificationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeVerificationPass)

  StringRef getArgument() const override { return "verify-native-functions"; }

  StringRef getDescription() const override {
    return "Verify that @native functions do not use py.* types";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LogicalResult result = success();

    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp->hasAttr("native"))
        return;

      StringRef funcName = funcOp.getSymName();

      auto checkType = [&](Type type, Operation *op,
                           StringRef context) -> LogicalResult {
        if (!isPyType(type))
          return success();

        op->emitError() << "native function '" << funcName
                        << "' must not use py.* types in " << context
                        << ": found " << type
                        << "\n  Note: @native functions operate in the "
                           "Primitive World and cannot use Python objects";
        return failure();
      };

      funcOp.walk([&](Operation *op) {
        if (isa<func::FuncOp>(op))
          return;

        for (Value operand : op->getOperands()) {
          if (failed(checkType(operand.getType(), op, "operand")))
            result = failure();
        }

        for (Type resultType : op->getResultTypes()) {
          if (failed(checkType(resultType, op, "result")))
            result = failure();
        }
      });
    });

    if (failed(result))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createNativeVerificationPass() {
  return std::make_unique<NativeVerificationPass>();
}

} // namespace py
