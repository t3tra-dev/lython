#include "Driver.h"
#include "DriverCodeGen.h"

#include "Passes/Runtime/Arch/Arm/PrimitiveTensorArmSME.h"
#include "Passes/Runtime/Arch/X86/PrimitiveTensorX86.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/Passes.h"

#include <mutex>

#include "PyDialect.h.inc"

using namespace mlir;

namespace lython::driver {

void registerLythonDialects(DialectRegistry &registry) {
  // apply_registered_pass in manifest lowering strategies resolves through
  // the global pass registry; register once even when multiple contexts are
  // created in the same process (tests, fuzzers).
  static std::once_flag passRegistration;
  std::call_once(passRegistration,
                 [] { mlir::registerTransformsPasses(); });

  registry.insert<py::PyDialect, affine::AffineDialect, async::AsyncDialect,
                  func::FuncDialect, arith::ArithDialect, scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, tensor::TensorDialect,
                  linalg::LinalgDialect, memref::MemRefDialect,
                  vector::VectorDialect, bufferization::BufferizationDialect,
                  LLVM::LLVMDialect, math::MathDialect,
                  transform::TransformDialect>();
  py::lowering::arch::arm::registerSMEDialects(registry);
  py::lowering::arch::x86::registerX86Dialects(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  py::lowering::arch::x86::registerX86Translations(registry);
  registerPySafetyLLVMIRTranslation(registry);
}

} // namespace lython::driver
