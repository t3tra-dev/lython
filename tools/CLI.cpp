#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/ExecutionEngine/AsyncRuntime.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Error.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "llvm/Transforms/Coroutines/CoroEarly.h"
#include "llvm/Transforms/Coroutines/CoroSplit.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "Common/RuntimeSupport.h"
#include "Common/ThreadSafetyKernel.h"
#include "lyrt.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

// Forward declare our custom lowering pass factories
namespace py {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRuntimeLoweringPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafetyVerifierPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncRuntimeRewritePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLinalgLoweringPass();
} // namespace py

namespace {
class FileObjectCache final : public llvm::ObjectCache {
public:
  explicit FileObjectCache(std::string outputPath)
      : outputPath(std::move(outputPath)) {}

  void notifyObjectCompiled(const llvm::Module *M,
                            llvm::MemoryBufferRef ObjBuffer) override {
    (void)M;
    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "error: failed to write JIT object to " << outputPath
                   << ": " << ec.message() << "\n";
      return;
    }
    os.write(ObjBuffer.getBufferStart(), ObjBuffer.getBufferSize());
    os.flush();
  }

  std::unique_ptr<llvm::MemoryBuffer>
  getObject(const llvm::Module *M) override {
    (void)M;
    return nullptr;
  }

private:
  std::string outputPath;
};

constexpr llvm::StringLiteral kPythonFrontendScript = R"PY(
import ast
import os
import sys
from lython.mlir import ir
from lython.visitors._base import BaseVisitor

def main():
    if len(sys.argv) < 2:
        raise SystemExit("expected input.py")
    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as source_file:
        source = source_file.read()
    tree = ast.parse(source, filename=input_path)
    abs_path = os.path.abspath(input_path)
    tree.lython_module_name = abs_path
    ctx = ir.Context()
    parser = BaseVisitor(ctx)
    parser._set_module_name(abs_path)
    parser.visit(tree)
    module = parser.module
    module.operation.print(file=sys.stdout, enable_debug_info=True)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
)PY";

void setEnvVar(llvm::StringRef name, llvm::StringRef value) {
#if defined(_WIN32)
  _putenv_s(name.str().c_str(), value.str().c_str());
#else
  setenv(name.str().c_str(), value.str().c_str(), 1);
#endif
}

void unsetEnvVar(llvm::StringRef name) {
#if defined(_WIN32)
  _putenv_s(name.str().c_str(), "");
#else
  unsetenv(name.str().c_str());
#endif
}

LogicalResult runPipeline(ModuleOp module, MLIRContext &context) {
  bool dumpIR =
      static_cast<bool>(llvm::sys::Process::GetEnv("LYTHON_DUMP_LOWERING_IR"));

  if (dumpIR) {
    llvm::errs() << "=== [Frontend Output (before any passes)] ===\n";
    module.dump();
  }

  // Phase 1: Native verification
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] NativeVerificationPass\n";
    PassManager pm(&context);
    pm.addPass(py::createNativeVerificationPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After NativeVerificationPass] ===\n";
    module.dump();
  }

  // Phase 2: Publication preparation
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] PublicationPreparationPass\n";
    PassManager pm(&context);
    pm.addPass(py::createPublicationPreparationPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After PublicationPreparationPass] ===\n";
    module.dump();
  }

  // Phase 3: Reference counting insertion
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] RefCountInsertionPass\n";
    PassManager pm(&context);
    pm.addPass(py::createRefCountInsertionPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After RefCountInsertionPass] ===\n";
    module.dump();
  }

  // Phase 3.1: Proven refcount pair elision
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] RefCountPairElisionPass\n";
    PassManager pm(&context);
    pm.addPass(py::createRefCountPairElisionPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After RefCountPairElisionPass] ===\n";
    module.dump();
  }

  // Phase 3.2: Quantitative ownership verification
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] OwnershipVerifierPass\n";
    PassManager pm(&context);
    pm.addPass(py::createOwnershipVerifierPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After OwnershipVerifierPass] ===\n";
    module.dump();
  }

  // Phase 4: Early canonicalization and CSE
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] Canonicalizer + CSE\n";
    PassManager pm(&context);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After Canonicalizer + CSE] ===\n";
    module.dump();
  }

  // Phase 5: Runtime lowering (Py dialect -> func/LLVM)
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] RuntimeLoweringPass\n";
    PassManager pm(&context);
    pm.addPass(py::createRuntimeLoweringPass());
    if (failed(pm.run(module)))
      return failure();
  }

  // Phase 5.5: Let generic MLIR cleanup remove artifacts created by lowering.
  {
    if (dumpIR)
      llvm::errs()
          << "[Pipeline] Post-lowering Canonicalizer + CSE + SymbolDCE\n";
    PassManager pm(&context);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After Post-lowering Canonicalizer + CSE + "
                    "SymbolDCE] ===\n";
    module.dump();
  }

  // Phase 5.6: Validate that post-lowering cleanup did not alter ownership of
  // runtime calls that return or consume owned LyObject-family references.
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] LLVMCallOwnershipVerifierPass\n";
    PassManager pm(&context);
    pm.addPass(py::createLLVMCallOwnershipVerifierPass());
    if (failed(pm.run(module)))
      return failure();
  }

  // Phase 5.7: Validate memref-level no-GIL contracts before final lowering.
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] LLVMThreadSafetyVerifierPass\n";
    PassManager pm(&context);
    pm.addPass(py::createLLVMThreadSafetyVerifierPass());
    if (failed(pm.run(module)))
      return failure();
  }

  // Phase 6: Bufferization (tensor -> memref)
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] OneShotBufferize + BufferDeallocation\n";
    PassManager pm(&context);
    bufferization::OneShotBufferizationOptions options;
    options.allowUnknownOps = true;
    pm.addPass(bufferization::createOneShotBufferizePass(options));
    // Deallocation is handled later; avoid failing on unmatched allocations.
    if (failed(pm.run(module)))
      return failure();
  }

  // Phase 6: Lower linalg to loops
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] ConvertLinalgToLoops\n";
    PassManager pm(&context);
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    if (failed(pm.run(module)))
      return failure();
  }

  // Phase 6.5: Lower async.func/async.await to the MLIR async runtime, then
  // convert those runtime ops to LLVM. Lython owns !py.coro/!py.task/!py.future
  // descriptors linearly, so async runtime refcounting is emitted by Py
  // lowering instead of the generic MLIR async refcount pass.
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] AsyncToLLVM\n";
    PassManager pm(&context);
    pm.addPass(mlir::createAsyncFuncToAsyncRuntimePass());
    pm.addPass(mlir::createAsyncToAsyncRuntimePass());
    if (failed(pm.run(module)))
      return failure();
  }
  if (dumpIR) {
    llvm::errs() << "[After async func/runtime conversion]\n";
    module.dump();
  }
  {
    PassManager pm(&context);
    pm.addPass(py::createAsyncRuntimeRewritePass());
    if (failed(pm.run(module)))
      return failure();
  }
  if (dumpIR) {
    llvm::errs() << "[After Lython async exception rewrites]\n";
    module.dump();
  }
  SmallVector<py::AsyncArgProvenanceContract> asyncArgContracts;
  py::collectAsyncArgProvenanceContracts(module, asyncArgContracts);
  {
    PassManager pm(&context);
    pm.addPass(mlir::createConvertAsyncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (failed(pm.run(module)))
      return failure();
  }
  if (failed(py::preserveLLVMAsyncArgProvenanceContracts(module,
                                                         asyncArgContracts)))
    return failure();
  {
    PassManager pm(&context);
    pm.addPass(py::createAsyncRuntimeRewritePass());
    if (failed(pm.run(module)))
      return failure();
  }
  if (dumpIR) {
    llvm::errs() << "[After ConvertAsyncToLLVM]\n";
    module.dump();
  }
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] Post-async LLVMCallOwnershipVerifierPass\n";
    PassManager pm(&context);
    pm.addPass(py::createLLVMCallOwnershipVerifierPass());
    if (failed(pm.run(module)))
      return failure();
  }

  // Phase 7: Final lowering to LLVM
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] ConvertToLLVM\n";
    py::LoweredSafetyContracts finalSafetyContracts;
    py::collectLoweredSafetyContracts(module, finalSafetyContracts);
    PassManager pm(&context);
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(pm.run(module)))
      return failure();
    if (failed(
            py::preserveLoweredSafetyContracts(module, finalSafetyContracts)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [Final LLVM IR] ===\n";
    module.dump();
  }

  // Phase 7.4: Re-check ownership after final conversion rewrites.
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] Final LLVMCallOwnershipVerifierPass\n";
    PassManager pm(&context);
    pm.addPass(py::createLLVMCallOwnershipVerifierPass());
    if (failed(pm.run(module)))
      return failure();
  }

  // Phase 7.5: Validate final LLVM atomic orderings after MemRefToLLVM.
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] Final LLVMThreadSafetyVerifierPass\n";
    PassManager pm(&context);
    pm.addPass(py::createLLVMThreadSafetyVerifierPass());
    if (failed(pm.run(module)))
      return failure();
  }

  return success();
}

llvm::orc::SymbolMap
buildRuntimeSymbolMap(llvm::orc::MangleAndInterner interner) {
  llvm::orc::SymbolMap symbolMap;
  auto add = [&](llvm::StringRef name, auto *ptr) {
    symbolMap[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(ptr),
                                 llvm::JITSymbolFlags::Exported};
  };
  add("Ly_IncRef", &Ly_IncRef);
  add("Ly_DecRef", &Ly_DecRef);
  add("LyUnicode_FromUTF8", &LyUnicode_FromUTF8);
  add("LyUnicode_InternStaticUTF8", &LyUnicode_InternStaticUTF8);
  add("LyUnicode_Concat", &LyUnicode_Concat);
  add("Ly_GetNone", &Ly_GetNone);
  add("Ly_GetBuiltinPrint", &Ly_GetBuiltinPrint);
  add("builtin_print_impl", &builtin_print_impl);
  add("LyTensorF16_Repr", &LyTensorF16_Repr);
  add("LyTensorF32_Repr", &LyTensorF32_Repr);
  add("LyTensorF64_Repr", &LyTensorF64_Repr);
  add("LyTensorF128_Repr", &LyTensorF128_Repr);
  add("_mlir_ciface_LyTensorF16_Repr", &_mlir_ciface_LyTensorF16_Repr);
  add("_mlir_ciface_LyTensorF32_Repr", &_mlir_ciface_LyTensorF32_Repr);
  add("_mlir_ciface_LyTensorF64_Repr", &_mlir_ciface_LyTensorF64_Repr);
  add("_mlir_ciface_LyTensorF128_Repr", &_mlir_ciface_LyTensorF128_Repr);
  add("LyListI64_Repr", &LyListI64_Repr);
  add("LyListBool_Repr", &LyListBool_Repr);
  add("LyListF64Bits_Repr", &LyListF64Bits_Repr);
  add("LyListPtr_Repr", &LyListPtr_Repr);
  add("_mlir_ciface_LyListI64_Repr", &_mlir_ciface_LyListI64_Repr);
  add("_mlir_ciface_LyListBool_Repr", &_mlir_ciface_LyListBool_Repr);
  add("_mlir_ciface_LyListF64Bits_Repr", &_mlir_ciface_LyListF64Bits_Repr);
  add("_mlir_ciface_LyListPtr_Repr", &_mlir_ciface_LyListPtr_Repr);
  add("LyTupleI64_Repr", &LyTupleI64_Repr);
  add("LyTupleBool_Repr", &LyTupleBool_Repr);
  add("LyTupleF64Bits_Repr", &LyTupleF64Bits_Repr);
  add("LyTuplePtr_Repr", &LyTuplePtr_Repr);
  add("_mlir_ciface_LyTupleI64_Repr", &_mlir_ciface_LyTupleI64_Repr);
  add("_mlir_ciface_LyTupleBool_Repr", &_mlir_ciface_LyTupleBool_Repr);
  add("_mlir_ciface_LyTupleF64Bits_Repr", &_mlir_ciface_LyTupleF64Bits_Repr);
  add("_mlir_ciface_LyTuplePtr_Repr", &_mlir_ciface_LyTuplePtr_Repr);
  add("LyDictPacked_Repr", &LyDictPacked_Repr);
  add("_mlir_ciface_LyDictPacked_Repr", &_mlir_ciface_LyDictPacked_Repr);
  add("LyLong_FromI64", &LyLong_FromI64);
  add("LyLong_AsI64", &LyLong_AsI64);
  add("LyLong_FromString", &LyLong_FromString);
  add("LyLong_Add", &LyLong_Add);
  add("LyLong_Sub", &LyLong_Sub);
  add("LyLong_Compare", &LyLong_Compare);
  add("LyFloat_FromDouble", &LyFloat_FromDouble);
  add("LyFloat_AsDouble", &LyFloat_AsDouble);
  add("LyFloat_Add", &LyFloat_Add);
  add("LyFloat_Sub", &LyFloat_Sub);
  add("LyBool_FromBool", &LyBool_FromBool);
  add("LyObject_Repr", &LyObject_Repr);
  add("LyObject_EqBool", &LyObject_EqBool);
  add("LyClass_ReprNamed", &LyClass_ReprNamed);
  add("LyMem_Alloc", &LyMem_Alloc);
  add("LyMem_Free", &LyMem_Free);
  add("LyException_New", &LyException_New);
  add("LyException_SetCurrent", &LyException_SetCurrent);
  add("LyException_GetCurrent", &LyException_GetCurrent);
  add("LyException_Clear", &LyException_Clear);
  add("LyEH_Throw", &LyEH_Throw);
  add("LyEH_Capture", &LyEH_Capture);
  add("LyEH_ReportUnhandled", &LyEH_ReportUnhandled);
  add("LyTraceback_Push", &LyTraceback_Push);
  add("LyTraceback_Pop", &LyTraceback_Pop);
  add("LyTraceback_Clear", &LyTraceback_Clear);
  add("LyTraceback_Print", &LyTraceback_Print);
  add("LyNumber_Add", &LyNumber_Add);
  add("LyNumber_Sub", &LyNumber_Sub);
  add("LyNumber_Lt", &LyNumber_Lt);
  add("LyNumber_Le", &LyNumber_Le);
  add("LyNumber_Gt", &LyNumber_Gt);
  add("LyNumber_Ge", &LyNumber_Ge);
  add("LyNumber_Eq", &LyNumber_Eq);
  add("LyNumber_Ne", &LyNumber_Ne);
  add("LyBool_AsBool", &LyBool_AsBool);
  add("mlirAsyncRuntimeAddRef", &mlir::runtime::mlirAsyncRuntimeAddRef);
  add("mlirAsyncRuntimeDropRef", &mlir::runtime::mlirAsyncRuntimeDropRef);
  add("mlirAsyncRuntimeCreateToken",
      &mlir::runtime::mlirAsyncRuntimeCreateToken);
  add("mlirAsyncRuntimeCreateValue",
      &mlir::runtime::mlirAsyncRuntimeCreateValue);
  add("mlirAsyncRuntimeCreateGroup",
      &mlir::runtime::mlirAsyncRuntimeCreateGroup);
  add("mlirAsyncRuntimeAddTokenToGroup",
      &mlir::runtime::mlirAsyncRuntimeAddTokenToGroup);
  add("mlirAsyncRuntimeEmplaceToken",
      &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
  add("mlirAsyncRuntimeEmplaceValue",
      &mlir::runtime::mlirAsyncRuntimeEmplaceValue);
  add("mlirAsyncRuntimeSetTokenError",
      &mlir::runtime::mlirAsyncRuntimeSetTokenError);
  add("mlirAsyncRuntimeSetValueError",
      &mlir::runtime::mlirAsyncRuntimeSetValueError);
  add("mlirAsyncRuntimeIsTokenError",
      &mlir::runtime::mlirAsyncRuntimeIsTokenError);
  add("mlirAsyncRuntimeIsValueError",
      &mlir::runtime::mlirAsyncRuntimeIsValueError);
  add("mlirAsyncRuntimeIsGroupError",
      &mlir::runtime::mlirAsyncRuntimeIsGroupError);
  add("mlirAsyncRuntimeAwaitToken", &mlir::runtime::mlirAsyncRuntimeAwaitToken);
  add("mlirAsyncRuntimeAwaitValue", &mlir::runtime::mlirAsyncRuntimeAwaitValue);
  add("mlirAsyncRuntimeAwaitAllInGroup",
      &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroup);
  add("mlirAsyncRuntimeExecute", &mlir::runtime::mlirAsyncRuntimeExecute);
  add("mlirAsyncRuntimeGetValueStorage",
      &mlir::runtime::mlirAsyncRuntimeGetValueStorage);
  add("mlirAsyncRuntimeAwaitTokenAndExecute",
      &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
  add("mlirAsyncRuntimeAwaitValueAndExecute",
      &mlir::runtime::mlirAsyncRuntimeAwaitValueAndExecute);
  add("mlirAsyncRuntimeAwaitAllInGroupAndExecute",
      &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupAndExecute);
  add("mlirAsyncRuntimGetNumWorkerThreads",
      &mlir::runtime::mlirAsyncRuntimGetNumWorkerThreads);
  add("mlirAsyncRuntimePrintCurrentThreadId",
      &mlir::runtime::mlirAsyncRuntimePrintCurrentThreadId);
  return symbolMap;
}

OwningOpRef<ModuleOp> parseModuleFromBuffer(StringRef buffer,
                                            MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(buffer, &context);
  if (!module)
    llvm::errs() << "error: failed to parse MLIR source\n";
  return module;
}

std::optional<std::string> findBuildRoot(StringRef argv0) {
  llvm::SmallString<256> exePath(argv0);
  if (auto ec = llvm::sys::fs::real_path(exePath, exePath)) {
    llvm::errs() << "error: unable to resolve executable path: " << ec.message()
                 << "\n";
    return std::nullopt;
  }
  llvm::sys::path::remove_filename(exePath);

  llvm::SmallString<256> current(exePath);
  while (!current.empty()) {
    llvm::SmallString<256> cachePath(current);
    llvm::sys::path::append(cachePath, "CMakeCache.txt");
    if (llvm::sys::fs::exists(cachePath))
      return std::string(current.str());
    if (!llvm::sys::path::has_parent_path(current))
      break;
    llvm::sys::path::remove_filename(current);
  }
  return std::nullopt;
}

std::optional<std::string> findSourceRoot(StringRef buildRoot) {
  llvm::SmallString<256> cachePath(buildRoot);
  llvm::sys::path::append(cachePath, "CMakeCache.txt");
  auto bufferOrErr = llvm::MemoryBuffer::getFile(cachePath);
  if (!bufferOrErr)
    return std::nullopt;

  llvm::StringRef content = bufferOrErr->get()->getBuffer();
  llvm::StringRef key = "CMAKE_HOME_DIRECTORY:";
  size_t pos = content.find(key);
  if (pos == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef line = content.substr(pos);
  line = line.substr(0, line.find('\n'));
  size_t eq = line.find('=');
  if (eq == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef value = line.substr(eq + 1).trim();
  if (value.empty())
    return std::nullopt;
  return value.str();
}

LogicalResult generateMlirFromPython(StringRef pythonFile, StringRef sourceRoot,
                                     std::string &mlirBuffer) {
  auto findProjectPython =
      [&](llvm::StringRef executable) -> std::optional<std::string> {
    llvm::SmallString<256> candidate(sourceRoot);
    llvm::sys::path::append(candidate, ".venv");
#if defined(_WIN32)
    llvm::sys::path::append(candidate, "Scripts");
#else
    llvm::sys::path::append(candidate, "bin");
#endif
    llvm::SmallString<256> binary(candidate);
    llvm::sys::path::append(binary, executable);
    if (llvm::sys::fs::exists(binary))
      return std::string(binary.str());
    return std::nullopt;
  };

  std::optional<std::string> pythonExe;
#if defined(_WIN32)
  pythonExe = findProjectPython("python.exe");
#else
  pythonExe = findProjectPython("python3");
  if (!pythonExe)
    pythonExe = findProjectPython("python");
#endif
  if (!pythonExe) {
    auto sysPython = llvm::sys::findProgramByName("python3");
    if (!sysPython)
      sysPython = llvm::sys::findProgramByName("python");
    if (!sysPython) {
      llvm::errs() << "error: could not find python3/python executable\n";
      return failure();
    }
    pythonExe = *sysPython;
  }

  llvm::SmallString<256> scriptPath;
  if (auto ec = llvm::sys::fs::createTemporaryFile("lython_frontend", "py",
                                                   scriptPath)) {
    llvm::errs() << "error: failed to create temporary frontend script: "
                 << ec.message() << "\n";
    return failure();
  }
  llvm::FileRemover scriptCleanup(scriptPath);
  {
    std::error_code ec;
    llvm::raw_fd_ostream scriptFile(scriptPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "error: failed to open temporary frontend script: "
                   << ec.message() << "\n";
      return failure();
    }
    scriptFile << kPythonFrontendScript.str();
  }

  auto existingPyPath = llvm::sys::Process::GetEnv("PYTHONPATH");
  llvm::SmallString<256> newPyPath(sourceRoot);
  llvm::sys::path::append(newPyPath, "src");
  if (existingPyPath && !existingPyPath->empty()) {
    newPyPath += llvm::sys::path::get_separator();
    newPyPath += *existingPyPath;
  }
  setEnvVar("PYTHONPATH", newPyPath);

  std::string command;
  {
    llvm::raw_string_ostream os(command);
    llvm::sys::printArg(os, *pythonExe, /*Quote=*/true);
    os << ' ';
    llvm::sys::printArg(os, scriptPath.str(), /*Quote=*/true);
    os << ' ';
    llvm::sys::printArg(os, pythonFile, /*Quote=*/true);
  }

#if defined(_WIN32)
  FILE *pipe = _popen(command.c_str(), "rb");
#else
  FILE *pipe = popen(command.c_str(), "r");
#endif
  if (!pipe) {
    llvm::errs() << "error: failed to invoke python frontend\n";
    if (existingPyPath)
      setEnvVar("PYTHONPATH", *existingPyPath);
    else
      unsetEnvVar("PYTHONPATH");
    return failure();
  }

  mlirBuffer.clear();
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe))
    mlirBuffer.append(buffer);

#if defined(_WIN32)
  int result = _pclose(pipe);
#else
  int result = pclose(pipe);
#endif

  if (existingPyPath)
    setEnvVar("PYTHONPATH", *existingPyPath);
  else
    unsetEnvVar("PYTHONPATH");

  if (result != 0) {
    llvm::errs() << "error: python frontend failed for '" << pythonFile
                 << "'\n";
    return failure();
  }
  return success();
}

LogicalResult dumpLLVMIR(llvm::Module &llvmModule, StringRef outputPath) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
    return failure();
  }
  llvmModule.print(out, nullptr);
  return success();
}

void runLLVMCoroLowering(llvm::Module &llvmModule) {
  llvm::LoopAnalysisManager loopAM;
  llvm::FunctionAnalysisManager functionAM;
  llvm::CGSCCAnalysisManager cgsccAM;
  llvm::ModuleAnalysisManager moduleAM;
  llvm::PassBuilder passBuilder;
  passBuilder.registerModuleAnalyses(moduleAM);
  passBuilder.registerCGSCCAnalyses(cgsccAM);
  passBuilder.registerFunctionAnalyses(functionAM);
  passBuilder.registerLoopAnalyses(loopAM);
  passBuilder.crossRegisterProxies(loopAM, functionAM, cgsccAM, moduleAM);

  llvm::ModulePassManager modulePM;
  modulePM.addPass(llvm::CoroEarlyPass());
  llvm::CGSCCPassManager cgsccPM;
  cgsccPM.addPass(llvm::CoroSplitPass());
  modulePM.addPass(
      llvm::createModuleToPostOrderCGSCCPassAdaptor(std::move(cgsccPM)));
  modulePM.addPass(llvm::CoroCleanupPass());
  modulePM.run(llvmModule, moduleAM);
}

bool isLLVMConstantInt(llvm::Value *value, int64_t expected) {
  auto *constant = llvm::dyn_cast<llvm::ConstantInt>(value);
  return constant && constant->getSExtValue() == expected;
}

std::optional<int64_t> getLLVMConstantInt(llvm::Value *value) {
  auto *constant = llvm::dyn_cast<llvm::ConstantInt>(value);
  if (!constant)
    return std::nullopt;
  return constant->getSExtValue();
}

bool isLLVMNullPointerValue(llvm::Value *value) {
  auto *constant = llvm::dyn_cast<llvm::Constant>(value);
  return constant && constant->isNullValue();
}

enum class LLVMSafetyEffectKind {
  RuntimeRetainCall,
  RuntimeReleaseCall,
  RuntimeTransferCall,
  AsyncRuntimeRefcountCall,
  AtomicRMW,
  AtomicCmpXchg,
  AtomicLoad,
  AtomicStore,
};

struct LLVMSafetyContract {
  int64_t id = -1;
  std::string functionName;
  LLVMSafetyEffectKind kind;
  std::string callee;
  std::string opcode;
  std::string role;
  std::string ordering;
  std::string failureOrdering;
  std::string provenance;
  std::string retainPremise;
  std::string resourceGroup;
  std::string resourceComponent;
  std::string containerKind;
  std::optional<int64_t> resourceSlot;
  int64_t value = 0;
  bool asyncExceptionCell = false;
  bool asyncCancelFlag = false;
};

struct LLVMSafetyProfile {
  llvm::SmallVector<LLVMSafetyContract, 64> contracts;
};

static constexpr llvm::StringLiteral kLythonSafetyMetadataName{"ly.safety"};
static constexpr llvm::StringLiteral kLythonSafetyMetadataVersion{
    "ly.safety.v1"};

std::optional<StringRef> getStringAttr(Operation *op, StringRef attrName) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<StringAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

std::optional<int64_t> getMLIRLLVMConstantInt(Value value) {
  if (auto constant = value.getDefiningOp<LLVM::ConstantOp>()) {
    auto attr = dyn_cast<IntegerAttr>(constant.getValue());
    if (attr)
      return attr.getInt();
  }
  if (value.getDefiningOp<LLVM::ZeroOp>())
    return 0;
  return std::nullopt;
}

bool hasMLIRFunctionArgumentAttr(Value value, llvm::StringRef attrName) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  auto func = dyn_cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
  return func && func.getArgAttr(arg.getArgNumber(), attrName);
}

Value stripMLIRAsyncExceptionCellPointer(Value value) {
  while (value) {
    if (auto bitcast = value.getDefiningOp<LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    return value;
  }
  return {};
}

Value stripMLIRAsyncCancelFlagPointer(Value value) {
  while (value) {
    if (auto bitcast = value.getDefiningOp<LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (auto extract = value.getDefiningOp<LLVM::ExtractValueOp>()) {
      value = extract.getContainer();
      continue;
    }
    if (auto gep = value.getDefiningOp<LLVM::GEPOp>()) {
      value = gep.getBase();
      continue;
    }
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    return value;
  }
  return {};
}

bool hasMLIRAsyncExceptionCellProvenance(Value value) {
  value = stripMLIRAsyncExceptionCellPointer(value);
  if (!value)
    return false;
  if (isa<BlockArgument>(value))
    return hasMLIRFunctionArgumentAttr(value,
                                       py::AsyncSafetyAttrs::kExceptionCell);
  Operation *def = value.getDefiningOp();
  return def && def->hasAttr(py::AsyncSafetyAttrs::kExceptionCell);
}

bool hasMLIRAsyncCancelFlagProvenance(Operation *op, Value value) {
  if (op && op->hasAttr(py::AsyncSafetyAttrs::kCancelFlag))
    return true;
  value = stripMLIRAsyncCancelFlagPointer(value);
  if (!value)
    return false;
  if (isa<BlockArgument>(value))
    return hasMLIRFunctionArgumentAttr(value,
                                       py::AsyncSafetyAttrs::kCancelFlag);
  if (value.getDefiningOp<LLVM::AllocaOp>())
    return true;
  Operation *def = value.getDefiningOp();
  return def && def->hasAttr(py::AsyncSafetyAttrs::kCancelFlag);
}

std::string getMLIRAtomicOrderingName(LLVM::AtomicOrdering ordering) {
  switch (ordering) {
  case LLVM::AtomicOrdering::not_atomic:
    return "not_atomic";
  case LLVM::AtomicOrdering::unordered:
    return "unordered";
  case LLVM::AtomicOrdering::monotonic:
    return py::ThreadSafetyAttrs::kOrderingMonotonic.str();
  case LLVM::AtomicOrdering::acquire:
    return py::ThreadSafetyAttrs::kOrderingAcquire.str();
  case LLVM::AtomicOrdering::release:
    return py::ThreadSafetyAttrs::kOrderingRelease.str();
  case LLVM::AtomicOrdering::acq_rel:
    return py::ThreadSafetyAttrs::kOrderingAcqRel.str();
  case LLVM::AtomicOrdering::seq_cst:
    return py::ThreadSafetyAttrs::kOrderingSeqCst.str();
  }
  return "unknown";
}

std::string getLLVMAtomicOrderingName(llvm::AtomicOrdering ordering) {
  switch (ordering) {
  case llvm::AtomicOrdering::NotAtomic:
    return "not_atomic";
  case llvm::AtomicOrdering::Unordered:
    return "unordered";
  case llvm::AtomicOrdering::Monotonic:
    return py::ThreadSafetyAttrs::kOrderingMonotonic.str();
  case llvm::AtomicOrdering::Acquire:
    return py::ThreadSafetyAttrs::kOrderingAcquire.str();
  case llvm::AtomicOrdering::Release:
    return py::ThreadSafetyAttrs::kOrderingRelease.str();
  case llvm::AtomicOrdering::AcquireRelease:
    return py::ThreadSafetyAttrs::kOrderingAcqRel.str();
  case llvm::AtomicOrdering::SequentiallyConsistent:
    return py::ThreadSafetyAttrs::kOrderingSeqCst.str();
  }
  return "unknown";
}

std::string getMLIRAtomicRMWOpcodeName(LLVM::AtomicBinOp op) {
  switch (op) {
  case LLVM::AtomicBinOp::xchg:
    return "xchg";
  case LLVM::AtomicBinOp::add:
    return "add";
  case LLVM::AtomicBinOp::sub:
    return "sub";
  case LLVM::AtomicBinOp::_and:
    return "and";
  case LLVM::AtomicBinOp::_or:
    return "or";
  case LLVM::AtomicBinOp::_xor:
    return "xor";
  case LLVM::AtomicBinOp::nand:
    return "nand";
  case LLVM::AtomicBinOp::max:
    return "max";
  case LLVM::AtomicBinOp::min:
    return "min";
  case LLVM::AtomicBinOp::umax:
    return "umax";
  case LLVM::AtomicBinOp::umin:
    return "umin";
  case LLVM::AtomicBinOp::fadd:
    return "fadd";
  case LLVM::AtomicBinOp::fsub:
    return "fsub";
  case LLVM::AtomicBinOp::fmax:
    return "fmax";
  case LLVM::AtomicBinOp::fmin:
    return "fmin";
  case LLVM::AtomicBinOp::uinc_wrap:
    return "uinc_wrap";
  case LLVM::AtomicBinOp::udec_wrap:
    return "udec_wrap";
  case LLVM::AtomicBinOp::usub_cond:
    return "usub_cond";
  case LLVM::AtomicBinOp::usub_sat:
    return "usub_sat";
  }
  return "unknown";
}

std::string getLLVMAtomicRMWOpcodeName(llvm::AtomicRMWInst::BinOp op) {
  switch (op) {
  case llvm::AtomicRMWInst::Xchg:
    return "xchg";
  case llvm::AtomicRMWInst::Add:
    return "add";
  case llvm::AtomicRMWInst::Sub:
    return "sub";
  case llvm::AtomicRMWInst::And:
    return "and";
  case llvm::AtomicRMWInst::Or:
    return "or";
  case llvm::AtomicRMWInst::Xor:
    return "xor";
  case llvm::AtomicRMWInst::Nand:
    return "nand";
  case llvm::AtomicRMWInst::Max:
    return "max";
  case llvm::AtomicRMWInst::Min:
    return "min";
  case llvm::AtomicRMWInst::UMax:
    return "umax";
  case llvm::AtomicRMWInst::UMin:
    return "umin";
  case llvm::AtomicRMWInst::FAdd:
    return "fadd";
  case llvm::AtomicRMWInst::FSub:
    return "fsub";
  case llvm::AtomicRMWInst::FMax:
    return "fmax";
  case llvm::AtomicRMWInst::FMin:
    return "fmin";
  case llvm::AtomicRMWInst::UIncWrap:
    return "uinc_wrap";
  case llvm::AtomicRMWInst::UDecWrap:
    return "udec_wrap";
  case llvm::AtomicRMWInst::USubCond:
    return "usub_cond";
  case llvm::AtomicRMWInst::USubSat:
    return "usub_sat";
  case llvm::AtomicRMWInst::BAD_BINOP:
    return "bad";
  }
  return "unknown";
}

std::optional<LLVMSafetyEffectKind>
getSafetyEffectKind(llvm::Instruction &inst) {
  if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
    llvm::Function *callee = call->getCalledFunction();
    if (!callee)
      return std::nullopt;
    llvm::StringRef name = callee->getName();
    if (py::runtime::Callee::retain(name))
      return LLVMSafetyEffectKind::RuntimeRetainCall;
    if (py::runtime::Callee::release(name))
      return LLVMSafetyEffectKind::RuntimeReleaseCall;
    if (py::runtime::Callee::transfer(name))
      return LLVMSafetyEffectKind::RuntimeTransferCall;
    if (py::runtime::mlir_async::Callee::refcount(name))
      return LLVMSafetyEffectKind::AsyncRuntimeRefcountCall;
    return std::nullopt;
  }
  if (llvm::isa<llvm::AtomicRMWInst>(inst))
    return LLVMSafetyEffectKind::AtomicRMW;
  if (llvm::isa<llvm::AtomicCmpXchgInst>(inst))
    return LLVMSafetyEffectKind::AtomicCmpXchg;
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst))
    if (load->isAtomic())
      return LLVMSafetyEffectKind::AtomicLoad;
  if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst))
    if (store->isAtomic())
      return LLVMSafetyEffectKind::AtomicStore;
  return std::nullopt;
}

void appendAtomicAttrs(LLVMSafetyContract &contract, Operation *op) {
  if (auto role = getStringAttr(op, py::ThreadSafetyAttrs::kAtomicRole))
    contract.role = role->str();
  if (auto ordering = getStringAttr(op, py::ThreadSafetyAttrs::kAtomicOrdering))
    contract.ordering = ordering->str();
  if (auto provenance =
          getStringAttr(op, py::ThreadSafetyAttrs::kAtomicProvenance))
    contract.provenance = provenance->str();
  if (auto premise = getStringAttr(op, py::ThreadSafetyAttrs::kRetainPremise))
    contract.retainPremise = premise->str();
  if (auto group = getStringAttr(op, py::ThreadSafetyAttrs::kAtomicMemRefGroup))
    contract.resourceGroup = group->str();
  if (auto component =
          getStringAttr(op, py::ThreadSafetyAttrs::kAtomicMemRefComponent))
    contract.resourceComponent = component->str();
  if (auto kind =
          getStringAttr(op, py::ThreadSafetyAttrs::kAtomicContainerKind))
    contract.containerKind = kind->str();
  if (auto slot = op->getAttrOfType<IntegerAttr>(
          py::ThreadSafetyAttrs::kAtomicMemRefSlot))
    contract.resourceSlot = slot.getInt();
}

void collectLLVMSafetyContracts(ModuleOp module, LLVMSafetyProfile &profile) {
  module.walk([&](LLVM::LLVMFuncOp func) {
    for (Block &block : func.getBody()) {
      for (Operation &op : block) {
        LLVMSafetyContract contract;
        contract.id = static_cast<int64_t>(profile.contracts.size());
        contract.functionName = func.getName().str();

        if (auto call = dyn_cast<LLVM::CallOp>(&op)) {
          auto callee = call.getCallee();
          if (!callee)
            continue;
          contract.callee = callee->str();
          if (py::runtime::Callee::retain(*callee))
            contract.kind = LLVMSafetyEffectKind::RuntimeRetainCall;
          else if (py::runtime::Callee::release(*callee))
            contract.kind = LLVMSafetyEffectKind::RuntimeReleaseCall;
          else if (py::runtime::Callee::transfer(*callee))
            contract.kind = LLVMSafetyEffectKind::RuntimeTransferCall;
          else if (py::runtime::mlir_async::Callee::refcount(*callee))
            contract.kind = LLVMSafetyEffectKind::AsyncRuntimeRefcountCall;
          else
            continue;
          appendAtomicAttrs(contract, call.getOperation());
          if (py::runtime::mlir_async::Callee::refcount(*callee) &&
              call.getNumOperands() >= 2) {
            if (auto count = getMLIRLLVMConstantInt(call.getOperand(1)))
              contract.value = *count;
          }
          profile.contracts.push_back(std::move(contract));
          continue;
        }

        if (auto atomic = dyn_cast<LLVM::AtomicRMWOp>(&op)) {
          contract.kind = LLVMSafetyEffectKind::AtomicRMW;
          contract.opcode = getMLIRAtomicRMWOpcodeName(atomic.getBinOp());
          contract.ordering = getMLIRAtomicOrderingName(atomic.getOrdering());
          if (auto value = getMLIRLLVMConstantInt(atomic.getVal()))
            contract.value = *value;
          appendAtomicAttrs(contract, atomic.getOperation());
          if (llvm::StringRef(contract.role) ==
              py::ThreadSafetyAttrs::kRoleAsyncCancelRequest)
            contract.asyncCancelFlag = hasMLIRAsyncCancelFlagProvenance(
                atomic.getOperation(), atomic.getPtr());
          profile.contracts.push_back(std::move(contract));
          continue;
        }

        if (auto cmpxchg = dyn_cast<LLVM::AtomicCmpXchgOp>(&op)) {
          contract.kind = LLVMSafetyEffectKind::AtomicCmpXchg;
          contract.ordering =
              getMLIRAtomicOrderingName(cmpxchg.getSuccessOrdering());
          contract.failureOrdering =
              getMLIRAtomicOrderingName(cmpxchg.getFailureOrdering());
          appendAtomicAttrs(contract, cmpxchg.getOperation());
          if (llvm::StringRef(contract.role) ==
              py::ThreadSafetyAttrs::kRoleAsyncExceptionStore)
            contract.asyncExceptionCell =
                hasMLIRAsyncExceptionCellProvenance(cmpxchg.getPtr());
          profile.contracts.push_back(std::move(contract));
          continue;
        }

        if (auto load = dyn_cast<LLVM::LoadOp>(&op)) {
          if (load.getOrdering() == LLVM::AtomicOrdering::not_atomic)
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicLoad;
          contract.ordering = getMLIRAtomicOrderingName(load.getOrdering());
          appendAtomicAttrs(contract, load.getOperation());
          if (llvm::StringRef(contract.role) ==
              py::ThreadSafetyAttrs::kRoleAsyncExceptionLoad)
            contract.asyncExceptionCell =
                hasMLIRAsyncExceptionCellProvenance(load.getAddr());
          if (llvm::StringRef(contract.role) ==
              py::ThreadSafetyAttrs::kRoleAsyncCancelLoad)
            contract.asyncCancelFlag = hasMLIRAsyncCancelFlagProvenance(
                load.getOperation(), load.getAddr());
          profile.contracts.push_back(std::move(contract));
          continue;
        }

        if (auto store = dyn_cast<LLVM::StoreOp>(&op)) {
          if (store.getOrdering() == LLVM::AtomicOrdering::not_atomic)
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicStore;
          contract.ordering = getMLIRAtomicOrderingName(store.getOrdering());
          appendAtomicAttrs(contract, store.getOperation());
          profile.contracts.push_back(std::move(contract));
        }
      }
    }
  });
}

void setLythonSafetyMetadata(llvm::Instruction &inst,
                             const LLVMSafetyContract &contract) {
  llvm::LLVMContext &ctx = inst.getContext();
  llvm::Metadata *operands[] = {
      llvm::MDString::get(ctx, kLythonSafetyMetadataVersion),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), contract.id))};
  inst.setMetadata(kLythonSafetyMetadataName, llvm::MDNode::get(ctx, operands));
}

std::optional<int64_t> getLythonSafetyMetadataId(llvm::Instruction &inst) {
  llvm::MDNode *node = inst.getMetadata(kLythonSafetyMetadataName);
  if (!node || node->getNumOperands() != 2)
    return std::nullopt;
  auto *version = llvm::dyn_cast<llvm::MDString>(node->getOperand(0));
  if (!version || version->getString() != kLythonSafetyMetadataVersion)
    return std::nullopt;
  auto *constant =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(node->getOperand(1));
  if (!constant)
    return std::nullopt;
  auto *intConstant = llvm::dyn_cast<llvm::ConstantInt>(constant->getValue());
  if (!intConstant)
    return std::nullopt;
  return intConstant->getSExtValue();
}

const LLVMSafetyContract *
lookupSafetyContract(llvm::Instruction &inst,
                     const LLVMSafetyProfile &profile) {
  auto id = getLythonSafetyMetadataId(inst);
  if (!id || *id < 0 || static_cast<size_t>(*id) >= profile.contracts.size())
    return nullptr;
  return &profile.contracts[static_cast<size_t>(*id)];
}

void emitPostCoroVerifierError(llvm::Instruction &inst, llvm::StringRef msg);

static llvm::StringRef getContractRole(const LLVMSafetyContract &contract) {
  return llvm::StringRef(contract.role);
}

static bool roleIs(const LLVMSafetyContract &contract, llvm::StringRef role) {
  return getContractRole(contract) == role;
}

static bool hasContainerAtomicProvenance(const LLVMSafetyContract &contract) {
  llvm::StringRef role = getContractRole(contract);
  if (!py::role::containerAtomic(role))
    return true;
  return llvm::StringRef(contract.provenance) ==
             py::ThreadSafetyAttrs::kProvenanceMemRefDescriptor &&
         !contract.resourceGroup.empty() &&
         llvm::StringRef(contract.resourceComponent) ==
             py::ContainerSafetyAttrs::kComponentHeader &&
         contract.resourceSlot.has_value();
}

static bool hasRetainPremise(const LLVMSafetyContract &contract) {
  return !contract.retainPremise.empty();
}

LogicalResult annotateLLVMIRSafetyContracts(llvm::Module &llvmModule,
                                            const LLVMSafetyProfile &profile) {
  std::map<std::string, llvm::SmallVector<size_t, 16>> byFunction;
  for (auto indexed : llvm::enumerate(profile.contracts))
    byFunction[indexed.value().functionName].push_back(indexed.index());

  std::map<std::string, size_t> cursors;
  llvm::SmallVector<bool, 64> used(profile.contracts.size(), false);
  bool failedAny = false;

  for (llvm::Function &function : llvmModule) {
    auto it = byFunction.find(function.getName().str());
    if (it == byFunction.end())
      continue;
    llvm::ArrayRef<size_t> sequence = it->second;
    size_t &cursor = cursors[function.getName().str()];
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto kind = getSafetyEffectKind(inst);
        if (!kind)
          continue;
        if (cursor >= sequence.size()) {
          emitPostCoroVerifierError(
              inst, "LLVM IR safety effect has no MLIR contract");
          failedAny = true;
          continue;
        }
        const LLVMSafetyContract &contract =
            profile.contracts[sequence[cursor]];
        if (contract.kind != *kind) {
          emitPostCoroVerifierError(
              inst, "LLVM IR safety effect order diverges from MLIR contract");
          failedAny = true;
          continue;
        }
        setLythonSafetyMetadata(inst, contract);
        used[sequence[cursor]] = true;
        ++cursor;
      }
    }
  }

  for (auto indexed : llvm::enumerate(used)) {
    if (indexed.value())
      continue;
    const LLVMSafetyContract &contract = profile.contracts[indexed.index()];
    llvm::errs() << "error: LLVM IR safety contract for @"
                 << contract.functionName
                 << " was not attached during LLVM translation\n";
    failedAny = true;
  }

  return failure(failedAny);
}

void emitPostCoroVerifierError(llvm::Instruction &inst, llvm::StringRef msg) {
  llvm::errs() << "error: post-coro LLVM safety verifier: " << msg << "\n";
  if (llvm::Function *function = inst.getFunction())
    llvm::errs() << "  in function: " << function->getName() << "\n";
  llvm::errs() << "  instruction: " << inst << "\n";
}

LogicalResult verifyPostCoroCall(llvm::CallBase &call,
                                 const LLVMSafetyContract &contract) {
  llvm::Function *callee = call.getCalledFunction();
  if (!callee || callee->getName() != contract.callee) {
    emitPostCoroVerifierError(call,
                              "safety call callee differs from MLIR contract");
    return failure();
  }
  if (contract.kind == LLVMSafetyEffectKind::RuntimeRetainCall ||
      contract.kind == LLVMSafetyEffectKind::RuntimeReleaseCall ||
      contract.kind == LLVMSafetyEffectKind::RuntimeTransferCall) {
    if (call.arg_size() != 1) {
      emitPostCoroVerifierError(
          call, "runtime ownership call must have one ownership operand");
      return failure();
    }
    if (!call.getArgOperand(0)->getType()->isPointerTy()) {
      emitPostCoroVerifierError(call,
                                "runtime ownership operand must be a pointer");
      return failure();
    }
  }
  if (contract.kind == LLVMSafetyEffectKind::RuntimeRetainCall &&
      !hasRetainPremise(contract)) {
    emitPostCoroVerifierError(
        call, "runtime retain call lost its MLIR retain premise contract");
    return failure();
  }
  if (contract.kind == LLVMSafetyEffectKind::AsyncRuntimeRefcountCall) {
    if (call.arg_size() != 2) {
      emitPostCoroVerifierError(call,
                                "MLIR async runtime refcount call must have "
                                "handle and count operands");
      return failure();
    }
    if (!call.getArgOperand(0)->getType()->isPointerTy()) {
      emitPostCoroVerifierError(call,
                                "MLIR async runtime refcount handle must be a "
                                "pointer");
      return failure();
    }
    std::optional<int64_t> count = getLLVMConstantInt(call.getArgOperand(1));
    if (!count || *count <= 0 || *count != contract.value) {
      emitPostCoroVerifierError(call,
                                "MLIR async runtime refcount count differs "
                                "from MLIR contract");
      return failure();
    }
  }
  return success();
}

LogicalResult verifyPostCoroAtomicRMW(llvm::AtomicRMWInst &inst,
                                      const LLVMSafetyContract &contract) {
  llvm::AtomicOrdering ordering = inst.getOrdering();
  llvm::AtomicRMWInst::BinOp op = inst.getOperation();
  llvm::Value *value = inst.getValOperand();
  std::optional<int64_t> intValue = getLLVMConstantInt(value);
  llvm::StringRef role = getContractRole(contract);

  if (contract.kind != LLVMSafetyEffectKind::AtomicRMW ||
      contract.opcode != getLLVMAtomicRMWOpcodeName(op) || !intValue ||
      *intValue != contract.value ||
      contract.ordering != getLLVMAtomicOrderingName(ordering) ||
      role.empty()) {
    emitPostCoroVerifierError(
        inst, "atomicrmw does not match its MLIR safety contract");
    return failure();
  }

  if (!hasContainerAtomicProvenance(contract)) {
    emitPostCoroVerifierError(
        inst, "container atomicrmw lost memref descriptor provenance");
    return failure();
  }

  if (py::role::retainRefcount(role)) {
    if (op == llvm::AtomicRMWInst::Add && isLLVMConstantInt(value, 1) &&
        py::ordering::refcountInc(ordering) && hasRetainPremise(contract))
      return success();
    emitPostCoroVerifierError(inst, "retain RMW must be add +1, carry a "
                                    "retain premise, and use monotonic or "
                                    "stronger ordering");
    return failure();
  }

  if (py::role::releaseRefcount(role)) {
    if (op == llvm::AtomicRMWInst::Add && isLLVMConstantInt(value, -1) &&
        py::ordering::atLeastAcqRel(ordering))
      return success();
    emitPostCoroVerifierError(inst, "release RMW must be add -1 and acq_rel or "
                                    "stronger");
    return failure();
  }

  if (roleIs(contract, py::ThreadSafetyAttrs::kRoleContainerRefcountLoad)) {
    if (op == llvm::AtomicRMWInst::Add && isLLVMConstantInt(value, 0) &&
        py::ordering::atLeastAcquire(ordering))
      return success();
    emitPostCoroVerifierError(
        inst, "container refcount load RMW must be add 0 and acquire or "
              "stronger");
    return failure();
  }

  if (py::role::lockAcquire(role)) {
    if (op == llvm::AtomicRMWInst::Xchg && isLLVMConstantInt(value, 1) &&
        py::ordering::atLeastAcquire(ordering))
      return success();
    emitPostCoroVerifierError(inst,
                              "lock acquire xchg must be acquire or stronger");
    return failure();
  }

  if (py::role::lockRelease(role)) {
    if (op == llvm::AtomicRMWInst::Xchg && isLLVMConstantInt(value, 0) &&
        py::ordering::atLeastRelease(ordering))
      return success();
    emitPostCoroVerifierError(inst,
                              "lock release xchg must be release or stronger");
    return failure();
  }

  if (roleIs(contract, py::ThreadSafetyAttrs::kRoleAsyncCancelRequest)) {
    if (op == llvm::AtomicRMWInst::UMax && isLLVMConstantInt(value, 1) &&
        py::ordering::atLeastAcqRel(ordering) && contract.asyncCancelFlag)
      return success();
    emitPostCoroVerifierError(
        inst, "async cancel request must target a proven cancel flag and use "
              "acq_rel or seq_cst");
    return failure();
  }

  emitPostCoroVerifierError(inst, "unsupported generated atomicrmw role");
  return failure();
}

LogicalResult verifyPostCoroAtomicCmpXchg(llvm::AtomicCmpXchgInst &inst,
                                          const LLVMSafetyContract &contract) {
  llvm::StringRef role = getContractRole(contract);
  if (contract.kind != LLVMSafetyEffectKind::AtomicCmpXchg ||
      contract.ordering !=
          getLLVMAtomicOrderingName(inst.getSuccessOrdering()) ||
      contract.failureOrdering !=
          getLLVMAtomicOrderingName(inst.getFailureOrdering()) ||
      role.empty()) {
    emitPostCoroVerifierError(
        inst, "cmpxchg does not match its MLIR safety contract");
    return failure();
  }
  if (role != py::ThreadSafetyAttrs::kRoleAsyncExceptionStore) {
    emitPostCoroVerifierError(inst,
                              "cmpxchg must be an async exception-cell store");
    return failure();
  }
  if (!contract.asyncExceptionCell) {
    emitPostCoroVerifierError(
        inst, "async exception-cell cmpxchg lost exception-cell provenance");
    return failure();
  }
  if (!isLLVMNullPointerValue(inst.getCompareOperand())) {
    emitPostCoroVerifierError(
        inst, "async exception-cell cmpxchg must compare against null");
    return failure();
  }
  if (!inst.getNewValOperand()->getType()->isPointerTy()) {
    emitPostCoroVerifierError(
        inst, "async exception-cell cmpxchg payload must be a pointer");
    return failure();
  }
  if (!py::ordering::atLeastAcqRel(inst.getSuccessOrdering())) {
    emitPostCoroVerifierError(
        inst, "cmpxchg success ordering must be acq_rel or seq_cst");
    return failure();
  }
  if (!py::ordering::atLeastAcquire(inst.getFailureOrdering())) {
    emitPostCoroVerifierError(
        inst, "cmpxchg failure ordering must be acquire or seq_cst");
    return failure();
  }
  return success();
}

LogicalResult verifyPostCoroAtomicLoad(llvm::LoadInst &inst,
                                       const LLVMSafetyContract &contract) {
  llvm::StringRef role = getContractRole(contract);
  if (contract.kind != LLVMSafetyEffectKind::AtomicLoad ||
      contract.ordering != getLLVMAtomicOrderingName(inst.getOrdering()) ||
      role.empty()) {
    emitPostCoroVerifierError(
        inst, "atomic load does not match its MLIR safety contract");
    return failure();
  }
  if (role != py::ThreadSafetyAttrs::kRoleAsyncExceptionLoad &&
      role != py::ThreadSafetyAttrs::kRoleAsyncCancelLoad) {
    emitPostCoroVerifierError(inst, "unsupported atomic load role");
    return failure();
  }
  if (role == py::ThreadSafetyAttrs::kRoleAsyncExceptionLoad &&
      !contract.asyncExceptionCell) {
    emitPostCoroVerifierError(
        inst, "async exception-cell load lost exception-cell provenance");
    return failure();
  }
  if (role == py::ThreadSafetyAttrs::kRoleAsyncCancelLoad &&
      !contract.asyncCancelFlag) {
    emitPostCoroVerifierError(inst,
                              "async cancel load lost cancel-flag provenance");
    return failure();
  }
  if (!py::ordering::atLeastAcquire(inst.getOrdering())) {
    emitPostCoroVerifierError(inst, "atomic load must be acquire or stronger");
    return failure();
  }
  return success();
}

LogicalResult verifyPostCoroAtomicStore(llvm::StoreInst &inst,
                                        const LLVMSafetyContract &contract) {
  llvm::StringRef role = getContractRole(contract);
  if (contract.kind != LLVMSafetyEffectKind::AtomicStore ||
      contract.ordering != getLLVMAtomicOrderingName(inst.getOrdering()) ||
      role.empty()) {
    emitPostCoroVerifierError(
        inst, "atomic store does not match its MLIR safety contract");
    return failure();
  }
  if (!py::role::lockRelease(role)) {
    emitPostCoroVerifierError(inst, "atomic store must be a lock release");
    return failure();
  }
  if (!hasContainerAtomicProvenance(contract)) {
    emitPostCoroVerifierError(
        inst, "container atomic store lost memref descriptor provenance");
    return failure();
  }
  if (!isLLVMConstantInt(inst.getValueOperand(), 0)) {
    emitPostCoroVerifierError(inst, "lock release store must write 0");
    return failure();
  }
  if (!py::ordering::atLeastRelease(inst.getOrdering())) {
    emitPostCoroVerifierError(inst, "atomic store must be release or stronger");
    return failure();
  }
  return success();
}

LogicalResult verifyPostCoroLLVMThreadSafety(llvm::Module &llvmModule,
                                             const LLVMSafetyProfile &profile) {
  if (llvm::verifyModule(llvmModule, &llvm::errs()))
    return failure();

  bool failedAny = false;
  std::vector<unsigned> contractUseCounts(profile.contracts.size(), 0);
  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto kind = getSafetyEffectKind(inst);
        if (!kind)
          continue;

        const LLVMSafetyContract *contract =
            lookupSafetyContract(inst, profile);
        if (!contract) {
          emitPostCoroVerifierError(
              inst,
              "safety effect is missing preserved MLIR contract metadata");
          failedAny = true;
          continue;
        }

        if (contract->kind != *kind) {
          emitPostCoroVerifierError(
              inst, "safety effect kind differs from preserved MLIR contract");
          failedAny = true;
          continue;
        }
        if (auto id = getLythonSafetyMetadataId(inst))
          ++contractUseCounts[static_cast<size_t>(*id)];

        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
          if (failed(verifyPostCoroCall(*call, *contract)))
            failedAny = true;
          continue;
        }
        if (auto *rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&inst)) {
          if (failed(verifyPostCoroAtomicRMW(*rmw, *contract)))
            failedAny = true;
          continue;
        }
        if (auto *cmpxchg = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&inst)) {
          if (failed(verifyPostCoroAtomicCmpXchg(*cmpxchg, *contract)))
            failedAny = true;
          continue;
        }
        if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
          if (failed(verifyPostCoroAtomicLoad(*load, *contract)))
            failedAny = true;
          continue;
        }
        if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
          if (failed(verifyPostCoroAtomicStore(*store, *contract)))
            failedAny = true;
        }
      }
    }
  }
  for (auto indexed : llvm::enumerate(contractUseCounts)) {
    if (indexed.value() != 0)
      continue;
    const LLVMSafetyContract &contract = profile.contracts[indexed.index()];
    llvm::errs() << "error: post-coro LLVM safety verifier: MLIR safety "
                    "contract disappeared after LLVM lowering\n"
                 << "  contract id: " << contract.id << "\n"
                 << "  original function: " << contract.functionName << "\n";
    failedAny = true;
  }
  return failure(failedAny);
}

LogicalResult emitObjectFile(llvm::Module &llvmModule,
                             const LLVMSafetyProfile &safetyProfile,
                             StringRef objectPath) {
  auto targetTriple = llvmModule.getTargetTriple();
  if (targetTriple.empty())
    targetTriple = llvm::sys::getDefaultTargetTriple();

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "Failed to lookup target: " << error << "\n";
    return failure();
  }

  llvm::TargetOptions opt;
  auto targetMachine =
      std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
          targetTriple, "generic", "", opt, std::nullopt));
  llvmModule.setTargetTriple(targetTriple);
  llvmModule.setDataLayout(targetMachine->createDataLayout());
  runLLVMCoroLowering(llvmModule);
  if (failed(verifyPostCoroLLVMThreadSafety(llvmModule, safetyProfile)))
    return failure();

  std::error_code ec;
  llvm::raw_fd_ostream dest(objectPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Failed to open object file: " << ec.message() << "\n";
    return failure();
  }

  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "Target machine cannot emit object file\n";
    return failure();
  }
  pass.run(llvmModule);
  return success();
}

LogicalResult linkExecutable(StringRef objectPath, StringRef runtimeLib,
                             StringRef asyncRuntimeLib, StringRef outputPath) {
  auto clangExe = llvm::sys::findProgramByName("clang++");
  if (!clangExe)
    clangExe = llvm::sys::findProgramByName("clang");
  if (!clangExe) {
    llvm::errs() << "error: clang++/clang executable not found in PATH\n";
    return failure();
  }

  std::string clangProgram = *clangExe;
  std::vector<std::string> argStorage;
  argStorage.push_back(clangProgram);
  argStorage.emplace_back(objectPath.str());
  argStorage.emplace_back(runtimeLib.str());
  if (!asyncRuntimeLib.empty()) {
    argStorage.emplace_back(asyncRuntimeLib.str());
    llvm::SmallString<256> asyncRuntimeDir(asyncRuntimeLib);
    llvm::sys::path::remove_filename(asyncRuntimeDir);
    argStorage.emplace_back("-Wl,-rpath," + asyncRuntimeDir.str().str());
  }
  argStorage.emplace_back("-O2");
#if defined(__linux__)
  argStorage.emplace_back("-no-pie");
#endif
  argStorage.emplace_back("-o");
  argStorage.emplace_back(outputPath.str());

  llvm::SmallVector<llvm::StringRef, 8> args;
  for (const auto &arg : argStorage)
    args.push_back(arg);

  std::string errorMessage;
  bool executionFailed = false;
  int result =
      llvm::sys::ExecuteAndWait(clangProgram, args, std::nullopt, std::nullopt,
                                0, 0, &errorMessage, &executionFailed);
  if (result != 0 || executionFailed) {
    if (!errorMessage.empty())
      llvm::errs() << errorMessage << "\n";
    llvm::errs() << "error: linking failed\n";
    return failure();
  }
  return success();
}

LogicalResult buildExecutable(llvm::Module &llvmModule,
                              const LLVMSafetyProfile &safetyProfile,
                              StringRef runtimeLib, StringRef asyncRuntimeLib,
                              StringRef outputPath) {
  llvm::SmallString<256> objectPath;
  if (auto ec =
          llvm::sys::fs::createTemporaryFile("lython", ".o", objectPath)) {
    llvm::errs() << "error: failed to create temporary object file: "
                 << ec.message() << "\n";
    return failure();
  }
  llvm::FileRemover objCleanup(objectPath);
  if (failed(emitObjectFile(llvmModule, safetyProfile, objectPath)))
    return failure();
  if (failed(
          linkExecutable(objectPath, runtimeLib, asyncRuntimeLib, outputPath)))
    return failure();
  return success();
}

LogicalResult runJIT(ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto tmBuilderOrErr = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrErr) {
    llvm::errs() << "Failed to create JITTargetMachineBuilder: "
                 << llvm::toString(tmBuilderOrErr.takeError()) << "\n";
    return failure();
  }
  auto tmBuilder = std::move(*tmBuilderOrErr);
  auto options = tmBuilder.getOptions();
  options.ExceptionModel = llvm::ExceptionHandling::DwarfCFI;
  options.MCOptions.EmitCompactUnwindNonCanonical = true;
  options.ForceDwarfFrameSection = true;
  options.MCOptions.EmitDwarfUnwind = llvm::EmitDwarfUnwindType::Always;
  tmBuilder.setOptions(options);

  std::unique_ptr<FileObjectCache> objectCache;
  if (const char *dumpEnv = std::getenv("LYTHON_DUMP_JIT_OBJECT")) {
    std::string dumpPath = dumpEnv;
    if (dumpPath.empty() || dumpPath == "1")
      dumpPath = "/tmp/lython_jit.o";
    objectCache = std::make_unique<FileObjectCache>(std::move(dumpPath));
  }

  auto compileFunctionCreator = [&](llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    auto tmOrErr = jtmb.createTargetMachine();
    if (!tmOrErr)
      return tmOrErr.takeError();
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
        std::move(*tmOrErr), objectCache.get());
  };

  auto objectLayerCreator =
      [](llvm::orc::ExecutionSession &session,
         const llvm::Triple &tt) -> std::unique_ptr<llvm::orc::ObjectLayer> {
    auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);
    if (tt.isOSBinFormatCOFF()) {
      layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      layer->setAutoClaimResponsibilityForObjectSymbols(true);
    }
    return layer;
  };

  llvm::orc::LLJITBuilder builder;
  builder.setJITTargetMachineBuilder(std::move(tmBuilder))
      .setCompileFunctionCreator(compileFunctionCreator)
      .setObjectLinkingLayerCreator(objectLayerCreator)
      .setPrePlatformSetup([](llvm::orc::LLJIT &jit) -> llvm::Error {
        auto psjd = jit.getProcessSymbolsJITDylib();
        if (!psjd)
          return llvm::make_error<llvm::StringError>(
              "Native platforms require a process symbols JITDylib",
              llvm::inconvertibleErrorCode());
        auto dlGen =
            llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                jit.getDataLayout().getGlobalPrefix());
        if (dlGen)
          psjd->addGenerator(std::move(*dlGen));
#if defined(__APPLE__)
        const char *libcppCandidates[] = {
            "/usr/lib/libc++.1.dylib",
            "libc++.1.dylib",
            "/usr/lib/libc++abi.dylib",
            "libc++abi.dylib",
        };
        for (const char *lib : libcppCandidates) {
          if (auto gen = llvm::orc::DynamicLibrarySearchGenerator::Load(
                  lib, jit.getDataLayout().getGlobalPrefix())) {
            psjd->addGenerator(std::move(*gen));
          } else {
            llvm::consumeError(gen.takeError());
          }
        }
#endif
        return llvm::Error::success();
      });
  auto jitExpected = builder.create();
  if (!jitExpected) {
    llvm::errs() << "Failed to create LLJIT\n";
    return failure();
  }
  auto jit = std::move(*jitExpected);

  LLVMSafetyProfile safetyProfile;
  collectLLVMSafetyContracts(module, safetyProfile);

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR\n";
    return failure();
  }
  if (failed(annotateLLVMIRSafetyContracts(*llvmModule, safetyProfile)))
    return failure();
  if (failed(verifyPostCoroLLVMThreadSafety(*llvmModule, safetyProfile)))
    return failure();
  for (auto &func : *llvmModule) {
    if (!func.isDeclaration())
      func.setUWTableKind(llvm::UWTableKind::Async);
  }

  llvmModule->setDataLayout(jit->getDataLayout());
  llvmModule->setTargetTriple(jit->getTargetTriple().getTriple());
  runLLVMCoroLowering(*llvmModule);
  if (failed(verifyPostCoroLLVMThreadSafety(*llvmModule, safetyProfile)))
    return failure();

  auto &jd = jit->getMainJITDylib();
  auto dlGen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      jit->getDataLayout().getGlobalPrefix());
  if (dlGen)
    jd.addGenerator(std::move(*dlGen));
  llvm::orc::MangleAndInterner interner(jd.getExecutionSession(),
                                        jit->getDataLayout());
  auto symbolMap = buildRuntimeSymbolMap(interner);
  cantFail(jd.define(absoluteSymbols(std::move(symbolMap))));

  auto tsm = llvm::orc::ThreadSafeModule(std::move(llvmModule),
                                         std::move(llvmContext));
  if (auto err = jit->addIRModule(std::move(tsm))) {
    llvm::errs() << "JIT add module error: " << err << "\n";
    return failure();
  }
  if (auto err = jit->initialize(jd)) {
    llvm::errs() << "JIT initialize error: " << err << "\n";
    return failure();
  }

  auto sym = jit->lookup("_mlir_ciface_main");
  if (!sym) {
    llvm::errs() << "JIT lookup failed: " << sym.takeError() << "\n";
    return failure();
  }
  auto *entry = sym->toPtr<int32_t (*)()>();
  int32_t exitCode = entry();
  return exitCode == 0 ? success() : failure();
}

} // namespace

static llvm::cl::OptionCategory LythonCategory("lython options");
static llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::cat(LythonCategory));
static llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Output file (default: a.out)"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("a.out"),
                   llvm::cl::cat(LythonCategory));
static llvm::cl::opt<bool> EmitLLVMOnly(
    "emit-llvm",
    llvm::cl::desc("Stop after emitting LLVM IR to the output file"),
    llvm::cl::init(false), llvm::cl::cat(LythonCategory));
static llvm::cl::SubCommand JitCommand("jit", "JIT execute an input file");
static llvm::cl::opt<std::string>
    JitInputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                     llvm::cl::Required, llvm::cl::sub(JitCommand));

int main(int argc, char **argv) {
  if (std::getenv("LYTHON_DEBUG_SIGNALS")) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  }
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &os) { os << "Lython CLI based on MLIR\n"; });
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Lython compiler driver (clang-style)\n");

  const bool jitMode = static_cast<bool>(JitCommand);
  std::string inputPath;
  std::string outputPath = OutputFilename;

  if (jitMode) {
    inputPath = JitInputFilename;
  } else {
    if (InputFilename.empty()) {
      llvm::errs() << "error: no input files\n";
      llvm::cl::PrintHelpMessage();
      return 1;
    }
    inputPath = InputFilename;
  }

  auto buildRoot = findBuildRoot(argv[0]);
  if (!buildRoot) {
    llvm::errs() << "error: unable to locate CMake build root\n";
    return 1;
  }

  llvm::SmallString<256> runtimeLibPath(*buildRoot);
  llvm::sys::path::append(runtimeLibPath, "src", "lython", "runtime",
                          "libLythonRuntime.a");
  if (!llvm::sys::fs::exists(runtimeLibPath)) {
    llvm::errs() << "error: runtime library not found at '" << runtimeLibPath
                 << "'\n";
    return 1;
  }

  auto projectSourceRoot = findSourceRoot(*buildRoot);
  if (!projectSourceRoot) {
    llvm::errs() << "error: unable to determine source root\n";
    return 1;
  }

  llvm::SmallString<256> asyncRuntimeLibPath(LYTHON_MLIR_ASYNC_RUNTIME_LIBRARY);
  if (!llvm::sys::fs::exists(asyncRuntimeLibPath))
    asyncRuntimeLibPath.clear();

  bool isPythonInput = llvm::sys::path::extension(inputPath) == ".py";
  std::optional<std::string> sourceRoot;
  if (isPythonInput) {
    sourceRoot = projectSourceRoot;
  }

  DialectRegistry registry;
  registry.insert<py::PyDialect, async::AsyncDialect, func::FuncDialect,
                  arith::ArithDialect, scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, tensor::TensorDialect,
                  linalg::LinalgDialect, memref::MemRefDialect,
                  bufferization::BufferizationDialect, LLVM::LLVMDialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerAllToLLVMIRTranslations(registry);

  MLIRContext context;
  context.appendDialectRegistry(registry);

  std::string mlirSource;
  if (isPythonInput) {
    if (failed(generateMlirFromPython(inputPath, *sourceRoot, mlirSource)))
      return 1;
  } else {
    auto file = mlir::openInputFile(inputPath);
    if (!file) {
      llvm::errs() << "error: could not open input file '" << inputPath
                   << "'\n";
      return 1;
    }
    mlirSource = std::string(file->getBuffer());
  }

  OwningOpRef<ModuleOp> module = parseModuleFromBuffer(mlirSource, context);
  if (!module)
    return 1;

  if (failed(runPipeline(*module, context))) {
    llvm::errs() << "Failed to run lowering pipeline\n";
    return 1;
  }

  if (jitMode) {
    return failed(runJIT(*module)) ? 1 : 0;
  }

  LLVMSafetyProfile safetyProfile;
  collectLLVMSafetyContracts(*module, safetyProfile);

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR\n";
    return 1;
  }
  if (failed(annotateLLVMIRSafetyContracts(*llvmModule, safetyProfile)))
    return 1;
  if (failed(verifyPostCoroLLVMThreadSafety(*llvmModule, safetyProfile)))
    return 1;

  if (EmitLLVMOnly)
    return failed(dumpLLVMIR(*llvmModule, outputPath)) ? 1 : 0;

  if (llvm::InitializeNativeTarget()) {
    llvm::errs() << "error: could not initialize native target\n";
    return 1;
  }
  if (llvm::InitializeNativeTargetAsmPrinter()) {
    llvm::errs() << "error: could not initialize native asm printer\n";
    return 1;
  }

  return failed(buildExecutable(*llvmModule, safetyProfile, runtimeLibPath,
                                asyncRuntimeLibPath, outputPath))
             ? 1
             : 0;
}
