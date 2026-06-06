#include "Common/Container.h"

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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Error.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
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
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "Common/Object.h"
#include "Common/RuntimeLibrary.h"
#include "Common/RuntimeSupport.h"
#include "Common/ThreadSafetyKernel.h"
#include "Passes/Runtime/Cleanup.h"
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
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPyOptimizationPass();
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
from lython.frontend import analyze_module_types
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
    analyze_module_types(tree, filename=abs_path)
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

  // Phase 3.15: High-level Py semantic optimizations. This must run before
  // runtime MLIR embedding so the imported runtime roots match the optimized
  // py dialect, and before OwnershipVerifierPass so all ownership rewrites are
  // still checked by the quantitative kernel.
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] PyOptimizationPass\n";
    PassManager pm(&context);
    pm.addPass(py::createPyOptimizationPass());
    if (failed(pm.run(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After PyOptimizationPass] ===\n";
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

  // Phase 4.5: Import runtime object definitions written in MLIR. These
  // fragments are kept at func/memref level and flow through the same lowering
  // pipeline as user code.
  {
    if (dumpIR)
      llvm::errs() << "[Pipeline] Runtime object MLIR embedding\n";
    if (failed(py::runtime_library::embedObjectModules(module)))
      return failure();
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After Runtime object MLIR embedding] ===\n";
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

  // Phase 5.5: Let generic MLIR cleanup remove artifacts created by lowering
  // and runtime embedding.
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
    while (py::lowering::runtime::cleanup::pointerRoundTrips(module))
      ;
  }

  if (dumpIR) {
    llvm::errs() << "\n=== [After Post-lowering Canonicalizer + CSE + "
                    "SymbolDCE] ===\n";
    module.dump();
  }

  // Phase 5.6: Validate that post-lowering cleanup did not alter ownership of
  // runtime calls that return or consume owned object-family descriptors.
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

  // Phase 6: Lower async.func/async.await to the MLIR async runtime, then
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
  py::optimizer::pipeline::finalLLVMCleanup(module);
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
    py::optimizer::pipeline::finalLLVMCleanup(module);
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
  add("LyHost_PrintLine", &LyHost_PrintLine);
  add("LyEH_ThrowException", &LyEH_ThrowException);
  add("LyEH_RethrowCurrent", &LyEH_RethrowCurrent);
  add("LyEH_TakeCurrentDescriptor", &LyEH_TakeCurrentDescriptor);
  add("LyTraceback_Push", &LyTraceback_Push);
  add("LyTraceback_Pop", &LyTraceback_Pop);
  add("LyTraceback_Clear", &LyTraceback_Clear);
  add("LyTraceback_PrintMessage", &LyTraceback_PrintMessage);
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

enum class LLVMSafetyEffectKind {
  AtomicRMW,
  AtomicCmpXchg,
  AtomicLoad,
  AtomicStore,
};

struct LLVMSafetyContract {
  int64_t id = -1;
  std::string functionName;
  LLVMSafetyEffectKind kind;
};

struct LLVMSafetyProfile {
  llvm::SmallVector<LLVMSafetyContract, 64> contracts;
};

static constexpr llvm::StringLiteral kLythonSafetyMetadataName{"ly.safety"};
static constexpr llvm::StringLiteral kLythonSafetyMetadataVersion{
    "ly.safety.v1"};
static constexpr llvm::StringLiteral kPySafetyContractIdAttr{
    "py.safety_contract_id"};

std::optional<LLVMSafetyEffectKind>
getStructuralSafetyEffectKind(llvm::Instruction &inst) {
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

static bool instructionMatchesContract(llvm::Instruction &inst,
                                       const LLVMSafetyContract &contract) {
  switch (contract.kind) {
  case LLVMSafetyEffectKind::AtomicRMW:
    return llvm::isa<llvm::AtomicRMWInst>(inst);
  case LLVMSafetyEffectKind::AtomicCmpXchg:
    return llvm::isa<llvm::AtomicCmpXchgInst>(inst);
  case LLVMSafetyEffectKind::AtomicLoad: {
    auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst);
    return load && load->isAtomic();
  }
  case LLVMSafetyEffectKind::AtomicStore: {
    auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst);
    return store && store->isAtomic();
  }
  }
  return false;
}

void collectLLVMSafetyContracts(ModuleOp module, LLVMSafetyProfile &profile) {
  module.walk([&](LLVM::LLVMFuncOp func) {
    for (Block &block : func.getBody()) {
      for (Operation &op : block) {
        LLVMSafetyContract contract;
        contract.id = static_cast<int64_t>(profile.contracts.size());
        contract.functionName = func.getName().str();
        auto markContract = [&] {
          op.setAttr(kPySafetyContractIdAttr,
                     IntegerAttr::get(IntegerType::get(op.getContext(), 64),
                                      contract.id));
          profile.contracts.push_back(std::move(contract));
        };

        if (auto atomic = dyn_cast<LLVM::AtomicRMWOp>(&op)) {
          contract.kind = LLVMSafetyEffectKind::AtomicRMW;
          markContract();
          continue;
        }

        if (auto cmpxchg = dyn_cast<LLVM::AtomicCmpXchgOp>(&op)) {
          contract.kind = LLVMSafetyEffectKind::AtomicCmpXchg;
          markContract();
          continue;
        }

        if (auto load = dyn_cast<LLVM::LoadOp>(&op)) {
          if (load.getOrdering() == LLVM::AtomicOrdering::not_atomic)
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicLoad;
          markContract();
          continue;
        }

        if (auto store = dyn_cast<LLVM::StoreOp>(&op)) {
          if (store.getOrdering() == LLVM::AtomicOrdering::not_atomic)
            continue;
          contract.kind = LLVMSafetyEffectKind::AtomicStore;
          markContract();
        }
      }
    }
  });
}

void setLythonSafetyMetadataId(llvm::Instruction &inst, int64_t id) {
  llvm::LLVMContext &ctx = inst.getContext();
  llvm::Metadata *operands[] = {
      llvm::MDString::get(ctx, kLythonSafetyMetadataVersion),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), id))};
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

class PySafetyLLVMIRTranslationInterface final
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult amendOperation(Operation *op,
                               ArrayRef<llvm::Instruction *> instructions,
                               NamedAttribute attribute,
                               LLVM::ModuleTranslation &) const final {
    if (attribute.getName() != kPySafetyContractIdAttr)
      return success();
    auto idAttr = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!idAttr)
      return op->emitOpError("py.safety_contract_id must be an integer");
    int64_t id = idAttr.getInt();
    for (llvm::Instruction *instruction : instructions)
      setLythonSafetyMetadataId(*instruction, id);
    return success();
  }
};

void registerPySafetyLLVMIRTranslation(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, py::PyDialect *dialect) {
    dialect->addInterfaces<PySafetyLLVMIRTranslationInterface>();
  });
}

void emitLLVMSafetyVerifierError(llvm::Instruction &inst, llvm::StringRef msg);

LogicalResult
verifyLLVMIRSafetyMetadataPreserved(llvm::Module &llvmModule,
                                    const LLVMSafetyProfile &profile,
                                    llvm::StringRef label) {
  std::vector<unsigned> used(profile.contracts.size(), 0);
  bool failedAny = false;

  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto id = getLythonSafetyMetadataId(inst);
        if (!id) {
          if (getStructuralSafetyEffectKind(inst)) {
            emitLLVMSafetyVerifierError(
                inst, "LLVM atomic safety effect has no preserved MLIR "
                      "contract id");
            failedAny = true;
          }
          continue;
        }
        if (*id < 0 || static_cast<size_t>(*id) >= profile.contracts.size()) {
          emitLLVMSafetyVerifierError(
              inst, "LLVM IR safety effect has no preserved MLIR contract id");
          failedAny = true;
          continue;
        }

        const LLVMSafetyContract &contract =
            profile.contracts[static_cast<size_t>(*id)];
        if (!instructionMatchesContract(inst, contract)) {
          emitLLVMSafetyVerifierError(
              inst, "LLVM IR safety effect shape differs from MLIR contract "
                    "id");
          failedAny = true;
          continue;
        }
        ++used[static_cast<size_t>(*id)];
      }
    }
  }

  for (auto indexed : llvm::enumerate(used)) {
    if (indexed.value() != 0)
      continue;
    const LLVMSafetyContract &contract = profile.contracts[indexed.index()];
    llvm::errs() << "error: " << label << ": MLIR safety contract for @"
                 << contract.functionName << " was not preserved\n";
    failedAny = true;
  }

  return failure(failedAny);
}

LogicalResult
verifyLLVMIRSafetyMetadataAttached(llvm::Module &llvmModule,
                                   const LLVMSafetyProfile &profile) {
  return verifyLLVMIRSafetyMetadataPreserved(
      llvmModule, profile, "LLVM IR safety metadata verifier");
}

void emitLLVMSafetyVerifierError(llvm::Instruction &inst, llvm::StringRef msg) {
  llvm::errs() << "error: LLVM safety verifier: " << msg << "\n";
  if (llvm::Function *function = inst.getFunction())
    llvm::errs() << "  in function: " << function->getName() << "\n";
  llvm::errs() << "  instruction: " << inst << "\n";
}

LogicalResult verifyPostCoroLLVMThreadSafety(llvm::Module &llvmModule,
                                             const LLVMSafetyProfile &profile) {
  if (llvm::verifyModule(llvmModule, &llvm::errs()))
    return failure();

  return verifyLLVMIRSafetyMetadataPreserved(llvmModule, profile,
                                             "post-coro LLVM safety verifier");
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
  if (failed(verifyLLVMIRSafetyMetadataAttached(*llvmModule, safetyProfile)))
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
  registerPySafetyLLVMIRTranslation(registry);

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
  if (failed(verifyLLVMIRSafetyMetadataAttached(*llvmModule, safetyProfile)))
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
