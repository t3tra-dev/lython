#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "PyDialect.h.inc"

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);

  llvm::cl::opt<std::string> input(
      llvm::cl::Positional, llvm::cl::desc("<input mlir>"), llvm::cl::Required);
  llvm::cl::opt<std::string> output("o", llvm::cl::desc("Output bytecode"),
                                    llvm::cl::value_desc("filename"),
                                    llvm::cl::Required);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Lython runtime MLIR bytecode emitter\n");

  mlir::DialectRegistry registry;
  registry.insert<
      py::PyDialect, mlir::arith::ArithDialect, mlir::async::AsyncDialect,
      mlir::bufferization::BufferizationDialect, mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
      mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module) {
    llvm::errs() << "failed to parse runtime MLIR module: " << input << "\n";
    return 1;
  }

  std::error_code error;
  llvm::raw_fd_ostream stream(output, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "failed to open bytecode output: " << output << ": "
                 << error.message() << "\n";
    return 1;
  }

  if (mlir::failed(mlir::writeBytecodeToFile(module.get(), stream))) {
    llvm::errs() << "failed to write runtime MLIR bytecode: " << output << "\n";
    return 1;
  }
  return 0;
}
