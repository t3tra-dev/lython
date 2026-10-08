#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
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

#include "RuntimeClasses.h"

#include <system_error>

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
      mlir::linalg::LinalgDialect, mlir::math::MathDialect,
      mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect, mlir::transform::TransformDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module) {
    llvm::errs() << "failed to parse runtime MLIR module: " << input << "\n";
    return 1;
  }

  // ⭐ A RUNTIME CLASS IS NAMED, NOT NUMBERED, in the manifests: a constant
  // that is a class word says whose (`ly.class_of = "builtins.ValueError"`),
  // so does a static object's header, and a function that constructs one is
  // marked (`ly.runtime.class`, beside its `ly.runtime.contract`). The
  // lowering makes each the class's type-object address (TypeObjects.h); here
  // every name is only checked against RuntimeClasses.h, so a misspelled class
  // fails the build rather than a program.
  bool resolved = true;
  module->walk([&](mlir::Operation *op) {
    if (auto name = op->getAttrOfType<mlir::StringAttr>("ly.class_of")) {
      bool placed = mlir::isa<mlir::memref::GlobalOp>(op) ||
                    (mlir::isa<mlir::arith::ConstantOp>(op) &&
                     op->getResult(0).getType().isInteger(64));
      if (!py::runtime_classes::isListed(name.getValue()) || !placed) {
        op->emitError() << "ly.class_of names '" << name.getValue()
                        << "'; it must be a runtime class (RuntimeClasses.h), "
                           "on an i64 constant or a static object";
        resolved = false;
      }
    }
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op))
      if (mlir::Attribute marker = function->getAttr("ly.runtime.class")) {
        auto contract =
            function->getAttrOfType<mlir::StringAttr>("ly.runtime.contract");
        if (!contract || !py::runtime_classes::isListed(contract.getValue()) ||
            !mlir::isa<mlir::UnitAttr>(marker)) {
          function.emitError()
              << "ly.runtime.class is a marker beside a ly.runtime.contract "
                 "that RuntimeClasses.h lists";
          resolved = false;
        }
      }
  });
  if (!resolved)
    return 1;

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
