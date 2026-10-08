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

#include "ClassIds.h"

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
  // that is a class number says whose (`ly.class_id_of = "builtins.ValueError"`)
  // and a function that constructs one marks it (`ly.runtime.class_id`, beside
  // its `ly.runtime.contract`). Both are given their number here, from
  // ClassIds.h, so every image of the runtime agrees on it.
  bool resolved = true;
  module->walk([&](mlir::Operation *op) {
    // A static object's header is a dense table; its word 1 is the class.
    if (auto global = mlir::dyn_cast<mlir::memref::GlobalOp>(op))
      if (auto name = op->getAttrOfType<mlir::StringAttr>("ly.class_id_of")) {
        std::int64_t id = py::class_ids::lookup(name.getValue());
        auto initial = mlir::dyn_cast_if_present<mlir::DenseIntElementsAttr>(
            global.getInitialValueAttr());
        // A byte image (a str's, laid out as LyUnicode_FromStatic reads it)
        // keeps word 1 in bytes 8..15, little-endian, once per record of
        // `ly.class_id_stride` bytes (the whole image when absent).
        if (id >= 0 && initial &&
            initial.getElementType().isInteger(8)) {
          llvm::SmallVector<llvm::APInt, 64> bytes(
              initial.getValues<llvm::APInt>());
          std::int64_t stride = static_cast<std::int64_t>(bytes.size());
          if (auto given =
                  op->getAttrOfType<mlir::IntegerAttr>("ly.class_id_stride"))
            stride = given.getInt();
          if (stride < 16 || bytes.size() % stride != 0) {
            op->emitError() << "ly.class_id_of on a byte image needs records "
                               "of at least 16 bytes";
            resolved = false;
            return;
          }
          for (std::size_t record = 0; record < bytes.size();
               record += static_cast<std::size_t>(stride))
            for (unsigned byte = 0; byte < 8; ++byte)
              bytes[record + 8 + byte] = llvm::APInt(
                  8, (static_cast<std::uint64_t>(id) >> (8 * byte)) & 0xff);
          global.setInitialValueAttr(
              mlir::DenseIntElementsAttr::get(initial.getType(), bytes));
          op->removeAttr("ly.class_id_of");
          op->removeAttr("ly.class_id_stride");
          return;
        }
        if (id < 0 || !initial || initial.getNumElements() < 2) {
          op->emitError() << "ly.class_id_of on a global needs a runtime class "
                             "name and a dense i64 header";
          resolved = false;
          return;
        }
        llvm::SmallVector<llvm::APInt, 8> words(initial.getValues<llvm::APInt>());
        words[1] = llvm::APInt(64, static_cast<std::uint64_t>(id), true);
        global.setInitialValueAttr(
            mlir::DenseIntElementsAttr::get(initial.getType(), words));
        op->removeAttr("ly.class_id_of");
        return;
      }
    if (auto name = op->getAttrOfType<mlir::StringAttr>("ly.class_id_of")) {
      std::int64_t id = py::class_ids::lookup(name.getValue());
      auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op);
      if (id < 0 || !constant || !constant.getType().isInteger(64)) {
        op->emitError() << "ly.class_id_of names '" << name.getValue()
                        << "', which is not a runtime class (ClassIds.h), on "
                           "something that is not an i64 constant";
        resolved = false;
        return;
      }
      constant.setValueAttr(mlir::IntegerAttr::get(constant.getType(), id));
      op->removeAttr("ly.class_id_of");
    }
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op))
      if (mlir::Attribute marker = function->getAttr("ly.runtime.class_id")) {
        auto contract =
            function->getAttrOfType<mlir::StringAttr>("ly.runtime.contract");
        std::int64_t id =
            contract ? py::class_ids::lookup(contract.getValue()) : -1;
        if (id < 0 || !mlir::isa<mlir::UnitAttr>(marker)) {
          function.emitError()
              << "ly.runtime.class_id is a marker beside a ly.runtime.contract "
                 "that ClassIds.h lists; the number comes from the table";
          resolved = false;
          return;
        }
        function->setAttr("ly.runtime.class_id",
                          mlir::IntegerAttr::get(
                              mlir::IntegerType::get(&context, 64), id));
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
