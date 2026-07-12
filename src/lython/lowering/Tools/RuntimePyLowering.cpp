// Build-time pre-lowering for runtime modules WRITTEN IN PYTHON
// (src/lython/runtime/lib/*.py).
//
// Compiles one runtime .py through the full Lython pipeline (parse -> emit ->
// runtime lowering -> refcount insertion/elision -> verifiers -> symbol DCE)
// for an explicit --target triple, INCLUDING the final LLVM-dialect
// conversion (the triple is known here, so the artifact is pure LLVM
// dialect). It is serialized as MLIR bytecode and embedded into lyc as a
// NativeMLIRBytecode module riding the same link path as the hand-written
// native support code (now built by RuntimeSupportBuilder).
//
// Platform branches inside the .py (`if sys.platform == ...`) fold at compile
// time against the --target triple, so ONE Python source yields one artifact
// per platform (<stem>_darwin / _linux / _windows), selected at link exactly
// like the hand-written per-platform support modules.
//
// Module-level statements other than imports and function definitions are
// rejected: a runtime module must not run code at import; __main__ is dropped
// from the artifact. Top-level functions become PUBLIC symbols (callable from
// runtime manifests); everything they pull in from the embedded runtime
// stays private and links as internal (duplicate-safe).

#include "Common/LoweringPipeline.h"
#include "Common/RuntimeSupport.h"

#include "Emitter.h"
#include "Parser.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "PyDialect.h.inc"

namespace {

mlir::LogicalResult stampTargetFacts(mlir::ModuleOp module,
                                     const llvm::Triple &triple) {
  std::uint64_t pointerWidth =
      triple.isArch64Bit() ? 64 : (triple.isArch32Bit() ? 32 : 0);
  if (pointerWidth == 0) {
    llvm::errs() << "error: cannot derive pointer width for target triple '"
                 << triple.normalize() << "'\n";
    return mlir::failure();
  }
  std::uint64_t cLongWidth =
      triple.isOSWindows() ? 32 : (pointerWidth == 64 ? 64 : 32);
  mlir::Builder builder(module.getContext());
  module->setAttr("ly.target.triple",
                  builder.getStringAttr(triple.normalize()));
  module->setAttr("ly.target.pointer_width",
                  builder.getI64IntegerAttr(pointerWidth));
  module->setAttr("ly.target.c_long_width",
                  builder.getI64IntegerAttr(cLongWidth));
  return mlir::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);

  llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                   llvm::cl::desc("<input python>"),
                                   llvm::cl::Required);
  llvm::cl::opt<std::string> output("o", llvm::cl::desc("Output bytecode"),
                                    llvm::cl::value_desc("filename"),
                                    llvm::cl::Required);
  llvm::cl::opt<std::string> target("target",
                                    llvm::cl::desc("Target triple"),
                                    llvm::cl::value_desc("triple"),
                                    llvm::cl::Required);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Lython runtime Python pre-lowering\n");

  mlir::registerTransformsPasses();

  mlir::DialectRegistry registry;
  registry.insert<py::PyDialect, mlir::affine::AffineDialect,
                  mlir::async::AsyncDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                  mlir::vector::VectorDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::LLVM::LLVMDialect, mlir::math::MathDialect,
                  mlir::transform::TransformDialect>();
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerAllToLLVMIRTranslations(registry);

  mlir::MLIRContext context(registry);

  auto file = llvm::MemoryBuffer::getFile(input);
  if (!file) {
    llvm::errs() << "error: could not open input file '" << input << "'\n";
    return 1;
  }

  lython::parser::ParseResult parsed =
      lython::parser::parse((*file)->getBuffer(), input);
  if (!parsed.ok()) {
    for (const lython::parser::Diagnostic &diagnostic : parsed.diagnostics)
      llvm::errs() << input << ':' << diagnostic.location.line << ':'
                   << diagnostic.location.column
                   << ": parse error: " << diagnostic.message << "\n";
    return 1;
  }

  llvm::Triple triple{llvm::Triple(target).normalize()};

  lython::emitter::EmitOptions emitOptions;
  emitOptions.targetTriple = triple.normalize();
  lython::emitter::EmitResult emitted = lython::emitter::emitModule(
      *parsed.tree, context, "__main__", input, emitOptions);
  if (!emitted.ok()) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics)
      llvm::errs() << input << ':' << diagnostic.location.line << ':'
                   << diagnostic.location.column
                   << ": emit error: " << diagnostic.message << "\n";
    return 1;
  }
  mlir::ModuleOp module = *emitted.module;

  // A runtime module must not execute code at import: __main__ may only
  // contain effect-free residue (the docstring constant, binding refs, the
  // implicit return) -- no calls, no stores, no control flow.
  if (auto mainFunction = module.lookupSymbol<mlir::func::FuncOp>("__main__")) {
    for (mlir::Block &block : mainFunction.getBody())
      for (mlir::Operation &op : block) {
        bool allowed = mlir::isa<mlir::func::ReturnOp>(op) ||
                       op.hasTrait<mlir::OpTrait::ConstantLike>();
        if (!allowed && op.getDialect() &&
            op.getDialect()->getNamespace() == "py") {
          llvm::StringRef name = op.getName().getStringRef();
          allowed = name == "py.str.constant" || name == "py.int.constant" ||
                    name == "py.bool.constant" || name == "py.binding.ref" ||
                    name == "py.none" || name == "py.class.upcast";
          // A module-global DECLARATION (`NAME: int = 0`) is allowed: the
          // storage cell lowers to a zero-initialized llvm.global and this
          // __main__ is never executed, so only a zero initializer keeps the
          // declared and effective values in agreement.
          if (!allowed && name == "py.global.set") {
            mlir::Value init = op.getOperand(0);
            while (mlir::Operation *def = init.getDefiningOp()) {
              if (def->getName().getStringRef() == "py.class.upcast") {
                init = def->getOperand(0);
                continue;
              }
              if (def->getName().getStringRef() == "py.int.constant")
                if (auto value = def->getAttrOfType<mlir::StringAttr>("value"))
                  allowed = value.getValue() == "0";
              break;
            }
            if (!allowed) {
              op.emitError()
                  << "runtime lib module globals must be declared with a "
                     "zero initializer (`NAME: int = 0`)";
              return 1;
            }
          }
        }
        if (!allowed) {
          op.emitError() << "runtime lib module must not run module-level "
                            "code; only imports and function definitions are "
                            "allowed";
          return 1;
        }
      }
  }

  // Top-level functions are the module's exports: public visibility protects
  // them from symbol DCE and makes them linkable from runtime manifests.
  llvm::StringSet<> exportedNames;
  for (auto function : module.getOps<mlir::func::FuncOp>())
    if (function.getSymName() != "__main__" && !function.isDeclaration()) {
      function.setPublic();
      exportedNames.insert(function.getSymName());
    }

  if (mlir::failed(stampTargetFacts(module, triple)))
    return 1;

  py::TensorLoweringTarget tensorTarget;
  py::LoweringPipelineOptions options;
  py::IRDumpConfig irDump = py::IRDumpConfig::fromEnv();
  if (mlir::failed(
          py::runLoweringPipeline(module, tensorTarget, irDump, options))) {
    llvm::errs() << "error: runtime pre-lowering pipeline failed for '"
                 << input << "'\n";
    return 1;
  }

  // Drop the (empty) module entry point from the artifact.
  if (auto mainFunction = module.lookupSymbol<mlir::func::FuncOp>("__main__"))
    mainFunction.erase();
  if (auto mainFunction =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__main__"))
    mainFunction.erase();

  // Everything the exports pulled in from the embedded runtime (manifest
  // implementations, globals) is a COPY of definitions the user program also
  // imports: internalize them so the artifact links without duplicate-symbol
  // clashes (the LLVM linker renames internal symbols). Only the .py-defined
  // exports stay external. Constraint (documented): runtime lib code must
  // not depend on SHARED MUTABLE runtime globals -- an internalized copy
  // would be a second instance.
  module.walk([&](mlir::LLVM::LLVMFuncOp function) {
    if (function.isExternal())
      return; // declarations resolve at final link (libc, runtime support)
    if (exportedNames.contains(function.getName()))
      return;
    function.setLinkage(mlir::LLVM::Linkage::Internal);
  });
  module.walk([&](mlir::LLVM::GlobalOp global) {
    bool isDefinition =
        global.getValueOrNull() || !global.getInitializerRegion().empty();
    if (!isDefinition || exportedNames.contains(global.getSymName()))
      return;
    global.setLinkage(mlir::LLVM::Linkage::Internal);
  });

  // Strip attributes that carry py-dialect types (callable_type and friends):
  // the artifact is a NATIVE runtime module parsed at link time in a context
  // without the py dialect.
  module.walk([&](mlir::Operation *op) {
    llvm::SmallVector<mlir::StringAttr, 4> drop;
    for (mlir::NamedAttribute named : op->getAttrs()) {
      bool hasPyType = false;
      mlir::AttrTypeWalker walker;
      walker.addWalk([&](mlir::Type type) {
        if (type.getDialect().getNamespace() == "py")
          hasPyType = true;
      });
      walker.walk(named.getValue());
      if (hasPyType)
        drop.push_back(named.getName());
    }
    for (mlir::StringAttr name : drop)
      op->removeAttr(name);
  });
  std::error_code error;
  llvm::raw_fd_ostream stream(output, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "failed to open bytecode output: " << output << ": "
                 << error.message() << "\n";
    return 1;
  }
  if (mlir::failed(mlir::writeBytecodeToFile(module, stream))) {
    llvm::errs() << "failed to write runtime MLIR bytecode: " << output
                 << "\n";
    return 1;
  }
  return 0;
}
