#include "runtime/Detail.h"
#include "runtime/Verification.h"

#include "Native.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace py::lowering {
namespace {

namespace native = py::native;

bool isNativePointerType(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

bool isNoneResultContract(llvm::StringRef contract) {
  return contract == "types.NoneType";
}

bool hasAnyNativeAttr(mlir::Operation *op) {
  static constexpr llvm::StringLiteral attrs[] = {
      native::kNativeSymbolAttr,
      native::kNativeArgTypesAttr,
      native::kNativeResultTypeAttr,
      native::kNativeABIAttr,
      native::kNativeProcessLibraryAttr,
      native::kNativeTargetTripleAttr,
      native::kNativeTargetPointerWidthAttr,
      native::kNativeTargetCLongWidthAttr,
  };
  for (llvm::StringRef attr : attrs)
    if (op->hasAttr(attr))
      return true;
  return false;
}

mlir::LogicalResult
verifyNativeABISurface(mlir::Operation *op, llvm::StringRef abi,
                       bool processLibrary,
                       const native::TargetPlatformFacts &facts) {
  if (!processLibrary)
    return op->emitError()
           << "native declaration does not preserve a concrete library handle; "
              "only process-library symbols are currently verifiable";

  if (abi == "cdecl")
    return mlir::success();
  if (abi == "stdcall") {
    llvm::Triple triple(facts.triple);
    if (!triple.isOSWindows())
      return op->emitError()
             << "stdcall native ABI requires a Windows target, got "
             << facts.triple;
    return mlir::success();
  }
  return op->emitError() << "unsupported native ABI '" << abi << "'";
}

mlir::LogicalResult
verifyNativeTargetSnapshot(mlir::Operation *op,
                           const native::TargetPlatformFacts &facts) {
  auto triple = readRequiredStringAttr(op, native::kNativeTargetTripleAttr,
                                       "native declaration");
  if (mlir::failed(triple))
    return mlir::failure();
  if (*triple != facts.triple)
    return op->emitError() << native::kNativeTargetTripleAttr << " is '"
                           << *triple << "' but module target is '"
                           << facts.triple << "'";

  auto pointerWidth =
      readRequiredUnsignedIntegerAttr(op, native::kNativeTargetPointerWidthAttr,
                                      "native declaration");
  if (mlir::failed(pointerWidth))
    return mlir::failure();
  if (*pointerWidth != facts.pointerWidth)
    return op->emitError() << native::kNativeTargetPointerWidthAttr << " is "
                           << *pointerWidth
                           << " but module target pointer width is "
                           << facts.pointerWidth;

  auto cLongWidth =
      readRequiredUnsignedIntegerAttr(op, native::kNativeTargetCLongWidthAttr,
                                      "native declaration");
  if (mlir::failed(cLongWidth))
    return mlir::failure();
  if (*cLongWidth != facts.cLongWidth)
    return op->emitError() << native::kNativeTargetCLongWidthAttr << " is "
                           << *cLongWidth
                           << " but module target c_long width is "
                           << facts.cLongWidth;

  return mlir::success();
}

mlir::LogicalResult
verifyNativeABIType(mlir::Operation *op, mlir::Type loweredType,
                    llvm::StringRef contract, bool result,
                    const native::TargetPlatformFacts &facts) {
  std::optional<native::NativeABIType> layout =
      native::ctypesScalarLayout(contract, facts);
  if (!layout)
    return op->emitError() << "native " << (result ? "result" : "argument")
                           << " contract '" << contract
                           << "' has no scalar ctypes ABI layout";

  unsigned bits = static_cast<unsigned>(layout->size * 8);
  if (native::isIntegerABI(*layout)) {
    if (result && layout->kind == native::NativeABIKind::UnsignedInteger &&
        layout->size == 8)
      return op->emitError()
             << "native unsigned 64-bit result '" << contract
             << "' requires explicit Python bigint materialization";
    if (!isIntegerType(loweredType, bits))
      return op->emitError()
             << "native " << (result ? "result" : "argument") << " contract '"
             << contract << "' expects i" << bits << ", got " << loweredType;
    return mlir::success();
  }

  if (native::isFloatingABI(*layout)) {
    if (layout->size != 8)
      return op->emitError() << "native " << (result ? "result" : "argument")
                             << " contract '" << contract
                             << "' requires an explicit f32 precision "
                                "conversion before ABI verification";
    if (!loweredType.isF64())
      return op->emitError()
             << "native " << (result ? "result" : "argument") << " contract '"
             << contract << "' expects f64, got " << loweredType;
    return mlir::success();
  }

  if (native::isPointerABI(*layout)) {
    if (!isNativePointerType(loweredType))
      return op->emitError()
             << "native " << (result ? "result" : "argument") << " contract '"
             << contract << "' expects !llvm.ptr, got " << loweredType;
    return mlir::success();
  }

  return op->emitError() << "native " << (result ? "result" : "argument")
                         << " contract '" << contract
                         << "' has an unsupported ABI kind";
}

mlir::LogicalResult
verifyNativeFunction(mlir::Operation *op, llvm::StringRef functionName,
                     llvm::ArrayRef<mlir::Type> inputTypes,
                     llvm::ArrayRef<mlir::Type> resultTypes,
                     const native::TargetPlatformFacts &facts) {
  auto symbol =
      readRequiredStringAttr(op, native::kNativeSymbolAttr,
                             "native declaration");
  if (mlir::failed(symbol))
    return mlir::failure();
  if (*symbol != functionName)
    return op->emitError() << native::kNativeSymbolAttr << " is '" << *symbol
                           << "' but declaration symbol is @" << functionName;

  auto argTypes =
      readRequiredStringArrayAttr(op, native::kNativeArgTypesAttr,
                                  "native declaration");
  if (mlir::failed(argTypes))
    return mlir::failure();
  auto resultType =
      readRequiredStringAttr(op, native::kNativeResultTypeAttr,
                             "native declaration");
  if (mlir::failed(resultType))
    return mlir::failure();
  auto abi =
      readRequiredStringAttr(op, native::kNativeABIAttr, "native declaration");
  if (mlir::failed(abi))
    return mlir::failure();
  auto processLibrary =
      readRequiredBoolAttr(op, native::kNativeProcessLibraryAttr,
                           "native declaration");
  if (mlir::failed(processLibrary))
    return mlir::failure();

  if (mlir::failed(verifyNativeTargetSnapshot(op, facts)))
    return mlir::failure();
  if (mlir::failed(verifyNativeABISurface(op, *abi, *processLibrary, facts)))
    return mlir::failure();

  if (inputTypes.size() != argTypes->size())
    return op->emitError() << "native declaration has " << inputTypes.size()
                           << " lowered inputs but " << argTypes->size()
                           << " declared ctypes argument contracts";
  for (auto [index, loweredType] : llvm::enumerate(inputTypes)) {
    if (mlir::failed(verifyNativeABIType(op, loweredType, (*argTypes)[index],
                                         /*result=*/false, facts)))
      return mlir::failure();
  }

  if (isNoneResultContract(*resultType)) {
    if (!resultTypes.empty())
      return op->emitError()
             << "native declaration result contract is types.NoneType but "
             << resultTypes.size() << " lowered results are present";
    return mlir::success();
  }

  if (resultTypes.size() != 1)
    return op->emitError() << "native declaration result contract '"
                           << *resultType
                           << "' requires exactly one lowered result, got "
                           << resultTypes.size();
  return verifyNativeABIType(op, resultTypes.front(), *resultType,
                             /*result=*/true, facts);
}

mlir::LogicalResult
verifyNativeFuncOp(mlir::func::FuncOp function,
                   const native::TargetPlatformFacts &facts) {
  mlir::FunctionType type = function.getFunctionType();
  return verifyNativeFunction(function, function.getSymName(), type.getInputs(),
                              type.getResults(), facts);
}

mlir::LogicalResult
verifyNativeLLVMFuncOp(mlir::LLVM::LLVMFuncOp function,
                       const native::TargetPlatformFacts &facts) {
  mlir::LLVM::LLVMFunctionType type = function.getFunctionType();
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  inputTypes.reserve(type.getNumParams());
  for (unsigned index = 0, count = type.getNumParams(); index < count; ++index)
    inputTypes.push_back(type.getParamType(index));
  return verifyNativeFunction(function, function.getSymName(), inputTypes,
                              function.getResultTypes(), facts);
}

mlir::LogicalResult verifyNativeDeclarations(mlir::ModuleOp module) {
  if (mlir::failed(native::verifyTargetPlatformFacts(module)))
    return mlir::failure();
  std::optional<native::TargetPlatformFacts> facts =
      native::readTargetPlatformFacts(module);
  if (!facts)
    return module.emitError()
           << "verified target platform facts could not be read";

  if (mlir::failed(walkVerifyOperations(module, [](mlir::Operation *op)
                                                   -> mlir::LogicalResult {
        if (!hasAnyNativeAttr(op))
          return mlir::success();
        if (mlir::isa<mlir::func::FuncOp, mlir::LLVM::LLVMFuncOp>(op))
          return mlir::success();
        return op->emitError()
               << "ly.native.* attributes are only valid on function "
                  "declarations";
      })))
    return mlir::failure();

  if (mlir::failed(walkVerify<mlir::func::FuncOp>(
          module, [&](mlir::func::FuncOp function) {
            if (!hasAnyNativeAttr(function))
              return mlir::success();
            return verifyNativeFuncOp(function, *facts);
          })))
    return mlir::failure();

  return walkVerify<mlir::LLVM::LLVMFuncOp>(
      module, [&](mlir::LLVM::LLVMFuncOp function) {
        if (!hasAnyNativeAttr(function))
          return mlir::success();
        return verifyNativeLLVMFuncOp(function, *facts);
      });
}

class NativeVerificationPass
    : public mlir::PassWrapper<NativeVerificationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeVerificationPass)

  llvm::StringRef getArgument() const final {
    return "lython-native-verification";
  }
  llvm::StringRef getDescription() const final {
    return "verify native target facts and ctypes declaration ABI contracts";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyNativeDeclarations(getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass() {
  return std::make_unique<lowering::NativeVerificationPass>();
}

} // namespace py
