#include "Runtime/Ctypes/CallbackThunks.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace py::lowering::ctypes {

namespace {

// "s32"/"u64" -> iN, "p" -> !llvm.ptr, "void" -> nullopt (result only).
std::optional<mlir::Type> nativeTypeForCode(mlir::MLIRContext *context,
                                            llvm::StringRef code) {
  if (code == "p")
    return mlir::LLVM::LLVMPointerType::get(context);
  if (code.size() < 2 || (code.front() != 's' && code.front() != 'u'))
    return std::nullopt;
  unsigned width = 0;
  if (code.drop_front().getAsInteger(10, width) || width == 0)
    return std::nullopt;
  return mlir::IntegerType::get(context, width);
}

// An LLVM callee is async-signal-UNSAFE if it can allocate or drive the
// garbage-collected object runtime: the Lython runtime object routines (`Ly*`)
// and the libc allocator family. Everything else (int/pointer arithmetic,
// control flow, plain memory loads/stores, and other libc calls such as
// write/_exit that the callback needs) is permitted.
bool isSignalUnsafeCallee(llvm::StringRef callee) {
  if (callee.starts_with("Ly"))
    return true;
  return callee == "malloc" || callee == "calloc" || callee == "realloc" ||
         callee == "free" || callee == "aligned_alloc" ||
         callee == "posix_memalign" || callee == "strdup";
}

mlir::LogicalResult verifyCallbackSignalSafety(mlir::ModuleOp module,
                                               mlir::LLVM::LLVMFuncOp entry,
                                               llvm::StringRef displayName) {
  llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 8> worklist{entry};
  llvm::DenseSet<mlir::Operation *> visited{entry};
  while (!worklist.empty()) {
    mlir::LLVM::LLVMFuncOp function = worklist.pop_back_val();
    mlir::LogicalResult result = mlir::success();
    function.walk([&](mlir::LLVM::CallOp call) {
      std::optional<llvm::StringRef> callee = call.getCallee();
      if (!callee)
        return mlir::WalkResult::advance();
      if (isSignalUnsafeCallee(*callee)) {
        call.emitError()
            << "ctypes callback '" << displayName
            << "' is not async-signal-safe: it calls '" << *callee
            << "', which allocates or drives the garbage-collected object "
               "runtime. A signal handler may only run allocation-free code "
               "(integer/pointer arithmetic and raw native calls such as "
               "write/_exit); it may not use str/print, build objects, or "
               "raise exceptions.";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      // Recurse into module-internal callees (the callback's own helpers);
      // external declarations (libc, ctypes symbols) have no body to inspect.
      if (auto next =
              module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(*callee))
        if (!next.isExternal() && visited.insert(next).second)
          worklist.push_back(next);
      return mlir::WalkResult::advance();
    });
    if (mlir::failed(result))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace

// Fill `ly.symbol_address` placeholders (from cast(named-symbol, pointer)):
// each declared `() -> i64` function returns the runtime address of a linked
// symbol. Runs at the LLVM layer where addresses exist; creates an external
// `llvm.func` declaration for the symbol if none exists.
mlir::LogicalResult materializeSymbolAddresses(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 4> placeholders;
  module.walk([&](mlir::LLVM::LLVMFuncOp function) {
    if (function->hasAttr("ly.symbol_address"))
      placeholders.push_back(function);
  });
  if (placeholders.empty())
    return mlir::success();

  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  mlir::Type i64 = builder.getI64Type();
  for (mlir::LLVM::LLVMFuncOp placeholder : placeholders) {
    auto symbol =
        placeholder->getAttrOfType<mlir::StringAttr>("ly.symbol_address");
    if (!symbol)
      return placeholder.emitError() << "symbol address placeholder is missing "
                                        "its symbol name";
    // Ensure a declaration of the target symbol exists to take its address.
    if (!module.lookupSymbol(symbol.getValue())) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(module.getBody());
      // Unknown location: an external declaration must not carry a !dbg
      // subprogram attachment.
      mlir::LLVM::LLVMFuncOp::create(
          builder, builder.getUnknownLoc(), symbol.getValue(),
          mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(context), {}, /*isVarArg=*/true));
    }
    mlir::Block *entry = placeholder.addEntryBlock(builder);
    builder.setInsertionPointToStart(entry);
    mlir::Value pointer = mlir::LLVM::AddressOfOp::create(
        builder, placeholder.getLoc(),
        mlir::LLVM::LLVMPointerType::get(context), symbol.getValue());
    mlir::Value address =
        mlir::LLVM::PtrToIntOp::create(builder, placeholder.getLoc(), i64,
                                       pointer);
    mlir::LLVM::ReturnOp::create(builder, placeholder.getLoc(), address);
    placeholder->removeAttr("ly.symbol_address");
  }
  return mlir::success();
}

mlir::LogicalResult materializeCallbackThunks(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 4> placeholders;
  module.walk([&](mlir::LLVM::LLVMFuncOp function) {
    if (function->hasAttr("ly.callback.thunk"))
      placeholders.push_back(function);
  });
  if (placeholders.empty())
    return mlir::success();

  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();

  for (mlir::LLVM::LLVMFuncOp placeholder : placeholders) {
    auto thunkName =
        placeholder->getAttrOfType<mlir::StringAttr>("ly.callback.thunk");
    auto targetName =
        placeholder->getAttrOfType<mlir::FlatSymbolRefAttr>(
            "ly.callback.target");
    auto argCodes =
        placeholder->getAttrOfType<mlir::ArrayAttr>("ly.callback.args");
    auto resultCode =
        placeholder->getAttrOfType<mlir::StringAttr>("ly.callback.result");
    if (!thunkName || !targetName || !argCodes || !resultCode)
      return placeholder.emitError()
             << "callback address placeholder has malformed attributes";

    auto target =
        module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(targetName.getAttr());
    if (!target)
      return placeholder.emitError()
             << "callback target '" << targetName.getValue()
             << "' was not lowered to an LLVM function";

    // SIGNAL-SAFETY POLICY (see docs/lowering-architecture.md): a ctypes
    // callback may fire in an async-signal context, so its body must be
    // async-signal-safe -- it must NOT allocate or touch the garbage-collected
    // heap. The transitive closure of the callback body must contain no call
    // to a runtime object/allocation routine. This is verified over the FINAL
    // LLVM IR (every boxing helper is now a concrete `Ly*`/`malloc` call), so
    // the check cannot be fooled by a helper that boxes internally.
    if (mlir::failed(verifyCallbackSignalSafety(module, target,
                                                targetName.getValue())))
      return mlir::failure();

    // The primitive-ABI clone takes (i64 raw, i1 valid) per parameter and
    // returns struct<(i64, i1)>.
    mlir::LLVM::LLVMFunctionType targetType = target.getFunctionType();
    if (targetType.getNumParams() != 2 * argCodes.size())
      return placeholder.emitError()
             << "callback target '" << targetName.getValue() << "' expects "
             << targetType.getNumParams() << " ABI inputs for "
             << argCodes.size() << " callback parameters";

    llvm::SmallVector<mlir::Type, 4> nativeArgTypes;
    for (mlir::Attribute codeAttr : argCodes) {
      auto code = mlir::dyn_cast<mlir::StringAttr>(codeAttr);
      std::optional<mlir::Type> nativeType =
          code ? nativeTypeForCode(context, code.getValue()) : std::nullopt;
      if (!nativeType)
        return placeholder.emitError()
               << "callback argument code is not a scalar native type";
      nativeArgTypes.push_back(*nativeType);
    }
    std::optional<mlir::Type> nativeResultType;
    if (resultCode.getValue() != "void") {
      nativeResultType = nativeTypeForCode(context, resultCode.getValue());
      if (!nativeResultType)
        return placeholder.emitError()
               << "callback result code is not a scalar native type";
    }

    builder.setInsertionPointToEnd(module.getBody());
    auto thunkType = mlir::LLVM::LLVMFunctionType::get(
        nativeResultType ? *nativeResultType
                         : mlir::LLVM::LLVMVoidType::get(context),
        nativeArgTypes, /*isVarArg=*/false);
    auto thunk = mlir::LLVM::LLVMFuncOp::create(
        builder, placeholder.getLoc(), thunkName.getValue(), thunkType,
        mlir::LLVM::Linkage::Internal);
    mlir::Block *entry = thunk.addEntryBlock(builder);
    builder.setInsertionPointToStart(entry);
    mlir::Value validTrue = mlir::LLVM::ConstantOp::create(
        builder, thunk.getLoc(), i1, builder.getBoolAttr(true));
    llvm::SmallVector<mlir::Value, 8> callOperands;
    for (auto [index, codeAttr] : llvm::enumerate(argCodes)) {
      llvm::StringRef code = mlir::cast<mlir::StringAttr>(codeAttr).getValue();
      mlir::Value argument = entry->getArgument(index);
      mlir::Value widened;
      if (code == "p") {
        widened = mlir::LLVM::PtrToIntOp::create(builder, thunk.getLoc(), i64,
                                                 argument);
      } else if (argument.getType() == i64) {
        widened = argument;
      } else if (code.front() == 'u') {
        widened = mlir::LLVM::ZExtOp::create(builder, thunk.getLoc(), i64,
                                             argument);
      } else {
        widened = mlir::LLVM::SExtOp::create(builder, thunk.getLoc(), i64,
                                             argument);
      }
      callOperands.push_back(widened);
      callOperands.push_back(validTrue);
    }
    auto call = mlir::LLVM::CallOp::create(builder, thunk.getLoc(), target,
                                           callOperands);
    if (!nativeResultType) {
      mlir::LLVM::ReturnOp::create(builder, thunk.getLoc(),
                                   mlir::ValueRange{});
    } else {
      mlir::Value raw = mlir::LLVM::ExtractValueOp::create(
          builder, thunk.getLoc(), call.getResult(), 0);
      mlir::Value narrowed;
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(*nativeResultType))
        narrowed = mlir::LLVM::IntToPtrOp::create(builder, thunk.getLoc(),
                                                  *nativeResultType, raw);
      else if (*nativeResultType == i64)
        narrowed = raw;
      else
        narrowed = mlir::LLVM::TruncOp::create(builder, thunk.getLoc(),
                                               *nativeResultType, raw);
      mlir::LLVM::ReturnOp::create(builder, thunk.getLoc(), narrowed);
    }

    // Fill the address placeholder: addressof(thunk) as an i64.
    mlir::Block *addressEntry = placeholder.addEntryBlock(builder);
    builder.setInsertionPointToStart(addressEntry);
    mlir::Value pointer = mlir::LLVM::AddressOfOp::create(
        builder, placeholder.getLoc(), thunk);
    mlir::Value address = mlir::LLVM::PtrToIntOp::create(
        builder, placeholder.getLoc(), i64, pointer);
    mlir::LLVM::ReturnOp::create(builder, placeholder.getLoc(), address);

    placeholder->removeAttr("ly.callback.thunk");
    placeholder->removeAttr("ly.callback.target");
    placeholder->removeAttr("ly.callback.args");
    placeholder->removeAttr("ly.callback.result");
  }
  return mlir::success();
}

} // namespace py::lowering::ctypes
