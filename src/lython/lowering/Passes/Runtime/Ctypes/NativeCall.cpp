#include "Runtime/Ctypes/Internal.h"

namespace py::lowering {

using namespace ctypes;

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesNativeCall(
    py::CallOp op, const RuntimeBundle &callable) {
  if (!callable.ctypes ||
      callable.ctypes->kind != RuntimeCtypesEvidence::Kind::Symbol)
    return mlir::failure();
  const RuntimeCtypesEvidence &evidence = *callable.ctypes;
  // A call target is either a named process symbol or a runtime address (a
  // CFuncPtr built from an integer, e.g. a pre-resolved libc pointer).
  bool callThroughAddress =
      evidence.symbolName.empty() && evidence.addressValue &&
      evidence.addressValid && isKnownTrue(evidence.addressValid);
  if (evidence.symbolName.empty() && !callThroughAddress)
    return op.emitError() << "ctypes native call has no symbol or address "
                             "evidence";
  if (!evidence.resultType)
    return op.emitError() << "ctypes call target requires a static restype "
                             "before calling";
  if (!callThroughAddress &&
      (!evidence.processLibrary || !evidence.libraryName.empty()))
    return op.emitError()
           << "ctypes native call lowering currently supports only "
              "ctypes.CDLL(None) process symbols";
  // The compiler itself emits declarations for the C allocator / mem
  // primitives during LLVM lowering, so a same-named ctypes declaration would
  // collide at link. Direct calls to these are rejected with a clear message;
  // resolve the symbol's address (`cast(fn, c_void_p).value`) and call through
  // it instead, or use a stack buffer.
  if (!callThroughAddress) {
    static constexpr llvm::StringLiteral kReserved[] = {
        "malloc",  "calloc",  "realloc", "free",
        "memcpy",  "memmove", "memset",  "aligned_alloc"};
    if (llvm::is_contained(kReserved, evidence.symbolName))
      return op.emitError()
             << "ctypes cannot call the compiler-provided libc symbol '"
             << evidence.symbolName
             << "' by name (it would clash with the runtime's own "
                "declaration); resolve its address with "
                "ctypes.cast(fn, ctypes.c_void_p).value and call through it, "
                "or use a stack buffer";
  }
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "ctypes native arguments",
                                              sources, &unpackedSources)))
    return mlir::failure();
  if (sources.size() != evidence.argTypes.size())
    return op.emitError() << "ctypes call target expects "
                          << evidence.argTypes.size() << " arguments but got "
                          << sources.size();

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  if (!facts)
    return op.emitError()
           << "ctypes native call requires TargetPlatformFacts before "
              "lowering";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Type, 4> nativeArgTypes;
  llvm::SmallVector<mlir::Value, 4> nativeArgs;
  for (auto [index, source] : llvm::enumerate(sources)) {
    if (!source)
      return op.emitError()
             << "ctypes native argument " << index << " has no evidence";
    llvm::StringRef argType = evidence.argTypes[index];
    std::optional<CtypesLayout> layout =
        ctypesStaticLayout(module, argType, facts);
    if (!layout)
      return op.emitError()
             << "ctypes native argument " << index << " type " << argType
             << " has no ABI layout for " << targetFactsLabel(facts);
    if (isIntegerScalarLayout(*layout)) {
      mlir::IntegerType nativeType = nativeIntegerType(builder, *layout);
      std::optional<mlir::Value> nativeValue = extractNativeIntegerArgument(
          op, builder, *source, argType, *layout, facts);
      if (!nativeValue)
        return op.emitError()
               << "ctypes native argument " << index << " for " << argType
               << " requires an exact ctypes scalar cell, a same-width signed "
                  "primitive integer, or a statically range-proven primitive "
                  "integer ("
               << describeNativeArgumentSource(*source) << ")";
      nativeArgTypes.push_back(nativeType);
      nativeArgs.push_back(*nativeValue);
      continue;
    }
    if (isFloatingScalarLayout(*layout)) {
      if (layout->size != 8)
        return op.emitError()
               << "ctypes native argument " << index << " type " << argType
               << " requires an explicit f32 precision conversion";
      std::optional<RuntimeSymbol> unbox =
          manifest.primitive(source->contractName(), "unbox.f64");
      if (!unbox)
        return op.emitError()
               << "ctypes native argument " << index << " for " << argType
               << " requires float-compatible source evidence ("
               << describeNativeArgumentSource(*source) << ")";
      mlir::func::CallOp unboxCall =
          createRuntimeCall(op.getLoc(), *unbox, source->physicalValues());
      if (unboxCall.getNumResults() != 1 ||
          !unboxCall.getResult(0).getType().isF64())
        return op.emitError() << "ctypes native argument " << index
                              << " float unbox primitive must return f64";
      nativeArgTypes.push_back(builder.getF64Type());
      nativeArgs.push_back(unboxCall.getResult(0));
      continue;
    }
    if (isPointerScalarLayout(*layout)) {
      std::optional<mlir::Value> pointer =
          extractNativePointerArgument(op, builder, *source, facts);
      if (!pointer)
        return op.emitError()
               << "ctypes native argument " << index << " for " << argType
               << " requires None, c_void_p, primitive pointer integer, or "
                  "call-region byref evidence ("
               << describeNativeArgumentSource(*source) << ")";
      nativeArgTypes.push_back(nativePointerType(context));
      nativeArgs.push_back(*pointer);
      continue;
    }
    return op.emitError() << "ctypes native argument " << index << " type "
                          << argType
                          << " is not supported by scalar native lowering";
  }

  llvm::SmallVector<mlir::Type, 1> nativeResultTypes;
  std::optional<CtypesLayout> resultLayout;
  if (*evidence.resultType != "types.NoneType") {
    resultLayout = ctypesStaticLayout(module, *evidence.resultType, facts);
    if (!resultLayout)
      return op.emitError()
             << "ctypes native result type " << *evidence.resultType
             << " has no ABI layout for " << targetFactsLabel(facts);
    if (isIntegerScalarLayout(*resultLayout)) {
      if (resultLayout->kind == CtypesLayout::ABIKind::UnsignedInteger &&
          resultLayout->size == 8)
        return op.emitError()
               << "ctypes native unsigned 64-bit result requires Python "
                  "bigint materialization";
      nativeResultTypes.push_back(nativeIntegerType(builder, *resultLayout));
    } else if (isFloatingScalarLayout(*resultLayout)) {
      if (resultLayout->size != 8)
        return op.emitError()
               << "ctypes native result type " << *evidence.resultType
               << " requires an explicit f32 precision conversion";
      nativeResultTypes.push_back(builder.getF64Type());
    } else if (isPointerScalarLayout(*resultLayout)) {
      nativeResultTypes.push_back(nativePointerType(context));
    } else {
      return op.emitError()
             << "ctypes native result type " << *evidence.resultType
             << " is not supported by scalar native lowering";
    }
  }

  mlir::FunctionType functionType =
      builder.getFunctionType(nativeArgTypes, nativeResultTypes);
  builder.setInsertionPoint(op);
  mlir::Value indirectResult;
  mlir::func::CallOp call;
  if (callThroughAddress) {
    // Indirect call through the target address: inttoptr + llvm.call with an
    // explicit LLVM function type. Args are already native LLVM-compatible
    // (iN / !llvm.ptr).
    llvm::SmallVector<mlir::Type, 1> llvmResults;
    if (!nativeResultTypes.empty())
      llvmResults.push_back(nativeResultTypes.front());
    mlir::Type llvmResult =
        llvmResults.empty() ? mlir::Type(mlir::LLVM::LLVMVoidType::get(context))
                            : llvmResults.front();
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(
        llvmResult, nativeArgTypes, /*isVarArg=*/false);
    mlir::Value pointer = mlir::LLVM::IntToPtrOp::create(
        builder, op.getLoc(), nativePointerType(context),
        evidence.addressValue);
    llvm::SmallVector<mlir::Value, 5> operands;
    operands.push_back(pointer);
    operands.append(nativeArgs.begin(), nativeArgs.end());
    auto llvmCall = mlir::LLVM::CallOp::create(builder, op.getLoc(), llvmFnType,
                                               operands);
    if (!nativeResultTypes.empty())
      indirectResult = llvmCall.getResult();
  } else {
    mlir::FailureOr<mlir::func::FuncOp> declaration =
        getOrCreateNativeDeclaration(op, module, builder, evidence.symbolName,
                                     functionType, evidence.argTypes,
                                     *evidence.resultType, evidence.abi,
                                     evidence.processLibrary, *facts);
    if (mlir::failed(declaration))
      return mlir::failure();
    builder.setInsertionPoint(op);
    call = mlir::func::CallOp::create(builder, op.getLoc(), *declaration,
                                      nativeArgs);
  }
  auto resultAt = [&](unsigned index) -> mlir::Value {
    return callThroughAddress ? indirectResult : call.getResult(index);
  };

  if (op.getNumResults() != 1)
    return op.emitError() << "ctypes native call expects one Python result";
  if (*evidence.resultType == "types.NoneType") {
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(0), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
  } else if (isIntegerScalarLayout(*resultLayout)) {
    mlir::Value raw = resultAt(0);
    mlir::IntegerType i64 = builder.getI64Type();
    if (raw.getType() != i64) {
      if (resultLayout->kind == CtypesLayout::ABIKind::UnsignedInteger)
        raw = mlir::arith::ExtUIOp::create(builder, op.getLoc(), i64, raw)
                  .getResult();
      else
        raw = mlir::arith::ExtSIOp::create(builder, op.getLoc(), i64, raw)
                  .getResult();
    }
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), raw, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
  } else if (isFloatingScalarLayout(*resultLayout)) {
    RuntimeBundle result;
    if (mlir::failed(initializeObjectFromRawValues(
            op, runtimeContractType(context, "builtins.float"),
            mlir::ValueRange{resultAt(0)}, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
  } else if (isPointerScalarLayout(*resultLayout)) {
    mlir::Value raw =
        nativePointerToInteger(builder, op.getLoc(), resultAt(0));
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), raw, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
  } else {
    return op.emitError() << "ctypes native call has unsupported result layout";
  }

  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
