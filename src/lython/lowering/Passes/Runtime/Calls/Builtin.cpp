#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

mlir::FailureOr<bool> RuntimeBundleLowerer::emitSourceClassReprCall(
    mlir::Operation *op, const RuntimeBundle &object, RuntimeBundle &result) {
  py::ClassOp classOp = RuntimeBundleLowerer::classForContract(object.contract);
  if (!classOp)
    return false;
  std::optional<std::string> symbol =
      RuntimeBundleLowerer::classMethodSymbol(classOp, "__repr__");
  if (!symbol)
    return false;
  auto function = module.lookupSymbol<mlir::func::FuncOp>(*symbol);
  if (!function || function.isExternal())
    return false;
  // The compiled method takes the instance's physical values (self box and
  // field views) directly; a shape mismatch is a real ABI error, not a miss.
  mlir::FunctionType type = function.getFunctionType();
  llvm::ArrayRef<mlir::Value> physicals = object.physicalValues();
  if (type.getNumInputs() != physicals.size())
    return op->emitError() << "source class __repr__ for "
                           << object.contractName() << " expects "
                           << type.getNumInputs() << " values, receiver has "
                           << physicals.size();
  for (auto [input, physical] : llvm::zip(type.getInputs(), physicals))
    if (physical.getType() != input)
      return op->emitError()
             << "source class __repr__ for " << object.contractName()
             << " receiver value type " << physical.getType()
             << " does not match parameter type " << input;
  builder.setInsertionPoint(op);
  mlir::func::CallOp call = mlir::func::CallOp::create(
      builder, op->getLoc(), function,
      llvm::SmallVector<mlir::Value, 4>(physicals.begin(), physicals.end()));
  if (mlir::failed(bundleRuntimeResults(
          op, runtimeContractType(context, "builtins.str"), call, result)))
    return mlir::failure();
  return true;
}

namespace {

// ⭐ KEEP THE TYPE ARGUMENTS THE OP'S OWN RESULT ALREADY CARRIES. The manifest
// answers with a contract NAME, and rebuilding a type from a name drops the
// parameters: `list(xs)` over a `list[str]` came out of the builtin as a bare
// `builtins.list`, and storing that into a declared `list[str]` field was
// "attribute value builtins.list is not assignable to field
// builtins.list<str>". The manifest METHOD path already keeps them
// (`bindRuntimeCallResult`); the builtin path rebuilt from the name.
//
// ⛔ Only when the names AGREE. Where the manifest's declared contract differs
// from the op's -- the fallbacks above pick `symbol.resultContract` precisely
// when the op says `object` -- the manifest's answer is the one to trust.
mlir::Type builtinResultTypeFor(mlir::Value resultValue,
                                llvm::StringRef resultContract) {
  if (resultValue &&
      runtimeContractName(resultValue.getType()) == resultContract)
    return resultValue.getType();
  return runtimeContractType(resultValue.getContext(), resultContract);
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::emitBoxedReprHookCall(
    mlir::Operation *op, const RuntimeBundle &object, RuntimeBundle &result) {
  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectPhysicalHeader(op, object.objectValue);
  if (mlir::failed(header))
    return mlir::failure();
  auto headerType = mlir::cast<mlir::MemRefType>(header->getType());
  if (!headerType.getLayout().isIdentity())
    return op->emitError()
           << "erased object repr requires an identity-layout box, got "
           << header->getType();

  mlir::func::FuncOp hook =
      module.lookupSymbol<mlir::func::FuncOp>("__ly_repr_boxed_by_contract");
  if (!hook) {
    // Declaring the hook is what requests its generation at the end of the
    // pass (the same declaration a merged container __repr__ carries).
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
    mlir::Type i64 = builder.getI64Type();
    auto strHeader = mlir::MemRefType::get({2}, i64);
    auto strBytes = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                          builder.getI8Type());
    hook = mlir::func::FuncOp::create(
        builder, module.getLoc(), "__ly_repr_boxed_by_contract",
        builder.getFunctionType({ptrType, i64},
                                {strHeader, strBytes, builder.getI1Type()}));
    hook.setPrivate();
    hook->setAttr("ly.ownership.owned_results", builder.getI64ArrayAttr({0}));
    // Str and bytes share the (header, byte payload) shape; the declared
    // result contract disambiguates the owned result's deallocator for the
    // refcount insertion pass.
    hook->setAttr("ly.runtime.result_contract",
                  builder.getStringAttr("builtins.str"));
  }

  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::Value classSlot =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  mlir::Value classId =
      mlir::memref::LoadOp::create(builder, loc, *header, classSlot)
          .getResult();
  mlir::Value pointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                           *header);
  mlir::Value pointerWord = mlir::arith::IndexCastOp::create(
                                builder, loc, builder.getI64Type(),
                                pointerIndex)
                                .getResult();
  mlir::Value boxPointer =
      mlir::LLVM::IntToPtrOp::create(
          builder, loc, mlir::LLVM::LLVMPointerType::get(context), pointerWord)
          .getResult();
  mlir::func::CallOp call = mlir::func::CallOp::create(
      builder, loc, hook, mlir::ValueRange{boxPointer, classId});
  // The hook consumed only a raw pointer word: pin the box's liveness past
  // the call with an explicit touch use (the liveness analysis cannot see
  // raw-pointer uses).
  if (std::optional<RuntimeSymbol> touch =
          manifest.primitive("builtins.object", "touch");
      touch && header->getType() ==
                   touch->function.getFunctionType().getInput(0))
    RuntimeBundleLowerer::createRuntimeCall(loc, *touch,
                                            mlir::ValueRange{*header});
  mlir::cf::AssertOp::create(builder, loc, call.getResult(2),
                             "repr: boxed object has no conforming __repr__");
  // Register the owned str result through the shared result bundling so the
  // refcount machinery roots it (including the unwind cleanup path). The
  // trailing handled flag is not part of the str's value group.
  return bundleRuntimeResults(op, runtimeContractType(context, "builtins.str"),
                              call.getResults().take_front(2), result);
}

mlir::LogicalResult RuntimeBundleLowerer::collectSingleBuiltinArgument(
    py::CallOp op, const RuntimeSymbol &symbol,
    const RuntimeBundle *&argument) {
  const RuntimeBundle *posargs =
      RuntimeBundleLowerer::bundleFor(op.getPosargs());
  if (!posargs || posargs->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' requires packed positional arguments";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();
  if (posargs->aggregateOperands.size() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' expects exactly one positional argument";

  // The argument can be an as-yet-unlowered merge block argument (e.g. a
  // ternary flowing straight into the call): demand its bundle first.
  if (mlir::failed(RuntimeBundleLowerer::ensureValueBundle(
          op, posargs->aggregateOperands[0])))
    return mlir::failure();
  argument = RuntimeBundleLowerer::bundleFor(posargs->aggregateOperands[0]);
  if (!argument)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' argument has no lowered runtime bundle";
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBuiltinMethodCall(py::CallOp op,
                                             const RuntimeSymbol &symbol) {
  if (op.getNumResults() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' method lowering must produce one result";

  const RuntimeBundle *argument = nullptr;
  if (mlir::failed(collectSingleBuiltinArgument(op, symbol, argument)))
    return mlir::failure();
  const RuntimeBundle *receiver =
      RuntimeBundleLowerer::concreteObjectForOwnership(*argument);
  if (!receiver)
    receiver = argument;

  if (symbol.builtinName == "repr" && symbol.builtinMethod == "__repr__") {
    RuntimeBundle rendered;
    mlir::FailureOr<bool> sourceRepr =
        RuntimeBundleLowerer::emitSourceClassReprCall(op, *receiver, rendered);
    if (mlir::failed(sourceRepr))
      return mlir::failure();
    if (*sourceRepr) {
      valueBundles[op.getResult(0)] = std::move(rendered);
      erase.push_back(op);
      return mlir::success();
    }
    if (RuntimeBundleLowerer::needsDefaultObjectRepr(*receiver)) {
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, *receiver, result)))
        return mlir::failure();
      valueBundles[op.getResult(0)] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{receiver};
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(RuntimeBundleLowerer::emitManifestMethodCall(
          op, *receiver, symbol.builtinMethod, sources,
          /*allowUnusedSources=*/false, emitted)))
    return mlir::failure();

  std::string resultContract = runtimeContractName(op.getResult(0).getType());
  if (resultContract.empty() || resultContract == "builtins.object")
    resultContract = symbol.resultContract;
  if (resultContract.empty())
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' needs a concrete result contract";

  RuntimeBundle result;
  if (mlir::failed(bundleRuntimeResults(
          op, builtinResultTypeFor(op.getResult(0), resultContract),
          emitted->call, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerDirectBuiltinCall(py::CallOp op,
                                             const RuntimeSymbol &symbol) {
  if (op.getNumResults() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' direct lowering must produce one result";

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  // ⭐ KEYWORD ARGUMENTS TO A MANIFEST FREE FUNCTION. Every one of them used to
  // stop here -- "kw names lowering is not keyword-aware yet" -- so a parameter
  // CPython makes keyword-only was unreachable by construction:
  // `math.isclose(a, b, rel_tol=1e-6)` had no spelling that compiled.
  //
  // The call carries what the mapping needs: the contract's arg_names followed
  // by its kw_names are the parameter ORDER, which is the order the manifest
  // function declares its inputs in. A keyword resolves to a position there,
  // and the positions nobody supplied stay null and take the function's own
  // ly.runtime.default_* value.
  const RuntimeBundle *kwNames = RuntimeBundleLowerer::bundleFor(op.getKwnames());
  const RuntimeBundle *kwValues =
      RuntimeBundleLowerer::bundleFor(op.getKwvalues());
  if (!kwNames || kwNames->kind != RuntimeBundle::Kind::Aggregate ||
      !kwValues || kwValues->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' keyword packs must be lowered aggregates";
  if (!kwNames->aggregateOperands.empty()) {
    if (kwNames->aggregateOperands.size() != kwValues->aggregateOperands.size())
      return op.emitError() << "builtin callable '" << symbol.builtinName
                            << "' keyword name/value count mismatch";
    auto contract =
        mlir::dyn_cast_if_present<py::CallableType>(op.getCallContract());
    if (!contract)
      return op.emitError() << "builtin callable '" << symbol.builtinName
                            << "' keyword call has no Callable contract";
    llvm::SmallVector<llvm::StringRef, 8> order;
    for (mlir::StringAttr parameter : contract.getPositionalNames())
      order.push_back(parameter.getValue());
    for (mlir::StringAttr parameter : contract.getKwOnlyNames())
      order.push_back(parameter.getValue());
    if (order.size() < sources.size())
      return op.emitError() << "builtin callable '" << symbol.builtinName
                            << "' contract names fewer parameters than the "
                               "call supplies";
    llvm::SmallVector<const RuntimeBundle *, 8> byPosition(order.size(),
                                                           nullptr);
    for (auto [index, source] : llvm::enumerate(sources))
      byPosition[index] = source;
    for (auto [index, nameValue] : llvm::enumerate(kwNames->aggregateOperands)) {
      std::optional<std::string> keyword =
          RuntimeBundleLowerer::keywordNameFromValue(nameValue);
      if (!keyword)
        return op.emitError() << "builtin callable '" << symbol.builtinName
                              << "' keyword name must be statically known";
      auto slot = llvm::find(order, llvm::StringRef(*keyword));
      if (slot == order.end())
        return op.emitError() << "builtin callable '" << symbol.builtinName
                              << "' has no parameter named '" << *keyword
                              << "'";
      unsigned position = static_cast<unsigned>(slot - order.begin());
      if (byPosition[position])
        return op.emitError() << "builtin callable '" << symbol.builtinName
                              << "' got two values for '" << *keyword << "'";
      mlir::Value keywordValue = kwValues->aggregateOperands[index];
      if (mlir::failed(
              RuntimeBundleLowerer::ensureValueBundle(op, keywordValue)))
        return mlir::failure();
      const RuntimeBundle *bundle =
          RuntimeBundleLowerer::bundleFor(keywordValue);
      if (!bundle)
        return op.emitError() << "builtin callable '" << symbol.builtinName
                              << "' keyword value has no lowered runtime "
                                 "bundle";
      byPosition[position] = bundle;
    }
    // Trailing parameters nobody supplied are not gaps, they are absent: the
    // operand walk fills them from the same defaults a short positional call
    // gets, and leaving them in the vector would ask it to do that twice.
    while (!byPosition.empty() && !byPosition.back())
      byPosition.pop_back();
    sources.assign(byPosition.begin(), byPosition.end());
  }

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 4> operands;
  if (mlir::failed(buildRuntimeCallOperands(op, symbol, sources, operands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), symbol, operands);

  std::string resultContract = runtimeContractName(op.getResult(0).getType());
  if (resultContract.empty() || resultContract == "builtins.object")
    resultContract = symbol.resultContract;
  if (resultContract.empty())
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' needs a concrete result contract";

  RuntimeBundle result;
  if (mlir::failed(bundleRuntimeResults(
          op, builtinResultTypeFor(op.getResult(0), resultContract), call,
          result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBuiltinMethodSinkCall(py::CallOp op,
                                                 const RuntimeSymbol &symbol) {
  const RuntimeBundle *argument = nullptr;
  if (mlir::failed(collectSingleBuiltinArgument(op, symbol, argument)))
    return mlir::failure();
  const RuntimeBundle *sinkArgument =
      RuntimeBundleLowerer::concreteObjectForOwnership(*argument);
  if (!sinkArgument)
    sinkArgument = argument;

  RuntimeBundle printable = *sinkArgument;
  // User exception classes have no manifest methods of their own but share
  // the taxonomy exception shape, and the manifest __str__/__repr__ resolve
  // the display name by DYNAMIC class id (which the instance header holds),
  // so rendering through the ancestor keeps the user class's name.
  if (printable.kind == RuntimeBundle::Kind::Object &&
      printable.physicalValues().size() == 3 &&
      !manifest.method(printable.contractName(), symbol.builtinMethod)) {
    if (std::optional<std::string> ancestor =
            RuntimeBundleLowerer::exceptionAncestorContractFor(
                printable.contract)) {
      mlir::Type ancestorType = runtimeContractType(context, *ancestor);
      printable.contract = ancestorType;
      printable.objectValue.contract = ancestorType;
    }
  }
  auto assignSinkResults = [&]() -> mlir::LogicalResult {
    std::string resultContract = symbol.resultContract.empty()
                                     ? "types.NoneType"
                                     : symbol.resultContract;
    for (mlir::Value result : op.getResults()) {
      if (mlir::failed(assignObjectBundle(
              op, result, runtimeContractType(context, resultContract), {})))
        return mlir::failure();
    }
    erase.push_back(op);
    return mlir::success();
  };

  if (symbol.builtinMethod == "__repr__" &&
      symbol.builtinSinkContract == "builtins.str" &&
      printable.contractName() == "builtins.object") {
    // print converts with str(), not repr() (an erased str prints without
    // quotes); the manifest object __str__ dispatches the payload class and
    // falls back to the repr form exactly where CPython's str(x) does.
    if (manifest.method("builtins.object", "__str__")) {
      llvm::SmallVector<const RuntimeBundle *, 1> strSources{&printable};
      std::optional<EmittedRuntimeCall> emitted;
      if (mlir::failed(emitManifestMethodCall(op, printable, "__str__",
                                              strSources,
                                              /*allowUnusedSources=*/false,
                                              emitted)))
        return mlir::failure();
      RuntimeBundle rendered;
      if (mlir::failed(bundleRuntimeResults(
              op, runtimeContractType(context, "builtins.str"), emitted->call,
              rendered)))
        return mlir::failure();
      printable = std::move(rendered);
    } else {
      RuntimeBundle rendered;
      if (mlir::failed(RuntimeBundleLowerer::emitBoxedReprHookCall(
              op, printable, rendered)))
        return mlir::failure();
      printable = std::move(rendered);
    }
  }

  // ⭐ None renders statically: it has no lanes, so there is no receiver to
  // hand a `__repr__`, and there is nothing to ask -- every None is "None".
  //
  //     print(None)   # types.NoneType runtime object has no physical header
  //                   # value
  //
  // The diagnostic named the ABI rather than the answer, and `str(None)`
  // already folds to the same four bytes, so this is that fold reached from
  // print's conversion instead of from `str`.
  if (printable.contractName() == "types.NoneType" &&
      symbol.builtinSinkContract == "builtins.str") {
    builder.setInsertionPoint(op);
    RuntimeBundle rendered;
    if (mlir::failed(
            RuntimeBundleLowerer::materializeStringObject(op, "None", rendered)))
      return mlir::failure();
    printable = std::move(rendered);
  }

  if (printable.contractName() != symbol.builtinSinkContract) {
    RuntimeBundle rendered;
    mlir::FailureOr<bool> sourceRepr =
        symbol.builtinMethod == "__repr__"
            ? RuntimeBundleLowerer::emitSourceClassReprCall(op, printable,
                                                            rendered)
            : mlir::FailureOr<bool>(false);
    if (mlir::failed(sourceRepr))
      return mlir::failure();
    if (*sourceRepr) {
      printable = std::move(rendered);
    } else if (symbol.builtinMethod == "__repr__" &&
               RuntimeBundleLowerer::needsDefaultObjectRepr(printable)) {
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, printable, rendered)))
        return mlir::failure();
      printable = std::move(rendered);
    } else {
      // CPython print() renders through str(); __repr__ is only the
      // fallback for contracts without a __str__ (containers). The two were
      // indistinguishable until exception __repr__ gained its
      // ClassName(...) form.
      llvm::StringRef sinkMethod = symbol.builtinMethod;
      if (sinkMethod == "__repr__" &&
          manifest.method(printable.contractName(), "__str__"))
        sinkMethod = "__str__";
      llvm::SmallVector<const RuntimeBundle *, 1> sources{&printable};
      std::optional<EmittedRuntimeCall> emitted;
      if (mlir::failed(emitManifestMethodCall(
              op, printable, sinkMethod, sources,
              /*allowUnusedSources=*/false, emitted)))
        return mlir::failure();
      if (mlir::failed(bundleRuntimeResults(
              op, runtimeContractType(context, symbol.builtinSinkContract),
              emitted->call, printable)))
        return mlir::failure();
    }
  }
  if (printable.contractName() != symbol.builtinSinkContract)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' requires a " << symbol.builtinSinkContract
                          << "-compatible argument";

  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), symbol,
                                          printable.physicalValues());
  return assignSinkResults();
}

} // namespace py::lowering
