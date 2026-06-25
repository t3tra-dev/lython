#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

RuntimeBundleLowerer::RuntimeBundleLowerer(mlir::ModuleOp module)
    : module(module), context(module.getContext()), builder(context),
      manifest(module) {}

mlir::LogicalResult RuntimeBundleLowerer::lowerModule() {
  if (mlir::failed(manifest.verify()))
    return mlir::failure();
  if (mlir::failed(buildReturnedCallableSummaries()))
    return mlir::failure();
  if (mlir::failed(prepareCallableFunctionABIs()))
    return mlir::failure();

  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::Operation *op) {
    if (!op->getDialect() || op->getDialect()->getNamespace() != "py")
      return mlir::WalkResult::advance();
    if (mlir::failed(lowerPyOp(op))) {
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (mlir::failed(result))
    return mlir::failure();
  if (mlir::failed(lowerFunctionReturns()))
    return mlir::failure();
  if (mlir::failed(eraseLoweredPyOps()))
    return mlir::failure();
  return RuntimeBundleLowerer::eraseCallableLogicalEntryArgs();
}

const RuntimeValueShape *
RuntimeBundleLowerer::runtimeValueShapeFor(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef purpose) const {
  std::string contract = runtimeContractName(type);
  if (contract.empty()) {
    op->emitError() << purpose << " has no concrete runtime contract: " << type;
    return nullptr;
  }
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  if (!shape)
    op->emitError() << "runtime manifest has no ABI shape for " << contract
                    << " " << purpose;
  return shape;
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeValueTypes(
    mlir::Operation *op, mlir::Type type,
    llvm::SmallVectorImpl<mlir::Type> &types) const {
  const RuntimeValueShape *shape =
      RuntimeBundleLowerer::runtimeValueShapeFor(op, type, "callable ABI type");
  if (!shape)
    return mlir::failure();
  types.append(shape->valueTypes.begin(), shape->valueTypes.end());
  return mlir::success();
}

llvm::SmallVector<mlir::Type, 4>
RuntimeBundleLowerer::callableClosureTypes(mlir::func::FuncOp function) const {
  llvm::SmallVector<mlir::Type, 4> types;
  auto attr = function->getAttrOfType<mlir::ArrayAttr>("closure_types");
  if (!attr)
    return types;
  for (mlir::Attribute entry : attr) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(entry);
    if (!typeAttr)
      return {};
    types.push_back(typeAttr.getValue());
  }
  return types;
}

static mlir::Value stripReturnedCallableView(mlir::Value value) {
  while (value) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def || def->getNumOperands() != 1 || def->getNumResults() != 1)
      return value;
    llvm::StringRef name = def->getName().getStringRef();
    if (name != "py.class.upcast" && name != "py.class.refine" &&
        name != "py.protocol.view")
      return value;
    value = def->getOperand(0);
  }
  return value;
}

mlir::LogicalResult RuntimeBundleLowerer::buildReturnedCallableSummaries() {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("callable_type") || function.isDeclaration())
      return mlir::WalkResult::advance();

    std::optional<ReturnedCallableSummary> summary;
    bool sawReturnedCallable = false;
    function.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != function)
        return;
      if (ret.getNumOperands() != 1)
        return;
      mlir::Value returned = stripReturnedCallableView(ret.getOperand(0));
      auto binding = returned.getDefiningOp<py::BindingRefOp>();
      if (!binding)
        return;
      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
      if (!target || !target->hasAttr("callable_type"))
        return;

      ReturnedCallableSummary candidate;
      candidate.target = binding.getBinding().str();
      mlir::Block &entry = function.getBody().front();
      for (mlir::Value capture : binding.getCaptures()) {
        auto arg = mlir::dyn_cast<mlir::BlockArgument>(capture);
        if (!arg || arg.getOwner() != &entry) {
          summary.reset();
          sawReturnedCallable = false;
          return;
        }
        candidate.captureArgumentIndices.push_back(arg.getArgNumber());
      }

      if (!summary) {
        summary = std::move(candidate);
      } else if (summary->target != candidate.target ||
                 summary->captureArgumentIndices !=
                     candidate.captureArgumentIndices) {
        summary.reset();
        sawReturnedCallable = false;
        return;
      }
      sawReturnedCallable = true;
    });

    if (summary && sawReturnedCallable)
      returnedCallableSummaries[function.getSymName()] = std::move(*summary);
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::LogicalResult RuntimeBundleLowerer::prepareCallableFunctionABIs() {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp function) {
    auto callableType =
        function->getAttrOfType<mlir::TypeAttr>("callable_type");
    if (!callableType)
      return mlir::WalkResult::advance();
    auto callable = mlir::dyn_cast<py::CallableType>(callableType.getValue());
    if (!callable) {
      function.emitError() << "callable_type must be Callable";
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    if (callable.hasVararg() || callable.hasKwarg()) {
      if (function.isDeclaration())
        return mlir::WalkResult::advance();
      function.emitError()
          << "callable function ABI lowering currently does not support "
             "variadic callables";
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }

    llvm::SmallVector<mlir::Type, 8> logicalInputTypes(
        callable.getPositionalTypes().begin(),
        callable.getPositionalTypes().end());
    logicalInputTypes.append(callable.getKwOnlyTypes().begin(),
                             callable.getKwOnlyTypes().end());
    llvm::SmallVector<mlir::Type, 4> closureTypes =
        callableClosureTypes(function);
    logicalInputTypes.append(closureTypes.begin(), closureTypes.end());

    llvm::SmallVector<mlir::Type, 8> inputTypes;
    for (mlir::Type inputType : logicalInputTypes) {
      if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
              function, inputType, inputTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
    }

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    for (mlir::Type resultType : callable.getResultTypes()) {
      if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
              function, resultType, resultTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
    }
    if (!function.isDeclaration()) {
      if (mlir::failed(
              seedCallableEntryArgumentBundles(function, logicalInputTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
    }
    function.setFunctionType(
        mlir::FunctionType::get(context, inputTypes, resultTypes));
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::LogicalResult RuntimeBundleLowerer::seedCallableEntryArgumentBundles(
    mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes) {
  if (function.isDeclaration())
    return mlir::success();
  mlir::Block &entry = function.getBody().front();
  if (entry.getNumArguments() != logicalTypes.size())
    return function.emitError()
           << "callable function entry argument count does not match "
              "callable_type";

  unsigned logicalArgCount = entry.getNumArguments();
  for (auto [index, logicalType] : llvm::enumerate(logicalTypes)) {
    mlir::BlockArgument logicalArg = entry.getArgument(index);
    const RuntimeValueShape *shape = RuntimeBundleLowerer::runtimeValueShapeFor(
        function, logicalType, "callable parameter ABI");
    if (!shape)
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> physicalArgs;
    for (mlir::Type physicalType : shape->valueTypes)
      physicalArgs.push_back(
          entry.addArgument(physicalType, logicalArg.getLoc()));

    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            function, logicalType, physicalArgs, bundle)))
      return mlir::failure();
    valueBundles[logicalArg] = std::move(bundle);
  }
  callableLogicalEntryArgCounts.push_back(
      CallableLogicalEntryArgs{function, logicalArgCount});
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::validateObjectShape(mlir::Operation *op,
                                          llvm::StringRef contract,
                                          mlir::ValueRange values) const {
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  if (!shape)
    return op->emitError() << "runtime manifest has no ABI shape for "
                           << contract;
  if (shape->valueTypes.size() != values.size())
    return op->emitError() << "runtime bundle for " << contract << " has "
                           << values.size()
                           << " values, but manifest shape from "
                           << shape->source << " expects "
                           << shape->valueTypes.size();
  for (auto [index, value] : llvm::enumerate(values)) {
    mlir::Type expected = shape->valueTypes[index];
    if (value.getType() != expected)
      return op->emitError()
             << "runtime bundle value " << index << " for " << contract
             << " has type " << value.getType() << ", but manifest shape from "
             << shape->source << " expects " << expected;
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::makeObjectBundle(mlir::Operation *op, mlir::Type contract,
                                       mlir::ValueRange values,
                                       RuntimeBundle &bundle) const {
  std::string contractName = runtimeContractName(contract);
  if (contractName.empty())
    return op->emitError() << "runtime bundle has no concrete contract";
  if (mlir::failed(validateObjectShape(op, contractName, values)))
    return mlir::failure();
  bundle = RuntimeBundle::object(contract, values);
  return mlir::success();
}

bool RuntimeBundleLowerer::objectShapeMatches(llvm::StringRef contract,
                                              mlir::ValueRange values) const {
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  if (!shape)
    return false;
  if (shape->valueTypes.size() != values.size())
    return false;
  for (auto [index, value] : llvm::enumerate(values))
    if (value.getType() != shape->valueTypes[index])
      return false;
  return true;
}

bool RuntimeBundleLowerer::rawValuesMatchRuntimeInputs(
    const RuntimeSymbol &symbol, mlir::ValueRange values) const {
  mlir::func::FuncOp function = symbol.function;
  mlir::FunctionType functionType = function.getFunctionType();
  if (functionType.getNumInputs() != values.size())
    return false;
  for (auto [index, value] : llvm::enumerate(values))
    if (value.getType() != functionType.getInput(index))
      return false;
  return true;
}

mlir::LogicalResult RuntimeBundleLowerer::initializeObjectFromRawValues(
    mlir::Operation *op, mlir::Type contract, mlir::ValueRange values,
    RuntimeBundle &bundle, bool emitErrors) {
  std::string contractName = runtimeContractName(contract);
  if (contractName.empty()) {
    if (emitErrors)
      return op->emitError()
             << "runtime initializer target has no concrete contract";
    return mlir::failure();
  }

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer(contractName, "__new__");
  if (!initializer) {
    if (emitErrors)
      return op->emitError()
             << "runtime manifest has no " << contractName << ".__new__";
    return mlir::failure();
  }
  if (!rawValuesMatchRuntimeInputs(*initializer, values)) {
    if (emitErrors)
      return op->emitError() << "runtime initializer " << contractName
                             << ".__new__ cannot accept raw input values "
                             << describeValueTypes(values);
    return mlir::failure();
  }

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *initializer, values);
  return RuntimeBundleLowerer::makeObjectBundle(op, contract, call.getResults(),
                                                bundle);
}

mlir::LogicalResult RuntimeBundleLowerer::bundleRawObjectValues(
    mlir::Operation *op, mlir::Type contract, mlir::ValueRange values,
    RuntimeBundle &bundle) {
  std::string contractName = runtimeContractName(contract);
  if (contractName.empty())
    return op->emitError()
           << "default argument has no concrete runtime contract";

  mlir::Type concreteContract = runtimeContractType(context, contractName);
  if (objectShapeMatches(contractName, values))
    return RuntimeBundleLowerer::makeObjectBundle(op, concreteContract, values,
                                                  bundle);
  if (mlir::succeeded(initializeObjectFromRawValues(
          op, concreteContract, values, bundle, /*emitErrors=*/false)))
    return mlir::success();
  return RuntimeBundleLowerer::makeObjectBundle(op, concreteContract, values,
                                                bundle);
}

mlir::LogicalResult RuntimeBundleLowerer::materializeDefaultValue(
    mlir::Operation *op, mlir::Type parameterType, mlir::Attribute attr,
    RuntimeBundle &bundle) {
  auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
  if (!dict)
    return op->emitError() << "callable default value metadata is malformed";
  auto kind = dict.getAs<mlir::StringAttr>("kind");
  if (!kind)
    return op->emitError() << "callable default value has no kind";

  mlir::Location loc = op->getLoc();
  llvm::StringRef spelling = kind.getValue();
  if (spelling == "none")
    return RuntimeBundleLowerer::bundleRawObjectValues(
        op, parameterType, mlir::ValueRange{}, bundle);
  if (spelling == "bool") {
    auto value = dict.getAs<mlir::BoolAttr>("value");
    if (!value)
      return op->emitError() << "bool default value has no payload";
    mlir::Value bit =
        builder.create<mlir::arith::ConstantIntOp>(loc, value.getValue(), 1)
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(op, parameterType, bit,
                                                       bundle);
  }
  if (spelling == "int") {
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (!value)
      return op->emitError() << "int default value has no payload";
    std::int64_t parsed = 0;
    if (value.getValue().getAsInteger(10, parsed))
      return op->emitError()
             << "integer default value is outside the lowered i64 path";
    mlir::Value integer =
        builder.create<mlir::arith::ConstantIntOp>(loc, parsed, 64).getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(op, parameterType,
                                                       integer, bundle);
  }
  if (spelling == "float") {
    auto value = dict.getAs<mlir::FloatAttr>("value");
    if (!value)
      return op->emitError() << "float default value has no payload";
    mlir::Value number = builder
                             .create<mlir::arith::ConstantFloatOp>(
                                 loc, value.getValue(), builder.getF64Type())
                             .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(op, parameterType,
                                                       number, bundle);
  }
  if (spelling == "str") {
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (!value)
      return op->emitError() << "str default value has no payload";
    mlir::Value bytes =
        RuntimeBundleLowerer::materializeByteBuffer(loc, value.getValue());
    mlir::Value start =
        builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
    mlir::Value length =
        builder
            .create<mlir::arith::ConstantIntOp>(
                loc, static_cast<std::int64_t>(value.getValue().size()), 64)
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(
        op, parameterType, mlir::ValueRange{bytes, start, length}, bundle);
  }
  if (spelling == "unsupported") {
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (value)
      return op->emitError()
             << "unsupported callable default expression " << value;
    return op->emitError() << "unsupported callable default expression";
  }
  return op->emitError() << "unknown callable default value kind '" << spelling
                         << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::assignObjectBundle(mlir::Operation *op, mlir::Value value,
                                         mlir::Type contract,
                                         mlir::ValueRange values) {
  RuntimeBundle bundle;
  if (mlir::failed(makeObjectBundle(op, contract, values, bundle)))
    return mlir::failure();
  valueBundles[value] = std::move(bundle);
  return mlir::success();
}

mlir::FailureOr<llvm::StringRef>
RuntimeBundleLowerer::requireMethodTarget(mlir::Operation *op,
                                          mlir::FlatSymbolRefAttr target,
                                          llvm::StringRef expectedName) const {
  if (target)
    return target.getValue();
  return op->emitError() << "resolved special-method op for " << expectedName
                         << " has no manifest method target";
}

} // namespace py::runtime_lowering
