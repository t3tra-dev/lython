#include "RuntimeLowering/RuntimeLowering.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace py::runtime_lowering {

namespace {

enum class PrimitiveI64ArithmeticKind { Add, Sub, Mul };

bool compatibleRankOneMemRefStorage(mlir::Type source, mlir::Type target,
                                    bool targetMustBeDynamic) {
  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(source);
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(target);
  if (!sourceMemRef || !targetMemRef || sourceMemRef.getRank() != 1 ||
      targetMemRef.getRank() != 1 ||
      sourceMemRef.getElementType() != targetMemRef.getElementType() ||
      sourceMemRef.getMemorySpace() != targetMemRef.getMemorySpace())
    return false;
  if (targetMustBeDynamic)
    return targetMemRef.getDimSize(0) == mlir::ShapedType::kDynamic;
  return sourceMemRef.getShape() == targetMemRef.getShape();
}

std::optional<PrimitiveI64ArithmeticKind>
primitiveI64ArithmeticKind(llvm::StringRef methodName) {
  return llvm::StringSwitch<std::optional<PrimitiveI64ArithmeticKind>>(
             methodName)
      .Case("__add__", PrimitiveI64ArithmeticKind::Add)
      .Case("__sub__", PrimitiveI64ArithmeticKind::Sub)
      .Case("__mul__", PrimitiveI64ArithmeticKind::Mul)
      .Default(std::nullopt);
}

std::optional<mlir::arith::CmpIPredicate>
primitiveI64ComparePredicate(llvm::StringRef methodName) {
  return llvm::StringSwitch<std::optional<mlir::arith::CmpIPredicate>>(
             methodName)
      .Case("__eq__", mlir::arith::CmpIPredicate::eq)
      .Case("__ne__", mlir::arith::CmpIPredicate::ne)
      .Case("__lt__", mlir::arith::CmpIPredicate::slt)
      .Case("__le__", mlir::arith::CmpIPredicate::sle)
      .Case("__gt__", mlir::arith::CmpIPredicate::sgt)
      .Case("__ge__", mlir::arith::CmpIPredicate::sge)
      .Default(std::nullopt);
}

bool hasPrimitiveI64Evidence(const RuntimeBundle *bundle) {
  return bundle && bundle->kind == RuntimeBundle::Kind::Object &&
         bundle->contractName() == "builtins.int" && bundle->primitiveI64 &&
         bundle->primitiveI64->value &&
         bundle->primitiveI64->value.getType().isInteger(64) &&
         bundle->primitiveI64->valid &&
         bundle->primitiveI64->valid.getType().isInteger(1);
}

mlir::Value boolConstant(mlir::OpBuilder &builder, mlir::Location loc,
                         bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1)
      .getResult();
}

mlir::Value i64Constant(mlir::OpBuilder &builder, mlir::Location loc,
                        std::int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64)
      .getResult();
}

mlir::Value logicalAnd(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value lhs, mlir::Value rhs) {
  return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs).getResult();
}

mlir::Value logicalNot(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value value) {
  return builder.create<mlir::arith::XOrIOp>(
                    loc, value, boolConstant(builder, loc, true))
      .getResult();
}

mlir::Value signedAddOverflow(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs,
                              mlir::Value result) {
  mlir::Value zero = i64Constant(builder, loc, 0);
  mlir::Value lhsNegative = builder.create<mlir::arith::CmpIOp>(
                                     loc, mlir::arith::CmpIPredicate::slt, lhs,
                                     zero)
                                 .getResult();
  mlir::Value rhsNegative = builder.create<mlir::arith::CmpIOp>(
                                     loc, mlir::arith::CmpIPredicate::slt, rhs,
                                     zero)
                                 .getResult();
  mlir::Value resultNegative =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                       result, zero)
          .getResult();
  mlir::Value sameSign =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                       lhsNegative, rhsNegative)
          .getResult();
  mlir::Value signChanged =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                       resultNegative, lhsNegative)
          .getResult();
  return logicalAnd(builder, loc, sameSign, signChanged);
}

mlir::Value signedSubOverflow(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs,
                              mlir::Value result) {
  mlir::Value zero = i64Constant(builder, loc, 0);
  mlir::Value lhsNegative = builder.create<mlir::arith::CmpIOp>(
                                     loc, mlir::arith::CmpIPredicate::slt, lhs,
                                     zero)
                                 .getResult();
  mlir::Value rhsNegative = builder.create<mlir::arith::CmpIOp>(
                                     loc, mlir::arith::CmpIPredicate::slt, rhs,
                                     zero)
                                 .getResult();
  mlir::Value resultNegative =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                       result, zero)
          .getResult();
  mlir::Value differentSign =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                       lhsNegative, rhsNegative)
          .getResult();
  mlir::Value signChanged =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                       resultNegative, lhsNegative)
          .getResult();
  return logicalAnd(builder, loc, differentSign, signChanged);
}

std::pair<mlir::Value, mlir::Value>
buildPrimitiveI64Arithmetic(mlir::OpBuilder &builder, mlir::Location loc,
                            PrimitiveI64ArithmeticKind kind, mlir::Value lhs,
                            mlir::Value rhs) {
  switch (kind) {
  case PrimitiveI64ArithmeticKind::Add: {
    mlir::Value result =
        builder.create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
    return {result, signedAddOverflow(builder, loc, lhs, rhs, result)};
  }
  case PrimitiveI64ArithmeticKind::Sub: {
    mlir::Value result =
        builder.create<mlir::arith::SubIOp>(loc, lhs, rhs).getResult();
    return {result, signedSubOverflow(builder, loc, lhs, rhs, result)};
  }
  case PrimitiveI64ArithmeticKind::Mul: {
    auto extended =
        builder.create<mlir::arith::MulSIExtendedOp>(loc, lhs, rhs);
    mlir::Value shift = i64Constant(builder, loc, 63);
    mlir::Value expectedHigh =
        builder.create<mlir::arith::ShRSIOp>(loc, extended.getLow(), shift)
            .getResult();
    mlir::Value overflow =
        builder
            .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                         extended.getHigh(), expectedHigh)
            .getResult();
    return {extended.getLow(), overflow};
  }
  }
  llvm_unreachable("unknown primitive i64 arithmetic kind");
}

} // namespace

bool RuntimeBundleLowerer::canAppendExactValueSequence(
    mlir::FunctionType functionType, unsigned inputIndex,
    const RuntimeBundle &source) const {
  llvm::ArrayRef<mlir::Value> sourceValues = source.physicalValues();
  if (sourceValues.empty() ||
      inputIndex + sourceValues.size() > functionType.getNumInputs())
    return false;
  for (auto [offset, value] : llvm::enumerate(sourceValues)) {
    if (value.getType() != functionType.getInput(inputIndex + offset))
      return false;
  }
  return true;
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeSource(
    mlir::Operation *op, const RuntimeSymbol &symbol,
    mlir::FunctionType functionType, unsigned &inputIndex,
    const RuntimeBundle &source, llvm::SmallVectorImpl<mlir::Value> &operands) {
  llvm::ArrayRef<mlir::Value> sourceValues = source.physicalValues();
  if (canAppendExactValueSequence(functionType, inputIndex, source)) {
    operands.append(sourceValues.begin(), sourceValues.end());
    inputIndex += static_cast<unsigned>(sourceValues.size());
    return mlir::success();
  }

  mlir::Type expected = functionType.getInput(inputIndex);
  if (source.kind == RuntimeBundle::Kind::Object &&
      isErasedObjectStorageType(expected)) {
    mlir::FailureOr<mlir::Value> storage =
        RuntimeBundleLowerer::erasedObjectStorageView(op, source.objectValue,
                                                      expected);
    if (mlir::failed(storage))
      return mlir::failure();
    operands.push_back(*storage);
    ++inputIndex;
    return mlir::success();
  }

  if (source.kind == RuntimeBundle::Kind::Object &&
      isBuiltinsObjectHeaderType(expected)) {
    mlir::FailureOr<mlir::Value> header =
        RuntimeBundleLowerer::objectHeaderView(op, source.objectValue);
    if (mlir::failed(header))
      return mlir::failure();
    operands.push_back(*header);
    ++inputIndex;
    return mlir::success();
  }

  if (source.kind == RuntimeBundle::Kind::TypeObject &&
      expected.isInteger(64)) {
    std::optional<std::int64_t> id =
        manifest.classId(source.instanceContractName());
    if (!id)
      return op->emitError() << "type object has no runtime class id for "
                             << source.instanceContractName();
    mlir::Value value =
        builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), *id, 64)
            .getResult();
    operands.push_back(value);
    ++inputIndex;
    return mlir::success();
  }

  if (expected.isInteger(64)) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.i64");
    if (unbox) {
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), *unbox, sourceValues);
      if (call.getNumResults() == 1 &&
          call.getResult(0).getType() == expected) {
        operands.push_back(call.getResult(0));
        ++inputIndex;
        return mlir::success();
      }
    }
  }

  if (expected == builder.getF64Type()) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.f64");
    if (unbox) {
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), *unbox, sourceValues);
      if (call.getNumResults() == 1 &&
          call.getResult(0).getType() == expected) {
        operands.push_back(call.getResult(0));
        ++inputIndex;
        return mlir::success();
      }
    }
  }

  return op->emitError() << "cannot adapt " << source.contractName()
                         << " to runtime input " << inputIndex << " of "
                         << symbol.contract << "." << symbol.name;
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeSourceAs(
    mlir::Operation *op, const RuntimeSymbol &symbol,
    mlir::FunctionType functionType, unsigned &inputIndex,
    const RuntimeBundle &source, mlir::Type expected,
    llvm::SmallVectorImpl<mlir::Value> &operands) {
  if (auto expectedUnion = mlir::dyn_cast_if_present<py::UnionType>(expected)) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(
            op, expectedUnion, "callable union argument ABI");
    if (mlir::failed(expectedTypes))
      return mlir::failure();
    if (inputIndex + expectedTypes->size() > functionType.getNumInputs())
      return op->emitError() << "union argument ABI for " << symbol.contract
                             << "." << symbol.name << " exceeds runtime inputs";
    for (auto [offset, expectedType] : llvm::enumerate(*expectedTypes)) {
      mlir::Type physicalType = functionType.getInput(inputIndex + offset);
      if (physicalType != expectedType)
        return op->emitError()
               << "union argument ABI for " << symbol.contract << "."
               << symbol.name << " expects runtime input "
               << inputIndex + offset << " to be " << expectedType
               << ", but function ABI has " << physicalType;
    }

    llvm::SmallVector<mlir::Value, 8> unionValues;
    if (mlir::failed(RuntimeBundleLowerer::appendUnionRuntimeValues(
            op, expectedUnion, source, source.contract, unionValues)))
      return mlir::failure();
    if (unionValues.size() != expectedTypes->size())
      return op->emitError()
             << "union argument produced " << unionValues.size()
             << " physical values, but ABI expects " << expectedTypes->size();
    operands.append(unionValues.begin(), unionValues.end());
    inputIndex += static_cast<unsigned>(unionValues.size());
    return mlir::success();
  }

  return RuntimeBundleLowerer::appendRuntimeSource(
      op, symbol, functionType, inputIndex, source, operands);
}

mlir::LogicalResult RuntimeBundleLowerer::appendPrimitiveI64EvidenceOperand(
    mlir::Operation *op, mlir::FunctionType functionType, unsigned &inputIndex,
    const RuntimeBundle &source, llvm::SmallVectorImpl<mlir::Value> &operands) {
  if (source.contractName() != "builtins.int" ||
      inputIndex + 2 > functionType.getNumInputs() ||
      !functionType.getInput(inputIndex).isInteger(64) ||
      !functionType.getInput(inputIndex + 1).isInteger(1))
    return mlir::success();

  if (source.primitiveI64) {
    operands.push_back(source.primitiveI64->value);
    operands.push_back(source.primitiveI64->valid);
  } else {
    operands.push_back(
        builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), 0, 64)
            .getResult());
    operands.push_back(boolConstant(builder, op->getLoc(), false));
  }
  inputIndex += 2;
  return mlir::success();
}

bool RuntimeBundleLowerer::canAppendRuntimeSource(
    const RuntimeSymbol &symbol, mlir::FunctionType functionType,
    unsigned &inputIndex, const RuntimeBundle &source) const {
  llvm::ArrayRef<mlir::Value> sourceValues = source.physicalValues();
  if (canAppendExactValueSequence(functionType, inputIndex, source)) {
    inputIndex += static_cast<unsigned>(sourceValues.size());
    return true;
  }
  if (inputIndex >= functionType.getNumInputs())
    return false;

  mlir::Type expected = functionType.getInput(inputIndex);
  if (source.kind == RuntimeBundle::Kind::Object &&
      isErasedObjectStorageType(expected) && !sourceValues.empty() &&
      compatibleRankOneMemRefStorage(sourceValues.front().getType(), expected,
                                     /*targetMustBeDynamic=*/true)) {
    ++inputIndex;
    return true;
  }

  if (source.kind == RuntimeBundle::Kind::Object &&
      isBuiltinsObjectHeaderType(expected) && !sourceValues.empty() &&
      (sourceValues.front().getType() == expected ||
       compatibleRankOneMemRefStorage(sourceValues.front().getType(), expected,
                                      /*targetMustBeDynamic=*/false))) {
    ++inputIndex;
    return true;
  }

  if (source.kind == RuntimeBundle::Kind::TypeObject &&
      expected.isInteger(64) &&
      manifest.classId(source.instanceContractName())) {
    ++inputIndex;
    return true;
  }

  if (expected.isInteger(64)) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.i64");
    if (unbox && unbox->function.getFunctionType().getNumResults() == 1 &&
        unbox->function.getFunctionType().getResult(0) == expected) {
      ++inputIndex;
      return true;
    }
  }

  if (expected.isF64()) {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source.contractName(), "unbox.f64");
    if (unbox && unbox->function.getFunctionType().getNumResults() == 1 &&
        unbox->function.getFunctionType().getResult(0) == expected) {
      ++inputIndex;
      return true;
    }
  }

  (void)symbol;
  return false;
}

mlir::LogicalResult RuntimeBundleLowerer::appendImplicitRuntimeArgument(
    mlir::Operation *op, const RuntimeSymbol &symbol, unsigned &inputIndex,
    llvm::SmallVectorImpl<mlir::Value> &operands) {
  if (const RuntimeDefaultArgument *defaultArgument =
          symbol.defaultArgument(inputIndex)) {
    if (defaultArgument->kind == RuntimeDefaultArgument::Kind::I64) {
      auto integerDefault =
          mlir::cast<mlir::IntegerAttr>(defaultArgument->value);
      operands.push_back(builder
                             .create<mlir::arith::ConstantIntOp>(
                                 op->getLoc(), integerDefault.getInt(), 64)
                             .getResult());
      ++inputIndex;
      return mlir::success();
    }
    auto floatDefault = mlir::cast<mlir::FloatAttr>(defaultArgument->value);
    operands.push_back(
        builder
            .create<mlir::arith::ConstantFloatOp>(
                op->getLoc(), floatDefault.getValue(), builder.getF64Type())
            .getResult());
    ++inputIndex;
    return mlir::success();
  }
  return op->emitError() << "runtime call is missing input " << inputIndex
                         << " for " << symbol.contract << "." << symbol.name;
}

bool RuntimeBundleLowerer::canAppendImplicitRuntimeArgument(
    const RuntimeSymbol &symbol, unsigned &inputIndex) const {
  if (!symbol.defaultArgument(inputIndex))
    return false;
  ++inputIndex;
  return true;
}

bool RuntimeBundleLowerer::canBuildRuntimeCallOperands(
    const RuntimeSymbol &symbol, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources, const RuntimeBundle *classObject) const {
  mlir::func::FuncOp function = symbol.function;
  mlir::FunctionType functionType = function.getFunctionType();
  unsigned inputIndex = 0;
  unsigned sourceIndex = 0;
  while (inputIndex < functionType.getNumInputs()) {
    if (symbol.hasClassIdArgument(inputIndex)) {
      if (!classObject)
        return false;
      if (!canAppendRuntimeSource(symbol, functionType, inputIndex,
                                  *classObject))
        return false;
      continue;
    }
    if (sourceIndex >= sources.size()) {
      if (!canAppendImplicitRuntimeArgument(symbol, inputIndex))
        return false;
      continue;
    }
    if (!sources[sourceIndex] ||
        !canAppendRuntimeSource(symbol, functionType, inputIndex,
                                *sources[sourceIndex]))
      return false;
    ++sourceIndex;
  }
  return allowUnusedSources || sourceIndex == sources.size();
}

mlir::LogicalResult RuntimeBundleLowerer::buildRuntimeCallOperands(
    mlir::Operation *op, const RuntimeSymbol &symbol,
    llvm::ArrayRef<const RuntimeBundle *> sources,
    llvm::SmallVectorImpl<mlir::Value> &operands, bool allowUnusedSources,
    const RuntimeBundle *classObject) {
  mlir::func::FuncOp function = symbol.function;
  mlir::FunctionType functionType = function.getFunctionType();
  unsigned inputIndex = 0;
  unsigned sourceIndex = 0;
  while (inputIndex < functionType.getNumInputs()) {
    if (symbol.hasClassIdArgument(inputIndex)) {
      if (!classObject)
        return op->emitError()
               << "runtime class id input " << inputIndex << " for "
               << symbol.contract << "." << symbol.name
               << " has no lowered class object source";
      if (mlir::failed(appendRuntimeSource(op, symbol, functionType, inputIndex,
                                           *classObject, operands)))
        return mlir::failure();
      continue;
    }
    if (sourceIndex >= sources.size()) {
      if (mlir::failed(
              appendImplicitRuntimeArgument(op, symbol, inputIndex, operands)))
        return mlir::failure();
      continue;
    }
    if (mlir::failed(appendRuntimeSource(op, symbol, functionType, inputIndex,
                                         *sources[sourceIndex], operands)))
      return mlir::failure();
    ++sourceIndex;
  }
  if (!allowUnusedSources && sourceIndex != sources.size())
    return op->emitError() << "too many runtime arguments for "
                           << symbol.contract << "." << symbol.name;
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerPrimitiveI64BinarySpecial(
    mlir::Operation *op, llvm::StringRef methodName,
    llvm::ArrayRef<const RuntimeBundle *> sources, mlir::Value resultValue) {
  if (sources.size() != 2 || !hasPrimitiveI64Evidence(sources[0]) ||
      !hasPrimitiveI64Evidence(sources[1]))
    return op->emitError()
           << "primitive i64 lowering requires two int operands with evidence";

  std::optional<PrimitiveI64ArithmeticKind> arithmetic =
      primitiveI64ArithmeticKind(methodName);
  std::optional<mlir::arith::CmpIPredicate> compare =
      primitiveI64ComparePredicate(methodName);
  if (!arithmetic && !compare)
    return op->emitError()
           << "unsupported primitive i64 special method " << methodName;

  mlir::FailureOr<RuntimeSymbol> selected =
      RuntimeBundleLowerer::selectManifestMethod(
          op, *sources.front(), methodName, sources,
          /*allowUnusedSources=*/false);
  if (mlir::failed(selected))
    return mlir::failure();

  std::string resultContract = RuntimeBundleLowerer::resultContractFor(
      resultValue, *selected, /*preferManifestObjectResult=*/true);
  if (resultContract.empty())
    return op->emitError() << "primitive i64 " << methodName
                           << " result needs a concrete runtime contract";
  if (arithmetic && resultContract != "builtins.int")
    return op->emitError() << "primitive i64 arithmetic " << methodName
                           << " must produce builtins.int, got "
                           << resultContract;
  if (compare && resultContract != "builtins.bool")
    return op->emitError() << "primitive i64 comparison " << methodName
                           << " must produce builtins.bool, got "
                           << resultContract;

  mlir::Type resultType = runtimeContractType(context, resultContract);
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(
          op, resultType, "primitive i64 guarded result ABI");
  if (mlir::failed(resultTypes))
    return mlir::failure();

  auto checkPhysicalTypes =
      [&](mlir::ValueRange values, llvm::StringRef label)
          -> mlir::LogicalResult {
    if (values.size() != resultTypes->size())
      return op->emitError()
             << label << " produced " << values.size()
             << " values, but primitive guarded result ABI expects "
             << resultTypes->size();
    for (auto [index, value] : llvm::enumerate(values)) {
      if (value.getType() != (*resultTypes)[index])
        return op->emitError()
               << label << " result " << index << " has type "
               << value.getType() << ", but primitive guarded result ABI "
               << "expects " << (*resultTypes)[index];
    }
    return mlir::success();
  };

  context->loadDialect<mlir::scf::SCFDialect>();
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  const RuntimePrimitiveI64Evidence &lhs = *sources[0]->primitiveI64;
  const RuntimePrimitiveI64Evidence &rhs = *sources[1]->primitiveI64;
  mlir::Value operandsValid = logicalAnd(builder, loc, lhs.valid, rhs.valid);

  auto emitFallbackYield = [&]() -> mlir::LogicalResult {
    llvm::SmallVector<mlir::Value, 8> operands;
    if (mlir::failed(RuntimeBundleLowerer::buildRuntimeCallOperands(
            op, *selected, sources, operands, /*allowUnusedSources=*/false)))
      return mlir::failure();
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *selected, operands);
    if (mlir::failed(checkPhysicalTypes(call.getResults(), "fallback call")))
      return mlir::failure();
    builder.create<mlir::scf::YieldOp>(loc, call.getResults());
    return mlir::success();
  };

  if (arithmetic) {
    auto [rawResult, overflow] = buildPrimitiveI64Arithmetic(
        builder, loc, *arithmetic, lhs.value, rhs.value);
    mlir::Value fastValid =
        logicalAnd(builder, loc, operandsValid, logicalNot(builder, loc,
                                                          overflow));
    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc, *resultTypes, fastValid, /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    RuntimeBundle fastBundle;
    if (mlir::failed(RuntimeBundleLowerer::initializeObjectFromRawValues(
            op, resultType, mlir::ValueRange{rawResult}, fastBundle)))
      return mlir::failure();
    if (mlir::failed(checkPhysicalTypes(fastBundle.physicalValues(),
                                        "primitive i64 fast path")))
      return mlir::failure();
    builder.create<mlir::scf::YieldOp>(loc, fastBundle.physicalValues());

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (mlir::failed(emitFallbackYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(ifOp);
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op, resultType, ifOp.getResults(), result)))
      return mlir::failure();
    result.primitiveI64 = RuntimePrimitiveI64Evidence{rawResult, fastValid};
    valueBundles[resultValue] = std::move(result);
    return mlir::success();
  }

  if (resultTypes->size() != 1 || !resultTypes->front().isInteger(1))
    return op->emitError() << "primitive i64 comparison " << methodName
                           << " expects a single i1 bool ABI result";
  mlir::Value fastResult =
      builder.create<mlir::arith::CmpIOp>(loc, *compare, lhs.value, rhs.value)
          .getResult();
  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc, *resultTypes, operandsValid, /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{fastResult});

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  if (mlir::failed(emitFallbackYield()))
    return mlir::failure();

  builder.setInsertionPointAfter(ifOp);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, resultType, ifOp.getResults(), result)))
    return mlir::failure();
  valueBundles[resultValue] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBinarySpecial(
    mlir::Operation *op, mlir::Value lhs, mlir::Value rhs,
    llvm::StringRef methodName, mlir::Value resultValue) {
  llvm::SmallVector<mlir::Value, 2> inputs{lhs, rhs};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "binary special method operands need runtime bundles",
          sources)))
    return mlir::failure();
  if (sources.size() == 2 && hasPrimitiveI64Evidence(sources[0]) &&
      hasPrimitiveI64Evidence(sources[1]) &&
      (primitiveI64ArithmeticKind(methodName) ||
       primitiveI64ComparePredicate(methodName))) {
    if (mlir::failed(RuntimeBundleLowerer::lowerPrimitiveI64BinarySpecial(
            op, methodName, sources, resultValue)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }
  if (mlir::failed(lowerManifestMethodResult(
          op, resultValue, *sources.front(), methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectSingleBuiltinArgument(
    py::CallOp op, const RuntimeSymbol &symbol,
    const RuntimeBundle *&argument) const {
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

  if (symbol.builtinName == "repr" && symbol.builtinMethod == "__repr__" &&
      RuntimeBundleLowerer::needsDefaultObjectRepr(*argument)) {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
            op, *argument, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{argument};
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(RuntimeBundleLowerer::emitManifestMethodCall(
          op, *argument, symbol.builtinMethod, sources,
          /*allowUnusedSources=*/false, emitted)))
    return mlir::failure();

  std::string resultContract = runtimeContractName(op.getResult(0).getType());
  if (resultContract.empty() || resultContract == "builtins.object")
    resultContract = symbol.resultContract;
  if (resultContract.empty())
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' needs a concrete result contract";

  RuntimeBundle result;
  if (mlir::failed(
          bundleRuntimeResults(op, runtimeContractType(context, resultContract),
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

  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

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
          op, runtimeContractType(context, resultContract), call, result)))
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

  RuntimeBundle printable = *argument;
  if (printable.contractName() != symbol.builtinSinkContract &&
      symbol.builtinMethod == "__repr__" &&
      symbol.builtinSinkContract == "builtins.str") {
    auto renderRepr =
        [&](auto &&self,
            const RuntimeBundle &bundle) -> std::optional<std::string> {
      if (bundle.literalText && bundle.contractName() == "builtins.str")
        return *bundle.literalText;
      if (!bundle.sequenceElementBundles.empty()) {
        std::string text = "[";
        for (auto [index, element] :
             llvm::enumerate(bundle.sequenceElementBundles)) {
          if (index)
            text += ", ";
          if (element) {
            std::optional<std::string> rendered = self(self, *element);
            text += rendered ? *rendered : element->contractName();
          } else {
            text += "builtins.object";
          }
        }
        text += "]";
        return text;
      }
      if (!bundle.fieldBundles.empty() &&
          RuntimeBundleLowerer::classDefinesMethod(bundle.contract,
                                                   "__repr__")) {
        std::string contractName = bundle.contractName();
        llvm::StringRef contract(contractName);
        std::string text = contract.rsplit('.').second.str();
        if (text.empty())
          text = contract.str();
        text += "(";
        llvm::SmallVector<llvm::StringRef, 4> names;
        for (const auto &entry : bundle.fieldBundles)
          names.push_back(entry.getKey());
        llvm::sort(names);
        for (auto [index, name] : llvm::enumerate(names)) {
          if (index)
            text += ", ";
          text += name.str();
          text += "=";
          auto field = bundle.fieldBundles.find(name);
          if (field != bundle.fieldBundles.end() && field->second) {
            std::optional<std::string> rendered = self(self, *field->second);
            text += rendered ? *rendered : field->second->contractName();
          } else {
            text += "builtins.object";
          }
        }
        text += ")";
        return text;
      }
      return std::nullopt;
    };
    if (std::optional<std::string> rendered =
            renderRepr(renderRepr, printable)) {
      if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
              op, *rendered, printable)))
        return mlir::failure();
      printable.literalText = std::move(*rendered);
    }
  }
  if (printable.contractName() != symbol.builtinSinkContract) {
    if (symbol.builtinMethod == "__repr__" &&
        RuntimeBundleLowerer::needsDefaultObjectRepr(*argument)) {
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, *argument, printable)))
        return mlir::failure();
    } else {
      llvm::SmallVector<const RuntimeBundle *, 1> sources{argument};
      std::optional<EmittedRuntimeCall> emitted;
      if (mlir::failed(emitManifestMethodCall(
              op, *argument, symbol.builtinMethod, sources,
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
  std::string resultContract =
      symbol.resultContract.empty() ? "types.NoneType" : symbol.resultContract;
  for (mlir::Value result : op.getResults()) {
    if (mlir::failed(assignObjectBundle(
            op, result, runtimeContractType(context, resultContract), {})))
      return mlir::failure();
  }
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
