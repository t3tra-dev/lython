#include "Runtime/Core/Lowerer.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace py::lowering {

namespace {

enum class PrimitiveI64ArithmeticKind { Add, Sub, Mul };

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

mlir::Value boolConstant(mlir::OpBuilder &builder, mlir::Location loc,
                         bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1)
      .getResult();
}

bool isKnownTrue(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>();
  if (constant)
    return constant.value() != 0;
  if (auto andOp = value.getDefiningOp<mlir::arith::AndIOp>())
    return isKnownTrue(andOp.getLhs()) && isKnownTrue(andOp.getRhs());
  return false;
}

mlir::Value i64Constant(mlir::OpBuilder &builder, mlir::Location loc,
                        std::int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64).getResult();
}

mlir::Value logicalAnd(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value lhs, mlir::Value rhs) {
  return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs).getResult();
}

mlir::Value logicalNot(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value value) {
  return builder
      .create<mlir::arith::XOrIOp>(loc, value, boolConstant(builder, loc, true))
      .getResult();
}

mlir::Value signedAddOverflow(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs,
                              mlir::Value result) {
  mlir::Value zero = i64Constant(builder, loc, 0);
  mlir::Value lhsNegative =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                       lhs, zero)
          .getResult();
  mlir::Value rhsNegative =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                       rhs, zero)
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
  mlir::Value lhsNegative =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                       lhs, zero)
          .getResult();
  mlir::Value rhsNegative =
      builder
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                       rhs, zero)
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
    auto extended = builder.create<mlir::arith::MulSIExtendedOp>(loc, lhs, rhs);
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

mlir::LogicalResult RuntimeBundleLowerer::lowerPrimitiveI64BinarySpecial(
    mlir::Operation *op, llvm::StringRef methodName,
    llvm::ArrayRef<const RuntimeBundle *> sources, mlir::Value resultValue) {
  if (sources.size() != 2 ||
      !RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[0]) ||
      !RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[1]))
    return op->emitError()
           << "primitive i64 lowering requires two int operands with evidence";

  std::optional<PrimitiveI64ArithmeticKind> arithmetic =
      primitiveI64ArithmeticKind(methodName);
  std::optional<mlir::arith::CmpIPredicate> compare =
      primitiveI64ComparePredicate(methodName);
  if (!arithmetic && !compare)
    return op->emitError() << "unsupported primitive i64 special method "
                           << methodName;

  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  const RuntimePrimitiveI64Evidence &lhs = *sources[0]->primitiveI64;
  const RuntimePrimitiveI64Evidence &rhs = *sources[1]->primitiveI64;
  mlir::Value operandsValid = logicalAnd(builder, loc, lhs.valid, rhs.valid);

  if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(
          op->getParentOfType<mlir::func::FuncOp>())) {
    if (arithmetic) {
      auto [rawResult, overflow] = buildPrimitiveI64Arithmetic(
          builder, loc, *arithmetic, lhs.value, rhs.value);
      mlir::Value valid = logicalAnd(builder, loc, operandsValid,
                                     logicalNot(builder, loc, overflow));
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
              op, resultValue.getType(), rawResult, valid, result)))
        return mlir::failure();
      valueBundles[resultValue] = std::move(result);
      return mlir::success();
    }

    mlir::Value compared =
        builder.create<mlir::arith::CmpIOp>(loc, *compare, lhs.value, rhs.value)
            .getResult();
    mlir::Value fastResult = logicalAnd(builder, loc, operandsValid, compared);
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, resultValue.getType(), mlir::ValueRange{fastResult}, result)))
      return mlir::failure();
    valueBundles[resultValue] = std::move(result);
    return mlir::success();
  }

  mlir::FailureOr<RuntimeSymbol> selected =
      RuntimeBundleLowerer::selectManifestMethod(op, *sources.front(),
                                                 methodName, sources,
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

  auto checkPhysicalTypes = [&](mlir::ValueRange values,
                                llvm::StringRef label) -> mlir::LogicalResult {
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

  auto emitFallbackYield = [&]() -> mlir::LogicalResult {
    mlir::Block *fallbackBlock = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(fallbackBlock);
    llvm::SmallVector<RuntimeBundle, 2> materializedSources;
    llvm::SmallVector<const RuntimeBundle *, 2> fallbackSources;
    materializedSources.reserve(sources.size());
    fallbackSources.reserve(sources.size());
    for (const RuntimeBundle *source : sources) {
      if (!source || !RuntimeBundleLowerer::hasLazyPrimitiveI64Object(*source)) {
        fallbackSources.push_back(source);
        continue;
      }
      mlir::FailureOr<RuntimeValue> materialized =
          RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
              op, *source);
      if (mlir::failed(materialized))
        return mlir::failure();
      RuntimeBundle updated = *source;
      updated.contract = materialized->contract;
      updated.objectValue = *materialized;
      materializedSources.push_back(std::move(updated));
      fallbackSources.push_back(&materializedSources.back());
    }
    llvm::SmallVector<mlir::Value, 8> operands;
    if (mlir::failed(RuntimeBundleLowerer::buildRuntimeCallOperands(
            op, *selected, fallbackSources, operands,
            /*allowUnusedSources=*/false)))
      return mlir::failure();
    builder.setInsertionPointToEnd(fallbackBlock);
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
    mlir::Value fastValid = logicalAnd(builder, loc, operandsValid,
                                       logicalNot(builder, loc, overflow));
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, *resultTypes, fastValid,
                                                /*withElseRegion=*/true);

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
  if (isKnownTrue(operandsValid)) {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, resultType, mlir::ValueRange{fastResult}, result)))
      return mlir::failure();
    valueBundles[resultValue] = std::move(result);
    return mlir::success();
  }
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, *resultTypes, operandsValid,
                                              /*withElseRegion=*/true);

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
  if (sources.size() == 2 &&
      RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[0]) &&
      RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[1]) &&
      (primitiveI64ArithmeticKind(methodName) ||
       primitiveI64ComparePredicate(methodName))) {
    if (mlir::failed(RuntimeBundleLowerer::lowerPrimitiveI64BinarySpecial(
            op, methodName, sources, resultValue)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }
  if (methodName == "__mul__" && sources.size() == 2)
    if (mlir::succeeded(RuntimeBundleLowerer::lowerStaticCtypesArrayTypeMul(
            op, *sources[0], *sources[1], resultValue))) {
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

} // namespace py::lowering
