#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

mlir::LogicalResult
RuntimeBundleLowerer::lowerStrConstant(py::StrConstantOp op) {
  if (isStaticKeywordName(op)) {
    erase.push_back(op);
    return mlir::success();
  }

  builder.setInsertionPoint(op);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
          op, op.getValue(), result)))
    return mlir::failure();
  if (objectShapeMatches(runtimeContractName(op.getResult().getType()),
                         result.physicalValues()))
    result = RuntimeBundle::object(op.getResult().getType(),
                                   result.physicalValues());
  result.literalText = op.getValue().str();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

bool RuntimeBundleLowerer::isStaticKeywordName(py::StrConstantOp op) const {
  if (op.getResult().use_empty())
    return false;
  for (mlir::OpOperand &use : op.getResult().getUses()) {
    auto pack = mlir::dyn_cast<py::PackOp>(use.getOwner());
    if (!pack)
      return false;
    bool feedsKeywordNames = false;
    for (mlir::OpOperand &packUse : pack.getResult().getUses()) {
      auto call = mlir::dyn_cast<py::CallOp>(packUse.getOwner());
      if (call && call.getKwnames() == pack.getResult()) {
        feedsKeywordNames = true;
        break;
      }
      auto init = mlir::dyn_cast<py::InitOp>(packUse.getOwner());
      if (init && init.getKwnames() == pack.getResult()) {
        feedsKeywordNames = true;
        break;
      }
      auto newOp = mlir::dyn_cast<py::NewOp>(packUse.getOwner());
      if (newOp && newOp.getKwnames() == pack.getResult()) {
        feedsKeywordNames = true;
        break;
      }
    }
    if (!feedsKeywordNames)
      return false;
  }
  return true;
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBytesConstant(py::BytesConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value buffer =
      RuntimeBundleLowerer::materializeByteBuffer(loc, op.getValue());
  mlir::Value start =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value length =
      mlir::arith::ConstantIntOp::create(
          builder, loc, static_cast<std::int64_t>(op.getValue().size()), 64)
          .getResult();
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::initializeObjectFromRawValues(
          op, runtimeContractType(context, "builtins.bytes"),
          mlir::ValueRange{buffer, start, length}, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerIntConstant(py::IntConstantOp op) {
  std::int64_t parsed = 0;
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  if (!op.getValue().getAsInteger(10, parsed)) {
    mlir::Value value =
        mlir::arith::ConstantIntOp::create(builder, loc, parsed, 64)
            .getResult();
    mlir::Value valid =
        mlir::arith::ConstantIntOp::create(builder, loc, 1, 1).getResult();
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
            op, op.getResult().getType(), value, valid, result)))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  // Beyond i64: split the decimal text into 30-bit limbs at compile time and
  // construct the boxed digit form. No primitiveI64 evidence lane is attached,
  // so downstream arithmetic takes the manifest (digit) path for this value.
  llvm::StringRef text = op.getValue();
  bool negative = text.consume_front("-");
  llvm::APInt magnitude;
  if (text.empty() || text.getAsInteger(10, magnitude))
    return op.emitError() << "invalid integer literal '" << op.getValue()
                          << "'";
  unsigned digitCount = (magnitude.getActiveBits() + 29) / 30;
  if (digitCount == 0)
    digitCount = 1;
  magnitude = magnitude.zext(digitCount * 30);

  mlir::Value dynamicSize =
      mlir::arith::ConstantIndexOp::create(builder, loc, digitCount)
          .getResult();
  auto memrefType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI32Type());
  mlir::Value buffer =
      mlir::memref::AllocaOp::create(builder, loc, memrefType,
                                     mlir::ValueRange{dynamicSize})
          .getResult();
  for (unsigned index = 0; index < digitCount; ++index) {
    std::uint64_t digit = magnitude.extractBitsAsZExtValue(30, index * 30);
    mlir::Value position =
        mlir::arith::ConstantIndexOp::create(builder, loc, index).getResult();
    mlir::Value value = mlir::arith::ConstantIntOp::create(
                            builder, loc, static_cast<std::int64_t>(digit), 32)
                            .getResult();
    mlir::memref::StoreOp::create(builder, loc, value, buffer,
                                  mlir::ValueRange{position});
  }
  mlir::Value sign =
      mlir::arith::ConstantIntOp::create(builder, loc, negative ? -1 : 1, 64)
          .getResult();

  std::optional<RuntimeSymbol> fromDigits =
      manifest.primitive("builtins.int", "from_digits");
  if (!fromDigits)
    return op.emitError()
           << "runtime manifest has no builtins.int from_digits primitive";
  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      loc, *fromDigits, mlir::ValueRange{sign, buffer});
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, op.getResult().getType(), call, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFloatConstant(py::FloatConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Value value =
      mlir::arith::ConstantFloatOp::create(builder, op.getLoc(),
                                           builder.getF64Type(), op.getValue())
          .getResult();
  RuntimeBundle result;
  if (mlir::failed(initializeObjectFromRawValues(
          op, op.getResult().getType(), mlir::ValueRange{value}, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBoolConstant(py::BoolConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Value bit = mlir::arith::ConstantIntOp::create(builder, op.getLoc(),
                                                       op.getValue() ? 1 : 0, 1)
                        .getResult();
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "builtins.bool"),
          bit)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNone(py::NoneOp op) {
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerCastFromPrim(py::CastFromPrimOp op) {
  std::string expected = runtimeContractName(op.getResult().getType());
  if (expected.empty())
    return op.emitError() << "primitive cast result has no runtime contract";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 1> inputValues{op.getInput()};
  if (expected == "builtins.int") {
    if (auto integer =
            mlir::dyn_cast<mlir::IntegerType>(op.getInput().getType())) {
      mlir::Value value = op.getInput();
      mlir::Type i64 = builder.getI64Type();
      if (integer.getWidth() < 64) {
        if (integer.getWidth() == 1)
          value = mlir::arith::ExtUIOp::create(builder, op.getLoc(), i64, value)
                      .getResult();
        else
          value = mlir::arith::ExtSIOp::create(builder, op.getLoc(), i64, value)
                      .getResult();
      } else if (integer.getWidth() > 64) {
        value = mlir::arith::TruncIOp::create(builder, op.getLoc(), i64, value)
                    .getResult();
      }
      mlir::Value valid =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1)
              .getResult();
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
              op, op.getResult().getType(), value, valid, result)))
        return mlir::failure();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }
  if (objectShapeMatches(expected, inputValues)) {
    if (mlir::failed(assignObjectBundle(op, op.getResult(),
                                        runtimeContractType(context, expected),
                                        inputValues)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  RuntimeBundle result;
  if (mlir::succeeded(initializeObjectFromRawValues(
          op, op.getResult().getType(), inputValues, result,
          /*emitErrors=*/false))) {
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  return op.emitError() << "primitive cast to " << expected
                        << " has no manifest-driven runtime lowering yet";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerCastToPrim(py::CastToPrimOp op) {
  builder.setInsertionPoint(op);
  if (mlir::failed(ensureValueBundle(op, op.getInput())))
    return mlir::failure();
  const RuntimeBundle *source = bundleFor(op.getInput());
  if (!source)
    return op.emitError() << "cast.to_prim input has no lowered runtime bundle";

  mlir::Location loc = op.getLoc();
  mlir::Type resultType = op.getResult().getType();
  llvm::StringRef mode = op.getMode();

  auto finish = [&](mlir::Value value) {
    op.getResult().replaceAllUsesWith(value);
    erase.push_back(op);
    return mlir::success();
  };

  // The i64 payload of an int-like source. The evidence lane carries a valid
  // bit because arithmetic may overflow into the boxed representation; using
  // the lane value while it is invalid would be a silent mis-execution, so it
  // is asserted rather than branched around (the boxed fallback has no
  // statically known shape here).
  auto materializeI64 = [&]() -> mlir::FailureOr<mlir::Value> {
    if (source->primitiveI64) {
      mlir::cf::AssertOp::create(
          builder, loc, source->primitiveI64->valid,
          "lython: int value left the primitive i64 range before a "
          "primitive cast");
      return source->primitiveI64->value;
    }
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source->contractName(), "unbox.i64");
    if (!unbox || unbox->function.getNumArguments() !=
                      source->physicalValues().size())
      return op.emitError()
             << "cast.to_prim from " << source->contractName()
             << " has no i64 unbox primitive";
    mlir::func::CallOp call =
        createRuntimeCall(loc, *unbox, source->physicalValues());
    if (call.getNumResults() != 1 ||
        !call.getResult(0).getType().isInteger(64))
      return op.emitError()
             << "cast.to_prim i64 unbox primitive must return i64";
    return call.getResult(0);
  };

  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(resultType)) {
    mlir::Value raw;
    if (source->primitiveI64) {
      mlir::FailureOr<mlir::Value> payload = materializeI64();
      if (mlir::failed(payload))
        return mlir::failure();
      raw = mlir::arith::SIToFPOp::create(builder, loc, floatType, *payload)
                .getResult();
      return finish(raw);
    }
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(source->contractName(), "unbox.f64");
    if (!unbox || unbox->function.getNumArguments() !=
                      source->physicalValues().size())
      return op.emitError()
             << "cast.to_prim from " << source->contractName()
             << " has no f64 unbox primitive";
    mlir::func::CallOp call =
        createRuntimeCall(loc, *unbox, source->physicalValues());
    if (call.getNumResults() != 1 || !call.getResult(0).getType().isF64())
      return op.emitError()
             << "cast.to_prim f64 unbox primitive must return f64";
    raw = call.getResult(0);
    unsigned width = floatType.getWidth();
    if (width < 64)
      raw = mlir::arith::TruncFOp::create(builder, loc, floatType, raw)
                .getResult();
    return finish(raw);
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(resultType)) {
    mlir::FailureOr<mlir::Value> payload = materializeI64();
    if (mlir::failed(payload))
      return mlir::failure();
    mlir::Value value = *payload;
    unsigned width = intType.getWidth();
    if (width > 64)
      return finish(
          mlir::arith::ExtSIOp::create(builder, loc, intType, value)
              .getResult());
    if (width == 64)
      return finish(value);

    mlir::Type i64 = builder.getI64Type();
    if (mode == "saturate") {
      std::int64_t max = (std::int64_t(1) << (width - 1)) - 1;
      std::int64_t min = -(std::int64_t(1) << (width - 1));
      mlir::Value maxValue =
          mlir::arith::ConstantIntOp::create(builder, loc, max, 64)
              .getResult();
      mlir::Value minValue =
          mlir::arith::ConstantIntOp::create(builder, loc, min, 64)
              .getResult();
      value = mlir::arith::MinSIOp::create(builder, loc, value, maxValue)
                  .getResult();
      value = mlir::arith::MaxSIOp::create(builder, loc, value, minValue)
                  .getResult();
    }
    mlir::Value narrowed =
        mlir::arith::TruncIOp::create(builder, loc, intType, value)
            .getResult();
    if (mode == "exact") {
      mlir::Value widened =
          mlir::arith::ExtSIOp::create(builder, loc, i64, narrowed)
              .getResult();
      mlir::Value fits =
          mlir::arith::CmpIOp::create(builder, loc,
                                      mlir::arith::CmpIPredicate::eq, widened,
                                      value)
              .getResult();
      mlir::cf::AssertOp::create(
          builder, loc, fits,
          "lython: int value is out of range for the primitive integer "
          "width");
    }
    return finish(narrowed);
  }

  return op.emitError() << "cast.to_prim result type is not a primitive "
                           "integer or float";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerTypeObject(py::TypeObjectOp op) {
  valueBundles[op.getResult()] = RuntimeBundle::typeObject(
      op.getResult().getType(), op.getInstanceContract());
  erase.push_back(op);
  return mlir::success();
}
} // namespace py::lowering
