#include "Runtime/Core/Lowerer.h"

#include "ExceptionTaxonomy.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include <cctype>

// Loop-body generators: transform an int-pure generator body into a
// primitive-i64 RESUME state machine (pure SSA — no in-body memory):
//
//   @body(args: int...)                     [py CFG with py.yield_value]
//     ==>
//   @body__lyrt_gen_resume(args..., state: int, sent: int, inject: int,
//                          lives: int x K)
//       -> (state': int, has: int, value: int,
//           ret: int, hasret: int, lives': int x K)
//
// Every yield splits its block; the continuation's externally-defined values
// become block arguments (the live set, all builtins.int in this tier), the
// yield becomes `return (k, 1, value, 0, 0, lives...)`, and a raw-i64
// dispatch chain in the entry branches on the state argument. The `sent`
// lane carries the send() value into the resumed continuation (the yield
// expression result aliases the entry argument), and the `inject` lane makes
// the continuation rethrow the EH TLS exception at the suspension point —
// which is how throw() and close() (GeneratorExit) reach the body's
// handlers. `ret`/`hasret` carry the generator's return value out so the
// exhaustion StopIteration can expose it.
//
// The generator object stores state (word 4) and the live slots (words 8..)
// inline. All resumes go through synthesized driver functions instead of
// inline call-site expansion:
//
//   @body...__step(gen, argpairs..., sent, sentvalid, inject)
//       -> (has, value, valid, ret, retvalid)
//   @body...__advance = step + StopIteration(value) on exhaustion
//   @body...__throw   = TLS-stage exception, resume with inject=1
//   @body...__close   = GeneratorExit injection + swallow + closed state
//
// A separate function keeps the multi-block EH CFG legal at call sites that
// sit inside single-block SCF regions (for-loop conditions), and it makes
// the unwind edge originate at a call — which is the shape the ownership
// inserter models when it places compensating releases at catch entries.

namespace py::lowering {

namespace {

constexpr unsigned kGeneratorFrameSlotBase = 8;
constexpr unsigned kGeneratorFrameSlotLimit = 64;
constexpr llvm::StringLiteral kGeneratorBodyResultAttr{
    "ly.generator.body_result"};
constexpr llvm::StringLiteral kGeneratorPublicResultAttr{
    "ly.generator.public_result"};

// Backward liveness over a function region: liveIn(B) = values used in B or
// live into a successor, minus values B itself defines (op results and block
// arguments). Deterministic (SetVector, block/op order).
llvm::DenseMap<mlir::Block *, llvm::SetVector<mlir::Value>>
computeLiveIns(mlir::Region &region) {
  llvm::DenseMap<mlir::Block *, llvm::SetVector<mlir::Value>> liveIns;
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Block &blockRef : llvm::reverse(region.getBlocks())) {
      mlir::Block *block = &blockRef;
      llvm::SetVector<mlir::Value> live;
      for (mlir::Block *successor : block->getSuccessors())
        for (mlir::Value value : liveIns[successor])
          live.insert(value);
      for (mlir::Operation &op : llvm::reverse(*block)) {
        for (mlir::Value result : op.getResults())
          live.remove(result);
        for (mlir::Value operand : op.getOperands())
          live.insert(operand);
      }
      for (mlir::BlockArgument argument : block->getArguments())
        live.remove(argument);
      llvm::SetVector<mlir::Value> &slot = liveIns[block];
      if (live.size() != slot.size() ||
          !llvm::all_of(live,
                        [&](mlir::Value v) { return slot.contains(v); })) {
        slot = std::move(live);
        changed = true;
      }
    }
  }
  return liveIns;
}

// Values live immediately AFTER `yield`, excluding the function's entry
// block arguments (real arguments are re-passed on every resume; everything
// else must ride in the frame) and the yield's own sent result (it does not
// exist at suspend time — the resume's sent lane materializes it).
llvm::SetVector<mlir::Value> liveAfterYield(
    py::YieldValueOp yield,
    llvm::DenseMap<mlir::Block *, llvm::SetVector<mlir::Value>> &liveIns,
    mlir::Block *entry) {
  mlir::Block *block = yield->getBlock();
  llvm::SetVector<mlir::Value> live;
  for (mlir::Block *successor : block->getSuccessors())
    for (mlir::Value value : liveIns[successor])
      live.insert(value);
  for (mlir::Operation &op : llvm::reverse(*block)) {
    if (&op == yield.getOperation())
      break;
    for (mlir::Value result : op.getResults())
      live.remove(result);
    for (mlir::Value operand : op.getOperands())
      live.insert(operand);
  }
  live.remove(yield.getSent());
  llvm::SetVector<mlir::Value> filtered;
  for (mlir::Value value : live) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (argument && argument.getOwner() == entry)
      continue;
    filtered.insert(value);
  }
  return filtered;
}

bool isIntContract(mlir::Type type) {
  return runtimeContractName(type) == "builtins.int";
}

bool isNoneLike(mlir::Type type) {
  if (runtimeContractName(type) == "types.NoneType")
    return true;
  auto literal = mlir::dyn_cast<py::LiteralType>(type);
  return literal && literal.getSpelling() == "None";
}

mlir::MemRefType generatorStorageType(mlir::OpBuilder &builder) {
  return mlir::MemRefType::get({kGeneratorFrameSlotLimit},
                               builder.getI64Type());
}

std::int64_t exceptionClassId(llvm::StringRef name) {
  const py::exceptions::BuiltinExceptionInfo *info =
      py::exceptions::findByName(name);
  return info ? info->classId : 0;
}

// Number of logical extra lanes before the frame slots: state, sent, inject.
constexpr unsigned kResumeControlLanes = 3;
// Number of logical result lanes before the frame slots:
// state', has, value, ret, hasret.
constexpr unsigned kResumeResultLanes = 5;
// Logical result index of the yielded-value lane.
constexpr unsigned kResumeValueLaneIndex = 2;

} // namespace

// Map a resume clone back to its GeneratorResumeInfo: the clone carries the
// original function's name in kPrimitiveI64CloneAttr, which keys the map.
RuntimeBundleLowerer::GeneratorResumeInfo *
RuntimeBundleLowerer::generatorResumeInfoForClone(mlir::func::FuncOp clone) {
  if (!clone || !clone->hasAttr("ly.generator.resume"))
    return nullptr;
  auto original =
      clone->getAttrOfType<mlir::StringAttr>(kPrimitiveI64CloneAttr);
  if (!original)
    return nullptr;
  auto found = generatorResumeClones.find(original.getValue());
  if (found == generatorResumeClones.end())
    return nullptr;
  return &found->second;
}

mlir::FailureOr<RuntimeBundleLowerer::GeneratorResumeLane>
RuntimeBundleLowerer::computeGeneratorResumeLane(mlir::Operation *op,
                                                 mlir::Type type) {
  GeneratorResumeLane lane;
  if (isNoneLike(type)) {
    lane.contract = "types.NoneType";
    lane.isNone = true;
    lane.physicalCount = 0;
    return lane;
  }
  std::string contract = runtimeContractName(type);
  if (contract.empty())
    return op->emitError()
           << "generator suspension lane has no concrete runtime contract: "
           << type;
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  if (!shape)
    return op->emitError()
           << "generator suspension lane contract " << contract
           << " has no runtime ABI shape";
  lane.contract = contract;
  lane.isInt = contract == "builtins.int";
  lane.physicalCount = static_cast<unsigned>(shape->valueTypes.size());
  return lane;
}

llvm::SmallVector<mlir::Type, 6> RuntimeBundleLowerer::generatorLanePhysicalTypes(
    const GeneratorResumeLane &lane) const {
  llvm::SmallVector<mlir::Type, 6> types;
  mlir::Type i64 = mlir::IntegerType::get(context, 64);
  mlir::Type i1 = mlir::IntegerType::get(context, 1);
  if (lane.isControl()) {
    types.push_back(i64);
    types.push_back(i1);
    return types;
  }
  if (const RuntimeValueShape *shape = manifest.valueShape(lane.contract))
    types.append(shape->valueTypes.begin(), shape->valueTypes.end());
  if (lane.isInt) {
    types.push_back(i64);
    types.push_back(i1);
  }
  return types;
}

// Release-safe placeholders for a value-bearing lane on non-yield exits:
// immortal static globals, so the consumer's owned-result release obligation
// stays dischargeable (a no-op) on paths where no value was yielded.
mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
RuntimeBundleLowerer::materializeGeneratorDeadLaneValues(
    mlir::Operation *op, const GeneratorResumeLane &lane) {
  llvm::SmallVector<mlir::Value, 4> values;
  if (lane.isControl() || lane.isNone)
    return values;
  mlir::FailureOr<RuntimeValue> dead =
      RuntimeBundleLowerer::materializeNonOwningDeadObjectValue(
          op, runtimeContractType(context, lane.contract),
          "generator suspend placeholder");
  if (mlir::failed(dead))
    return mlir::failure();
  values.append(dead->values.begin(), dead->values.end());
  if (lane.isInt) {
    mlir::Location loc = op->getLoc();
    values.push_back(
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult());
    values.push_back(
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 1).getResult());
  }
  return values;
}

// One suspend lane's physical return operands. Ownership: the suspend result
// range is covered by the clone's owned-results contract, so appending an
// OWNED bundle transfers its token to the resumer; borrowed/immortal sources
// are retained first (CPython: the caller of next() receives a new
// reference). Pair-only ints materialize their object once, at the boundary
// (never silently mis-execute: an invalid pair here means the value left the
// i64 lane without a boxed representation, which is a loud runtime error).
mlir::LogicalResult RuntimeBundleLowerer::appendGeneratorLaneReturnOperands(
    mlir::func::ReturnOp op, const GeneratorResumeLane &lane,
    const RuntimeBundle &bundle, llvm::SmallVectorImpl<mlir::Value> &operands,
    bool forceRetain) {
  mlir::Location loc = op.getLoc();
  builder.setInsertionPoint(op);
  if (lane.isControl())
    return op.emitError()
           << "generator control lanes do not take object return operands";
  if (lane.isNone)
    return mlir::success();

  bool bundleIsNone = bundle.contractName() == "types.NoneType";
  if (bundleIsNone) {
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> dead =
        RuntimeBundleLowerer::materializeGeneratorDeadLaneValues(
            op.getOperation(), lane);
    if (mlir::failed(dead))
      return mlir::failure();
    operands.append(dead->begin(), dead->end());
    return mlir::success();
  }

  if (lane.isInt) {
    if (bundle.contractName() != "builtins.int")
      return op.emitError() << "generator yield lane expects builtins.int, got "
                            << bundle.contract;
    if (!bundle.physicalValues().empty()) {
      if (bundle.physicalValues().size() != lane.physicalCount)
        return op.emitError()
               << "generator int yield lane has "
               << bundle.physicalValues().size()
               << " physical values, but the lane ABI expects "
               << lane.physicalCount;
      if ((forceRetain ||
           bundle.objectValue.ownership != ownership::OwnershipKind::Own) &&
          mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
              op, bundle, "generator yield lane")))
        return mlir::failure();
      operands.append(bundle.physicalValues().begin(),
                      bundle.physicalValues().end());
      if (bundle.primitiveI64) {
        operands.push_back(bundle.primitiveI64->value);
        operands.push_back(bundle.primitiveI64->valid);
      } else {
        operands.push_back(
            mlir::arith::ConstantIntOp::create(builder, loc, 0, 64)
                .getResult());
        operands.push_back(
            mlir::arith::ConstantIntOp::create(builder, loc, 0, 1)
                .getResult());
      }
      return mlir::success();
    }
    if (!bundle.primitiveI64)
      return op.emitError()
             << "generator int yield lane has neither physical values nor "
                "primitive evidence";
    // Pair-only: the raw lane is authoritative only while valid. Reject an
    // invalid pair loudly before materializing.
    mlir::Value valid = bundle.primitiveI64->valid;
    mlir::Value trueValue =
        mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
    mlir::Value invalid =
        mlir::arith::XOrIOp::create(builder, loc, valid, trueValue);
    auto guard = mlir::scf::IfOp::create(builder, loc, invalid,
                                         /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard guardScope(builder);
      builder.setInsertionPoint(guard.getThenRegion().front().getTerminator());
      if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
              op.getOperation(), "builtins.ValueError",
              "int too large to convert to a native 64-bit integer")))
        return mlir::failure();
    }
    builder.setInsertionPoint(op);
    std::optional<RuntimeSymbol> initializer =
        manifest.initializer("builtins.int", "__new__");
    if (!initializer)
      return op.emitError() << "runtime manifest has no builtins.int.__new__ "
                               "for the generator yield lane";
    mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
        loc, *initializer, mlir::ValueRange{bundle.primitiveI64->value});
    if (call.getNumResults() != lane.physicalCount)
      return op.emitError() << "builtins.int.__new__ does not match the "
                               "generator yield lane ABI";
    operands.append(call.getResults().begin(), call.getResults().end());
    operands.push_back(bundle.primitiveI64->value);
    operands.push_back(trueValue);
    return mlir::success();
  }

  if (bundle.contractName() != lane.contract)
    return op.emitError() << "generator yield lane expects " << lane.contract
                          << ", got " << bundle.contract;
  if (bundle.physicalValues().size() != lane.physicalCount)
    return op.emitError() << "generator yield lane for " << lane.contract
                          << " has " << bundle.physicalValues().size()
                          << " physical values, but the lane ABI expects "
                          << lane.physicalCount;
  if ((forceRetain ||
       bundle.objectValue.ownership != ownership::OwnershipKind::Own) &&
      mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
          op, bundle, "generator yield lane")))
    return mlir::failure();
  operands.append(bundle.physicalValues().begin(),
                  bundle.physicalValues().end());
  return mlir::success();
}

unsigned RuntimeBundleLowerer::generatorLaneFrameWords(
    const GeneratorResumeLane &lane) const {
  if (lane.isControl() || lane.isNone)
    return lane.isControl() ? 2 : 0;
  return (lane.isInt ? 2 : 0) + 2 * lane.physicalCount;
}

namespace {

std::string generatorLaneSymbolComponent(llvm::StringRef contract) {
  std::string text;
  text.reserve(contract.size());
  for (char ch : contract)
    text.push_back(std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_');
  return text;
}

} // namespace

mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorClaimFunction(
    mlir::Operation *op, const GeneratorResumeLane &lane) {
  std::string name =
      "__ly_generator_claim_" + generatorLaneSymbolComponent(lane.contract);
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = op->getLoc();
  llvm::SmallVector<mlir::Type, 6> types =
      RuntimeBundleLowerer::generatorLanePhysicalTypes(lane);
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(types, types));
  function.setPrivate();
  function->setAttr(ownership::kOwnedResultsAttr,
                    mlir::DenseI64ArrayAttr::get(
                        context, llvm::ArrayRef<std::int64_t>{0}));
  function->setAttr(
      ownership::kOwnedResultContractsAttr,
      builder.getArrayAttr({builder.getStringAttr(lane.contract)}));
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::func::ReturnOp::create(builder, loc, entry->getArguments());
  return function;
}

mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorFrameStoreFunction(
    mlir::Operation *op, const GeneratorResumeLane &lane) {
  std::string name = "__ly_generator_frame_store_" +
                     generatorLaneSymbolComponent(lane.contract);
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = op->getLoc();
  mlir::Type i64 = builder.getI64Type();
  llvm::SmallVector<mlir::Type, 8> laneTypes =
      RuntimeBundleLowerer::generatorLanePhysicalTypes(lane);
  llvm::SmallVector<mlir::Type, 10> inputs;
  inputs.push_back(generatorStorageType(builder));
  inputs.push_back(i64); // frame word base
  inputs.append(laneTypes.begin(), laneTypes.end());
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, {}));
  function.setPrivate();
  // The frame absorbs the span's token: callers hand it over for good. The
  // transfer anchors at the lane's header (operand 2); the remaining parts
  // are interior views of the same entity.
  if (lane.physicalCount > 0) {
    function->setAttr(ownership::kTransferArgsAttr,
                      builder.getI64ArrayAttr({2}));
    function.setArgAttr(2, ownership::kObjectHeaderAttr, builder.getUnitAttr());
  }

  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::Value storage = entry->getArgument(0);
  mlir::Value base = entry->getArgument(1);
  auto wordIndex = [&](unsigned offset) -> mlir::Value {
    mlir::Value word = mlir::arith::AddIOp::create(
        builder, loc, base,
        mlir::arith::ConstantIntOp::create(builder, loc, offset, 64)
            .getResult());
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), word)
        .getResult();
  };
  unsigned word = 0;
  if (lane.isInt) {
    mlir::Value raw = entry->getArgument(2 + lane.physicalCount);
    mlir::Value valid = entry->getArgument(2 + lane.physicalCount + 1);
    mlir::Value validWord =
        mlir::arith::ExtUIOp::create(builder, loc, i64, valid).getResult();
    mlir::memref::StoreOp::create(builder, loc, raw, storage, wordIndex(0));
    mlir::memref::StoreOp::create(builder, loc, validWord, storage,
                                  wordIndex(1));
    word = 2;
  }
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  for (unsigned part = 0; part < lane.physicalCount; ++part) {
    mlir::Value value = entry->getArgument(2 + part);
    auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    mlir::Value pointerWord = zero;
    mlir::Value sizeWord = zero;
    if (memref && memref.getRank() == 1) {
      mlir::Value pointerIndex =
          mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                               value);
      pointerWord = mlir::arith::IndexCastOp::create(builder, loc, i64,
                                                     pointerIndex)
                        .getResult();
      if (memref.hasStaticShape()) {
        sizeWord = mlir::arith::ConstantIntOp::create(
                       builder, loc, memref.getDimSize(0), 64)
                       .getResult();
      } else {
        mlir::Value dim =
            mlir::memref::DimOp::create(builder, loc, value, 0).getResult();
        sizeWord =
            mlir::arith::IndexCastOp::create(builder, loc, i64, dim)
                .getResult();
      }
    }
    mlir::memref::StoreOp::create(builder, loc, pointerWord, storage,
                                  wordIndex(word));
    mlir::memref::StoreOp::create(builder, loc, sizeWord, storage,
                                  wordIndex(word + 1));
    word += 2;
  }
  mlir::func::ReturnOp::create(builder, loc);
  return function;
}

mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorFrameLoadFunction(
    mlir::Operation *op, const GeneratorResumeLane &lane) {
  std::string name = "__ly_generator_frame_load_" +
                     generatorLaneSymbolComponent(lane.contract);
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = op->getLoc();
  mlir::Type i64 = builder.getI64Type();
  llvm::SmallVector<mlir::Type, 8> laneTypes =
      RuntimeBundleLowerer::generatorLanePhysicalTypes(lane);
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name,
      builder.getFunctionType({generatorStorageType(builder), i64},
                              laneTypes));
  function.setPrivate();
  function->setAttr(ownership::kOwnedResultsAttr,
                    mlir::DenseI64ArrayAttr::get(
                        context, llvm::ArrayRef<std::int64_t>{0}));
  function->setAttr(
      ownership::kOwnedResultContractsAttr,
      builder.getArrayAttr({builder.getStringAttr(lane.contract)}));

  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::Value storage = entry->getArgument(0);
  mlir::Value base = entry->getArgument(1);
  auto wordIndex = [&](unsigned offset) -> mlir::Value {
    mlir::Value word = mlir::arith::AddIOp::create(
        builder, loc, base,
        mlir::arith::ConstantIntOp::create(builder, loc, offset, 64)
            .getResult());
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), word)
        .getResult();
  };
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  unsigned word = 0;
  mlir::Value raw;
  mlir::Value valid;
  if (lane.isInt) {
    raw = mlir::memref::LoadOp::create(builder, loc, storage, wordIndex(0))
              .getResult();
    mlir::Value validWord =
        mlir::memref::LoadOp::create(builder, loc, storage, wordIndex(1))
            .getResult();
    valid = mlir::arith::CmpIOp::create(
                builder, loc, mlir::arith::CmpIPredicate::ne, validWord, zero)
                .getResult();
    word = 2;
  }
  llvm::SmallVector<mlir::Value, 6> results;
  for (unsigned part = 0; part < lane.physicalCount; ++part) {
    mlir::Value pointerWord =
        mlir::memref::LoadOp::create(builder, loc, storage, wordIndex(word))
            .getResult();
    mlir::Value sizeWord =
        mlir::memref::LoadOp::create(builder, loc, storage,
                                     wordIndex(word + 1))
            .getResult();
    results.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, pointerWord, sizeWord,
        mlir::cast<mlir::MemRefType>(laneTypes[part])));
    word += 2;
  }
  // Ownership moves out with the loaded span: zero the slot so a later
  // drop/close finalization releases only what the frame still holds.
  for (unsigned index = 0; index < word; ++index)
    mlir::memref::StoreOp::create(builder, loc, zero, storage,
                                  wordIndex(index));
  if (lane.isInt) {
    results.push_back(raw);
    results.push_back(valid);
  }
  mlir::func::ReturnOp::create(builder, loc, results);
  return function;
}

// Entry seeding for the resume clone's mixed lane ABI: control lanes get
// raw (i64, i1) pair arguments and pair-only bundles; frame lanes get their
// physical span (+ trailing pair for int) and borrowed object bundles — the
// entry arguments only feed the resume dispatch, which forwards them into
// the continuations where the claim calls take ownership.
mlir::LogicalResult RuntimeBundleLowerer::seedGeneratorResumeCloneEntry(
    mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes,
    GeneratorResumeInfo &info) {
  if (function.isDeclaration())
    return mlir::success();
  mlir::Block &entry = function.getBody().front();
  if (entry.getNumArguments() != logicalTypes.size())
    return function.emitError()
           << "generator resume clone entry argument count does not match "
              "callable_type";

  unsigned logicalArgCount = entry.getNumArguments();
  unsigned controlCount = info.argumentCount + kResumeControlLanes;
  for (auto [index, logicalType] : llvm::enumerate(logicalTypes)) {
    mlir::BlockArgument logicalArg = entry.getArgument(index);
    const GeneratorResumeLane *lane = nullptr;
    if (index >= controlCount && index - controlCount < info.frameLanes.size())
      lane = &info.frameLanes[index - controlCount];
    if (!lane || lane->isControl()) {
      if (runtimeContractName(logicalType) != "builtins.int")
        return function.emitError()
               << "generator resume clone control lane " << index
               << " must be builtins.int, got " << logicalType;
      mlir::BlockArgument raw = entry.addArgument(
          mlir::IntegerType::get(context, 64), logicalArg.getLoc());
      mlir::BlockArgument valid = entry.addArgument(
          mlir::IntegerType::get(context, 1), logicalArg.getLoc());
      RuntimeBundle bundle = RuntimeBundle::objectWithOwnership(
          logicalType, mlir::ValueRange{},
          ownership::logicalOwnershipKind(logicalType,
                                          /*ownsObject=*/false));
      bundle.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
      valueBundles[logicalArg] = std::move(bundle);
      continue;
    }
    llvm::SmallVector<mlir::Type, 6> laneTypes =
        RuntimeBundleLowerer::generatorLanePhysicalTypes(*lane);
    llvm::SmallVector<mlir::Value, 6> physicalArgs;
    for (mlir::Type laneType : laneTypes)
      physicalArgs.push_back(entry.addArgument(laneType, logicalArg.getLoc()));
    RuntimeBundle bundle;
    llvm::ArrayRef<mlir::Value> parts(physicalArgs);
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            function, logicalType, parts.take_front(lane->physicalCount),
            bundle, /*ownsObject=*/false)))
      return mlir::failure();
    if (lane->isInt)
      bundle.primitiveI64 = RuntimePrimitiveI64Evidence{
          physicalArgs[lane->physicalCount],
          physicalArgs[lane->physicalCount + 1]};
    valueBundles[logicalArg] = std::move(bundle);
  }
  callableLogicalEntryArgCounts.push_back(
      CallableLogicalEntryArgs{function, logicalArgCount});
  return mlir::success();
}

// Phase 1 (before prepareCallableFunctionABIs): decide eligibility, compute
// the frame width K, and create the resume clone with its extended logical
// signature so the primitive-i64 ABI machinery seeds it like any clone.
mlir::LogicalResult RuntimeBundleLowerer::buildGeneratorResumeCloneSignatures() {
  llvm::SmallVector<mlir::func::FuncOp, 4> bodies;
  module.walk([&](mlir::func::FuncOp fn) {
    if (fn->hasAttr(kGeneratorBodyResultAttr) && !fn.isDeclaration())
      bodies.push_back(fn);
  });

  for (mlir::func::FuncOp body : bodies) {
    bool hasYield = false;
    bool hasYieldFrom = false;
    bool unsupported = false;
    body.walk([&](mlir::Operation *op) {
      if (mlir::isa<py::YieldValueOp>(op)) {
        hasYield = true;
        return;
      }
      if (mlir::isa<py::YieldFromOp>(op)) {
        // Statically-bound `yield from inner(...)` is inlined into the
        // clone below; other delegation shapes fall back to the legacy
        // inline dispatch.
        hasYieldFrom = true;
        return;
      }
      // py.try is fine: the clone's tries are flattened to CFG below,
      // before the yield split, so suspension points inside try/finally
      // stay inside their (by then block-registered) handler scopes.
      if (op != body.getOperation() && op->getNumRegions() != 0 &&
          !mlir::isa<py::TryOp>(op))
        unsupported = true;
    });
    if (unsupported || (!hasYield && !hasYieldFrom))
      continue;
    // Straight-line int bodies also take the state machine: the inline
    // re-dispatch raises its exhaustion StopIteration inside an scf.if that
    // falls through, and an unwind that starts after the caller's releases
    // double-frees at the catch entry's compensating release. Ineligible
    // bodies (non-int surface, non-static delegation) still fall back to it.

    auto callableAttr =
        body->getAttrOfType<mlir::TypeAttr>(ownership::kCallableTypeAttr);
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable || callable.hasVararg() || callable.hasKwarg())
      continue;
    if (!llvm::all_of(callable.getPositionalTypes(), isIntContract))
      continue;

    std::string cloneName = (body.getSymName() + "__lyrt_gen_resume").str();
    mlir::func::FuncOp clone;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(body);
      clone = body.clone();
      clone.setSymName(cloneName);
      clone->removeAttr(kGeneratorBodyResultAttr);
      clone->removeAttr(kGeneratorPublicResultAttr);
      clone->setAttr(kPrimitiveI64CloneAttr,
                     builder.getStringAttr(body.getSymName()));
      // The affine-ownership verifier keys its generator-frame rule
      // (suspension boundaries carry NonObject lanes only) on this marker.
      clone->setAttr("ly.generator.resume", builder.getUnitAttr());
      mlir::SymbolTable::setSymbolVisibility(
          clone, mlir::SymbolTable::Visibility::Private);
      builder.insert(clone);
    }

    if (hasYieldFrom) {
      mlir::FailureOr<bool> inlined =
          RuntimeBundleLowerer::inlineDelegatedYieldFroms(clone);
      if (mlir::failed(inlined))
        return mlir::failure();
      if (!*inlined) {
        clone.erase();
        continue;
      }
    }

    // Eligibility on the (possibly delegation-inlined) clone: yielded values
    // may ride any manifest-shaped object lane (they cross the suspension
    // boundary through the owned-result lane contract); sent values and
    // returns still ride primitive int lanes in this tier. All yields must
    // agree on one lane contract — mixed-type yields would need a union
    // lane, which has no suspension ABI yet.
    bool eligible = true;
    std::string valueContract;
    auto laneEligibleContract = [&](mlir::Type type) -> std::string {
      if (isIntContract(type))
        return "builtins.int";
      std::string contract = runtimeContractName(type);
      if (contract.empty() || !manifest.valueShape(contract))
        return std::string();
      return contract;
    };
    llvm::SmallVector<py::YieldValueOp, 8> yields;
    clone.walk([&](mlir::Operation *op) {
      if (auto yield = mlir::dyn_cast<py::YieldValueOp>(op)) {
        yields.push_back(yield);
        std::string contract = laneEligibleContract(yield.getValue().getType());
        if (contract.empty())
          eligible = false;
        else if (valueContract.empty())
          valueContract = contract;
        else if (valueContract != contract)
          eligible = false;
        if (!yield.getSent().use_empty() &&
            !isIntContract(yield.getSent().getType()))
          eligible = false;
        return;
      }
      if (mlir::isa<py::YieldFromOp>(op) ||
          (op != clone.getOperation() && op->getNumRegions() != 0 &&
           !mlir::isa<py::TryOp>(op)))
        eligible = false;
      // The return value rides the int ret lane; anything else would be
      // silently dropped from StopIteration.value, so reject it here and
      // let the inline path emit its diagnostic.
      if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
        for (mlir::Value operand : ret.getOperands())
          if (!isIntContract(operand.getType()) &&
              !isNoneLike(operand.getType()))
            eligible = false;
    });
    if (!eligible || yields.empty()) {
      clone.erase();
      continue;
    }

    // Flatten the clone's structured tries before anything looks at the
    // CFG: the yield split can only place suspends in function-level blocks
    // (func.return is illegal inside a py.try region), and the flattening
    // registers the try blocks' handler ids — which the split continuations
    // inherit, so injected exceptions unwind into the body's handlers.
    {
      llvm::SmallVector<py::TryOp, 4> tries;
      clone.walk([&](py::TryOp tryOp) { tries.push_back(tryOp); });
      for (py::TryOp tryOp : llvm::reverse(tries))
        if (mlir::failed(lowerTry(tryOp)))
          return mlir::failure();
    }

    // Frame lanes: values live across a yield, from backward liveness on
    // the flattened clone CFG (phase 2 recomputes on the same clone; the
    // sent alias inserted there replaces the yield result one-for-one, so
    // the live sets agree). Lanes are grouped per contract — sized by the
    // maximum same-contract live count over all yields — so every
    // suspension state maps its live values onto type-stable lanes.
    bool livesEligible = true;
    llvm::StringMap<unsigned> laneCounts;
    {
      llvm::SmallVector<py::YieldValueOp, 8> cloneYields;
      clone.walk(
          [&](py::YieldValueOp yield) { cloneYields.push_back(yield); });
      mlir::Block &cloneEntry = clone.getBody().front();
      auto liveIns = computeLiveIns(clone.getBody());
      for (py::YieldValueOp yield : cloneYields) {
        llvm::SetVector<mlir::Value> lives =
            liveAfterYield(yield, liveIns, &cloneEntry);
        llvm::StringMap<unsigned> counts;
        for (mlir::Value live : lives) {
          std::string contract = laneEligibleContract(live.getType());
          if (contract.empty()) {
            livesEligible = false;
            break;
          }
          ++counts[contract];
        }
        for (auto &entry : counts) {
          unsigned &lanes = laneCounts[entry.getKey()];
          lanes = std::max(lanes, entry.getValue());
        }
      }
    }
    llvm::SmallVector<GeneratorResumeLane, 8> frameLanes;
    unsigned frameWords = 0;
    if (livesEligible) {
      llvm::SmallVector<llvm::StringRef, 8> laneContracts;
      for (auto &entry : laneCounts)
        laneContracts.push_back(entry.getKey());
      llvm::sort(laneContracts);
      for (llvm::StringRef contract : laneContracts) {
        mlir::FailureOr<GeneratorResumeLane> lane =
            RuntimeBundleLowerer::computeGeneratorResumeLane(
                clone.getOperation(),
                runtimeContractType(context, contract.str()));
        if (mlir::failed(lane))
          return mlir::failure();
        for (unsigned count = 0; count < laneCounts[contract]; ++count) {
          frameLanes.push_back(*lane);
          frameWords += RuntimeBundleLowerer::generatorLaneFrameWords(*lane);
        }
      }
    }
    unsigned frameWidth = static_cast<unsigned>(frameLanes.size());
    if (!livesEligible ||
        kGeneratorFrameSlotBase + frameWords > kGeneratorFrameSlotLimit) {
      clone.erase();
      continue;
    }

    mlir::FailureOr<GeneratorResumeLane> valueLane =
        RuntimeBundleLowerer::computeGeneratorResumeLane(
            clone.getOperation(),
            runtimeContractType(context, valueContract));
    if (mlir::failed(valueLane))
      return mlir::failure();

    mlir::Type intContract = runtimeContractType(context, "builtins.int");
    llvm::SmallVector<mlir::Type, 8> argTypes(
        callable.getPositionalTypes().begin(),
        callable.getPositionalTypes().end());
    for (unsigned lane = 0; lane < kResumeControlLanes; ++lane)
      argTypes.push_back(intContract); // state, sent, inject
    for (const GeneratorResumeLane &lane : frameLanes)
      argTypes.push_back(runtimeContractType(context, lane.contract));
    llvm::SmallVector<mlir::Type, 8> resultTypes;
    for (unsigned lane = 0; lane < kResumeResultLanes; ++lane)
      resultTypes.push_back(
          lane == kResumeValueLaneIndex
              ? runtimeContractType(context, valueContract)
              : intContract); // state', has, value, ret, hasret
    for (const GeneratorResumeLane &lane : frameLanes)
      resultTypes.push_back(runtimeContractType(context, lane.contract));

    auto newCallable = py::CallableType::get(
        context, argTypes, /*kwonly=*/{}, /*varargType=*/mlir::Type(),
        /*kwargsType=*/mlir::Type(), resultTypes);
    clone->setAttr(ownership::kCallableTypeAttr,
                   mlir::TypeAttr::get(newCallable));
    clone.setFunctionType(
        mlir::FunctionType::get(context, argTypes, resultTypes));
    mlir::Block &cloneEntry = clone.getBody().front();
    for (unsigned extra = 0; extra < kResumeControlLanes; ++extra)
      cloneEntry.addArgument(intContract, clone.getLoc());
    for (const GeneratorResumeLane &lane : frameLanes)
      cloneEntry.addArgument(runtimeContractType(context, lane.contract),
                             clone.getLoc());

    GeneratorResumeInfo info;
    info.cloneName = cloneName;
    info.frameWidth = frameWidth;
    info.argumentCount = static_cast<unsigned>(
        callable.getPositionalTypes().size());
    info.valueLane = *valueLane;
    info.frameLanes = frameLanes;
    generatorResumeClones[body.getSymName().str()] = info;
  }
  return mlir::success();
}

// PEP 380 delegation by frame merging: a statically-bound
// `yield from inner(...)` clones inner's py-level body into the outer resume
// clone. The inner yields become outer suspension points, so send() values
// flow to whichever yield is active, throw()/close() injections unwind
// through inner handlers first, and the delegate's return value is simply
// the SSA value flowing into the yield-from result — no StopIteration
// transport is involved. Runs to fixpoint so an inlined body's own
// delegations inline too; the budget bounds (mutually) recursive delegation,
// which has no static expansion.
mlir::FailureOr<bool>
RuntimeBundleLowerer::inlineDelegatedYieldFroms(mlir::func::FuncOp clone) {
  for (unsigned round = 0; round < 64; ++round) {
    py::YieldFromOp yieldFrom;
    clone.walk([&](py::YieldFromOp op) {
      yieldFrom = op;
      return mlir::WalkResult::interrupt();
    });
    if (!yieldFrom)
      return true;

    auto call = yieldFrom.getSource().getDefiningOp<py::CallOp>();
    if (!call)
      return false;
    auto binding = call.getCallable().getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return false;
    auto target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!target || target.isDeclaration() ||
        !target->hasAttr(kGeneratorBodyResultAttr))
      return false;
    py::CallableType callableType = callableTypeOf(target);
    if (!callableType || callableType.hasVararg() || callableType.hasKwarg() ||
        !callableType.getKwOnlyTypes().empty())
      return false;
    std::optional<StaticCallableInvocation> invocation =
        RuntimeBundleLowerer::collectStaticCallableInvocation(call);
    if (!invocation)
      return false;
    std::optional<CallableArgumentPlan> plan =
        RuntimeBundleLowerer::collectCallableArgumentPlan(call, callableType,
                                                          /*emitErrors=*/false);
    if (!plan || !plan->defaultedFixed.empty() ||
        !plan->varargActuals.empty() || !plan->kwargActuals.empty())
      return false;
    llvm::ArrayRef<mlir::Type> positionalTypes =
        callableType.getPositionalTypes();
    if (plan->fixedActuals.size() != positionalTypes.size())
      return false;
    llvm::SmallVector<mlir::Value, 8> mappedArgs;
    for (std::optional<unsigned> actualIndex : plan->fixedActuals) {
      if (!actualIndex || *actualIndex >= invocation->actualValues.size())
        return false;
      mappedArgs.push_back(invocation->actualValues[*actualIndex]);
    }
    llvm::SmallVector<mlir::Type, 4> closureTypes =
        RuntimeBundleLowerer::callableClosureTypes(target);
    if (closureTypes.size() != binding.getCaptures().size())
      return false;
    for (mlir::Value capture : binding.getCaptures())
      mappedArgs.push_back(capture);
    mlir::Block &targetEntry = target.getBody().front();
    if (targetEntry.getNumArguments() != mappedArgs.size())
      return false;

    mlir::Value completion = yieldFrom.getResult();
    bool completionUsed = !completion.use_empty();
    bool completionIsInt = isIntContract(completion.getType());
    if (completionUsed && !completionIsInt && !isNoneLike(completion.getType()))
      return false;
    if (completionUsed && completionIsInt) {
      bool everyReturnCarriesInt = true;
      target.walk([&](mlir::func::ReturnOp ret) {
        if (llvm::none_of(ret.getOperands(), [](mlir::Value v) {
              return isIntContract(v.getType());
            }))
          everyReturnCarriesInt = false;
      });
      if (!everyReturnCarriesInt)
        return false;
    }
    for (auto [index, value] : llvm::enumerate(mappedArgs)) {
      mlir::Type expected = targetEntry.getArgument(index).getType();
      if (value.getType() != expected &&
          !py::isAssignableTo(value.getType(), expected, call))
        return false;
    }

    mlir::Location loc = yieldFrom.getLoc();
    mlir::Block *block = yieldFrom->getBlock();
    mlir::Block *after = block->splitBlock(
        std::next(mlir::Block::iterator(yieldFrom.getOperation())));
    // Coerce actuals whose type is narrower than the inner parameter.
    {
      mlir::OpBuilder b(yieldFrom);
      for (auto [index, value] : llvm::enumerate(mappedArgs)) {
        mlir::Type expected = targetEntry.getArgument(index).getType();
        if (value.getType() != expected)
          mappedArgs[index] =
              py::ClassUpcastOp::create(b, loc, expected, value).getResult();
      }
    }
    if (completionUsed) {
      if (completionIsInt) {
        mlir::BlockArgument arg =
            after->addArgument(completion.getType(), loc);
        completion.replaceAllUsesWith(arg);
      } else {
        mlir::OpBuilder b = mlir::OpBuilder::atBlockBegin(after);
        mlir::Value none =
            py::NoneOp::create(b, loc,
                               py::LiteralType::get(context, "None"))
                .getResult();
        if (none.getType() != completion.getType())
          none = py::ClassUpcastOp::create(b, loc, completion.getType(), none)
                     .getResult();
        completion.replaceAllUsesWith(none);
      }
    }

    mlir::IRMapping mapping;
    target.getBody().cloneInto(&clone.getBody(), after->getIterator(),
                               mapping);
    mlir::Block *innerEntry = mapping.lookup(&targetEntry);
    for (auto [index, value] : llvm::enumerate(mappedArgs))
      innerEntry->getArgument(static_cast<unsigned>(index))
          .replaceAllUsesWith(value);
    innerEntry->eraseArguments(0, innerEntry->getNumArguments());

    llvm::SmallVector<mlir::func::ReturnOp, 4> innerReturns;
    for (mlir::Block &targetBlock : target.getBody())
      if (mlir::Block *mapped = mapping.lookupOrNull(&targetBlock))
        for (mlir::Operation &op : *mapped)
          if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
            innerReturns.push_back(ret);
    for (mlir::func::ReturnOp ret : innerReturns) {
      mlir::OpBuilder b(ret);
      llvm::SmallVector<mlir::Value, 1> operands;
      if (completionUsed && completionIsInt) {
        mlir::Value returned;
        for (mlir::Value operand : ret.getOperands())
          if (isIntContract(operand.getType()))
            returned = operand;
        operands.push_back(returned);
      }
      mlir::cf::BranchOp::create(b, ret.getLoc(), after, operands);
      ret.erase();
    }

    {
      mlir::OpBuilder b(yieldFrom);
      mlir::cf::BranchOp::create(b, loc, innerEntry);
    }
    yieldFrom.erase();
    if (llvm::all_of(call->getResults(),
                     [](mlir::Value v) { return v.use_empty(); })) {
      llvm::SmallVector<mlir::Operation *, 4> packs;
      for (mlir::Value operand :
           {call.getPosargs(), call.getKwnames(), call.getKwvalues()})
        if (mlir::Operation *pack = operand.getDefiningOp())
          packs.push_back(pack);
      call.erase();
      for (mlir::Operation *pack : packs)
        if (pack->use_empty())
          pack->erase();
      if (binding->use_empty())
        binding->erase();
    }
  }
  return clone.emitError()
         << "yield from delegation exceeded the static inlining budget "
            "(recursive delegation has no static expansion)";
}

// Phase 2 (after ABI seeding): CFG surgery on the seeded clone.
mlir::LogicalResult RuntimeBundleLowerer::buildGeneratorResumeBodies() {
  for (auto &entry : generatorResumeClones) {
    GeneratorResumeInfo &info = entry.second;
    mlir::func::FuncOp clone =
        module.lookupSymbol<mlir::func::FuncOp>(info.cloneName);
    if (!clone)
      return module.emitError()
             << "generator resume clone " << info.cloneName << " is missing";
    unsigned argumentCount = info.argumentCount;
    unsigned frameWidth = info.frameWidth;
    unsigned controlCount = argumentCount + kResumeControlLanes;
    unsigned logicalCount = controlCount + frameWidth;
    mlir::Block &entryBlock = clone.getBody().front();
    // Per-logical-lane physical widths: control lanes ride the (i64, i1)
    // pair, frame lanes their physical span (+ pair for int).
    llvm::SmallVector<unsigned, 16> laneWidths(logicalCount, 2);
    for (auto [index, lane] : llvm::enumerate(info.frameLanes))
      laneWidths[controlCount + index] = static_cast<unsigned>(
          RuntimeBundleLowerer::generatorLanePhysicalTypes(lane).size());
    llvm::SmallVector<unsigned, 16> laneOffsets(logicalCount, 0);
    unsigned physicalTotal = 0;
    for (unsigned index = 0; index < logicalCount; ++index) {
      laneOffsets[index] = logicalCount + physicalTotal;
      physicalTotal += laneWidths[index];
    }
    if (entryBlock.getNumArguments() != logicalCount + physicalTotal)
      return clone.emitError()
             << "generator resume clone entry was not seeded as a "
                "primitive-i64 callable";
    mlir::Location loc = clone.getLoc();
    mlir::Type intContract = runtimeContractType(context, "builtins.int");

    auto rawOf = [&](unsigned logicalIndex) {
      return entryBlock.getArgument(laneOffsets[logicalIndex]);
    };
    auto validOf = [&](unsigned logicalIndex) {
      return entryBlock.getArgument(laneOffsets[logicalIndex] + 1);
    };
    unsigned stateIndex = argumentCount;
    unsigned sentIndex = argumentCount + 1;
    unsigned injectIndex = argumentCount + 2;
    mlir::Value rawState = rawOf(stateIndex);

    // Frame lane assignment: lanes are grouped per contract (phase 1
    // ordering); a live value maps onto lane (start of its contract group +
    // its ordinal among same-contract lives, in live-set order).
    llvm::StringMap<unsigned> laneStarts;
    for (auto [index, lane] : llvm::enumerate(info.frameLanes))
      laneStarts.try_emplace(lane.contract, static_cast<unsigned>(index));
    auto liveContract = [&](mlir::Value value) -> std::string {
      if (isIntContract(value.getType()))
        return "builtins.int";
      if (isNoneLike(value.getType()))
        return "types.NoneType";
      return runtimeContractName(value.getType());
    };
    auto assignFrameLanes =
        [&](llvm::ArrayRef<mlir::Value> lives,
            llvm::SmallVectorImpl<unsigned> &liveLanes,
            llvm::SmallVectorImpl<mlir::Value> &laneValues)
        -> mlir::LogicalResult {
      laneValues.assign(frameWidth, mlir::Value());
      llvm::StringMap<unsigned> counts;
      for (mlir::Value live : lives) {
        std::string contract = liveContract(live);
        auto start = laneStarts.find(contract);
        unsigned lane =
            start == laneStarts.end() ? frameWidth : start->second + counts[contract];
        if (lane >= frameWidth || info.frameLanes[lane].contract != contract)
          return clone.emitError()
                 << "generator resume live value of contract " << contract
                 << " exceeds the computed frame lanes";
        ++counts[contract];
        liveLanes.push_back(lane);
        laneValues[lane] = live;
      }
      return mlir::success();
    };

    // Logical int constant with primitive evidence: an unbound py constant
    // would need the full py lowering chain; instead reuse the entry's own
    // logical/raw pairing trick — materialize via py.int.constant and let
    // the walk lower it (it runs after this transform).
    auto intLiteral = [&](mlir::OpBuilder &b,
                          std::int64_t value) -> mlir::Value {
      std::string text = std::to_string(value);
      mlir::Type literalType = py::LiteralType::get(context, text);
      mlir::Value literal =
          py::IntConstantOp::create(b, loc, literalType,
                                    b.getStringAttr(text))
              .getResult();
      return py::ClassUpcastOp::create(b, loc, intContract, literal)
          .getResult();
    };
    // Placeholder for the yielded-value lane on non-yield exits: a None
    // literal, which the generator return lowering maps to immortal dead
    // placeholders (release-safe on the consumer's owned-result contract).
    auto deadValueLaneOperand = [&](mlir::OpBuilder &b) -> mlir::Value {
      return py::NoneOp::create(b, loc, py::LiteralType::get(context, "None"))
          .getResult();
    };
    // Placeholder operand for a frame lane with no live value in a given
    // state: int lanes reuse the zero literal (its materialization is the
    // cached-zero singleton); object lanes cross dead immortal placeholders.
    auto framePlaceholderOperand = [&](mlir::OpBuilder &b,
                                       unsigned lane) -> mlir::Value {
      if (info.frameLanes[lane].isInt)
        return intLiteral(b, 0);
      return deadValueLaneOperand(b);
    };

    // Collect yields in deterministic order and the original returns.
    llvm::SmallVector<py::YieldValueOp, 8> yields;
    llvm::SmallVector<mlir::func::ReturnOp, 4> originalReturns;
    clone.walk([&](mlir::Operation *op) {
      if (auto yield = mlir::dyn_cast<py::YieldValueOp>(op))
        yields.push_back(yield);
      else if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
        originalReturns.push_back(ret);
    });
    std::int64_t doneState = static_cast<std::int64_t>(yields.size()) + 1;

    // Sent values: the yield expression result does not exist at suspend
    // time — it is the NEXT resume's sent lane. Materialize an alias op so
    // the value participates in liveness like any other int (a send() value
    // held across a later yield must ride the frame), and swap its operand
    // to the entry sent argument after the split.
    llvm::DenseMap<mlir::Operation *, py::ClassUpcastOp> sentAliases;
    for (py::YieldValueOp yield : yields) {
      if (yield.getSent().use_empty())
        continue;
      mlir::OpBuilder b(context);
      b.setInsertionPointAfter(yield);
      auto alias = py::ClassUpcastOp::create(b, loc, yield.getSent().getType(),
                                             yield.getSent());
      yield.getSent().replaceAllUsesExcept(alias.getResult(), alias);
      sentAliases[yield.getOperation()] = alias;
    }

    // Exhausted / falling-off-the-end returns. A `return X` return threads X
    // through the ret lane so the exhaustion StopIteration can carry it.
    for (mlir::func::ReturnOp ret : originalReturns) {
      mlir::OpBuilder b(ret);
      mlir::Value returnedValue;
      for (mlir::Value operand : ret.getOperands())
        if (isIntContract(operand.getType()))
          returnedValue = operand;
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.push_back(intLiteral(b, doneState));
      operands.push_back(intLiteral(b, 0)); // has
      operands.push_back(deadValueLaneOperand(b)); // value
      operands.push_back(returnedValue ? returnedValue : intLiteral(b, 0));
      operands.push_back(intLiteral(b, returnedValue ? 1 : 0));
      for (unsigned slot = 0; slot < frameWidth; ++slot)
        operands.push_back(framePlaceholderOperand(b, slot));
      mlir::func::ReturnOp::create(b, ret.getLoc(), operands);
      ret.erase();
    }

    // Live sets from backward liveness on the still-unsplit body, so the
    // frame covers uses reached through successor blocks (joins, loop
    // headers), not just the block suffix.
    auto liveIns = computeLiveIns(clone.getBody());

    // Split at each yield; the yield's live-across set becomes the
    // continuation's logical block arguments and the suspend payload.
    struct Continuation {
      mlir::Block *block = nullptr;
      unsigned liveCount = 0;
      // Frame lane index per live position (contract-grouped assignment).
      llvm::SmallVector<unsigned, 8> liveLanes;
    };
    llvm::SmallVector<Continuation, 8> continuations;
    llvm::DenseSet<mlir::Block *> continuationBlocks;
    llvm::DenseMap<std::pair<mlir::Block *, mlir::Value>, mlir::BlockArgument>
        blockValueArguments;
    for (auto [index, yield] : llvm::enumerate(yields)) {
      std::int64_t state = static_cast<std::int64_t>(index) + 1;
      mlir::Block *block = yield->getBlock();
      llvm::SetVector<mlir::Value> lives =
          liveAfterYield(yield, liveIns, &entryBlock);
      if (lives.size() > frameWidth)
        return yield.emitError()
               << "generator resume live set exceeds the computed frame";
      llvm::SmallVector<unsigned, 8> liveLanes;
      llvm::SmallVector<mlir::Value, 8> laneValues;
      if (mlir::failed(
              assignFrameLanes(lives.getArrayRef(), liveLanes, laneValues)))
        return mlir::failure();
      mlir::Block *cont = block->splitBlock(
          std::next(mlir::Block::iterator(yield.getOperation())));
      // A suspension point inside a (flattened) try keeps its handler scope:
      // the continuation inherits the block's handler id so the injected
      // rethrow and the continuation's calls unwind into the body handlers.
      {
        auto handlerIt = tryHandlerIds.find(block);
        if (handlerIt != tryHandlerIds.end())
          tryHandlerIds.try_emplace(cont, handlerIt->second);
      }
      llvm::SmallVector<mlir::Value, 8> liveValues(lives.begin(), lives.end());
      for (mlir::Value live : liveValues) {
        mlir::BlockArgument arg = cont->addArgument(live.getType(), loc);
        blockValueArguments[{cont, live}] = arg;
        live.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
          return use.getOwner()->getBlock() == cont;
        });
        // Register with the logical block-argument machinery so the standard
        // post-lowering passes drop the logical operands/arguments (the raw
        // evidence pairs are the runtime ABI).
        if (controlFlowLogicalBlockArgumentSet.insert(arg).second)
          controlFlowLogicalBlockArguments.push_back(
              ControlFlowLogicalBlockArgumentABI{arg});
      }
      // The resumed continuation reads the CURRENT resume's sent lane: the
      // alias op moved into the continuation with the split; wire it to the
      // entry sent argument (visible everywhere — entry dominates).
      auto aliasIt = sentAliases.find(yield.getOperation());
      if (aliasIt != sentAliases.end())
        aliasIt->second->setOperand(0, entryBlock.getArgument(sentIndex));
      // The suspend: return (state, 1, value, 0, 0, frame lanes...). Frame
      // operands ride LANE order (contract-grouped); lanes with no live
      // value in this state cross placeholders. External live operands are
      // rethreaded by the normalization pass below.
      {
        mlir::OpBuilder b = mlir::OpBuilder::atBlockEnd(block);
        llvm::SmallVector<mlir::Value, 8> operands;
        operands.push_back(intLiteral(b, state));
        operands.push_back(intLiteral(b, 1));
        operands.push_back(yield.getValue());
        operands.push_back(intLiteral(b, 0));
        operands.push_back(intLiteral(b, 0));
        for (unsigned slot = 0; slot < frameWidth; ++slot)
          operands.push_back(laneValues[slot]
                                 ? laneValues[slot]
                                 : framePlaceholderOperand(b, slot));
        mlir::func::ReturnOp::create(b, loc, operands);
      }
      // Continuation entry guards, in order: (1) an injected exception
      // (throw/close) rethrows the TLS current exception at the suspension
      // point, inside the body's handler scope; (2) resuming a sent-value
      // continuation via next() would materialize None in an int slot, so
      // it is rejected at this boundary (CPython raises TypeError when the
      // None reaches the int operation; there is no None-in-int runtime
      // representation here, so the boundary is the earliest sound point).
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(cont);
        mlir::Value zero =
            mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
        mlir::Value injected = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::ne, rawOf(injectIndex),
            zero);
        auto injectIf = mlir::scf::IfOp::create(builder, loc, injected,
                                                /*withElseRegion=*/false);
        builder.setInsertionPoint(
            injectIf.getThenRegion().front().getTerminator());
        emitTryCallSiteMarkerIfNeeded(loc);
        mlir::func::CallOp::create(builder, loc,
                                   getOrCreateRethrowCurrent(module, builder),
                                   mlir::ValueRange{});
        builder.setInsertionPointAfter(injectIf);
        if (aliasIt != sentAliases.end()) {
          mlir::Value trueValue =
              mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
          mlir::Value sentMissing = mlir::arith::XOrIOp::create(
              builder, loc, validOf(sentIndex), trueValue);
          auto sentIf = mlir::scf::IfOp::create(builder, loc, sentMissing,
                                                /*withElseRegion=*/false);
          builder.setInsertionPoint(
              sentIf.getThenRegion().front().getTerminator());
          if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
                  clone.getOperation(), "builtins.TypeError",
                  "generator expects a builtins.int sent value; resume it "
                  "with send(), not next()")))
            return mlir::failure();
        }
      }
      yield.erase();
      continuationBlocks.insert(cont);
      continuations.push_back(Continuation{
          cont, static_cast<unsigned>(liveValues.size()), liveLanes});
    }

    // Detach the original first block from the entry so the entry can become
    // the dispatch. Entry-defined ops move with it; only entry ARGUMENTS
    // remain visible everywhere.
    mlir::Block *originalStart =
        entryBlock.splitBlock(entryBlock.begin());

    // Normalization: resume dispatch adds CFG edges into the middle of the
    // body, so plain dominance no longer holds. Make every cross-block value
    // an explicit block argument (iterated to fixpoint); afterwards each
    // block only reads its own arguments, its own ops, and entry arguments.
    // The liveness-derived continuation arguments are closed under this
    // rewrite, so continuations never grow here.
    bool normalizing = true;
    while (normalizing) {
      normalizing = false;
      for (mlir::Block &blockRef : clone.getBody()) {
        mlir::Block *block = &blockRef;
        if (block == &entryBlock)
          continue;
        llvm::SetVector<mlir::Value> external;
        for (mlir::Operation &op : *block)
          for (mlir::Value operand : op.getOperands()) {
            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
              if (argument.getOwner() == block ||
                  argument.getOwner() == &entryBlock)
                continue;
            } else if (operand.getDefiningOp()->getBlock() == block) {
              continue;
            }
            external.insert(operand);
          }
        if (external.empty())
          continue;
        if (continuationBlocks.contains(block))
          return clone.emitError()
                 << "generator resume continuation live closure violated";
        normalizing = true;
        for (mlir::Value value : external) {
          mlir::BlockArgument arg = block->addArgument(value.getType(), loc);
          blockValueArguments[{block, value}] = arg;
          value.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
            return use.getOwner()->getBlock() == block;
          });
          for (mlir::Block *pred : block->getPredecessors()) {
            mlir::BlockArgument threaded =
                blockValueArguments.lookup({pred, value});
            mlir::Value passed = threaded ? mlir::Value(threaded) : value;
            mlir::Operation *terminator = pred->getTerminator();
            if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
              branch.getDestOperandsMutable().append(passed);
            } else if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(
                           terminator)) {
              if (cond.getTrueDest() == block)
                cond.getTrueDestOperandsMutable().append(passed);
              if (cond.getFalseDest() == block)
                cond.getFalseDestOperandsMutable().append(passed);
            } else {
              return terminator->emitError()
                     << "unsupported terminator in generator resume "
                        "normalization";
            }
          }
        }
      }
    }

    // Physical spans for the logical continuation arguments, and bundles so
    // the py ops in the continuations lower on them. Each lane's incoming
    // ownership is re-rooted at a claim call (identity + owned-results
    // contract): continuation block arguments alone are invisible to the
    // refcount inserter and the affine verifier, so without the claim the
    // frame's token would silently leak on paths where the value dies
    // before the next suspension.
    for (Continuation &continuation : continuations) {
      mlir::Block *cont = continuation.block;
      llvm::SmallVector<llvm::SmallVector<mlir::Value, 6>, 8> lanePhysicals;
      for (unsigned position = 0; position < continuation.liveCount;
           ++position) {
        const GeneratorResumeLane &lane =
            info.frameLanes[continuation.liveLanes[position]];
        llvm::SmallVector<mlir::Type, 6> laneTypes =
            RuntimeBundleLowerer::generatorLanePhysicalTypes(lane);
        llvm::SmallVector<mlir::Value, 6> physicals;
        for (mlir::Type laneType : laneTypes)
          physicals.push_back(cont->addArgument(laneType, loc));
        lanePhysicals.push_back(std::move(physicals));
      }
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(cont);
      for (unsigned position = 0; position < continuation.liveCount;
           ++position) {
        const GeneratorResumeLane &lane =
            info.frameLanes[continuation.liveLanes[position]];
        llvm::ArrayRef<mlir::Value> physicals = lanePhysicals[position];
        RuntimeBundle bundle;
        if (lane.physicalCount == 0) {
          if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
                  clone.getOperation(),
                  runtimeContractType(context, lane.contract), physicals,
                  bundle)))
            return mlir::failure();
        } else {
          mlir::FailureOr<mlir::func::FuncOp> claim =
              RuntimeBundleLowerer::getOrCreateGeneratorClaimFunction(
                  clone.getOperation(), lane);
          if (mlir::failed(claim))
            return mlir::failure();
          mlir::func::CallOp claimed =
              mlir::func::CallOp::create(builder, loc, *claim, physicals);
          llvm::SmallVector<mlir::Value, 6> owned(
              claimed.getResults().begin(), claimed.getResults().end());
          llvm::ArrayRef<mlir::Value> parts(owned);
          if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
                  clone.getOperation(),
                  runtimeContractType(context, lane.contract),
                  parts.take_front(lane.physicalCount), bundle,
                  /*ownsObject=*/true)))
            return mlir::failure();
          if (lane.isInt)
            bundle.primitiveI64 = RuntimePrimitiveI64Evidence{
                owned[lane.physicalCount], owned[lane.physicalCount + 1]};
        }
        valueBundles[cont->getArgument(position)] = std::move(bundle);
      }
    }

    // Entry dispatch: chain raw-state comparisons in the (now empty) entry.
    // State 0 falls through to the original body.
    mlir::OpBuilder b = mlir::OpBuilder::atBlockEnd(&entryBlock);
    mlir::Block *current = &entryBlock;
    for (auto [index, continuation] : llvm::enumerate(continuations)) {
      std::int64_t state = static_cast<std::int64_t>(index) + 1;
      mlir::Block *next = clone.addBlock();
      b.setInsertionPointToEnd(current);
      mlir::Value expected =
          mlir::arith::ConstantIntOp::create(b, loc, state, 64);
      mlir::Value matches = mlir::arith::CmpIOp::create(
          b, loc, mlir::arith::CmpIPredicate::eq, rawState, expected);
      llvm::SmallVector<mlir::Value, 8> destOperands;
      for (unsigned position = 0; position < continuation.liveCount;
           ++position)
        destOperands.push_back(entryBlock.getArgument(
            controlCount + continuation.liveLanes[position]));
      for (unsigned position = 0; position < continuation.liveCount;
           ++position) {
        unsigned logicalIndex =
            controlCount + continuation.liveLanes[position];
        for (unsigned part = 0; part < laneWidths[logicalIndex]; ++part)
          destOperands.push_back(
              entryBlock.getArgument(laneOffsets[logicalIndex] + part));
      }
      mlir::cf::CondBranchOp::create(b, loc, matches, continuation.block,
                                     destOperands, next, mlir::ValueRange{});
      current = next;
    }
    // Default: state 0 → run from the start; anything else is exhausted.
    {
      b.setInsertionPointToEnd(current);
      mlir::Value zero = mlir::arith::ConstantIntOp::create(b, loc, 0, 64);
      mlir::Value fresh = mlir::arith::CmpIOp::create(
          b, loc, mlir::arith::CmpIPredicate::eq, rawState, zero);
      mlir::Block *exhausted = clone.addBlock();
      mlir::cf::CondBranchOp::create(b, loc, fresh, originalStart,
                                     mlir::ValueRange{}, exhausted,
                                     mlir::ValueRange{});
      b.setInsertionPointToEnd(exhausted);
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.push_back(intLiteral(b, doneState));
      for (unsigned lane = 1; lane < kResumeResultLanes; ++lane)
        operands.push_back(lane == kResumeValueLaneIndex
                               ? deadValueLaneOperand(b)
                               : intLiteral(b, 0));
      for (unsigned slot = 0; slot < frameWidth; ++slot)
        operands.push_back(framePlaceholderOperand(b, slot));
      mlir::func::ReturnOp::create(b, loc, operands);
    }
  }
  return mlir::success();
}

// step: resume the generator once. Loads state + live slots from the
// generator object, calls the resume clone through the primitive ABI, stores
// them back, and keeps the lifecycle word (word 2) coherent. When the body
// escapes with an exception the generator is marked exhausted first
// (resuming after a raise must report exhaustion, not re-run the body), and
// an escaping StopIteration becomes RuntimeError (PEP 479).
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.stepName.empty()) {
    if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(info.stepName))
      return existing;
  }
  mlir::func::FuncOp clone =
      module.lookupSymbol<mlir::func::FuncOp>(info.cloneName);
  if (!clone)
    return op->emitError() << "generator resume clone " << info.cloneName
                           << " is not defined";
  mlir::Location loc = clone.getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();

  llvm::SmallVector<mlir::Type, 16> inputs;
  inputs.push_back(generatorStorageType(builder));
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    inputs.push_back(i64);
    inputs.push_back(i1);
  }
  inputs.push_back(i64); // sent
  inputs.push_back(i1);  // sent valid
  inputs.push_back(i64); // inject
  llvm::SmallVector<mlir::Type, 6> valueLaneTypes =
      RuntimeBundleLowerer::generatorLanePhysicalTypes(info.valueLane);
  unsigned valueLaneWidth = static_cast<unsigned>(valueLaneTypes.size());
  llvm::SmallVector<mlir::Type, 8> results;
  results.push_back(i1); // has
  results.append(valueLaneTypes.begin(), valueLaneTypes.end());
  results.push_back(i64); // ret
  results.push_back(i1);  // hasret

  std::string name = info.cloneName + "__step";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, results));
  function.setPrivate();
  // The yielded value crosses the driver as an owned object span (the clone
  // transferred it through its own owned-results contract).
  if (!info.valueLane.isControl() && !info.valueLane.isNone) {
    function->setAttr(ownership::kOwnedResultsAttr,
                      mlir::DenseI64ArrayAttr::get(
                          context, llvm::ArrayRef<std::int64_t>{1}));
    function->setAttr(
        ownership::kOwnedResultContractsAttr,
        builder.getArrayAttr({builder.getStringAttr(info.valueLane.contract)}));
  }
  info.stepName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  mlir::Value generator = entry->getArgument(0);
  unsigned sentValueIndex = 1 + 2 * info.argumentCount;
  builder.setInsertionPointToStart(entry);
  auto slotIndex = [&](std::int64_t slot) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, slot)
        .getResult();
  };
  auto i64Const = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };

  mlir::Value lifecycle =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(2))
          .getResult();
  mlir::Value state =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(4))
          .getResult();
  // Lifecycle gate (word 2: 3 = exhausted, 4 = closed): a finished generator
  // must not run body code. Feeding the clone an out-of-range state routes
  // it to the pure exhausted return, so no branch is needed.
  mlir::Value finished = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, lifecycle, i64Const(3));
  state = mlir::arith::SelectOp::create(builder, loc, finished, i64Const(-1),
                                        state)
              .getResult();
  // send(non-None) before the first resume never reaches body code in
  // CPython either; reject it here where the state is known.
  mlir::Value created = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(0));
  mlir::Value sentValid = entry->getArgument(sentValueIndex + 1);
  mlir::Value inject = entry->getArgument(sentValueIndex + 2);
  mlir::Value noInject = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, inject, i64Const(0));
  mlir::Value freshSend =
      mlir::arith::AndIOp::create(builder, loc, created, sentValid);
  freshSend = mlir::arith::AndIOp::create(builder, loc, freshSend, noInject);
  {
    auto freshIf = mlir::scf::IfOp::create(builder, loc, freshSend,
                                           /*withElseRegion=*/false);
    mlir::OpBuilder::InsertionGuard freshGuard(builder);
    builder.setInsertionPoint(freshIf.getThenRegion().front().getTerminator());
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            clone.getOperation(), "builtins.TypeError",
            "can't send non-None value to a just-started generator")))
      return mlir::failure();
  }

  std::int64_t handlerId = nextTryHandlerId++;
  mlir::Block *callBlock = builder.createBlock(&body);
  mlir::Block *catchBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(entry);
  auto anchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(handlerId)});
  mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0), catchBlock,
                                 mlir::ValueRange{}, callBlock,
                                 mlir::ValueRange{});

  // Normal resume.
  builder.setInsertionPointToEnd(callBlock);
  mlir::Value trueValue =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
  llvm::SmallVector<mlir::Value, 24> operands;
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    operands.push_back(entry->getArgument(1 + 2 * index));
    operands.push_back(entry->getArgument(2 + 2 * index));
  }
  operands.push_back(state);
  operands.push_back(trueValue);
  operands.push_back(entry->getArgument(sentValueIndex));
  operands.push_back(sentValid);
  operands.push_back(inject);
  operands.push_back(trueValue);
  // Frame lanes: load each lane's span out of the generator storage. The
  // load transfers the frame's token into the clone (and zeroes the slot),
  // so ownership lives in exactly one place while the body runs.
  llvm::SmallVector<unsigned, 8> laneWordOffsets;
  {
    unsigned frameWord = kGeneratorFrameSlotBase;
    for (const GeneratorResumeLane &lane : info.frameLanes) {
      laneWordOffsets.push_back(frameWord);
      frameWord += RuntimeBundleLowerer::generatorLaneFrameWords(lane);
    }
  }
  llvm::SmallVector<unsigned, 8> laneResultWidths;
  for (const GeneratorResumeLane &lane : info.frameLanes) {
    laneResultWidths.push_back(static_cast<unsigned>(
        RuntimeBundleLowerer::generatorLanePhysicalTypes(lane).size()));
  }
  for (auto [slot, lane] : llvm::enumerate(info.frameLanes)) {
    mlir::FailureOr<mlir::func::FuncOp> load =
        RuntimeBundleLowerer::getOrCreateGeneratorFrameLoadFunction(op, lane);
    if (mlir::failed(load))
      return mlir::failure();
    mlir::func::CallOp loaded = mlir::func::CallOp::create(
        builder, loc, *load,
        mlir::ValueRange{generator, i64Const(laneWordOffsets[slot])});
    operands.append(loaded.getResults().begin(), loaded.getResults().end());
  }
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::func::CallOp call =
      mlir::func::CallOp::create(builder, loc, clone, operands);
  // Clone result layout: control lanes are (i64, i1) pairs; the value lane
  // (logical result 2) and the frame lanes are their physical spans.
  unsigned frameResultWidth = 0;
  for (unsigned width : laneResultWidths)
    frameResultWidth += width;
  unsigned expectedResults =
      2 * (kResumeResultLanes - 1) + valueLaneWidth + frameResultWidth;
  if (call.getNumResults() != expectedResults)
    return op->emitError() << "generator resume clone ABI mismatch";
  unsigned valueSpanBegin = 4;
  unsigned retRawIndex = valueSpanBegin + valueLaneWidth;
  unsigned hasretRawIndex = retRawIndex + 2;
  unsigned frameBaseIndex = hasretRawIndex + 2;
  mlir::Value newState = call.getResult(0);
  mlir::Value hasRaw = call.getResult(2);
  llvm::SmallVector<mlir::Value, 6> valueSpan;
  for (unsigned lane = 0; lane < valueLaneWidth; ++lane)
    valueSpan.push_back(call.getResult(valueSpanBegin + lane));
  mlir::Value returnedValue = call.getResult(retRawIndex);
  mlir::Value returnedRaw = call.getResult(hasretRawIndex);
  mlir::memref::StoreOp::create(builder, loc, newState, generator,
                                slotIndex(4));
  // The frame absorbs each lane's token (rfc/stdlib-semantics.md R3): the
  // store helper consumes the span into the generator storage words.
  {
    unsigned resultOffset = frameBaseIndex;
    for (auto [slot, lane] : llvm::enumerate(info.frameLanes)) {
      mlir::FailureOr<mlir::func::FuncOp> store =
          RuntimeBundleLowerer::getOrCreateGeneratorFrameStoreFunction(op,
                                                                       lane);
      if (mlir::failed(store))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 10> storeOperands{
          generator, i64Const(laneWordOffsets[slot])};
      for (unsigned part = 0; part < laneResultWidths[slot]; ++part)
        storeOperands.push_back(call.getResult(resultOffset + part));
      mlir::func::CallOp::create(builder, loc, *store, storeOperands);
      resultOffset += laneResultWidths[slot];
    }
  }
  mlir::Value hasValue = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, hasRaw, i64Const(0));
  mlir::Value hasReturned = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, returnedRaw, i64Const(0));
  // Suspended while values flow, exhausted when the body finishes; a
  // finished generator keeps its previous (exhausted/closed) state.
  mlir::Value doneLifecycle =
      mlir::arith::SelectOp::create(builder, loc, finished, lifecycle,
                                    i64Const(3))
          .getResult();
  mlir::Value nextLifecycle =
      mlir::arith::SelectOp::create(builder, loc, hasValue, i64Const(2),
                                    doneLifecycle)
          .getResult();
  mlir::memref::StoreOp::create(builder, loc, nextLifecycle, generator,
                                slotIndex(2));
  llvm::SmallVector<mlir::Value, 8> stepResults;
  stepResults.push_back(hasValue);
  stepResults.append(valueSpan.begin(), valueSpan.end());
  stepResults.push_back(returnedValue);
  stepResults.push_back(hasReturned);
  mlir::func::ReturnOp::create(builder, loc, stepResults);

  // Body escaped with an exception: the generator is finished for good, and
  // PEP 479 turns an escaping StopIteration into RuntimeError. Everything
  // else propagates to the resumer.
  builder.setInsertionPointToEnd(catchBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::memref::StoreOp::create(builder, loc, i64Const(-1), generator,
                                slotIndex(4));
  mlir::memref::StoreOp::create(builder, loc, i64Const(3), generator,
                                slotIndex(2));
  auto currentMatches = getOrCreatePrivateFunction(
      module, builder, "LyEH_CurrentExceptionMatches",
      builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()}));
  auto isStopIteration = mlir::func::CallOp::create(
      builder, loc, currentMatches,
      mlir::ValueRange{i64Const(exceptionClassId("StopIteration"))});
  mlir::Block *pep479Block = builder.createBlock(&body);
  mlir::Block *rethrowBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(catchBlock);
  mlir::cf::CondBranchOp::create(builder, loc, isStopIteration.getResult(0),
                                 pep479Block, mlir::ValueRange{}, rethrowBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(pep479Block);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          clone.getOperation(), "builtins.RuntimeError",
          "generator raised StopIteration")))
    return mlir::failure();
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(pep479Block);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::func::CallOp::create(builder, loc,
                             getOrCreateRethrowCurrent(module, builder),
                             mlir::ValueRange{});
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  return function;
}

// advance: step + the next()/send() exhaustion protocol — StopIteration
// carrying str(return value) as its message (CPython's str(StopIteration(v))
// is str(v); the typed .value attribute is not represented yet).
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorAdvanceFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.advanceName.empty()) {
    if (auto existing =
            module.lookupSymbol<mlir::func::FuncOp>(info.advanceName))
      return existing;
  }
  mlir::FailureOr<mlir::func::FuncOp> step =
      RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(op, info);
  if (mlir::failed(step))
    return mlir::failure();
  mlir::Location loc = step->getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);

  llvm::SmallVector<mlir::Type, 6> valueLaneTypes =
      RuntimeBundleLowerer::generatorLanePhysicalTypes(info.valueLane);
  unsigned valueLaneWidth = static_cast<unsigned>(valueLaneTypes.size());
  std::string name = info.cloneName + "__advance";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name,
      builder.getFunctionType(step->getFunctionType().getInputs(),
                              valueLaneTypes));
  function.setPrivate();
  if (!info.valueLane.isControl() && !info.valueLane.isNone) {
    function->setAttr(ownership::kOwnedResultsAttr,
                      mlir::DenseI64ArrayAttr::get(
                          context, llvm::ArrayRef<std::int64_t>{0}));
    function->setAttr(
        ownership::kOwnedResultContractsAttr,
        builder.getArrayAttr({builder.getStringAttr(info.valueLane.contract)}));
  }
  info.advanceName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  builder.setInsertionPointToStart(entry);
  mlir::func::CallOp call = mlir::func::CallOp::create(
      builder, loc, *step, entry->getArguments());
  unsigned retIndex = 1 + valueLaneWidth;
  unsigned hasretIndex = retIndex + 1;
  mlir::Block *yieldedBlock = builder.createBlock(&body);
  mlir::Block *stoppedBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(entry);
  mlir::cf::CondBranchOp::create(builder, loc, call.getResult(0), yieldedBlock,
                                 mlir::ValueRange{}, stoppedBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(yieldedBlock);
  llvm::SmallVector<mlir::Value, 6> yieldedSpan;
  for (unsigned lane = 0; lane < valueLaneWidth; ++lane)
    yieldedSpan.push_back(call.getResult(1 + lane));
  mlir::func::ReturnOp::create(builder, loc, yieldedSpan);

  builder.setInsertionPointToEnd(stoppedBlock);
  // The exhausted path raises; its value span holds immortal dead
  // placeholders, whose release discharges the step call's owned-result
  // contract as a no-op before the raise.
  if (!info.valueLane.isControl() && !info.valueLane.isNone) {
    llvm::SmallVector<mlir::Value, 4> placeholderValues;
    for (unsigned lane = 0; lane < info.valueLane.physicalCount; ++lane)
      placeholderValues.push_back(call.getResult(1 + lane));
    RuntimeBundle placeholder;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, runtimeContractType(context, info.valueLane.contract),
            placeholderValues, placeholder, /*ownsObject=*/true)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, placeholder, "generator exhausted value lane")))
      return mlir::failure();
  }
  mlir::Block *valueBlock = builder.createBlock(&body);
  mlir::Block *plainBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(stoppedBlock);
  mlir::cf::CondBranchOp::create(builder, loc, call.getResult(hasretIndex),
                                 valueBlock, mlir::ValueRange{}, plainBlock,
                                 mlir::ValueRange{});

  // return X → StopIteration whose message is str(X). Exhaustion can be
  // observed while another exception is the pending current one (next()
  // inside an except handler); the single-token TLS slot requires the same
  // discard-before-raise that py.raise lowering performs.
  builder.setInsertionPointToEnd(valueBlock);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  {
    std::optional<RuntimeSymbol> intNew =
        manifest.initializer("builtins.int", "__new__");
    std::optional<RuntimeSymbol> intStr =
        manifest.method("builtins.int", "__str__");
    if (!intNew || !intStr)
      return op->emitError() << "runtime manifest cannot render the generator "
                                "return value (builtins.int __new__/__str__)";
    mlir::func::CallOp boxed = RuntimeBundleLowerer::createRuntimeCall(
        loc, *intNew, mlir::ValueRange{call.getResult(retIndex)});
    mlir::func::CallOp text = RuntimeBundleLowerer::createRuntimeCall(
        loc, *intStr, boxed.getResults());
    // The boxed int is only a rendering temporary; drop it through the
    // manifest deallocator rather than teaching this synthesized body the
    // elision machinery.
    RuntimeBundle boxedBundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, runtimeContractType(context, "builtins.int"),
            boxed.getResults(), boxedBundle)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, boxedBundle, "generator return value rendering")))
      return mlir::failure();
    RuntimeBundle message;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, runtimeContractType(context, "builtins.str"),
            text.getResults(), message)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeExceptionFromMessageObject(
            op, "builtins.StopIteration", message)))
      return mlir::failure();
  }
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(valueBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);

  builder.setInsertionPointToEnd(plainBlock);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, "builtins.StopIteration", "")))
    return mlir::failure();
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(plainBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  return function;
}

// throw: stage the exception in the EH TLS slot (raise + immediate catch),
// then resume with inject=1 so the continuation rethrows it at the
// suspension point, inside the body's handler scope. A created/finished
// generator never runs body code: the exception propagates directly (and a
// never-started generator closes).
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorThrowFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.throwName.empty()) {
    if (auto existing =
            module.lookupSymbol<mlir::func::FuncOp>(info.throwName))
      return existing;
  }
  mlir::FailureOr<mlir::func::FuncOp> advance =
      RuntimeBundleLowerer::getOrCreateGeneratorAdvanceFunction(op, info);
  if (mlir::failed(advance))
    return mlir::failure();
  mlir::Location loc = advance->getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();
  auto headerType = mlir::MemRefType::get({3}, i64);
  auto messageType = mlir::MemRefType::get({2}, i64);
  auto bytesType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI8Type());

  llvm::SmallVector<mlir::Type, 16> inputs;
  inputs.push_back(generatorStorageType(builder));
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    inputs.push_back(i64);
    inputs.push_back(i1);
  }
  unsigned headerIndex = 1 + 2 * info.argumentCount;
  inputs.push_back(headerType);
  inputs.push_back(messageType);
  inputs.push_back(bytesType);

  llvm::SmallVector<mlir::Type, 6> valueLaneTypes =
      RuntimeBundleLowerer::generatorLanePhysicalTypes(info.valueLane);
  std::string name = info.cloneName + "__throw";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, valueLaneTypes));
  function.setPrivate();
  if (!info.valueLane.isControl() && !info.valueLane.isNone) {
    function->setAttr(ownership::kOwnedResultsAttr,
                      mlir::DenseI64ArrayAttr::get(
                          context, llvm::ArrayRef<std::int64_t>{0}));
    function->setAttr(
        ownership::kOwnedResultContractsAttr,
        builder.getArrayAttr({builder.getStringAttr(info.valueLane.contract)}));
  }
  // The caller hands the exception over for good — like the *_Raise
  // primitives (the TLS slot takes it).
  function->setAttr(
      "ly.ownership.transfer_args",
      builder.getI64ArrayAttr({static_cast<std::int64_t>(headerIndex),
                               static_cast<std::int64_t>(headerIndex + 1)}));
  function.setArgAttr(headerIndex, "ly.ownership.object_header",
                      builder.getUnitAttr());
  function.setArgAttr(headerIndex + 1, "ly.ownership.object_header",
                      builder.getUnitAttr());
  info.throwName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  builder.setInsertionPointToStart(entry);
  auto i64Const = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };
  auto throwException = getOrCreatePrivateFunction(
      module, builder, "LyEH_ThrowException",
      builder.getFunctionType({headerType, messageType, bytesType}, {}));

  std::int64_t handlerId = nextTryHandlerId++;
  mlir::Block *raiseBlock = builder.createBlock(&body);
  mlir::Block *caughtBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(entry);
  auto anchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(handlerId)});
  mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0),
                                 caughtBlock, mlir::ValueRange{}, raiseBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::func::CallOp::create(
      builder, loc, throwException,
      mlir::ValueRange{entry->getArgument(headerIndex),
                       entry->getArgument(headerIndex + 1),
                       entry->getArgument(headerIndex + 2)});
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);

  builder.setInsertionPointToEnd(caughtBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::Value generator = entry->getArgument(0);
  auto slotIndex = [&](std::int64_t slot) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, slot)
        .getResult();
  };
  mlir::Value lifecycle =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(2))
          .getResult();
  mlir::Value suspended = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(2));
  mlir::Block *injectBlock = builder.createBlock(&body);
  mlir::Block *directBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(caughtBlock);
  mlir::cf::CondBranchOp::create(builder, loc, suspended, injectBlock,
                                 mlir::ValueRange{}, directBlock,
                                 mlir::ValueRange{});

  builder.setInsertionPointToEnd(injectBlock);
  llvm::SmallVector<mlir::Value, 16> advanceOperands;
  advanceOperands.push_back(generator);
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    advanceOperands.push_back(entry->getArgument(1 + 2 * index));
    advanceOperands.push_back(entry->getArgument(2 + 2 * index));
  }
  advanceOperands.push_back(i64Const(0));
  advanceOperands.push_back(constantI1(builder, loc, false));
  advanceOperands.push_back(i64Const(1)); // inject
  mlir::func::CallOp resumed =
      mlir::func::CallOp::create(builder, loc, *advance, advanceOperands);
  mlir::func::ReturnOp::create(builder, loc, resumed.getResults());

  // Not suspended: the body never sees the exception. CPython closes a
  // just-started generator and raises the thrown exception as-is.
  builder.setInsertionPointToEnd(directBlock);
  mlir::Value wasCreated = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(0));
  auto closeIf = mlir::scf::IfOp::create(builder, loc, wasCreated,
                                         /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard closeGuard(builder);
    builder.setInsertionPoint(closeIf.getThenRegion().front().getTerminator());
    mlir::memref::StoreOp::create(builder, loc, i64Const(4), generator,
                                  slotIndex(2));
  }
  mlir::func::CallOp::create(builder, loc,
                             getOrCreateRethrowCurrent(module, builder),
                             mlir::ValueRange{});
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(directBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  return function;
}

// close: PEP 342 close(). GeneratorExit is injected at the suspension point
// (so finally blocks run); the generator swallowing it (returning) or the
// exception coming back out both close the generator, while yielding again
// is the "generator ignored GeneratorExit" RuntimeError.
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorCloseFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.closeName.empty()) {
    if (auto existing =
            module.lookupSymbol<mlir::func::FuncOp>(info.closeName))
      return existing;
  }
  mlir::FailureOr<mlir::func::FuncOp> step =
      RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(op, info);
  if (mlir::failed(step))
    return mlir::failure();
  std::optional<RuntimeSymbol> exitNew =
      manifest.initializer("builtins.GeneratorExit", "__new__");
  std::optional<RuntimeSymbol> exitRaise =
      manifest.primitive("builtins.GeneratorExit", "raise");
  if (!exitNew || !exitRaise)
    return op->emitError()
           << "runtime manifest has no builtins.GeneratorExit support";
  mlir::Location loc = step->getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();

  llvm::SmallVector<mlir::Type, 16> inputs;
  inputs.push_back(generatorStorageType(builder));
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    inputs.push_back(i64);
    inputs.push_back(i1);
  }
  std::string name = info.cloneName + "__close";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, {}));
  function.setPrivate();
  info.closeName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  mlir::Value generator = entry->getArgument(0);
  builder.setInsertionPointToStart(entry);
  auto i64Const = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };
  auto slotIndex = [&](std::int64_t slot) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, slot)
        .getResult();
  };

  mlir::Block *checkFinished = builder.createBlock(&body);
  mlir::Block *injectBlock = builder.createBlock(&body);
  mlir::Block *markClosed = builder.createBlock(&body);
  mlir::Block *returnBlock = builder.createBlock(&body);

  builder.setInsertionPointToEnd(entry);
  mlir::Value lifecycle =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(2))
          .getResult();
  mlir::Value created = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(0));
  mlir::cf::CondBranchOp::create(builder, loc, created, markClosed,
                                 mlir::ValueRange{}, checkFinished,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(checkFinished);
  mlir::Value finished = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, lifecycle, i64Const(3));
  mlir::cf::CondBranchOp::create(builder, loc, finished, returnBlock,
                                 mlir::ValueRange{}, injectBlock,
                                 mlir::ValueRange{});

  // Stage GeneratorExit in the TLS slot (raise + immediate catch), resume
  // with inject=1, and interpret the outcome.
  builder.setInsertionPointToEnd(injectBlock);
  std::int64_t stageId = nextTryHandlerId++;
  std::int64_t resumeId = nextTryHandlerId++;
  mlir::Block *raiseBlock = builder.createBlock(&body);
  mlir::Block *stagedBlock = builder.createBlock(&body);
  mlir::Block *resumeBlock = builder.createBlock(&body);
  mlir::Block *swallowBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(injectBlock);
  auto anchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(stageId)});
  mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0),
                                 stagedBlock, mlir::ValueRange{}, raiseBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::func::CallOp exitObject = RuntimeBundleLowerer::createRuntimeCall(
      loc, *exitNew,
      mlir::ValueRange{i64Const(exceptionClassId("GeneratorExit"))});
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(stageId)});
  RuntimeBundleLowerer::createRuntimeCall(loc, *exitRaise,
                                          exitObject.getResults());
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);

  builder.setInsertionPointToEnd(stagedBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(stageId)});
  auto resumeAnchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(resumeId)});
  mlir::cf::CondBranchOp::create(builder, loc, resumeAnchor.getResult(0),
                                 swallowBlock, mlir::ValueRange{}, resumeBlock,
                                 mlir::ValueRange{});

  builder.setInsertionPointToEnd(resumeBlock);
  llvm::SmallVector<mlir::Value, 16> stepOperands;
  stepOperands.push_back(generator);
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    stepOperands.push_back(entry->getArgument(1 + 2 * index));
    stepOperands.push_back(entry->getArgument(2 + 2 * index));
  }
  stepOperands.push_back(i64Const(0));
  stepOperands.push_back(constantI1(builder, loc, false));
  stepOperands.push_back(i64Const(1)); // inject
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(resumeId)});
  mlir::func::CallOp resumed =
      mlir::func::CallOp::create(builder, loc, *step, stepOperands);
  // Discharge the step call's owned value lane on both outcomes: immortal
  // dead placeholders when the body finished (release is a no-op), the
  // ignored yielded value when the body kept yielding.
  if (!info.valueLane.isControl() && !info.valueLane.isNone) {
    llvm::SmallVector<mlir::Value, 4> laneValues;
    for (unsigned lane = 0; lane < info.valueLane.physicalCount; ++lane)
      laneValues.push_back(resumed.getResult(1 + lane));
    RuntimeBundle laneBundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, runtimeContractType(context, info.valueLane.contract),
            laneValues, laneBundle, /*ownsObject=*/true)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, laneBundle, "generator close ignored value lane")))
      return mlir::failure();
  }
  auto ignoredIf = mlir::scf::IfOp::create(builder, loc, resumed.getResult(0),
                                           /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard ignoredGuard(builder);
    builder.setInsertionPoint(
        ignoredIf.getThenRegion().front().getTerminator());
    // The staged GeneratorExit may still be the pending current exception
    // (the body can be suspended inside its own handler); the EH TLS slot is
    // single-token, so raising without discarding would trap.
    mlir::func::CallOp::create(
        builder, loc, getOrCreateDiscardCurrentException(module, builder),
        mlir::ValueRange{});
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            op, "builtins.RuntimeError", "generator ignored GeneratorExit")))
      return mlir::failure();
  }
  mlir::cf::BranchOp::create(builder, loc, markClosed);

  // The injected GeneratorExit came back out: that is the normal close path.
  // Anything else (body exceptions, PEP 479 RuntimeError) propagates.
  builder.setInsertionPointToEnd(swallowBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(resumeId)});
  auto currentMatches = getOrCreatePrivateFunction(
      module, builder, "LyEH_CurrentExceptionMatches",
      builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()}));
  auto isExit = mlir::func::CallOp::create(
      builder, loc, currentMatches,
      mlir::ValueRange{i64Const(exceptionClassId("GeneratorExit"))});
  mlir::Block *discardBlock = builder.createBlock(&body);
  mlir::Block *rethrowBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(swallowBlock);
  mlir::cf::CondBranchOp::create(builder, loc, isExit.getResult(0),
                                 discardBlock, mlir::ValueRange{},
                                 rethrowBlock, mlir::ValueRange{});
  builder.setInsertionPointToEnd(discardBlock);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  mlir::cf::BranchOp::create(builder, loc, markClosed);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::func::CallOp::create(builder, loc,
                             getOrCreateRethrowCurrent(module, builder),
                             mlir::ValueRange{});
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);

  builder.setInsertionPointToEnd(markClosed);
  mlir::memref::StoreOp::create(builder, loc, i64Const(4), generator,
                                slotIndex(2));
  mlir::cf::BranchOp::create(builder, loc, returnBlock);
  builder.setInsertionPointToEnd(returnBlock);
  mlir::func::ReturnOp::create(builder, loc);
  return function;
}

// Resume over a state-machine generator: collect the generator storage and
// argument lanes and call the synthesized step/advance driver.
mlir::FailureOr<RuntimeBundleLowerer::SourceGeneratorResumeResult>
RuntimeBundleLowerer::emitStateMachineGeneratorResume(
    mlir::Operation *op, const RuntimeBundle &iterator,
    GeneratorResumeInfo &info, bool useCurrentInsertionPoint,
    std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence,
    bool raiseWhenExhausted) {
  if (iterator.physicalValues().empty())
    return op->emitError() << "generator object has no physical storage";
  mlir::Value generator = iterator.physicalValues().front();
  if (iterator.generatorSourceBundles.size() != info.argumentCount)
    return op->emitError()
           << "generator frame source count does not match the resume clone";

  mlir::FailureOr<mlir::func::FuncOp> driver =
      raiseWhenExhausted
          ? RuntimeBundleLowerer::getOrCreateGeneratorAdvanceFunction(op, info)
          : RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(op, info);
  if (mlir::failed(driver))
    return mlir::failure();

  if (!useCurrentInsertionPoint)
    builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  llvm::SmallVector<mlir::Value, 16> operands;
  operands.push_back(generator);
  for (const std::shared_ptr<RuntimeBundle> &source :
       iterator.generatorSourceBundles) {
    if (!source || !source->primitiveI64)
      return op->emitError() << "state-machine generator frame sources must "
                                "carry primitive int evidence";
    operands.push_back(source->primitiveI64->value);
    operands.push_back(source->primitiveI64->valid);
  }
  if (sentI64Evidence) {
    operands.push_back(sentI64Evidence->value);
    operands.push_back(sentI64Evidence->valid);
  } else {
    operands.push_back(
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult());
    operands.push_back(constantI1(builder, loc, false));
  }
  operands.push_back(
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult());
  emitTryCallSiteMarkerIfNeeded(loc);
  mlir::func::CallOp call =
      mlir::func::CallOp::create(builder, loc, *driver, operands);
  unsigned spanBegin = raiseWhenExhausted ? 0 : 1;
  unsigned spanWidth = static_cast<unsigned>(
      RuntimeBundleLowerer::generatorLanePhysicalTypes(info.valueLane).size());
  SourceGeneratorResumeResult result;
  result.hasValue = raiseWhenExhausted ? constantI1(builder, loc, true)
                                       : call.getResult(0);
  for (unsigned lane = 0; lane < spanWidth; ++lane)
    result.lanePhysicals.push_back(call.getResult(spanBegin + lane));
  if (info.valueLane.isInt && spanWidth >= 2) {
    // The trailing evidence pair doubles as the legacy (value, valid) view
    // for pure-int consumers (yield-from delegation, for-loop rewiring).
    result.value = result.lanePhysicals[spanWidth - 2];
    result.valid = result.lanePhysicals[spanWidth - 1];
  } else {
    result.value =
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
    result.valid = constantI1(builder, loc, false);
  }
  return result;
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStateMachineGeneratorThrow(
    py::CallOp op, const RuntimeBundle &receiver, GeneratorResumeInfo &info,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() != 2 || !sources[1])
    return op.emitError()
           << "generator throw expects exactly one exception value";
  if (op.getNumResults() != 1 ||
      runtimeContractName(op.getResult(0).getType()) != info.valueLane.contract)
    return op.emitError()
           << "generator throw result must match the yield lane contract "
           << info.valueLane.contract;
  const RuntimeBundle &exception = *sources[1];
  if (exception.physicalValues().size() != 3)
    return op.emitError()
           << "generator throw currently requires a builtin exception "
              "instance (header, message, bytes), got "
           << exception.contractName();
  if (receiver.physicalValues().empty())
    return op.emitError() << "generator object has no physical storage";

  mlir::FailureOr<mlir::func::FuncOp> throwFn =
      RuntimeBundleLowerer::getOrCreateGeneratorThrowFunction(
          op.getOperation(), info);
  if (mlir::failed(throwFn))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  if (mlir::failed(emitTracebackFrame(op.getOperation())))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 16> operands;
  operands.push_back(receiver.physicalValues().front());
  if (receiver.generatorSourceBundles.size() != info.argumentCount)
    return op.emitError()
           << "generator frame source count does not match the resume clone";
  for (const std::shared_ptr<RuntimeBundle> &source :
       receiver.generatorSourceBundles) {
    if (!source || !source->primitiveI64)
      return op.emitError() << "state-machine generator frame sources must "
                               "carry primitive int evidence";
    operands.push_back(source->primitiveI64->value);
    operands.push_back(source->primitiveI64->valid);
  }
  for (mlir::Value value : exception.physicalValues())
    operands.push_back(value);
  emitTryCallSiteMarkerIfNeeded(loc);
  mlir::func::CallOp call =
      mlir::func::CallOp::create(builder, loc, *throwFn, operands);

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op.getOperation(), op.getResult(0).getType(), call, result)))
    return mlir::failure();
  result.setObjectLogicalOwnership(/*ownsObject=*/true);
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStateMachineGeneratorClose(
    py::CallOp op, const RuntimeBundle &receiver, GeneratorResumeInfo &info) {
  if (op.getNumResults() != 1)
    return op.emitError() << "generator close expects one (None) result";
  if (receiver.physicalValues().empty())
    return op.emitError() << "generator object has no physical storage";
  mlir::FailureOr<mlir::func::FuncOp> closeFn =
      RuntimeBundleLowerer::getOrCreateGeneratorCloseFunction(op.getOperation(),
                                                              info);
  if (mlir::failed(closeFn))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 16> operands;
  operands.push_back(receiver.physicalValues().front());
  if (receiver.generatorSourceBundles.size() != info.argumentCount)
    return op.emitError()
           << "generator frame source count does not match the resume clone";
  for (const std::shared_ptr<RuntimeBundle> &source :
       receiver.generatorSourceBundles) {
    if (!source || !source->primitiveI64)
      return op.emitError() << "state-machine generator frame sources must "
                               "carry primitive int evidence";
    operands.push_back(source->primitiveI64->value);
    operands.push_back(source->primitiveI64->valid);
  }
  emitTryCallSiteMarkerIfNeeded(loc);
  mlir::func::CallOp::create(builder, loc, *closeFn, operands);
  if (mlir::failed(assignObjectBundle(
          op.getOperation(), op.getResult(0),
          runtimeContractType(context, "types.NoneType"), mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
