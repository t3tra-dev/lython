// This file implements shared optimization helpers specific to the Py dialect
// and its lowered LLVM representation. These optimizations include:
//   - Dead tuple cleanup (removing tuples only used by DecRefOp)
//   - Integer constant hoisting and CSE
//   - Small integer DecRef removal (immortal integers -5 to 256)
//   - String creation CSE (LyUnicode_FromUTF8)
//   - Singleton getter CSE (Ly_GetBuiltinPrint, Ly_GetNone, etc.)
//   - Bool boxing/unboxing elimination (LyBool_FromBool + LyBool_AsBool)
//   - LLVM constant CSE
//   - Dead code elimination for unused LLVM operations

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py::optimizer {

Value stripIdentityCasts(Value value);

bool isCallTo(LLVM::CallOp callOp, llvm::StringRef calleeName) {
  auto callee = callOp.getCallee();
  return callee && *callee == calleeName;
}

LLVM::CallOp getOptionalDecRefUser(Value value) {
  LLVM::CallOp decRefCall = nullptr;
  for (Operation *user : value.getUsers()) {
    auto callUser = dyn_cast<LLVM::CallOp>(user);
    if (!callUser || !isCallTo(callUser, RuntimeSymbols::kDecRef))
      return nullptr;
    if (decRefCall)
      return nullptr;
    decRefCall = callUser;
  }
  return decRefCall;
}

void eraseCallAndOptionalDecRefUsers(LLVM::CallOp callOp) {
  if (!callOp || callOp->getNumResults() != 1)
    return;

  Value result = callOp.getResult();
  LLVM::CallOp decRefCall = getOptionalDecRefUser(result);
  if (decRefCall)
    decRefCall->erase();

  if (result.use_empty())
    callOp->erase();
}

LLVM::LLVMFuncOp getOrCreateRuntimeFunc(ModuleOp module, StringRef name,
                                        Type resultType, ValueRange operands) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;

  OpBuilder moduleBuilder(module.getBodyRegion());
  SmallVector<Type> argTypes;
  argTypes.reserve(operands.size());
  for (Value operand : operands)
    argTypes.push_back(operand.getType());

  auto fnType = LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return moduleBuilder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

Value materializeI64FromLong(ModuleOp module, OpBuilder &builder, Location loc,
                             Value boxedLong) {
  if (auto fromI64Call = boxedLong.getDefiningOp<LLVM::CallOp>()) {
    if (isCallTo(fromI64Call, RuntimeSymbols::kLongFromI64) &&
        fromI64Call.getNumOperands() == 1) {
      return fromI64Call.getOperand(0);
    }
  }

  auto asI64Func =
      getOrCreateRuntimeFunc(module, RuntimeSymbols::kLongAsI64,
                             builder.getI64Type(), ValueRange{boxedLong});
  return builder.create<LLVM::CallOp>(loc, asI64Func, ValueRange{boxedLong})
      .getResult();
}

/// Remove py.incref on !py.class values when it only survives because a
/// tuple-consuming call was rewritten into a direct func.call that borrows
/// the class slot.
bool cleanupRedundantClassIncrefsAfterDirectCalls(ModuleOp module) {
  SmallVector<IncRefOp> toErase;

  module.walk([&](IncRefOp incref) {
    Value object = incref.getObject();
    if (!isa<ClassType>(object.getType()))
      return;

    for (Operation *cursor = incref->getNextNode(); cursor;
         cursor = cursor->getNextNode()) {
      if (isa<CastIdentityOp, func::ConstantOp>(cursor))
        continue;

      auto call = dyn_cast<func::CallOp>(cursor);
      if (!call)
        break;

      bool passesClass = llvm::any_of(call.getOperands(), [&](Value operand) {
        if (operand == object)
          return true;
        if (auto cast = operand.getDefiningOp<CastIdentityOp>())
          return cast.getInput() == object;
        return false;
      });
      if (passesClass)
        toErase.push_back(incref);
      break;
    }
  });

  for (IncRefOp incref : toErase)
    incref.erase();

  return !toErase.empty();
}

void removeUnusedNoneOps(ModuleOp module) {
  SmallVector<NoneOp> toErase;
  module.walk([&](NoneOp op) {
    if (op.getResult().use_empty())
      toErase.push_back(op);
  });
  for (NoneOp op : toErase)
    op.erase();
}

void removeNoneDecrefs(ModuleOp module) {
  SmallVector<DecRefOp> toErase;
  module.walk([&](DecRefOp op) {
    if (isa<NoneType>(op.getObject().getType()))
      toErase.push_back(op);
  });
  for (DecRefOp op : toErase)
    op.erase();
}

void markConsumedAttrSetValues(ModuleOp module) {
  SmallVector<DecRefOp> toErase;

  module.walk([&](AttrSetOp op) {
    Value value = op.getValue();
    if (!isPyType(value.getType()) || isa<NoneType, BoolType>(value.getType()))
      return;

    DecRefOp trailingDrop;
    for (Operation *user : value.getUsers()) {
      if (user == op.getOperation())
        continue;
      auto decref = dyn_cast<DecRefOp>(user);
      if (!decref || trailingDrop)
        return;
      trailingDrop = decref;
    }

    if (!trailingDrop)
      return;
    if (trailingDrop->getBlock() != op->getBlock())
      return;
    if (!op->isBeforeInBlock(trailingDrop))
      return;

    op->setAttr("lython.consume_value", UnitAttr::get(module.getContext()));
    toErase.push_back(trailingDrop);
  });

  for (DecRefOp op : toErase)
    op.erase();
}

void markConsumedListAppendValues(ModuleOp module) {
  SmallVector<DecRefOp> toErase;

  module.walk([&](ListAppendOp op) {
    Value value = op.getValue();
    if (!isPyType(value.getType()) || isa<NoneType, BoolType>(value.getType()))
      return;
    // Only steal ownership that was materialized explicitly for a publication
    // boundary. Borrowed values such as function arguments must keep the
    // append-side retain.
    if (!value.getDefiningOp<PublishOp>())
      return;

    DecRefOp trailingDrop;
    for (Operation *user : value.getUsers()) {
      if (user == op.getOperation())
        continue;
      auto decref = dyn_cast<DecRefOp>(user);
      if (!decref || trailingDrop)
        return;
      trailingDrop = decref;
    }

    if (!trailingDrop)
      return;
    if (trailingDrop->getBlock() != op->getBlock())
      return;
    if (!op->isBeforeInBlock(trailingDrop))
      return;

    op->setAttr("lython.consume_value", UnitAttr::get(module.getContext()));
    toErase.push_back(trailingDrop);
  });

  for (DecRefOp op : toErase)
    op.erase();
}

Value stripIdentityCasts(Value value) {
  while (auto identity = value.getDefiningOp<CastIdentityOp>())
    value = identity.getInput();
  return value;
}

Value stripTransparentPublicationOps(Value value) {
  while (true) {
    if (auto identity = value.getDefiningOp<CastIdentityOp>()) {
      value = identity.getInput();
      continue;
    }
    if (auto publish = value.getDefiningOp<PublishOp>()) {
      value = publish.getInput();
      continue;
    }
    break;
  }
  return value;
}

bool isPublicationSummaryResultType(Type type) {
  return isa<ListType, DictType, ObjectType, ClassType>(type);
}

bool arrayAttrContainsIndex(ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (Attribute element : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    if (!intAttr)
      continue;
    if (intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

FuncOp resolveDirectPyFuncSymbol(Operation *from, Value callable) {
  callable = stripIdentityCasts(callable);

  if (auto funcObject = callable.getDefiningOp<FuncObjectOp>()) {
    Operation *symbol =
        SymbolTable::lookupNearestSymbolFrom(from, funcObject.getTargetAttr());
    return dyn_cast_or_null<FuncOp>(symbol);
  }
  if (auto makeFunc = callable.getDefiningOp<MakeFunctionOp>()) {
    Operation *symbol =
        SymbolTable::lookupNearestSymbolFrom(from, makeFunc.getTargetAttr());
    return dyn_cast_or_null<FuncOp>(symbol);
  }
  return nullptr;
}

bool funcResultIsPublished(Operation *funcLike, unsigned resultIndex) {
  return funcLike && arrayAttrContainsIndex(funcLike->getAttrOfType<ArrayAttr>(
                                                "lython.returns_published"),
                                            resultIndex);
}

int getEntryArgumentIndex(FuncOp func, Value value) {
  value = stripTransparentPublicationOps(value);
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return -1;
  if (arg.getOwner() != &func.getBody().front())
    return -1;
  return static_cast<int>(arg.getArgNumber());
}

ArrayAttr buildSortedIndexArrayAttr(MLIRContext *ctx,
                                    const llvm::DenseSet<int> &indices) {
  SmallVector<int64_t, 4> sorted(indices.begin(), indices.end());
  std::sort(sorted.begin(), sorted.end());

  SmallVector<Attribute, 4> attrs;
  attrs.reserve(sorted.size());
  for (int64_t idx : sorted)
    attrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), idx));
  return ArrayAttr::get(ctx, attrs);
}

bool updateFuncPublicationSummaryAttrs(
    FuncOp func, const llvm::DenseSet<int> &publishesArgs,
    const llvm::DenseSet<int> &capturesPublished,
    const llvm::DenseSet<int> &returnsPublished) {
  bool changed = false;
  auto clearOrSetArrayAttr = [&](StringRef name,
                                 const llvm::DenseSet<int> &indices) {
    ArrayAttr current = func->getAttrOfType<ArrayAttr>(name);
    if (indices.empty()) {
      if (current) {
        func->removeAttr(name);
        changed = true;
      }
      return;
    }
    ArrayAttr updated = buildSortedIndexArrayAttr(func.getContext(), indices);
    if (current == updated)
      return;
    func->setAttr(name, updated);
    changed = true;
  };

  clearOrSetArrayAttr("lython.publishes_args", publishesArgs);
  clearOrSetArrayAttr("lython.captures_published", capturesPublished);
  clearOrSetArrayAttr("lython.returns_published", returnsPublished);
  return changed;
}

bool isDefinitelyPublishedStaticClassValue(Value value);

static bool insertPublishBeforeOperand(Operation *owner,
                                       unsigned operandIndex) {
  Value operand = owner->getOperand(operandIndex);
  if (!isPublicationSummaryResultType(operand.getType()))
    return false;
  Value root = stripIdentityCasts(operand);
  if (isa_and_nonnull<PublishOp>(root.getDefiningOp()))
    return false;
  if (isa<ClassType>(operand.getType()) &&
      isDefinitelyPublishedStaticClassValue(operand))
    return false;

  OpBuilder builder(owner);
  auto publish =
      builder.create<PublishOp>(owner->getLoc(), operand.getType(), operand);
  owner->setOperand(operandIndex, publish.getResult());
  return true;
}

static bool insertPublishBeforeTupleElement(TupleCreateOp tupleCreate,
                                            unsigned operandIndex) {
  if (!tupleCreate || operandIndex >= tupleCreate->getNumOperands())
    return false;

  Value operand = tupleCreate->getOperand(operandIndex);
  if (!isPublicationSummaryResultType(operand.getType()))
    return false;
  Value root = stripIdentityCasts(operand);
  if (isa_and_nonnull<PublishOp>(root.getDefiningOp()))
    return false;
  if (isa<ClassType>(operand.getType()) &&
      isDefinitelyPublishedStaticClassValue(operand))
    return false;

  OpBuilder builder(tupleCreate);
  auto publish = builder.create<PublishOp>(tupleCreate.getLoc(),
                                           operand.getType(), operand);
  tupleCreate->setOperand(operandIndex, publish.getResult());
  return true;
}

template <typename CallbackT>
static void forEachDirectPositionalOperand(CallVectorOp op,
                                           CallbackT &&callback) {
  auto kwnames = stripIdentityCasts(op.getKwnames());
  auto kwvalues = stripIdentityCasts(op.getKwvalues());
  if (!isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  Value posargs = stripIdentityCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>()) {
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
  }
}

template <typename CallbackT>
static void forEachDirectPositionalOperand(CallOp op, CallbackT &&callback) {
  Value kwargs = stripIdentityCasts(op.getKwargs());
  if (!isa_and_nonnull<NoneOp>(kwargs.getDefiningOp()))
    return;

  Value posargs = stripIdentityCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>()) {
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
  }
}

template <typename CallbackT>
static void forEachDirectPositionalOperand(InvokeOp op, CallbackT &&callback) {
  auto kwnames = stripIdentityCasts(op.getKwnames());
  auto kwvalues = stripIdentityCasts(op.getKwvalues());
  if (!isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  Value posargs = stripIdentityCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>()) {
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
  }
}

static SmallVector<unsigned, 4>
getPublicationArgsForDirectCallee(CallVectorOp op) {
  auto callee = resolveDirectPyFuncSymbol(op, op.getCallable());
  if (!callee)
    return {};
  llvm::SmallDenseSet<unsigned, 4> indices;
  auto collect = [&](ArrayAttr attr) {
    if (!attr)
      return;
    for (Attribute element : attr) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        continue;
      int64_t argIndex = intAttr.getInt();
      if (argIndex < 0)
        continue;
      indices.insert(static_cast<unsigned>(argIndex));
    }
  };
  collect(callee->getAttrOfType<ArrayAttr>("lython.publishes_args"));
  collect(callee->getAttrOfType<ArrayAttr>("lython.captures_published"));
  SmallVector<unsigned, 4> result(indices.begin(), indices.end());
  llvm::sort(result);
  return result;
}

static SmallVector<unsigned, 4> getPublicationArgsForDirectCallee(CallOp op) {
  auto callee = resolveDirectPyFuncSymbol(op, op.getCallable());
  if (!callee)
    return {};
  llvm::SmallDenseSet<unsigned, 4> indices;
  auto collect = [&](ArrayAttr attr) {
    if (!attr)
      return;
    for (Attribute element : attr) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        continue;
      int64_t argIndex = intAttr.getInt();
      if (argIndex < 0)
        continue;
      indices.insert(static_cast<unsigned>(argIndex));
    }
  };
  collect(callee->getAttrOfType<ArrayAttr>("lython.publishes_args"));
  collect(callee->getAttrOfType<ArrayAttr>("lython.captures_published"));
  SmallVector<unsigned, 4> result(indices.begin(), indices.end());
  llvm::sort(result);
  return result;
}

static SmallVector<unsigned, 4> getPublicationArgsForDirectCallee(InvokeOp op) {
  auto callee = resolveDirectPyFuncSymbol(op, op.getCallable());
  if (!callee)
    return {};
  llvm::SmallDenseSet<unsigned, 4> indices;
  auto collect = [&](ArrayAttr attr) {
    if (!attr)
      return;
    for (Attribute element : attr) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        continue;
      int64_t argIndex = intAttr.getInt();
      if (argIndex < 0)
        continue;
      indices.insert(static_cast<unsigned>(argIndex));
    }
  };
  collect(callee->getAttrOfType<ArrayAttr>("lython.publishes_args"));
  collect(callee->getAttrOfType<ArrayAttr>("lython.captures_published"));
  SmallVector<unsigned, 4> result(indices.begin(), indices.end());
  llvm::sort(result);
  return result;
}

static void collectMakeFunctionPublicationSites(
    MakeFunctionOp op,
    SmallVectorImpl<std::pair<Operation *, unsigned>> &insertionSites,
    SmallVectorImpl<std::pair<TupleCreateOp, unsigned>> &tupleInsertionSites) {
  unsigned operandIndex = 0;

  auto collectTupleElements = [&](Value tupleValue) {
    Value root = stripIdentityCasts(tupleValue);
    if (auto tupleCreate = root.getDefiningOp<TupleCreateOp>()) {
      for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands())) {
        if (!isPublicationSummaryResultType(operand.getType()))
          continue;
        tupleInsertionSites.emplace_back(tupleCreate,
                                         static_cast<unsigned>(idx));
      }
      return true;
    }
    return false;
  };

  if (Value defaults = op.getDefaults()) {
    if (!collectTupleElements(defaults))
      insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (Value kwdefaults = op.getKwdefaults()) {
    insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (Value closure = op.getClosure()) {
    if (!collectTupleElements(closure))
      insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (Value annotations = op.getAnnotations()) {
    insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (Value moduleName = op.getModule()) {
    insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
}

void insertPublishesAtPublicationBoundaries(ModuleOp module) {
  SmallVector<std::pair<Operation *, unsigned>, 16> insertionSites;
  SmallVector<std::pair<TupleCreateOp, unsigned>, 16> tupleInsertionSites;

  module.walk([&](ListAppendOp op) {
    insertionSites.emplace_back(op.getOperation(), 1);
  });

  module.walk([&](DictInsertOp op) {
    insertionSites.emplace_back(op.getOperation(), 1);
    insertionSites.emplace_back(op.getOperation(), 2);
  });

  module.walk([&](AttrSetOp op) {
    if (isDefinitelyPublishedStaticClassValue(op.getObject()))
      insertionSites.emplace_back(op.getOperation(), 1);
  });

  module.walk([&](ReturnOp op) {
    for (auto [operandIndex, operand] : llvm::enumerate(op.getOperands())) {
      if (!isPublicationSummaryResultType(operand.getType()))
        continue;
      insertionSites.emplace_back(op.getOperation(),
                                  static_cast<unsigned>(operandIndex));
    }
  });

  module.walk([&](MakeFunctionOp op) {
    collectMakeFunctionPublicationSites(op, insertionSites,
                                        tupleInsertionSites);
  });

  module.walk([&](CallVectorOp op) {
    auto publicationArgs = getPublicationArgsForDirectCallee(op);
    if (publicationArgs.empty())
      return;
    auto tupleCreate = op.getPosargs().getDefiningOp<TupleCreateOp>();
    if (!tupleCreate)
      return;
    for (unsigned argIndex : publicationArgs)
      tupleInsertionSites.emplace_back(tupleCreate, argIndex);
  });

  module.walk([&](CallOp op) {
    auto publicationArgs = getPublicationArgsForDirectCallee(op);
    if (publicationArgs.empty())
      return;
    auto tupleCreate = op.getPosargs().getDefiningOp<TupleCreateOp>();
    if (!tupleCreate)
      return;
    for (unsigned argIndex : publicationArgs)
      tupleInsertionSites.emplace_back(tupleCreate, argIndex);
  });

  module.walk([&](InvokeOp op) {
    auto publicationArgs = getPublicationArgsForDirectCallee(op);
    if (publicationArgs.empty())
      return;
    auto tupleCreate = op.getPosargs().getDefiningOp<TupleCreateOp>();
    if (!tupleCreate)
      return;
    for (unsigned argIndex : publicationArgs)
      tupleInsertionSites.emplace_back(tupleCreate, argIndex);
  });

  for (auto [owner, operandIndex] : insertionSites)
    insertPublishBeforeOperand(owner, operandIndex);
  for (auto [tupleCreate, operandIndex] : tupleInsertionSites)
    insertPublishBeforeTupleElement(tupleCreate, operandIndex);
}

void computeLocalPublicationSummaries(ModuleOp module) {
  bool changed = false;
  do {
    changed = false;

    module.walk([&](FuncOp func) {
      llvm::DenseSet<int> publishesArgs;
      llvm::DenseSet<int> capturesPublished;
      llvm::DenseSet<int> returnsPublished;

      func.walk([&](ListAppendOp op) {
        int argIndex = getEntryArgumentIndex(func, op.getValue());
        if (argIndex < 0)
          return;
        publishesArgs.insert(argIndex);
        capturesPublished.insert(argIndex);
      });

      func.walk([&](DictInsertOp op) {
        for (Value operand : {op.getKey(), op.getValue()}) {
          int argIndex = getEntryArgumentIndex(func, operand);
          if (argIndex < 0)
            continue;
          publishesArgs.insert(argIndex);
          capturesPublished.insert(argIndex);
        }
      });

      func.walk([&](AttrSetOp op) {
        if (!isDefinitelyPublishedStaticClassValue(op.getObject()))
          return;
        int argIndex = getEntryArgumentIndex(func, op.getValue());
        if (argIndex < 0)
          return;
        publishesArgs.insert(argIndex);
        capturesPublished.insert(argIndex);
      });

      auto recordEscapingArgument = [&](Value value) {
        int argIndex = getEntryArgumentIndex(func, value);
        if (argIndex < 0)
          return;
        publishesArgs.insert(argIndex);
        capturesPublished.insert(argIndex);
      };

      func.walk([&](MakeFunctionOp op) {
        auto recordTupleElements = [&](Value tupleValue) {
          Value root = stripIdentityCasts(tupleValue);
          if (auto tupleCreate = root.getDefiningOp<TupleCreateOp>()) {
            for (Value element : tupleCreate.getElements()) {
              if (!isPublicationSummaryResultType(element.getType()))
                continue;
              recordEscapingArgument(element);
            }
            return true;
          }
          return false;
        };

        if (Value defaults = op.getDefaults())
          if (!recordTupleElements(defaults) &&
              isPublicationSummaryResultType(defaults.getType()))
            recordEscapingArgument(defaults);

        if (Value kwdefaults = op.getKwdefaults())
          if (isPublicationSummaryResultType(kwdefaults.getType()))
            recordEscapingArgument(kwdefaults);

        if (Value closure = op.getClosure())
          if (!recordTupleElements(closure) &&
              isPublicationSummaryResultType(closure.getType()))
            recordEscapingArgument(closure);

        if (Value annotations = op.getAnnotations())
          if (isPublicationSummaryResultType(annotations.getType()))
            recordEscapingArgument(annotations);
      });

      auto propagateFromCalleeSummary = [&](auto op, FuncOp callee) {
        if (!callee)
          return;

        llvm::DenseMap<unsigned, Value> operandByIndex;
        forEachDirectPositionalOperand(op, [&](unsigned idx, Value operand) {
          operandByIndex[idx] = operand;
        });
        if (operandByIndex.empty())
          return;

        auto recordIndices = [&](ArrayAttr indices, bool capture) {
          if (!indices)
            return;
          for (Attribute element : indices) {
            auto intAttr = dyn_cast<IntegerAttr>(element);
            if (!intAttr)
              continue;
            int64_t formalIndex = intAttr.getInt();
            if (formalIndex < 0)
              continue;
            auto it = operandByIndex.find(static_cast<unsigned>(formalIndex));
            if (it == operandByIndex.end())
              continue;
            int argIndex = getEntryArgumentIndex(func, it->second);
            if (argIndex < 0)
              continue;
            publishesArgs.insert(argIndex);
            if (capture)
              capturesPublished.insert(argIndex);
          }
        };

        recordIndices(callee->getAttrOfType<ArrayAttr>("lython.publishes_args"),
                      /*capture=*/false);
        recordIndices(
            callee->getAttrOfType<ArrayAttr>("lython.captures_published"),
            /*capture=*/true);
      };

      func.walk([&](CallVectorOp op) {
        propagateFromCalleeSummary(
            op, resolveDirectPyFuncSymbol(op, op.getCallable()));
      });

      func.walk([&](CallOp op) {
        propagateFromCalleeSummary(
            op, resolveDirectPyFuncSymbol(op, op.getCallable()));
      });

      func.walk([&](InvokeOp op) {
        propagateFromCalleeSummary(
            op, resolveDirectPyFuncSymbol(op, op.getCallable()));
      });

      func.walk([&](ReturnOp op) {
        for (auto [resultIndex, operand] : llvm::enumerate(op.getOperands())) {
          if (!isPublicationSummaryResultType(operand.getType()))
            continue;
          if (isa<ClassType>(operand.getType()) &&
              !isDefinitelyPublishedStaticClassValue(operand))
            continue;
          returnsPublished.insert(static_cast<int>(resultIndex));
        }
      });

      changed |= updateFuncPublicationSummaryAttrs(
          func, publishesArgs, capturesPublished, returnsPublished);
    });
  } while (changed);
}

bool isDefinitelyLocalStaticClassValue(Value value) {
  value = stripIdentityCasts(value);

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *owner = arg.getOwner();
    auto func =
        owner ? dyn_cast_or_null<func::FuncOp>(owner->getParentOp()) : nullptr;
    if (!func)
      return false;
    return arg.getArgNumber() == 0 &&
           (static_cast<bool>(func->getAttr("lython.zero_initialized_self")) ||
            static_cast<bool>(func->getAttr("lython.local_self_arg0")));
  }

  if (!isa<ClassType>(value.getType()))
    return false;

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<ClassNewOp>(def))
    return true;
  if (isa<PublishOp, ClassPromoteOp, ListGetOp>(def))
    return false;
  return false;
}

bool isDefinitelyFreshStaticClassValue(Value value) {
  value = stripIdentityCasts(value);

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *owner = arg.getOwner();
    auto func =
        owner ? dyn_cast_or_null<func::FuncOp>(owner->getParentOp()) : nullptr;
    return func && arg.getArgNumber() == 0 &&
           static_cast<bool>(func->getAttr("lython.zero_initialized_self"));
  }

  return isa_and_nonnull<ClassNewOp>(value.getDefiningOp());
}

bool isDefinitelyPublishedStaticClassValue(Value value) {
  value = stripIdentityCasts(value);
  if (!isa<ClassType>(value.getType()))
    return false;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Block *owner = arg.getOwner();
    if (!owner)
      return false;
    Block *pred = owner->getUniquePredecessor();
    if (!pred)
      return false;
    auto invoke = dyn_cast_or_null<InvokeOp>(pred->getTerminator());
    if (!invoke || invoke.getNormalDest() != owner)
      return false;
    auto callee = resolveDirectPyFuncSymbol(invoke, invoke.getCallable());
    return funcResultIsPublished(callee, arg.getArgNumber());
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<PublishOp, ClassPromoteOp, ListGetOp>(def))
    return true;

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;

  if (auto call = dyn_cast<CallOp>(def))
    return funcResultIsPublished(
        resolveDirectPyFuncSymbol(call, call.getCallable()),
        result.getResultNumber());

  if (auto call = dyn_cast<CallVectorOp>(def))
    return funcResultIsPublished(
        resolveDirectPyFuncSymbol(call, call.getCallable()),
        result.getResultNumber());

  if (auto call = dyn_cast<func::CallOp>(def)) {
    ModuleOp module = call->getParentOfType<ModuleOp>();
    if (!module)
      return false;
    return funcResultIsPublished(
        module.lookupSymbol<func::FuncOp>(call.getCallee()),
        result.getResultNumber());
  }

  return false;
}

func::FuncOp resolveLocalSelfHelper(func::FuncOp callee, ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<SymbolRefAttr>("lython.local_self_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

func::FuncOp resolveFreshInitHelper(func::FuncOp callee, ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<SymbolRefAttr>("lython.fresh_init_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

std::string getPublishedBorrowHelperAttrName(unsigned argIndex) {
  return "lython.published_borrow_helper_arg" + std::to_string(argIndex);
}

func::FuncOp resolvePublishedBorrowHelper(func::FuncOp callee,
                                          unsigned argIndex, ModuleOp module) {
  auto helperAttr = callee->getAttrOfType<SymbolRefAttr>(
      getPublishedBorrowHelperAttrName(argIndex));
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<func::FuncOp>(helperName);
}

func::FuncOp resolvePreferredDirectCallTarget(func::FuncOp callee,
                                              func::CallOp call,
                                              ModuleOp module) {
  func::FuncOp preferredTarget = callee;

  if (call.getNumOperands() != 0) {
    if (isDefinitelyFreshStaticClassValue(call.getOperand(0))) {
      if (auto freshHelper = resolveFreshInitHelper(preferredTarget, module))
        preferredTarget = freshHelper;
    }
    if (preferredTarget == callee &&
        isDefinitelyLocalStaticClassValue(call.getOperand(0))) {
      if (auto localHelper = resolveLocalSelfHelper(preferredTarget, module))
        preferredTarget = localHelper;
    }
  }

  for (unsigned idx = 1; idx < call.getNumOperands(); ++idx) {
    if (!isDefinitelyPublishedStaticClassValue(call.getOperand(idx)))
      continue;
    if (auto publishedHelper =
            resolvePublishedBorrowHelper(preferredTarget, idx, module)) {
      preferredTarget = publishedHelper;
    }
  }

  return preferredTarget;
}

void rewriteDirectFuncCallsToPreferredHelpers(ModuleOp module) {
  SmallVector<func::CallOp> toRewrite;

  module.walk([&](func::CallOp call) {
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (!callee)
      return;

    auto preferredHelper =
        resolvePreferredDirectCallTarget(callee, call, module);
    if (!preferredHelper || preferredHelper == callee)
      return;

    auto calleeType = preferredHelper.getFunctionType();
    if (calleeType.getNumInputs() != call.getNumOperands() ||
        calleeType.getNumResults() != call.getNumResults())
      return;

    for (auto [lhs, rhs] :
         llvm::zip(calleeType.getInputs(), call.getOperandTypes()))
      if (lhs != rhs)
        return;
    for (auto [lhs, rhs] :
         llvm::zip(calleeType.getResults(), call.getResultTypes()))
      if (lhs != rhs)
        return;

    toRewrite.push_back(call);
  });

  for (func::CallOp call : toRewrite) {
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (!callee)
      continue;
    auto preferredHelper =
        resolvePreferredDirectCallTarget(callee, call, module);
    if (!preferredHelper || preferredHelper == callee)
      continue;
    OpBuilder builder(call);
    auto replacement = builder.create<func::CallOp>(
        call.getLoc(), preferredHelper, call.getOperands());
    call.replaceAllUsesWith(replacement.getResults());
    call.erase();
  }
}

void eliminateRedundantClassPublishes(ModuleOp module) {
  SmallVector<PublishOp> toErase;

  module.walk([&](PublishOp op) {
    if (!isa<ClassType>(op.getResult().getType()))
      return;
    if (!isDefinitelyPublishedStaticClassValue(op.getInput()))
      return;

    op.getResult().replaceAllUsesWith(op.getInput());
    toErase.push_back(op);
  });

  for (PublishOp op : toErase)
    op.erase();
}

void markKnownLocalStaticClassAccesses(ModuleOp module) {
  module.walk([&](Operation *op) {
    Value object;
    if (auto attrGet = dyn_cast<AttrGetOp>(op)) {
      object = attrGet.getObject();
    } else if (auto attrSet = dyn_cast<AttrSetOp>(op)) {
      object = attrSet.getObject();
    } else {
      return;
    }

    if (!isDefinitelyLocalStaticClassValue(object))
      return;

    op->setAttr("lython.known_local_class_access",
                UnitAttr::get(module.getContext()));
  });
}

void markZeroInitializedSelfFirstStores(ModuleOp module) {
  module.walk([&](AttrSetOp op) {
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc || !parentFunc->getAttr("lython.zero_initialized_self"))
      return;
    if (!parentFunc.getBody().hasOneBlock())
      return;

    Value objectRoot = stripIdentityCasts(op.getObject());
    auto selfArg = dyn_cast<BlockArgument>(objectRoot);
    if (!selfArg || selfArg.getOwner() != &parentFunc.getBody().front() ||
        selfArg.getArgNumber() != 0)
      return;

    for (Operation *cursor = op->getPrevNode(); cursor;
         cursor = cursor->getPrevNode()) {
      bool touchesSelf =
          llvm::any_of(cursor->getOperands(), [&](Value operand) {
            return stripIdentityCasts(operand) == objectRoot;
          });
      if (!touchesSelf)
        continue;

      if (auto prevSet = dyn_cast<AttrSetOp>(cursor)) {
        if (stripIdentityCasts(prevSet.getObject()) == objectRoot &&
            prevSet.getNameAttr() == op.getNameAttr())
          return;
        continue;
      }

      if (isa<CastIdentityOp, AttrGetOp>(cursor))
        continue;

      return;
    }

    op->setAttr("lython.zero_init_first_store",
                UnitAttr::get(module.getContext()));
  });
}

/// Hoist integer constants to entry block and perform CSE.
void hoistIntConstants(ModuleOp module) {
  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;

    Block &entryBlock = func.getBody().front();
    llvm::StringMap<IntConstantOp> constantMap;

    // First pass: collect all IntConstantOp
    SmallVector<IntConstantOp> allConstants;
    func.walk([&](IntConstantOp op) { allConstants.push_back(op); });

    // Second pass: CSE and hoist (using string value as key)
    for (IntConstantOp op : allConstants) {
      llvm::StringRef value = op.getValue();
      auto it = constantMap.find(value);
      if (it != constantMap.end()) {
        // Replace with existing constant
        op.getResult().replaceAllUsesWith(it->second.getResult());
        op->erase();
      } else {
        // Move to entry block if not already there
        if (op->getBlock() != &entryBlock) {
          op->moveBefore(&entryBlock, entryBlock.begin());
        }
        constantMap[value] = op;
      }
    }
  });
}

/// Remove DecRef for small integer constants (-5 to 256).
/// These are immortal in Python's small integer cache.
void removeSmallIntDecrefs(ModuleOp module) {
  SmallVector<DecRefOp> toErase;

  module.walk([&](DecRefOp decref) {
    Value obj = decref.getObject();
    if (auto intConst = obj.getDefiningOp<IntConstantOp>()) {
      llvm::StringRef valueStr = intConst.getValue();
      // Parse the string to check if it's in small int range
      char *end;
      long long value = std::strtoll(valueStr.data(), &end, 10);
      // Only apply optimization if parsing succeeded and value is in range
      if (end == valueStr.data() + valueStr.size() && value >= -5 &&
          value <= 256) {
        toErase.push_back(decref);
      }
    }
  });

  for (auto op : toErase)
    op->erase();
}

// Post-lowering optimizations (operate on LLVM dialect ops)

/// CSE for LyUnicode_FromUTF8 calls within each function.
/// This reduces redundant string key creation for attribute access.
/// After CSE, adds DecRef at function end for the cached strings.
void cseStringCreation(ModuleOp module) {
  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;

    DominanceInfo dom(func);

    // Map from (string_global_symbol, length) -> first call result
    llvm::DenseMap<std::pair<StringRef, int64_t>, LLVM::CallOp> stringCache;
    SmallVector<LLVM::CallOp> toErase;
    SmallVector<LLVM::CallOp> cachedStrings;

    func.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee || *callee != "LyUnicode_FromUTF8")
        return;

      // Get the string global and length arguments
      if (callOp.getNumOperands() != 2)
        return;

      // First operand should be a GEP pointing to a global string
      auto gepOp = callOp.getOperand(0).getDefiningOp<LLVM::GEPOp>();
      if (!gepOp)
        return;
      auto addrOp = gepOp.getBase().getDefiningOp<LLVM::AddressOfOp>();
      if (!addrOp)
        return;
      StringRef globalName = addrOp.getGlobalName();

      // Second operand should be a constant length
      auto lenConst = callOp.getOperand(1).getDefiningOp<LLVM::ConstantOp>();
      if (!lenConst)
        return;
      auto lenAttr = llvm::dyn_cast<IntegerAttr>(lenConst.getValue());
      if (!lenAttr)
        return;
      int64_t len = lenAttr.getInt();

      auto key = std::make_pair(globalName, len);
      auto it = stringCache.find(key);
      if (it != stringCache.end()) {
        // Replace with cached result
        callOp.getResult().replaceAllUsesWith(it->second.getResult());
        toErase.push_back(callOp);
      } else {
        stringCache[key] = callOp;
        cachedStrings.push_back(callOp);
      }
    });

    for (auto op : toErase)
      op->erase();

    // Add DecRef for cached strings before each return
    if (!cachedStrings.empty()) {
      func.walk([&](func::ReturnOp returnOp) {
        OpBuilder builder(returnOp);
        auto decrefFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("Ly_DecRef");
        if (decrefFunc) {
          for (auto cachedCall : cachedStrings) {
            if (!dom.dominates(cachedCall, returnOp))
              continue;
            builder.create<LLVM::CallOp>(returnOp.getLoc(), decrefFunc,
                                         ValueRange{cachedCall.getResult()});
          }
        }
      });
    }
  });
}

/// CSE for singleton getter calls (Ly_GetBuiltinPrint, Ly_GetNone). These
/// return borrowed references to singletons, so no DecRef needed.
void cseSingletonGetters(ModuleOp module) {
  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;

    // Cache for each singleton getter function
    llvm::StringMap<LLVM::CallOp> singletonCache;
    SmallVector<LLVM::CallOp> toErase;

    func.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee)
        return;

      // Singleton getters that we want to CSE
      if (*callee != "Ly_GetBuiltinPrint" && *callee != "Ly_GetNone")
        return;

      StringRef funcName = *callee;
      auto it = singletonCache.find(funcName);
      if (it != singletonCache.end()) {
        // Replace with cached result
        callOp.getResult().replaceAllUsesWith(it->second.getResult());
        toErase.push_back(callOp);
      } else {
        singletonCache[funcName] = callOp;
      }
    });

    for (auto op : toErase)
      op->erase();
  });
}

/// Eliminate redundant Bool boxing/unboxing patterns.
/// Pattern:
///   %boxed = llvm.call @LyBool_FromBool(%i1_val) : (i1) -> !llvm.ptr
///   %unboxed = llvm.call @LyBool_AsBool(%boxed) : (!llvm.ptr) -> i1
///   llvm.call @Ly_DecRef(%boxed) : (!llvm.ptr) -> ()
/// Replaced with direct use of %i1_val.
void eliminateBoolBoxingUnboxing(ModuleOp module) {
  SmallVector<LLVM::CallOp> fromBoolCalls;

  // First pass: find all LyBool_FromBool calls
  module.walk([&](LLVM::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (callee && *callee == "LyBool_FromBool")
      fromBoolCalls.push_back(callOp);
  });

  // Second pass: check each FromBool call for the pattern
  for (auto fromBoolCall : fromBoolCalls) {
    if (fromBoolCall.getNumOperands() != 1)
      continue;

    Value i1Value = fromBoolCall.getOperand(0);
    Value boxedValue = fromBoolCall.getResult();

    // Find users of the boxed value
    LLVM::CallOp asBoolCall = nullptr;
    LLVM::CallOp decRefCall = nullptr;
    bool hasOtherUsers = false;

    for (Operation *user : boxedValue.getUsers()) {
      if (auto callUser = dyn_cast<LLVM::CallOp>(user)) {
        auto callee = callUser.getCallee();
        if (callee && *callee == "LyBool_AsBool" && !asBoolCall) {
          asBoolCall = callUser;
        } else if (callee && *callee == "Ly_DecRef" && !decRefCall) {
          decRefCall = callUser;
        } else {
          hasOtherUsers = true;
        }
      } else {
        hasOtherUsers = true;
      }
    }

    // Only optimize if the pattern matches:
    // - One LyBool_AsBool call
    // - Optional Ly_DecRef call (bool singletons are immortal)
    // - No other users
    if (!asBoolCall || hasOtherUsers)
      continue;

    // Replace uses of the AsBool result with the original i1 value
    asBoolCall.getResult().replaceAllUsesWith(i1Value);

    // Erase the operations (in reverse order of dependencies)
    if (decRefCall)
      decRefCall->erase();
    asBoolCall->erase();
    fromBoolCall->erase();
  }
}

/// Eliminate redundant long boxing/unboxing patterns.
/// Pattern:
///   %boxed = llvm.call @LyLong_FromI64(%i64_val) : (i64) -> !llvm.ptr
///   %unboxed = llvm.call @LyLong_AsI64(%boxed) : (!llvm.ptr) -> i64
///   llvm.call @Ly_DecRef(%boxed) : (!llvm.ptr) -> ()
/// Replaced with direct use of %i64_val.
void eliminateLongBoxingUnboxing(ModuleOp module) {
  SmallVector<LLVM::CallOp> fromI64Calls;

  module.walk([&](LLVM::CallOp callOp) {
    if (isCallTo(callOp, RuntimeSymbols::kLongFromI64))
      fromI64Calls.push_back(callOp);
  });

  for (auto fromI64Call : fromI64Calls) {
    if (fromI64Call.getNumOperands() != 1)
      continue;

    Value i64Value = fromI64Call.getOperand(0);
    Value boxedValue = fromI64Call.getResult();

    LLVM::CallOp asI64Call = nullptr;
    LLVM::CallOp decRefCall = nullptr;
    bool hasOtherUsers = false;

    for (Operation *user : boxedValue.getUsers()) {
      auto callUser = dyn_cast<LLVM::CallOp>(user);
      if (!callUser) {
        hasOtherUsers = true;
        break;
      }

      if (isCallTo(callUser, RuntimeSymbols::kLongAsI64) && !asI64Call) {
        asI64Call = callUser;
      } else if (isCallTo(callUser, RuntimeSymbols::kDecRef) && !decRefCall) {
        decRefCall = callUser;
      } else {
        hasOtherUsers = true;
        break;
      }
    }

    if (!asI64Call || hasOtherUsers)
      continue;

    asI64Call.getResult().replaceAllUsesWith(i64Value);

    if (decRefCall)
      decRefCall->erase();
    asI64Call->erase();
    fromI64Call->erase();
  }
}

/// Rewrites LyLong_Add/Sub round-trips through heap objects into direct i64
/// arithmetic when the result is immediately unboxed.
/// Pattern:
///   %lhs_box = llvm.call @LyLong_FromI64(%lhs) : (i64) -> !llvm.ptr
///   %res_box = llvm.call @LyLong_Add|Sub(%lhs_box, %rhs_box) : ... ->
///   !llvm.ptr %res = llvm.call @LyLong_AsI64(%res_box) : (!llvm.ptr) -> i64
///   llvm.call @Ly_DecRef(%res_box) : (!llvm.ptr) -> ()
/// Replaced with direct i64 add/sub, materializing LyLong_AsI64 only for
/// boxed operands that are not already from LyLong_FromI64.
void eliminateLongArithmeticRoundTrips(ModuleOp module) {
  SmallVector<LLVM::CallOp> asI64Calls;

  module.walk([&](LLVM::CallOp callOp) {
    if (isCallTo(callOp, RuntimeSymbols::kLongAsI64))
      asI64Calls.push_back(callOp);
  });

  for (auto asI64Call : asI64Calls) {
    if (asI64Call.getNumOperands() != 1)
      continue;

    auto arithmeticCall = asI64Call.getOperand(0).getDefiningOp<LLVM::CallOp>();
    if (!arithmeticCall || arithmeticCall.getNumOperands() != 2)
      continue;

    bool isAdd = isCallTo(arithmeticCall, RuntimeSymbols::kLongAdd);
    bool isSub = isCallTo(arithmeticCall, RuntimeSymbols::kLongSub);
    if (!isAdd && !isSub)
      continue;

    Value boxedResult = arithmeticCall.getResult();
    LLVM::CallOp decRefCall = nullptr;
    bool hasOtherUsers = false;
    for (Operation *user : boxedResult.getUsers()) {
      auto callUser = dyn_cast<LLVM::CallOp>(user);
      if (!callUser) {
        hasOtherUsers = true;
        break;
      }

      if (callUser == asI64Call) {
        continue;
      }
      if (isCallTo(callUser, RuntimeSymbols::kDecRef) && !decRefCall) {
        decRefCall = callUser;
      } else {
        hasOtherUsers = true;
        break;
      }
    }

    if (hasOtherUsers)
      continue;

    OpBuilder builder(asI64Call);
    Location loc = asI64Call.getLoc();
    Value lhsBoxed = arithmeticCall.getOperand(0);
    Value rhsBoxed = arithmeticCall.getOperand(1);

    Value lhs = materializeI64FromLong(module, builder, loc, lhsBoxed);
    Value rhs = materializeI64FromLong(module, builder, loc, rhsBoxed);

    Value arithmeticResult;
    if (isAdd) {
      arithmeticResult =
          builder.create<LLVM::AddOp>(loc, lhs.getType(), lhs, rhs).getResult();
    } else {
      arithmeticResult =
          builder.create<LLVM::SubOp>(loc, lhs.getType(), lhs, rhs).getResult();
    }

    asI64Call.getResult().replaceAllUsesWith(arithmeticResult);

    if (decRefCall)
      decRefCall->erase();
    asI64Call->erase();
    arithmeticCall->erase();

    if (auto fromI64Call = lhsBoxed.getDefiningOp<LLVM::CallOp>();
        fromI64Call && isCallTo(fromI64Call, RuntimeSymbols::kLongFromI64)) {
      eraseCallAndOptionalDecRefUsers(fromI64Call);
    }
    if (auto fromI64Call = rhsBoxed.getDefiningOp<LLVM::CallOp>();
        fromI64Call && isCallTo(fromI64Call, RuntimeSymbols::kLongFromI64)) {
      eraseCallAndOptionalDecRefUsers(fromI64Call);
    }
  }
}

/// CSE for LyLong_FromI64 for small integers (-5 to 256).
/// Small integers are immortal, so sharing is safe.
void cseSmallIntFromI64(ModuleOp module) {
  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;

    llvm::DenseMap<int64_t, LLVM::CallOp> cache;
    SmallVector<LLVM::CallOp> toErase;

    func.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee || *callee != "LyLong_FromI64")
        return;
      if (callOp.getNumOperands() != 1)
        return;
      auto constOp = callOp.getOperand(0).getDefiningOp<LLVM::ConstantOp>();
      if (!constOp)
        return;
      auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
      if (!intAttr)
        return;
      int64_t value = intAttr.getInt();
      if (value < -5 || value > 256)
        return;

      auto it = cache.find(value);
      if (it != cache.end()) {
        callOp.getResult().replaceAllUsesWith(it->second.getResult());
        toErase.push_back(callOp);
      } else {
        cache[value] = callOp;
      }
    });

    for (auto op : toErase)
      op->erase();
  });
}

/// CSE for LLVM constants within each function.
void cseConstants(ModuleOp module) {
  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;

    // Map from (type, value) -> first constant op
    llvm::DenseMap<std::pair<Type, Attribute>, LLVM::ConstantOp> constantCache;
    SmallVector<LLVM::ConstantOp> toErase;

    func.walk([&](LLVM::ConstantOp constOp) {
      auto key = std::make_pair(constOp.getType(), constOp.getValue());
      auto it = constantCache.find(key);
      if (it != constantCache.end()) {
        constOp.getResult().replaceAllUsesWith(it->second.getResult());
        toErase.push_back(constOp);
      } else {
        constantCache[key] = constOp;
      }
    });

    for (auto op : toErase)
      op->erase();
  });
}

/// Dead code elimination for unused LLVM operations after CSE.
/// This cleans up AddressOf, GEP, and constants that were only used by CSE'd
/// calls.
void eliminateDeadCode(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;

    module.walk([&](Operation *op) {
      // Only consider operations with no side effects
      if (!isa<LLVM::AddressOfOp, LLVM::GEPOp, LLVM::ConstantOp>(op))
        return;

      // Check if all results are unused
      bool allUnused = true;
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          allUnused = false;
          break;
        }
      }
      if (allUnused)
        toErase.push_back(op);
    });

    for (Operation *op : toErase) {
      op->erase();
      changed = true;
    }
  }
}

} // namespace py::optimizer
