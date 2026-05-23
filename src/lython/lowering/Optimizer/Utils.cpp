// This file implements shared optimization helpers specific to the Py dialect
// and its lowered LLVM representation. These optimizations include:
//   - Dead tuple cleanup (removing tuples only used by DecRefOp)
//   - Integer constant hoisting and CSE
//   - Small integer DecRef removal (immortal integers -5 to 256)
//   - Singleton getter CSE (Ly_GetBuiltinPrint, Ly_GetNone, etc.)
//   - Bool boxing/unboxing elimination (LyBool_FromBool + LyBool_AsBool)
//   - LLVM constant CSE
//   - Dead code elimination for unused LLVM/tensor producer operations

#include "Optimizer/Utils.h"

#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>

namespace py::optimizer {

mlir::Value value::stripCasts(mlir::Value value);

bool runtime::Call::is(mlir::LLVM::CallOp callOp, llvm::StringRef calleeName) {
  auto callee = callOp.getCallee();
  return callee && *callee == calleeName;
}

mlir::LLVM::CallOp runtime::Call::optionalReleaseUser(mlir::Value value) {
  mlir::LLVM::CallOp decRefCall = nullptr;
  for (mlir::Operation *user : value.getUsers()) {
    auto callUser = mlir::dyn_cast<mlir::LLVM::CallOp>(user);
    if (!callUser || !runtime::Call::is(callUser, RuntimeSymbols::kDecRef))
      return nullptr;
    if (decRefCall)
      return nullptr;
    decRefCall = callUser;
  }
  return decRefCall;
}

void runtime::Call::eraseWithOptionalRelease(mlir::LLVM::CallOp callOp) {
  if (!callOp || callOp->getNumResults() != 1)
    return;

  mlir::Value result = callOp.getResult();
  mlir::LLVM::CallOp decRefCall = runtime::Call::optionalReleaseUser(result);
  if (decRefCall)
    decRefCall->erase();

  if (result.use_empty())
    callOp->erase();
}

mlir::LLVM::LLVMFuncOp runtime::Func::getOrCreate(mlir::ModuleOp module,
                                                  llvm::StringRef name,
                                                  mlir::Type resultType,
                                                  mlir::ValueRange operands) {
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;

  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
    argTypes.push_back(operand.getType());

  auto fnType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name,
                                                      fnType);
}

mlir::Value runtime::Long::asI64(mlir::ModuleOp module,
                                 mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value boxedLong) {
  if (auto fromI64Call = boxedLong.getDefiningOp<mlir::LLVM::CallOp>()) {
    if (runtime::Call::is(fromI64Call, RuntimeSymbols::kLongFromI64) &&
        fromI64Call.getNumOperands() == 1) {
      return fromI64Call.getOperand(0);
    }
  }

  auto asI64Func = runtime::Func::getOrCreate(
      module, RuntimeSymbols::kLongAsI64, builder.getI64Type(),
      mlir::ValueRange{boxedLong});
  return builder
      .create<mlir::LLVM::CallOp>(loc, asI64Func, mlir::ValueRange{boxedLong})
      .getResult();
}

/// Remove py.incref on !py.class values when it only survives because a
/// tuple-consuming call was rewritten into a direct func.call that borrows
/// the class slot.
bool call::cleanupClassIncrefs(mlir::ModuleOp module) {
  llvm::SmallVector<IncRefOp> toErase;

  module.walk([&](IncRefOp incref) {
    mlir::Value object = incref.getObject();
    if (!mlir::isa<ClassType>(object.getType()))
      return;

    for (mlir::Operation *cursor = incref->getNextNode(); cursor;
         cursor = cursor->getNextNode()) {
      if (mlir::isa<mlir::UnrealizedConversionCastOp, mlir::func::ConstantOp>(
              cursor))
        continue;

      auto call = mlir::dyn_cast<mlir::func::CallOp>(cursor);
      if (!call)
        break;

      bool passesClass =
          llvm::any_of(call.getOperands(), [&](mlir::Value operand) {
            if (operand == object)
              return true;
            if (auto cast =
                    operand.getDefiningOp<mlir::UnrealizedConversionCastOp>())
              return cast->getNumOperands() == 1 &&
                     cast.getOperand(0) == object;
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

void scalar::removeUnusedNone(mlir::ModuleOp module) {
  llvm::SmallVector<NoneOp> toErase;
  module.walk([&](NoneOp op) {
    if (op.getResult().use_empty())
      toErase.push_back(op);
  });
  for (NoneOp op : toErase)
    op.erase();
}

void scalar::dropNoneDecrefs(mlir::ModuleOp module) {
  llvm::SmallVector<DecRefOp> toErase;
  module.walk([&](DecRefOp op) {
    if (mlir::isa<NoneType>(op.getObject().getType()))
      toErase.push_back(op);
  });
  for (DecRefOp op : toErase)
    op.erase();
}

void consume::attrSetValues(mlir::ModuleOp module) {
  llvm::SmallVector<DecRefOp> toErase;

  module.walk([&](AttrSetOp op) {
    mlir::Value value = op.getValue();
    if (!isPyType(value.getType()) ||
        mlir::isa<NoneType, BoolType>(value.getType()))
      return;

    DecRefOp trailingDrop;
    for (mlir::Operation *user : value.getUsers()) {
      if (user == op.getOperation())
        continue;
      auto decref = mlir::dyn_cast<DecRefOp>(user);
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

    op->setAttr("ly.consume_value", mlir::UnitAttr::get(module.getContext()));
    toErase.push_back(trailingDrop);
  });

  for (DecRefOp op : toErase)
    op.erase();
}

void consume::listAppendValues(mlir::ModuleOp module) {
  llvm::SmallVector<DecRefOp> toErase;

  module.walk([&](ListAppendOp op) {
    mlir::Value value = op.getValue();
    if (!isPyType(value.getType()) ||
        mlir::isa<NoneType, BoolType>(value.getType()))
      return;
    // Only steal ownership that was materialized explicitly for a publication
    // boundary. Borrowed values such as function arguments must keep the
    // append-side retain.
    if (!value.getDefiningOp<PublishOp>())
      return;

    DecRefOp trailingDrop;
    for (mlir::Operation *user : value.getUsers()) {
      if (user == op.getOperation())
        continue;
      auto decref = mlir::dyn_cast<DecRefOp>(user);
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

    op->setAttr("ly.consume_value", mlir::UnitAttr::get(module.getContext()));
    toErase.push_back(trailingDrop);
  });

  for (DecRefOp op : toErase)
    op.erase();
}

mlir::Value value::stripCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

mlir::Value value::stripPublications(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        break;
      value = cast.getOperand(0);
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

bool publication::tracks(mlir::Type type) {
  return mlir::isa<ListType, DictType, TupleType, ObjectType, ClassType>(type);
}

bool attr::containsIndex(mlir::ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (mlir::Attribute element : attr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
    if (!intAttr)
      continue;
    if (intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

FuncOp call::pyFunc(mlir::Operation *from, mlir::Value callable) {
  callable = value::stripCasts(callable);

  if (auto funcObject = callable.getDefiningOp<FuncObjectOp>()) {
    mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
        from, funcObject.getTargetAttr());
    return mlir::dyn_cast_or_null<FuncOp>(symbol);
  }
  if (auto makeFunc = callable.getDefiningOp<MakeFunctionOp>()) {
    mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
        from, makeFunc.getTargetAttr());
    return mlir::dyn_cast_or_null<FuncOp>(symbol);
  }
  return nullptr;
}

bool publication::result(mlir::Operation *funcLike, unsigned resultIndex) {
  return funcLike &&
         attr::containsIndex(
             funcLike->getAttrOfType<mlir::ArrayAttr>("ly.returns_published"),
             resultIndex);
}

int publication::entryArg(FuncOp func, mlir::Value value) {
  value = ::py::optimizer::value::stripPublications(value);
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg)
    return -1;
  if (arg.getOwner() != &func.getBody().front())
    return -1;
  return static_cast<int>(arg.getArgNumber());
}

mlir::ArrayAttr attr::indexArray(mlir::MLIRContext *ctx,
                                 const llvm::DenseSet<int> &indices) {
  llvm::SmallVector<int64_t, 4> sorted(indices.begin(), indices.end());
  std::sort(sorted.begin(), sorted.end());

  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.reserve(sorted.size());
  for (int64_t idx : sorted)
    attrs.push_back(
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), idx));
  return mlir::ArrayAttr::get(ctx, attrs);
}

bool publication::update(FuncOp func, const llvm::DenseSet<int> &publishesArgs,
                         const llvm::DenseSet<int> &capturesPublished,
                         const llvm::DenseSet<int> &returnsPublished,
                         const llvm::DenseSet<int> &readonlyArgs,
                         const llvm::DenseSet<int> &mutableArgs) {
  bool changed = false;
  auto clearOrSetArrayAttr = [&](llvm::StringRef name,
                                 const llvm::DenseSet<int> &indices) {
    mlir::ArrayAttr current = func->getAttrOfType<mlir::ArrayAttr>(name);
    if (indices.empty()) {
      if (current) {
        func->removeAttr(name);
        changed = true;
      }
      return;
    }
    mlir::ArrayAttr updated = attr::indexArray(func.getContext(), indices);
    if (current == updated)
      return;
    func->setAttr(name, updated);
    changed = true;
  };

  clearOrSetArrayAttr("ly.publishes_args", publishesArgs);
  clearOrSetArrayAttr("ly.captures_published", capturesPublished);
  clearOrSetArrayAttr("ly.returns_published", returnsPublished);
  clearOrSetArrayAttr("ly.readonly_args", readonlyArgs);
  clearOrSetArrayAttr("ly.mutable_args", mutableArgs);
  return changed;
}

bool class_state::published(mlir::Value value);

static TupleCreateOp getStaticTupleCreate(mlir::Value value) {
  value = value::stripCasts(value);
  return mlir::dyn_cast_or_null<TupleCreateOp>(value.getDefiningOp());
}

static mlir::Value stripStaticMetadataValue(mlir::Value value) {
  while (true) {
    value = value::stripPublications(value);
    if (auto upcast = value.getDefiningOp<UpcastOp>()) {
      if (isCompilerOwnedMemRefContainerType(upcast.getInput().getType()))
        return value;
      value = upcast.getInput();
      continue;
    }
    return value;
  }
}

static bool
collectStaticDictStringValues(mlir::Value dictValue, mlir::Operation *beforeOp,
                              llvm::StringMap<mlir::Value> &values) {
  mlir::Value root = value::stripPublications(dictValue);
  if (!root.getDefiningOp<DictEmptyOp>())
    return false;

  llvm::SmallVector<DictInsertOp, 8> inserts;
  for (mlir::Operation *user : root.getUsers()) {
    auto insert = mlir::dyn_cast<DictInsertOp>(user);
    if (!insert)
      continue;
    if (beforeOp && insert->getBlock() != beforeOp->getBlock())
      return false;
    if (beforeOp && !insert->isBeforeInBlock(beforeOp))
      continue;
    inserts.push_back(insert);
  }
  llvm::sort(inserts, [](DictInsertOp lhs, DictInsertOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  for (DictInsertOp insert : inserts) {
    mlir::Value key = value::stripCasts(insert.getKey());
    auto strConst = key.getDefiningOp<StrConstantOp>();
    if (!strConst)
      return false;
    values[strConst.getValueAttr().getValue()] =
        stripStaticMetadataValue(insert.getValue());
  }
  return true;
}

static bool
collectStaticKeywordArgumentValues(mlir::Value namesValue,
                                   mlir::Value valuesValue,
                                   llvm::StringMap<mlir::Value> &values) {
  namesValue = value::stripCasts(namesValue);
  valuesValue = value::stripCasts(valuesValue);
  if (mlir::isa_and_nonnull<TupleEmptyOp>(namesValue.getDefiningOp()) &&
      mlir::isa_and_nonnull<TupleEmptyOp>(valuesValue.getDefiningOp()))
    return true;

  auto names = namesValue.getDefiningOp<TupleCreateOp>();
  auto args = valuesValue.getDefiningOp<TupleCreateOp>();
  if (!names || !args ||
      names.getElements().size() != args.getElements().size())
    return false;

  for (auto [name, value] :
       llvm::zip(names.getElements(), args.getElements())) {
    name = value::stripCasts(name);
    auto strConst = name.getDefiningOp<StrConstantOp>();
    if (!strConst)
      return false;
    values[strConst.getValueAttr().getValue()] =
        stripStaticMetadataValue(value);
  }
  return true;
}

void call::staticDefaults(mlir::ModuleOp module) {
  llvm::SmallVector<CallVectorOp> calls;
  module.walk([&](CallVectorOp op) { calls.push_back(op); });

  for (CallVectorOp call : calls) {
    mlir::Value callable = value::stripCasts(call.getCallable());
    auto makeFunc = callable.getDefiningOp<MakeFunctionOp>();
    if (!makeFunc || (!makeFunc.getDefaults() && !makeFunc.getKwdefaults() &&
                      !makeFunc.getClosure()))
      continue;

    auto funcType = mlir::dyn_cast<FuncType>(makeFunc.getResult().getType());
    if (!funcType)
      continue;
    auto signature = funcType.getSignature();
    if (signature.hasVararg())
      continue;
    llvm::ArrayRef<mlir::Type> positionalTypes = signature.getPositionalTypes();
    unsigned positionalCount = static_cast<unsigned>(positionalTypes.size());
    llvm::ArrayRef<mlir::Type> kwonlyTypes = signature.getKwOnlyTypes();
    unsigned kwonlyCount = static_cast<unsigned>(kwonlyTypes.size());

    TupleCreateOp posargs = getStaticTupleCreate(call.getPosargs());
    if (!posargs)
      continue;
    unsigned providedCount =
        static_cast<unsigned>(posargs.getElements().size());
    if (providedCount > positionalCount)
      continue;

    llvm::SmallVector<mlir::Value, 8> elements;
    elements.append(posargs.getElements().begin(), posargs.getElements().end());
    if (providedCount < positionalCount) {
      TupleCreateOp defaults = getStaticTupleCreate(makeFunc.getDefaults());
      if (!defaults)
        continue;
      unsigned defaultsCount =
          static_cast<unsigned>(defaults.getElements().size());
      if (providedCount + defaultsCount < positionalCount)
        continue;

      unsigned firstDefaultIndex = positionalCount - defaultsCount;
      if (providedCount < firstDefaultIndex)
        continue;
      for (unsigned index = providedCount; index < positionalCount; ++index)
        elements.push_back(defaults.getElements()[index - firstDefaultIndex]);
    }

    if (kwonlyCount > 0) {
      mlir::ArrayAttr kwonlyNames = nullptr;
      if (mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
              call, makeFunc.getTargetAttr()))
        if (auto targetFunc = mlir::dyn_cast<FuncOp>(symbol))
          kwonlyNames = targetFunc.getKwonlyNamesAttr();
      if (!kwonlyNames || kwonlyNames.size() != kwonlyCount)
        continue;
      llvm::StringMap<mlir::Value> kwdefaultValues;
      if (!makeFunc.getKwdefaults() ||
          !collectStaticDictStringValues(makeFunc.getKwdefaults(),
                                         makeFunc.getOperation(),
                                         kwdefaultValues))
        continue;
      llvm::StringMap<mlir::Value> explicitKwValues;
      if (!collectStaticKeywordArgumentValues(
              call.getKwnames(), call.getKwvalues(), explicitKwValues))
        continue;
      bool missingDefault = false;
      for (mlir::Attribute attr : kwonlyNames) {
        auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!nameAttr) {
          missingDefault = true;
          break;
        }
        auto explicitIt = explicitKwValues.find(nameAttr.getValue());
        if (explicitIt != explicitKwValues.end()) {
          elements.push_back(explicitIt->second);
          continue;
        }
        auto defaultIt = kwdefaultValues.find(nameAttr.getValue());
        if (defaultIt == kwdefaultValues.end()) {
          missingDefault = true;
          break;
        }
        elements.push_back(defaultIt->second);
      }
      if (missingDefault)
        continue;
    }

    if (mlir::Value closureValue = makeFunc.getClosure()) {
      TupleCreateOp closure = getStaticTupleCreate(closureValue);
      if (!closure)
        continue;
      elements.append(closure.getElements().begin(),
                      closure.getElements().end());
    }

    if (elements.size() == posargs.getElements().size())
      continue;

    llvm::SmallVector<mlir::Type, 8> elementTypes;
    elementTypes.reserve(elements.size());
    for (mlir::Value element : elements)
      elementTypes.push_back(element.getType());

    mlir::OpBuilder builder(call);
    auto newTuple = builder.create<TupleCreateOp>(
        call.getLoc(), TupleType::get(module.getContext(), elementTypes),
        elements);
    auto funcObject = builder.create<FuncObjectOp>(
        call.getLoc(), makeFunc.getResult().getType(),
        makeFunc.getTargetAttr());
    auto emptyKwTuple = builder.create<TupleEmptyOp>(
        call.getLoc(), TupleType::get(module.getContext(), {}));
    call.getPosargsMutable().assign(newTuple.getResult());
    call.getCallableMutable().assign(funcObject.getResult());
    call.getKwnamesMutable().assign(emptyKwTuple.getResult());
    call.getKwvaluesMutable().assign(emptyKwTuple.getResult());
  }
}

static bool insertPublishBeforeOperand(mlir::Operation *owner,
                                       unsigned operandIndex) {
  mlir::Value operand = owner->getOperand(operandIndex);
  if (!publication::tracks(operand.getType()))
    return false;
  mlir::Value root = value::stripCasts(operand);
  if (mlir::isa_and_nonnull<PublishOp>(root.getDefiningOp()))
    return false;
  if (mlir::isa<ClassType>(operand.getType()) &&
      class_state::published(operand))
    return false;

  mlir::OpBuilder builder(owner);
  auto publish =
      builder.create<PublishOp>(owner->getLoc(), operand.getType(), operand);
  owner->setOperand(operandIndex, publish.getResult());
  return true;
}

static bool insertPublishBeforeTupleElement(TupleCreateOp tupleCreate,
                                            unsigned operandIndex) {
  if (!tupleCreate || operandIndex >= tupleCreate->getNumOperands())
    return false;

  mlir::Value operand = tupleCreate->getOperand(operandIndex);
  if (!publication::tracks(operand.getType()))
    return false;
  mlir::Value root = value::stripCasts(operand);
  if (mlir::isa_and_nonnull<PublishOp>(root.getDefiningOp()))
    return false;
  if (mlir::isa<ClassType>(operand.getType()) &&
      class_state::published(operand))
    return false;

  mlir::OpBuilder builder(tupleCreate);
  auto publish = builder.create<PublishOp>(tupleCreate.getLoc(),
                                           operand.getType(), operand);
  tupleCreate->setOperand(operandIndex, publish.getResult());
  return true;
}

template <typename CallbackT>
static void forEachDirectPositionalOperand(CallVectorOp op,
                                           CallbackT &&callback) {
  auto kwnames = value::stripCasts(op.getKwnames());
  auto kwvalues = value::stripCasts(op.getKwvalues());
  if (!mlir::isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !mlir::isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  mlir::Value posargs = value::stripCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>()) {
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
  }
}

template <typename CallbackT>
static void forEachDirectPositionalOperand(CallOp op, CallbackT &&callback) {
  mlir::Value kwargs = value::stripCasts(op.getKwargs());
  if (!mlir::isa_and_nonnull<NoneOp>(kwargs.getDefiningOp()))
    return;

  mlir::Value posargs = value::stripCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>()) {
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
  }
}

template <typename CallbackT>
static void forEachDirectPositionalOperand(InvokeOp op, CallbackT &&callback) {
  auto kwnames = value::stripCasts(op.getKwnames());
  auto kwvalues = value::stripCasts(op.getKwvalues());
  if (!mlir::isa_and_nonnull<TupleEmptyOp>(kwnames.getDefiningOp()) ||
      !mlir::isa_and_nonnull<TupleEmptyOp>(kwvalues.getDefiningOp()))
    return;

  mlir::Value posargs = value::stripCasts(op.getPosargs());
  if (auto tupleCreate = posargs.getDefiningOp<TupleCreateOp>()) {
    for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands()))
      callback(static_cast<unsigned>(idx), operand);
  }
}

static llvm::SmallVector<unsigned, 4>
getPublicationArgsForDirectCallee(CallVectorOp op) {
  auto callee = call::pyFunc(op, op.getCallable());
  if (!callee)
    return {};
  llvm::SmallDenseSet<unsigned, 4> indices;
  auto collect = [&](mlir::ArrayAttr attr) {
    if (!attr)
      return;
    for (mlir::Attribute element : attr) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
      if (!intAttr)
        continue;
      int64_t argIndex = intAttr.getInt();
      if (argIndex < 0)
        continue;
      indices.insert(static_cast<unsigned>(argIndex));
    }
  };
  collect(callee->getAttrOfType<mlir::ArrayAttr>("ly.publishes_args"));
  collect(callee->getAttrOfType<mlir::ArrayAttr>("ly.captures_published"));
  llvm::SmallVector<unsigned, 4> result(indices.begin(), indices.end());
  llvm::sort(result);
  return result;
}

static llvm::SmallVector<unsigned, 4>
getPublicationArgsForDirectCallee(CallOp op) {
  auto callee = call::pyFunc(op, op.getCallable());
  if (!callee)
    return {};
  llvm::SmallDenseSet<unsigned, 4> indices;
  auto collect = [&](mlir::ArrayAttr attr) {
    if (!attr)
      return;
    for (mlir::Attribute element : attr) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
      if (!intAttr)
        continue;
      int64_t argIndex = intAttr.getInt();
      if (argIndex < 0)
        continue;
      indices.insert(static_cast<unsigned>(argIndex));
    }
  };
  collect(callee->getAttrOfType<mlir::ArrayAttr>("ly.publishes_args"));
  collect(callee->getAttrOfType<mlir::ArrayAttr>("ly.captures_published"));
  llvm::SmallVector<unsigned, 4> result(indices.begin(), indices.end());
  llvm::sort(result);
  return result;
}

static llvm::SmallVector<unsigned, 4>
getPublicationArgsForDirectCallee(InvokeOp op) {
  auto callee = call::pyFunc(op, op.getCallable());
  if (!callee)
    return {};
  llvm::SmallDenseSet<unsigned, 4> indices;
  auto collect = [&](mlir::ArrayAttr attr) {
    if (!attr)
      return;
    for (mlir::Attribute element : attr) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
      if (!intAttr)
        continue;
      int64_t argIndex = intAttr.getInt();
      if (argIndex < 0)
        continue;
      indices.insert(static_cast<unsigned>(argIndex));
    }
  };
  collect(callee->getAttrOfType<mlir::ArrayAttr>("ly.publishes_args"));
  collect(callee->getAttrOfType<mlir::ArrayAttr>("ly.captures_published"));
  llvm::SmallVector<unsigned, 4> result(indices.begin(), indices.end());
  llvm::sort(result);
  return result;
}

static void collectMakeFunctionPublicationSites(
    MakeFunctionOp op,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, unsigned>>
        &insertionSites,
    llvm::SmallVectorImpl<std::pair<TupleCreateOp, unsigned>>
        &tupleInsertionSites) {
  unsigned operandIndex = 0;

  auto collectTupleElements = [&](mlir::Value tupleValue) {
    mlir::Value root = value::stripCasts(tupleValue);
    if (auto tupleCreate = root.getDefiningOp<TupleCreateOp>()) {
      for (auto [idx, operand] : llvm::enumerate(tupleCreate.getOperands())) {
        if (!publication::tracks(operand.getType()))
          continue;
        tupleInsertionSites.emplace_back(tupleCreate,
                                         static_cast<unsigned>(idx));
      }
      return true;
    }
    return false;
  };

  if (mlir::Value defaults = op.getDefaults()) {
    if (!collectTupleElements(defaults))
      insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (mlir::Value kwdefaults = op.getKwdefaults()) {
    insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (mlir::Value closure = op.getClosure()) {
    if (!collectTupleElements(closure))
      insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (mlir::Value annotations = op.getAnnotations()) {
    insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
  if (mlir::Value moduleName = op.getModule()) {
    insertionSites.emplace_back(op.getOperation(), operandIndex);
    ++operandIndex;
  }
}

void publication::insertBoundaries(mlir::ModuleOp module) {
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 16> insertionSites;
  llvm::SmallVector<std::pair<TupleCreateOp, unsigned>, 16> tupleInsertionSites;

  module.walk([&](ListAppendOp op) {
    insertionSites.emplace_back(op.getOperation(), 1);
  });

  module.walk([&](DictInsertOp op) {
    insertionSites.emplace_back(op.getOperation(), 1);
    insertionSites.emplace_back(op.getOperation(), 2);
  });

  module.walk([&](AttrSetOp op) {
    if (class_state::published(op.getObject()))
      insertionSites.emplace_back(op.getOperation(), 1);
  });

  module.walk([&](ReturnOp op) {
    for (auto [operandIndex, operand] : llvm::enumerate(op.getOperands())) {
      if (!publication::tracks(operand.getType()))
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

void publication::compute(mlir::ModuleOp module) {
  bool changed = false;
  do {
    changed = false;

    module.walk([&](FuncOp func) {
      llvm::DenseSet<int> publishesArgs;
      llvm::DenseSet<int> capturesPublished;
      llvm::DenseSet<int> returnsPublished;
      llvm::DenseSet<int> readonlyArgs;
      llvm::DenseSet<int> mutableArgs;

      func.walk([&](ListAppendOp op) {
        int listArgIndex = publication::entryArg(func, op.getList());
        if (listArgIndex >= 0)
          mutableArgs.insert(listArgIndex);

        int argIndex = publication::entryArg(func, op.getValue());
        if (argIndex < 0)
          return;
        publishesArgs.insert(argIndex);
        capturesPublished.insert(argIndex);
      });

      func.walk([&](ListRemoveOp op) {
        int argIndex = publication::entryArg(func, op.getList());
        if (argIndex >= 0)
          mutableArgs.insert(argIndex);
      });

      func.walk([&](ListGetOp op) {
        int argIndex = publication::entryArg(func, op.getList());
        if (argIndex >= 0)
          readonlyArgs.insert(argIndex);
      });

      func.walk([&](DictInsertOp op) {
        int dictArgIndex = publication::entryArg(func, op.getDict());
        if (dictArgIndex >= 0)
          mutableArgs.insert(dictArgIndex);

        for (mlir::Value operand : {op.getKey(), op.getValue()}) {
          int argIndex = publication::entryArg(func, operand);
          if (argIndex < 0)
            continue;
          publishesArgs.insert(argIndex);
          capturesPublished.insert(argIndex);
        }
      });

      func.walk([&](DictGetOp op) {
        int argIndex = publication::entryArg(func, op.getDict());
        if (argIndex >= 0)
          readonlyArgs.insert(argIndex);
      });

      func.walk([&](AttrSetOp op) {
        if (!class_state::published(op.getObject()))
          return;
        int argIndex = publication::entryArg(func, op.getValue());
        if (argIndex < 0)
          return;
        publishesArgs.insert(argIndex);
        capturesPublished.insert(argIndex);
      });

      auto recordEscapingArgument = [&](mlir::Value value) {
        int argIndex = publication::entryArg(func, value);
        if (argIndex < 0)
          return;
        publishesArgs.insert(argIndex);
        capturesPublished.insert(argIndex);
      };

      func.walk([&](MakeFunctionOp op) {
        auto recordTupleElements = [&](mlir::Value tupleValue) {
          mlir::Value root = value::stripCasts(tupleValue);
          if (auto tupleCreate = root.getDefiningOp<TupleCreateOp>()) {
            for (mlir::Value element : tupleCreate.getElements()) {
              if (!publication::tracks(element.getType()))
                continue;
              recordEscapingArgument(element);
            }
            return true;
          }
          return false;
        };

        if (mlir::Value defaults = op.getDefaults())
          if (!recordTupleElements(defaults) &&
              publication::tracks(defaults.getType()))
            recordEscapingArgument(defaults);

        if (mlir::Value kwdefaults = op.getKwdefaults())
          if (publication::tracks(kwdefaults.getType()))
            recordEscapingArgument(kwdefaults);

        if (mlir::Value closure = op.getClosure())
          if (!recordTupleElements(closure) &&
              publication::tracks(closure.getType()))
            recordEscapingArgument(closure);

        if (mlir::Value annotations = op.getAnnotations())
          if (publication::tracks(annotations.getType()))
            recordEscapingArgument(annotations);
      });

      auto propagateFromCalleeSummary = [&](auto op, FuncOp callee) {
        if (!callee)
          return;

        llvm::DenseMap<unsigned, mlir::Value> operandByIndex;
        forEachDirectPositionalOperand(op,
                                       [&](unsigned idx, mlir::Value operand) {
                                         operandByIndex[idx] = operand;
                                       });
        if (operandByIndex.empty())
          return;

        auto recordPublicationIndices = [&](mlir::ArrayAttr indices,
                                            bool capture) {
          if (!indices)
            return;
          for (mlir::Attribute element : indices) {
            auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
            if (!intAttr)
              continue;
            int64_t formalIndex = intAttr.getInt();
            if (formalIndex < 0)
              continue;
            auto it = operandByIndex.find(static_cast<unsigned>(formalIndex));
            if (it == operandByIndex.end())
              continue;
            int argIndex = publication::entryArg(func, it->second);
            if (argIndex < 0)
              continue;
            publishesArgs.insert(argIndex);
            if (capture)
              capturesPublished.insert(argIndex);
          }
        };

        auto recordAccessIndices = [&](mlir::ArrayAttr indices,
                                       llvm::DenseSet<int> &target) {
          if (!indices)
            return;
          for (mlir::Attribute element : indices) {
            auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
            if (!intAttr)
              continue;
            int64_t formalIndex = intAttr.getInt();
            if (formalIndex < 0)
              continue;
            auto it = operandByIndex.find(static_cast<unsigned>(formalIndex));
            if (it == operandByIndex.end())
              continue;
            int argIndex = publication::entryArg(func, it->second);
            if (argIndex >= 0)
              target.insert(argIndex);
          }
        };

        recordPublicationIndices(
            callee->getAttrOfType<mlir::ArrayAttr>("ly.publishes_args"),
            /*capture=*/false);
        recordPublicationIndices(
            callee->getAttrOfType<mlir::ArrayAttr>("ly.captures_published"),
            /*capture=*/true);
        recordAccessIndices(
            callee->getAttrOfType<mlir::ArrayAttr>("ly.readonly_args"),
            readonlyArgs);
        recordAccessIndices(
            callee->getAttrOfType<mlir::ArrayAttr>("ly.mutable_args"),
            mutableArgs);
      };

      func.walk([&](CallVectorOp op) {
        propagateFromCalleeSummary(op, call::pyFunc(op, op.getCallable()));
      });

      func.walk([&](CallOp op) {
        propagateFromCalleeSummary(op, call::pyFunc(op, op.getCallable()));
      });

      func.walk([&](InvokeOp op) {
        propagateFromCalleeSummary(op, call::pyFunc(op, op.getCallable()));
      });

      func.walk([&](ReturnOp op) {
        for (auto [resultIndex, operand] : llvm::enumerate(op.getOperands())) {
          if (!publication::tracks(operand.getType()))
            continue;
          if (mlir::isa<ClassType>(operand.getType()) &&
              !class_state::published(operand))
            continue;
          returnsPublished.insert(static_cast<int>(resultIndex));
        }
      });

      for (int argIndex : mutableArgs)
        readonlyArgs.erase(argIndex);

      changed |=
          publication::update(func, publishesArgs, capturesPublished,
                              returnsPublished, readonlyArgs, mutableArgs);
    });
  } while (changed);
}

bool class_state::local(mlir::Value value) {
  value = ::py::optimizer::value::stripCasts(value);

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto *owner = arg.getOwner();
    auto func =
        owner ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(owner->getParentOp())
              : nullptr;
    if (!func)
      return false;
    return arg.getArgNumber() == 0 &&
           (static_cast<bool>(func->getAttr("ly.zero_initialized_self")) ||
            static_cast<bool>(func->getAttr("ly.local_self_arg0")));
  }

  if (!mlir::isa<ClassType>(value.getType()))
    return false;

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (mlir::isa<ClassNewOp>(def))
    return true;
  if (mlir::isa<PublishOp, ClassPromoteOp, ListGetOp>(def))
    return false;
  return false;
}

bool class_state::fresh(mlir::Value value) {
  value = ::py::optimizer::value::stripCasts(value);

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto *owner = arg.getOwner();
    auto func =
        owner ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(owner->getParentOp())
              : nullptr;
    return func && arg.getArgNumber() == 0 &&
           static_cast<bool>(func->getAttr("ly.zero_initialized_self"));
  }

  return mlir::isa_and_nonnull<ClassNewOp>(value.getDefiningOp());
}

bool class_state::published(mlir::Value value) {
  value = ::py::optimizer::value::stripCasts(value);
  if (!mlir::isa<ClassType>(value.getType()))
    return false;

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = arg.getOwner();
    if (!owner)
      return false;
    mlir::Block *pred = owner->getUniquePredecessor();
    if (!pred)
      return false;
    auto invoke = mlir::dyn_cast_or_null<InvokeOp>(pred->getTerminator());
    if (!invoke || invoke.getNormalDest() != owner)
      return false;
    auto callee = call::pyFunc(invoke, invoke.getCallable());
    return publication::result(callee, arg.getArgNumber());
  }

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (mlir::isa<PublishOp, ClassPromoteOp, ListGetOp>(def))
    return true;

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return false;

  if (auto call = mlir::dyn_cast<CallOp>(def))
    return publication::result(
        ::py::optimizer::call::pyFunc(call, call.getCallable()),
        result.getResultNumber());

  if (auto call = mlir::dyn_cast<CallVectorOp>(def))
    return publication::result(
        ::py::optimizer::call::pyFunc(call, call.getCallable()),
        result.getResultNumber());

  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(def)) {
    mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return false;
    return publication::result(
        module.lookupSymbol<mlir::func::FuncOp>(call.getCallee()),
        result.getResultNumber());
  }

  return false;
}

mlir::func::FuncOp call::localSelfHelper(mlir::func::FuncOp callee,
                                         mlir::ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<mlir::SymbolRefAttr>("ly.local_self_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(helperName);
}

mlir::func::FuncOp call::freshInitHelper(mlir::func::FuncOp callee,
                                         mlir::ModuleOp module) {
  auto helperAttr =
      callee->getAttrOfType<mlir::SymbolRefAttr>("ly.fresh_init_helper");
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(helperName);
}

std::string call::publishedBorrowAttr(unsigned argIndex) {
  return ::py::publication::borrow::Attr::name(argIndex);
}

mlir::func::FuncOp call::publishedBorrowHelper(mlir::func::FuncOp callee,
                                               unsigned argIndex,
                                               mlir::ModuleOp module) {
  auto helperAttr = callee->getAttrOfType<mlir::SymbolRefAttr>(
      call::publishedBorrowAttr(argIndex));
  if (!helperAttr)
    return nullptr;
  auto helperName = helperAttr.getLeafReference().empty()
                        ? helperAttr.getRootReference().getValue()
                        : helperAttr.getLeafReference().getValue();
  return module.lookupSymbol<mlir::func::FuncOp>(helperName);
}

mlir::func::FuncOp call::preferredTarget(mlir::func::FuncOp callee,
                                         mlir::func::CallOp call,
                                         mlir::ModuleOp module) {
  mlir::func::FuncOp preferredTarget = callee;

  if (call.getNumOperands() != 0) {
    if (class_state::fresh(call.getOperand(0))) {
      if (auto freshHelper =
              ::py::optimizer::call::freshInitHelper(preferredTarget, module))
        preferredTarget = freshHelper;
    }
    if (preferredTarget == callee && class_state::local(call.getOperand(0))) {
      if (auto localHelper =
              ::py::optimizer::call::localSelfHelper(preferredTarget, module))
        preferredTarget = localHelper;
    }
  }

  for (unsigned idx = 1; idx < call.getNumOperands(); ++idx) {
    if (!class_state::published(call.getOperand(idx)))
      continue;
    if (auto publishedHelper = ::py::optimizer::call::publishedBorrowHelper(
            preferredTarget, idx, module)) {
      preferredTarget = publishedHelper;
    }
  }

  return preferredTarget;
}

void call::rewritePreferred(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::func::CallOp> toRewrite;

  module.walk([&](mlir::func::CallOp call) {
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
    if (!callee)
      return;

    auto preferredHelper =
        ::py::optimizer::call::preferredTarget(callee, call, module);
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

  for (mlir::func::CallOp call : toRewrite) {
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
    if (!callee)
      continue;
    auto preferredHelper =
        ::py::optimizer::call::preferredTarget(callee, call, module);
    if (!preferredHelper || preferredHelper == callee)
      continue;
    mlir::OpBuilder builder(call);
    auto replacement = builder.create<mlir::func::CallOp>(
        call.getLoc(), preferredHelper, call.getOperands());
    call.replaceAllUsesWith(replacement.getResults());
    call.erase();
  }
}

void class_state::eliminatePublishes(mlir::ModuleOp module) {
  llvm::SmallVector<PublishOp> toErase;

  module.walk([&](PublishOp op) {
    if (!mlir::isa<ClassType>(op.getResult().getType()))
      return;
    if (!class_state::published(op.getInput()))
      return;

    op.getResult().replaceAllUsesWith(op.getInput());
    toErase.push_back(op);
  });

  for (PublishOp op : toErase)
    op.erase();
}

static bool isZeroCostLocalClassFieldCandidate(mlir::Type fieldType) {
  return !mlir::isa<ClassType, ListType, TupleType, DictType>(fieldType);
}

void class_state::proveLocalAccess(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation *op) {
    mlir::Value object;
    mlir::Type fieldType;
    if (auto attrGet = mlir::dyn_cast<AttrGetOp>(op)) {
      object = attrGet.getObject();
      fieldType = attrGet.getResult().getType();
    } else if (auto attrSet = mlir::dyn_cast<AttrSetOp>(op)) {
      object = attrSet.getObject();
      fieldType = attrSet.getValue().getType();
    } else {
      return;
    }

    if (!class_state::local(object))
      return;
    if (!isZeroCostLocalClassFieldCandidate(fieldType))
      return;

    op->setAttr("ly.proven_local_class_access",
                mlir::UnitAttr::get(module.getContext()));
  });
}

void class_state::markFirstStores(mlir::ModuleOp module) {
  module.walk([&](AttrSetOp op) {
    auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
    if (!parentFunc || !parentFunc->getAttr("ly.zero_initialized_self"))
      return;
    if (!parentFunc.getBody().hasOneBlock())
      return;

    mlir::Value objectRoot = value::stripCasts(op.getObject());
    auto selfArg = mlir::dyn_cast<mlir::BlockArgument>(objectRoot);
    if (!selfArg || selfArg.getOwner() != &parentFunc.getBody().front() ||
        selfArg.getArgNumber() != 0)
      return;

    for (mlir::Operation *cursor = op->getPrevNode(); cursor;
         cursor = cursor->getPrevNode()) {
      bool touchesSelf =
          llvm::any_of(cursor->getOperands(), [&](mlir::Value operand) {
            return value::stripCasts(operand) == objectRoot;
          });
      if (!touchesSelf)
        continue;

      if (auto prevSet = mlir::dyn_cast<AttrSetOp>(cursor)) {
        if (value::stripCasts(prevSet.getObject()) == objectRoot &&
            prevSet.getNameAttr() == op.getNameAttr())
          return;
        continue;
      }

      if (mlir::isa<mlir::UnrealizedConversionCastOp, AttrGetOp>(cursor))
        continue;

      return;
    }

    op->setAttr("ly.zero_init_first_store",
                mlir::UnitAttr::get(module.getContext()));
  });
}

/// Hoist integer constants to entry block and perform CSE.
void scalar::hoistInts(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    if (func.isExternal())
      return;

    mlir::Block &entryBlock = func.getBody().front();
    llvm::StringMap<IntConstantOp> constantMap;

    // First pass: collect all IntConstantOp
    llvm::SmallVector<IntConstantOp> allConstants;
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
void scalar::dropSmallIntDecrefs(mlir::ModuleOp module) {
  llvm::SmallVector<DecRefOp> toErase;

  module.walk([&](DecRefOp decref) {
    mlir::Value obj = decref.getObject();
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

/// CSE for singleton getter calls (Ly_GetBuiltinPrint, Ly_GetNone). These
/// return borrowed references to singletons, so no DecRef needed.
void scalar::cseSingletons(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    if (func.isExternal())
      return;

    // Cache for each singleton getter function
    llvm::StringMap<mlir::LLVM::CallOp> singletonCache;
    llvm::SmallVector<mlir::LLVM::CallOp> toErase;

    func.walk([&](mlir::LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee)
        return;

      // Singleton getters that we want to CSE
      if (*callee != "Ly_GetBuiltinPrint" && *callee != "Ly_GetNone")
        return;

      llvm::StringRef funcName = *callee;
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
void scalar::elideBoolBoxing(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::LLVM::CallOp> fromBoolCalls;

  // First pass: find all LyBool_FromBool calls
  module.walk([&](mlir::LLVM::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (callee && *callee == "LyBool_FromBool")
      fromBoolCalls.push_back(callOp);
  });

  // Second pass: check each FromBool call for the pattern
  for (auto fromBoolCall : fromBoolCalls) {
    if (fromBoolCall.getNumOperands() != 1)
      continue;

    mlir::Value i1Value = fromBoolCall.getOperand(0);
    mlir::Value boxedValue = fromBoolCall.getResult();

    // Find users of the boxed value
    mlir::LLVM::CallOp asBoolCall = nullptr;
    mlir::LLVM::CallOp decRefCall = nullptr;
    bool hasOtherUsers = false;

    for (mlir::Operation *user : boxedValue.getUsers()) {
      if (auto callUser = mlir::dyn_cast<mlir::LLVM::CallOp>(user)) {
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
void scalar::elideLongBoxing(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::LLVM::CallOp> fromI64Calls;

  module.walk([&](mlir::LLVM::CallOp callOp) {
    if (runtime::Call::is(callOp, RuntimeSymbols::kLongFromI64))
      fromI64Calls.push_back(callOp);
  });

  for (auto fromI64Call : fromI64Calls) {
    if (fromI64Call.getNumOperands() != 1)
      continue;

    mlir::Value i64Value = fromI64Call.getOperand(0);
    mlir::Value boxedValue = fromI64Call.getResult();

    mlir::LLVM::CallOp asI64Call = nullptr;
    mlir::LLVM::CallOp decRefCall = nullptr;
    bool hasOtherUsers = false;

    for (mlir::Operation *user : boxedValue.getUsers()) {
      auto callUser = mlir::dyn_cast<mlir::LLVM::CallOp>(user);
      if (!callUser) {
        hasOtherUsers = true;
        break;
      }

      if (runtime::Call::is(callUser, RuntimeSymbols::kLongAsI64) &&
          !asI64Call) {
        asI64Call = callUser;
      } else if (runtime::Call::is(callUser, RuntimeSymbols::kDecRef) &&
                 !decRefCall) {
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
void scalar::elideLongArithRoundTrips(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::LLVM::CallOp> asI64Calls;

  module.walk([&](mlir::LLVM::CallOp callOp) {
    if (runtime::Call::is(callOp, RuntimeSymbols::kLongAsI64))
      asI64Calls.push_back(callOp);
  });

  for (auto asI64Call : asI64Calls) {
    if (asI64Call.getNumOperands() != 1)
      continue;

    auto arithmeticCall =
        asI64Call.getOperand(0).getDefiningOp<mlir::LLVM::CallOp>();
    if (!arithmeticCall || arithmeticCall.getNumOperands() != 2)
      continue;

    bool isAdd = runtime::Call::is(arithmeticCall, RuntimeSymbols::kLongAdd);
    bool isSub = runtime::Call::is(arithmeticCall, RuntimeSymbols::kLongSub);
    if (!isAdd && !isSub)
      continue;

    mlir::Value boxedResult = arithmeticCall.getResult();
    mlir::LLVM::CallOp decRefCall = nullptr;
    bool hasOtherUsers = false;
    for (mlir::Operation *user : boxedResult.getUsers()) {
      auto callUser = mlir::dyn_cast<mlir::LLVM::CallOp>(user);
      if (!callUser) {
        hasOtherUsers = true;
        break;
      }

      if (callUser == asI64Call) {
        continue;
      }
      if (runtime::Call::is(callUser, RuntimeSymbols::kDecRef) && !decRefCall) {
        decRefCall = callUser;
      } else {
        hasOtherUsers = true;
        break;
      }
    }

    if (hasOtherUsers)
      continue;

    mlir::OpBuilder builder(asI64Call);
    mlir::Location loc = asI64Call.getLoc();
    mlir::Value lhsBoxed = arithmeticCall.getOperand(0);
    mlir::Value rhsBoxed = arithmeticCall.getOperand(1);

    mlir::Value lhs = runtime::Long::asI64(module, builder, loc, lhsBoxed);
    mlir::Value rhs = runtime::Long::asI64(module, builder, loc, rhsBoxed);

    mlir::Value arithmeticResult;
    if (isAdd) {
      arithmeticResult =
          builder.create<mlir::LLVM::AddOp>(loc, lhs.getType(), lhs, rhs)
              .getResult();
    } else {
      arithmeticResult =
          builder.create<mlir::LLVM::SubOp>(loc, lhs.getType(), lhs, rhs)
              .getResult();
    }

    asI64Call.getResult().replaceAllUsesWith(arithmeticResult);

    if (decRefCall)
      decRefCall->erase();
    asI64Call->erase();
    arithmeticCall->erase();

    if (auto fromI64Call = lhsBoxed.getDefiningOp<mlir::LLVM::CallOp>();
        fromI64Call &&
        runtime::Call::is(fromI64Call, RuntimeSymbols::kLongFromI64)) {
      runtime::Call::eraseWithOptionalRelease(fromI64Call);
    }
    if (auto fromI64Call = rhsBoxed.getDefiningOp<mlir::LLVM::CallOp>();
        fromI64Call &&
        runtime::Call::is(fromI64Call, RuntimeSymbols::kLongFromI64)) {
      runtime::Call::eraseWithOptionalRelease(fromI64Call);
    }
  }
}

/// CSE for LyLong_FromI64 for small integers (-5 to 256).
/// Small integers are immortal, so sharing is safe.
void scalar::cseSmallInts(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    if (func.isExternal())
      return;

    llvm::DenseMap<int64_t, mlir::LLVM::CallOp> cache;
    llvm::SmallVector<mlir::LLVM::CallOp> toErase;

    func.walk([&](mlir::LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee || *callee != "LyLong_FromI64")
        return;
      if (callOp.getNumOperands() != 1)
        return;
      auto constOp =
          callOp.getOperand(0).getDefiningOp<mlir::LLVM::ConstantOp>();
      if (!constOp)
        return;
      auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
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
void scalar::cseConstants(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    if (func.isExternal())
      return;

    // Map from (type, value) -> first constant op
    llvm::DenseMap<std::pair<mlir::Type, mlir::Attribute>,
                   mlir::LLVM::ConstantOp>
        constantCache;
    llvm::SmallVector<mlir::LLVM::ConstantOp> toErase;

    func.walk([&](mlir::LLVM::ConstantOp constOp) {
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

static bool hasOnlyTensorResults(mlir::Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  return llvm::all_of(op->getResultTypes(), [](mlir::Type type) {
    return mlir::isa<mlir::RankedTensorType>(type);
  });
}

static bool isDeadCodeCandidate(mlir::Operation *op) {
  if (mlir::isa<mlir::LLVM::AddressOfOp, mlir::LLVM::GEPOp,
                mlir::LLVM::ConstantOp>(op))
    return true;
  if (mlir::isa<mlir::tensor::EmptyOp, mlir::tensor::FromElementsOp>(op))
    return true;
  if (mlir::isa<mlir::arith::ConstantOp, mlir::arith::AddFOp,
                mlir::arith::SubFOp, mlir::arith::MulFOp>(op))
    return hasOnlyTensorResults(op);
  if (mlir::isa<mlir::linalg::FillOp, mlir::linalg::GenericOp>(op))
    return hasOnlyTensorResults(op);
  return false;
}

/// Dead code elimination for unused operations after CSE/lowering.
/// This cleans up LLVM address constants and tensor/linalg producers that
/// became dead after compile-time tensor repr materialization.
void scalar::dce(mlir::ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> toErase;

    module.walk([&](mlir::Operation *op) {
      // Only consider operations with no side effects
      if (!isDeadCodeCandidate(op))
        return;

      // Check if all results are unused
      bool allUnused = true;
      for (mlir::Value result : op->getResults()) {
        if (!result.use_empty()) {
          allUnused = false;
          break;
        }
      }
      if (allUnused)
        toErase.push_back(op);
    });

    for (mlir::Operation *op : toErase) {
      op->erase();
      changed = true;
    }
  }
}

} // namespace py::optimizer
