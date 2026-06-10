// This file implements shared optimization helpers specific to the Py dialect
// and its lowered LLVM representation. These optimizations include:
//   - Dead tuple cleanup (removing tuples only used by DecRefOp)
//   - Integer constant hoisting and CSE
//   - LLVM constant CSE
//   - Dead code elimination for unused LLVM/tensor producer operations

#include "Optimizer/Utils.h"

#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <optional>

namespace py::optimizer {

mlir::Value value::stripCasts(mlir::Value value);

template <typename OpT>
static OpT findDominatingCached(const llvm::SmallVectorImpl<OpT> &candidates,
                                OpT current, mlir::DominanceInfo &dominance) {
  for (OpT candidate : candidates)
    if (dominance.dominates(candidate.getOperation(), current.getOperation()))
      return candidate;
  return {};
}

bool runtime::Call::is(mlir::LLVM::CallOp callOp, llvm::StringRef calleeName) {
  auto callee = callOp.getCallee();
  return callee && *callee == calleeName;
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

bool runtime::Func::eraseUnusedDecls(mlir::ModuleOp module) {
  llvm::DenseSet<llvm::StringRef> referenced;
  module.walk([&](mlir::LLVM::CallOp call) {
    if (auto callee = call.getCallee())
      referenced.insert(*callee);
  });
  module.walk([&](mlir::LLVM::InvokeOp invoke) {
    if (auto callee = invoke.getCallee())
      referenced.insert(*callee);
  });
  module.walk([&](mlir::LLVM::AddressOfOp addressOf) {
    referenced.insert(addressOf.getGlobalName());
  });
  module.walk([&](mlir::LLVM::LLVMFuncOp fn) {
    auto personality =
        fn->getAttrOfType<mlir::FlatSymbolRefAttr>("personality");
    if (personality)
      referenced.insert(personality.getValue());
  });

  llvm::SmallVector<mlir::LLVM::LLVMFuncOp> unused;
  module.walk([&](mlir::LLVM::LLVMFuncOp fn) {
    if (!fn.getBody().empty())
      return;
    if (referenced.contains(fn.getSymName()))
      return;
    unused.push_back(fn);
  });

  for (mlir::LLVM::LLVMFuncOp fn : unused)
    fn.erase();
  return !unused.empty();
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
    if (!class_state::local(object))
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
    if (mlir::isa<ClassType, ListType, TupleType, DictType>(value.getType()))
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
  return mlir::isa<ListType, DictType, TupleType, ClassType>(type);
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

static bool
collectStaticTupleElements(mlir::Value value,
                           llvm::SmallVectorImpl<mlir::Value> &elements) {
  value = value::stripCasts(value);
  if (mlir::isa_and_nonnull<TupleEmptyOp>(value.getDefiningOp()))
    return true;
  if (auto tuple = value.getDefiningOp<TupleCreateOp>()) {
    elements.append(tuple.getElements().begin(), tuple.getElements().end());
    return true;
  }
  return false;
}

static mlir::Value stripStaticMetadataValue(mlir::Value value) {
  while (true) {
    value = value::stripPublications(value);
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

static void eraseDeadStaticMetadataTree(mlir::Value value) {
  if (!value)
    return;

  if (auto publish = value.getDefiningOp<PublishOp>()) {
    mlir::Value input = publish.getInput();
    if (publish->use_empty())
      publish.erase();
    eraseDeadStaticMetadataTree(input);
    return;
  }

  if (auto tuple = value.getDefiningOp<TupleCreateOp>()) {
    if (!tuple->use_empty())
      return;
    llvm::SmallVector<mlir::Value, 8> elements(tuple.getElements().begin(),
                                               tuple.getElements().end());
    tuple.erase();
    for (mlir::Value element : elements)
      eraseDeadStaticMetadataTree(element);
    return;
  }

  if (auto tuple = value.getDefiningOp<TupleEmptyOp>()) {
    if (tuple->use_empty())
      tuple.erase();
    return;
  }

  if (auto dict = value.getDefiningOp<DictEmptyOp>()) {
    llvm::SmallVector<mlir::Value, 8> operands;
    llvm::SmallVector<DictInsertOp, 8> inserts;
    for (mlir::Operation *user : dict.getResult().getUsers()) {
      auto insert = mlir::dyn_cast<DictInsertOp>(user);
      if (!insert || insert.getDict() != dict.getResult())
        return;
      operands.push_back(insert.getKey());
      operands.push_back(insert.getValue());
      inserts.push_back(insert);
    }
    for (DictInsertOp insert : inserts)
      insert.erase();
    if (dict->use_empty())
      dict.erase();
    for (mlir::Value operand : operands)
      eraseDeadStaticMetadataTree(operand);
    return;
  }

  mlir::Operation *producer = value.getDefiningOp();
  if (mlir::isa_and_nonnull<StrConstantOp, IntConstantOp, FloatConstantOp>(
          producer) &&
      producer->use_empty())
    producer->erase();
}

static llvm::StringRef symbolName(mlir::FlatSymbolRefAttr symbol) {
  if (!symbol)
    return {};
  return symbol.getValue();
}

static bool isPureReturnedCallableCloneOp(mlir::Operation *op) {
  return mlir::isa_and_nonnull<
      mlir::arith::ConstantOp, CastFromPrimOp, CastToPrimOp, TupleCreateOp,
      TupleEmptyOp, NoneOp, IntConstantOp, FloatConstantOp, StrConstantOp>(op);
}

static bool returnedCallableBodyIsPure(FuncOp func, MakeFunctionOp makeFunc,
                                       ReturnOp returnOp) {
  if (!func || !makeFunc || !returnOp || !func.getBody().hasOneBlock())
    return false;
  for (mlir::Operation &op : func.getBody().front()) {
    if (&op == makeFunc.getOperation() || &op == returnOp.getOperation())
      continue;
    if (!isPureReturnedCallableCloneOp(&op))
      return false;
  }
  return true;
}

static MakeFunctionOp findReturnedMakeFunction(FuncOp func,
                                               llvm::StringRef targetName) {
  if (!func || !func.getBody().hasOneBlock())
    return {};

  ReturnOp returnOp;
  for (mlir::Operation &op : func.getBody().front()) {
    if (auto candidate = mlir::dyn_cast<ReturnOp>(op)) {
      if (returnOp)
        return {};
      returnOp = candidate;
    }
  }
  if (!returnOp || returnOp->getNumOperands() != 1)
    return {};

  mlir::Value returned = value::stripCasts(returnOp->getOperand(0));
  auto makeFunc = returned.getDefiningOp<MakeFunctionOp>();
  if (!makeFunc || symbolName(makeFunc.getTargetAttr()) != targetName)
    return {};
  if (!returnedCallableBodyIsPure(func, makeFunc, returnOp))
    return {};
  return makeFunc;
}

static mlir::FailureOr<mlir::Value>
clonePureReturnedCallableValue(mlir::Value value, mlir::OpBuilder &builder,
                               mlir::IRMapping &mapping) {
  if (!value)
    return mlir::Value();
  if (mlir::Value mapped = mapping.lookupOrNull(value))
    return mapped;

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    (void)blockArg;
    return mlir::failure();
  }

  mlir::Operation *def = value.getDefiningOp();
  if (!isPureReturnedCallableCloneOp(def) || def->getNumRegions() != 0)
    return mlir::failure();

  for (mlir::Value operand : def->getOperands())
    if (mlir::failed(clonePureReturnedCallableValue(operand, builder, mapping)))
      return mlir::failure();

  mlir::Operation *cloned = builder.clone(*def, mapping);
  for (auto [original, copy] :
       llvm::zip(def->getResults(), cloned->getResults()))
    mapping.map(original, copy);
  return mapping.lookup(value);
}

static bool mapReturnedCallableArguments(CallVectorOp sourceCall, FuncOp source,
                                         mlir::IRMapping &mapping) {
  if (!source || !source.getBody().hasOneBlock())
    return false;
  if (!mlir::isa<TupleEmptyOp>(
          value::stripCasts(sourceCall.getKwnames()).getDefiningOp()) ||
      !mlir::isa<TupleEmptyOp>(
          value::stripCasts(sourceCall.getKwvalues()).getDefiningOp()))
    return false;

  llvm::SmallVector<mlir::Value, 8> args;
  if (!collectStaticTupleElements(sourceCall.getPosargs(), args))
    return false;

  mlir::Block &entry = source.getBody().front();
  if (entry.getNumArguments() != args.size())
    return false;
  for (auto [arg, value] : llvm::zip(entry.getArguments(), args))
    mapping.map(arg, value);
  return true;
}

static mlir::FailureOr<mlir::Value> cloneOptionalReturnedCallableMetadata(
    mlir::Value value, mlir::OpBuilder &builder, mlir::IRMapping &mapping) {
  if (!value)
    return mlir::Value();
  return clonePureReturnedCallableValue(value, builder, mapping);
}

static bool materializeReturnedCallable(CallVectorOp call) {
  mlir::Value callable = value::stripCasts(call.getCallable());
  auto sourceCall = callable.getDefiningOp<CallVectorOp>();
  if (!sourceCall)
    return false;
  if (!llvm::hasSingleElement(sourceCall.getResult(0).getUsers()))
    return false;

  auto returnedSymbol = sourceCall->getAttrOfType<mlir::FlatSymbolRefAttr>(
      "ly.returned_callable_symbol");
  llvm::StringRef targetName = symbolName(returnedSymbol);
  if (targetName.empty())
    return false;

  FuncOp sourceFunc =
      call::pyFunc(sourceCall.getOperation(), sourceCall.getCallable());
  MakeFunctionOp sourceMakeFunc =
      findReturnedMakeFunction(sourceFunc, targetName);
  if (!sourceMakeFunc)
    return false;

  mlir::OpBuilder builder(call);
  mlir::IRMapping mapping;
  if (!mapReturnedCallableArguments(sourceCall, sourceFunc, mapping))
    return false;

  mlir::FailureOr<mlir::Value> defaults = cloneOptionalReturnedCallableMetadata(
      sourceMakeFunc.getDefaults(), builder, mapping);
  if (mlir::failed(defaults))
    return false;
  mlir::FailureOr<mlir::Value> kwdefaults =
      cloneOptionalReturnedCallableMetadata(sourceMakeFunc.getKwdefaults(),
                                            builder, mapping);
  if (mlir::failed(kwdefaults))
    return false;
  mlir::FailureOr<mlir::Value> closure = cloneOptionalReturnedCallableMetadata(
      sourceMakeFunc.getClosure(), builder, mapping);
  if (mlir::failed(closure))
    return false;
  mlir::FailureOr<mlir::Value> annotations =
      cloneOptionalReturnedCallableMetadata(sourceMakeFunc.getAnnotations(),
                                            builder, mapping);
  if (mlir::failed(annotations))
    return false;
  mlir::FailureOr<mlir::Value> moduleName =
      cloneOptionalReturnedCallableMetadata(sourceMakeFunc.getModule(), builder,
                                            mapping);
  if (mlir::failed(moduleName))
    return false;

  auto materialized = builder.create<MakeFunctionOp>(
      call.getLoc(), call.getCallable().getType(),
      sourceMakeFunc.getTargetAttr(), *defaults, *kwdefaults, *closure,
      *annotations, *moduleName);
  call.getCallableMutable().assign(materialized.getResult());

  mlir::Value oldPosargs = sourceCall.getPosargs();
  mlir::Value oldKwnames = sourceCall.getKwnames();
  mlir::Value oldKwvalues = sourceCall.getKwvalues();
  sourceCall.erase();
  eraseDeadStaticMetadataTree(oldPosargs);
  eraseDeadStaticMetadataTree(oldKwnames);
  eraseDeadStaticMetadataTree(oldKwvalues);
  return true;
}

static void materializeReturnedCallables(mlir::ModuleOp module) {
  llvm::SmallVector<CallVectorOp> calls;
  module.walk([&](CallVectorOp op) { calls.push_back(op); });
  for (CallVectorOp call : calls)
    materializeReturnedCallable(call);
}

void call::staticDefaults(mlir::ModuleOp module) {
  materializeReturnedCallables(module);

  llvm::SmallVector<CallVectorOp> calls;
  module.walk([&](CallVectorOp op) { calls.push_back(op); });

  for (CallVectorOp call : calls) {
    mlir::Value callable = value::stripCasts(call.getCallable());
    auto makeFunc = callable.getDefiningOp<MakeFunctionOp>();
    if (!makeFunc)
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

    llvm::StringMap<mlir::Value> explicitKwValues;
    if (!collectStaticKeywordArgumentValues(
            call.getKwnames(), call.getKwvalues(), explicitKwValues))
      continue;
    bool hasExplicitKeywords = !explicitKwValues.empty();
    if (!makeFunc.getDefaults() && !makeFunc.getKwdefaults() &&
        !makeFunc.getClosure() && !hasExplicitKeywords)
      continue;

    llvm::SmallVector<mlir::Value, 8> providedElements;
    if (!collectStaticTupleElements(call.getPosargs(), providedElements))
      continue;
    unsigned providedCount = static_cast<unsigned>(providedElements.size());
    if (providedCount > positionalCount)
      continue;

    FuncOp targetFunc = nullptr;
    if (mlir::Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
            call, makeFunc.getTargetAttr()))
      targetFunc = mlir::dyn_cast<FuncOp>(symbol);

    llvm::StringMap<unsigned> positionalNameToIndex;
    if (targetFunc) {
      mlir::ArrayAttr argNames = targetFunc.getArgNamesAttr();
      if (argNames && argNames.size() == positionalCount) {
        for (auto [index, attr] : llvm::enumerate(argNames)) {
          auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
          if (!nameAttr)
            continue;
          positionalNameToIndex[nameAttr.getValue()] = index;
        }
      }
    }

    llvm::StringMap<unsigned> kwonlyNameToIndex;
    llvm::SmallVector<llvm::StringRef, 8> kwonlyNames;
    if (kwonlyCount > 0) {
      if (!targetFunc)
        continue;
      mlir::ArrayAttr kwonlyNamesAttr = targetFunc.getKwonlyNamesAttr();
      if (!kwonlyNamesAttr || kwonlyNamesAttr.size() != kwonlyCount)
        continue;
      kwonlyNames.reserve(kwonlyCount);
      for (auto [index, attr] : llvm::enumerate(kwonlyNamesAttr)) {
        auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!nameAttr)
          continue;
        kwonlyNames.push_back(nameAttr.getValue());
        kwonlyNameToIndex[nameAttr.getValue()] = index;
      }
      if (kwonlyNames.size() != kwonlyCount)
        continue;
    }

    llvm::SmallVector<mlir::Value, 8> positionalElements(positionalCount);
    for (auto [index, element] : llvm::enumerate(providedElements))
      positionalElements[index] = element;

    bool invalidKeyword = false;
    llvm::StringSet<> consumedKeywords;
    for (auto &entry : explicitKwValues) {
      auto positionalIt = positionalNameToIndex.find(entry.getKey());
      if (positionalIt == positionalNameToIndex.end())
        continue;
      unsigned index = positionalIt->second;
      if (index < providedCount || positionalElements[index]) {
        invalidKeyword = true;
        break;
      }
      positionalElements[index] = entry.getValue();
      consumedKeywords.insert(entry.getKey());
    }
    if (invalidKeyword)
      continue;

    if (llvm::any_of(positionalElements,
                     [](mlir::Value value) { return !value; })) {
      llvm::SmallVector<mlir::Value, 8> defaultElements;
      bool hasStaticDefaults =
          makeFunc.getDefaults() &&
          collectStaticTupleElements(makeFunc.getDefaults(), defaultElements);
      unsigned defaultsCount =
          hasStaticDefaults ? static_cast<unsigned>(defaultElements.size()) : 0;
      if (defaultsCount > positionalCount)
        continue;

      unsigned firstDefaultIndex = positionalCount - defaultsCount;
      for (unsigned index = providedCount; index < positionalCount; ++index)
        if (!positionalElements[index] && index >= firstDefaultIndex)
          positionalElements[index] =
              defaultElements[index - firstDefaultIndex];
      if (llvm::any_of(positionalElements,
                       [](mlir::Value value) { return !value; }))
        continue;
    }

    llvm::SmallVector<mlir::Value, 8> elements;
    elements.append(positionalElements.begin(), positionalElements.end());

    if (kwonlyCount > 0) {
      llvm::StringMap<mlir::Value> kwdefaultValues;
      if (makeFunc.getKwdefaults() &&
          !collectStaticDictStringValues(makeFunc.getKwdefaults(),
                                         makeFunc.getOperation(),
                                         kwdefaultValues))
        continue;
      bool missingDefault = false;
      for (llvm::StringRef name : kwonlyNames) {
        auto explicitIt = explicitKwValues.find(name);
        if (explicitIt != explicitKwValues.end()) {
          elements.push_back(explicitIt->second);
          consumedKeywords.insert(name);
          continue;
        }
        auto defaultIt = kwdefaultValues.find(name);
        if (defaultIt == kwdefaultValues.end()) {
          missingDefault = true;
          break;
        }
        elements.push_back(defaultIt->second);
      }
      if (missingDefault)
        continue;
    }

    if (consumedKeywords.size() != explicitKwValues.size())
      continue;

    if (mlir::Value closureValue = makeFunc.getClosure()) {
      llvm::SmallVector<mlir::Value, 8> closureElements;
      if (!collectStaticTupleElements(closureValue, closureElements))
        continue;
      elements.append(closureElements.begin(), closureElements.end());
    }

    mlir::Value oldPosargs = call.getPosargs();
    mlir::Value oldKwnames = call.getKwnames();
    mlir::Value oldKwvalues = call.getKwvalues();
    mlir::Value defaults = makeFunc.getDefaults();
    mlir::Value kwdefaults = makeFunc.getKwdefaults();
    mlir::Value closure = makeFunc.getClosure();

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
    eraseDeadStaticMetadataTree(oldPosargs);
    eraseDeadStaticMetadataTree(oldKwnames);
    eraseDeadStaticMetadataTree(oldKwvalues);
    if (makeFunc->use_empty()) {
      makeFunc.erase();
      eraseDeadStaticMetadataTree(defaults);
      eraseDeadStaticMetadataTree(kwdefaults);
      eraseDeadStaticMetadataTree(closure);
    }
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

static bool storesInlineClassAggregate(ListAppendOp op) {
  auto listType = mlir::dyn_cast<ListType>(op.getList().getType());
  return listType && mlir::isa<ClassType>(listType.getElementType());
}

void publication::insertBoundaries(mlir::ModuleOp module) {
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 16> insertionSites;
  llvm::SmallVector<std::pair<TupleCreateOp, unsigned>, 16> tupleInsertionSites;

  module.walk([&](ListAppendOp op) {
    if (storesInlineClassAggregate(op))
      return;
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

        if (storesInlineClassAggregate(op))
          return;

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

static bool isBorrowOnlyLocalFieldUser(mlir::Operation *user) {
  return mlir::isa<AddOp, StrConcat3Op, SubOp, MulOp, DivOp, FloorDivOp, ModOp,
                   LShiftOp, RShiftOp, BitAndOp, BitOrOp, BitXorOp, LeOp, LtOp,
                   GtOp, GeOp, EqOp, NeOp, ReprOp, ListAppendOp, ListRemoveOp,
                   ListGetOp>(user);
}

static bool hasSingleBorrowThenDrop(mlir::Value value) {
  DecRefOp drop;
  mlir::Operation *borrowUser = nullptr;

  for (mlir::Operation *user : value.getUsers()) {
    if (auto decref = mlir::dyn_cast<DecRefOp>(user)) {
      if (drop)
        return false;
      drop = decref;
      continue;
    }
    if (!isBorrowOnlyLocalFieldUser(user))
      return false;
    if (borrowUser)
      return false;
    borrowUser = user;
  }

  if (!borrowUser || !drop)
    return false;
  if (borrowUser->getBlock() != drop->getBlock())
    return false;
  return borrowUser->isBeforeInBlock(drop);
}

void class_state::proveLocalAccess(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation *op) {
    mlir::Value object;
    mlir::Type fieldType;
    bool borrowedContainerField = false;
    if (auto attrGet = mlir::dyn_cast<AttrGetOp>(op)) {
      object = attrGet.getObject();
      fieldType = attrGet.getResult().getType();
      borrowedContainerField =
          mlir::isa<ListType, TupleType, DictType>(fieldType) &&
          hasSingleBorrowThenDrop(attrGet.getResult());
    } else if (auto attrSet = mlir::dyn_cast<AttrSetOp>(op)) {
      object = attrSet.getObject();
      fieldType = attrSet.getValue().getType();
    } else {
      return;
    }

    if (!class_state::local(object))
      return;
    if (!isZeroCostLocalClassFieldCandidate(fieldType) &&
        !borrowedContainerField)
      return;

    op->setAttr("ly.proven_local_class_access",
                mlir::UnitAttr::get(module.getContext()));
    if (borrowedContainerField)
      op->setAttr(ClassSafetyAttrs::kBorrowedLocalField,
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

static std::optional<llvm::APInt> parsePyIntLiteral(llvm::StringRef text) {
  bool negative = text.consume_front("-");
  if (!negative)
    (void)text.consume_front("+");
  if (text.empty())
    return std::nullopt;
  for (char c : text)
    if (c < '0' || c > '9')
      return std::nullopt;

  unsigned bitWidth =
      std::max<unsigned>(2, static_cast<unsigned>(text.size()) * 4 + 2);
  llvm::APInt magnitude(bitWidth, text, 10);
  if (!negative)
    return magnitude;
  return -magnitude;
}

static std::string formatPyIntLiteral(const llvm::APInt &value) {
  llvm::SmallString<64> out;
  value.toStringSigned(out, 10);
  return out.str().str();
}

static bool isStrAdd(AddOp op) {
  return op && mlir::isa<StrType>(op.getLhs().getType()) &&
         mlir::isa<StrType>(op.getRhs().getType()) &&
         mlir::isa<StrType>(op.getResult().getType());
}

static DecRefOp movableDropBefore(mlir::Value value, mlir::Operation *onlyUser,
                                  mlir::Operation *before) {
  DecRefOp drop;
  for (mlir::Operation *user : value.getUsers()) {
    if (user == onlyUser)
      continue;
    auto candidate = mlir::dyn_cast<DecRefOp>(user);
    if (!candidate || candidate->getBlock() != before->getBlock())
      return nullptr;
    if (!onlyUser->isBeforeInBlock(candidate) ||
        !candidate->isBeforeInBlock(before))
      return nullptr;
    if (drop)
      return nullptr;
    drop = candidate;
  }
  return drop;
}

static DecRefOp soleDropAfter(mlir::Value value, mlir::Operation *onlyUser,
                              mlir::Operation *after) {
  DecRefOp drop;
  for (mlir::Operation *user : value.getUsers()) {
    if (user == onlyUser)
      continue;
    auto candidate = mlir::dyn_cast<DecRefOp>(user);
    if (!candidate || candidate->getBlock() != after->getBlock())
      return nullptr;
    if (!after->isBeforeInBlock(candidate))
      return nullptr;
    if (drop)
      return nullptr;
    drop = candidate;
  }
  return drop;
}

struct StrConcat3Candidate {
  AddOp inner;
  AddOp outer;
  DecRefOp lhsDrop;
  DecRefOp middleDrop;
  DecRefOp innerDrop;
};

static std::optional<StrConcat3Candidate> matchStrConcat3(AddOp outer) {
  if (!isStrAdd(outer))
    return std::nullopt;
  auto inner = value::stripCasts(outer.getLhs()).getDefiningOp<AddOp>();
  if (!isStrAdd(inner) || inner->getBlock() != outer->getBlock())
    return std::nullopt;

  DecRefOp lhsDrop =
      movableDropBefore(inner.getLhs(), inner.getOperation(), outer);
  DecRefOp middleDrop =
      movableDropBefore(inner.getRhs(), inner.getOperation(), outer);
  DecRefOp innerDrop =
      soleDropAfter(inner.getResult(), outer.getOperation(), outer);
  if (!lhsDrop || !middleDrop || !innerDrop)
    return std::nullopt;

  return StrConcat3Candidate{inner, outer, lhsDrop, middleDrop, innerDrop};
}

static bool fuseOneStrConcat3(mlir::ModuleOp module) {
  std::optional<StrConcat3Candidate> candidate;
  module.walk([&](AddOp outer) -> mlir::WalkResult {
    candidate = matchStrConcat3(outer);
    return candidate ? mlir::WalkResult::interrupt()
                     : mlir::WalkResult::advance();
  });
  if (!candidate)
    return false;

  mlir::OpBuilder builder(candidate->outer);
  auto fused = builder.create<StrConcat3Op>(
      candidate->outer.getLoc(), candidate->outer.getResult().getType(),
      candidate->inner.getLhs(), candidate->inner.getRhs(),
      candidate->outer.getRhs());
  candidate->outer.getResult().replaceAllUsesWith(fused.getResult());

  mlir::Operation *after = fused.getOperation();
  candidate->lhsDrop->moveAfter(after);
  after = candidate->lhsDrop.getOperation();
  candidate->middleDrop->moveAfter(after);

  candidate->innerDrop.erase();
  candidate->outer.erase();
  if (candidate->inner.getResult().use_empty())
    candidate->inner.erase();
  return true;
}

void scalar::fuseStrConcat3(mlir::ModuleOp module) {
  while (fuseOneStrConcat3(module))
    ;
}

static bool foldIntBinaryConstant(mlir::Operation *op, mlir::Value lhs,
                                  mlir::Value rhs, mlir::Type resultType,
                                  llvm::StringRef operation) {
  if (!mlir::isa<IntType>(resultType))
    return false;

  auto lhsConst = value::stripCasts(lhs).getDefiningOp<IntConstantOp>();
  auto rhsConst = value::stripCasts(rhs).getDefiningOp<IntConstantOp>();
  if (!lhsConst || !rhsConst)
    return false;

  std::optional<llvm::APInt> lhsValue = parsePyIntLiteral(lhsConst.getValue());
  std::optional<llvm::APInt> rhsValue = parsePyIntLiteral(rhsConst.getValue());
  if (!lhsValue || !rhsValue)
    return false;

  unsigned resultBits =
      std::max(lhsValue->getBitWidth(), rhsValue->getBitWidth()) + 2;
  llvm::APInt lhsWide = lhsValue->sextOrTrunc(resultBits);
  llvm::APInt rhsWide = rhsValue->sextOrTrunc(resultBits);
  llvm::APInt folded(resultBits, 0, /*isSigned=*/true);
  if (operation == "+") {
    folded = lhsWide + rhsWide;
  } else if (operation == "-") {
    folded = lhsWide - rhsWide;
  } else {
    return false;
  }

  mlir::OpBuilder builder(op);
  auto replacement = builder.create<IntConstantOp>(op->getLoc(), resultType,
                                                   formatPyIntLiteral(folded));
  op->getResult(0).replaceAllUsesWith(replacement.getResult());
  op->erase();
  return true;
}

void scalar::foldStaticBuiltinPrintRepr(mlir::ModuleOp module) {
  // This optimization used to bypass typed string lowering by emitting an
  // erased host print call directly.  That violates the
  // memref-centered ABI because the high-level optimizer would introduce a host
  // boundary before value lowering can prove ownership. Keep the pass disabled
  // until it can emit py/unicode memref operations.
  (void)module;
}

void scalar::foldIntConstants(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> candidates;
  module.walk([&](mlir::Operation *op) {
    if (mlir::isa<AddOp, SubOp>(op))
      candidates.push_back(op);
  });

  for (mlir::Operation *op : candidates) {
    if (auto add = mlir::dyn_cast<AddOp>(op)) {
      foldIntBinaryConstant(add.getOperation(), add.getLhs(), add.getRhs(),
                            add.getResult().getType(), "+");
      continue;
    }
    if (auto sub = mlir::dyn_cast<SubOp>(op))
      foldIntBinaryConstant(sub.getOperation(), sub.getLhs(), sub.getRhs(),
                            sub.getResult().getType(), "-");
  }
}

/// Hoist integer constants to entry block and perform CSE.
void scalar::hoistInts(mlir::ModuleOp module) {
  auto crossesPyStructuredResult = [](mlir::Operation *op,
                                      mlir::Operation *root) {
    for (mlir::Operation *parent = op->getParentOp(); parent && parent != root;
         parent = parent->getParentOp()) {
      auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(parent);
      if (!ifOp)
        continue;
      if (llvm::any_of(ifOp.getResultTypes(), isPyType))
        return true;
    }
    return false;
  };

  auto hoistInRegion = [&](mlir::Operation *op, mlir::Region &region) {
    if (region.empty())
      return;

    mlir::Block &entryBlock = region.front();
    llvm::StringMap<IntConstantOp> constantMap;

    // First pass: collect all IntConstantOp
    llvm::SmallVector<IntConstantOp> allConstants;
    op->walk([&](IntConstantOp op) { allConstants.push_back(op); });

    // Second pass: CSE and hoist (using string value as key)
    for (IntConstantOp constant : allConstants) {
      if (constant->getBlock() != &entryBlock &&
          crossesPyStructuredResult(constant.getOperation(), op))
        continue;

      llvm::StringRef value = constant.getValue();
      auto existing = constantMap.find(value);
      if (existing != constantMap.end()) {
        // Replace with existing constant
        constant.getResult().replaceAllUsesWith(existing->second.getResult());
        constant->erase();
      } else {
        // Move to entry block if not already there
        if (constant->getBlock() != &entryBlock) {
          constant->moveBefore(&entryBlock, entryBlock.begin());
        }
        constantMap[value] = constant;
      }
    }
  };

  module.walk([&](mlir::func::FuncOp func) {
    if (!func.isExternal())
      hoistInRegion(func.getOperation(), func.getBody());
  });
  module.walk([&](FuncOp func) {
    if (func->getNumRegions() != 0)
      hoistInRegion(func.getOperation(), func->getRegion(0));
  });
}

/// CSE for LLVM constants within each function.
void scalar::cseConstants(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    if (func.isExternal())
      return;

    mlir::DominanceInfo dominance(func.getOperation());
    // Map from (type, value) -> first constant op
    llvm::DenseMap<std::pair<mlir::Type, mlir::Attribute>,
                   llvm::SmallVector<mlir::LLVM::ConstantOp, 4>>
        constantCache;
    llvm::SmallVector<mlir::LLVM::ConstantOp> toErase;

    func.walk([&](mlir::LLVM::ConstantOp constOp) {
      auto key = std::make_pair(constOp.getType(), constOp.getValue());
      auto &candidates = constantCache[key];
      if (mlir::LLVM::ConstantOp dominating =
              findDominatingCached(candidates, constOp, dominance)) {
        constOp.getResult().replaceAllUsesWith(dominating.getResult());
        toErase.push_back(constOp);
      } else {
        candidates.push_back(constOp);
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
  if (mlir::isa<TupleCreateOp>(op) && op->hasAttr("ly.dead_call_pack"))
    return true;
  if (mlir::isa<StrConstantOp, IntConstantOp, FloatConstantOp, ReprOp>(op))
    return true;
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
