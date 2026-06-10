#include "Passes/OwnershipAnalysis.h"

#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

bool isPyOwnershipTrackedType(mlir::Type type);

bool isPyOwnershipTrackedType(mlir::Type type) {
  if (mlir::isa<FuncSignatureType, FuncType, PrimFuncType, NoneType, BoolType,
                FloatType>(type))
    return false;
  return isPyType(type);
}

bool isPyOwnershipImmortalOp(mlir::Operation *op) {
  if (mlir::isa<IntConstantOp>(op))
    return true;
  if (op->getNumResults() == 1 &&
      mlir::isa<BoolType>(op->getResult(0).getType()))
    return true;
  if (auto classNew = mlir::dyn_cast<ClassNewOp>(op))
    return classNew.getClassNameAttr().getValue() == "Exception";
  return mlir::isa<NoneOp, FuncObjectOp, TupleEmptyOp, ExceptionNullOp,
                   TracebackNullOp, LocationCurrentOp>(op);
}

bool isPyOwnershipIdentityTransform(mlir::Operation *op) {
  (void)op;
  return false;
}

bool createsPyOwnedResult(mlir::Operation *op) {
  if (op->hasAttr(OwnershipContractAttrs::kOwnedResults))
    return true;
  if (isPyOwnershipImmortalOp(op))
    return false;
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return llvm::any_of(ifOp.getResults(), [](mlir::Value result) {
      return isPyOwnershipTrackedType(result.getType());
    });
  if (mlir::isa<TryOp, TupleCreateOp, DictEmptyOp, StrConstantOp, IntConstantOp,
                FloatConstantOp, AddOp, StrConcat3Op, SubOp, MulOp, ReprOp,
                DivOp, FloorDivOp, ModOp, LShiftOp, RShiftOp, BitAndOp, BitOrOp,
                BitXorOp, MakeFunctionOp, CastFromPrimOp, ClassNewOp,
                ClassPromoteOp, PublishOp, ListNewOp, ListGetOp, TupleGetOp,
                DictGetOp, AttrGetOp, AttrGetLocalOp, ExceptionNewOp, AwaitOp,
                AsyncGatherOp, CoroCreateOp, TaskCreateOp, AsyncSleepOp>(op))
    return true;

  return mlir::isa<CallOp, CallVectorOp, NativeCallOp>(op);
}

static bool tupleCreateConsumesOperand(TupleCreateOp tuple,
                                       mlir::Value operand) {
  auto isDirectCallPack = [](TupleCreateOp tuple) {
    llvm::SmallVector<mlir::Value, 4> worklist{tuple.getResult()};
    llvm::SmallPtrSet<mlir::Value, 4> seen;
    bool sawCallPackUse = false;
    while (!worklist.empty()) {
      mlir::Value value = worklist.pop_back_val();
      if (!value || !seen.insert(value).second)
        continue;
      for (mlir::Operation *user : value.getUsers()) {
        if (mlir::isa<DecRefOp>(user))
          continue;
        if (auto cast =
                mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user)) {
          for (mlir::Value result : cast.getResults())
            worklist.push_back(result);
          continue;
        }
        if (auto call = mlir::dyn_cast<CallVectorOp>(user)) {
          if (call.getPosargs() != value)
            return false;
          sawCallPackUse = true;
          continue;
        }
        if (auto call = mlir::dyn_cast<CallOp>(user)) {
          if (call.getPosargs() != value)
            return false;
          sawCallPackUse = true;
          continue;
        }
        if (auto invoke = mlir::dyn_cast<InvokeOp>(user)) {
          if (invoke.getPosargs() != value)
            return false;
          sawCallPackUse = true;
          continue;
        }
        return false;
      }
    }
    return sawCallPackUse;
  };

  if (isDirectCallPack(tuple))
    return false;

  auto tupleType = mlir::dyn_cast<TupleType>(tuple.getResult().getType());
  if (!tupleType)
    return true;
  auto elementTypes = tupleType.getElementTypes();
  for (auto [index, element] : llvm::enumerate(tuple.getElements())) {
    if (element != operand)
      continue;
    if (elementTypes.empty())
      return true;
    mlir::Type elementType =
        elementTypes.size() == 1 ? elementTypes.front() : elementTypes[index];
    switch (container::Slot::policy(elementType)) {
    case container::SlotPolicy::NativeInteger:
    case container::SlotPolicy::NativeBool:
    case container::SlotPolicy::NativeFloat:
      return false;
    case container::SlotPolicy::ObjectParts:
    case container::SlotPolicy::Unsupported:
      return true;
    }
  }
  return false;
}

bool consumesPyOwnedOperand(mlir::Operation *op, mlir::Value operand) {
  if (auto decRef = mlir::dyn_cast<DecRefOp>(op))
    return decRef.getObject() == operand;
  if (mlir::isa<ReturnOp, RaiseOp, TryYieldOp, ExceptYieldOp, FinallyYieldOp,
                mlir::async::ReturnOp>(op))
    return true;
  if (auto tuple = mlir::dyn_cast<TupleCreateOp>(op))
    return tupleCreateConsumesOperand(tuple, operand);
  if (auto awaitOp = mlir::dyn_cast<AwaitOp>(op))
    return awaitOp.getAwaitable() == operand;
  if (auto taskCreate = mlir::dyn_cast<TaskCreateOp>(op))
    return taskCreate.getCoroutine() == operand;
  if (auto gather = mlir::dyn_cast<AsyncGatherOp>(op))
    return llvm::is_contained(gather.getAwaitables(), operand);
  if (auto exception = mlir::dyn_cast<ExceptionNewOp>(op))
    return llvm::is_contained(exception.getArgs(), operand);
  if (auto attrSet = mlir::dyn_cast<AttrSetOp>(op))
    return attrSet.getValue() == operand && op->hasAttr("ly.consume_value");
  if (auto attrSetLocal = mlir::dyn_cast<AttrSetLocalOp>(op))
    return attrSetLocal.getValue() == operand &&
           op->hasAttr("ly.consume_value");
  if (auto listAppend = mlir::dyn_cast<ListAppendOp>(op))
    return listAppend.getValue() == operand && op->hasAttr("ly.consume_value");
  if (auto dictInsert = mlir::dyn_cast<DictInsertOp>(op))
    return (dictInsert.getKey() == operand ||
            dictInsert.getValue() == operand) &&
           op->hasAttr("ly.consume_value");
  if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
    auto ifOp = yield->getParentOfType<mlir::scf::IfOp>();
    if (!ifOp)
      return false;
    for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) {
      if (yielded == operand && index < ifOp.getNumResults())
        return isPyOwnershipTrackedType(ifOp.getResult(index).getType());
    }
    return false;
  }
  return false;
}

OwnershipAliasAnalysis::OwnershipAliasAnalysis(
    mlir::Region &region, TypePredicate tracksType,
    IdentityPredicate isIdentityTransform) {
  for (mlir::Block &block : region) {
    for (mlir::BlockArgument arg : block.getArguments())
      parent.try_emplace(arg, arg);
    for (mlir::Operation &op : block) {
      for (mlir::Value result : op.getResults())
        parent.try_emplace(result, result);
      for (mlir::Value operand : op.getOperands())
        parent.try_emplace(operand, operand);
    }
  }

  region.walk([&](mlir::Operation *op) {
    if (isIdentityTransform(op)) {
      unionSets(op->getOperand(0), op->getResult(0));
      return;
    }
  });
  llvm::SmallVector<mlir::LLVM::ExtractValueOp, 32> extracts;
  region.walk([&](mlir::LLVM::ExtractValueOp extract) {
    for (mlir::LLVM::ExtractValueOp prior : extracts) {
      if (prior.getContainer() == extract.getContainer() &&
          prior.getPosition() == extract.getPosition()) {
        unionSets(prior.getResult(), extract.getResult());
        break;
      }
    }
    extracts.push_back(extract);
  });

  for (mlir::Block &block : region) {
    mlir::Operation *terminator = block.getTerminator();
    auto unionBranchValues = [&](mlir::ValueRange operands,
                                 mlir::Block *successor) {
      if (!successor)
        return;
      unsigned count =
          std::min<unsigned>(operands.size(), successor->getNumArguments());
      for (unsigned j = 0; j != count; ++j) {
        mlir::BlockArgument argument = successor->getArgument(j);
        mlir::Value operand = operands[j];
        if (operand && ((tracksType(operand.getType()) &&
                         tracksType(argument.getType())) ||
                        operand.getType() == argument.getType()))
          unionSets(operand, argument);
      }
    };
    if (auto br = mlir::dyn_cast_or_null<mlir::LLVM::BrOp>(terminator))
      unionBranchValues(br.getDestOperands(), br.getDest());
    if (auto cond = mlir::dyn_cast_or_null<mlir::LLVM::CondBrOp>(terminator)) {
      unionBranchValues(cond.getTrueDestOperands(), cond.getTrueDest());
      unionBranchValues(cond.getFalseDestOperands(), cond.getFalseDest());
    }
    if (auto br = mlir::dyn_cast_or_null<mlir::cf::BranchOp>(terminator))
      unionBranchValues(br.getDestOperands(), br.getDest());
    if (auto cond =
            mlir::dyn_cast_or_null<mlir::cf::CondBranchOp>(terminator)) {
      unionBranchValues(cond.getTrueDestOperands(), cond.getTrueDest());
      unionBranchValues(cond.getFalseDestOperands(), cond.getFalseDest());
    }
    auto branch = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(terminator);
    if (branch) {
      for (unsigned i = 0, e = branch->getNumSuccessors(); i != e; ++i) {
        mlir::Block *successor = branch->getSuccessor(i);
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(i);
        unionBranchValues(operands.getForwardedOperands(), successor);
      }
    }

    if (auto invoke = mlir::dyn_cast_or_null<InvokeOp>(terminator)) {
      unionSuccessorOperands(invoke.getNormalDestOperands(),
                             invoke.getNormalDest(), tracksType);
      unionSuccessorOperands(invoke.getUnwindDestOperands(),
                             invoke.getUnwindDest(), tracksType);
    }
  }

  for (auto &kv : parent)
    if (tracksType(kv.first.getType()))
      carrierRoots.insert(find(kv.first));

  for (auto &kv : parent) {
    mlir::Value root = find(kv.first);
    members[root].push_back(kv.first);
  }
}

OwnershipAliasAnalysis::OwnershipAliasAnalysis(
    mlir::Region &region, ValuePredicate tracksValue,
    IdentityPredicate isIdentityTransform) {
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 16> branchPairs;
  for (mlir::Block &block : region) {
    for (mlir::BlockArgument arg : block.getArguments())
      parent.try_emplace(arg, arg);
    for (mlir::Operation &op : block) {
      for (mlir::Value result : op.getResults())
        parent.try_emplace(result, result);
      for (mlir::Value operand : op.getOperands())
        parent.try_emplace(operand, operand);
    }
  }

  region.walk([&](mlir::Operation *op) {
    if (isIdentityTransform(op) && op->getNumOperands() == 1 &&
        op->getNumResults() == 1)
      unionSets(op->getOperand(0), op->getResult(0));

    auto addBranchPairs = [&](mlir::ValueRange operands,
                              mlir::Block *successor) {
      if (!successor)
        return;
      unsigned count =
          std::min<unsigned>(operands.size(), successor->getNumArguments());
      for (unsigned j = 0; j != count; ++j) {
        mlir::Value operand = operands[j];
        mlir::BlockArgument argument = successor->getArgument(j);
        if (operand)
          branchPairs.push_back({operand, argument});
      }
    };

    if (auto br = mlir::dyn_cast<mlir::LLVM::BrOp>(op))
      addBranchPairs(br.getDestOperands(), br.getDest());
    if (auto cond = mlir::dyn_cast<mlir::LLVM::CondBrOp>(op)) {
      addBranchPairs(cond.getTrueDestOperands(), cond.getTrueDest());
      addBranchPairs(cond.getFalseDestOperands(), cond.getFalseDest());
    }
    if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(op))
      addBranchPairs(br.getDestOperands(), br.getDest());
    if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(op)) {
      addBranchPairs(cond.getTrueDestOperands(), cond.getTrueDest());
      addBranchPairs(cond.getFalseDestOperands(), cond.getFalseDest());
    }

    auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(op);
    if (!branch)
      return;
    for (unsigned i = 0, e = branch->getNumSuccessors(); i != e; ++i) {
      mlir::Block *successor = branch->getSuccessor(i);
      mlir::SuccessorOperands operands = branch.getSuccessorOperands(i);
      unsigned count =
          std::min<unsigned>(operands.size(), successor->getNumArguments());
      for (unsigned j = 0; j != count; ++j) {
        if (operands.isOperandProduced(j))
          continue;
        mlir::Value operand = operands[j];
        mlir::BlockArgument argument = successor->getArgument(j);
        if (operand)
          branchPairs.push_back({operand, argument});
      }
    }
  });
  llvm::SmallVector<mlir::LLVM::ExtractValueOp, 32> extracts;
  region.walk([&](mlir::LLVM::ExtractValueOp extract) {
    for (mlir::LLVM::ExtractValueOp prior : extracts) {
      if (prior.getContainer() == extract.getContainer() &&
          prior.getPosition() == extract.getPosition()) {
        unionSets(prior.getResult(), extract.getResult());
        break;
      }
    }
    extracts.push_back(extract);
  });

  for (auto &kv : parent)
    if (tracksValue(kv.first))
      carrierRoots.insert(find(kv.first));

  bool changed = false;
  do {
    changed = false;
    for (auto [operand, argument] : branchPairs) {
      mlir::Value operandRoot = find(operand);
      mlir::Value argumentRoot = find(argument);
      if (!carrierRoots.contains(operandRoot) &&
          !carrierRoots.contains(argumentRoot))
        continue;
      unionSets(operand, argument);
      mlir::Value root = find(operand);
      if (carrierRoots.insert(root).second)
        changed = true;
    }
  } while (changed);

  llvm::SmallVector<mlir::Value, 16> carrierRootValues(carrierRoots.begin(),
                                                       carrierRoots.end());
  for (mlir::Value root : carrierRootValues)
    carrierRoots.insert(find(root));

  for (auto &kv : parent) {
    mlir::Value root = find(kv.first);
    members[root].push_back(kv.first);
  }
}

mlir::Value OwnershipAliasAnalysis::getRoot(mlir::Value value) const {
  return find(value);
}

bool OwnershipAliasAnalysis::sameRoot(mlir::Value lhs, mlir::Value rhs) const {
  return getRoot(lhs) == getRoot(rhs);
}

bool OwnershipAliasAnalysis::isCarrier(mlir::Value value) const {
  return carrierRoots.contains(find(value));
}

bool OwnershipAliasAnalysis::tracksThroughAlias(mlir::Value value) const {
  if (isPyOwnershipTrackedType(value.getType()))
    return true;
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  collectAliases(value, aliasSet);
  return llvm::any_of(aliasSet, [](mlir::Value alias) {
    return isPyOwnershipTrackedType(alias.getType());
  });
}

bool OwnershipAliasAnalysis::rootIsImmortal(mlir::Value root) const {
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  collectAliases(root, aliasSet);
  for (mlir::Value member : aliasSet) {
    mlir::Operation *def = member.getDefiningOp();
    if (def && isPyOwnershipImmortalOp(def))
      return true;
  }
  return false;
}

bool OwnershipAliasAnalysis::rootHasAggregateBorrow(mlir::Value root) const {
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  collectAliases(root, aliasSet);
  for (mlir::Value member : aliasSet) {
    mlir::Operation *def = member.getDefiningOp();
    if (!def)
      continue;
    if (def->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return true;
    auto role =
        def->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    if (role && role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionLoad)
      return true;
  }
  return false;
}

bool OwnershipAliasAnalysis::rootIsEntryBorrowed(mlir::Value value,
                                                 mlir::Block &entry) const {
  if (!isPyOwnershipTrackedType(value.getType()))
    return false;
  mlir::Value root = getRoot(value);
  for (mlir::BlockArgument arg : entry.getArguments())
    if (sameRoot(arg, root))
      return true;
  return false;
}

bool OwnershipAliasAnalysis::rootIsCapturedBorrow(mlir::Value root,
                                                  mlir::Region &region) const {
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  collectAliases(root, aliasSet);
  for (mlir::Value member : aliasSet) {
    if (!isPyOwnershipTrackedType(member.getType()))
      continue;
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(member))
      if (arg.getOwner()->getParent() != &region)
        return true;
    if (mlir::Operation *def = member.getDefiningOp())
      if (def->getParentRegion() != &region)
        return true;
  }
  return false;
}

void OwnershipAliasAnalysis::collectAliases(
    mlir::Value value, llvm::SmallVectorImpl<mlir::Value> &aliases) const {
  mlir::Value root = find(value);
  auto it = members.find(root);
  if (it != members.end()) {
    aliases.append(it->second.begin(), it->second.end());
    return;
  }
  aliases.push_back(value);
}

mlir::Value OwnershipAliasAnalysis::find(mlir::Value value) const {
  auto it = parent.find(value);
  if (it == parent.end())
    return value;
  if (it->second == value)
    return value;
  mlir::Value root = find(it->second);
  parent[value] = root;
  return root;
}

void OwnershipAliasAnalysis::unionSets(mlir::Value lhs, mlir::Value rhs) {
  mlir::Value lhsRoot = find(lhs);
  mlir::Value rhsRoot = find(rhs);
  if (lhsRoot != rhsRoot)
    parent[rhsRoot] = lhsRoot;
}

void OwnershipAliasAnalysis::unionSuccessorOperands(mlir::ValueRange operands,
                                                    mlir::Block *successor,
                                                    TypePredicate tracksType) {
  if (!successor)
    return;
  for (auto [operand, argument] :
       llvm::zip_equal(operands, successor->getArguments())) {
    if ((tracksType(operand.getType()) && tracksType(argument.getType())) ||
        operand.getType() == argument.getType())
      unionSets(operand, argument);
  }
}

} // namespace py
