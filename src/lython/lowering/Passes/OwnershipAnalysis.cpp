#include "Passes/OwnershipAnalysis.h"

#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdlib>
#include <string>

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

bool isPyOwnershipTrackedType(mlir::Type type);

static bool isSmallIntConstant(mlir::Operation *op) {
  auto intConst = mlir::dyn_cast<IntConstantOp>(op);
  if (!intConst)
    return false;

  std::string text = intConst.getValue().str();
  const char *begin = text.c_str();
  char *end = nullptr;
  long long value = std::strtoll(begin, &end, 10);
  return end == begin + text.size() && value >= -5 && value <= 256;
}

static bool isSmallLLVMIntegerConstant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>();
  if (!constant)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return false;
  int64_t raw = integer.getInt();
  return raw >= -5 && raw <= 256;
}

static bool isSmallRuntimeLongBridge(mlir::Operation *op) {
  auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op);
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
      !mlir::isa<IntType>(cast.getResult(0).getType()))
    return false;
  auto call = cast.getOperand(0).getDefiningOp<mlir::LLVM::CallOp>();
  if (!call)
    return false;
  auto callee = call.getCallee();
  return callee && *callee == RuntimeSymbols::kLongFromI64 &&
         call.getNumOperands() == 1 &&
         isSmallLLVMIntegerConstant(call.getOperand(0));
}

static bool isOwnedRuntimeObjectBridge(mlir::Operation *op) {
  auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op);
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
      !isPyOwnershipTrackedType(cast.getResult(0).getType()) ||
      isPyOwnershipTrackedType(cast.getOperand(0).getType()))
    return false;
  auto call = cast.getOperand(0).getDefiningOp<mlir::LLVM::CallOp>();
  if (!call)
    return false;
  auto callee = call.getCallee();
  if (!callee)
    return false;
  if (runtime::Callee::alwaysOwnedResult(*callee))
    return true;
  if (*callee == RuntimeSymbols::kLongFromI64)
    return call.getNumOperands() != 1 ||
           !isSmallLLVMIntegerConstant(call.getOperand(0));
  return false;
}

static bool isOwnedAsyncPayloadBridge(mlir::Operation *op) {
  auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op);
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
      !isPyOwnershipTrackedType(cast.getResult(0).getType()) ||
      isPyOwnershipTrackedType(cast.getOperand(0).getType()))
    return false;
  mlir::Operation *def = cast.getOperand(0).getDefiningOp();
  return def &&
         mlir::isa<mlir::async::AwaitOp, mlir::async::RuntimeLoadOp>(def);
}

bool isPyOwnershipTrackedType(mlir::Type type) {
  if (mlir::isa<FuncSignatureType, FuncType, PrimFuncType, NoneType, BoolType>(
          type))
    return false;
  return isPyType(type);
}

bool isPyOwnershipImmortalOp(mlir::Operation *op) {
  if (isSmallRuntimeLongBridge(op))
    return true;
  if (isSmallIntConstant(op))
    return true;
  if (op->getNumResults() == 1 &&
      mlir::isa<BoolType>(op->getResult(0).getType()))
    return true;
  if (auto classNew = mlir::dyn_cast<ClassNewOp>(op))
    return classNew.getClassNameAttr().getValue() == "Exception";
  return mlir::isa<NoneOp, FuncObjectOp, TupleEmptyOp, StrConstantOp,
                   ExceptionNullOp, TracebackNullOp, LocationCurrentOp>(op);
}

bool isPyOwnershipIdentityTransform(mlir::Operation *op) {
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op))
    return cast->getNumOperands() == 1 && cast->getNumResults() == 1;
  auto upcast = mlir::dyn_cast<UpcastOp>(op);
  return upcast &&
         !isCompilerOwnedMemRefContainerType(upcast.getInput().getType());
}

bool isPyOwnershipMaterializedObjectBridge(mlir::Operation *op) {
  auto upcast = mlir::dyn_cast<UpcastOp>(op);
  return upcast &&
         isCompilerOwnedMemRefContainerType(upcast.getInput().getType());
}

bool createsPyOwnedResult(mlir::Operation *op) {
  if (isPyOwnershipImmortalOp(op))
    return false;
  if (isPyOwnershipMaterializedObjectBridge(op))
    return true;
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1 &&
        isPyOwnershipTrackedType(cast.getResult(0).getType()) &&
        !isPyOwnershipTrackedType(cast.getOperand(0).getType())) {
      if (isOwnedRuntimeObjectBridge(op))
        return true;
      if (isOwnedAsyncPayloadBridge(op))
        return true;
      if (mlir::Operation *def = cast.getOperand(0).getDefiningOp())
        if (createsPyOwnedResult(def))
          return true;
    }
  }
  if (isPyOwnershipIdentityTransform(op))
    return false;

  if (mlir::isa<TryOp, TupleCreateOp, DictEmptyOp, IntConstantOp,
                FloatConstantOp, AddOp, SubOp, ReprOp, MakeFunctionOp,
                CastFromPrimOp, ClassNewOp, ClassPromoteOp, PublishOp,
                ListNewOp, ListGetOp, DictGetOp, AttrGetOp, AttrGetLocalOp,
                ExceptionNewOp, AwaitOp, AsyncGatherOp, CoroCreateOp,
                TaskCreateOp, AsyncSleepOp>(op))
    return true;

  return mlir::isa<CallOp, CallVectorOp, NativeCallOp>(op);
}

bool consumesPyOwnedOperand(mlir::Operation *op, mlir::Value operand) {
  if (mlir::isa<ReturnOp, RaiseOp, TryYieldOp, ExceptYieldOp, FinallyYieldOp,
                mlir::async::ReturnOp>(op))
    return true;
  if (mlir::isa<TupleCreateOp>(op))
    return true;
  if (auto awaitOp = mlir::dyn_cast<AwaitOp>(op))
    return awaitOp.getAwaitable() == operand;
  if (auto taskCreate = mlir::dyn_cast<TaskCreateOp>(op))
    return taskCreate.getCoroutine() == operand;
  if (auto gather = mlir::dyn_cast<AsyncGatherOp>(op))
    return llvm::is_contained(gather.getAwaitables(), operand);
  if (auto attrSet = mlir::dyn_cast<AttrSetOp>(op))
    return attrSet.getValue() == operand && op->hasAttr("ly.consume_value");
  if (auto listAppend = mlir::dyn_cast<ListAppendOp>(op))
    return listAppend.getValue() == operand && op->hasAttr("ly.consume_value");
  if (auto dictInsert = mlir::dyn_cast<DictInsertOp>(op))
    return (dictInsert.getKey() == operand ||
            dictInsert.getValue() == operand) &&
           op->hasAttr("ly.consume_value");
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
    if (!isIdentityTransform(op))
      return;
    unionSets(op->getOperand(0), op->getResult(0));
  });

  for (mlir::Block &block : region) {
    mlir::Operation *terminator = block.getTerminator();
    auto branch = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(terminator);
    if (branch) {
      for (unsigned i = 0, e = branch->getNumSuccessors(); i != e; ++i) {
        mlir::Block *successor = branch->getSuccessor(i);
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(i);
        unsigned count =
            std::min<unsigned>(operands.size(), successor->getNumArguments());
        for (unsigned j = 0; j != count; ++j) {
          mlir::BlockArgument argument = successor->getArgument(j);
          if (operands.isOperandProduced(j)) {
            continue;
          }

          mlir::Value operand = operands[j];
          if (operand && ((tracksType(operand.getType()) &&
                           tracksType(argument.getType())) ||
                          operand.getType() == argument.getType()))
            unionSets(operand, argument);
        }
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

bool OwnershipAliasAnalysis::isCarrier(mlir::Value value) const {
  return carrierRoots.contains(find(value));
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
