#include "Passes/Runtime/Async.h"

#include "Passes/Runtime/Conversion.h"
#include "Passes/Runtime/Helpers.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace py {
namespace {

bool stackMemRef(mlir::Value value, llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!value || !seen.insert(value).second)
    return false;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = arg.getOwner();
    if (!owner)
      return false;
    for (mlir::Block *predecessor : owner->getPredecessors()) {
      mlir::Operation *terminator = predecessor->getTerminator();
      auto branch = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(terminator);
      if (!branch)
        continue;
      for (unsigned i = 0, e = branch->getNumSuccessors(); i != e; ++i) {
        if (branch->getSuccessor(i) != owner)
          continue;
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(i);
        unsigned argIndex = arg.getArgNumber();
        if (argIndex >= operands.size() || operands.isOperandProduced(argIndex))
          continue;
        if (stackMemRef(operands[argIndex], seen))
          return true;
      }
    }
    return false;
  }
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (mlir::isa<mlir::memref::AllocaOp>(def))
    return true;
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
    for (mlir::Value operand : cast.getOperands())
      if (stackMemRef(operand, seen))
        return true;
  }
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(def))
    return stackMemRef(cast.getSource(), seen);
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return false;
    unsigned resultIndex = result.getResultNumber();
    bool captures = false;
    auto checkYield = [&](mlir::scf::YieldOp yield) {
      if (resultIndex >= yield.getNumOperands())
        return;
      if (stackMemRef(yield.getOperand(resultIndex), seen))
        captures = true;
    };
    ifOp.getThenRegion().walk(checkYield);
    ifOp.getElseRegion().walk(checkYield);
    return captures;
  }
  return false;
}

} // namespace

namespace lowering::runtime::async {

mlir::LogicalResult verifyReturnPayloads(mlir::ModuleOp module) {
  bool failedAny = false;
  module.walk([&](mlir::async::ReturnOp op) {
    for (mlir::Value operand : op.getOperands()) {
      llvm::SmallPtrSet<mlir::Value, 8> seen;
      if (!stackMemRef(operand, seen))
        continue;
      op->emitOpError("captures a stack memref in an async result payload; "
                      "escaping async payloads must be promoted to managed "
                      "heap descriptors before async.return");
      failedAny = true;
      return;
    }
  });
  return mlir::failure(failedAny);
}

mlir::LogicalResult
normalizeFuncSignatures(mlir::ModuleOp module,
                        PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::async::FuncOp> funcs;
  module.walk([&](mlir::async::FuncOp op) { funcs.push_back(op); });

  mlir::OpBuilder builder(module.getContext());
  for (mlir::async::FuncOp op : funcs) {
    mlir::FunctionType oldType = op.getFunctionType();
    llvm::SmallVector<llvm::SmallVector<mlir::Type>> convertedArgTypes;
    llvm::SmallVector<mlir::Type> flattenedArgTypes;
    convertedArgTypes.reserve(oldType.getNumInputs());

    for (mlir::Type inputType : oldType.getInputs()) {
      llvm::SmallVector<mlir::Type> converted;
      if (mlir::failed(typeConverter.convertType(inputType, converted)) ||
          converted.empty())
        return op.emitOpError("failed to convert async function argument type");
      flattenedArgTypes.append(converted.begin(), converted.end());
      convertedArgTypes.push_back(std::move(converted));
    }

    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(
            typeConverter.convertTypes(oldType.getResults(), resultTypes)))
      return op.emitOpError("failed to convert async function result types");

    if (::py::lowering::runtime::conversion::types::same(oldType.getInputs(),
                                                         flattenedArgTypes) &&
        ::py::lowering::runtime::conversion::types::same(oldType.getResults(),
                                                         resultTypes))
      continue;

    op.setFunctionTypeAttr(mlir::TypeAttr::get(
        builder.getFunctionType(flattenedArgTypes, resultTypes)));
    ::py::lowering::runtime::async_args::mark(
        op.getOperation(), oldType.getInputs(), typeConverter,
        /*trailingExceptionCell=*/true);

    if (op.isDeclaration())
      continue;
    if (op.getBody().empty())
      return op.emitOpError("has no entry block");

    mlir::Block &entry = op.getBody().front();
    llvm::SmallVector<mlir::BlockArgument> oldArgs(entry.args_begin(),
                                                   entry.args_end());
    if (oldArgs.size() != convertedArgTypes.size())
      return op.emitOpError("entry block argument count does not match type");

    llvm::SmallVector<llvm::SmallVector<mlir::BlockArgument>>
        convertedArgValues;
    convertedArgValues.reserve(oldArgs.size());
    for (auto [oldArg, convertedTypes] :
         llvm::zip(oldArgs, convertedArgTypes)) {
      llvm::SmallVector<mlir::Location> locs(convertedTypes.size(),
                                             oldArg.getLoc());
      auto addedArgs =
          entry.addArguments(mlir::TypeRange(convertedTypes), locs);
      convertedArgValues.push_back(llvm::SmallVector<mlir::BlockArgument>(
          addedArgs.begin(), addedArgs.end()));
    }

    builder.setInsertionPointToStart(&entry);
    for (auto [oldArg, convertedValues] :
         llvm::zip(oldArgs, convertedArgValues)) {
      mlir::Value replacement;
      if (convertedValues.size() == 1 &&
          convertedValues.front().getType() == oldArg.getType()) {
        replacement = convertedValues.front();
      } else {
        llvm::SmallVector<mlir::Value> operands(convertedValues.begin(),
                                                convertedValues.end());
        replacement = builder
                          .create<mlir::UnrealizedConversionCastOp>(
                              oldArg.getLoc(),
                              mlir::TypeRange{oldArg.getType()}, operands)
                          .getResult(0);
      }
      oldArg.replaceAllUsesWith(replacement);
    }
    entry.eraseArguments(0, static_cast<unsigned>(oldArgs.size()));
  }

  return mlir::success();
}

} // namespace lowering::runtime::async

} // namespace py
