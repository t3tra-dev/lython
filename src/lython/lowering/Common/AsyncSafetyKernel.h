#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace py {

inline bool hasPresplitCoroutinePassthrough(mlir::Operation *op) {
  if (!op)
    return false;
  auto passthrough = op->getAttrOfType<mlir::ArrayAttr>("passthrough");
  if (!passthrough)
    return false;
  return llvm::any_of(passthrough, [](mlir::Attribute attr) {
    auto string = mlir::dyn_cast<mlir::StringAttr>(attr);
    return string && string.getValue() == "presplitcoroutine";
  });
}

namespace async_runtime {
namespace contract {

inline bool has(mlir::Operation *op, llvm::StringRef attrName) {
  return op && op->hasAttr(attrName);
}

inline llvm::SmallVector<unsigned, 2> indices(mlir::Operation *op,
                                              llvm::StringRef attrName) {
  llvm::SmallVector<unsigned, 2> values;
  auto read = [&](mlir::Operation *source) -> bool {
    if (!source)
      return false;
    auto array = source->getAttrOfType<mlir::ArrayAttr>(attrName);
    if (!array)
      return false;
    for (mlir::Attribute attr : array) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      if (integer && integer.getInt() >= 0)
        values.push_back(static_cast<unsigned>(integer.getInt()));
    }
    return true;
  };
  read(op);
  return values;
}

inline bool pointerLike(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

inline bool validOperandIndices(mlir::Operation *op,
                                llvm::ArrayRef<unsigned> indices) {
  if (!op)
    return false;
  for (unsigned index : indices)
    if (index >= op->getNumOperands() ||
        !pointerLike(op->getOperand(index).getType()))
      return false;
  return true;
}

inline bool validResultIndices(mlir::Operation *op,
                               llvm::ArrayRef<unsigned> indices) {
  if (!op)
    return false;
  for (unsigned index : indices)
    if (index >= op->getNumResults() ||
        !pointerLike(op->getResult(index).getType()))
      return false;
  return true;
}

inline bool hasBorrowedOperands(mlir::Operation *op) {
  llvm::SmallVector<unsigned, 2> indices =
      contract::indices(op, AsyncSafetyAttrs::kRuntimeHandleBorrowArgs);
  return !indices.empty() && validOperandIndices(op, indices);
}

inline bool hasTransferredOperands(mlir::Operation *op) {
  llvm::SmallVector<unsigned, 2> indices =
      contract::indices(op, AsyncSafetyAttrs::kRuntimeHandleTransferArgs);
  return !indices.empty() && validOperandIndices(op, indices);
}

inline bool hasOwnedResults(mlir::Operation *op) {
  llvm::SmallVector<unsigned, 2> indices =
      contract::indices(op, AsyncSafetyAttrs::kRuntimeHandleOwnedResults);
  return !indices.empty() && validResultIndices(op, indices);
}

} // namespace contract

struct Handle {
  static bool ownedResult(mlir::Operation *op) {
    return contract::hasOwnedResults(op);
  }

  static bool isCallee(mlir::Operation *op) {
    return contract::hasBorrowedOperands(op) ||
           contract::hasTransferredOperands(op) ||
           contract::hasOwnedResults(op);
  }

  static llvm::SmallVector<unsigned, 2>
  transferredOperands(mlir::Operation *op) {
    llvm::SmallVector<unsigned, 2> indices =
        contract::indices(op, AsyncSafetyAttrs::kRuntimeHandleTransferArgs);
    if (!contract::validOperandIndices(op, indices))
      indices.clear();
    return indices;
  }

  static llvm::SmallVector<unsigned, 2> borrowedOperands(mlir::Operation *op) {
    llvm::SmallVector<unsigned, 2> indices =
        contract::indices(op, AsyncSafetyAttrs::kRuntimeHandleBorrowArgs);
    if (!contract::validOperandIndices(op, indices))
      indices.clear();
    return indices;
  }

  static bool ownedChildProducer(mlir::Operation *op) {
    return contract::has(op, AsyncSafetyAttrs::kRuntimeExecuteEntry);
  }
};

struct Entry {
  static bool isFunction(mlir::Operation *parent) {
    return contract::has(parent, AsyncSafetyAttrs::kRuntimeExecuteEntry) ||
           hasPresplitCoroutinePassthrough(parent);
  }
};
} // namespace async_runtime

inline bool isCoroSuspendStatus(mlir::Value value) {
  while (value) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    llvm::StringRef opName = def->getName().getStringRef();
    if ((opName == "llvm.sext" || opName == "llvm.zext" ||
         opName == "llvm.trunc") &&
        def->getNumOperands() == 1) {
      value = def->getOperand(0);
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    return opName == "llvm.intr.coro.suspend";
  }
  return false;
}

inline std::optional<unsigned>
findCoroSuspendCleanupSuccessorIndex(mlir::LLVM::SwitchOp switchOp) {
  auto caseValues = switchOp.getCaseValues();
  if (!caseValues)
    return std::nullopt;
  unsigned caseIndex = 0;
  for (llvm::APInt value : caseValues->getValues<llvm::APInt>()) {
    if (value == 1)
      return caseIndex + 1;
    ++caseIndex;
  }
  return std::nullopt;
}

namespace async_runtime {
struct ValueStorage {
  static bool isAddress(mlir::Value value) {
    if (!value)
      return false;
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
      return isAddress(bitcast.getArg());
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
      if (cast->getNumOperands() == 1)
        return isAddress(cast.getOperand(0));
    if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
      return isAddress(gep.getBase());
    if (auto call = value.getDefiningOp<mlir::LLVM::CallOp>()) {
      return contract::has(call.getOperation(),
                           AsyncSafetyAttrs::kRuntimeValueStorage);
    }
    if (auto call = value.getDefiningOp<mlir::func::CallOp>()) {
      return contract::has(call.getOperation(),
                           AsyncSafetyAttrs::kRuntimeValueStorage);
    }
    return false;
  }
};
} // namespace async_runtime

} // namespace py
