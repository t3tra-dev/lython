#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace py {

inline std::optional<llvm::StringRef> getDirectCallee(mlir::Operation *op) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    return call.getCallee();
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
    return call.getCallee();
  return std::nullopt;
}

inline mlir::Operation *lookupCallableSymbol(mlir::Operation *op,
                                             llvm::StringRef name) {
  if (!op)
    return nullptr;
  auto symbol = mlir::StringAttr::get(op->getContext(), name);
  if (auto func =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              op, symbol))
    return func.getOperation();
  if (auto func =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              op, symbol))
    return func.getOperation();
  return nullptr;
}

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
struct Callee {
  static bool executeFunction(llvm::StringRef callee) {
    return callee.starts_with("async_execute_fn");
  }

  static bool awaitAndExecute(llvm::StringRef callee) {
    return runtime::mlir_async::Callee::awaitAndExecute(callee);
  }

  static bool refcount(llvm::StringRef callee) {
    return runtime::mlir_async::Callee::refcount(callee);
  }

  static bool createHandle(llvm::StringRef callee) {
    return runtime::mlir_async::Callee::createHandle(callee);
  }

  static bool isError(llvm::StringRef callee) {
    return runtime::mlir_async::Callee::isError(callee);
  }

  static bool known(llvm::StringRef callee) {
    return runtime::mlir_async::Callee::known(callee);
  }
};

struct Handle {
  static bool ownedResult(llvm::StringRef callee, mlir::Operation *calleeOp) {
    return Callee::createHandle(callee) || Callee::executeFunction(callee) ||
           hasPresplitCoroutinePassthrough(calleeOp);
  }

  static bool isCallee(llvm::StringRef callee, mlir::Operation *calleeOp) {
    return Callee::known(callee) || Callee::executeFunction(callee) ||
           hasPresplitCoroutinePassthrough(calleeOp);
  }

  static bool transferToExecute(llvm::StringRef callee) {
    return Callee::executeFunction(callee);
  }

  static llvm::SmallVector<unsigned, 2>
  borrowedOperands(llvm::StringRef callee) {
    return runtime::mlir_async::Callee::borrowedHandleOperands(callee);
  }

  static bool ownedChildProducer(mlir::Operation *op) {
    auto callee = getDirectCallee(op);
    if (!callee)
      return false;
    mlir::Operation *calleeOp = lookupCallableSymbol(op, *callee);
    return Callee::executeFunction(*callee) ||
           hasPresplitCoroutinePassthrough(calleeOp);
  }
};

struct Entry {
  static bool isFunction(mlir::Operation *parent) {
    auto symbol = mlir::dyn_cast_or_null<mlir::SymbolOpInterface>(parent);
    return symbol && (Callee::executeFunction(symbol.getName()) ||
                      hasPresplitCoroutinePassthrough(parent));
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
      auto callee = call.getCallee();
      return callee && runtime::mlir_async::Callee::valueStorage(*callee);
    }
    if (auto call = value.getDefiningOp<mlir::func::CallOp>())
      return runtime::mlir_async::Callee::valueStorage(call.getCallee());
    return false;
  }
};
} // namespace async_runtime

} // namespace py
