// This file implements optimization passes specific to the Py dialect and
// its lowered LLVM representation. These optimizations include:
//   - Dead tuple cleanup (removing tuples only used by DecRefOp)
//   - Integer constant hoisting and CSE
//   - Small integer DecRef removal (immortal integers -5 to 256)
//   - String creation CSE (LyUnicode_FromUTF8)
//   - Empty tuple replacement (LyTuple_New(0) -> Ly_GetEmptyTuple)
//   - Singleton getter CSE (Ly_GetBuiltinPrint, Ly_GetNone, etc.)
//   - Bool boxing/unboxing elimination (LyBool_FromBool + LyBool_AsBool)
//   - LLVM constant CSE
//   - Dead code elimination for unused LLVM operations

#include "RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

// Pre-lowering optimizations (operate on Py dialect ops)

/// Clean up dead tuple operations whose only users are DecRefOps.
/// When removing a TupleCreateOp, we must add DecRefs for its elements
/// since the tuple's destructor would have handled them.
/// Returns true if any operations were erased.
bool cleanupDeadTuples(ModuleOp module) {
  SmallVector<Operation *> toErase;
  SmallVector<std::pair<TupleCreateOp, SmallVector<Operation *>>> tupleCreates;

  module.walk([&](Operation *tupleOp) {
    if (auto tupleEmpty = dyn_cast<TupleEmptyOp>(tupleOp)) {
      // Empty tuples have no elements, just check for DecRef-only users
      Value result = tupleEmpty->getResult(0);
      SmallVector<Operation *> decrefsToErase;
      bool canErase = true;

      for (Operation *user : result.getUsers()) {
        if (auto decref = dyn_cast<DecRefOp>(user)) {
          decrefsToErase.push_back(decref);
        } else {
          canErase = false;
          break;
        }
      }

      if (canErase && !decrefsToErase.empty()) {
        for (Operation *decref : decrefsToErase)
          toErase.push_back(decref);
        toErase.push_back(tupleOp);
      }
    } else if (auto tupleCreate = dyn_cast<TupleCreateOp>(tupleOp)) {
      Value result = tupleCreate->getResult(0);
      SmallVector<Operation *> decrefsToErase;
      bool canErase = true;

      for (Operation *user : result.getUsers()) {
        if (auto decref = dyn_cast<DecRefOp>(user)) {
          decrefsToErase.push_back(decref);
        } else {
          canErase = false;
          break;
        }
      }

      if (canErase && !decrefsToErase.empty()) {
        tupleCreates.push_back({tupleCreate, std::move(decrefsToErase)});
      }
    }
  });

  // For TupleCreateOps, add DecRefs for elements before erasing
  for (auto &[tupleCreate, decrefs] : tupleCreates) {
    // Get the elements before we erase anything
    SmallVector<Value> elements(tupleCreate.getElements());

    // For each DecRef on the tuple, we need to add DecRefs for elements
    // We only need to do this once (not per-decref), since the tuple
    // would only be destroyed once
    if (!decrefs.empty() && !elements.empty()) {
      // Insert DecRefs for elements after the last user of each element
      // For simplicity, insert right before where the tuple DecRef was
      OpBuilder builder(decrefs.front());
      for (Value element : elements) {
        // Skip if the element is from an immortal operation
        if (Operation *defOp = element.getDefiningOp()) {
          if (isa<NoneOp, FuncObjectOp, TupleEmptyOp>(defOp))
            continue;
        }

        // Check if the element has other users besides this tuple
        // If it does, those users are responsible for managing its refcount
        // Note: CastIdentityOp is transparent and doesn't affect refcount,
        // so we don't count it as a "real" user.
        bool hasOtherUsers = false;
        for (Operation *user : element.getUsers()) {
          if (user == tupleCreate.getOperation())
            continue;
          // CastIdentityOp is transparent - check its users recursively
          if (isa<CastIdentityOp>(user)) {
            // The cast itself doesn't count, but if the cast result
            // has users beyond calls that consume it, that counts
            continue;
          }
          hasOtherUsers = true;
          break;
        }

        // Only add DecRef if the element was created solely for this tuple
        if (!hasOtherUsers) {
          builder.create<DecRefOp>(tupleCreate.getLoc(), element);
        }
      }
    }

    // Now erase the tuple and its decrefs
    for (Operation *decref : decrefs)
      toErase.push_back(decref);
    toErase.push_back(tupleCreate.getOperation());
  }

  for (Operation *op : toErase)
    op->erase();

  return !toErase.empty();
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
            builder.create<LLVM::CallOp>(returnOp.getLoc(), decrefFunc,
                                         ValueRange{cachedCall.getResult()});
          }
        }
      });
    }
  });
}

/// Replace LyTuple_New(0) with Ly_GetEmptyTuple().
/// This uses a pre-allocated immortal empty tuple singleton.
void replaceEmptyTupleNew(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  SmallVector<LLVM::CallOp> toReplace;

  module.walk([&](LLVM::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (!callee || *callee != "LyTuple_New")
      return;

    // Check if argument is constant 0
    if (callOp.getNumOperands() != 1)
      return;
    auto sizeConst = callOp.getOperand(0).getDefiningOp<LLVM::ConstantOp>();
    if (!sizeConst)
      return;
    auto sizeAttr = llvm::dyn_cast<IntegerAttr>(sizeConst.getValue());
    if (!sizeAttr || sizeAttr.getInt() != 0)
      return;

    toReplace.push_back(callOp);
  });

  for (auto callOp : toReplace) {
    OpBuilder builder(callOp);

    // Get or create Ly_GetEmptyTuple function declaration
    auto emptyTupleFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>("Ly_GetEmptyTuple");
    if (!emptyTupleFunc) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      auto ptrType = LLVM::LLVMPointerType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(ptrType, {});
      emptyTupleFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          module.getLoc(), "Ly_GetEmptyTuple", funcType);
    }

    auto newCall = builder.create<LLVM::CallOp>(callOp.getLoc(), emptyTupleFunc,
                                                ValueRange{});
    callOp.getResult().replaceAllUsesWith(newCall.getResult());
    callOp->erase();
  }
}

/// CSE for singleton getter calls (Ly_GetBuiltinPrint, Ly_GetNone,
/// Ly_GetEmptyTuple). These return borrowed references to singletons, so no
/// DecRef needed.
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
      if (*callee != "Ly_GetBuiltinPrint" && *callee != "Ly_GetNone" &&
          *callee != "Ly_GetEmptyTuple")
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

    // Only optimize if the pattern matches exactly:
    // - One LyBool_AsBool call
    // - One Ly_DecRef call
    // - No other users
    if (!asBoolCall || !decRefCall || hasOtherUsers)
      continue;

    // Replace uses of the AsBool result with the original i1 value
    asBoolCall.getResult().replaceAllUsesWith(i1Value);

    // Erase the operations (in reverse order of dependencies)
    decRefCall->erase();
    asBoolCall->erase();
    fromBoolCall->erase();
  }
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

// PyOptimizationPass: Aggregates all Py-specific optimizations

struct PyOptimizationPass
    : public PassWrapper<PyOptimizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PyOptimizationPass)

  StringRef getArgument() const override { return "py-optimize"; }

  StringRef getDescription() const override {
    return "Apply Py dialect-specific optimizations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Delegate to helper functions to avoid code duplication
    runPreLoweringOptimizations(module);
    runPostLoweringOptimizations(module);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPyOptimizationPass() {
  return std::make_unique<PyOptimizationPass>();
}

void runPreLoweringOptimizations(ModuleOp module) {
  cleanupDeadTuples(module);
  hoistIntConstants(module);
  removeSmallIntDecrefs(module);
}

void runPostLoweringOptimizations(ModuleOp module) {
  cseStringCreation(module);
  replaceEmptyTupleNew(module);
  cseSingletonGetters(module);
  eliminateBoolBoxingUnboxing(module);
  cseConstants(module);
  eliminateDeadCode(module);
}

} // namespace py
