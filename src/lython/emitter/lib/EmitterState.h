#pragma once

#include "TypeSystem.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <memory>
#include <string>

namespace lython::emitter {

struct BoundMethodValue;

struct Value {
  mlir::Value value;
  mlir::Type type;
  std::shared_ptr<BoundMethodValue> boundMethod = nullptr;
};

struct Capture {
  std::string name;
  Value value;
};

struct MethodBinding {
  const parser::Node *method = nullptr;
  FunctionSignature bodySignature;
  FunctionSignature signature;
  std::string kind = "instance";
  std::string symbolName;
  bool async = false;
};

struct BoundMethodValue {
  Value receiver;
  MethodBinding method;
};

struct WithCleanup {
  Value manager;
  bool async = false;
};

struct InlineReturnContext {
  mlir::Block *target = nullptr;
  mlir::Type resultType;
  bool carryResult = true;
};

// One loop-carried local: a pre-existing local reassigned in a loop body,
// threaded as a loop-header / after-block argument in this order.
struct CarriedLoopLocal {
  std::string name;
  mlir::Type type;
  // Shaped primitives only: the pre-loop buffer every rebinding of this local
  // is copied into (a bufferization.alloc_tensor dominating the loop). A
  // pre-loop destination keeps the block arguments' buffer types inferable --
  // pinning into the header argument itself leaves nested loops with no
  // cycle-free incoming edge to infer from.
  mlir::Value pinBuffer;
};

struct LoopControlContext {
  LoopControlContext() = default;
  // Two-target form used by every loop site; carriedLocals / headerBlock keep
  // their defaults and are assigned afterwards where needed. A constructor
  // (not aggregate init) so partially-braced sites do not trip
  // -Wmissing-field-initializers.
  LoopControlContext(mlir::Block *breakTarget, mlir::Block *continueTarget)
      : breakTarget(breakTarget), continueTarget(continueTarget) {}

  mlir::Block *breakTarget = nullptr;
  mlir::Block *continueTarget = nullptr;
  // Loop-carried local names, in the block-argument order of breakTarget /
  // continueTarget. Empty when those blocks take no arguments. When non-empty,
  // `break` / `continue` forward the current values of these locals as branch
  // operands and release the replaced loop-header value.
  llvm::SmallVector<CarriedLoopLocal, 4> carriedLocals;
  // The loop-header block whose arguments hold the current-iteration carried
  // values (used to detect replacement for the decref-on-replace on
  // break/continue edges).
  mlir::Block *headerBlock = nullptr;
};

struct PrimitiveConstant {
  mlir::Type type;
  std::int64_t integerValue = 0;
};

struct CallOperands {
  bool valid = true;
  std::string failureReason;
  llvm::SmallVector<Value, 8> positional;
  llvm::SmallVector<char, 8> positionalUnpacked;
  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  llvm::SmallVector<Value, 4> keywordNames;
  llvm::SmallVector<Value, 4> keywordValues;
  llvm::SmallVector<CallKeywordType, 4> keywordTypes;
};

class ScopedEmitterScope {
public:
  ScopedEmitterScope(llvm::StringMap<Value> &values, const TypeSystem &types)
      : values(values), savedValues(values), typeScope(types.pushScope()) {}

  ScopedEmitterScope(const ScopedEmitterScope &) = delete;
  ScopedEmitterScope &operator=(const ScopedEmitterScope &) = delete;

  ~ScopedEmitterScope() { values = savedValues; }

private:
  llvm::StringMap<Value> &values;
  llvm::StringMap<Value> savedValues;
  TypeSystem::Scope typeScope;
};

class ScopedCallableEmission {
public:
  ScopedCallableEmission(llvm::StringMap<Value> &values,
                         mlir::Type &currentReturnType,
                         std::string &currentFunctionPrefix,
                         mlir::Type &currentGeneratorSendType,
                         const TypeSystem &types)
      : values(values), savedValues(values),
        currentReturnType(currentReturnType),
        savedReturnType(currentReturnType),
        currentFunctionPrefix(currentFunctionPrefix),
        savedFunctionPrefix(currentFunctionPrefix),
        currentGeneratorSendType(currentGeneratorSendType),
        savedGeneratorSendType(currentGeneratorSendType),
        typeScope(types.pushScope()) {}

  ScopedCallableEmission(const ScopedCallableEmission &) = delete;
  ScopedCallableEmission &operator=(const ScopedCallableEmission &) = delete;

  ~ScopedCallableEmission() {
    values = savedValues;
    currentReturnType = savedReturnType;
    currentFunctionPrefix = savedFunctionPrefix;
    currentGeneratorSendType = savedGeneratorSendType;
  }

private:
  llvm::StringMap<Value> &values;
  llvm::StringMap<Value> savedValues;
  mlir::Type &currentReturnType;
  mlir::Type savedReturnType;
  std::string &currentFunctionPrefix;
  std::string savedFunctionPrefix;
  mlir::Type &currentGeneratorSendType;
  mlir::Type savedGeneratorSendType;
  TypeSystem::Scope typeScope;
};

} // namespace lython::emitter
