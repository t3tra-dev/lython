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
};

struct PrimitiveConstant {
  mlir::Type type;
  std::int64_t integerValue = 0;
};

struct CallOperands {
  llvm::SmallVector<Value, 8> positional;
  llvm::SmallVector<char, 8> positionalUnpacked;
  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  llvm::SmallVector<Value, 4> keywordNames;
  llvm::SmallVector<Value, 4> keywordValues;
  llvm::SmallVector<CallKeywordType, 4> keywordTypes;
};

class ScopedEmitterScope {
public:
  ScopedEmitterScope(llvm::StringMap<Value> &values, const AlgorithmM &types)
      : values(values), savedValues(values), typeScope(types.pushScope()) {}

  ScopedEmitterScope(const ScopedEmitterScope &) = delete;
  ScopedEmitterScope &operator=(const ScopedEmitterScope &) = delete;

  ~ScopedEmitterScope() { values = savedValues; }

private:
  llvm::StringMap<Value> &values;
  llvm::StringMap<Value> savedValues;
  AlgorithmM::Scope typeScope;
};

class ScopedCallableEmission {
public:
  ScopedCallableEmission(llvm::StringMap<Value> &values,
                         mlir::Type &currentReturnType,
                         std::string &currentFunctionPrefix,
                         const AlgorithmM &types)
      : values(values), savedValues(values),
        currentReturnType(currentReturnType),
        savedReturnType(currentReturnType),
        currentFunctionPrefix(currentFunctionPrefix),
        savedFunctionPrefix(currentFunctionPrefix),
        typeScope(types.pushScope()) {}

  ScopedCallableEmission(const ScopedCallableEmission &) = delete;
  ScopedCallableEmission &operator=(const ScopedCallableEmission &) = delete;

  ~ScopedCallableEmission() {
    values = savedValues;
    currentReturnType = savedReturnType;
    currentFunctionPrefix = savedFunctionPrefix;
  }

private:
  llvm::StringMap<Value> &values;
  llvm::StringMap<Value> savedValues;
  mlir::Type &currentReturnType;
  mlir::Type savedReturnType;
  std::string &currentFunctionPrefix;
  std::string savedFunctionPrefix;
  AlgorithmM::Scope typeScope;
};

} // namespace lython::emitter
