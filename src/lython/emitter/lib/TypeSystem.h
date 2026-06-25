#pragma once

#include "Ast.h"
#include "cpp/PyDialectTypes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <optional>
#include <string>

namespace lython::emitter {

struct FunctionSignature {
  py::CallableType callable;
  llvm::SmallVector<std::string, 8> positionalNames;
  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  llvm::SmallVector<std::string, 4> kwOnlyNames;
  llvm::SmallVector<mlir::Type, 4> kwOnlyTypes;
  llvm::SmallVector<bool, 8> positionalDefaults;
  llvm::SmallVector<bool, 4> kwOnlyDefaults;
  mlir::Type varargType;
  mlir::Type kwargType;
  std::optional<std::string> varargName;
  std::optional<std::string> kwargName;
  unsigned positionalOnlyCount = 0;
  mlir::Type resultType;
};

struct CallKeywordType {
  std::string name;
  mlir::Type type;
};

struct CallInferenceEvidence {
  mlir::Type callableContract;
  std::string methodName;
  std::optional<std::string> receiverManifestClass;
};

struct CallInferenceResult {
  mlir::Type resultType;
  CallInferenceEvidence evidence;
  bool resolved = false;

  explicit operator bool() const {
    return resolved && static_cast<bool>(resultType);
  }
};

class AlgorithmM {
public:
  class Scope {
  public:
    Scope() = default;
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    Scope(Scope &&other) noexcept;
    Scope &operator=(Scope &&other) noexcept;
    ~Scope();

  private:
    friend class AlgorithmM;
    explicit Scope(const AlgorithmM &owner) : owner(&owner) {}
    void reset();

    const AlgorithmM *owner = nullptr;
  };

  explicit AlgorithmM(mlir::MLIRContext &context);

  mlir::MLIRContext &getContext() const { return context; }
  void seedBuiltins();

  mlir::Type object() const;
  mlir::Type none() const;
  mlir::Type boolType() const;
  mlir::Type intType() const;
  mlir::Type strType() const;
  mlir::Type floatType() const;
  mlir::Type contract(llvm::StringRef name,
                      mlir::ArrayRef<mlir::Type> arguments = {}) const;
  mlir::Type protocol(llvm::StringRef name,
                      mlir::ArrayRef<mlir::Type> arguments = {}) const;
  mlir::Type literal(llvm::StringRef spelling) const;
  mlir::Type typeObject(mlir::Type instanceType) const;
  mlir::Type tupleOf(mlir::Type elementType) const;
  mlir::Type listOf(mlir::Type elementType) const;
  mlir::Type dictOf(mlir::Type keyType, mlir::Type valueType) const;
  mlir::Type iteratorOf(mlir::Type elementType) const;

  Scope pushScope() const;
  void bindSymbol(llvm::StringRef name, mlir::Type type);
  std::optional<mlir::Type> lookupSymbol(llvm::StringRef name) const;
  void bindClass(llvm::StringRef name, mlir::Type instanceType);
  std::optional<mlir::Type> lookupClass(llvm::StringRef name) const;

  mlir::Type annotationType(const parser::Node *node) const;
  mlir::Type inferExpr(const parser::Node *node) const;
  CallInferenceResult
  inferCallWithEvidence(mlir::Type calleeType,
                        mlir::ArrayRef<mlir::Type> positional,
                        mlir::ArrayRef<CallKeywordType> keywords) const;
  CallInferenceResult inferMethodCallWithEvidence(
      mlir::Type receiverType, llvm::StringRef methodName,
      mlir::ArrayRef<mlir::Type> positional,
      mlir::ArrayRef<CallKeywordType> keywords = {}) const;
  mlir::Type inferCall(mlir::Type calleeType,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords) const;
  mlir::Type join(mlir::ArrayRef<mlir::Type> types) const;
  mlir::Type widenLiteral(mlir::Type type) const;

  FunctionSignature
  functionSignature(const parser::Node &function,
                    std::optional<llvm::StringRef> selfName = std::nullopt,
                    py::CallableType expectedCallable = {}) const;
  void refreshCallable(FunctionSignature &sig) const;

private:
  void popScope() const;
  void bindLocalSymbol(llvm::StringRef name, mlir::Type type) const;

  mlir::MLIRContext &context;
  llvm::StringMap<mlir::Type> symbols;
  llvm::StringMap<mlir::Type> classes;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> scopes;
};

} // namespace lython::emitter
