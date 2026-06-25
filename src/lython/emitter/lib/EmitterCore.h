#pragma once

#include "Emitter.h"
#include "TypeSystem.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace lython::emitter {

class ModuleEmitter {
public:
  ModuleEmitter(const parser::Node &moduleNode, mlir::MLIRContext &context,
                std::string moduleName, std::string sourceName);

  EmitResult emit();

private:
  struct Value {
    mlir::Value value;
    mlir::Type type;
  };

  struct Capture {
    std::string name;
    Value value;
  };

  mlir::Location loc(const parser::Node &node) const;
  mlir::Type callableProtocol() const;
  mlir::Type callProtocolFor(mlir::Type calleeType) const;
  mlir::Type callProtocolFor(const CallInferenceResult &inference,
                             mlir::Type fallback = {}) const;
  mlir::Type boolProtocol() const;

  void predeclareTopLevel();
  void emitTopLevelDeclarations();
  void emitFunctionDecl(const parser::Node &function);
  void emitCallableFunction(const parser::Node &callable,
                            llvm::StringRef symbolName,
                            const FunctionSignature &sig,
                            llvm::ArrayRef<Capture> captures, bool isLambda);
  Value emitNestedFunctionDecl(const parser::Node &function);
  Value emitLambda(const parser::Node &expr, py::CallableType expected = {});
  void emitClassContract(const parser::Node &classDef);
  void collectClassFields(const parser::Node &classDef,
                          llvm::StringMap<mlir::Type> &fields) const;

  void emitStatements(const std::vector<parser::NodePtr> *statements,
                      bool skipDeclarations = false);
  void emitStatement(const parser::Node &statement);
  void emitAssignTarget(const parser::Node &target, Value value);
  void emitIf(const parser::Node &statement);
  void emitFor(const parser::Node &statement);
  void emitWith(const parser::Node &statement, bool async);

  Value emitExpr(const parser::Node *expr);
  Value emitConstant(const parser::Node &expr);
  Value emitCall(const parser::Node &expr);
  Value emitClassInstantiation(const parser::Node &expr, llvm::StringRef name,
                               mlir::Type instanceType);
  Value emitUnary(const parser::Node &expr);
  Value emitBinary(const parser::Node &expr);
  Value emitCompare(const parser::Node &expr);
  Value emitSubscript(const parser::Node &expr);
  Value emitAttribute(const parser::Node &expr);
  Value emitContainerLiteral(const parser::Node &expr);
  Value emitBindingRef(const parser::Node &anchor, llvm::StringRef binding,
                       mlir::Type type, llvm::ArrayRef<Value> captures = {});
  Value emitFunctionObject(const parser::Node &anchor,
                           llvm::StringRef symbolName, mlir::Type type,
                           llvm::ArrayRef<Capture> captures);
  Value emitNone(const parser::Node &anchor);
  Value emitPack(mlir::ArrayRef<Value> values);
  Value coerceValue(Value value, mlir::Type targetType,
                    const parser::Node &anchor);
  mlir::Value emitBoolValue(Value value, const parser::Node &anchor);

  template <typename Op>
  Value emitBinarySpecial(const parser::Node &anchor, llvm::StringRef method,
                          Value lhs, Value rhs, mlir::Type resultType);
  template <typename Op>
  Value emitUnarySpecial(const parser::Node &anchor, llvm::StringRef method,
                         Value input, mlir::Type resultType);

  mlir::ModuleOp module;
  const parser::Node &moduleNode;
  mlir::MLIRContext &context;
  std::string moduleName;
  std::string sourceName;
  mlir::OpBuilder builder;
  AlgorithmM types;
  parser::Diagnostics diagnostics;
  llvm::StringMap<Value> values;
  mlir::Type currentReturnType;
  std::string currentFunctionPrefix;
  unsigned syntheticFunctionCounter = 0;
};

} // namespace lython::emitter
