#pragma once

#include "Emitter.h"
#include "EmitterState.h"
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
  mlir::Location loc(const parser::Node &node) const;
  mlir::Type callableProtocol() const;
  mlir::Type callProtocolFor(mlir::Type calleeType) const;
  mlir::Type callProtocolFor(const CallInferenceResult &inference,
                             mlir::Type fallback = {}) const;
  mlir::Type boolProtocol() const;
  mlir::Type coroutineType(mlir::Type resultType) const;
  FunctionSignature asyncPublicSignature(FunctionSignature sig) const;

  void predeclareTopLevel();
  void emitTopLevelDeclarations();
  bool bindImportStatement(const parser::Node &statement,
                           bool diagnoseUnsupported);
  void emitFunctionDecl(const parser::Node &function);
  void emitCallableFunction(const parser::Node &callable,
                            llvm::StringRef symbolName,
                            const FunctionSignature &sig,
                            llvm::ArrayRef<Capture> captures, bool isLambda);
  std::optional<MethodBinding>
  lookupClassMethod(mlir::Type receiverType, llvm::StringRef methodName) const;
  std::optional<mlir::Type> lookupClassField(mlir::Type receiverType,
                                             llvm::StringRef fieldName) const;
  Value emitNestedFunctionDecl(const parser::Node &function);
  Value emitLambda(const parser::Node &expr, py::CallableType expected = {});
  void emitClassContract(const parser::Node &classDef);
  void collectClassFields(const parser::Node &classDef,
                          llvm::StringMap<mlir::Type> &fields) const;
  void collectStaticClassAssignments(
      const parser::Node &classDef, llvm::SmallVectorImpl<std::string> &names,
      llvm::SmallVectorImpl<mlir::Attribute> &values) const;
  void collectStaticModuleAssignments(
      const parser::Node &moduleNode, llvm::SmallVectorImpl<std::string> &names,
      llvm::SmallVectorImpl<mlir::Attribute> &values) const;

  void emitStatements(const std::vector<parser::NodePtr> *statements,
                      bool skipDeclarations = false);
  void emitStatement(const parser::Node &statement);
  void emitAssignTarget(const parser::Node &target, Value value);
  void emitIf(const parser::Node &statement);
  void emitFor(const parser::Node &statement);
  void emitAsyncFor(const parser::Node &statement);
  void emitTry(const parser::Node &statement);
  void emitWith(const parser::Node &statement, bool async);
  void emitWithCleanup(const parser::Node &anchor, const WithCleanup &cleanup);
  void emitActiveCleanups(const parser::Node &anchor);

  Value emitExpr(const parser::Node *expr);
  Value emitConstant(const parser::Node &expr);
  Value emitCall(const parser::Node &expr);
  CallOperands emitCallOperands(const parser::Node &expr,
                                llvm::ArrayRef<Value> leadingPositional = {},
                                bool includeAstArguments = true);
  Value emitCallableDispatch(const parser::Node &anchor, Value callee,
                             const CallOperands &operands,
                             mlir::Type resultOverride = {});
  Value emitInlineMethodCall(const parser::Node &expr, Value receiver,
                             const MethodBinding &method);
  Value emitInlineMethodBody(const parser::Node &anchor, Value receiver,
                             const MethodBinding &method,
                             llvm::ArrayRef<Value> positional,
                             const llvm::StringMap<Value> &keywords);
  Value emitClassInstantiation(const parser::Node &expr, llvm::StringRef name,
                               mlir::Type instanceType);
  Value emitUnary(const parser::Node &expr);
  Value emitBinary(const parser::Node &expr);
  Value emitCompare(const parser::Node &expr);
  Value emitSubscript(const parser::Node &expr);
  Value emitAttribute(const parser::Node &expr);
  Value emitAwait(const parser::Node &expr);
  Value emitAwaitValue(const parser::Node &anchor, Value awaitable);
  Value emitContainerLiteral(const parser::Node &expr);
  Value emitBindingRef(const parser::Node &anchor, llvm::StringRef binding,
                       mlir::Type type, llvm::ArrayRef<Value> captures = {});
  Value emitFunctionObject(const parser::Node &anchor,
                           llvm::StringRef symbolName, mlir::Type type,
                           llvm::ArrayRef<Capture> captures);
  std::optional<Value>
  emitPrimitiveConstructorCall(const parser::Node &expr,
                               const parser::Node *calleeNode);
  std::optional<Value> emitPrimitiveRuntimeCall(const parser::Node &expr,
                                                const parser::Node *calleeNode);
  std::optional<Value>
  emitDirectPrimitiveFunctionCall(const parser::Node &expr,
                                  const parser::Node *calleeNode);
  std::optional<Value> emitPrimitiveBinary(const parser::Node &expr, Value lhs,
                                           Value rhs, const parser::Node *op);
  std::optional<Value> emitPrimitiveCompare(const parser::Node &expr, Value lhs,
                                            Value rhs, const parser::Node *op);
  Value emitPrimitiveConstant(const parser::Node &anchor,
                              const PrimitiveConstant &constant);
  Value coercePrimitiveInteger(Value value, mlir::IntegerType targetType,
                               const parser::Node &anchor);
  Value emitNone(const parser::Node &anchor);
  Value emitPack(mlir::ArrayRef<Value> values,
                 llvm::ArrayRef<char> unpacked = {});
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
  llvm::StringMap<PrimitiveConstant> primitiveConstants;
  llvm::StringMap<llvm::StringMap<mlir::Type>> classFieldBindings;
  llvm::StringMap<llvm::StringMap<MethodBinding>> classMethodBindings;
  mlir::Type currentReturnType;
  std::string currentFunctionPrefix;
  unsigned syntheticFunctionCounter = 0;
  llvm::SmallVector<WithCleanup, 8> activeWithCleanups;
  llvm::SmallVector<InlineReturnContext, 4> inlineReturnContexts;
};

} // namespace lython::emitter
