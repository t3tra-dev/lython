#pragma once

#include "cpp/PyDialectTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::runtime_lowering {

inline constexpr llvm::StringLiteral kManifestContractsAttr{
    "ly.runtime.contracts"};
inline constexpr llvm::StringLiteral kManifestContractAttr{
    "ly.runtime.contract"};
inline constexpr llvm::StringLiteral kManifestMethodAttr{"ly.runtime.method"};
inline constexpr llvm::StringLiteral kManifestInitializerAttr{
    "ly.runtime.initializer"};
inline constexpr llvm::StringLiteral kManifestPrimitiveAttr{
    "ly.runtime.primitive"};
inline constexpr llvm::StringLiteral kManifestBuiltinAttr{"ly.runtime.builtin"};
inline constexpr llvm::StringLiteral kManifestBuiltinLoweringAttr{
    "ly.runtime.builtin_lowering"};
inline constexpr llvm::StringLiteral kManifestBuiltinMethodAttr{
    "ly.runtime.builtin_method"};
inline constexpr llvm::StringLiteral kManifestBuiltinSinkContractAttr{
    "ly.runtime.builtin_sink_contract"};
inline constexpr llvm::StringLiteral kManifestShapeAttr{"ly.runtime.shape"};
inline constexpr llvm::StringLiteral kManifestDeallocatorAttr{
    "ly.runtime.deallocator"};
inline constexpr llvm::StringLiteral kManifestClassIdAttr{
    "ly.runtime.class_id"};
inline constexpr llvm::StringLiteral kManifestClassIdArgumentAttr{
    "ly.runtime.class_id_argument"};
inline constexpr llvm::StringLiteral kManifestDefaultI64Attr{
    "ly.runtime.default_i64"};
inline constexpr llvm::StringLiteral kManifestDefaultF64Attr{
    "ly.runtime.default_f64"};
inline constexpr llvm::StringLiteral kManifestResultContractAttr{
    "ly.runtime.result_contract"};
inline constexpr llvm::StringLiteral kManifestElementContractAttr{
    "ly.runtime.element_contract"};
inline constexpr llvm::StringLiteral kManifestNextContractAttr{
    "ly.runtime.next_contract"};
inline constexpr llvm::StringLiteral kManifestValidResultIndexAttr{
    "ly.runtime.valid_result_index"};
inline constexpr llvm::StringLiteral kCallableDefaultValuesAttr{
    "callable_default_values"};

std::string runtimeKey(llvm::StringRef contract, llvm::StringRef role,
                       llvm::StringRef name);
bool isIntegerLiteralSpelling(llvm::StringRef spelling);
std::string runtimeContractName(mlir::Type type);
mlir::Type runtimeContractType(mlir::MLIRContext *context,
                               llvm::StringRef contract);
bool sameTypeSequence(llvm::ArrayRef<mlir::Type> lhs,
                      llvm::ArrayRef<mlir::Type> rhs);
std::string describeTypeSequence(llvm::ArrayRef<mlir::Type> types);
std::string describeValueTypes(mlir::ValueRange values);
llvm::SmallVector<mlir::Type, 4> takePrefix(llvm::ArrayRef<mlir::Type> types,
                                            unsigned count);
llvm::SmallVector<mlir::Type, 4> takeSlice(llvm::ArrayRef<mlir::Type> types,
                                           unsigned begin, unsigned end);

struct RuntimeDefaultArgument {
  enum class Kind { I64, F64 };

  unsigned inputIndex = 0;
  Kind kind = Kind::I64;
  mlir::Attribute value;
};

struct RuntimeSymbol {
  mlir::func::FuncOp function;
  std::string contract;
  std::string role;
  std::string name;
  std::string resultContract;
  std::string elementContract;
  std::string nextContract;
  std::string builtinName;
  std::string builtinLowering;
  std::string builtinMethod;
  std::string builtinSinkContract;
  llvm::SmallVector<unsigned, 1> classIdArgumentIndices;
  llvm::SmallVector<RuntimeDefaultArgument, 2> defaultArguments;
  std::optional<unsigned> validResultIndex;

  bool hasClassIdArgument(unsigned inputIndex) const;
  const RuntimeDefaultArgument *defaultArgument(unsigned inputIndex) const;
};

struct RuntimeValueShape {
  llvm::SmallVector<mlir::Type, 4> valueTypes;
  std::string source;
};

struct RuntimeShapeDefinition {
  mlir::func::FuncOp function;
  std::string contract;
  llvm::SmallVector<mlir::Type, 4> valueTypes;
  std::string source;
};

struct RuntimeClassIdDefinition {
  mlir::func::FuncOp function;
  std::string contract;
  std::int64_t classId = 0;
};

struct RuntimeSymbolDuplicate {
  mlir::func::FuncOp first;
  mlir::func::FuncOp duplicate;
  std::string contract;
  std::string role;
  std::string name;
};

struct RuntimeBuiltinDuplicate {
  mlir::func::FuncOp first;
  mlir::func::FuncOp duplicate;
  std::string name;
};

class RuntimeManifestIndex {
public:
  explicit RuntimeManifestIndex(mlir::ModuleOp module);

  std::optional<RuntimeSymbol> lookup(llvm::StringRef contract,
                                      llvm::StringRef role,
                                      llvm::StringRef name) const;
  std::optional<RuntimeSymbol> initializer(llvm::StringRef contract,
                                           llvm::StringRef name) const;
  std::optional<RuntimeSymbol> method(llvm::StringRef contract,
                                      llvm::StringRef name) const;
  std::optional<RuntimeSymbol> primitive(llvm::StringRef contract,
                                         llvm::StringRef name) const;
  std::optional<RuntimeSymbol> builtinCallable(llvm::StringRef name) const;
  const RuntimeValueShape *valueShape(llvm::StringRef contract) const;
  std::optional<std::int64_t> classId(llvm::StringRef contract) const;
  mlir::LogicalResult verify();

private:
  void recordDeclaredContracts(mlir::ModuleOp module);
  void recordValueShape(llvm::StringRef contract,
                        mlir::ArrayRef<mlir::Type> types,
                        llvm::StringRef source);
  void recordDeallocatorShape(mlir::func::FuncOp function,
                              llvm::StringRef contract);
  void recordResultShape(mlir::func::FuncOp function, llvm::StringRef contract);
  void recordClassId(mlir::func::FuncOp function, llvm::StringRef contract);
  void record(mlir::func::FuncOp function, llvm::StringRef contract,
              llvm::StringRef role, llvm::StringRef name);
  void recordBuiltin(const RuntimeSymbol &symbol);
  void build(mlir::ModuleOp module);
  const RuntimeValueShape *requireShape(mlir::func::FuncOp function,
                                        llvm::StringRef contract,
                                        llvm::StringRef purpose);
  mlir::LogicalResult verifyTypeSequence(mlir::func::FuncOp function,
                                         llvm::StringRef label,
                                         llvm::StringRef contract,
                                         llvm::ArrayRef<mlir::Type> actual,
                                         const RuntimeValueShape &expected);
  mlir::LogicalResult verifyReceiverShape(RuntimeSymbol &symbol);
  mlir::LogicalResult verifyResultShape(RuntimeSymbol &symbol,
                                        llvm::StringRef resultContract,
                                        llvm::StringRef label);
  mlir::LogicalResult verifyNextResultPartition(RuntimeSymbol &symbol);
  mlir::LogicalResult verifyClassIdArguments(RuntimeSymbol &symbol);
  mlir::LogicalResult verifyDefaultArguments(RuntimeSymbol &symbol);
  mlir::LogicalResult verifyBuiltinCallable(RuntimeSymbol &symbol);
  mlir::LogicalResult verifySymbol(RuntimeSymbol &symbol);

  llvm::StringMap<RuntimeSymbol> symbols;
  llvm::StringMap<RuntimeSymbol> builtinCallables;
  llvm::StringMap<RuntimeValueShape> valueShapes;
  llvm::StringSet<> declaredContracts;
  llvm::StringMap<std::int64_t> classIds;
  llvm::SmallVector<RuntimeShapeDefinition, 8> shapeDefinitions;
  llvm::SmallVector<RuntimeClassIdDefinition, 8> classIdDefinitions;
  llvm::SmallVector<RuntimeSymbolDuplicate, 8> duplicateSymbols;
  llvm::SmallVector<RuntimeBuiltinDuplicate, 8> duplicateBuiltins;
  mlir::ModuleOp module;
  bool malformedContractsAttr = false;
};

struct RuntimeValue {
  mlir::Type contract;
  llvm::SmallVector<mlir::Value, 4> values;

  static RuntimeValue object(mlir::Type contract, mlir::ValueRange values);
  std::string contractName() const;
};

struct RuntimeBundle {
  enum class Kind { Object, Aggregate, BuiltinCallable, TypeObject };

  Kind kind = Kind::Object;
  mlir::Type contract;
  mlir::Type instanceContract;
  RuntimeValue objectValue;
  llvm::SmallVector<mlir::Value, 4> aggregateOperands;
  std::string binding;
  std::string functionTarget;
  llvm::SmallVector<RuntimeValue, 4> closureValues;

  static RuntimeBundle object(mlir::Type contract, mlir::ValueRange values);
  static RuntimeBundle aggregate(mlir::Type contract,
                                 mlir::ValueRange operands);
  static RuntimeBundle builtinCallable(mlir::Type contract,
                                       llvm::StringRef binding);
  static RuntimeBundle typeObject(mlir::Type typeContract,
                                  mlir::Type instanceContract);
  llvm::ArrayRef<mlir::Value> physicalValues() const;
  std::string contractName() const;
  std::string instanceContractName() const;
};

struct CallableLogicalEntryArgs {
  mlir::func::FuncOp function;
  unsigned count = 0;
};

struct ReturnedCallableSummary {
  std::string target;
  llvm::SmallVector<unsigned, 4> captureArgumentIndices;
};

class RuntimeBundleLowerer {
public:
  explicit RuntimeBundleLowerer(mlir::ModuleOp module);
  mlir::LogicalResult lowerModule();

private:
  struct EmittedRuntimeCall {
    RuntimeSymbol symbol;
    mlir::func::CallOp call;
  };

  const RuntimeValueShape *runtimeValueShapeFor(mlir::Operation *op,
                                                mlir::Type type,
                                                llvm::StringRef purpose) const;
  mlir::LogicalResult
  appendRuntimeValueTypes(mlir::Operation *op, mlir::Type type,
                          llvm::SmallVectorImpl<mlir::Type> &types) const;
  llvm::SmallVector<mlir::Type, 4>
  callableClosureTypes(mlir::func::FuncOp function) const;
  mlir::LogicalResult buildReturnedCallableSummaries();
  mlir::LogicalResult prepareCallableFunctionABIs();
  mlir::LogicalResult
  seedCallableEntryArgumentBundles(mlir::func::FuncOp function,
                                   mlir::ArrayRef<mlir::Type> logicalTypes);
  mlir::LogicalResult validateObjectShape(mlir::Operation *op,
                                          llvm::StringRef contract,
                                          mlir::ValueRange values) const;
  mlir::LogicalResult makeObjectBundle(mlir::Operation *op, mlir::Type contract,
                                       mlir::ValueRange values,
                                       RuntimeBundle &bundle) const;
  bool objectShapeMatches(llvm::StringRef contract,
                          mlir::ValueRange values) const;
  bool rawValuesMatchRuntimeInputs(const RuntimeSymbol &symbol,
                                   mlir::ValueRange values) const;
  mlir::LogicalResult initializeObjectFromRawValues(mlir::Operation *op,
                                                    mlir::Type contract,
                                                    mlir::ValueRange values,
                                                    RuntimeBundle &bundle,
                                                    bool emitErrors = true);
  mlir::LogicalResult bundleRawObjectValues(mlir::Operation *op,
                                            mlir::Type contract,
                                            mlir::ValueRange values,
                                            RuntimeBundle &bundle);
  mlir::LogicalResult materializeDefaultValue(mlir::Operation *op,
                                              mlir::Type parameterType,
                                              mlir::Attribute attr,
                                              RuntimeBundle &bundle);
  mlir::LogicalResult assignObjectBundle(mlir::Operation *op, mlir::Value value,
                                         mlir::Type contract,
                                         mlir::ValueRange values);
  mlir::FailureOr<llvm::StringRef>
  requireMethodTarget(mlir::Operation *op, mlir::FlatSymbolRefAttr target,
                      llvm::StringRef expectedName) const;

  template <typename Op> mlir::LogicalResult lowerAliasViewOp(Op op) {
    return lowerAliasView(op.getOperation(), op.getInput(), op.getResult());
  }

  template <typename Op> mlir::LogicalResult lowerUnaryMethodOp(Op op) {
    mlir::FailureOr<llvm::StringRef> methodName = requireMethodTarget(
        op.getOperation(), op.getTargetAttr(), op.getMethodName());
    if (mlir::failed(methodName))
      return mlir::failure();
    return lowerUnarySpecial(op.getOperation(), op.getInput(), *methodName,
                             op.getResult());
  }

  template <typename Op> mlir::LogicalResult lowerBinaryMethodOp(Op op) {
    mlir::FailureOr<llvm::StringRef> methodName = requireMethodTarget(
        op.getOperation(), op.getTargetAttr(), op.getMethodName());
    if (mlir::failed(methodName))
      return mlir::failure();
    return lowerBinarySpecial(op.getOperation(), op.getLhs(), op.getRhs(),
                              *methodName, op.getResult());
  }

  template <typename Op>
  mlir::LogicalResult lowerNamedUnaryMethodOp(Op op,
                                              llvm::StringRef methodName) {
    mlir::FailureOr<llvm::StringRef> target =
        requireMethodTarget(op.getOperation(), op.getTargetAttr(), methodName);
    if (mlir::failed(target))
      return mlir::failure();
    return lowerUnarySpecial(op.getOperation(), op.getInput(), *target,
                             op.getResult());
  }

  mlir::LogicalResult lowerPyOp(mlir::Operation *op);
  mlir::LogicalResult lowerStrConstant(py::StrConstantOp op);
  bool isStaticKeywordName(py::StrConstantOp op) const;
  mlir::LogicalResult lowerIntConstant(py::IntConstantOp op);
  mlir::LogicalResult lowerFloatConstant(py::FloatConstantOp op);
  mlir::LogicalResult lowerBoolConstant(py::BoolConstantOp op);
  mlir::LogicalResult lowerNone(py::NoneOp op);
  mlir::LogicalResult lowerCastFromPrim(py::CastFromPrimOp op);
  mlir::LogicalResult lowerTypeObject(py::TypeObjectOp op);
  mlir::LogicalResult lowerPack(py::PackOp op);
  mlir::LogicalResult lowerBindingRef(py::BindingRefOp op);
  mlir::LogicalResult lowerFunctionBindingRef(py::BindingRefOp op,
                                              mlir::func::FuncOp function);
  mlir::LogicalResult appendClosureValues(py::BindingRefOp op,
                                          mlir::func::FuncOp function,
                                          RuntimeBundle &bundle);
  mlir::LogicalResult lowerAliasView(mlir::Operation *op, mlir::Value input,
                                     mlir::Value resultValue);
  mlir::LogicalResult collectPackedObjectSources(
      mlir::Operation *op, mlir::Value packValue, llvm::StringRef label,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources) const;
  mlir::LogicalResult requireEmptyAggregate(mlir::Operation *op,
                                            mlir::Value packValue,
                                            llvm::StringRef label) const;
  mlir::LogicalResult verifySelectedRuntimeTarget(mlir::Operation *op,
                                                  RuntimeSymbol &symbol);
  mlir::LogicalResult emitManifestMethodCall(
      mlir::Operation *op, const RuntimeBundle &receiver,
      llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
      bool allowUnusedSources, std::optional<EmittedRuntimeCall> &emitted);
  std::string resultContractFor(mlir::Value resultValue,
                                const RuntimeSymbol &symbol,
                                bool preferManifestObjectResult) const;
  mlir::LogicalResult
  bindRuntimeCallResult(mlir::Operation *op, mlir::Value resultValue,
                        const EmittedRuntimeCall &emitted,
                        bool preferManifestObjectResult = false);
  mlir::LogicalResult lowerManifestMethodResult(
      mlir::Operation *op, mlir::Value resultValue,
      const RuntimeBundle &receiver, llvm::StringRef methodName,
      llvm::ArrayRef<const RuntimeBundle *> sources, bool allowUnusedSources,
      bool preferManifestObjectResult = false);
  mlir::LogicalResult lowerManifestI1MethodResult(
      mlir::Operation *op, mlir::Value resultValue,
      const RuntimeBundle &receiver, llvm::StringRef methodName,
      llvm::ArrayRef<const RuntimeBundle *> sources, bool allowUnusedSources);
  mlir::LogicalResult
  lowerManifestVoidMethod(mlir::Operation *op, const RuntimeBundle &receiver,
                          llvm::StringRef methodName,
                          llvm::ArrayRef<const RuntimeBundle *> sources,
                          bool allowUnusedSources);
  mlir::LogicalResult lowerNew(py::NewOp op);
  mlir::LogicalResult lowerInit(py::InitOp op);
  mlir::LogicalResult lowerRaise(py::RaiseOp op);
  mlir::LogicalResult lowerRaiseCurrent(py::RaiseCurrentOp op);
  mlir::LogicalResult emitTracebackFrame(mlir::Operation *op);
  mlir::LogicalResult lowerCall(py::CallOp op);
  mlir::LogicalResult lowerObjectCallableCall(py::CallOp op,
                                              const RuntimeBundle &callable);
  mlir::LogicalResult lowerFunctionTargetCall(py::CallOp op,
                                              const RuntimeBundle &callable);
  mlir::LogicalResult collectFunctionCallSources(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
      llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults);
  mlir::LogicalResult materializeDefaultArgument(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      unsigned index, mlir::Type parameterType,
      llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
      const RuntimeBundle *&source);
  mlir::LogicalResult applyFunctionKeywordSources(
      py::CallOp op, py::CallableType callable, llvm::StringRef targetName,
      llvm::MutableArrayRef<const RuntimeBundle *> ordered) const;
  std::optional<std::string> keywordNameFromValue(mlir::Value value) const;
  std::optional<unsigned> keywordParameterIndex(py::CallableType callable,
                                                llvm::StringRef keyword) const;
  mlir::LogicalResult lowerReceiverMethodResult(
      mlir::Operation *op, mlir::Value receiverValue, mlir::Value resultValue,
      llvm::StringRef missingSubject, llvm::StringRef methodName,
      bool preferManifestObjectResult = false);
  mlir::LogicalResult lowerBool(py::BoolOp op);
  mlir::LogicalResult lowerLen(py::LenOp op);
  mlir::LogicalResult lowerGetItem(py::GetItemOp op);
  mlir::LogicalResult lowerSetItem(py::SetItemOp op);
  mlir::LogicalResult lowerDelItem(py::DelItemOp op);
  mlir::LogicalResult lowerContains(py::ContainsOp op);
  mlir::LogicalResult lowerIter(py::IterOp op);
  mlir::LogicalResult lowerNext(py::NextOp op);
  mlir::LogicalResult lowerEnter(py::EnterOp op);
  mlir::LogicalResult lowerExit(py::ExitOp op);
  mlir::LogicalResult lowerAEnter(py::AEnterOp op);
  mlir::LogicalResult lowerAExit(py::AExitOp op);
  mlir::LogicalResult lowerAIter(py::AIterOp op);
  mlir::LogicalResult lowerANext(py::ANextOp op);
  mlir::LogicalResult lowerRound(py::RoundOp op);
  mlir::LogicalResult lowerUnarySpecial(mlir::Operation *op, mlir::Value input,
                                        llvm::StringRef methodName,
                                        mlir::Value resultValue);
  bool canAppendExactValueSequence(mlir::FunctionType functionType,
                                   unsigned inputIndex,
                                   const RuntimeBundle &source) const;
  mlir::LogicalResult
  appendRuntimeSource(mlir::Operation *op, const RuntimeSymbol &symbol,
                      mlir::FunctionType functionType, unsigned &inputIndex,
                      const RuntimeBundle &source,
                      llvm::SmallVectorImpl<mlir::Value> &operands);
  mlir::LogicalResult appendImplicitRuntimeArgument(
      mlir::Operation *op, const RuntimeSymbol &symbol, unsigned &inputIndex,
      llvm::SmallVectorImpl<mlir::Value> &operands);
  mlir::LogicalResult
  buildRuntimeCallOperands(mlir::Operation *op, const RuntimeSymbol &symbol,
                           llvm::ArrayRef<const RuntimeBundle *> sources,
                           llvm::SmallVectorImpl<mlir::Value> &operands,
                           bool allowUnusedSources,
                           const RuntimeBundle *classObject = nullptr);
  mlir::LogicalResult lowerBinarySpecial(mlir::Operation *op, mlir::Value lhs,
                                         mlir::Value rhs,
                                         llvm::StringRef methodName,
                                         mlir::Value resultValue);
  mlir::LogicalResult
  collectSingleBuiltinArgument(py::CallOp op, const RuntimeSymbol &symbol,
                               const RuntimeBundle *&argument) const;
  mlir::LogicalResult lowerBuiltinMethodCall(py::CallOp op,
                                             const RuntimeSymbol &symbol);
  mlir::LogicalResult lowerBuiltinMethodSinkCall(py::CallOp op,
                                                 const RuntimeSymbol &symbol);
  mlir::LogicalResult bundleRuntimeResults(mlir::Operation *op,
                                           mlir::Type expectedContract,
                                           mlir::func::CallOp call,
                                           RuntimeBundle &result);
  const RuntimeBundle *bundleFor(mlir::Value value) const;
  mlir::Value materializeByteBuffer(mlir::Location loc, llvm::StringRef text);
  mlir::func::CallOp createRuntimeCall(mlir::Location loc,
                                       const RuntimeSymbol &symbol,
                                       mlir::ValueRange operands);
  std::int64_t functionTargetId(llvm::StringRef target);
  mlir::LogicalResult lowerFunctionReturns();
  mlir::LogicalResult eraseCallableLogicalEntryArgs();
  mlir::LogicalResult eraseLoweredPyOps();

  mlir::ModuleOp module;
  mlir::MLIRContext *context;
  mlir::OpBuilder builder;
  RuntimeManifestIndex manifest;
  llvm::DenseMap<mlir::Value, RuntimeBundle> valueBundles;
  llvm::StringMap<ReturnedCallableSummary> returnedCallableSummaries;
  llvm::StringMap<std::int64_t> functionTargetIds;
  llvm::SmallVector<CallableLogicalEntryArgs, 8> callableLogicalEntryArgCounts;
  std::int64_t nextFunctionTargetId = 1;
  llvm::SmallVector<mlir::Operation *, 32> erase;
};

} // namespace py::runtime_lowering
