#pragma once

#include "Runtime/Model/Contracts.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::runtime_lowering {

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
  std::string resultEvidence;
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
  llvm::ArrayRef<RuntimeSymbol> lookupAll(llvm::StringRef contract,
                                          llvm::StringRef role,
                                          llvm::StringRef name) const;
  std::optional<RuntimeSymbol> initializer(llvm::StringRef contract,
                                           llvm::StringRef name) const;
  std::optional<RuntimeSymbol> method(llvm::StringRef contract,
                                      llvm::StringRef name) const;
  llvm::ArrayRef<RuntimeSymbol>
  methodCandidates(llvm::StringRef contract, llvm::StringRef name) const;
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
  llvm::StringMap<llvm::SmallVector<RuntimeSymbol, 2>> symbolSets;
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

} // namespace py::runtime_lowering
