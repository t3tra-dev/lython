#include "Runtime/Manifest/Index.h"

namespace py::runtime_lowering {

namespace {

class VerificationResult {
public:
  void fail() { result = mlir::failure(); }

  void check(mlir::LogicalResult candidate) {
    if (mlir::failed(candidate))
      fail();
  }

  mlir::LogicalResult get() const { return result; }

private:
  mlir::LogicalResult result = mlir::success();
};

} // namespace

RuntimeManifestIndex::RuntimeManifestIndex(mlir::ModuleOp module)
    : module(module) {
  build(module);
}

std::optional<RuntimeSymbol>
RuntimeManifestIndex::lookup(llvm::StringRef contract, llvm::StringRef role,
                             llvm::StringRef name) const {
  llvm::ArrayRef<RuntimeSymbol> candidates = lookupAll(contract, role, name);
  if (candidates.empty())
    return std::nullopt;
  return candidates.front();
}

llvm::ArrayRef<RuntimeSymbol>
RuntimeManifestIndex::lookupAll(llvm::StringRef contract, llvm::StringRef role,
                                llvm::StringRef name) const {
  auto found = symbolSets.find(runtimeKey(contract, role, name));
  if (found == symbolSets.end())
    return {};
  return found->second;
}

std::optional<RuntimeSymbol>
RuntimeManifestIndex::initializer(llvm::StringRef contract,
                                  llvm::StringRef name) const {
  return lookup(contract, "initializer", name);
}

std::optional<RuntimeSymbol>
RuntimeManifestIndex::method(llvm::StringRef contract,
                             llvm::StringRef name) const {
  return lookup(contract, "method", name);
}

llvm::ArrayRef<RuntimeSymbol>
RuntimeManifestIndex::methodCandidates(llvm::StringRef contract,
                                       llvm::StringRef name) const {
  return lookupAll(contract, "method", name);
}

std::optional<RuntimeSymbol>
RuntimeManifestIndex::primitive(llvm::StringRef contract,
                                llvm::StringRef name) const {
  return lookup(contract, "primitive", name);
}

std::optional<RuntimeSymbol>
RuntimeManifestIndex::builtinCallable(llvm::StringRef name) const {
  auto found = builtinCallables.find(name);
  if (found == builtinCallables.end())
    return std::nullopt;
  return found->second;
}

const RuntimeValueShape *
RuntimeManifestIndex::valueShape(llvm::StringRef contract) const {
  auto found = valueShapes.find(contract);
  if (found == valueShapes.end())
    return nullptr;
  return &found->second;
}

std::optional<std::int64_t>
RuntimeManifestIndex::classId(llvm::StringRef contract) const {
  auto found = classIds.find(contract);
  if (found == classIds.end())
    return std::nullopt;
  return found->second;
}

mlir::LogicalResult RuntimeManifestIndex::verify() {
  VerificationResult verified;

  for (RuntimeSymbolDuplicate &duplicate : duplicateSymbols) {
    duplicate.duplicate.emitError()
        << "duplicate runtime manifest symbol for " << duplicate.contract << " "
        << duplicate.role << " " << duplicate.name << "; first definition is @"
        << duplicate.first.getSymName();
    verified.fail();
  }

  for (RuntimeBuiltinDuplicate &duplicate : duplicateBuiltins) {
    duplicate.duplicate.emitError()
        << "duplicate runtime builtin binding " << duplicate.name
        << "; first definition is @" << duplicate.first.getSymName();
    verified.fail();
  }

  if (malformedContractsAttr)
    verified.fail();

  for (const auto &entry : declaredContracts) {
    llvm::StringRef contract = entry.getKey();
    if (valueShape(contract))
      continue;
    module.emitError() << "runtime contract " << contract << " is declared in "
                       << kManifestContractsAttr << " but has no ABI shape";
    verified.fail();
  }

  for (RuntimeShapeDefinition &definition : shapeDefinitions) {
    const RuntimeValueShape *shape = valueShape(definition.contract);
    if (!shape)
      continue;
    if (sameTypeSequence(definition.valueTypes, shape->valueTypes))
      continue;
    definition.function.emitError()
        << "runtime value shape for " << definition.contract << " from "
        << definition.source << " is "
        << describeTypeSequence(definition.valueTypes)
        << ", but canonical shape from " << shape->source << " is "
        << describeTypeSequence(shape->valueTypes);
    verified.fail();
  }

  for (RuntimeClassIdDefinition &definition : classIdDefinitions) {
    std::optional<std::int64_t> expected = classId(definition.contract);
    if (!expected || *expected == definition.classId)
      continue;
    definition.function.emitError()
        << "runtime class id for " << definition.contract << " is "
        << definition.classId << ", but canonical class id is " << *expected;
    verified.fail();
  }

  for (auto &entry : symbolSets)
    for (RuntimeSymbol &symbol : entry.second)
      verified.check(verifySymbol(symbol));

  return verified.get();
}

void RuntimeManifestIndex::recordDeclaredContracts(mlir::ModuleOp module) {
  auto contracts =
      module->getAttrOfType<mlir::ArrayAttr>(kManifestContractsAttr);
  if (!contracts)
    return;
  for (mlir::Attribute attr : contracts) {
    auto contract = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!contract) {
      module.emitError() << kManifestContractsAttr
                         << " entries must be strings";
      malformedContractsAttr = true;
      continue;
    }
    declaredContracts.insert(contract.getValue());
  }
}

void RuntimeManifestIndex::recordValueShape(llvm::StringRef contract,
                                            mlir::ArrayRef<mlir::Type> types,
                                            llvm::StringRef source) {
  if (valueShapes.find(contract) != valueShapes.end())
    return;
  RuntimeValueShape &shape = valueShapes[contract];
  shape.valueTypes.assign(types.begin(), types.end());
  shape.source = source.str();
}

void RuntimeManifestIndex::recordDeallocatorShape(mlir::func::FuncOp function,
                                                  llvm::StringRef contract) {
  llvm::SmallVector<mlir::Type, 4> types;
  types.append(function.getFunctionType().getInputs().begin(),
               function.getFunctionType().getInputs().end());
  shapeDefinitions.push_back(RuntimeShapeDefinition{
      function, contract.str(), types, function.getSymName().str()});
  recordValueShape(contract, types, function.getSymName());
}

void RuntimeManifestIndex::recordResultShape(mlir::func::FuncOp function,
                                             llvm::StringRef contract) {
  llvm::SmallVector<mlir::Type, 4> types;
  types.append(function.getFunctionType().getResults().begin(),
               function.getFunctionType().getResults().end());
  std::string source = (function.getSymName() + ".shape").str();
  shapeDefinitions.push_back(
      RuntimeShapeDefinition{function, contract.str(), types, source});
  recordValueShape(contract, types, source);
}

void RuntimeManifestIndex::recordClassId(mlir::func::FuncOp function,
                                         llvm::StringRef contract) {
  auto attr = function->getAttrOfType<mlir::IntegerAttr>(kManifestClassIdAttr);
  if (!attr)
    return;
  std::int64_t classIdValue = attr.getInt();
  classIdDefinitions.push_back(
      RuntimeClassIdDefinition{function, contract.str(), classIdValue});
  if (classIds.find(contract) == classIds.end())
    classIds[contract] = classIdValue;
}

void RuntimeManifestIndex::record(mlir::func::FuncOp function,
                                  llvm::StringRef contract,
                                  llvm::StringRef role, llvm::StringRef name) {
  auto stringAttr = [&](llvm::StringLiteral attrName) -> std::string {
    if (auto attr = function->getAttrOfType<mlir::StringAttr>(attrName))
      return attr.getValue().str();
    return "";
  };

  std::optional<unsigned> validResultIndex;
  if (auto attr = function->getAttrOfType<mlir::IntegerAttr>(
          kManifestValidResultIndexAttr))
    validResultIndex = static_cast<unsigned>(attr.getInt());

  llvm::SmallVector<unsigned, 1> classIdArgumentIndices;
  llvm::SmallVector<RuntimeDefaultArgument, 2> defaultArguments;
  for (unsigned index = 0, end = function.getFunctionType().getNumInputs();
       index < end; ++index) {
    if (function.getArgAttr(index, kManifestClassIdArgumentAttr))
      classIdArgumentIndices.push_back(index);
    if (mlir::Attribute attr =
            function.getArgAttr(index, kManifestDefaultI64Attr))
      defaultArguments.push_back(RuntimeDefaultArgument{
          index, RuntimeDefaultArgument::Kind::I64, attr});
    if (mlir::Attribute attr =
            function.getArgAttr(index, kManifestDefaultF64Attr))
      defaultArguments.push_back(RuntimeDefaultArgument{
          index, RuntimeDefaultArgument::Kind::F64, attr});
  }

  RuntimeSymbol symbol{function,
                       contract.str(),
                       role.str(),
                       name.str(),
                       stringAttr(kManifestResultContractAttr),
                       stringAttr(kManifestResultEvidenceAttr),
                       stringAttr(kManifestElementContractAttr),
                       stringAttr(kManifestNextContractAttr),
                       stringAttr(kManifestBuiltinAttr),
                       stringAttr(kManifestBuiltinLoweringAttr),
                       stringAttr(kManifestBuiltinMethodAttr),
                       stringAttr(kManifestBuiltinSinkContractAttr),
                       std::move(classIdArgumentIndices),
                       std::move(defaultArguments),
                       validResultIndex};
  std::string key = runtimeKey(contract, role, name);
  llvm::SmallVector<RuntimeSymbol, 2> &candidates = symbolSets[key];
  auto existing = symbols.find(key);
  if (existing != symbols.end()) {
    if (role != "method") {
      duplicateSymbols.push_back(
          RuntimeSymbolDuplicate{existing->second.function, function,
                                 contract.str(), role.str(), name.str()});
      candidates.push_back(std::move(symbol));
      return;
    }
    candidates.push_back(std::move(symbol));
    return;
  }
  symbols[key] = std::move(symbol);
  RuntimeSymbol &stored = symbols.find(key)->second;
  candidates.push_back(stored);
  if (!stored.builtinName.empty())
    recordBuiltin(stored);
}

void RuntimeManifestIndex::recordBuiltin(const RuntimeSymbol &symbol) {
  auto existing = builtinCallables.find(symbol.builtinName);
  if (existing != builtinCallables.end()) {
    duplicateBuiltins.push_back(RuntimeBuiltinDuplicate{
        existing->second.function, symbol.function, symbol.builtinName});
    return;
  }
  builtinCallables[symbol.builtinName] = symbol;
}

void RuntimeManifestIndex::build(mlir::ModuleOp module) {
  recordDeclaredContracts(module);
  module.walk([&](mlir::func::FuncOp function) {
    auto contract =
        function->getAttrOfType<mlir::StringAttr>(kManifestContractAttr);
    if (!contract)
      return;
    if (auto method =
            function->getAttrOfType<mlir::StringAttr>(kManifestMethodAttr))
      record(function, contract.getValue(), "method", method.getValue());
    if (auto initializer =
            function->getAttrOfType<mlir::StringAttr>(kManifestInitializerAttr))
      record(function, contract.getValue(), "initializer",
             initializer.getValue());
    if (auto primitive =
            function->getAttrOfType<mlir::StringAttr>(kManifestPrimitiveAttr))
      record(function, contract.getValue(), "primitive", primitive.getValue());
    if (function->hasAttr(kManifestShapeAttr))
      recordResultShape(function, contract.getValue());
    if (function->hasAttr(kManifestDeallocatorAttr))
      recordDeallocatorShape(function, contract.getValue());
    recordClassId(function, contract.getValue());
  });
}

const RuntimeValueShape *
RuntimeManifestIndex::requireShape(mlir::func::FuncOp function,
                                   llvm::StringRef contract,
                                   llvm::StringRef purpose) {
  const RuntimeValueShape *shape = valueShape(contract);
  if (shape)
    return shape;
  function.emitError() << "runtime manifest needs ABI shape for " << contract
                       << " to verify " << purpose;
  return nullptr;
}

mlir::LogicalResult RuntimeManifestIndex::verifyTypeSequence(
    mlir::func::FuncOp function, llvm::StringRef label,
    llvm::StringRef contract, llvm::ArrayRef<mlir::Type> actual,
    const RuntimeValueShape &expected) {
  if (sameTypeSequence(actual, expected.valueTypes))
    return mlir::success();
  return function.emitError()
         << "runtime manifest " << label << " for " << contract << " is "
         << describeTypeSequence(actual) << ", but ABI shape from "
         << expected.source << " is "
         << describeTypeSequence(expected.valueTypes);
}

mlir::LogicalResult
RuntimeManifestIndex::verifyReceiverShape(RuntimeSymbol &symbol) {
  const RuntimeValueShape *shape =
      requireShape(symbol.function, symbol.contract, "method receiver");
  if (!shape)
    return mlir::failure();
  mlir::FunctionType functionType = symbol.function.getFunctionType();
  if (functionType.getNumInputs() < shape->valueTypes.size())
    return symbol.function.emitError()
           << "runtime manifest method receiver for " << symbol.contract << "."
           << symbol.name << " has only " << functionType.getNumInputs()
           << " inputs, but ABI shape from " << shape->source << " requires "
           << shape->valueTypes.size();
  llvm::SmallVector<mlir::Type, 4> receiverTypes =
      takePrefix(functionType.getInputs(), shape->valueTypes.size());
  return verifyTypeSequence(symbol.function, "method receiver", symbol.contract,
                            receiverTypes, *shape);
}

mlir::LogicalResult
RuntimeManifestIndex::verifyResultShape(RuntimeSymbol &symbol,
                                        llvm::StringRef resultContract,
                                        llvm::StringRef label) {
  const RuntimeValueShape *shape =
      requireShape(symbol.function, resultContract, label);
  if (!shape)
    return mlir::failure();
  mlir::FunctionType functionType = symbol.function.getFunctionType();
  return verifyTypeSequence(symbol.function, label, resultContract,
                            functionType.getResults(), *shape);
}

mlir::LogicalResult
RuntimeManifestIndex::verifyNextResultPartition(RuntimeSymbol &symbol) {
  mlir::FunctionType functionType = symbol.function.getFunctionType();
  unsigned validIndex = *symbol.validResultIndex;
  if (validIndex >= functionType.getNumResults())
    return symbol.function.emitError()
           << "runtime manifest valid_result_index for " << symbol.contract
           << "." << symbol.name << " is outside the result list";
  if (!functionType.getResult(validIndex).isInteger(1))
    return symbol.function.emitError()
           << "runtime manifest valid_result_index for " << symbol.contract
           << "." << symbol.name << " must point at an i1 result";

  VerificationResult verified;
  if (!symbol.elementContract.empty()) {
    const RuntimeValueShape *elementShape = requireShape(
        symbol.function, symbol.elementContract, "next element result");
    if (!elementShape) {
      verified.fail();
    } else {
      llvm::SmallVector<mlir::Type, 4> elementTypes =
          takeSlice(functionType.getResults(), 0, validIndex);
      verified.check(verifyTypeSequence(symbol.function, "next element result",
                                        symbol.elementContract, elementTypes,
                                        *elementShape));
    }
  }

  if (!symbol.nextContract.empty()) {
    const RuntimeValueShape *nextShape =
        requireShape(symbol.function, symbol.nextContract, "next state result");
    if (!nextShape) {
      verified.fail();
    } else {
      llvm::SmallVector<mlir::Type, 4> nextTypes =
          takeSlice(functionType.getResults(), validIndex + 1,
                    functionType.getNumResults());
      verified.check(verifyTypeSequence(symbol.function, "next state result",
                                        symbol.nextContract, nextTypes,
                                        *nextShape));
    }
  }
  return verified.get();
}

mlir::LogicalResult
RuntimeManifestIndex::verifyClassIdArguments(RuntimeSymbol &symbol) {
  if (symbol.classIdArgumentIndices.empty())
    return mlir::success();
  if (symbol.role != "initializer")
    return symbol.function.emitError()
           << "runtime class id arguments are only supported on initializers";

  VerificationResult verified;
  mlir::FunctionType functionType = symbol.function.getFunctionType();
  for (unsigned inputIndex : symbol.classIdArgumentIndices) {
    if (inputIndex >= functionType.getNumInputs()) {
      symbol.function.emitError() << "runtime class id argument index "
                                  << inputIndex << " is outside the input list";
      verified.fail();
      continue;
    }
    if (!functionType.getInput(inputIndex).isInteger(64)) {
      symbol.function.emitError()
          << "runtime class id argument " << inputIndex << " for "
          << symbol.contract << "." << symbol.name
          << " must be an i64 input, got " << functionType.getInput(inputIndex);
      verified.fail();
    }
  }
  if (!classId(symbol.contract)) {
    symbol.function.emitError()
        << "runtime class id argument for " << symbol.contract
        << " requires a ly.runtime.class_id declaration";
    verified.fail();
  }
  return verified.get();
}

mlir::LogicalResult
RuntimeManifestIndex::verifyDefaultArguments(RuntimeSymbol &symbol) {
  VerificationResult verified;
  mlir::FunctionType functionType = symbol.function.getFunctionType();
  for (unsigned inputIndex = 0; inputIndex < functionType.getNumInputs();
       ++inputIndex) {
    mlir::Type inputType = functionType.getInput(inputIndex);
    mlir::Attribute defaultI64 =
        symbol.function.getArgAttr(inputIndex, kManifestDefaultI64Attr);
    mlir::Attribute defaultF64 =
        symbol.function.getArgAttr(inputIndex, kManifestDefaultF64Attr);
    if (defaultI64 && defaultF64) {
      symbol.function.emitError()
          << "runtime input " << inputIndex
          << " cannot declare both i64 and f64 defaults";
      verified.fail();
    }
    if (defaultI64) {
      if (!inputType.isInteger(64)) {
        symbol.function.emitError()
            << "runtime default_i64 input " << inputIndex
            << " must be an i64 input, got " << inputType;
        verified.fail();
      }
      if (!mlir::isa<mlir::IntegerAttr>(defaultI64)) {
        symbol.function.emitError() << "runtime default_i64 input "
                                    << inputIndex << " must be an IntegerAttr";
        verified.fail();
      }
    }
    if (defaultF64) {
      if (!inputType.isF64()) {
        symbol.function.emitError()
            << "runtime default_f64 input " << inputIndex
            << " must be an f64 input, got " << inputType;
        verified.fail();
      }
      if (!mlir::isa<mlir::FloatAttr>(defaultF64)) {
        symbol.function.emitError() << "runtime default_f64 input "
                                    << inputIndex << " must be a FloatAttr";
        verified.fail();
      }
    }
  }
  return verified.get();
}

mlir::LogicalResult
RuntimeManifestIndex::verifyBuiltinCallable(RuntimeSymbol &symbol) {
  if (symbol.builtinName.empty())
    return mlir::success();
  if (symbol.builtinLowering.empty())
    return symbol.function.emitError()
           << "runtime builtin binding " << symbol.builtinName
           << " must declare ly.runtime.builtin_lowering";

  if (symbol.builtinLowering == "method") {
    if (symbol.builtinMethod.empty())
      return symbol.function.emitError()
             << "runtime builtin binding " << symbol.builtinName
             << " with method lowering must declare ly.runtime.builtin_method";
    if (symbol.resultContract.empty())
      return symbol.function.emitError()
             << "runtime builtin binding " << symbol.builtinName
             << " with method lowering must declare ly.runtime.result_contract";
    return mlir::success();
  }

  if (symbol.builtinLowering == "method_sink") {
    if (symbol.builtinMethod.empty())
      return symbol.function.emitError()
             << "runtime builtin binding " << symbol.builtinName
             << " with method_sink lowering must declare "
                "ly.runtime.builtin_method";
    if (symbol.builtinSinkContract.empty())
      return symbol.function.emitError()
             << "runtime builtin binding " << symbol.builtinName
             << " with method_sink lowering must declare "
                "ly.runtime.builtin_sink_contract";
    if (symbol.resultContract.empty())
      return symbol.function.emitError()
             << "runtime builtin binding " << symbol.builtinName
             << " with method_sink lowering must declare "
                "ly.runtime.result_contract";
    const RuntimeValueShape *shape = requireShape(
        symbol.function, symbol.builtinSinkContract, "builtin sink input");
    if (!shape)
      return mlir::failure();
    mlir::FunctionType functionType = symbol.function.getFunctionType();
    return verifyTypeSequence(symbol.function, "builtin sink input",
                              symbol.builtinSinkContract,
                              functionType.getInputs(), *shape);
  }

  if (symbol.builtinLowering == "direct") {
    if (symbol.resultContract.empty())
      return symbol.function.emitError()
             << "runtime builtin binding " << symbol.builtinName
             << " with direct lowering must declare "
                "ly.runtime.result_contract";
    return mlir::success();
  }

  if (symbol.builtinLowering == "asyncio_sleep") {
    if (symbol.resultContract != "types.CoroutineType")
      return symbol.function.emitError()
             << "runtime builtin binding " << symbol.builtinName
             << " with asyncio_sleep lowering must declare "
                "ly.runtime.result_contract = \"types.CoroutineType\"";
    return mlir::success();
  }

  return symbol.function.emitError()
         << "runtime builtin binding " << symbol.builtinName
         << " has unsupported lowering strategy " << symbol.builtinLowering;
}

mlir::LogicalResult RuntimeManifestIndex::verifySymbol(RuntimeSymbol &symbol) {
  VerificationResult verified;

  verified.check(verifyBuiltinCallable(symbol));
  verified.check(verifyDefaultArguments(symbol));
  verified.check(verifyClassIdArguments(symbol));
  if (symbol.role == "initializer")
    verified.check(
        verifyResultShape(symbol, symbol.contract, "initializer result"));
  if (symbol.role == "method")
    verified.check(verifyReceiverShape(symbol));
  if (!symbol.resultContract.empty())
    verified.check(verifyResultShape(symbol, symbol.resultContract,
                                     "declared result_contract"));
  if (!symbol.resultEvidence.empty()) {
    if (symbol.resultEvidence != "receiver") {
      symbol.function.emitError() << kManifestResultEvidenceAttr
                                  << " must currently be empty or \"receiver\"";
      verified.fail();
    } else if (symbol.role != "method") {
      symbol.function.emitError()
          << kManifestResultEvidenceAttr
          << " = \"receiver\" is only valid for runtime methods";
      verified.fail();
    }
  }
  if (symbol.validResultIndex)
    verified.check(verifyNextResultPartition(symbol));

  return verified.get();
}

} // namespace py::runtime_lowering
