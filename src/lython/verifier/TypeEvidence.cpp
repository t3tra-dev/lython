#include "runtime/Verification.h"

#include "Contracts.h"

#include "Contracts.h"
#include "PyDialectTypes.h"
#include "PyTypeObject.h"
#include "PyProtocols.h"
#include "Support.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <memory>
#include <optional>

namespace py {
namespace {

using py::contracts::manifestClassNameForContract;

namespace contracts = py::contracts;
using verifier::VerificationResult;
using verifier::walkVerifyOperations;

bool isTypingAny(mlir::Type type) {
  auto contract = mlir::dyn_cast_if_present<ContractType>(type);
  return contract && contract.getContractName() == "typing.Any";
}

bool containsDisallowedEvidenceType(mlir::Type type) {
  if (!type)
    return false;

  // Unification variables are inference-engine internal; one reaching
  // evidence means the TypeSystem facade's zonk boundary was bypassed.
  if (mlir::isa<InferVarType>(type))
    return true;

  if (auto callable = mlir::dyn_cast<CallableType>(type)) {
    for (mlir::Type nested : callable.getPositionalTypes())
      if (containsDisallowedEvidenceType(nested))
        return true;
    for (mlir::Type nested : callable.getKwOnlyTypes())
      if (containsDisallowedEvidenceType(nested))
        return true;
    for (mlir::Type nested : callable.getResultTypes())
      if (containsDisallowedEvidenceType(nested))
        return true;
    if (callable.hasVararg() &&
        containsDisallowedEvidenceType(callable.getVarargType()))
      return true;
    if (callable.hasKwarg() &&
        containsDisallowedEvidenceType(callable.getKwargType()))
      return true;
    return false;
  }

  if (auto contract = mlir::dyn_cast<ContractType>(type))
    return llvm::any_of(contract.getArguments(),
                        containsDisallowedEvidenceType);
  if (auto protocol = mlir::dyn_cast<ProtocolType>(type))
    return llvm::any_of(protocol.getArguments(),
                        containsDisallowedEvidenceType);
  if (auto unionType = mlir::dyn_cast<UnionType>(type))
    return llvm::any_of(unionType.getMemberTypes(),
                        containsDisallowedEvidenceType);
  if (auto typeType = mlir::dyn_cast<TypeType>(type))
    return containsDisallowedEvidenceType(typeType.getInstanceType());
  if (auto overload = mlir::dyn_cast<OverloadType>(type))
    return llvm::any_of(overload.getCandidateTypes(),
                        containsDisallowedEvidenceType);
  if (auto unpack = mlir::dyn_cast<UnpackType>(type))
    return containsDisallowedEvidenceType(unpack.getPackedType());
  return false;
}

mlir::LogicalResult verifyStableEvidenceType(mlir::Operation *op,
                                             mlir::Type type,
                                             llvm::StringRef what) {
  if (isTypingAny(type))
    return op->emitError()
           << what
           << " is erased typing.Any evidence; TypeSystem must resolve stable "
              "lowering boundaries to a concrete manifest contract";
  if (containsPyInferVar(type))
    return op->emitError()
           << what
           << " contains an unresolved inference variable that escaped "
              "TypeSystem; inference must zonk evidence before emit";
  if (!containsDisallowedEvidenceType(type))
    return mlir::success();
  return op->emitError()
         << what
         << " contains erased or legacy dialect evidence; TypeSystem must "
            "resolve it to manifest contracts before lowering";
}

bool isNoneLike(mlir::Type type) {
  if (auto literal = mlir::dyn_cast_if_present<LiteralType>(type))
    if (literal.getSpelling() == "None")
      return true;
  return contracts::runtimeContractName(type) == "types.NoneType";
}

bool isBoolLike(mlir::Type type) {
  if (auto literal = mlir::dyn_cast_if_present<LiteralType>(type)) {
    llvm::StringRef spelling = literal.getSpelling();
    if (spelling == "True" || spelling == "False")
      return true;
  }
  return contracts::runtimeContractName(type) == "builtins.bool";
}

bool isPrimitiveBool(mlir::Type type) {
  auto integer = mlir::dyn_cast_if_present<mlir::IntegerType>(type);
  return integer && integer.getWidth() == 1;
}

bool evidenceTypeSame(mlir::Type lhs, mlir::Type rhs);

bool callableEvidenceSame(CallableType lhs, CallableType rhs) {
  if (!lhs || !rhs)
    return lhs == rhs;
  if (lhs.getPositionalTypes().size() != rhs.getPositionalTypes().size() ||
      lhs.getKwOnlyTypes().size() != rhs.getKwOnlyTypes().size() ||
      lhs.getResultTypes().size() != rhs.getResultTypes().size() ||
      lhs.hasVararg() != rhs.hasVararg() || lhs.hasKwarg() != rhs.hasKwarg())
    return false;

  auto sameRange = [](llvm::ArrayRef<mlir::Type> lhs,
                      llvm::ArrayRef<mlir::Type> rhs) {
    for (auto [left, right] : llvm::zip(lhs, rhs))
      if (!evidenceTypeSame(left, right))
        return false;
    return true;
  };
  if (!sameRange(lhs.getPositionalTypes(), rhs.getPositionalTypes()) ||
      !sameRange(lhs.getKwOnlyTypes(), rhs.getKwOnlyTypes()) ||
      !sameRange(lhs.getResultTypes(), rhs.getResultTypes()))
    return false;
  if (lhs.hasVararg() &&
      !evidenceTypeSame(lhs.getVarargType(), rhs.getVarargType()))
    return false;
  if (lhs.hasKwarg() &&
      !evidenceTypeSame(lhs.getKwargType(), rhs.getKwargType()))
    return false;
  return true;
}

bool evidenceTypeSame(mlir::Type lhs, mlir::Type rhs) {
  if (lhs == rhs)
    return true;

  CallableType lhsCallable = getCallableContract(lhs);
  CallableType rhsCallable = getCallableContract(rhs);
  if (lhsCallable || rhsCallable)
    return callableEvidenceSame(lhsCallable, rhsCallable);

  std::string lhsContract = contracts::runtimeContractName(lhs);
  std::string rhsContract = contracts::runtimeContractName(rhs);
  if (!lhsContract.empty() && lhsContract == rhsContract)
    return true;

  if (auto lhsType = mlir::dyn_cast_if_present<TypeType>(lhs))
    if (auto rhsType = mlir::dyn_cast_if_present<TypeType>(rhs))
      return evidenceTypeSame(lhsType.getInstanceType(),
                              rhsType.getInstanceType());

  if (auto lhsProtocol = mlir::dyn_cast_if_present<ProtocolType>(lhs)) {
    auto rhsProtocol = mlir::dyn_cast_if_present<ProtocolType>(rhs);
    if (!rhsProtocol ||
        lhsProtocol.getProtocolName() != rhsProtocol.getProtocolName() ||
        lhsProtocol.getArguments().size() != rhsProtocol.getArguments().size())
      return false;
    for (auto [left, right] :
         llvm::zip(lhsProtocol.getArguments(), rhsProtocol.getArguments()))
      if (!evidenceTypeSame(left, right))
        return false;
    return true;
  }

  return false;
}

bool evidenceAssignable(mlir::Type actual, mlir::Type expected,
                        mlir::Operation *from) {
  if (evidenceTypeSame(actual, expected))
    return true;
  if (isPrimitiveBool(actual) && isBoolLike(expected))
    return true;
  if (isNoneLike(actual) && isNoneLike(expected))
    return true;
  if (auto expectedContract =
          mlir::dyn_cast_if_present<ContractType>(expected)) {
    const protocols::Table &table = protocols::Table::get(*from->getContext());
    if (table.isManifestSubclassOf(actual, expectedContract.getContractName()))
      return true;
    std::string manifestName =
        manifestClassNameForContract(expectedContract.getContractName());
    const protocols::ProtocolInfo *info = table.lookup(manifestName);
    if (info && info->isProtocol) {
      bool satisfies = true;
      for (const auto &entry : info->methods) {
        bool methodMatches = false;
        for (const protocols::ProtocolMethod &required : entry.second) {
          CallableType requiredBound =
              protocols::bindReceiverCallable(required.signature);
          for (const protocols::ContractResolution &candidate :
               table.methodContractCandidatesWithEvidence(actual,
                                                          entry.first)) {
            CallableType candidateBound =
                protocols::bindReceiverCallable(candidate.method.signature);
            if (callableEvidenceSame(candidateBound, requiredBound)) {
              methodMatches = true;
              break;
            }
          }
          if (methodMatches)
            break;
        }
        if (!methodMatches) {
          satisfies = false;
          break;
        }
      }
      if (satisfies)
        return true;
    }
  }
  return isAssignableTo(actual, expected, from);
}

bool evidenceMatchesCandidate(mlir::Type selected, mlir::Type candidate,
                              mlir::Operation *from);

bool callableMatchesCandidate(CallableType selected, CallableType candidate,
                              mlir::Operation *from) {
  if (!selected || !candidate)
    return selected == candidate;
  if (selected.getKwOnlyTypes().size() != candidate.getKwOnlyTypes().size() ||
      selected.getResultTypes().size() != candidate.getResultTypes().size() ||
      selected.hasVararg() != candidate.hasVararg() ||
      selected.hasKwarg() != candidate.hasKwarg())
    return false;

  auto matchesRange = [&](llvm::ArrayRef<mlir::Type> selectedTypes,
                          llvm::ArrayRef<mlir::Type> candidateTypes) {
    if (selectedTypes.size() != candidateTypes.size())
      return false;
    for (auto [selectedType, candidateType] :
         llvm::zip(selectedTypes, candidateTypes))
      if (!evidenceMatchesCandidate(selectedType, candidateType, from))
        return false;
    return true;
  };

  llvm::ArrayRef<mlir::Type> selectedPositional = selected.getPositionalTypes();
  llvm::ArrayRef<mlir::Type> candidatePositional =
      candidate.getPositionalTypes();
  unsigned selectedIndex = 0;
  for (unsigned candidateIndex = 0; candidateIndex < candidatePositional.size();
       ++candidateIndex) {
    mlir::Type candidateType = candidatePositional[candidateIndex];
    bool isPack = mlir::isa<ParamSpecType, TypeVarTupleType>(candidateType);
    if (auto unpack = mlir::dyn_cast<UnpackType>(candidateType))
      isPack = mlir::isa<TypeVarTupleType>(unpack.getPackedType());
    if (isPack) {
      unsigned remainingCandidate =
          candidatePositional.size() - candidateIndex - 1;
      if (selectedPositional.size() < selectedIndex + remainingCandidate)
        return false;
      selectedIndex = selectedPositional.size() - remainingCandidate;
      continue;
    }
    if (selectedIndex >= selectedPositional.size())
      return false;
    if (!evidenceMatchesCandidate(selectedPositional[selectedIndex],
                                  candidateType, from))
      return false;
    ++selectedIndex;
  }
  if (selectedIndex != selectedPositional.size())
    return false;

  if (!matchesRange(selected.getKwOnlyTypes(), candidate.getKwOnlyTypes()) ||
      !matchesRange(selected.getResultTypes(), candidate.getResultTypes()))
    return false;
  if (selected.hasVararg() &&
      !evidenceMatchesCandidate(selected.getVarargType(),
                                candidate.getVarargType(), from))
    return false;
  if (selected.hasKwarg() &&
      !evidenceMatchesCandidate(selected.getKwargType(),
                                candidate.getKwargType(), from))
    return false;
  return true;
}

bool evidenceMatchesCandidate(mlir::Type selected, mlir::Type candidate,
                              mlir::Operation *from) {
  if (!candidate)
    return !selected;
  if (mlir::isa<SelfType, TypeVarType, ParamSpecType, TypeVarTupleType>(
          candidate))
    return true;
  if (auto unpack = mlir::dyn_cast<UnpackType>(candidate))
    if (mlir::isa<TypeVarTupleType>(unpack.getPackedType()))
      return true;

  CallableType selectedCallable = getCallableContract(selected);
  CallableType candidateCallable = getCallableContract(candidate);
  if (selectedCallable || candidateCallable)
    return callableMatchesCandidate(selectedCallable, candidateCallable, from);

  if (evidenceAssignable(selected, candidate, from))
    return true;

  if (auto selectedType = mlir::dyn_cast_if_present<TypeType>(selected))
    if (auto candidateType = mlir::dyn_cast_if_present<TypeType>(candidate))
      return evidenceMatchesCandidate(selectedType.getInstanceType(),
                                      candidateType.getInstanceType(), from);

  if (auto selectedContract =
          mlir::dyn_cast_if_present<ContractType>(selected)) {
    auto candidateContract = mlir::dyn_cast_if_present<ContractType>(candidate);
    if (!candidateContract ||
        selectedContract.getContractName() !=
            candidateContract.getContractName() ||
        selectedContract.getArguments().size() !=
            candidateContract.getArguments().size())
      return false;
    for (auto [selectedArg, candidateArg] : llvm::zip(
             selectedContract.getArguments(), candidateContract.getArguments()))
      if (!evidenceMatchesCandidate(selectedArg, candidateArg, from))
        return false;
    return true;
  }

  if (auto selectedProtocol =
          mlir::dyn_cast_if_present<ProtocolType>(selected)) {
    auto candidateProtocol = mlir::dyn_cast_if_present<ProtocolType>(candidate);
    if (!candidateProtocol ||
        selectedProtocol.getProtocolName() !=
            candidateProtocol.getProtocolName() ||
        selectedProtocol.getArguments().size() !=
            candidateProtocol.getArguments().size())
      return false;
    for (auto [selectedArg, candidateArg] : llvm::zip(
             selectedProtocol.getArguments(), candidateProtocol.getArguments()))
      if (!evidenceMatchesCandidate(selectedArg, candidateArg, from))
        return false;
    return true;
  }

  if (auto candidateUnion = mlir::dyn_cast_if_present<UnionType>(candidate))
    return llvm::any_of(
        candidateUnion.getMemberTypes(), [&](mlir::Type member) {
          return evidenceMatchesCandidate(selected, member, from);
        });

  return false;
}

std::optional<llvm::StringRef> contractAttrName(mlir::Operation *op) {
  if (op->hasAttr("callee_contract"))
    return llvm::StringRef("callee_contract");
  if (op->hasAttr("call_contract"))
    return llvm::StringRef("call_contract");
  if (op->hasAttr("new_contract"))
    return llvm::StringRef("new_contract");
  if (op->hasAttr("init_contract"))
    return llvm::StringRef("init_contract");
  if (op->hasAttr("await_contract"))
    return llvm::StringRef("await_contract");
  return std::nullopt;
}

mlir::FailureOr<CallableType> readCallableContract(mlir::Operation *op,
                                                   llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::TypeAttr>(attrName);
  if (!attr)
    return op->emitError() << "missing " << attrName << " evidence";
  if (mlir::failed(verifyStableEvidenceType(op, attr.getValue(), attrName)))
    return mlir::failure();
  CallableType callable = getCallableContract(attr.getValue());
  if (!callable)
    return op->emitError() << attrName
                           << " must resolve to a Callable protocol contract";
  return callable;
}

llvm::StringRef opName(mlir::Operation *op) {
  return op->getName().getStringRef();
}

bool isBinaryMethodOp(llvm::StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Cases({"py.add", "py.sub", "py.mul", "py.div", "py.floordiv"}, true)
      .Cases({"py.mod", "py.lshift", "py.rshift", "py.bitand"}, true)
      .Cases({"py.bitor", "py.bitxor", "py.pow", "py.le", "py.lt"}, true)
      .Cases({"py.gt", "py.ge", "py.eq", "py.ne"}, true)
      .Default(false);
}

bool isReflectedBinaryMethod(mlir::Operation *op) {
  if (!isBinaryMethodOp(opName(op)))
    return false;
  auto method = op->getAttrOfType<mlir::StringAttr>("method_name");
  if (!method)
    return false;
  llvm::StringRef name = method.getValue();
  return name.starts_with("__r") && name != "__repr__";
}

std::optional<llvm::StringRef> methodNameFor(mlir::Operation *op) {
  if (auto method = op->getAttrOfType<mlir::StringAttr>("method_name"))
    return method.getValue();
  return llvm::StringSwitch<std::optional<llvm::StringRef>>(opName(op))
      .Case("py.init", llvm::StringRef("__init__"))
      .Case("py.enter", llvm::StringRef("__enter__"))
      .Case("py.exit", llvm::StringRef("__exit__"))
      .Case("py.aenter", llvm::StringRef("__aenter__"))
      .Case("py.aexit", llvm::StringRef("__aexit__"))
      .Case("py.round", llvm::StringRef("__round__"))
      .Case("py.getitem", llvm::StringRef("__getitem__"))
      .Case("py.setitem", llvm::StringRef("__setitem__"))
      .Case("py.delitem", llvm::StringRef("__delitem__"))
      .Case("py.contains", llvm::StringRef("__contains__"))
      .Case("py.bool", llvm::StringRef("__bool__"))
      .Case("py.repr", llvm::StringRef("__repr__"))
      .Case("py.str", llvm::StringRef("__str__"))
      .Case("py.iter", llvm::StringRef("__iter__"))
      .Case("py.next", llvm::StringRef("__next__"))
      .Case("py.aiter", llvm::StringRef("__aiter__"))
      .Case("py.anext", llvm::StringRef("__anext__"))
      .Case("py.len", llvm::StringRef("__len__"))
      .Default(std::nullopt);
}

bool shouldValidateManifestCandidate(mlir::Operation *op) {
  llvm::StringRef name = opName(op);
  if (name == "py.call" || name == "py.invoke" || name == "py.new" ||
      name == "py.await")
    return false;
  return methodNameFor(op).has_value();
}

std::optional<mlir::Type> receiverTypeFor(mlir::Operation *op) {
  if (op->getNumOperands() == 0)
    return std::nullopt;
  if (isReflectedBinaryMethod(op) && op->getNumOperands() >= 2)
    return op->getOperand(1).getType();
  return op->getOperand(0).getType();
}

llvm::SmallVector<mlir::Value, 4>
explicitCallableOperands(mlir::Operation *op) {
  llvm::SmallVector<mlir::Value, 4> operands;
  llvm::StringRef name = opName(op);

  if (name == "py.call")
    return operands;
  if (name == "py.new" || name == "py.invoke" || name == "py.await")
    return operands;
  if (name == "py.init") {
    if (op->getNumOperands() > 0)
      operands.push_back(op->getOperand(0));
    return operands;
  }

  if (isReflectedBinaryMethod(op) && op->getNumOperands() >= 2) {
    operands.push_back(op->getOperand(1));
    operands.push_back(op->getOperand(0));
    return operands;
  }

  operands.append(op->operand_begin(), op->operand_end());
  return operands;
}

llvm::SmallVector<mlir::Type, 2> callableResultSurface(mlir::Operation *op) {
  llvm::SmallVector<mlir::Type, 2> results;
  llvm::StringRef name = opName(op);
  if (name == "py.invoke" || name == "py.await")
    return results;
  if (name == "py.next") {
    if (op->getNumResults() > 0)
      results.push_back(op->getResult(0).getType());
    return results;
  }
  for (mlir::Type type : op->getResultTypes())
    results.push_back(type);
  // A structural-mutation call (`ly.structural_mutation`) carries one extra
  // receiver-typed result that rebinds the receiver local; it is not part of
  // the Callable contract's result surface and is checked separately.
  if (op->hasAttr("ly.structural_mutation") && !results.empty())
    results.pop_back();
  return results;
}

mlir::LogicalResult verifyCallableOperands(mlir::Operation *op,
                                           CallableType callable) {
  llvm::SmallVector<mlir::Value, 4> actuals = explicitCallableOperands(op);
  if (actuals.empty())
    return mlir::success();

  llvm::ArrayRef<mlir::Type> expected = callable.getPositionalTypes();
  if (expected.size() < actuals.size())
    return op->emitError() << "selected Callable evidence has "
                           << expected.size()
                           << " positional operands, but op exposes "
                           << actuals.size();

  for (auto [index, pair] : llvm::enumerate(llvm::zip(actuals, expected))) {
    mlir::Value actual = std::get<0>(pair);
    mlir::Type expectedType = std::get<1>(pair);
    if (evidenceAssignable(actual.getType(), expectedType, op))
      continue;
    return op->emitError() << "operand " << index << " type "
                           << actual.getType()
                           << " does not match selected Callable evidence "
                           << expectedType;
  }
  return mlir::success();
}

mlir::LogicalResult verifyCallableResults(mlir::Operation *op,
                                          CallableType callable) {
  llvm::StringRef name = opName(op);
  if (name == "py.invoke" || name == "py.await")
    return mlir::success();

  if (op->hasAttr("ly.structural_mutation")) {
    // The rebind result is always last; ops whose contract yields no value
    // surface (e.g. py.setitem) carry exactly the rebind result.
    if (op->getNumResults() < 1 || op->getNumOperands() < 1 ||
        op->getResult(op->getNumResults() - 1).getType() !=
            op->getOperand(0).getType())
      return op->emitError()
             << "structural-mutation op must expose one extra result whose "
                "type matches the receiver operand";
  }

  llvm::SmallVector<mlir::Type, 2> actuals = callableResultSurface(op);
  llvm::ArrayRef<mlir::Type> expected = callable.getResultTypes();

  if (actuals.empty() && expected.empty())
    return mlir::success();
  if (actuals.empty() && expected.size() == 1 && isNoneLike(expected.front()))
    return mlir::success();
  if (actuals.size() != expected.size())
    return op->emitError() << "op result surface has " << actuals.size()
                           << " values, but selected Callable evidence returns "
                           << expected.size();

  for (auto [index, pair] : llvm::enumerate(llvm::zip(actuals, expected))) {
    mlir::Type actual = std::get<0>(pair);
    mlir::Type expectedType = std::get<1>(pair);
    if (evidenceAssignable(actual, expectedType, op))
      continue;
    return op->emitError() << "result " << index << " type " << actual
                           << " does not match selected Callable evidence "
                           << expectedType;
  }
  return mlir::success();
}

mlir::LogicalResult verifyManifestCandidate(mlir::Operation *op,
                                            CallableType selected) {
  if (!shouldValidateManifestCandidate(op))
    return mlir::success();
  std::optional<llvm::StringRef> methodName = methodNameFor(op);
  std::optional<mlir::Type> receiver = receiverTypeFor(op);
  if (!methodName || !receiver)
    return mlir::success();

  const protocols::Table &table = protocols::Table::get(*op->getContext());
  std::vector<protocols::ContractResolution> candidates =
      table.methodContractCandidatesWithEvidence(*receiver, *methodName);
  for (const protocols::ContractResolution &candidate : candidates)
    if (callableEvidenceSame(selected, candidate.method.signature) ||
        callableMatchesCandidate(selected, candidate.method.signature, op))
      return mlir::success();

  // Field-record constructor: a SOURCE class accepts its declared fields
  // positionally (all optional) when no source __init__ exists. The
  // synthesized contract validates against the py.class field schema instead
  // of a manifest candidate.
  if (*methodName == "__init__") {
    auto contract = mlir::dyn_cast<ContractType>(*receiver);
    py::ClassOp classOp =
        contract ? type_object::lookup(op, contract.getContractName())
                 : py::ClassOp{};
    auto fieldTypes =
        classOp ? classOp->getAttrOfType<mlir::ArrayAttr>("field_contract_types")
                : mlir::ArrayAttr{};
    if (fieldTypes) {
      llvm::ArrayRef<mlir::Type> positional = selected.getPositionalTypes();
      llvm::ArrayRef<mlir::Type> results = selected.getResultTypes();
      bool matches = positional.size() == fieldTypes.size() + 1 &&
                     !positional.empty() && positional.front() == *receiver &&
                     results.size() == 1 && isNoneLike(results.front());
      if (matches)
        for (auto [index, attr] : llvm::enumerate(fieldTypes)) {
          auto fieldType = mlir::dyn_cast<mlir::TypeAttr>(attr);
          if (!fieldType || fieldType.getValue() != positional[index + 1]) {
            matches = false;
            break;
          }
        }
      if (matches)
        return mlir::success();
    }
  }

  return op->emitError()
         << "selected " << *methodName
         << " Callable evidence is not a manifest candidate for receiver type "
         << *receiver;
}

mlir::LogicalResult verifyYieldFromEvidenceOp(mlir::Operation *op) {
  if (opName(op) != "py.yield_from")
    return mlir::success();

  auto attr = op->getAttrOfType<mlir::TypeAttr>("yield_from_contract");
  if (!attr)
    return op->emitError() << "missing yield_from_contract evidence";
  mlir::Type selected = attr.getValue();
  if (mlir::failed(
          verifyStableEvidenceType(op, selected, "yield_from_contract")))
    return mlir::failure();
  auto protocol = mlir::dyn_cast<ProtocolType>(selected);
  if (!protocol)
    return op->emitError()
           << "yield_from_contract must resolve to a protocol contract";
  llvm::StringRef name = protocol.getProtocolName();
  if (name != "Generator" && name != "Iterator" && name != "Iterable")
    return op->emitError()
           << "yield_from_contract must be Generator, Iterator, or Iterable";
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitError()
           << "yield_from evidence verifier expects one source and one result";

  VerificationResult result;
  mlir::Type sourceType = op->getOperand(0).getType();
  result.check(verifyStableEvidenceType(op, sourceType, "source type"));
  result.check(
      verifyStableEvidenceType(op, op->getResult(0).getType(), "result type"));
  if (!evidenceAssignable(sourceType, selected, op))
    result.check(op->emitError()
                 << "yield_from_contract " << selected
                 << " is not satisfied by source type " << sourceType);
  return result.get();
}

// Escape check for py.infervar across the whole op surface, including ops
// that carry no contract attribute: contract-typed operands/results are
// already rejected by op verifiers (InferVarType is not a Py_ContractType),
// but a variable nested inside a container argument or inside a type-bearing
// attribute (callable_type, ly.generator.*, closure_types) passes those
// predicates and must be caught here.
mlir::LogicalResult verifyNoEscapedInferenceVariable(mlir::Operation *op) {
  VerificationResult result;
  auto check = [&](mlir::Type type, llvm::StringRef what) {
    if (auto function = mlir::dyn_cast_if_present<mlir::FunctionType>(type)) {
      bool contaminated =
          llvm::any_of(function.getInputs(), containsPyInferVar) ||
          llvm::any_of(function.getResults(), containsPyInferVar);
      if (!contaminated)
        return;
    } else if (!containsPyInferVar(type)) {
      return;
    }
    result.check(op->emitError()
                 << what << " contains an unresolved inference variable that "
                 << "escaped TypeSystem; inference must zonk types before "
                 << "emit");
  };
  for (mlir::Type type : op->getResultTypes())
    check(type, "result type");
  for (mlir::Type type : op->getOperandTypes())
    check(type, "operand type");
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr.getValue())) {
      check(typeAttr.getValue(), attr.getName().strref());
      continue;
    }
    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue()))
      for (mlir::Attribute element : arrayAttr)
        if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(element))
          check(typeAttr.getValue(), attr.getName().strref());
  }
  return result.get();
}

mlir::LogicalResult verifyEvidenceOp(mlir::Operation *op) {
  if (mlir::failed(verifyNoEscapedInferenceVariable(op)))
    return mlir::failure();
  if (opName(op) == "py.yield_from")
    return verifyYieldFromEvidenceOp(op);

  std::optional<llvm::StringRef> attrName = contractAttrName(op);
  if (!attrName)
    // P0 verifier gate: every Python operation op that invokes user-visible
    // behavior already carries its selected callable/protocol contract as a
    // *mandatory* dialect attribute (call/invoke/new/init/await and every
    // special-method op declare `*_contract` as a required `TypeAttrOf`), so
    // the operation verifier rejects a dropped-evidence op before this pass
    // runs. Static attribute access is gated separately by `py.attr.get` /
    // `py.attr.set`'s own field-declaration verifier. Reaching this branch
    // therefore means the op is not a behavior-invoking op (constants, packs,
    // binding refs, ref-count and control-flow ops), which needs no contract.
    return mlir::success();

  if (opName(op) == "py.new") {
    VerificationResult result;
    if (auto attr = op->getAttrOfType<mlir::TypeAttr>(*attrName))
      result.check(verifyStableEvidenceType(op, attr.getValue(), *attrName));
    for (mlir::Type resultType : op->getResultTypes())
      result.check(verifyStableEvidenceType(op, resultType, "result type"));
    return result.get();
  }

  mlir::FailureOr<CallableType> callable = readCallableContract(op, *attrName);
  if (mlir::failed(callable))
    return mlir::failure();

  VerificationResult result;
  result.check(verifyCallableOperands(op, *callable));
  result.check(verifyCallableResults(op, *callable));
  result.check(verifyManifestCandidate(op, *callable));
  for (mlir::Type resultType : op->getResultTypes())
    result.check(verifyStableEvidenceType(op, resultType, "result type"));
  return result.get();
}

mlir::LogicalResult verifyTypeEvidenceImpl(mlir::ModuleOp module) {
  return walkVerifyOperations(module, verifyEvidenceOp);
}

class TypeEvidenceVerifierPass
    : public mlir::PassWrapper<TypeEvidenceVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeEvidenceVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-type-evidence-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify TypeSystem-selected Callable evidence before Py lowering";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyTypeEvidenceImpl(getOperation())))
      signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult verifyTypeEvidence(mlir::ModuleOp module) {
  return verifyTypeEvidenceImpl(module);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTypeEvidenceVerifierPass() {
  return std::make_unique<TypeEvidenceVerifierPass>();
}

} // namespace py
