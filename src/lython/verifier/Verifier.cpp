#include "PyDialectTypes.h"
#include "PyTypeObject.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

mlir::LogicalResult requireTypeAttr(mlir::Operation *op, llvm::StringRef name) {
  if (!op->getAttrOfType<mlir::TypeAttr>(name))
    return op->emitOpError("requires TypeAttr '") << name << "'";
  return mlir::success();
}

mlir::LogicalResult requireProtocolAttr(mlir::Operation *op,
                                        llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::TypeAttr>(name);
  if (!attr)
    return op->emitOpError("requires TypeAttr '") << name << "'";
  if (!isPyProtocolType(attr.getValue()))
    return op->emitOpError("'") << name << "' must be a !py.protocol type";
  return mlir::success();
}

mlir::LogicalResult requireContractTerm(mlir::Operation *op, mlir::Type type,
                                        llvm::StringRef what) {
  if (!isPyContractType(type))
    return op->emitOpError() << what << " must be a Python contract term";
  return mlir::success();
}

mlir::LogicalResult requireI1(mlir::Operation *op, mlir::Type type,
                              llvm::StringRef what) {
  if (!type.isInteger(1))
    return op->emitOpError() << what << " must be i1";
  return mlir::success();
}

bool isLiteralSpelling(mlir::Type type, llvm::StringRef spelling) {
  auto literal = mlir::dyn_cast<LiteralType>(type);
  return literal && literal.getSpelling() == spelling;
}

mlir::LogicalResult requireLiteralSpelling(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef spelling,
                                           llvm::StringRef what) {
  if (!isLiteralSpelling(type, spelling))
    return op->emitOpError()
           << what << " must be !py.literal<" << spelling << ">";
  return mlir::success();
}

mlir::LogicalResult verifyStringArray(mlir::Operation *op,
                                      mlir::ArrayAttr array,
                                      llvm::StringRef name) {
  if (!array)
    return mlir::success();
  for (mlir::Attribute attr : array)
    if (!mlir::isa<mlir::StringAttr>(attr))
      return op->emitOpError("'")
             << name << "' must contain only StringAttr values";
  return mlir::success();
}

mlir::LogicalResult verifyTypeArray(mlir::Operation *op, mlir::ArrayAttr array,
                                    llvm::StringRef name,
                                    bool requireContract = false,
                                    bool requireProtocol = false) {
  if (!array)
    return mlir::success();
  for (mlir::Attribute attr : array) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr)
      return op->emitOpError("'")
             << name << "' must contain only TypeAttr values";
    mlir::Type type = typeAttr.getValue();
    if (requireProtocol && !mlir::isa<ProtocolType>(type))
      return op->emitOpError("'")
             << name << "' must contain only !py.protocol terms";
    if (requireContract && !isPyContractType(type))
      return op->emitOpError("'")
             << name << "' must contain Python contract terms";
  }
  return mlir::success();
}

mlir::LogicalResult verifySameLength(mlir::Operation *op, mlir::ArrayAttr lhs,
                                     llvm::StringRef lhsName,
                                     mlir::ArrayAttr rhs,
                                     llvm::StringRef rhsName) {
  if (!lhs || !rhs)
    return mlir::success();
  if (lhs.size() != rhs.size())
    return op->emitOpError("'") << lhsName << "' and '" << rhsName
                                << "' must have the same number of elements";
  return mlir::success();
}

mlir::LogicalResult verifyDestination(mlir::Operation *op, mlir::Block *dest,
                                      mlir::ValueRange operands,
                                      llvm::StringRef name) {
  if (!dest)
    return op->emitOpError(name) << " destination is missing";
  if (dest->getNumArguments() != operands.size())
    return op->emitOpError(name)
           << " destination argument count must match operands";
  for (auto [operand, arg] : llvm::zip(operands, dest->getArguments()))
    if (operand.getType() != arg.getType())
      return op->emitOpError(name)
             << " destination argument types must match operands";
  return mlir::success();
}

mlir::LogicalResult verifyTryYieldTypes(mlir::Operation *op,
                                        mlir::Region &region,
                                        llvm::StringRef regionName) {
  if (region.empty())
    return mlir::success();
  mlir::Operation *term = region.front().getTerminator();
  if (!term)
    return op->emitOpError(regionName) << " region must have a terminator";
  if (regionName == "try" && !mlir::isa<TryYieldOp>(term))
    return mlir::success();
  if (regionName == "except" && !mlir::isa<ExceptYieldOp>(term))
    return mlir::success();
  if (term->getNumOperands() != op->getNumResults())
    return op->emitOpError(regionName)
           << " yield operand count must match py.try result count";
  for (auto [operand, result] :
       llvm::zip(term->getOperands(), op->getResults()))
    if (operand.getType() != result.getType())
      return op->emitOpError(regionName)
             << " yield operand types must match py.try result types";
  return mlir::success();
}

mlir::LogicalResult verifyResolvedProtocolCall(mlir::Operation *op,
                                               llvm::StringRef attrName) {
  return requireProtocolAttr(op, attrName);
}

mlir::LogicalResult verifyContractResults(mlir::Operation *op) {
  for (mlir::Value result : op->getResults())
    if (mlir::failed(requireContractTerm(op, result.getType(), "result type")))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult verifyOptionalStringAttr(mlir::Operation *op,
                                             llvm::StringRef name) {
  mlir::Attribute attr = op->getAttr(name);
  if (!attr)
    return mlir::success();
  auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
  if (!stringAttr || stringAttr.getValue().empty())
    return op->emitOpError("'") << name << "' must be a non-empty StringAttr";
  return mlir::success();
}

mlir::LogicalResult verifyDescriptorKindAttr(mlir::Operation *op,
                                             llvm::StringRef name,
                                             bool allowField) {
  mlir::Attribute attr = op->getAttr(name);
  if (!attr)
    return mlir::success();
  auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
  if (!stringAttr)
    return op->emitOpError("'") << name << "' must be a StringAttr";
  llvm::StringRef kind = stringAttr.getValue();
  if ((allowField && kind == "field") || kind == "instance" ||
      kind == "static" || kind == "class" || kind == "classmethod")
    return mlir::success();
  return op->emitOpError("unsupported descriptor kind '") << kind << "'";
}

mlir::LogicalResult verifyBinarySpecialMethod(mlir::Operation *op) {
  auto methodName = op->getAttrOfType<mlir::StringAttr>("method_name");
  if (!methodName || methodName.getValue().empty())
    return op->emitOpError("requires non-empty 'method_name'");
  if (mlir::failed(requireProtocolAttr(op, "callee_contract")))
    return mlir::failure();
  return verifyContractResults(op);
}

mlir::LogicalResult verifyUnarySpecialMethod(mlir::Operation *op) {
  auto methodName = op->getAttrOfType<mlir::StringAttr>("method_name");
  if (!methodName || methodName.getValue().empty())
    return op->emitOpError("requires non-empty 'method_name'");
  if (mlir::failed(requireProtocolAttr(op, "callee_contract")))
    return mlir::failure();
  return verifyContractResults(op);
}

bool unionContainsOrCovers(UnionType unionType, mlir::Type member) {
  if (unionType.hasMember(member))
    return true;
  if (auto memberUnion = mlir::dyn_cast<UnionType>(member))
    return llvm::all_of(memberUnion.getMemberTypes(), [&](mlir::Type type) {
      return unionType.hasMember(type);
    });
  return false;
}

std::optional<std::string> nominalContractName(mlir::Type type) {
  if (auto contract = mlir::dyn_cast<ContractType>(type))
    return contract.getContractName().str();
  return std::nullopt;
}

std::optional<std::string> nominalClassSymbolName(mlir::Operation *op,
                                                  mlir::Type type) {
  std::optional<std::string> name = nominalContractName(type);
  if (!name)
    return std::nullopt;
  if (type_object::lookup(op, *name))
    return name;
  llvm::StringRef shortName = llvm::StringRef(*name).rsplit('.').second;
  if (!shortName.empty() && shortName != *name &&
      type_object::lookup(op, shortName))
    return shortName.str();
  return name;
}

mlir::LogicalResult verifyNominalSubclass(mlir::Operation *op,
                                          mlir::Type derivedType,
                                          mlir::Type baseType,
                                          llvm::StringRef relation) {
  std::optional<std::string> derived = nominalClassSymbolName(op, derivedType);
  std::optional<std::string> base = nominalClassSymbolName(op, baseType);
  if (!derived || !base)
    return mlir::success();
  if (!type_object::lookup(op, *derived) || !type_object::lookup(op, *base))
    return mlir::success();
  mlir::FailureOr<bool> isSubclass =
      type_object::isSubclassOf(op, *derived, *base);
  if (mlir::failed(isSubclass))
    return mlir::failure();
  if (!*isSubclass)
    return op->emitOpError() << relation << " requires " << *derived
                             << " to inherit from " << *base;
  return mlir::success();
}

} // namespace

mlir::LogicalResult ClassOp::verify() {
  mlir::Operation *op = getOperation();
  mlir::Region &body = getBody();
  if (!body.empty() && !body.hasOneBlock())
    return emitOpError("body must contain at most one block");

  mlir::ArrayAttr fieldNames = getFieldNamesAttr();
  mlir::ArrayAttr fieldTypes = getFieldTypesAttr();
  mlir::ArrayAttr fieldContracts = getFieldContractTypesAttr();
  mlir::ArrayAttr methodNames = getMethodNamesAttr();
  mlir::ArrayAttr methodContracts = getMethodContractsAttr();
  mlir::ArrayAttr methodKinds = getMethodKindsAttr();
  mlir::ArrayAttr methodSymbols =
      op->getAttrOfType<mlir::ArrayAttr>("method_symbols");

  if (mlir::failed(verifyStringArray(op, getBaseNamesAttr(), "base_names")) ||
      mlir::failed(verifyStringArray(op, fieldNames, "field_names")) ||
      mlir::failed(verifyStringArray(op, methodNames, "method_names")) ||
      mlir::failed(verifyStringArray(op, methodSymbols, "method_symbols")) ||
      mlir::failed(verifyTypeArray(op, fieldTypes, "field_types")) ||
      mlir::failed(
          verifyTypeArray(op, fieldContracts, "field_contract_types", true)) ||
      mlir::failed(
          verifyTypeArray(op, methodContracts, "method_contracts", true)) ||
      mlir::failed(verifyStringArray(op, methodKinds, "method_kinds")))
    return mlir::failure();

  if ((fieldTypes || fieldContracts) && !fieldNames)
    return emitOpError("'field_names' must be provided when field schema "
                       "attributes are present");
  if ((methodNames || methodContracts || methodKinds) &&
      (!methodNames || !methodContracts || !methodKinds))
    return emitOpError("'method_names', 'method_contracts', and "
                       "'method_kinds' must be provided together");
  if (methodSymbols && !methodNames)
    return emitOpError("'method_symbols' requires 'method_names'");

  if (mlir::failed(verifySameLength(op, fieldNames, "field_names", fieldTypes,
                                    "field_types")) ||
      mlir::failed(verifySameLength(op, fieldNames, "field_names",
                                    fieldContracts, "field_contract_types")) ||
      mlir::failed(verifySameLength(op, methodNames, "method_names",
                                    methodContracts, "method_contracts")) ||
      mlir::failed(verifySameLength(op, methodNames, "method_names",
                                    methodKinds, "method_kinds")) ||
      mlir::failed(verifySameLength(op, methodNames, "method_names",
                                    methodSymbols, "method_symbols")))
    return mlir::failure();

  llvm::StringSet<> fieldSet;
  if (fieldNames) {
    for (mlir::Attribute attr : fieldNames) {
      auto name = mlir::cast<mlir::StringAttr>(attr).getValue();
      if (!fieldSet.insert(name).second)
        return emitOpError("duplicate field name '") << name << "'";
    }
  }

  if (methodKinds) {
    for (mlir::Attribute attr : methodKinds) {
      llvm::StringRef kind = mlir::cast<mlir::StringAttr>(attr).getValue();
      if (kind != "instance" && kind != "static" && kind != "class" &&
          kind != "classmethod")
        return emitOpError("unsupported method kind '") << kind << "'";
    }
  }

  if (mlir::ModuleOp module = getOperation()->getParentOfType<mlir::ModuleOp>())
    if (module->hasAttr("ly.typing.module") ||
        module->hasAttr("ly.typing.manifest"))
      return mlir::success(); // cross-manifest bases resolve in the table
  return type_object::verifyBases(*this);
}

mlir::LogicalResult CallOp::verify() {
  if (mlir::failed(requireProtocolAttr(getOperation(), "call_contract")))
    return mlir::failure();
  return verifyContractResults(getOperation());
}

mlir::LogicalResult InvokeOp::verify() {
  if (mlir::failed(requireProtocolAttr(getOperation(), "call_contract")))
    return mlir::failure();
  return verifyDestination(getOperation(), getUnwindDest(),
                           getUnwindDestOperands(), "unwind");
}

mlir::LogicalResult TryOp::verify() {
  mlir::Operation *op = getOperation();
  if (op->getRegion(0).empty())
    return emitOpError("try region must not be empty");
  if (op->getRegion(1).empty() && op->getRegion(2).empty())
    return emitOpError("py.try requires an except or finally region");
  if (mlir::failed(verifyTryYieldTypes(op, op->getRegion(0), "try")) ||
      mlir::failed(verifyTryYieldTypes(op, op->getRegion(1), "except")))
    return mlir::failure();
  if (!op->getRegion(2).empty()) {
    bool hasFinallyYield = false;
    op->getRegion(2).walk([&](FinallyYieldOp) { hasFinallyYield = true; });
    if (!hasFinallyYield)
      return emitOpError("finally region must contain py.finally.yield");
  }
  return mlir::success();
}

mlir::LogicalResult FinallyYieldOp::verify() {
  mlir::Operation *parent = getOperation()->getParentOp();
  auto tryOp = mlir::dyn_cast_or_null<TryOp>(parent);
  if (!tryOp)
    return emitOpError("must be nested inside py.try");
  if (getOperation()->getParentRegion() != &tryOp.getOperation()->getRegion(2))
    return emitOpError("must be inside py.try finally region");
  if (getOperands().empty())
    return mlir::success();
  if (getNumOperands() != tryOp.getNumResults())
    return emitOpError("with operands must yield exactly ")
           << tryOp.getNumResults() << " values";
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip_equal(getOperands(), tryOp.getResults()))) {
    mlir::Value operand = std::get<0>(pair);
    mlir::Value result = std::get<1>(pair);
    if (operand.getType() != result.getType())
      return emitOpError("operand ")
             << index << " type " << operand.getType()
             << " does not match py.try result type " << result.getType();
  }
  return mlir::success();
}

mlir::LogicalResult RaiseOp::verify() {
  if (mlir::failed(requireContractTerm(getOperation(), getException().getType(),
                                       "exception operand")))
    return mlir::failure();
  if (getCause() && getFromNone())
    return emitOpError("cannot carry both a cause operand and from_none");
  if (getCause())
    return requireContractTerm(getOperation(), getCause().getType(),
                               "cause operand");
  return mlir::success();
}

mlir::LogicalResult RaiseCurrentOp::verify() { return mlir::success(); }

mlir::LogicalResult ExceptMatchOp::verify() {
  if (mlir::failed(requireI1(getOperation(), getResult().getType(), "result")))
    return mlir::failure();
  return requireTypeAttr(getOperation(), "handler");
}

mlir::LogicalResult ExceptCurrentMatchOp::verify() {
  if (mlir::failed(requireI1(getOperation(), getResult().getType(), "result")))
    return mlir::failure();
  return requireTypeAttr(getOperation(), "handler");
}

mlir::LogicalResult ExceptCurrentValueOp::verify() {
  if (mlir::failed(requireTypeAttr(getOperation(), "handler")))
    return mlir::failure();
  auto handler = mlir::cast<TypeType>(getHandler());
  if (getResult().getType() != handler.getInstanceType())
    return emitOpError("result type must match handler instance type");
  return mlir::success();
}

mlir::LogicalResult AwaitOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "await_contract");
}

mlir::LogicalResult YieldFromOp::verify() {
  if (mlir::failed(requireProtocolAttr(getOperation(), "yield_from_contract")))
    return mlir::failure();
  if (mlir::failed(requireContractTerm(getOperation(), getSource().getType(),
                                       "source")))
    return mlir::failure();
  return requireContractTerm(getOperation(), getResult().getType(), "result");
}

mlir::LogicalResult BindingRefOp::verify() {
  if (getBinding().empty())
    return emitOpError("binding must not be empty");
  for (mlir::Value capture : getCaptures())
    if (mlir::failed(requireContractTerm(getOperation(), capture.getType(),
                                         "capture operand")))
      return mlir::failure();
  return verifyContractResults(getOperation());
}

mlir::LogicalResult PackOp::verify() {
  return verifyContractResults(getOperation());
}

mlir::LogicalResult NewOp::verify() {
  if (mlir::failed(requireProtocolAttr(getOperation(), "new_contract")))
    return mlir::failure();
  if (!mlir::isa<TypeType>(getClassObject().getType()))
    return emitOpError("class object must be !py.type");
  if (mlir::failed(requireContractTerm(getOperation(), getInstance().getType(),
                                       "instance result")) ||
      mlir::failed(
          verifyOptionalStringAttr(getOperation(), "ly.constructor.owner")) ||
      mlir::failed(verifyDescriptorKindAttr(getOperation(),
                                            "ly.constructor.new_kind",
                                            /*allowField=*/false)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult InitOp::verify() {
  if (mlir::failed(requireProtocolAttr(getOperation(), "init_contract")))
    return mlir::failure();
  if (mlir::failed(requireLiteralSpelling(getOperation(), getResult().getType(),
                                          "None", "result")) ||
      mlir::failed(
          verifyOptionalStringAttr(getOperation(), "ly.constructor.owner")) ||
      mlir::failed(verifyDescriptorKindAttr(getOperation(),
                                            "ly.constructor.init_kind",
                                            /*allowField=*/false)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult EnterOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult ExitOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult AEnterOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult AExitOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult RoundOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least a receiver operand");
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult GetItemOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult SetItemOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult DelItemOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult ContainsOp::verify() {
  if (mlir::failed(requireI1(getOperation(), getResult().getType(), "result")))
    return mlir::failure();
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}
mlir::LogicalResult BoolOp::verify() {
  if (mlir::failed(requireI1(getOperation(), getResult().getType(), "result")))
    return mlir::failure();
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult NoneOp::verify() {
  return requireLiteralSpelling(getOperation(), getResult().getType(), "None",
                                "result");
}

mlir::LogicalResult BoolConstantOp::verify() {
  llvm::StringRef expected = getValue() ? "True" : "False";
  return requireLiteralSpelling(getOperation(), getResult().getType(), expected,
                                "result");
}

mlir::LogicalResult StrConstantOp::verify() {
  return requireContractTerm(getOperation(), getResult().getType(), "result");
}

mlir::LogicalResult FloatConstantOp::verify() {
  return requireContractTerm(getOperation(), getResult().getType(), "result");
}

mlir::LogicalResult CastFromPrimOp::verify() {
  return requireContractTerm(getOperation(), getResult().getType(), "result");
}

mlir::LogicalResult CastToPrimOp::verify() {
  llvm::StringRef mode = getMode();
  if (mode != "exact" && mode != "truncate" && mode != "saturate")
    return emitOpError("mode must be one of exact, truncate, or saturate");
  return requireContractTerm(getOperation(), getInput().getType(), "input");
}

mlir::LogicalResult NegOp::verify() {
  return verifyUnarySpecialMethod(getOperation());
}
mlir::LogicalResult PosOp::verify() {
  return verifyUnarySpecialMethod(getOperation());
}
mlir::LogicalResult InvertOp::verify() {
  return verifyUnarySpecialMethod(getOperation());
}

mlir::LogicalResult AddOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult SubOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult MulOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult DivOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult FloorDivOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult ModOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult LShiftOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult RShiftOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult BitAndOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult BitOrOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult BitXorOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult PowOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult LeOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult LtOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult GtOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult GeOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult EqOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}
mlir::LogicalResult NeOp::verify() {
  return verifyBinarySpecialMethod(getOperation());
}

mlir::LogicalResult ReprOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult StrOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult IntOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult FloatOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult TypeObjectOp::verify() {
  auto resultType = mlir::dyn_cast<TypeType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !py.type");
  auto instanceAttr =
      getOperation()->getAttrOfType<mlir::TypeAttr>("instance_contract");
  if (!instanceAttr)
    return emitOpError("requires instance_contract");
  if (resultType.getInstanceType() != instanceAttr.getValue())
    return emitOpError("result type must match instance_contract");
  return mlir::success();
}

mlir::LogicalResult ClassUpcastOp::verify() {
  if (mlir::failed(
          requireContractTerm(getOperation(), getInput().getType(), "input")) ||
      mlir::failed(
          requireContractTerm(getOperation(), getResult().getType(), "result")))
    return mlir::failure();
  return verifyNominalSubclass(getOperation(), getInput().getType(),
                               getResult().getType(), "class upcast");
}

mlir::LogicalResult ClassRefineOp::verify() {
  if (mlir::failed(
          requireContractTerm(getOperation(), getInput().getType(), "input")) ||
      mlir::failed(
          requireContractTerm(getOperation(), getResult().getType(), "result")))
    return mlir::failure();
  return verifyNominalSubclass(getOperation(), getResult().getType(),
                               getInput().getType(), "class refine");
}

mlir::LogicalResult ProtocolViewOp::verify() {
  if (!mlir::isa<ProtocolType>(getResult().getType()))
    return emitOpError("result must be !py.protocol");
  return requireContractTerm(getOperation(), getInput().getType(), "input");
}

mlir::LogicalResult ClassTestOp::verify() {
  if (mlir::failed(
          requireI1(getOperation(), getResult().getType(), "result")) ||
      mlir::failed(
          requireContractTerm(getOperation(), getInput().getType(), "input")))
    return mlir::failure();
  std::optional<std::string> target =
      nominalClassSymbolName(getOperation(), getTarget());
  if (!target)
    return emitOpError("target must be a nominal class contract");
  if (!type_object::lookup(getOperation(), *target))
    return emitOpError("target has no class schema '") << *target << "'";
  return mlir::success();
}

mlir::LogicalResult AttrGetOp::verify() {
  if (getName().empty())
    return emitOpError("attribute name must not be empty");
  if (mlir::failed(requireContractTerm(getOperation(), getResult().getType(),
                                       "result")) ||
      mlir::failed(verifyOptionalStringAttr(getOperation(), "ly.attr.owner")) ||
      mlir::failed(verifyDescriptorKindAttr(getOperation(), "ly.attr.kind",
                                            /*allowField=*/true)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult AttrSetOp::verify() {
  if (getName().empty())
    return emitOpError("attribute name must not be empty");
  if (mlir::failed(
          requireContractTerm(getOperation(), getValue().getType(), "value")) ||
      mlir::failed(verifyOptionalStringAttr(getOperation(), "ly.attr.owner")) ||
      mlir::failed(verifyDescriptorKindAttr(getOperation(), "ly.attr.kind",
                                            /*allowField=*/true)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult IterOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult NextOp::verify() {
  if (mlir::failed(requireI1(getOperation(), getValid().getType(), "valid")))
    return mlir::failure();
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult AIterOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult ANextOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult LenOp::verify() {
  return verifyResolvedProtocolCall(getOperation(), "callee_contract");
}

mlir::LogicalResult UnionWrapOp::verify() {
  auto unionType = mlir::dyn_cast<UnionType>(getResult().getType());
  if (!unionType)
    return emitOpError("result must be !py.union");
  if (!unionContainsOrCovers(unionType, getInput().getType()))
    return emitOpError("input type must be a member of the result union");
  return mlir::success();
}

mlir::LogicalResult UnionTestOp::verify() {
  if (mlir::failed(requireI1(getOperation(), getResult().getType(), "result")))
    return mlir::failure();
  auto unionType = mlir::dyn_cast<UnionType>(getInput().getType());
  if (!unionType)
    return emitOpError("input must be !py.union");
  if (!unionType.hasMember(getMember()))
    return emitOpError("tested member must be part of the input union");
  return mlir::success();
}

mlir::LogicalResult UnionUnwrapOp::verify() {
  auto unionType = mlir::dyn_cast<UnionType>(getInput().getType());
  if (!unionType)
    return emitOpError("input must be !py.union");
  if (!unionContainsOrCovers(unionType, getResult().getType()))
    return emitOpError("result type must be a member of the input union");
  return mlir::success();
}

} // namespace py
