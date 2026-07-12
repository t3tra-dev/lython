#include "Runtime/Ctypes/Internal.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSwitch.h"

namespace py::lowering::ctypes {

std::optional<TargetPlatformFacts> targetPlatformFacts(mlir::ModuleOp module) {
  return py::native::readTargetPlatformFacts(module);
}

std::optional<CtypesLayout>
ctypesLayout(llvm::StringRef contract,
             const std::optional<TargetPlatformFacts> &facts) {
  std::optional<py::native::NativeABIType> native =
      py::native::ctypesScalarLayout(contract, facts);
  if (!native)
    return std::nullopt;
  CtypesLayout layout;
  layout.size = native->size;
  layout.align = native->align;
  switch (native->kind) {
  case py::native::NativeABIKind::SignedInteger:
    layout.kind = CtypesLayout::ABIKind::SignedInteger;
    break;
  case py::native::NativeABIKind::UnsignedInteger:
    layout.kind = CtypesLayout::ABIKind::UnsignedInteger;
    break;
  case py::native::NativeABIKind::Floating:
    layout.kind = CtypesLayout::ABIKind::Floating;
    break;
  case py::native::NativeABIKind::Pointer:
    layout.kind = CtypesLayout::ABIKind::Pointer;
    break;
  }
  return layout;
}

std::uint64_t alignTo(std::uint64_t value, std::uint64_t align) {
  if (align <= 1)
    return value;
  std::uint64_t remainder = value % align;
  return remainder == 0 ? value : value + (align - remainder);
}

py::ClassOp lookupClassForContract(mlir::ModuleOp module,
                                   llvm::StringRef contract) {
  auto lookup = [&](llvm::StringRef name) -> py::ClassOp {
    return mlir::dyn_cast_or_null<py::ClassOp>(
        mlir::SymbolTable::lookupSymbolIn(module.getOperation(), name));
  };
  if (py::ClassOp classOp = lookup(contract))
    return classOp;
  llvm::StringRef shortName = contract.rsplit('.').second;
  if (!shortName.empty() && shortName != contract)
    return lookup(shortName);
  return {};
}

std::optional<std::string> ctypesContractName(mlir::Type type) {
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(type))
    type = typeObject.getInstanceType();
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract)
    return std::nullopt;
  return contract.getContractName().str();
}

std::optional<llvm::StringRef> stringValue(mlir::DictionaryAttr dict,
                                           llvm::StringRef name) {
  auto attr = mlir::dyn_cast_or_null<mlir::StringAttr>(dict.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

std::optional<mlir::Type> resolveCtypesSourceExpr(mlir::MLIRContext *context,
                                                  mlir::ModuleOp module,
                                                  mlir::Attribute attr,
                                                  unsigned depth = 0);

std::optional<std::int64_t> sourceExprStaticInteger(mlir::Attribute attr) {
  auto dict = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(attr);
  if (!dict)
    return std::nullopt;
  std::optional<llvm::StringRef> kind = stringValue(dict, "kind");
  if (!kind || *kind != "constant.int")
    return std::nullopt;
  std::optional<llvm::StringRef> value = stringValue(dict, "value");
  if (!value)
    return std::nullopt;
  std::int64_t parsed = 0;
  if (value->getAsInteger(10, parsed))
    return std::nullopt;
  return parsed;
}

std::optional<mlir::Attribute> moduleStaticValue(mlir::ModuleOp module,
                                                 llvm::StringRef name) {
  auto names =
      module->getAttrOfType<mlir::ArrayAttr>("ly.module_static_attr_names");
  auto values =
      module->getAttrOfType<mlir::ArrayAttr>("ly.module_static_attr_values");
  if (!names || !values || names.size() != values.size())
    return std::nullopt;
  for (auto [index, attr] : llvm::enumerate(names)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (stringAttr && stringAttr.getValue() == name)
      return values[index];
  }
  return std::nullopt;
}

std::optional<mlir::Type> resolveCtypesSourceExpr(mlir::MLIRContext *context,
                                                  mlir::ModuleOp module,
                                                  mlir::Attribute attr,
                                                  unsigned depth) {
  if (depth > 16)
    return std::nullopt;
  auto dict = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(attr);
  if (!dict)
    return std::nullopt;
  std::optional<llvm::StringRef> kind = stringValue(dict, "kind");
  if (!kind)
    return std::nullopt;

  if (*kind == "ref") {
    std::optional<llvm::StringRef> name = stringValue(dict, "name");
    if (!name)
      return std::nullopt;
    if (std::optional<std::string> contract =
            ctypesQualifiedNameContract(*context, *name))
      return ctypesContractType(context, *contract);
    if (py::ClassOp classOp = lookupClassForContract(module, *name))
      return ctypesContractType(context, classOp.getSymName());
    if (std::optional<mlir::Attribute> alias = moduleStaticValue(module, *name))
      return resolveCtypesSourceExpr(context, module, *alias, depth + 1);
    return std::nullopt;
  }

  if (*kind == "call") {
    mlir::Attribute callee = dict.get("callee");
    auto args = mlir::dyn_cast_or_null<mlir::ArrayAttr>(dict.get("args"));
    auto calleeDict = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(callee);
    if (!calleeDict || !args || args.size() != 1)
      return std::nullopt;
    std::optional<llvm::StringRef> calleeKind = stringValue(calleeDict, "kind");
    std::optional<llvm::StringRef> calleeName = stringValue(calleeDict, "name");
    if (!calleeKind || *calleeKind != "ref" || !calleeName)
      return std::nullopt;
    if (*calleeName != "ctypes.POINTER" && *calleeName != "POINTER")
      return std::nullopt;
    std::optional<mlir::Type> pointee =
        resolveCtypesSourceExpr(context, module, args[0], depth + 1);
    if (!pointee)
      return std::nullopt;
    return py::ContractType::get(context, "_ctypes._Pointer", {*pointee});
  }

  if (*kind == "binop") {
    std::optional<llvm::StringRef> op = stringValue(dict, "op");
    if (!op || *op != "Mult")
      return std::nullopt;
    mlir::Attribute left = dict.get("left");
    mlir::Attribute right = dict.get("right");
    std::optional<mlir::Type> element =
        resolveCtypesSourceExpr(context, module, left, depth + 1);
    std::optional<std::int64_t> count = sourceExprStaticInteger(right);
    if (!element || !count) {
      element = resolveCtypesSourceExpr(context, module, right, depth + 1);
      count = sourceExprStaticInteger(left);
    }
    if (!element || !count || *count < 0)
      return std::nullopt;
    return py::ContractType::get(
        context, "_ctypes.Array",
        {*element, py::LiteralType::get(context, std::to_string(*count))});
  }

  return std::nullopt;
}

std::optional<mlir::Attribute> classStaticValue(py::ClassOp classOp,
                                                llvm::StringRef name) {
  auto names =
      classOp->getAttrOfType<mlir::ArrayAttr>("class_static_attr_names");
  auto values =
      classOp->getAttrOfType<mlir::ArrayAttr>("class_static_attr_values");
  if (!names || !values || names.size() != values.size())
    return std::nullopt;
  for (auto [index, attr] : llvm::enumerate(names)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (stringAttr && stringAttr.getValue() == name)
      return values[index];
  }
  return std::nullopt;
}

std::optional<std::uint64_t> classStaticPositiveInteger(py::ClassOp classOp,
                                                        llvm::StringRef name) {
  std::optional<mlir::Attribute> value = classStaticValue(classOp, name);
  if (!value)
    return std::nullopt;
  std::optional<std::int64_t> integer = sourceExprStaticInteger(*value);
  if (!integer || *integer <= 0)
    return std::nullopt;
  return static_cast<std::uint64_t>(*integer);
}

std::optional<std::string> ctypesAggregateKind(mlir::ModuleOp module,
                                               py::ClassOp classOp) {
  auto legacyKind = classOp->getAttrOfType<mlir::StringAttr>("ly.ctypes.kind");
  if (legacyKind &&
      (legacyKind.getValue() == "struct" || legacyKind.getValue() == "union"))
    return legacyKind.getValue().str();

  auto bases = classOp->getAttrOfType<mlir::ArrayAttr>("base_names");
  if (!bases)
    return std::nullopt;
  for (mlir::Attribute attr : bases) {
    auto base = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!base)
      continue;
    llvm::StringRef name = base.getValue();
    if (name == "ctypes.Structure" || name == "_ctypes.Structure" ||
        name == "Structure")
      return std::string("struct");
    if (name == "ctypes.Union" || name == "_ctypes.Union" || name == "Union")
      return std::string("union");
    if (py::ClassOp baseClass = lookupClassForContract(module, name))
      if (std::optional<std::string> kind =
              ctypesAggregateKind(module, baseClass))
        return kind;
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<std::pair<std::string, mlir::Type>, 8>>
ctypesStaticFields(mlir::ModuleOp module, py::ClassOp classOp) {
  llvm::SmallVector<std::pair<std::string, mlir::Type>, 8> fields;
  auto legacyNames =
      classOp->getAttrOfType<mlir::ArrayAttr>("ly.ctypes.field_names");
  auto legacyTypes =
      classOp->getAttrOfType<mlir::ArrayAttr>("ly.ctypes.field_types");
  if (legacyNames || legacyTypes) {
    if (!legacyNames || !legacyTypes ||
        legacyNames.size() != legacyTypes.size())
      return std::nullopt;
    for (auto [index, typeAttr] : llvm::enumerate(legacyTypes)) {
      auto name = mlir::dyn_cast<mlir::StringAttr>(legacyNames[index]);
      auto type = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
      if (!name || !type)
        return std::nullopt;
      fields.push_back({name.getValue().str(), type.getValue()});
    }
    return fields;
  }

  std::optional<mlir::Attribute> staticFields =
      classStaticValue(classOp, "_fields_");
  if (!staticFields)
    return fields;
  auto fieldsDict =
      mlir::dyn_cast_if_present<mlir::DictionaryAttr>(*staticFields);
  if (!fieldsDict)
    return std::nullopt;
  std::optional<llvm::StringRef> fieldsKind = stringValue(fieldsDict, "kind");
  if (!fieldsKind || (*fieldsKind != "list" && *fieldsKind != "tuple"))
    return std::nullopt;
  auto entries =
      mlir::dyn_cast_or_null<mlir::ArrayAttr>(fieldsDict.get("elts"));
  if (!entries)
    return std::nullopt;
  for (mlir::Attribute entryAttr : entries) {
    auto entry = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(entryAttr);
    if (!entry)
      return std::nullopt;
    std::optional<llvm::StringRef> entryKind = stringValue(entry, "kind");
    if (!entryKind || *entryKind != "tuple")
      return std::nullopt;
    auto elts = mlir::dyn_cast_or_null<mlir::ArrayAttr>(entry.get("elts"));
    if (!elts || elts.size() < 2 || elts.size() > 2)
      return std::nullopt;
    auto nameDict = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(elts[0]);
    if (!nameDict)
      return std::nullopt;
    std::optional<llvm::StringRef> nameKind = stringValue(nameDict, "kind");
    std::optional<llvm::StringRef> name = stringValue(nameDict, "value");
    if (!nameKind || *nameKind != "constant.str" || !name)
      return std::nullopt;
    std::optional<mlir::Type> fieldType =
        resolveCtypesSourceExpr(classOp.getContext(), module, elts[1]);
    if (!fieldType)
      return std::nullopt;
    fields.push_back({name->str(), *fieldType});
  }
  return fields;
}

std::optional<CtypesAggregateLayout>
ctypesAggregateLayout(mlir::ModuleOp module, py::ClassOp classOp,
                      const std::optional<TargetPlatformFacts> &facts,
                      unsigned depth);

std::optional<CtypesLayout>
ctypesStaticLayout(mlir::ModuleOp module, llvm::StringRef contract,
                   const std::optional<TargetPlatformFacts> &facts,
                   unsigned depth) {
  if (std::optional<CtypesLayout> scalar = ctypesLayout(contract, facts))
    return scalar;
  if (depth > 16)
    return std::nullopt;
  py::ClassOp classOp = lookupClassForContract(module, contract);
  if (!classOp)
    return std::nullopt;
  std::optional<CtypesAggregateLayout> aggregate =
      ctypesAggregateLayout(module, classOp, facts, depth + 1);
  if (!aggregate)
    return std::nullopt;
  return aggregate->layout;
}

std::optional<std::int64_t> ctypesLiteralInteger(mlir::Type type) {
  auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
  if (!literal)
    return std::nullopt;
  std::int64_t value = 0;
  if (literal.getSpelling().getAsInteger(10, value))
    return std::nullopt;
  return value;
}

std::optional<CtypesLayout>
ctypesStaticLayoutForType(mlir::ModuleOp module, mlir::Type type,
                          const std::optional<TargetPlatformFacts> &facts,
                          unsigned depth) {
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(type))
    type = typeObject.getInstanceType();
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract)
    return std::nullopt;
  if (contract.getContractName() == "_ctypes.Array" &&
      contract.getArguments().size() >= 2) {
    mlir::Type elementType = contract.getArguments().front();
    std::optional<CtypesLayout> element =
        ctypesStaticLayoutForType(module, elementType, facts, depth + 1);
    std::optional<std::int64_t> count =
        ctypesLiteralInteger(contract.getArguments()[1]);
    if (!element || !count || *count < 0)
      return std::nullopt;
    CtypesLayout layout;
    layout.kind = CtypesLayout::ABIKind::Aggregate;
    layout.align = element->align;
    layout.size = element->size * static_cast<std::uint64_t>(*count);
    return layout;
  }
  return ctypesStaticLayout(module, contract.getContractName(), facts, depth);
}

std::optional<CtypesArrayType>
ctypesArrayType(mlir::ModuleOp module, mlir::Type type,
                const std::optional<TargetPlatformFacts> &facts,
                unsigned depth) {
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(type))
    type = typeObject.getInstanceType();
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract || contract.getContractName() != "_ctypes.Array" ||
      contract.getArguments().size() < 2 || depth > 16)
    return std::nullopt;
  mlir::Type elementType = contract.getArguments().front();
  std::optional<std::string> elementContract = ctypesContractName(elementType);
  std::optional<CtypesLayout> elementLayout =
      ctypesStaticLayoutForType(module, elementType, facts, depth + 1);
  std::optional<std::int64_t> count =
      ctypesLiteralInteger(contract.getArguments()[1]);
  if (!elementContract || !elementLayout || !count || *count < 0)
    return std::nullopt;

  CtypesArrayType array;
  array.elementType = elementType;
  array.elementContract = *elementContract;
  array.elementLayout = *elementLayout;
  array.layout.kind = CtypesLayout::ABIKind::Aggregate;
  array.layout.align = elementLayout->align;
  array.count = static_cast<std::uint64_t>(*count);
  array.layout.size = elementLayout->size * array.count;
  return array;
}

std::optional<CtypesAggregateLayout>
ctypesAggregateLayout(mlir::ModuleOp module, py::ClassOp classOp,
                      const std::optional<TargetPlatformFacts> &facts,
                      unsigned depth) {
  std::optional<std::string> kind = ctypesAggregateKind(module, classOp);
  if (!kind || (*kind != "struct" && *kind != "union"))
    return std::nullopt;
  std::optional<llvm::SmallVector<std::pair<std::string, mlir::Type>, 8>>
      staticFields = ctypesStaticFields(module, classOp);
  if (!staticFields)
    return std::nullopt;

  CtypesAggregateLayout result;
  result.isUnion = *kind == "union";
  result.layout.kind = CtypesLayout::ABIKind::Aggregate;
  result.layout.align = 1;
  std::optional<std::uint64_t> pack =
      classStaticPositiveInteger(classOp, "_pack_");
  std::optional<std::uint64_t> alignOverride =
      classStaticPositiveInteger(classOp, "_align_");
  std::uint64_t offset = 0;
  for (const auto &staticField : *staticFields) {
    std::optional<std::string> contract =
        ctypesContractName(staticField.second);
    if (!contract)
      return std::nullopt;
    std::optional<CtypesLayout> fieldLayout =
        ctypesStaticLayoutForType(module, staticField.second, facts, depth + 1);
    if (!fieldLayout || fieldLayout->size == 0 || fieldLayout->align == 0)
      return std::nullopt;

    CtypesFieldLayout field;
    field.name = staticField.first;
    field.contract = *contract;
    field.type = staticField.second;
    field.layout = *fieldLayout;
    std::uint64_t fieldAlign = fieldLayout->align;
    if (pack)
      fieldAlign = std::min(fieldAlign, *pack);
    if (fieldAlign == 0)
      return std::nullopt;
    if (result.isUnion) {
      field.offset = 0;
      result.layout.size = std::max(result.layout.size, fieldLayout->size);
    } else {
      offset = alignTo(offset, fieldAlign);
      field.offset = offset;
      offset += fieldLayout->size;
      result.layout.size = offset;
    }
    result.layout.align = std::max(result.layout.align, fieldAlign);
    result.fields.push_back(std::move(field));
  }
  if (alignOverride)
    result.layout.align = std::max(result.layout.align, *alignOverride);
  result.layout.size = alignTo(result.layout.size, result.layout.align);
  return result;
}

} // namespace py::lowering::ctypes
