#include "RuntimeLowering/RuntimeLowering.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"

namespace py::runtime_lowering {
namespace {

struct CtypesLayout {
  enum class ABIKind {
    SignedInteger,
    UnsignedInteger,
    Floating,
    Pointer,
    Aggregate
  };

  std::uint64_t size = 0;
  std::uint64_t align = 0;
  ABIKind kind = ABIKind::SignedInteger;
};

struct CtypesFieldLayout {
  std::string name;
  std::string contract;
  mlir::Type type;
  CtypesLayout layout;
  std::uint64_t offset = 0;
};

struct CtypesAggregateLayout {
  CtypesLayout layout;
  llvm::SmallVector<CtypesFieldLayout, 8> fields;
  bool isUnion = false;
};

struct CtypesArrayType {
  mlir::Type elementType;
  std::string elementContract;
  CtypesLayout elementLayout;
  CtypesLayout layout;
  std::uint64_t count = 0;
};

struct TargetPlatformFacts {
  std::string triple;
  std::uint64_t pointerWidth = 0;
  std::uint64_t cLongWidth = 0;

  std::uint64_t pointerBytes() const { return pointerWidth / 8; }
  std::uint64_t cLongBytes() const { return cLongWidth / 8; }
};

struct CtypesNameMap {
  llvm::StringLiteral name;
  llvm::StringLiteral contract;
};

constexpr CtypesNameMap kCtypesPublicTypes[] = {
    {"CDLL", "ctypes.CDLL"},
    {"WinDLL", "ctypes.WinDLL"},
    {"OleDLL", "ctypes.OleDLL"},
    {"PyDLL", "ctypes.PyDLL"},
    {"LibraryLoader", "ctypes.LibraryLoader"},
    {"ArgumentError", "ctypes.ArgumentError"},
    {"Structure", "_ctypes.Structure"},
    {"Union", "_ctypes.Union"},
    {"Array", "_ctypes.Array"},
    {"_CData", "_ctypes._CData"},
    {"_SimpleCData", "_ctypes._SimpleCData"},
    {"_Pointer", "_ctypes._Pointer"},
    {"_CArgObject", "_ctypes._CArgObject"},
    {"CFuncPtr", "_ctypes.CFuncPtr"},
    {"CField", "_ctypes.CField"},
    {"py_object", "ctypes.py_object"},
    {"c_bool", "ctypes.c_bool"},
    {"c_byte", "ctypes.c_byte"},
    {"c_ubyte", "ctypes.c_ubyte"},
    {"c_short", "ctypes.c_short"},
    {"c_ushort", "ctypes.c_ushort"},
    {"c_int", "ctypes.c_int"},
    {"c_uint", "ctypes.c_uint"},
    {"c_long", "ctypes.c_long"},
    {"c_ulong", "ctypes.c_ulong"},
    {"c_longlong", "ctypes.c_longlong"},
    {"c_ulonglong", "ctypes.c_ulonglong"},
    {"c_int8", "ctypes.c_int8"},
    {"c_uint8", "ctypes.c_uint8"},
    {"c_int16", "ctypes.c_int16"},
    {"c_uint16", "ctypes.c_uint16"},
    {"c_int32", "ctypes.c_int32"},
    {"c_uint32", "ctypes.c_uint32"},
    {"c_int64", "ctypes.c_int64"},
    {"c_uint64", "ctypes.c_uint64"},
    {"c_ssize_t", "ctypes.c_ssize_t"},
    {"c_size_t", "ctypes.c_size_t"},
    {"c_float", "ctypes.c_float"},
    {"c_double", "ctypes.c_double"},
    {"c_longdouble", "ctypes.c_longdouble"},
    {"c_char", "ctypes.c_char"},
    {"c_wchar", "ctypes.c_wchar"},
    {"c_void_p", "ctypes.c_void_p"},
    {"c_voidp", "ctypes.c_void_p"},
    {"c_char_p", "ctypes.c_char_p"},
    {"c_wchar_p", "ctypes.c_wchar_p"},
    {"HRESULT", "ctypes.HRESULT"},
    {"c_time_t", "ctypes.c_time_t"},
};

constexpr CtypesNameMap kCtypesInternalTypes[] = {
    {"_CData", "_ctypes._CData"},
    {"_CanCastTo", "_ctypes._CanCastTo"},
    {"_PointerLike", "_ctypes._PointerLike"},
    {"_CArgObject", "_ctypes._CArgObject"},
    {"_SimpleCData", "_ctypes._SimpleCData"},
    {"_Pointer", "_ctypes._Pointer"},
    {"Array", "_ctypes.Array"},
    {"CFuncPtr", "_ctypes.CFuncPtr"},
    {"CField", "_ctypes.CField"},
    {"Structure", "_ctypes.Structure"},
    {"Union", "_ctypes.Union"},
};

constexpr CtypesNameMap kCtypesWinTypes[] = {
    {"DWORD", "ctypes.c_ulong"},    {"WORD", "ctypes.c_ushort"},
    {"BYTE", "ctypes.c_ubyte"},     {"BOOL", "ctypes.c_long"},
    {"HANDLE", "ctypes.c_void_p"},  {"LPVOID", "ctypes.c_void_p"},
    {"LPCVOID", "ctypes.c_void_p"},
};

llvm::StringRef stripCtypesModule(llvm::StringRef contract) {
  if (contract.consume_front("ctypes."))
    return contract;
  if (contract.consume_front("_ctypes."))
    return contract;
  return contract;
}

std::optional<std::string> lookupCtypesName(llvm::ArrayRef<CtypesNameMap> table,
                                            llvm::StringRef name) {
  for (const CtypesNameMap &entry : table)
    if (entry.name == name)
      return entry.contract.str();
  return std::nullopt;
}

std::optional<std::string> ctypesModuleAttrContract(llvm::StringRef moduleName,
                                                    llvm::StringRef attr) {
  if (moduleName == "ctypes")
    return lookupCtypesName(kCtypesPublicTypes, attr);
  if (moduleName == "_ctypes")
    return lookupCtypesName(kCtypesInternalTypes, attr);
  if (moduleName == "ctypes.wintypes")
    return lookupCtypesName(kCtypesWinTypes, attr);
  return std::nullopt;
}

std::optional<std::string> ctypesBareNameContract(llvm::StringRef name) {
  if (std::optional<std::string> publicName =
          lookupCtypesName(kCtypesPublicTypes, name))
    return publicName;
  if (std::optional<std::string> internalName =
          lookupCtypesName(kCtypesInternalTypes, name))
    return internalName;
  return lookupCtypesName(kCtypesWinTypes, name);
}

std::optional<std::string> ctypesQualifiedNameContract(llvm::StringRef name) {
  if (name.starts_with("ctypes.") || name.starts_with("_ctypes.")) {
    auto split = name.rsplit('.');
    if (std::optional<std::string> contract =
            ctypesModuleAttrContract(split.first, split.second))
      return contract;
    if (name.consume_front("ctypes.wintypes."))
      return ctypesModuleAttrContract("ctypes.wintypes", name);
  }
  return ctypesBareNameContract(name);
}

bool isStaticCtypesFunctionName(llvm::StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Cases("sizeof", "alignment", "byref", "pointer", "POINTER", "cast",
             "addressof", true)
      .Default(false);
}

mlir::Type ctypesContractType(mlir::MLIRContext *context,
                              llvm::StringRef contract) {
  return py::ContractType::get(context, contract);
}

RuntimeBundle makeCtypesModuleBundle(mlir::Type resultType,
                                     llvm::StringRef moduleName) {
  RuntimeBundle bundle = RuntimeBundle::object(resultType, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Module;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = moduleName.str();
  evidence.ctype = resultType;
  bundle.ctypes = std::move(evidence);
  return bundle;
}

std::optional<TargetPlatformFacts> targetPlatformFacts(mlir::ModuleOp module) {
  auto triple = module->getAttrOfType<mlir::StringAttr>("ly.target.triple");
  auto pointerWidth =
      module->getAttrOfType<mlir::IntegerAttr>("ly.target.pointer_width");
  auto cLongWidth =
      module->getAttrOfType<mlir::IntegerAttr>("ly.target.c_long_width");
  if (!triple || !pointerWidth || !cLongWidth)
    return std::nullopt;
  if (pointerWidth.getInt() <= 0 || cLongWidth.getInt() <= 0)
    return std::nullopt;
  TargetPlatformFacts facts;
  facts.triple = triple.getValue().str();
  facts.pointerWidth = static_cast<std::uint64_t>(pointerWidth.getInt());
  facts.cLongWidth = static_cast<std::uint64_t>(cLongWidth.getInt());
  if (facts.pointerWidth == 0 || facts.pointerWidth % 8 != 0 ||
      facts.cLongWidth == 0 || facts.cLongWidth % 8 != 0)
    return std::nullopt;
  return facts;
}

std::optional<CtypesLayout>
ctypesLayout(llvm::StringRef contract,
             const std::optional<TargetPlatformFacts> &facts) {
  llvm::StringRef name = stripCtypesModule(contract);
  if (std::optional<CtypesLayout> fixed =
          llvm::StringSwitch<std::optional<CtypesLayout>>(name)
              .Cases("c_bool", "c_ubyte", "c_uint8",
                     CtypesLayout{1, 1, CtypesLayout::ABIKind::UnsignedInteger})
              .Cases("c_byte", "c_int8", "c_char",
                     CtypesLayout{1, 1, CtypesLayout::ABIKind::SignedInteger})
              .Cases("c_ushort", "c_uint16",
                     CtypesLayout{2, 2, CtypesLayout::ABIKind::UnsignedInteger})
              .Cases("c_short", "c_int16",
                     CtypesLayout{2, 2, CtypesLayout::ABIKind::SignedInteger})
              .Cases("c_uint", "c_uint32",
                     CtypesLayout{4, 4, CtypesLayout::ABIKind::UnsignedInteger})
              .Cases("c_int", "c_int32",
                     CtypesLayout{4, 4, CtypesLayout::ABIKind::SignedInteger})
              .Case("c_float",
                    CtypesLayout{4, 4, CtypesLayout::ABIKind::Floating})
              .Cases("c_ulonglong", "c_uint64",
                     CtypesLayout{8, 8, CtypesLayout::ABIKind::UnsignedInteger})
              .Cases("c_longlong", "c_int64",
                     CtypesLayout{8, 8, CtypesLayout::ABIKind::SignedInteger})
              .Case("c_double",
                    CtypesLayout{8, 8, CtypesLayout::ABIKind::Floating})
              .Default(std::nullopt))
    return fixed;

  if (!facts)
    return std::nullopt;
  return llvm::StringSwitch<std::optional<CtypesLayout>>(name)
      .Case("c_ulong", CtypesLayout{facts->cLongBytes(), facts->cLongBytes(),
                                    CtypesLayout::ABIKind::UnsignedInteger})
      .Cases("c_long", "HRESULT",
             CtypesLayout{facts->cLongBytes(), facts->cLongBytes(),
                          CtypesLayout::ABIKind::SignedInteger})
      .Case("c_size_t",
            CtypesLayout{facts->pointerBytes(), facts->pointerBytes(),
                         CtypesLayout::ABIKind::UnsignedInteger})
      .Case("c_ssize_t",
            CtypesLayout{facts->pointerBytes(), facts->pointerBytes(),
                         CtypesLayout::ABIKind::SignedInteger})
      .Cases("c_void_p", "_Pointer",
             CtypesLayout{facts->pointerBytes(), facts->pointerBytes(),
                          CtypesLayout::ABIKind::Pointer})
      .Default(std::nullopt);
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
            ctypesQualifiedNameContract(*name))
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
                   unsigned depth = 0) {
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
                          unsigned depth = 0) {
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
                unsigned depth = 0) {
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

std::string targetFactsLabel(const std::optional<TargetPlatformFacts> &facts) {
  if (!facts)
    return "missing target facts";
  return "target '" + facts->triple + "'";
}

bool isFixedOrTargetDependentCtypesScalar(llvm::StringRef contract) {
  llvm::StringRef name = stripCtypesModule(contract);
  return llvm::StringSwitch<bool>(name)
      .Cases("c_bool", "c_byte", "c_ubyte", "c_short", "c_ushort", "c_int",
             "c_uint", true)
      .Cases("c_long", "c_ulong", "c_longlong", "c_ulonglong", true)
      .Cases("c_int8", "c_uint8", "c_int16", "c_uint16", "c_int32", "c_uint32",
             "c_int64", "c_uint64", true)
      .Cases("c_ssize_t", "c_size_t", "c_void_p", true)
      .Default(false);
}

bool isCtypesIntegralLike(llvm::StringRef contract) {
  llvm::StringRef name = stripCtypesModule(contract);
  return llvm::StringSwitch<bool>(name)
      .Cases("c_bool", "c_byte", "c_ubyte", "c_short", "c_ushort", "c_int",
             "c_uint", true)
      .Cases("c_long", "c_ulong", "c_longlong", "c_ulonglong", true)
      .Cases("c_int8", "c_uint8", "c_int16", "c_uint16", "c_int32", "c_uint32",
             "c_int64", "c_uint64", true)
      .Cases("c_ssize_t", "c_size_t", "c_void_p", true)
      .Default(false);
}

bool isCtypesVoidPointer(llvm::StringRef contract) {
  return stripCtypesModule(contract) == "c_void_p";
}

bool isCtypesPointerContract(llvm::StringRef contract) {
  return stripCtypesModule(contract) == "_Pointer";
}

bool isNoneContractName(llvm::StringRef contract) {
  return contract == "types.NoneType";
}

bool isNoneBundle(const RuntimeBundle &bundle) {
  return isNoneContractName(bundle.contractName());
}

bool isStaticSequenceBundle(const RuntimeBundle &bundle) {
  return bundle.contractName() == "builtins.list" ||
         bundle.contractName() == "builtins.tuple" ||
         !bundle.sequenceElementBundles.empty();
}

std::string ctypesContractFromBundle(const RuntimeBundle &bundle) {
  if (bundle.ctypes)
    return bundle.ctypes->ctypeName;
  if (bundle.kind == RuntimeBundle::Kind::TypeObject)
    return bundle.instanceContractName();
  return bundle.contractName();
}

mlir::Type ctypesTypeFromBundle(const RuntimeBundle &bundle) {
  if (bundle.ctypes && bundle.ctypes->ctype)
    return bundle.ctypes->ctype;
  if (bundle.kind == RuntimeBundle::Kind::TypeObject)
    return bundle.instanceContract;
  return bundle.contract;
}

std::optional<std::string> ctypesTypeObjectName(const RuntimeBundle &bundle) {
  if (bundle.kind != RuntimeBundle::Kind::TypeObject)
    return std::nullopt;
  std::string contract = bundle.instanceContractName();
  llvm::StringRef name(contract);
  if (name.starts_with("ctypes.") || name.starts_with("_ctypes."))
    return contract;
  return std::nullopt;
}

std::optional<llvm::StringRef>
ctypesFromAddressTarget(llvm::StringRef binding) {
  if (!binding.consume_front("ctypes.from_address:"))
    return std::nullopt;
  if (binding.empty())
    return std::nullopt;
  return binding;
}

std::optional<llvm::StringRef> ctypesFromBufferTarget(llvm::StringRef binding) {
  if (!binding.consume_front("ctypes.from_buffer:"))
    return std::nullopt;
  if (binding.empty())
    return std::nullopt;
  return binding;
}

std::optional<llvm::StringRef>
ctypesFromBufferCopyTarget(llvm::StringRef binding) {
  if (!binding.consume_front("ctypes.from_buffer_copy:"))
    return std::nullopt;
  if (binding.empty())
    return std::nullopt;
  return binding;
}

void keepAliveSource(RuntimeCtypesEvidence &evidence,
                     const RuntimeBundle &source) {
  if (source.objectValue.contract)
    evidence.keepAlive.push_back(source.objectValue);
  if (source.ctypes)
    evidence.keepAlive.append(source.ctypes->keepAlive.begin(),
                              source.ctypes->keepAlive.end());
}

void keepAliveBufferSource(RuntimeBufferEvidence &evidence,
                           const RuntimeBundle &source) {
  if (source.objectValue.contract)
    evidence.keepAlive.push_back(source.objectValue);
  if (source.buffer)
    evidence.keepAlive.append(source.buffer->keepAlive.begin(),
                              source.buffer->keepAlive.end());
  if (source.ctypes)
    evidence.keepAlive.append(source.ctypes->keepAlive.begin(),
                              source.ctypes->keepAlive.end());
}

mlir::Value cdataStorageAddress(const RuntimeCtypesEvidence &evidence) {
  return evidence.storageAddressValue;
}

mlir::Value cdataStorageAddressValid(const RuntimeCtypesEvidence &evidence) {
  return evidence.storageAddressValid;
}

mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                       bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1)
      .getResult();
}

mlir::Value constantI64(mlir::OpBuilder &builder, mlir::Location loc,
                        std::int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64).getResult();
}

mlir::Value constantIndex(mlir::OpBuilder &builder, mlir::Location loc,
                          std::int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value).getResult();
}

std::string ctypesLibraryABI(llvm::StringRef contract) {
  return llvm::StringSwitch<std::string>(contract)
      .Case("ctypes.CDLL", "cdecl")
      .Case("ctypes.WinDLL", "stdcall")
      .Default("");
}

bool isKnownTrue(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>();
  return constant && constant.value() != 0;
}

bool isIntegerScalarLayout(const CtypesLayout &layout) {
  return layout.kind == CtypesLayout::ABIKind::SignedInteger ||
         layout.kind == CtypesLayout::ABIKind::UnsignedInteger;
}

bool isFloatingScalarLayout(const CtypesLayout &layout) {
  return layout.kind == CtypesLayout::ABIKind::Floating;
}

bool isPointerScalarLayout(const CtypesLayout &layout) {
  return layout.kind == CtypesLayout::ABIKind::Pointer;
}

llvm::StringRef
ctypesProvenanceName(RuntimeCtypesEvidence::Provenance provenance) {
  switch (provenance) {
  case RuntimeCtypesEvidence::Provenance::None:
    return "none";
  case RuntimeCtypesEvidence::Provenance::NativeCell:
    return "native_cell";
  case RuntimeCtypesEvidence::Provenance::CallRegionBorrow:
    return "call_region_borrow";
  case RuntimeCtypesEvidence::Provenance::ExternalAddress:
    return "external_address";
  case RuntimeCtypesEvidence::Provenance::Cast:
    return "cast";
  case RuntimeCtypesEvidence::Provenance::BufferView:
    return "buffer_view";
  case RuntimeCtypesEvidence::Provenance::CallbackThunk:
    return "callback_thunk";
  }
  llvm_unreachable("unhandled ctypes provenance");
}

llvm::StringRef ctypesLifetimeName(RuntimeCtypesEvidence::Lifetime lifetime) {
  switch (lifetime) {
  case RuntimeCtypesEvidence::Lifetime::Unknown:
    return "unknown";
  case RuntimeCtypesEvidence::Lifetime::CallRegion:
    return "call_region";
  case RuntimeCtypesEvidence::Lifetime::Owner:
    return "owner";
  case RuntimeCtypesEvidence::Lifetime::External:
    return "external";
  case RuntimeCtypesEvidence::Lifetime::Static:
    return "static";
  }
  llvm_unreachable("unhandled ctypes lifetime");
}

std::optional<std::int64_t> knownI64Constant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constant)
    return std::nullopt;
  return constant.value();
}

bool fitsStaticIntegerLayout(std::int64_t value, const CtypesLayout &layout) {
  if (!isIntegerScalarLayout(layout))
    return false;
  unsigned bits = static_cast<unsigned>(layout.size * 8);
  if (layout.kind == CtypesLayout::ABIKind::SignedInteger) {
    if (bits >= 64)
      return true;
    std::int64_t min = -(std::int64_t{1} << (bits - 1));
    std::int64_t max = (std::int64_t{1} << (bits - 1)) - 1;
    return min <= value && value <= max;
  }
  if (value < 0)
    return false;
  if (bits >= 64)
    return true;
  std::uint64_t max = (std::uint64_t{1} << bits) - 1;
  return static_cast<std::uint64_t>(value) <= max;
}

mlir::IntegerType nativeIntegerType(mlir::Builder &builder,
                                    const CtypesLayout &layout) {
  return builder.getIntegerType(static_cast<unsigned>(layout.size * 8));
}

mlir::IntegerType
nativePointerIntegerType(mlir::Builder &builder,
                         const std::optional<TargetPlatformFacts> &facts) {
  unsigned width = facts ? static_cast<unsigned>(facts->pointerWidth) : 64;
  return builder.getIntegerType(width);
}

mlir::LLVM::LLVMPointerType nativePointerType(mlir::MLIRContext *context) {
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  return mlir::LLVM::LLVMPointerType::get(context);
}

mlir::Value coerceNativeInteger(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value,
                                mlir::IntegerType targetType) {
  auto sourceType = mlir::cast<mlir::IntegerType>(value.getType());
  if (sourceType == targetType)
    return value;
  if (sourceType.getWidth() > targetType.getWidth())
    return builder.create<mlir::arith::TruncIOp>(loc, targetType, value)
        .getResult();
  return builder.create<mlir::arith::ExtSIOp>(loc, targetType, value)
      .getResult();
}

mlir::Value
loadNativeIntegerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value address, mlir::IntegerType nativeType,
                             const std::optional<TargetPlatformFacts> &facts);

std::optional<mlir::Value> extractNativeIntegerArgument(
    mlir::Operation *op, mlir::OpBuilder &builder, const RuntimeBundle &source,
    llvm::StringRef expectedContract, const CtypesLayout &layout,
    const std::optional<TargetPlatformFacts> &facts) {
  if (!isIntegerScalarLayout(layout) && !isPointerScalarLayout(layout))
    return std::nullopt;
  mlir::IntegerType nativeType = nativeIntegerType(builder, layout);
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      stripCtypesModule(source.ctypes->ctypeName) ==
          stripCtypesModule(expectedContract) &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    return loadNativeIntegerFromAddress(builder, op->getLoc(),
                                        source.ctypes->storageAddressValue,
                                        nativeType, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      stripCtypesModule(source.ctypes->ctypeName) ==
          stripCtypesModule(expectedContract) &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid)) {
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->scalarValue, nativeType);
  }
  if (layout.kind == CtypesLayout::ABIKind::SignedInteger && layout.size == 8 &&
      source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid)) {
    return source.primitiveI64->value;
  }
  if (isPointerScalarLayout(layout) && source.primitiveI64 &&
      source.primitiveI64->value && source.primitiveI64->valid &&
      isKnownTrue(source.primitiveI64->valid)) {
    return coerceNativeInteger(builder, op->getLoc(),
                               source.primitiveI64->value, nativeType);
  }
  if (source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid)) {
    std::optional<std::int64_t> constant =
        knownI64Constant(source.primitiveI64->value);
    if (constant && fitsStaticIntegerLayout(*constant, layout))
      return coerceNativeInteger(builder, op->getLoc(),
                                 source.primitiveI64->value, nativeType);
  }
  return std::nullopt;
}

mlir::Value
integerToNativePointer(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value value,
                       const std::optional<TargetPlatformFacts> &facts) {
  mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
  mlir::Value raw = coerceNativeInteger(builder, loc, value, pointerInteger);
  return builder.create<mlir::LLVM::IntToPtrOp>(
      loc, nativePointerType(builder.getContext()), raw);
}

mlir::Value nativePointerToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value pointer) {
  return builder.create<mlir::LLVM::PtrToIntOp>(loc, builder.getI64Type(),
                                                pointer);
}

mlir::Value addressWithOffset(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value address, std::int64_t offset,
                              const std::optional<TargetPlatformFacts> &facts) {
  mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
  mlir::Value base = coerceNativeInteger(builder, loc, address, pointerInteger);
  if (offset == 0)
    return base;
  mlir::Value delta = builder.create<mlir::arith::ConstantIntOp>(
      loc, offset, pointerInteger.getWidth());
  return builder.create<mlir::arith::AddIOp>(loc, base, delta).getResult();
}

mlir::Value
nativePointerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value address,
                         const std::optional<TargetPlatformFacts> &facts) {
  return integerToNativePointer(builder, loc, address, facts);
}

mlir::Value
addressOfNativeCellAlloca(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value nativeValue,
                          const std::optional<TargetPlatformFacts> &facts) {
  auto nativeType = mlir::cast<mlir::IntegerType>(nativeValue.getType());
  auto bufferType = mlir::MemRefType::get({1}, nativeType);
  mlir::Value buffer = builder.create<mlir::memref::AllocaOp>(loc, bufferType);
  mlir::Value zero = constantIndex(builder, loc, 0);
  builder.create<mlir::memref::StoreOp>(loc, nativeValue, buffer,
                                        mlir::ValueRange{zero});
  mlir::Value pointerIndex =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
  return builder
      .create<mlir::arith::IndexCastOp>(
          loc, nativePointerIntegerType(builder, facts), pointerIndex)
      .getResult();
}

mlir::Value addressOfZeroedNativeBytesAlloca(
    mlir::OpBuilder &builder, mlir::Location loc, std::uint64_t size,
    const std::optional<TargetPlatformFacts> &facts) {
  std::uint64_t allocationSize = std::max<std::uint64_t>(size, 1);
  auto byteType = builder.getIntegerType(8);
  auto bufferType = mlir::MemRefType::get(
      {static_cast<std::int64_t>(allocationSize)}, byteType);
  mlir::Value buffer = builder.create<mlir::memref::AllocaOp>(loc, bufferType);
  mlir::Value lower = constantIndex(builder, loc, 0);
  mlir::Value upper =
      constantIndex(builder, loc, static_cast<std::int64_t>(allocationSize));
  mlir::Value step = constantIndex(builder, loc, 1);
  mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
  auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    builder.create<mlir::memref::StoreOp>(
        loc, zero, buffer, mlir::ValueRange{loop.getInductionVar()});
  }
  mlir::Value pointerIndex =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
  return builder
      .create<mlir::arith::IndexCastOp>(
          loc, nativePointerIntegerType(builder, facts), pointerIndex)
      .getResult();
}

mlir::Value
loadNativeIntegerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value address, mlir::IntegerType nativeType,
                             const std::optional<TargetPlatformFacts> &facts) {
  mlir::Value pointer = nativePointerFromAddress(builder, loc, address, facts);
  return builder.create<mlir::LLVM::LoadOp>(loc, nativeType, pointer)
      .getResult();
}

void storeNativeIntegerToAddress(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value address,
    mlir::Value value, mlir::IntegerType nativeType,
    const std::optional<TargetPlatformFacts> &facts) {
  mlir::Value pointer = nativePointerFromAddress(builder, loc, address, facts);
  mlir::Value nativeValue =
      coerceNativeInteger(builder, loc, value, nativeType);
  builder.create<mlir::LLVM::StoreOp>(loc, nativeValue, pointer);
}

mlir::Value widenNativeInteger(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value value, const CtypesLayout &layout) {
  auto sourceType = mlir::cast<mlir::IntegerType>(value.getType());
  mlir::IntegerType i64 = builder.getI64Type();
  if (sourceType == i64)
    return value;
  if (sourceType.getWidth() > i64.getWidth())
    return builder.create<mlir::arith::TruncIOp>(loc, i64, value).getResult();
  if (layout.kind == CtypesLayout::ABIKind::UnsignedInteger ||
      layout.kind == CtypesLayout::ABIKind::Pointer)
    return builder.create<mlir::arith::ExtUIOp>(loc, i64, value).getResult();
  return builder.create<mlir::arith::ExtSIOp>(loc, i64, value).getResult();
}

std::string describeNativeArgumentSource(const RuntimeBundle &source);

void copyNativeBytes(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value destinationAddress, mlir::Value sourceAddress,
                     std::uint64_t byteCount,
                     const std::optional<TargetPlatformFacts> &facts) {
  mlir::Type byteType = builder.getIntegerType(8);
  for (std::uint64_t offset = 0; offset < byteCount; ++offset) {
    mlir::Value sourceByteAddress = addressWithOffset(
        builder, loc, sourceAddress, static_cast<std::int64_t>(offset), facts);
    mlir::Value destinationByteAddress =
        addressWithOffset(builder, loc, destinationAddress,
                          static_cast<std::int64_t>(offset), facts);
    mlir::Value sourcePointer =
        nativePointerFromAddress(builder, loc, sourceByteAddress, facts);
    mlir::Value destinationPointer =
        nativePointerFromAddress(builder, loc, destinationByteAddress, facts);
    mlir::Value byte =
        builder.create<mlir::LLVM::LoadOp>(loc, byteType, sourcePointer)
            .getResult();
    builder.create<mlir::LLVM::StoreOp>(loc, byte, destinationPointer);
  }
}

mlir::LogicalResult storeCtypesValueToAddress(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Value destinationAddress, mlir::Type expectedType,
    llvm::StringRef expectedContract, const CtypesLayout &layout,
    const RuntimeBundle &source,
    const std::optional<TargetPlatformFacts> &facts) {
  if (isIntegerScalarLayout(layout) || isPointerScalarLayout(layout)) {
    std::optional<mlir::Value> value = extractNativeIntegerArgument(
        op, builder, source, expectedContract, layout, facts);
    if (!value)
      return op->emitError() << "ctypes value for " << expectedContract
                             << " has no compatible scalar evidence ("
                             << describeNativeArgumentSource(source) << ")";
    storeNativeIntegerToAddress(builder, op->getLoc(), destinationAddress,
                                *value, nativeIntegerType(builder, layout),
                                facts);
    return mlir::success();
  }

  if (layout.kind != CtypesLayout::ABIKind::Aggregate)
    return op->emitError() << "ctypes value for " << expectedContract
                           << " has unsupported ABI layout";
  if (!source.ctypes ||
      source.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return op->emitError() << "ctypes aggregate value for " << expectedContract
                           << " requires _CData cell evidence";
  if (expectedType && source.ctypes->ctype &&
      expectedType != source.ctypes->ctype)
    return op->emitError() << "ctypes aggregate assignment expects "
                           << expectedType << " but got "
                           << source.ctypes->ctype;

  mlir::Value sourceAddress = cdataStorageAddress(*source.ctypes);
  mlir::Value sourceValid = cdataStorageAddressValid(*source.ctypes);
  if (!sourceAddress || !sourceValid || !isKnownTrue(sourceValid))
    return op->emitError() << "ctypes aggregate value for " << expectedContract
                           << " has no materialized storage address";

  std::optional<CtypesLayout> sourceLayout = ctypesStaticLayoutForType(
      module,
      source.ctypes->ctype
          ? source.ctypes->ctype
          : ctypesContractType(op->getContext(), source.ctypes->ctypeName),
      facts);
  if (!sourceLayout)
    sourceLayout = ctypesStaticLayout(module, source.ctypes->ctypeName, facts);
  if (!sourceLayout || sourceLayout->size != layout.size)
    return op->emitError() << "ctypes aggregate value for " << expectedContract
                           << " has incompatible storage size";

  copyNativeBytes(builder, op->getLoc(), destinationAddress, sourceAddress,
                  layout.size, facts);
  return mlir::success();
}

RuntimeBufferEvidence
makeCtypesBufferEvidence(mlir::OpBuilder &builder, mlir::Location loc,
                         const CtypesLayout &layout, mlir::Value address,
                         mlir::Value valid, const RuntimeBundle &owner,
                         bool writable) {
  RuntimeBufferEvidence buffer;
  buffer.addressValue = address;
  buffer.addressValid = valid;
  buffer.byteLength =
      constantI64(builder, loc, static_cast<std::int64_t>(layout.size));
  buffer.byteLengthValid = constantI1(builder, loc, true);
  buffer.readable = true;
  buffer.writable = writable;
  buffer.cContiguous = true;
  keepAliveBufferSource(buffer, owner);
  return buffer;
}

void attachCtypesBufferEvidence(mlir::OpBuilder &builder, mlir::Location loc,
                                RuntimeBundle &bundle,
                                RuntimeCtypesEvidence &ctypes,
                                const CtypesLayout &layout, bool writable) {
  mlir::Value storageAddress = cdataStorageAddress(ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return;
  bundle.buffer = makeCtypesBufferEvidence(builder, loc, layout, storageAddress,
                                           storageValid, bundle, writable);
}

mlir::FailureOr<RuntimeBundle> materializeCtypesAddressView(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Type ctype, llvm::StringRef ctypeName, const CtypesLayout &layout,
    mlir::Value storageAddress, mlir::Value storageValid,
    const RuntimeBundle &owner) {
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op->emitError() << "ctypes view for " << ctypeName
                           << " requires a statically valid storage address";

  RuntimeBundle result = RuntimeBundle::object(ctype, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  evidence.provenance = RuntimeCtypesEvidence::Provenance::BufferView;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
  evidence.ctypeName = ctypeName.str();
  evidence.ctype = ctype;
  evidence.storageAddressValue = storageAddress;
  evidence.storageAddressValid = storageValid;
  evidence.addressValue = storageAddress;
  evidence.addressValid = storageValid;
  evidence.materializedObject = true;
  keepAliveSource(evidence, owner);

  if (isIntegerScalarLayout(layout) || isPointerScalarLayout(layout)) {
    mlir::Value nativeValue = loadNativeIntegerFromAddress(
        builder, op->getLoc(), storageAddress,
        nativeIntegerType(builder, layout), targetPlatformFacts(module));
    evidence.scalarValue =
        widenNativeInteger(builder, op->getLoc(), nativeValue, layout);
    evidence.scalarValid = constantI1(builder, op->getLoc(), true);
  }

  attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence, layout,
                             /*writable=*/true);
  result.ctypes = std::move(evidence);
  return result;
}

mlir::FailureOr<RuntimeBundle> materializeCtypesPythonReadResult(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Type ctype, llvm::StringRef ctypeName, const CtypesLayout &layout,
    mlir::Value storageAddress, mlir::Value storageValid,
    const RuntimeBundle &owner) {
  if (isIntegerScalarLayout(layout) || isPointerScalarLayout(layout)) {
    if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
      return op->emitError() << "ctypes scalar read for " << ctypeName
                             << " requires a statically valid storage address";
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    mlir::Value nativeValue =
        loadNativeIntegerFromAddress(builder, op->getLoc(), storageAddress,
                                     nativeIntegerType(builder, layout), facts);
    RuntimeBundle result = RuntimeBundle::object(
        runtimeContractType(op->getContext(), "builtins.int"), {});
    result.primitiveI64 = RuntimePrimitiveI64Evidence{
        widenNativeInteger(builder, op->getLoc(), nativeValue, layout),
        constantI1(builder, op->getLoc(), true)};
    return result;
  }

  return materializeCtypesAddressView(op, builder, module, ctype, ctypeName,
                                      layout, storageAddress, storageValid,
                                      owner);
}

std::string describeNativeArgumentSource(const RuntimeBundle &source);

mlir::FailureOr<RuntimeBundle>
materializeCtypesCell(mlir::Operation *op, mlir::OpBuilder &builder,
                      mlir::ModuleOp module, mlir::Type ctype,
                      llvm::StringRef ctypeName,
                      llvm::ArrayRef<const RuntimeBundle *> sources) {
  RuntimeBundle result = RuntimeBundle::object(ctype, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  evidence.provenance = RuntimeCtypesEvidence::Provenance::NativeCell;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
  evidence.ctypeName = ctypeName.str();
  evidence.ctype = ctype;
  evidence.ownsNativeStorage = true;

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesArrayType> array = ctypesArrayType(module, ctype, facts);
  if (array) {
    if (!facts)
      return op->emitError() << evidence.ctypeName
                             << " requires TargetPlatformFacts before lowering";
    if (sources.size() > array->count)
      return op->emitError() << evidence.ctypeName << " initializer received "
                             << sources.size() << " positional values for "
                             << array->count << " ctypes array elements";

    builder.setInsertionPoint(op);
    mlir::Value storageAddress = addressOfZeroedNativeBytesAlloca(
        builder, op->getLoc(), array->layout.size, facts);
    mlir::Value valid = constantI1(builder, op->getLoc(), true);
    evidence.addressValue = storageAddress;
    evidence.addressValid = valid;
    evidence.storageAddressValue = storageAddress;
    evidence.storageAddressValid = valid;

    for (auto [index, source] : llvm::enumerate(sources)) {
      if (!source)
        return op->emitError() << "ctypes array initializer argument " << index
                               << " has no evidence";
      mlir::Value elementAddress = addressWithOffset(
          builder, op->getLoc(), storageAddress,
          static_cast<std::int64_t>(index * array->elementLayout.size), facts);
      if (mlir::failed(storeCtypesValueToAddress(
              op, builder, module, elementAddress, array->elementType,
              array->elementContract, array->elementLayout, *source, facts)))
        return mlir::failure();
    }

    attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence,
                               array->layout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    return result;
  }

  py::ClassOp aggregateClass = lookupClassForContract(module, ctypeName);
  std::optional<CtypesAggregateLayout> aggregateLayout;
  if (aggregateClass)
    aggregateLayout = ctypesAggregateLayout(module, aggregateClass, facts, 0);
  if (aggregateLayout) {
    if (!facts)
      return op->emitError() << evidence.ctypeName
                             << " requires TargetPlatformFacts before lowering";
    if (sources.size() > aggregateLayout->fields.size())
      return op->emitError()
             << evidence.ctypeName << " initializer received " << sources.size()
             << " positional field values for "
             << aggregateLayout->fields.size() << " ctypes fields";

    builder.setInsertionPoint(op);
    mlir::Value storageAddress = addressOfZeroedNativeBytesAlloca(
        builder, op->getLoc(), aggregateLayout->layout.size, facts);
    mlir::Value valid = constantI1(builder, op->getLoc(), true);
    evidence.addressValue = storageAddress;
    evidence.addressValid = valid;
    evidence.storageAddressValue = storageAddress;
    evidence.storageAddressValid = valid;
    evidence.ownsNativeStorage = true;

    for (auto [index, source] : llvm::enumerate(sources)) {
      const CtypesFieldLayout &field = aggregateLayout->fields[index];
      if (!source)
        return op->emitError() << "ctypes aggregate initializer argument "
                               << index << " has no evidence";
      mlir::Value fieldAddress =
          addressWithOffset(builder, op->getLoc(), storageAddress,
                            static_cast<std::int64_t>(field.offset), facts);
      if (mlir::failed(storeCtypesValueToAddress(
              op, builder, module, fieldAddress, field.type, field.contract,
              field.layout, *source, facts)))
        return mlir::failure();
    }

    attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence,
                               aggregateLayout->layout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    return result;
  }

  if (!isCtypesIntegralLike(evidence.ctypeName))
    return op->emitError() << evidence.ctypeName
                           << " erased initializer is not implemented yet";
  if (sources.size() > 1)
    return op->emitError()
           << evidence.ctypeName
           << " erased initializer supports at most one value argument";

  builder.setInsertionPoint(op);
  if (sources.empty()) {
    evidence.scalarValue = constantI64(builder, op->getLoc(), 0);
    evidence.scalarValid = constantI1(builder, op->getLoc(), true);
  } else {
    const RuntimeBundle *source = sources.front();
    if (!source)
      return op->emitError() << "ctypes initializer argument has no evidence";
    if (source->primitiveI64) {
      evidence.scalarValue = source->primitiveI64->value;
      evidence.scalarValid = source->primitiveI64->valid;
    } else if (isCtypesVoidPointer(evidence.ctypeName) &&
               isNoneBundle(*source)) {
      evidence.scalarValue = constantI64(builder, op->getLoc(), 0);
      evidence.scalarValid = constantI1(builder, op->getLoc(), true);
    } else if (source->ctypes && source->ctypes->scalarValue &&
               source->ctypes->scalarValid) {
      evidence.scalarValue = source->ctypes->scalarValue;
      evidence.scalarValid = source->ctypes->scalarValid;
    } else {
      return op->emitError()
             << evidence.ctypeName
             << " erased initializer requires primitive integer evidence";
    }
  }

  std::optional<CtypesLayout> layout =
      ctypesStaticLayoutForType(module, ctype, facts);
  if (!layout)
    layout = ctypesStaticLayout(module, evidence.ctypeName, facts);
  if (layout && evidence.scalarValue && evidence.scalarValid &&
      isKnownTrue(evidence.scalarValid) &&
      (isIntegerScalarLayout(*layout) || isPointerScalarLayout(*layout))) {
    mlir::IntegerType nativeType = nativeIntegerType(builder, *layout);
    mlir::Value nativeValue = coerceNativeInteger(
        builder, op->getLoc(), evidence.scalarValue, nativeType);
    evidence.addressValue =
        addressOfNativeCellAlloca(builder, op->getLoc(), nativeValue, facts);
    evidence.addressValid = constantI1(builder, op->getLoc(), true);
    evidence.storageAddressValue = evidence.addressValue;
    evidence.storageAddressValid = evidence.addressValid;
    attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence, *layout,
                               /*writable=*/true);
  }

  result.ctypes = std::move(evidence);
  return result;
}

mlir::FailureOr<RuntimeBundle>
materializeCtypesLibrary(mlir::Operation *op, mlir::ModuleOp module,
                         mlir::Type ctype, llvm::StringRef ctypeName,
                         llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() > 1)
    return op->emitError()
           << ctypeName << " static initializer supports at most one library "
           << "name";

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  if (!facts)
    return op->emitError() << ctypeName
                           << " requires TargetPlatformFacts before lowering";
  llvm::Triple triple(facts->triple);
  if (ctypeName == "ctypes.WinDLL" && !triple.isOSWindows())
    return op->emitError()
           << "ctypes.WinDLL is only supported for Windows targets, got "
           << facts->triple;

  RuntimeBundle result = RuntimeBundle::object(ctype, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Library;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = ctypeName.str();
  evidence.ctype = ctype;
  evidence.abi = ctypesLibraryABI(ctypeName);
  if (sources.empty() || isNoneBundle(*sources.front())) {
    evidence.processLibrary = true;
    evidence.libraryName.clear();
  } else if (sources.front() && sources.front()->literalText) {
    evidence.processLibrary = false;
    evidence.libraryName = *sources.front()->literalText;
  } else {
    return op->emitError()
           << ctypeName
           << " requires a literal library name or None on the static path";
  }
  result.ctypes = std::move(evidence);
  return result;
}

std::optional<mlir::Value>
stackPointerForBorrowedScalar(mlir::Operation *op, mlir::OpBuilder &builder,
                              const RuntimeCtypesEvidence &evidence,
                              const std::optional<TargetPlatformFacts> &facts) {
  if (!evidence.callRegionBorrow || evidence.pointeeType.empty() ||
      !evidence.pointeeScalarValue || !evidence.pointeeScalarValid ||
      !isKnownTrue(evidence.pointeeScalarValid))
    return std::nullopt;
  std::optional<CtypesLayout> pointeeLayout =
      ctypesLayout(evidence.pointeeType, facts);
  if (!pointeeLayout || !isIntegerScalarLayout(*pointeeLayout))
    return std::nullopt;

  mlir::Location loc = op->getLoc();
  mlir::IntegerType nativeType = nativeIntegerType(builder, *pointeeLayout);
  mlir::Value nativeValue = coerceNativeInteger(
      builder, loc, evidence.pointeeScalarValue, nativeType);
  auto bufferType = mlir::MemRefType::get({1}, nativeType);
  mlir::Value buffer = builder.create<mlir::memref::AllocaOp>(loc, bufferType);
  mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  builder.create<mlir::memref::StoreOp>(loc, nativeValue, buffer,
                                        mlir::ValueRange{zero});
  mlir::Value pointerIndex =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
  mlir::Value pointerInteger = builder.create<mlir::arith::IndexCastOp>(
      loc, nativePointerIntegerType(builder, facts), pointerIndex);
  return builder.create<mlir::LLVM::IntToPtrOp>(
      loc, nativePointerType(builder.getContext()), pointerInteger);
}

std::optional<mlir::Value>
extractNativePointerArgument(mlir::Operation *op, mlir::OpBuilder &builder,
                             const RuntimeBundle &source,
                             const std::optional<TargetPlatformFacts> &facts) {
  if (isNoneBundle(source)) {
    mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
        op->getLoc(), 0, nativePointerIntegerType(builder, facts));
    return builder.create<mlir::LLVM::IntToPtrOp>(
        op->getLoc(), nativePointerType(builder.getContext()), zero);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer) {
    if (source.ctypes->storageAddressValue &&
        source.ctypes->storageAddressValid &&
        isKnownTrue(source.ctypes->storageAddressValid)) {
      mlir::IntegerType pointerInteger =
          nativePointerIntegerType(builder, facts);
      mlir::Value loaded = loadNativeIntegerFromAddress(
          builder, op->getLoc(), source.ctypes->storageAddressValue,
          pointerInteger, facts);
      return integerToNativePointer(builder, op->getLoc(), loaded, facts);
    }
    if (source.ctypes->addressValue && source.ctypes->addressValid &&
        isKnownTrue(source.ctypes->addressValid))
      return integerToNativePointer(builder, op->getLoc(),
                                    source.ctypes->addressValue, facts);
    return stackPointerForBorrowedScalar(op, builder, *source.ctypes, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
    mlir::Value loaded = loadNativeIntegerFromAddress(
        builder, op->getLoc(), source.ctypes->storageAddressValue,
        pointerInteger, facts);
    return integerToNativePointer(builder, op->getLoc(), loaded, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid))
    return integerToNativePointer(builder, op->getLoc(),
                                  source.ctypes->scalarValue, facts);
  if (source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid))
    return integerToNativePointer(builder, op->getLoc(),
                                  source.primitiveI64->value, facts);
  return std::nullopt;
}

std::optional<mlir::Value>
extractPointerAddressInteger(mlir::Operation *op, mlir::OpBuilder &builder,
                             const RuntimeBundle &source,
                             const std::optional<TargetPlatformFacts> &facts) {
  if (isNoneBundle(source))
    return constantI64(builder, op->getLoc(), 0);
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
    return loadNativeIntegerFromAddress(builder, op->getLoc(),
                                        source.ctypes->storageAddressValue,
                                        pointerInteger, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->scalarValue,
                               nativePointerIntegerType(builder, facts));
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
    return loadNativeIntegerFromAddress(builder, op->getLoc(),
                                        source.ctypes->storageAddressValue,
                                        pointerInteger, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer &&
      source.ctypes->addressValue && source.ctypes->addressValid &&
      isKnownTrue(source.ctypes->addressValid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->addressValue,
                               nativePointerIntegerType(builder, facts));
  if (source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.primitiveI64->value,
                               nativePointerIntegerType(builder, facts));
  return std::nullopt;
}

mlir::FailureOr<mlir::func::FuncOp>
getOrCreateNativeDeclaration(mlir::Operation *op, mlir::ModuleOp module,
                             mlir::OpBuilder &builder, llvm::StringRef name,
                             mlir::FunctionType type) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    if (existing.getFunctionType() != type)
      return op->emitError() << "native symbol '" << name
                             << "' was already declared with incompatible type";
    return existing;
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      builder.create<mlir::func::FuncOp>(module.getLoc(), name, type);
  function.setPrivate();
  return function;
}

std::string describeNativeArgumentSource(const RuntimeBundle &source) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "source contract=" << source.contractName();
  os << ", primitive_i64=" << (source.primitiveI64 ? "yes" : "no");
  if (source.primitiveI64 && source.primitiveI64->valid)
    os << ", primitive_valid_known_true="
       << (isKnownTrue(source.primitiveI64->valid) ? "yes" : "no");
  if (source.ctypes) {
    os << ", ctypes=" << source.ctypes->ctypeName;
    os << ", ctypes_kind=";
    switch (source.ctypes->kind) {
    case RuntimeCtypesEvidence::Kind::Module:
      os << "module";
      break;
    case RuntimeCtypesEvidence::Kind::Cell:
      os << "cell";
      break;
    case RuntimeCtypesEvidence::Kind::Pointer:
      os << "pointer";
      break;
    case RuntimeCtypesEvidence::Kind::Library:
      os << "library";
      break;
    case RuntimeCtypesEvidence::Kind::Symbol:
      os << "symbol";
      break;
    case RuntimeCtypesEvidence::Kind::FieldDescriptor:
      os << "field_descriptor";
      break;
    }
    os << ", provenance=" << ctypesProvenanceName(source.ctypes->provenance);
    os << ", lifetime=" << ctypesLifetimeName(source.ctypes->lifetime);
    os << ", scalar=" << (source.ctypes->scalarValue ? "yes" : "no");
    if (source.ctypes->scalarValid)
      os << ", scalar_valid_known_true="
         << (isKnownTrue(source.ctypes->scalarValid) ? "yes" : "no");
    os << ", storage_address="
       << (source.ctypes->storageAddressValue ? "yes" : "no");
    if (source.ctypes->storageAddressValid)
      os << ", storage_address_valid_known_true="
         << (isKnownTrue(source.ctypes->storageAddressValid) ? "yes" : "no");
    os << ", address=" << (source.ctypes->addressValue ? "yes" : "no");
    if (source.ctypes->addressValid)
      os << ", address_valid_known_true="
         << (isKnownTrue(source.ctypes->addressValid) ? "yes" : "no");
    os << ", keepalive_edges=" << source.ctypes->keepAlive.size();
  }
  return os.str();
}

} // namespace

bool RuntimeBundleLowerer::isStaticCtypesBinding(
    llvm::StringRef binding) const {
  if (RuntimeBundleLowerer::isStaticCtypesModuleBinding(binding) ||
      RuntimeBundleLowerer::isStaticCtypesCallable(binding))
    return true;
  return ctypesQualifiedNameContract(binding).has_value();
}

bool RuntimeBundleLowerer::isStaticCtypesModuleBinding(
    llvm::StringRef binding) const {
  return binding == "ctypes" || binding == "_ctypes" ||
         binding == "ctypes.wintypes";
}

bool RuntimeBundleLowerer::isStaticCtypesCallable(
    llvm::StringRef binding) const {
  if (ctypesFromAddressTarget(binding) || ctypesFromBufferTarget(binding) ||
      ctypesFromBufferCopyTarget(binding))
    return true;
  return llvm::StringSwitch<bool>(binding)
      .Cases("ctypes.sizeof", "ctypes.alignment", "ctypes.byref",
             "ctypes.pointer", "ctypes.POINTER", "ctypes.cast",
             "ctypes.addressof", true)
      .Default(false);
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesBindingRef(py::BindingRefOp op) {
  if (RuntimeBundleLowerer::isStaticCtypesModuleBinding(op.getBinding()))
    return RuntimeBundleLowerer::lowerStaticCtypesModuleBindingRef(op);
  if (RuntimeBundleLowerer::isStaticCtypesCallable(op.getBinding())) {
    valueBundles[op.getResult()] = RuntimeBundle::builtinCallable(
        op.getResult().getType(), op.getBinding());
    erase.push_back(op);
    return mlir::success();
  }
  if (std::optional<std::string> contract =
          ctypesQualifiedNameContract(op.getBinding())) {
    valueBundles[op.getResult()] = RuntimeBundle::typeObject(
        op.getResult().getType(), ctypesContractType(context, *contract));
    erase.push_back(op);
    return mlir::success();
  }
  return mlir::failure();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesModuleBindingRef(py::BindingRefOp op) {
  valueBundles[op.getResult()] =
      makeCtypesModuleBundle(op.getResult().getType(), op.getBinding());
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesModuleAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Module)
    return mlir::failure();
  llvm::StringRef moduleName = object.ctypes->ctypeName;
  if (moduleName == "ctypes" && op.getName() == "wintypes") {
    valueBundles[op.getResult()] =
        makeCtypesModuleBundle(op.getResult().getType(), "ctypes.wintypes");
    erase.push_back(op);
    return mlir::success();
  }
  if (std::optional<std::string> contract =
          ctypesModuleAttrContract(moduleName, op.getName())) {
    valueBundles[op.getResult()] = RuntimeBundle::typeObject(
        op.getResult().getType(), ctypesContractType(context, *contract));
    erase.push_back(op);
    return mlir::success();
  }
  if (moduleName == "ctypes" && isStaticCtypesFunctionName(op.getName())) {
    valueBundles[op.getResult()] = RuntimeBundle::builtinCallable(
        op.getResult().getType(),
        (llvm::Twine("ctypes.") + op.getName()).str());
    erase.push_back(op);
    return mlir::success();
  }
  return op.emitError() << "ctypes module '" << moduleName
                        << "' has no static attribute '" << op.getName() << "'";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesValueAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();
  if (!object.ctypes->scalarValue || !object.ctypes->scalarValid)
    return op.emitError() << "ctypes value attribute requires scalar evidence";
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, runtimeContractType(context, "builtins.int"),
          object.ctypes->scalarValue, object.ctypes->scalarValid, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesFieldDescriptorAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::FieldDescriptor)
    return mlir::failure();

  if (op.getName() == "offset" || op.getName() == "size") {
    std::int64_t value =
        op.getName() == "offset"
            ? static_cast<std::int64_t>(object.ctypes->fieldOffset)
            : static_cast<std::int64_t>(object.ctypes->fieldSize);
    RuntimeBundle result;
    builder.setInsertionPoint(op);
    if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"),
            constantI64(builder, op.getLoc(), value),
            constantI1(builder, op.getLoc(), true), result)))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (op.getName() == "name") {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
            op, object.ctypes->fieldName, result)))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  return op.emitError() << "ctypes CField descriptor has no static attribute '"
                        << op.getName() << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesTypeFieldDescriptorGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (object.kind != RuntimeBundle::Kind::TypeObject)
    return mlir::failure();
  std::string contract = object.instanceContractName();
  py::ClassOp classOp = lookupClassForContract(module, contract);
  if (!classOp)
    return mlir::failure();
  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesAggregateLayout> aggregate =
      ctypesAggregateLayout(module, classOp, facts, 0);
  if (!aggregate)
    return mlir::failure();

  auto field =
      llvm::find_if(aggregate->fields, [&](const CtypesFieldLayout &it) {
        return it.name == op.getName();
      });
  if (field == aggregate->fields.end())
    return mlir::failure();

  RuntimeBundle result =
      RuntimeBundle::object(ctypesContractType(context, "_ctypes.CField"), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::FieldDescriptor;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = "_ctypes.CField";
  evidence.ctype = ctypesContractType(context, "_ctypes.CField");
  evidence.fieldName = field->name;
  evidence.fieldType = field->contract;
  evidence.fieldOffset = field->offset;
  evidence.fieldSize = field->layout.size;
  result.ctypes = std::move(evidence);
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesFieldAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();

  py::ClassOp classOp =
      lookupClassForContract(module, object.ctypes->ctypeName);
  if (!classOp)
    return mlir::failure();
  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesAggregateLayout> aggregate =
      ctypesAggregateLayout(module, classOp, facts, 0);
  if (!aggregate)
    return mlir::failure();

  auto field =
      llvm::find_if(aggregate->fields, [&](const CtypesFieldLayout &it) {
        return it.name == op.getName();
      });
  if (field == aggregate->fields.end())
    return mlir::failure();

  mlir::Value storageAddress = cdataStorageAddress(*object.ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(*object.ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op.emitError() << "ctypes field '" << op.getName()
                          << "' requires materialized _CData storage";

  builder.setInsertionPoint(op);
  mlir::Value fieldAddress =
      addressWithOffset(builder, op.getLoc(), storageAddress,
                        static_cast<std::int64_t>(field->offset), facts);
  mlir::FailureOr<RuntimeBundle> result = materializeCtypesPythonReadResult(
      op, builder, module, field->type, field->contract, field->layout,
      fieldAddress, storageValid, object);
  if (mlir::failed(result))
    return mlir::failure();
  if ((*result).ctypes) {
    (*result).fieldAliasOwner = op.getObject();
    (*result).fieldAliasName = op.getName().str();
  }
  valueBundles[op.getResult()] = std::move(*result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesFieldAttrSet(
    py::AttrSetOp op, const RuntimeBundle &object, const RuntimeBundle *value) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();

  py::ClassOp classOp =
      lookupClassForContract(module, object.ctypes->ctypeName);
  if (!classOp)
    return mlir::failure();
  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesAggregateLayout> aggregate =
      ctypesAggregateLayout(module, classOp, facts, 0);
  if (!aggregate)
    return mlir::failure();

  auto field =
      llvm::find_if(aggregate->fields, [&](const CtypesFieldLayout &it) {
        return it.name == op.getName();
      });
  if (field == aggregate->fields.end())
    return mlir::failure();
  if (!value)
    return op.emitError() << "ctypes field assignment has no value evidence";

  mlir::Value storageAddress = cdataStorageAddress(*object.ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(*object.ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op.emitError() << "ctypes field '" << op.getName()
                          << "' requires materialized _CData storage";

  builder.setInsertionPoint(op);
  mlir::Value fieldAddress =
      addressWithOffset(builder, op.getLoc(), storageAddress,
                        static_cast<std::int64_t>(field->offset), facts);
  if (mlir::failed(storeCtypesValueToAddress(op, builder, module, fieldAddress,
                                             field->type, field->contract,
                                             field->layout, *value, facts)))
    return mlir::failure();

  valueBundles[op.getObject()] = object;
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesGetItem(py::GetItemOp op,
                                               const RuntimeBundle &container,
                                               const RuntimeBundle &index) {
  if (!container.ctypes ||
      container.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  mlir::Type containerType =
      container.ctypes->ctype ? container.ctypes->ctype : container.contract;
  std::optional<CtypesArrayType> array =
      ctypesArrayType(module, containerType, facts);
  if (!array)
    return mlir::failure();
  if (!index.primitiveI64 || !index.primitiveI64->value ||
      !index.primitiveI64->valid || !isKnownTrue(index.primitiveI64->valid))
    return op.emitError()
           << "ctypes array indexing requires primitive integer evidence";
  std::optional<std::int64_t> rawIndex =
      knownI64Constant(index.primitiveI64->value);
  if (!rawIndex)
    return op.emitError()
           << "ctypes array indexing currently requires a statically known "
              "integer index";

  std::int64_t normalized = *rawIndex;
  std::int64_t count = static_cast<std::int64_t>(array->count);
  if (normalized < 0)
    normalized += count;
  if (normalized < 0 || normalized >= count) {
    builder.setInsertionPoint(op);
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            op, "builtins.IndexError", "ctypes array index out of range")))
      return mlir::failure();
    mlir::FailureOr<RuntimeValue> dead =
        RuntimeBundleLowerer::materializeDeadObjectValue(
            op, op.getResult().getType(), "ctypes array index miss");
    if (mlir::failed(dead))
      return mlir::failure();
    valueBundles[op.getResult()] =
        RuntimeBundle::object(dead->contract, dead->values);
    erase.push_back(op);
    return mlir::success();
  }

  mlir::Value storageAddress = cdataStorageAddress(*container.ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(*container.ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op.emitError()
           << "ctypes array indexing requires materialized _CData storage";

  builder.setInsertionPoint(op);
  mlir::Value elementAddress = addressWithOffset(
      builder, op.getLoc(), storageAddress,
      static_cast<std::int64_t>(static_cast<std::uint64_t>(normalized) *
                                array->elementLayout.size),
      facts);
  mlir::FailureOr<RuntimeBundle> result = materializeCtypesPythonReadResult(
      op, builder, module, array->elementType, array->elementContract,
      array->elementLayout, elementAddress, storageValid, container);
  if (mlir::failed(result))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(*result);
  erase.push_back(op);
  return mlir::success();
}

bool RuntimeBundleLowerer::isErasedCtypesContract(
    llvm::StringRef contract) const {
  return isFixedOrTargetDependentCtypesScalar(contract);
}

bool RuntimeBundleLowerer::isStaticCtypesLibraryContract(
    llvm::StringRef contract) const {
  return contract == "ctypes.CDLL" || contract == "ctypes.WinDLL";
}

mlir::LogicalResult
RuntimeBundleLowerer::bindErasedCtypesNew(py::NewOp op,
                                          llvm::StringRef contract) {
  bool scalar = RuntimeBundleLowerer::isErasedCtypesContract(contract);
  py::ClassOp classOp =
      scalar
          ? py::ClassOp{}
          : RuntimeBundleLowerer::classForContract(op.getInstance().getType());
  bool aggregate = false;
  if (classOp) {
    std::optional<std::string> kind = ctypesAggregateKind(module, classOp);
    aggregate = kind && (*kind == "struct" || *kind == "union");
  }
  if (!scalar && !aggregate)
    return mlir::failure();

  RuntimeBundle bundle = RuntimeBundle::object(op.getInstance().getType(), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  evidence.provenance = RuntimeCtypesEvidence::Provenance::NativeCell;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
  evidence.ctypeName = contract.str();
  evidence.ctype = op.getInstance().getType();
  evidence.ownsNativeStorage = true;
  bundle.ctypes = std::move(evidence);
  valueBundles[op.getInstance()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::bindStaticCtypesLibraryNew(py::NewOp op,
                                                 llvm::StringRef contract) {
  if (!RuntimeBundleLowerer::isStaticCtypesLibraryContract(contract))
    return mlir::failure();
  RuntimeBundle bundle = RuntimeBundle::object(op.getInstance().getType(), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Library;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = contract.str();
  evidence.ctype = op.getInstance().getType();
  evidence.abi = ctypesLibraryABI(contract);
  bundle.ctypes = std::move(evidence);
  valueBundles[op.getInstance()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerErasedCtypesInit(
    py::InitOp op, const RuntimeBundle &instance,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (!instance.ctypes ||
      instance.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();
  if (sources.empty() || sources.front() != &instance)
    return op.emitError() << "ctypes initializer source evidence mismatch";

  mlir::FailureOr<RuntimeBundle> updated = materializeCtypesCell(
      op, builder, module,
      instance.ctypes->ctype ? instance.ctypes->ctype
                             : op.getInstance().getType(),
      instance.ctypes->ctypeName,
      llvm::ArrayRef<const RuntimeBundle *>(sources).drop_front());
  if (mlir::failed(updated))
    return mlir::failure();
  valueBundles[op.getInstance()] = std::move(*updated);
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesLibraryInit(
    py::InitOp op, const RuntimeBundle &instance,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (!instance.ctypes ||
      instance.ctypes->kind != RuntimeCtypesEvidence::Kind::Library)
    return mlir::failure();
  if (sources.empty() || sources.front() != &instance)
    return op.emitError() << "ctypes library initializer source evidence "
                          << "mismatch";

  mlir::FailureOr<RuntimeBundle> updated = materializeCtypesLibrary(
      op, module,
      instance.ctypes->ctype ? instance.ctypes->ctype
                             : op.getInstance().getType(),
      instance.ctypes->ctypeName, sources.drop_front());
  if (mlir::failed(updated))
    return mlir::failure();
  valueBundles[op.getInstance()] = std::move(*updated);
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesTypeObjectCall(
    py::CallOp op, const RuntimeBundle &callable) {
  if (op.getNumResults() != 1)
    return op.emitError() << "ctypes type object call expects one result";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  std::optional<std::string> contract = ctypesTypeObjectName(callable);
  if (!contract)
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "ctypes constructor arguments",
                                              sources, &unpackedSources)))
    return mlir::failure();

  mlir::Type ctype = callable.instanceContract
                         ? callable.instanceContract
                         : ctypesContractType(context, *contract);
  mlir::FailureOr<RuntimeBundle> result =
      RuntimeBundleLowerer::isStaticCtypesLibraryContract(*contract)
          ? materializeCtypesLibrary(op, module, ctype, *contract, sources)
          : materializeCtypesCell(op, builder, module, ctype, *contract,
                                  sources);
  if (mlir::failed(result))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(*result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesModuleCall(
    py::CallOp op, const RuntimeBundle &receiver, llvm::StringRef methodName) {
  if (!receiver.ctypes ||
      receiver.ctypes->kind != RuntimeCtypesEvidence::Kind::Module)
    return mlir::failure();
  llvm::StringRef moduleName = receiver.ctypes->ctypeName;
  if (moduleName == "ctypes" && isStaticCtypesFunctionName(methodName)) {
    RuntimeBundle callable = RuntimeBundle::builtinCallable(
        op.getCallable().getType(),
        (llvm::Twine("ctypes.") + methodName).str());
    return RuntimeBundleLowerer::lowerStaticCtypesCall(op, callable);
  }
  if (std::optional<std::string> contract =
          ctypesModuleAttrContract(moduleName, methodName)) {
    RuntimeBundle typeObject = RuntimeBundle::typeObject(
        op.getCallable().getType(), ctypesContractType(context, *contract));
    return RuntimeBundleLowerer::lowerStaticCtypesTypeObjectCall(op,
                                                                 typeObject);
  }
  return op.emitError() << "ctypes module '" << moduleName
                        << "' has no static callable attribute '" << methodName
                        << "'";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesTypeObjectMethodCall(
    py::CallOp op, const RuntimeBundle &receiver, llvm::StringRef methodName) {
  std::optional<std::string> contract = ctypesTypeObjectName(receiver);
  if (!contract)
    return mlir::failure();
  if (methodName != "from_address" && methodName != "from_buffer" &&
      methodName != "from_buffer_copy")
    return mlir::failure();
  RuntimeBundle callable = RuntimeBundle::builtinCallable(
      op.getCallable().getType(),
      (llvm::Twine("ctypes.") + methodName + ":" + *contract).str());
  return RuntimeBundleLowerer::lowerStaticCtypesCall(op, callable);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesArrayTypeMul(
    mlir::Operation *op, const RuntimeBundle &lhs, const RuntimeBundle &rhs,
    mlir::Value resultValue) {
  const RuntimeBundle *typeObject = nullptr;
  const RuntimeBundle *countSource = nullptr;
  if (ctypesTypeObjectName(lhs) && rhs.primitiveI64) {
    typeObject = &lhs;
    countSource = &rhs;
  } else if (ctypesTypeObjectName(rhs) && lhs.primitiveI64) {
    typeObject = &rhs;
    countSource = &lhs;
  } else {
    return mlir::failure();
  }
  std::optional<std::int64_t> count =
      knownI64Constant(countSource->primitiveI64->value);
  if (!count || *count < 0)
    return mlir::failure();

  mlir::Type element = typeObject->instanceContract;
  if (!element)
    return mlir::failure();
  mlir::Type arrayType = py::ContractType::get(
      context, "_ctypes.Array",
      {element, py::LiteralType::get(context, std::to_string(*count))});
  valueBundles[resultValue] =
      RuntimeBundle::typeObject(resultValue.getType(), arrayType);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesAttrGet(py::AttrGetOp op,
                                               const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Library)
    return mlir::failure();

  auto existing = object.fieldBundles.find(op.getName());
  if (existing != object.fieldBundles.end()) {
    if (!existing->second)
      return op.emitError()
             << "ctypes symbol evidence for '" << op.getName() << "' is empty";
    RuntimeBundle result = *existing->second;
    result.fieldAliasOwner = op.getObject();
    result.fieldAliasName = op.getName().str();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  RuntimeBundle result = RuntimeBundle::object(op.getResult().getType(), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Symbol;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = "_ctypes.CFuncPtr";
  evidence.ctype = op.getResult().getType();
  evidence.libraryName = object.ctypes->libraryName;
  evidence.abi = object.ctypes->abi;
  evidence.processLibrary = object.ctypes->processLibrary;
  evidence.symbolName = op.getName().str();
  result.ctypes = std::move(evidence);
  result.fieldAliasOwner = op.getObject();
  result.fieldAliasName = op.getName().str();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesAttrSet(
    py::AttrSetOp op, const RuntimeBundle &object, const RuntimeBundle *value) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Symbol)
    return mlir::failure();
  if (!value)
    return op.emitError() << "ctypes symbol attribute value has no evidence";

  RuntimeBundle updated = object;
  RuntimeCtypesEvidence evidence = *object.ctypes;
  llvm::StringRef name = op.getName();
  if (name == "argtypes") {
    if (!isStaticSequenceBundle(*value))
      return op.emitError()
             << "ctypes argtypes must be a static list or tuple of ctypes "
             << "type objects";
    evidence.argTypes.clear();
    evidence.argTypes.reserve(value->sequenceElementBundles.size());
    for (auto [index, element] :
         llvm::enumerate(value->sequenceElementBundles)) {
      if (!element)
        return op.emitError()
               << "ctypes argtypes element " << index << " has no evidence";
      std::optional<std::string> ctype = ctypesTypeObjectName(*element);
      if (!ctype)
        return op.emitError() << "ctypes argtypes element " << index
                              << " must be a ctypes type object";
      evidence.argTypes.push_back(std::move(*ctype));
    }
  } else if (name == "restype") {
    if (isNoneBundle(*value)) {
      evidence.resultType = std::string("types.NoneType");
    } else {
      std::optional<std::string> ctype = ctypesTypeObjectName(*value);
      if (!ctype)
        return op.emitError()
               << "ctypes restype must be a ctypes type object or None";
      evidence.resultType = std::move(*ctype);
    }
  } else {
    return op.emitError() << "ctypes symbol attribute '" << name
                          << "' is not supported on the static path";
  }

  updated.ctypes = std::move(evidence);
  valueBundles[op.getObject()] = updated;
  if (updated.fieldAliasOwner && !updated.fieldAliasName.empty()) {
    auto owner = valueBundles.find(updated.fieldAliasOwner);
    if (owner != valueBundles.end()) {
      RuntimeBundle ownerBundle = owner->second;
      ownerBundle.fieldBundles[updated.fieldAliasName] =
          std::make_shared<RuntimeBundle>(updated);
      valueBundles[updated.fieldAliasOwner] = std::move(ownerBundle);
    }
  }
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesNativeCall(
    py::CallOp op, const RuntimeBundle &callable) {
  if (!callable.ctypes ||
      callable.ctypes->kind != RuntimeCtypesEvidence::Kind::Symbol)
    return mlir::failure();
  const RuntimeCtypesEvidence &evidence = *callable.ctypes;
  if (evidence.symbolName.empty())
    return op.emitError() << "ctypes native call has no symbol evidence";
  if (!evidence.resultType)
    return op.emitError() << "ctypes symbol '" << evidence.symbolName
                          << "' requires a static restype before calling";
  if (!evidence.processLibrary || !evidence.libraryName.empty())
    return op.emitError()
           << "ctypes native call lowering currently supports only "
              "ctypes.CDLL(None) process symbols";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "ctypes native arguments",
                                              sources, &unpackedSources)))
    return mlir::failure();
  if (sources.size() != evidence.argTypes.size())
    return op.emitError() << "ctypes symbol '" << evidence.symbolName
                          << "' expects " << evidence.argTypes.size()
                          << " arguments but got " << sources.size();

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  if (!facts)
    return op.emitError()
           << "ctypes native call requires TargetPlatformFacts before "
              "lowering";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Type, 4> nativeArgTypes;
  llvm::SmallVector<mlir::Value, 4> nativeArgs;
  for (auto [index, source] : llvm::enumerate(sources)) {
    if (!source)
      return op.emitError()
             << "ctypes native argument " << index << " has no evidence";
    llvm::StringRef argType = evidence.argTypes[index];
    std::optional<CtypesLayout> layout =
        ctypesStaticLayout(module, argType, facts);
    if (!layout)
      return op.emitError()
             << "ctypes native argument " << index << " type " << argType
             << " has no ABI layout for " << targetFactsLabel(facts);
    if (isIntegerScalarLayout(*layout)) {
      mlir::IntegerType nativeType = nativeIntegerType(builder, *layout);
      std::optional<mlir::Value> nativeValue = extractNativeIntegerArgument(
          op, builder, *source, argType, *layout, facts);
      if (!nativeValue)
        return op.emitError()
               << "ctypes native argument " << index << " for " << argType
               << " requires an exact ctypes scalar cell, a same-width signed "
                  "primitive integer, or a statically range-proven primitive "
                  "integer ("
               << describeNativeArgumentSource(*source) << ")";
      nativeArgTypes.push_back(nativeType);
      nativeArgs.push_back(*nativeValue);
      continue;
    }
    if (isFloatingScalarLayout(*layout)) {
      if (layout->size != 8)
        return op.emitError()
               << "ctypes native argument " << index << " type " << argType
               << " requires an explicit f32 precision conversion";
      std::optional<RuntimeSymbol> unbox =
          manifest.primitive(source->contractName(), "unbox.f64");
      if (!unbox)
        return op.emitError()
               << "ctypes native argument " << index << " for " << argType
               << " requires float-compatible source evidence ("
               << describeNativeArgumentSource(*source) << ")";
      mlir::func::CallOp unboxCall =
          createRuntimeCall(op.getLoc(), *unbox, source->physicalValues());
      if (unboxCall.getNumResults() != 1 ||
          !unboxCall.getResult(0).getType().isF64())
        return op.emitError() << "ctypes native argument " << index
                              << " float unbox primitive must return f64";
      nativeArgTypes.push_back(builder.getF64Type());
      nativeArgs.push_back(unboxCall.getResult(0));
      continue;
    }
    if (isPointerScalarLayout(*layout)) {
      std::optional<mlir::Value> pointer =
          extractNativePointerArgument(op, builder, *source, facts);
      if (!pointer)
        return op.emitError()
               << "ctypes native argument " << index << " for " << argType
               << " requires None, c_void_p, primitive pointer integer, or "
                  "call-region byref evidence ("
               << describeNativeArgumentSource(*source) << ")";
      nativeArgTypes.push_back(nativePointerType(context));
      nativeArgs.push_back(*pointer);
      continue;
    }
    return op.emitError() << "ctypes native argument " << index << " type "
                          << argType
                          << " is not supported by scalar native lowering";
  }

  llvm::SmallVector<mlir::Type, 1> nativeResultTypes;
  std::optional<CtypesLayout> resultLayout;
  if (*evidence.resultType != "types.NoneType") {
    resultLayout = ctypesStaticLayout(module, *evidence.resultType, facts);
    if (!resultLayout)
      return op.emitError()
             << "ctypes native result type " << *evidence.resultType
             << " has no ABI layout for " << targetFactsLabel(facts);
    if (isIntegerScalarLayout(*resultLayout)) {
      if (resultLayout->kind == CtypesLayout::ABIKind::UnsignedInteger &&
          resultLayout->size == 8)
        return op.emitError()
               << "ctypes native unsigned 64-bit result requires Python "
                  "bigint materialization";
      nativeResultTypes.push_back(nativeIntegerType(builder, *resultLayout));
    } else if (isFloatingScalarLayout(*resultLayout)) {
      if (resultLayout->size != 8)
        return op.emitError()
               << "ctypes native result type " << *evidence.resultType
               << " requires an explicit f32 precision conversion";
      nativeResultTypes.push_back(builder.getF64Type());
    } else if (isPointerScalarLayout(*resultLayout)) {
      nativeResultTypes.push_back(nativePointerType(context));
    } else {
      return op.emitError()
             << "ctypes native result type " << *evidence.resultType
             << " is not supported by scalar native lowering";
    }
  }

  mlir::FunctionType functionType =
      builder.getFunctionType(nativeArgTypes, nativeResultTypes);
  mlir::FailureOr<mlir::func::FuncOp> declaration =
      getOrCreateNativeDeclaration(op, module, builder, evidence.symbolName,
                                   functionType);
  if (mlir::failed(declaration))
    return mlir::failure();
  builder.setInsertionPoint(op);
  mlir::func::CallOp call =
      builder.create<mlir::func::CallOp>(op.getLoc(), *declaration, nativeArgs);

  if (op.getNumResults() != 1)
    return op.emitError() << "ctypes native call expects one Python result";
  if (*evidence.resultType == "types.NoneType") {
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(0), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
  } else if (isIntegerScalarLayout(*resultLayout)) {
    mlir::Value raw = call.getResult(0);
    mlir::IntegerType i64 = builder.getI64Type();
    if (raw.getType() != i64) {
      if (resultLayout->kind == CtypesLayout::ABIKind::UnsignedInteger)
        raw = builder.create<mlir::arith::ExtUIOp>(op.getLoc(), i64, raw)
                  .getResult();
      else
        raw = builder.create<mlir::arith::ExtSIOp>(op.getLoc(), i64, raw)
                  .getResult();
    }
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), raw, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
  } else if (isFloatingScalarLayout(*resultLayout)) {
    RuntimeBundle result;
    if (mlir::failed(initializeObjectFromRawValues(
            op, runtimeContractType(context, "builtins.float"),
            mlir::ValueRange{call.getResult(0)}, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
  } else if (isPointerScalarLayout(*resultLayout)) {
    mlir::Value raw =
        nativePointerToInteger(builder, op.getLoc(), call.getResult(0));
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), raw, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
  } else {
    return op.emitError() << "ctypes native call has unsupported result layout";
  }

  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesCall(py::CallOp op,
                                            const RuntimeBundle &callable) {
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "ctypes arguments", sources, &unpackedSources)))
    return mlir::failure();

  if (callable.binding == "ctypes.sizeof" ||
      callable.binding == "ctypes.alignment") {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError()
             << callable.binding << " expects one static ctypes argument";
    std::string contract = ctypesContractFromBundle(*sources.front());
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    std::optional<CtypesLayout> layout = ctypesStaticLayoutForType(
        module, ctypesTypeFromBundle(*sources.front()), facts);
    if (!layout)
      layout = ctypesStaticLayout(module, contract, facts);
    if (!layout)
      return op.emitError()
             << callable.binding << "(" << contract
             << ") requires complete TargetPlatformFacts before lowering ("
             << targetFactsLabel(facts) << ")";
    std::int64_t value = callable.binding == "ctypes.sizeof"
                             ? static_cast<std::int64_t>(layout->size)
                             : static_cast<std::int64_t>(layout->align);
    builder.setInsertionPoint(op);
    mlir::Value scalar = constantI64(builder, op.getLoc(), value);
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), scalar, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (std::optional<llvm::StringRef> target =
          ctypesFromAddressTarget(callable.binding)) {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError()
             << "ctypes.from_address expects one integer address";
    const RuntimeBundle *address = sources.front();
    if (!address || !address->primitiveI64 || !address->primitiveI64->value ||
        !address->primitiveI64->valid ||
        !isKnownTrue(address->primitiveI64->valid))
      return op.emitError()
             << "ctypes.from_address requires a statically available pointer "
                "integer address";
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.from_address requires TargetPlatformFacts before "
                "lowering";
    builder.setInsertionPoint(op);
    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
    evidence.provenance = RuntimeCtypesEvidence::Provenance::ExternalAddress;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::External;
    evidence.ctypeName = target->str();
    evidence.ctype = ctypesContractType(context, *target);
    evidence.addressValue =
        coerceNativeInteger(builder, op.getLoc(), address->primitiveI64->value,
                            nativePointerIntegerType(builder, facts));
    evidence.addressValid = constantI1(builder, op.getLoc(), true);
    evidence.storageAddressValue = evidence.addressValue;
    evidence.storageAddressValid = evidence.addressValid;
    if (std::optional<CtypesLayout> layout =
            ctypesStaticLayout(module, *target, facts))
      attachCtypesBufferEvidence(builder, op.getLoc(), result, evidence,
                                 *layout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (std::optional<llvm::StringRef> target =
          ctypesFromBufferTarget(callable.binding)) {
    if (op.getNumResults() != 1 || sources.empty() || sources.size() > 2)
      return op.emitError() << "ctypes.from_buffer expects a buffer object and "
                               "optional offset";
    const RuntimeBundle *source = sources.front();
    if (!source || !source->buffer || !source->buffer->addressValue ||
        !source->buffer->addressValid ||
        !isKnownTrue(source->buffer->addressValid))
      return op.emitError()
             << "ctypes.from_buffer(" << *target
             << ") requires statically available buffer address evidence";
    if (!source->buffer->readable || !source->buffer->writable ||
        !source->buffer->cContiguous)
      return op.emitError() << "ctypes.from_buffer(" << *target
                            << ") requires a writable C-contiguous buffer";

    std::int64_t offset = 0;
    if (sources.size() == 2) {
      const RuntimeBundle *offsetSource = sources[1];
      if (!offsetSource || !offsetSource->primitiveI64 ||
          !offsetSource->primitiveI64->value ||
          !offsetSource->primitiveI64->valid ||
          !isKnownTrue(offsetSource->primitiveI64->valid))
        return op.emitError()
               << "ctypes.from_buffer offset requires primitive integer "
                  "evidence";
      std::optional<std::int64_t> staticOffset =
          knownI64Constant(offsetSource->primitiveI64->value);
      if (!staticOffset)
        return op.emitError()
               << "ctypes.from_buffer offset must be statically known";
      if (*staticOffset < 0)
        return op.emitError() << "ctypes.from_buffer offset must be >= 0";
      offset = *staticOffset;
    }

    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.from_buffer requires TargetPlatformFacts before "
                "lowering";
    std::optional<CtypesLayout> targetLayout =
        ctypesStaticLayout(module, *target, facts);
    if (!targetLayout)
      return op.emitError()
             << "ctypes.from_buffer(" << *target << ") has no ABI layout for "
             << targetFactsLabel(facts);
    if (!source->buffer->byteLength || !source->buffer->byteLengthValid ||
        !isKnownTrue(source->buffer->byteLengthValid))
      return op.emitError()
             << "ctypes.from_buffer requires statically available buffer "
                "length evidence";
    std::optional<std::int64_t> sourceLength =
        knownI64Constant(source->buffer->byteLength);
    if (!sourceLength)
      return op.emitError()
             << "ctypes.from_buffer buffer length must be statically known";
    if (offset > *sourceLength ||
        static_cast<std::uint64_t>(*sourceLength - offset) < targetLayout->size)
      return op.emitError() << "ctypes.from_buffer(" << *target
                            << ") exceeds the statically proven buffer size";

    builder.setInsertionPoint(op);
    mlir::Value address = addressWithOffset(
        builder, op.getLoc(), source->buffer->addressValue, offset, facts);
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
    evidence.provenance = RuntimeCtypesEvidence::Provenance::BufferView;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
    evidence.ctypeName = target->str();
    evidence.ctype = ctypesContractType(context, *target);
    evidence.addressValue = address;
    evidence.addressValid = valid;
    evidence.storageAddressValue = address;
    evidence.storageAddressValid = valid;
    keepAliveSource(evidence, *source);
    evidence.keepAlive.append(source->buffer->keepAlive.begin(),
                              source->buffer->keepAlive.end());
    result.ctypes = std::move(evidence);
    result.buffer = makeCtypesBufferEvidence(
        builder, op.getLoc(), *targetLayout, address, valid, *source,
        /*writable=*/true);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (std::optional<llvm::StringRef> target =
          ctypesFromBufferCopyTarget(callable.binding)) {
    if (op.getNumResults() != 1 || sources.empty() || sources.size() > 2)
      return op.emitError()
             << "ctypes.from_buffer_copy expects a buffer object and optional "
                "offset";
    const RuntimeBundle *source = sources.front();
    if (!source || !source->buffer || !source->buffer->addressValue ||
        !source->buffer->addressValid ||
        !isKnownTrue(source->buffer->addressValid))
      return op.emitError()
             << "ctypes.from_buffer_copy(" << *target
             << ") requires statically available buffer address evidence";
    if (!source->buffer->readable || !source->buffer->cContiguous)
      return op.emitError() << "ctypes.from_buffer_copy(" << *target
                            << ") requires a readable C-contiguous buffer";

    std::int64_t offset = 0;
    if (sources.size() == 2) {
      const RuntimeBundle *offsetSource = sources[1];
      if (!offsetSource || !offsetSource->primitiveI64 ||
          !offsetSource->primitiveI64->value ||
          !offsetSource->primitiveI64->valid ||
          !isKnownTrue(offsetSource->primitiveI64->valid))
        return op.emitError()
               << "ctypes.from_buffer_copy offset requires primitive integer "
                  "evidence";
      std::optional<std::int64_t> staticOffset =
          knownI64Constant(offsetSource->primitiveI64->value);
      if (!staticOffset)
        return op.emitError()
               << "ctypes.from_buffer_copy offset must be statically known";
      if (*staticOffset < 0)
        return op.emitError() << "ctypes.from_buffer_copy offset must be >= 0";
      offset = *staticOffset;
    }

    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.from_buffer_copy requires TargetPlatformFacts before "
                "lowering";
    std::optional<CtypesLayout> targetLayout =
        ctypesStaticLayout(module, *target, facts);
    if (!targetLayout)
      return op.emitError()
             << "ctypes.from_buffer_copy(" << *target
             << ") has no ABI layout for " << targetFactsLabel(facts);
    if (!isIntegerScalarLayout(*targetLayout) &&
        !isPointerScalarLayout(*targetLayout))
      return op.emitError()
             << "ctypes.from_buffer_copy(" << *target
             << ") currently requires an integer or pointer scalar layout";
    if (!source->buffer->byteLength || !source->buffer->byteLengthValid ||
        !isKnownTrue(source->buffer->byteLengthValid))
      return op.emitError()
             << "ctypes.from_buffer_copy requires statically available buffer "
                "length evidence";
    std::optional<std::int64_t> sourceLength =
        knownI64Constant(source->buffer->byteLength);
    if (!sourceLength)
      return op.emitError() << "ctypes.from_buffer_copy buffer length must be "
                               "statically known";
    if (offset > *sourceLength ||
        static_cast<std::uint64_t>(*sourceLength - offset) < targetLayout->size)
      return op.emitError() << "ctypes.from_buffer_copy(" << *target
                            << ") exceeds the statically proven buffer size";

    builder.setInsertionPoint(op);
    mlir::Value sourceAddress = addressWithOffset(
        builder, op.getLoc(), source->buffer->addressValue, offset, facts);
    mlir::IntegerType nativeType = nativeIntegerType(builder, *targetLayout);
    mlir::Value nativeValue = loadNativeIntegerFromAddress(
        builder, op.getLoc(), sourceAddress, nativeType, facts);
    mlir::Value ownedAddress =
        addressOfNativeCellAlloca(builder, op.getLoc(), nativeValue, facts);
    mlir::Value valid = constantI1(builder, op.getLoc(), true);

    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
    evidence.provenance = RuntimeCtypesEvidence::Provenance::NativeCell;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
    evidence.ctypeName = target->str();
    evidence.ctype = ctypesContractType(context, *target);
    evidence.scalarValue =
        widenNativeInteger(builder, op.getLoc(), nativeValue, *targetLayout);
    evidence.scalarValid = valid;
    evidence.addressValue = ownedAddress;
    evidence.addressValid = valid;
    evidence.storageAddressValue = ownedAddress;
    evidence.storageAddressValid = valid;
    evidence.ownsNativeStorage = true;
    attachCtypesBufferEvidence(builder, op.getLoc(), result, evidence,
                               *targetLayout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.addressof") {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError() << "ctypes.addressof expects one ctypes object";
    const RuntimeBundle *source = sources.front();
    if (!source || !source->ctypes)
      return op.emitError()
             << "ctypes.addressof requires materialized _CData evidence";
    mlir::Value storageAddress = cdataStorageAddress(*source->ctypes);
    mlir::Value storageValid = cdataStorageAddressValid(*source->ctypes);
    if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
      return op.emitError()
             << "ctypes.addressof requires a materialized _CData storage "
                "address";
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.addressof requires TargetPlatformFacts before lowering";
    builder.setInsertionPoint(op);
    mlir::Value raw =
        coerceNativeInteger(builder, op.getLoc(), storageAddress,
                            nativePointerIntegerType(builder, facts));
    mlir::Value i64 = raw;
    auto rawType = mlir::cast<mlir::IntegerType>(raw.getType());
    if (rawType.getWidth() < 64)
      i64 = builder
                .create<mlir::arith::ExtUIOp>(op.getLoc(), builder.getI64Type(),
                                              raw)
                .getResult();
    else if (rawType.getWidth() > 64)
      i64 = builder
                .create<mlir::arith::TruncIOp>(op.getLoc(),
                                               builder.getI64Type(), raw)
                .getResult();
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), i64, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.cast") {
    if (op.getNumResults() != 1 || sources.size() != 2)
      return op.emitError() << "ctypes.cast expects a source and ctypes type";
    const RuntimeBundle *source = sources[0];
    const RuntimeBundle *target = sources[1];
    if (!source || !target)
      return op.emitError() << "ctypes.cast arguments have no evidence";
    std::optional<std::string> targetContract = ctypesTypeObjectName(*target);
    if (!targetContract)
      return op.emitError()
             << "ctypes.cast target must be a static ctypes type object";

    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.cast requires TargetPlatformFacts before lowering";
    builder.setInsertionPoint(op);
    std::optional<mlir::Value> address =
        extractPointerAddressInteger(op, builder, *source, facts);
    if (!address)
      return op.emitError()
             << "ctypes.cast source requires None, primitive pointer integer, "
                "c_void_p, or pointer evidence with a concrete external "
                "address ("
             << describeNativeArgumentSource(*source) << ")";

    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.ctypeName = *targetContract;
    evidence.ctype = ctypesTypeFromBundle(*target);
    evidence.provenance = RuntimeCtypesEvidence::Provenance::Cast;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::External;
    evidence.addressValue = *address;
    evidence.addressValid = constantI1(builder, op.getLoc(), true);
    keepAliveSource(evidence, *source);
    if (isCtypesVoidPointer(*targetContract)) {
      evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
      evidence.scalarValue = *address;
      evidence.scalarValid = evidence.addressValid;
    } else if (isCtypesPointerContract(*targetContract)) {
      evidence.kind = RuntimeCtypesEvidence::Kind::Pointer;
      evidence.pointeeType = *targetContract;
    } else {
      return op.emitError() << "ctypes.cast target " << *targetContract
                            << " is not a supported pointer-like ctypes type";
    }

    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.byref" ||
      callable.binding == "ctypes.pointer") {
    if (op.getNumResults() != 1 || sources.empty() || sources.size() > 2)
      return op.emitError() << callable.binding
                            << " expects a ctypes object and optional offset";
    const RuntimeBundle *pointee = sources.front();
    if (!pointee || !pointee->ctypes ||
        pointee->ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
      return op.emitError()
             << callable.binding << " requires an erased ctypes cell";
    if (sources.size() == 2) {
      const RuntimeBundle *offset = sources[1];
      if (!offset || !offset->primitiveI64)
        return op.emitError()
               << callable.binding << " offset requires primitive integer "
               << "evidence";
      // Offset provenance is a separate proof term. Until it exists, only the
      // default zero-offset form is accepted on the erased path.
      return op.emitError()
             << callable.binding << " erased offset is not implemented yet";
    }

    builder.setInsertionPoint(op);
    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Pointer;
    evidence.ctypeName = callable.binding == "ctypes.pointer"
                             ? "_ctypes._Pointer"
                             : pointee->ctypes->ctypeName;
    if (callable.binding == "ctypes.pointer") {
      mlir::Type pointeeType =
          pointee->ctypes->ctype
              ? pointee->ctypes->ctype
              : ctypesContractType(context, pointee->ctypes->ctypeName);
      evidence.ctype =
          py::ContractType::get(context, "_ctypes._Pointer", {pointeeType});
    } else {
      evidence.ctype = pointee->ctypes->ctype ? pointee->ctypes->ctype
                                              : op.getResult(0).getType();
    }
    evidence.provenance =
        callable.binding == "ctypes.byref"
            ? RuntimeCtypesEvidence::Provenance::CallRegionBorrow
            : RuntimeCtypesEvidence::Provenance::NativeCell;
    evidence.lifetime = callable.binding == "ctypes.byref"
                            ? RuntimeCtypesEvidence::Lifetime::CallRegion
                            : RuntimeCtypesEvidence::Lifetime::Owner;
    evidence.pointeeType = pointee->ctypes->ctypeName;
    evidence.pointeeScalarValue = pointee->ctypes->scalarValue;
    evidence.pointeeScalarValid = pointee->ctypes->scalarValid;
    mlir::Value pointeeAddress = cdataStorageAddress(*pointee->ctypes);
    mlir::Value pointeeAddressValid =
        cdataStorageAddressValid(*pointee->ctypes);
    if (!pointeeAddress || !pointeeAddressValid ||
        !isKnownTrue(pointeeAddressValid))
      return op.emitError()
             << callable.binding
             << " requires a materialized pointee _CData storage address";
    evidence.addressValue = pointeeAddress;
    evidence.addressValid = pointeeAddressValid;
    keepAliveSource(evidence, *pointee);
    evidence.callRegionBorrow = callable.binding == "ctypes.byref";
    if (evidence.callRegionBorrow) {
      evidence.provenance = RuntimeCtypesEvidence::Provenance::CallRegionBorrow;
      evidence.lifetime = RuntimeCtypesEvidence::Lifetime::CallRegion;
    } else {
      std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
      if (!facts)
        return op.emitError()
               << "ctypes.pointer requires TargetPlatformFacts before "
                  "materializing pointer storage";
      std::optional<CtypesLayout> pointerLayout =
          ctypesLayout("_ctypes._Pointer", facts);
      if (!pointerLayout)
        return op.emitError()
               << "ctypes.pointer has no pointer object layout for "
               << targetFactsLabel(facts);
      mlir::Value pointerInteger =
          coerceNativeInteger(builder, op.getLoc(), pointeeAddress,
                              nativePointerIntegerType(builder, facts));
      evidence.storageAddressValue = addressOfNativeCellAlloca(
          builder, op.getLoc(), pointerInteger, facts);
      evidence.storageAddressValid = constantI1(builder, op.getLoc(), true);
      evidence.ownsNativeStorage = true;
      result.buffer = makeCtypesBufferEvidence(
          builder, op.getLoc(), *pointerLayout, evidence.storageAddressValue,
          evidence.storageAddressValid, *pointee,
          /*writable=*/true);
    }
    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.POINTER") {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError() << "ctypes.POINTER expects one static ctypes type";
    const RuntimeBundle *pointee = sources.front();
    if (!pointee)
      return op.emitError() << "ctypes.POINTER argument has no evidence";
    std::optional<std::string> pointeeName = ctypesTypeObjectName(*pointee);
    if (!pointeeName)
      return op.emitError()
             << "ctypes.POINTER expects a static ctypes type object";
    mlir::Type pointeeType = ctypesTypeFromBundle(*pointee);
    if (!pointeeType)
      pointeeType = ctypesContractType(context, *pointeeName);
    mlir::Type pointerType =
        py::ContractType::get(context, "_ctypes._Pointer", {pointeeType});
    valueBundles[op.getResult(0)] =
        RuntimeBundle::typeObject(op.getResult(0).getType(), pointerType);
    erase.push_back(op);
    return mlir::success();
  }

  return op.emitError() << callable.binding
                        << " has no erased ctypes lowering yet";
}

} // namespace py::runtime_lowering
