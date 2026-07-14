#include "PyDialectTypes.h"
#include "PyProtocols.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {

void PyDialect::initialize() {
  addTypes<ContractType, LiteralType, TypeVarType, ParamSpecType,
           TypeVarTupleType, UnpackType, InferVarType, TypeType, ProtocolType,
           ExceptionType, ExceptionCellType, TracebackType, LocationType,
           CallableType, UnionType, OverloadType, SelfType>();

  addOperations<
#define GET_OP_LIST
#include "PyOps.cpp.inc"
      >();

  addInterfaces<protocols::TableCache>();
}

mlir::Type PyDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return mlir::Type();

  mlir::MLIRContext *ctx = parser.getContext();

  auto parseTypeList =
      [&](llvm::SmallVectorImpl<mlir::Type> &out) -> mlir::LogicalResult {
    if (parser.parseLSquare())
      return mlir::failure();
    if (mlir::succeeded(parser.parseOptionalRSquare()))
      return mlir::success();
    do {
      mlir::Type elem;
      if (parser.parseType(elem))
        return mlir::failure();
      out.push_back(elem);
    } while (mlir::succeeded(parser.parseOptionalComma()));
    return parser.parseRSquare();
  };
  auto parseStringArray =
      [&](llvm::SmallVectorImpl<mlir::StringAttr> &out) -> mlir::LogicalResult {
    mlir::ArrayAttr arrayAttr;
    if (parser.parseAttribute(arrayAttr))
      return mlir::failure();
    for (mlir::Attribute element : arrayAttr) {
      auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(element);
      if (!stringAttr) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected string elements in Callable metadata");
        return mlir::failure();
      }
      out.push_back(stringAttr);
    }
    return mlir::success();
  };
  auto parseBoolArray =
      [&](llvm::SmallVectorImpl<mlir::BoolAttr> &out) -> mlir::LogicalResult {
    mlir::ArrayAttr arrayAttr;
    if (parser.parseAttribute(arrayAttr))
      return mlir::failure();
    for (mlir::Attribute element : arrayAttr) {
      auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(element);
      if (!boolAttr) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected bool elements in Callable metadata");
        return mlir::failure();
      }
      out.push_back(boolAttr);
    }
    return mlir::success();
  };
  auto parseCallableType = [&](bool arrowResults) -> mlir::Type {
    llvm::SmallVector<mlir::Type, 4> positionalTypes;
    llvm::SmallVector<mlir::Type, 4> kwonlyTypes;
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    llvm::SmallVector<mlir::StringAttr, 4> positionalNames;
    llvm::SmallVector<mlir::StringAttr, 4> kwOnlyNames;
    llvm::SmallVector<mlir::BoolAttr, 4> positionalDefaults;
    llvm::SmallVector<mlir::BoolAttr, 4> kwOnlyDefaults;
    mlir::Type varargType;
    mlir::Type kwargsType;
    mlir::StringAttr varargName;
    mlir::StringAttr kwargsName;
    uint64_t positionalOnlyCount = 0;
    bool varargSeen = false;
    bool kwonlySeen = false;
    bool kwargsSeen = false;
    bool posonlySeen = false;
    bool argNamesSeen = false;
    bool kwNamesSeen = false;
    bool argDefaultsSeen = false;
    bool kwDefaultsSeen = false;
    bool varargNameSeen = false;
    bool kwargsNameSeen = false;
    bool returnsSeen = false;

    if (mlir::failed(parseTypeList(positionalTypes)))
      return mlir::Type();

    while (mlir::succeeded(parser.parseOptionalComma())) {
      llvm::StringRef section;
      if (parser.parseKeyword(&section))
        return mlir::Type();
      if (section == "vararg") {
        if (varargSeen || parser.parseEqual() || parser.parseType(varargType))
          return mlir::Type();
        varargSeen = true;
        continue;
      }
      if (section == "posonly") {
        if (posonlySeen || parser.parseEqual() ||
            parser.parseInteger(positionalOnlyCount))
          return mlir::Type();
        posonlySeen = true;
        continue;
      }
      if (section == "kwonly") {
        if (kwonlySeen || parser.parseEqual() ||
            mlir::failed(parseTypeList(kwonlyTypes)))
          return mlir::Type();
        kwonlySeen = true;
        continue;
      }
      if (section == "kwargs") {
        if (kwargsSeen || parser.parseEqual() || parser.parseType(kwargsType))
          return mlir::Type();
        kwargsSeen = true;
        continue;
      }
      if (section == "arg_names") {
        if (argNamesSeen || parser.parseEqual() ||
            mlir::failed(parseStringArray(positionalNames)))
          return mlir::Type();
        argNamesSeen = true;
        continue;
      }
      if (section == "kw_names") {
        if (kwNamesSeen || parser.parseEqual() ||
            mlir::failed(parseStringArray(kwOnlyNames)))
          return mlir::Type();
        kwNamesSeen = true;
        continue;
      }
      if (section == "arg_defaults") {
        if (argDefaultsSeen || parser.parseEqual() ||
            mlir::failed(parseBoolArray(positionalDefaults)))
          return mlir::Type();
        argDefaultsSeen = true;
        continue;
      }
      if (section == "kw_defaults") {
        if (kwDefaultsSeen || parser.parseEqual() ||
            mlir::failed(parseBoolArray(kwOnlyDefaults)))
          return mlir::Type();
        kwDefaultsSeen = true;
        continue;
      }
      if (section == "vararg_name") {
        if (varargNameSeen || parser.parseEqual() ||
            parser.parseAttribute(varargName))
          return mlir::Type();
        varargNameSeen = true;
        continue;
      }
      if (section == "kwargs_name") {
        if (kwargsNameSeen || parser.parseEqual() ||
            parser.parseAttribute(kwargsName))
          return mlir::Type();
        kwargsNameSeen = true;
        continue;
      }
      if (!arrowResults && section == "returns") {
        if (returnsSeen || parser.parseEqual() ||
            mlir::failed(parseTypeList(resultTypes)))
          return mlir::Type();
        returnsSeen = true;
        continue;
      }
      parser.emitError(parser.getCurrentLocation(),
                       "unexpected token '" + section +
                           "' in Callable contract declaration");
      return mlir::Type();
    }

    if (arrowResults) {
      if (parser.parseArrow())
        return mlir::Type();
      if (mlir::failed(parseTypeList(resultTypes)))
        return mlir::Type();
    } else if (!returnsSeen) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'returns = [...]' in Callable contract");
      return mlir::Type();
    }

    auto wrongSize = [&](llvm::StringRef name, std::size_t got,
                         std::size_t expected) {
      parser.emitError(parser.getCurrentLocation(),
                       "Callable metadata '" + name + "' has " +
                           std::to_string(got) + " entries but expected " +
                           std::to_string(expected));
      return mlir::Type();
    };
    if (!positionalNames.empty() &&
        positionalNames.size() != positionalTypes.size())
      return wrongSize("arg_names", positionalNames.size(),
                       positionalTypes.size());
    if (!positionalDefaults.empty() &&
        positionalDefaults.size() != positionalTypes.size())
      return wrongSize("arg_defaults", positionalDefaults.size(),
                       positionalTypes.size());
    if (!kwOnlyNames.empty() && kwOnlyNames.size() != kwonlyTypes.size())
      return wrongSize("kw_names", kwOnlyNames.size(), kwonlyTypes.size());
    if (!kwOnlyDefaults.empty() && kwOnlyDefaults.size() != kwonlyTypes.size())
      return wrongSize("kw_defaults", kwOnlyDefaults.size(),
                       kwonlyTypes.size());
    if (varargName && !varargType)
      return wrongSize("vararg_name", 1, 0);
    if (kwargsName && !kwargsType)
      return wrongSize("kwargs_name", 1, 0);
    if (positionalOnlyCount > positionalTypes.size())
      return wrongSize("posonly", positionalOnlyCount, positionalTypes.size());

    return CallableType::get(
        ctx, positionalTypes, kwonlyTypes, varargType, kwargsType, resultTypes,
        positionalNames, kwOnlyNames, positionalDefaults, kwOnlyDefaults,
        varargName, kwargsName, static_cast<unsigned>(positionalOnlyCount));
  };

  if (keyword == "int")
    return pyIntContractType(ctx);
  if (keyword == "float")
    return pyFloatContractType(ctx);
  if (keyword == "bool")
    return pyBoolContractType(ctx);
  if (keyword == "str")
    return pyStrContractType(ctx);
  if (keyword == "none")
    return pyNoneContractType(ctx);
  if (keyword == "object")
    return pyObjectContractType(ctx);
  if (keyword == "self")
    return SelfType::get(ctx);
  if (keyword == "contract") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::StringAttr contractNameAttr;
    llvm::SmallVector<mlir::Type> arguments;
    if (parser.parseAttribute(contractNameAttr))
      return mlir::Type();
    if (mlir::succeeded(parser.parseOptionalComma()) &&
        mlir::failed(parseTypeList(arguments)))
      return mlir::Type();
    if (parser.parseGreater())
      return mlir::Type();
    return ContractType::get(ctx, contractNameAttr.getValue(), arguments);
  }
  if (keyword == "literal") {
    if (parser.parseLess())
      return mlir::Type();
    llvm::StringRef literalKeyword;
    std::string spelling;
    if (mlir::succeeded(parser.parseOptionalKeyword(&literalKeyword))) {
      spelling = literalKeyword.str();
    } else {
      mlir::StringAttr literalAttr;
      if (parser.parseAttribute(literalAttr))
        return mlir::Type();
      spelling = literalAttr.getValue().str();
    }
    if (parser.parseGreater())
      return mlir::Type();
    return LiteralType::get(ctx, spelling);
  }
  if (keyword == "typevar") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::StringAttr nameAttr;
    if (parser.parseAttribute(nameAttr) || parser.parseGreater())
      return mlir::Type();
    return TypeVarType::get(ctx, nameAttr.getValue());
  }
  if (keyword == "paramspec") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::StringAttr nameAttr;
    if (parser.parseAttribute(nameAttr) || parser.parseGreater())
      return mlir::Type();
    return ParamSpecType::get(ctx, nameAttr.getValue());
  }
  if (keyword == "typevartuple") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::StringAttr nameAttr;
    if (parser.parseAttribute(nameAttr) || parser.parseGreater())
      return mlir::Type();
    return TypeVarTupleType::get(ctx, nameAttr.getValue());
  }
  if (keyword == "unpack") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type packedType;
    if (parser.parseType(packedType) || parser.parseGreater())
      return mlir::Type();
    return UnpackType::get(ctx, packedType);
  }
  if (keyword == "infervar") {
    if (parser.parseLess())
      return mlir::Type();
    unsigned id = 0;
    if (parser.parseInteger(id) || parser.parseGreater())
      return mlir::Type();
    return InferVarType::get(ctx, id);
  }
  if (keyword == "tuple") {
    if (parser.parseLess())
      return mlir::Type();
    llvm::SmallVector<mlir::Type> elementTypes;
    if (mlir::succeeded(parser.parseOptionalGreater()))
      return ContractType::get(ctx, "builtins.tuple", elementTypes);
    do {
      mlir::Type element;
      if (parser.parseType(element))
        return mlir::Type();
      elementTypes.push_back(element);
    } while (mlir::succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater())
      return mlir::Type();
    return ContractType::get(ctx, "builtins.tuple", elementTypes);
  }
  if (keyword == "dict") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type keyType, valueType;
    if (parser.parseType(keyType) || parser.parseComma() ||
        parser.parseType(valueType) || parser.parseGreater())
      return mlir::Type();
    return ContractType::get(ctx, "builtins.dict", {keyType, valueType});
  }
  if (keyword == "list") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type elementType;
    if (parser.parseType(elementType) || parser.parseGreater())
      return mlir::Type();
    return ContractType::get(ctx, "builtins.list", {elementType});
  }
  if (keyword == "iterator_state") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type sourceType, elementType;
    if (parser.parseType(sourceType) || parser.parseComma() ||
        parser.parseType(elementType) || parser.parseGreater())
      return mlir::Type();
    (void)sourceType;
    return iteratorProtocolType(ctx, elementType);
  }
  if (keyword == "union") {
    if (parser.parseLess())
      return mlir::Type();
    llvm::SmallVector<mlir::Type> memberTypes;
    do {
      mlir::Type member;
      if (parser.parseType(member))
        return mlir::Type();
      memberTypes.push_back(member);
    } while (mlir::succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater())
      return mlir::Type();
    if (memberTypes.size() < 2) {
      parser.emitError(parser.getCurrentLocation(),
                       "py.union requires at least two member types");
      return mlir::Type();
    }
    mlir::Type normalized = UnionType::getNormalized(ctx, memberTypes);
    if (!normalized || !mlir::isa<UnionType>(normalized)) {
      parser.emitError(parser.getCurrentLocation(),
                       "py.union requires at least two distinct member types");
      return mlir::Type();
    }
    return normalized;
  }
  if (keyword == "overload") {
    if (parser.parseLess())
      return mlir::Type();
    llvm::SmallVector<mlir::Type> candidateTypes;
    if (mlir::failed(parseTypeList(candidateTypes)) || parser.parseGreater())
      return mlir::Type();
    if (candidateTypes.size() < 2) {
      parser.emitError(parser.getCurrentLocation(),
                       "py.overload requires at least two candidates");
      return mlir::Type();
    }
    for (mlir::Type candidate : candidateTypes) {
      if (mlir::isa<CallableType>(candidate))
        continue;
      parser.emitError(parser.getCurrentLocation(),
                       "py.overload candidates must be Callable contracts");
      return mlir::Type();
    }
    return OverloadType::get(ctx, candidateTypes);
  }
  if (keyword == "callable") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type callable = parseCallableType(/*arrowResults=*/false);
    if (!callable || parser.parseGreater())
      return mlir::Type();
    return callable;
  }
  if (keyword == "class") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::StringAttr classNameAttr;
    if (parser.parseAttribute(classNameAttr) || parser.parseGreater())
      return mlir::Type();
    return ContractType::get(ctx, classNameAttr.getValue());
  }
  if (keyword == "type") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type instanceType;
    if (parser.parseType(instanceType) || parser.parseGreater())
      return mlir::Type();
    return TypeType::get(ctx, instanceType);
  }
  if (keyword == "protocol") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::StringAttr protocolNameAttr;
    llvm::SmallVector<mlir::Type> arguments;
    if (parser.parseAttribute(protocolNameAttr) || parser.parseComma())
      return mlir::Type();
    if (protocolNameAttr.getValue() == "Callable") {
      mlir::Type callable = parseCallableType(/*arrowResults=*/true);
      if (!callable || parser.parseGreater())
        return mlir::Type();
      return callable;
    }
    if (mlir::failed(parseTypeList(arguments)) || parser.parseGreater())
      return mlir::Type();
    return ProtocolType::get(ctx, protocolNameAttr.getValue(), arguments);
  }
  if (keyword == "exception")
    return ExceptionType::get(ctx);
  if (keyword == "exception_cell")
    return ExceptionCellType::get(ctx);
  if (keyword == "traceback")
    return TracebackType::get(ctx);
  if (keyword == "location")
    return LocationType::get(ctx);
  parser.emitError(parser.getCurrentLocation(), "unknown py dialect type '")
      << keyword << "'";
  return mlir::Type();
}

void PyDialect::printType(mlir::Type type,
                          mlir::DialectAsmPrinter &printer) const {
  auto printTypeList = [&](llvm::ArrayRef<mlir::Type> types) {
    printer << "[";
    llvm::interleaveComma(types, printer,
                          [&](mlir::Type element) { printer << element; });
    printer << "]";
  };
  auto printAttrList = [&](auto attrs) {
    printer << "[";
    for (std::size_t index = 0; index < attrs.size(); ++index) {
      if (index != 0)
        printer << ", ";
      if (attrs[index])
        printer << attrs[index];
      else
        printer << "\"\"";
    }
    printer << "]";
  };

  llvm::TypeSwitch<mlir::Type>(type)
      .Case<SelfType>([&](SelfType) { printer << "self"; })
      .Case<UnionType>([&](UnionType unionTy) {
        printer << "union<";
        llvm::interleaveComma(unionTy.getMemberTypes(), printer,
                              [&](mlir::Type member) { printer << member; });
        printer << ">";
      })
      .Case<OverloadType>([&](OverloadType overloadTy) {
        printer << "overload<";
        printTypeList(overloadTy.getCandidateTypes());
        printer << ">";
      })
      .Case<ContractType>([&](ContractType contractTy) {
        printer << "contract<\"" << contractTy.getContractName() << "\"";
        if (!contractTy.getArguments().empty()) {
          printer << ", ";
          printTypeList(contractTy.getArguments());
        }
        printer << ">";
      })
      .Case<LiteralType>([&](LiteralType literalTy) {
        printer << "literal<" << literalTy.getSpelling() << ">";
      })
      .Case<TypeVarType>([&](TypeVarType typeVarTy) {
        printer << "typevar<\"" << typeVarTy.getName() << "\">";
      })
      .Case<ParamSpecType>([&](ParamSpecType paramSpecTy) {
        printer << "paramspec<\"" << paramSpecTy.getName() << "\">";
      })
      .Case<TypeVarTupleType>([&](TypeVarTupleType typeVarTupleTy) {
        printer << "typevartuple<\"" << typeVarTupleTy.getName() << "\">";
      })
      .Case<UnpackType>([&](UnpackType unpackTy) {
        printer << "unpack<" << unpackTy.getPackedType() << ">";
      })
      .Case<InferVarType>([&](InferVarType inferVarTy) {
        printer << "infervar<" << inferVarTy.getId() << ">";
      })
      .Case<TypeType>([&](TypeType typeTy) {
        printer << "type<" << typeTy.getInstanceType() << ">";
      })
      .Case<ProtocolType>([&](ProtocolType protocolTy) {
        printer << "protocol<\"" << protocolTy.getProtocolName() << "\", ";
        printTypeList(protocolTy.getArguments());
        printer << ">";
      })
      .Case<CallableType>([&](CallableType sigTy) {
        printer << "callable<";
        printTypeList(sigTy.getPositionalTypes());
        if (sigTy.getPositionalOnlyCount() != 0)
          printer << ", posonly = " << sigTy.getPositionalOnlyCount();
        if (sigTy.hasVararg()) {
          printer << ", vararg = " << sigTy.getVarargType();
        }
        if (!sigTy.getKwOnlyTypes().empty()) {
          printer << ", kwonly = ";
          printTypeList(sigTy.getKwOnlyTypes());
        }
        if (sigTy.hasKwarg()) {
          printer << ", kwargs = " << sigTy.getKwargType();
        }
        if (!sigTy.getPositionalNames().empty()) {
          printer << ", arg_names = ";
          printAttrList(sigTy.getPositionalNames());
        }
        if (!sigTy.getPositionalDefaults().empty()) {
          printer << ", arg_defaults = ";
          printAttrList(sigTy.getPositionalDefaults());
        }
        if (!sigTy.getKwOnlyNames().empty()) {
          printer << ", kw_names = ";
          printAttrList(sigTy.getKwOnlyNames());
        }
        if (!sigTy.getKwOnlyDefaults().empty()) {
          printer << ", kw_defaults = ";
          printAttrList(sigTy.getKwOnlyDefaults());
        }
        if (sigTy.getVarargName())
          printer << ", vararg_name = " << sigTy.getVarargName();
        if (sigTy.getKwargName())
          printer << ", kwargs_name = " << sigTy.getKwargName();
        printer << ", returns = ";
        printTypeList(sigTy.getResultTypes());
        printer << ">";
      })
      .Case<ExceptionType>([&](ExceptionType) { printer << "exception"; })
      .Case<ExceptionCellType>(
          [&](ExceptionCellType) { printer << "exception_cell"; })
      .Case<TracebackType>([&](TracebackType) { printer << "traceback"; })
      .Case<LocationType>([&](LocationType) { printer << "location"; })
      .Default(
          [&](mlir::Type) { llvm_unreachable("unknown py type to print"); });
}

mlir::ParseResult ClassOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &state) {
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return mlir::failure();
  mlir::Region *body = state.addRegion();
  if (parser.parseRegion(*body))
    return mlir::failure();
  // `{}` parses as an empty region; materialize the block the SymbolTable
  // trait requires so declaration-only classes stay writable by hand.
  if (body->empty())
    body->emplaceBlock();
  return mlir::success();
}

void ClassOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {mlir::SymbolTable::getSymbolAttrName()});
  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

} // namespace py

#include "PyDialect.cpp.inc"

// ODSが生成したOpクラスの定義本体を取り込む
#define GET_OP_CLASSES
#include "PyOps.cpp.inc"
#undef GET_OP_CLASSES
