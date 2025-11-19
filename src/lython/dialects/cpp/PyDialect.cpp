#include "PyDialectTypes.h"

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

using namespace mlir;

namespace py {

void PyDialect::initialize() {
  addTypes<IntType, FloatType, BoolType, StrType, ObjectType, NoneType,
           TupleType, DictType, ClassType, FuncSignatureType, FuncType,
           PrimFuncType>();

  addOperations<
#define GET_OP_LIST
#include "PyOps.cpp.inc"
      >();
}

Type PyDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  MLIRContext *ctx = parser.getContext();

  auto parseTypeList = [&](SmallVectorImpl<Type> &out) -> LogicalResult {
    if (parser.parseLSquare())
      return failure();
    if (succeeded(parser.parseOptionalRSquare()))
      return success();
    do {
      Type elem;
      if (parser.parseType(elem))
        return failure();
      out.push_back(elem);
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseRSquare();
  };

  if (keyword == "int")
    return IntType::get(ctx);
  if (keyword == "float")
    return FloatType::get(ctx);
  if (keyword == "bool")
    return BoolType::get(ctx);
  if (keyword == "str")
    return StrType::get(ctx);
  if (keyword == "object")
    return ObjectType::get(ctx);
  if (keyword == "none")
    return NoneType::get(ctx);
  if (keyword == "tuple") {
    if (parser.parseLess())
      return Type();
    SmallVector<Type> elementTypes;
    if (succeeded(parser.parseOptionalGreater()))
      return TupleType::get(ctx, elementTypes);
    do {
      Type element;
      if (parser.parseType(element))
        return Type();
      elementTypes.push_back(element);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater())
      return Type();
    return TupleType::get(ctx, elementTypes);
  }
  if (keyword == "dict") {
    if (parser.parseLess())
      return Type();
    Type keyType, valueType;
    if (parser.parseType(keyType) || parser.parseComma() ||
        parser.parseType(valueType) || parser.parseGreater())
      return Type();
    return DictType::get(ctx, keyType, valueType);
  }
  if (keyword == "class") {
    if (parser.parseLess())
      return Type();
    StringAttr classNameAttr;
    if (parser.parseAttribute(classNameAttr) || parser.parseGreater())
      return Type();
    return ClassType::get(ctx, classNameAttr.getValue());
  }
  if (keyword == "func") {
    if (parser.parseLess())
      return Type();
    Type signatureType;
    if (parser.parseType(signatureType) || parser.parseGreater())
      return Type();
    auto signature = dyn_cast<FuncSignatureType>(signatureType);
    if (!signature) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected FuncSignatureType after 'func<'");
      return Type();
    }
    return FuncType::get(ctx, signature);
  }
  if (keyword == "funcsig") {
    if (parser.parseLess())
      return Type();
    SmallVector<Type, 4> positionalTypes;
    SmallVector<Type, 4> kwonlyTypes;
    SmallVector<Type, 4> resultTypes;
    Type varargType;
    Type kwargsType;
    bool varargSeen = false;
    bool kwonlySeen = false;
    bool kwargsSeen = false;

    if (failed(parseTypeList(positionalTypes)))
      return Type();

    while (succeeded(parser.parseOptionalComma())) {
      StringRef section;
      if (parser.parseKeyword(&section))
        return Type();
      if (section == "vararg") {
        if (varargSeen || parser.parseEqual() || parser.parseType(varargType))
          return Type();
        varargSeen = true;
        continue;
      }
      if (section == "kwonly") {
        if (kwonlySeen || parser.parseEqual() ||
            failed(parseTypeList(kwonlyTypes)))
          return Type();
        kwonlySeen = true;
        continue;
      }
      if (section == "kwargs") {
        if (kwargsSeen || parser.parseEqual() || parser.parseType(kwargsType))
          return Type();
        kwargsSeen = true;
        continue;
      }
      parser.emitError(parser.getCurrentLocation(),
                       "unexpected token '" + section +
                           "' in funcsig type declaration");
      return Type();
    }

    if (parser.parseArrow())
      return Type();
    if (failed(parseTypeList(resultTypes)) || parser.parseGreater())
      return Type();

    return FuncSignatureType::get(ctx, positionalTypes, kwonlyTypes, varargType,
                                  kwargsType, resultTypes);
  }
  if (keyword == "prim.func") {
    if (parser.parseLess())
      return Type();
    Type signatureType;
    if (parser.parseType(signatureType) || parser.parseGreater())
      return Type();
    auto signature = dyn_cast<FunctionType>(signatureType);
    if (!signature) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected FunctionType after 'prim.func<'");
      return Type();
    }
    return PrimFuncType::get(ctx, signature);
  }

  parser.emitError(parser.getCurrentLocation(), "unknown py dialect type '")
      << keyword << "'";
  return Type();
}

void PyDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto printTypeList = [&](ArrayRef<Type> types) {
    printer << "[";
    llvm::interleaveComma(types, printer,
                          [&](Type element) { printer << element; });
    printer << "]";
  };

  llvm::TypeSwitch<Type>(type)
      .Case<IntType>([&](IntType) { printer << "int"; })
      .Case<FloatType>([&](FloatType) { printer << "float"; })
      .Case<BoolType>([&](BoolType) { printer << "bool"; })
      .Case<StrType>([&](StrType) { printer << "str"; })
      .Case<ObjectType>([&](ObjectType) { printer << "object"; })
      .Case<NoneType>([&](NoneType) { printer << "none"; })
      .Case<TupleType>([&](TupleType tupleTy) {
        printer << "tuple<";
        auto elements = tupleTy.getElementTypes();
        llvm::interleaveComma(elements, printer,
                              [&](Type element) { printer << element; });
        printer << ">";
      })
      .Case<DictType>([&](DictType dictTy) {
        printer << "dict<" << dictTy.getKeyType() << ", "
                << dictTy.getValueType() << ">";
      })
      .Case<ClassType>([&](ClassType classTy) {
        printer << "class<\"" << classTy.getClassName() << "\">";
      })
      .Case<FuncSignatureType>([&](FuncSignatureType sigTy) {
        printer << "funcsig<";
        printTypeList(sigTy.getPositionalTypes());
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
        printer << " -> ";
        printTypeList(sigTy.getResultTypes());
        printer << ">";
      })
      .Case<FuncType>([&](FuncType funcTy) {
        printer << "func<" << funcTy.getSignature() << ">";
      })
      .Case<PrimFuncType>([&](PrimFuncType primTy) {
        printer << "prim.func<" << primTy.getSignature() << ">";
      })
      .Default([&](Type) { llvm_unreachable("unknown py type to print"); });
}

} // namespace py

#include "PyDialect.cpp.inc"

// ODSが生成したOpクラスの定義本体を取り込む
#define GET_OP_CLASSES
#include "PyOps.cpp.inc"
#undef GET_OP_CLASSES
