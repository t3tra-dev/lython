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

namespace py {

void PyDialect::initialize() {
  addTypes<IntType, FloatType, BoolType, StrType, NoneType, TupleType, DictType,
           ListType, ClassType, ExceptionType, ExceptionCellType, TracebackType,
           LocationType, FuncSignatureType, FuncType, PrimFuncType,
           CoroutineType, TaskType, FutureType>();

  addOperations<
#define GET_OP_LIST
#include "PyOps.cpp.inc"
      >();
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

  if (keyword == "int")
    return IntType::get(ctx);
  if (keyword == "float")
    return FloatType::get(ctx);
  if (keyword == "bool")
    return BoolType::get(ctx);
  if (keyword == "str")
    return StrType::get(ctx);
  if (keyword == "none")
    return NoneType::get(ctx);
  if (keyword == "tuple") {
    if (parser.parseLess())
      return mlir::Type();
    llvm::SmallVector<mlir::Type> elementTypes;
    if (mlir::succeeded(parser.parseOptionalGreater()))
      return TupleType::get(ctx, elementTypes);
    do {
      mlir::Type element;
      if (parser.parseType(element))
        return mlir::Type();
      elementTypes.push_back(element);
    } while (mlir::succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater())
      return mlir::Type();
    return TupleType::get(ctx, elementTypes);
  }
  if (keyword == "dict") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type keyType, valueType;
    if (parser.parseType(keyType) || parser.parseComma() ||
        parser.parseType(valueType) || parser.parseGreater())
      return mlir::Type();
    return DictType::get(ctx, keyType, valueType);
  }
  if (keyword == "list") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type elementType;
    if (parser.parseType(elementType) || parser.parseGreater())
      return mlir::Type();
    return ListType::get(ctx, elementType);
  }
  if (keyword == "coro") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type resultType;
    if (parser.parseType(resultType) || parser.parseGreater())
      return mlir::Type();
    return CoroutineType::get(ctx, resultType);
  }
  if (keyword == "task") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type resultType;
    if (parser.parseType(resultType) || parser.parseGreater())
      return mlir::Type();
    return TaskType::get(ctx, resultType);
  }
  if (keyword == "future") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type resultType;
    if (parser.parseType(resultType) || parser.parseGreater())
      return mlir::Type();
    return FutureType::get(ctx, resultType);
  }
  if (keyword == "class") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::StringAttr classNameAttr;
    if (parser.parseAttribute(classNameAttr) || parser.parseGreater())
      return mlir::Type();
    return ClassType::get(ctx, classNameAttr.getValue());
  }
  if (keyword == "exception")
    return ExceptionType::get(ctx);
  if (keyword == "exception_cell")
    return ExceptionCellType::get(ctx);
  if (keyword == "traceback")
    return TracebackType::get(ctx);
  if (keyword == "location")
    return LocationType::get(ctx);
  if (keyword == "func") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type signatureType;
    if (parser.parseType(signatureType) || parser.parseGreater())
      return mlir::Type();
    auto signature = mlir::dyn_cast<FuncSignatureType>(signatureType);
    if (!signature) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected FuncSignatureType after 'func<'");
      return mlir::Type();
    }
    return FuncType::get(ctx, signature);
  }
  if (keyword == "funcsig") {
    if (parser.parseLess())
      return mlir::Type();
    llvm::SmallVector<mlir::Type, 4> positionalTypes;
    llvm::SmallVector<mlir::Type, 4> kwonlyTypes;
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    mlir::Type varargType;
    mlir::Type kwargsType;
    bool varargSeen = false;
    bool kwonlySeen = false;
    bool kwargsSeen = false;

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
      parser.emitError(parser.getCurrentLocation(),
                       "unexpected token '" + section +
                           "' in funcsig type declaration");
      return mlir::Type();
    }

    if (parser.parseArrow())
      return mlir::Type();
    if (mlir::failed(parseTypeList(resultTypes)) || parser.parseGreater())
      return mlir::Type();

    return FuncSignatureType::get(ctx, positionalTypes, kwonlyTypes, varargType,
                                  kwargsType, resultTypes);
  }
  if (keyword == "prim.func") {
    if (parser.parseLess())
      return mlir::Type();
    mlir::Type signatureType;
    if (parser.parseType(signatureType) || parser.parseGreater())
      return mlir::Type();
    auto signature = mlir::dyn_cast<mlir::FunctionType>(signatureType);
    if (!signature) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected FunctionType after 'prim.func<'");
      return mlir::Type();
    }
    return PrimFuncType::get(ctx, signature);
  }

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

  llvm::TypeSwitch<mlir::Type>(type)
      .Case<IntType>([&](IntType) { printer << "int"; })
      .Case<FloatType>([&](FloatType) { printer << "float"; })
      .Case<BoolType>([&](BoolType) { printer << "bool"; })
      .Case<StrType>([&](StrType) { printer << "str"; })
      .Case<NoneType>([&](NoneType) { printer << "none"; })
      .Case<TupleType>([&](TupleType tupleTy) {
        printer << "tuple<";
        auto elements = tupleTy.getElementTypes();
        llvm::interleaveComma(elements, printer,
                              [&](mlir::Type element) { printer << element; });
        printer << ">";
      })
      .Case<DictType>([&](DictType dictTy) {
        printer << "dict<" << dictTy.getKeyType() << ", "
                << dictTy.getValueType() << ">";
      })
      .Case<ListType>([&](ListType listTy) {
        printer << "list<" << listTy.getElementType() << ">";
      })
      .Case<CoroutineType>([&](CoroutineType coroTy) {
        printer << "coro<" << coroTy.getResultType() << ">";
      })
      .Case<TaskType>([&](TaskType taskTy) {
        printer << "task<" << taskTy.getResultType() << ">";
      })
      .Case<FutureType>([&](FutureType futureTy) {
        printer << "future<" << futureTy.getResultType() << ">";
      })
      .Case<ClassType>([&](ClassType classTy) {
        printer << "class<\"" << classTy.getClassName() << "\">";
      })
      .Case<ExceptionType>([&](ExceptionType) { printer << "exception"; })
      .Case<ExceptionCellType>(
          [&](ExceptionCellType) { printer << "exception_cell"; })
      .Case<TracebackType>([&](TracebackType) { printer << "traceback"; })
      .Case<LocationType>([&](LocationType) { printer << "location"; })
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
      .Default(
          [&](mlir::Type) { llvm_unreachable("unknown py type to print"); });
}

} // namespace py

#include "PyDialect.cpp.inc"

// ODSが生成したOpクラスの定義本体を取り込む
#define GET_OP_CLASSES
#include "PyOps.cpp.inc"
#undef GET_OP_CLASSES
