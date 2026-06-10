#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <utility>

#include "PyDialect.h.inc"

namespace lython::emitter {

Builder::Impl::Impl(mlir::MLIRContext &context, std::string moduleName)
    : moduleName(std::move(moduleName)), context(context), builder(&context) {
  context.loadDialect<py::PyDialect, mlir::arith::ArithDialect,
                      mlir::async::AsyncDialect, mlir::cf::ControlFlowDialect,
                      mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                      mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
}

EmitResult Builder::Impl::emit(const parser::Node &moduleNode) {
  if (moduleNode.kind != "Module") {
    error(moduleNode, "C++ emitter expected a Module root");
    return finish();
  }

  scanStaticImports(moduleNode);
  scanClasses(moduleNode);
  scanFunctions(moduleNode);
  module = mlir::ModuleOp::create(loc(), moduleName);
  builder.setInsertionPointToEnd(module->getBody());
  emitPrelude();
  emitMain(moduleNode);
  emitClassDefs(moduleNode);
  emitUserFunctions(moduleNode);

  EmitResult result;
  result.module = std::move(module);
  result.diagnostics = std::move(diagnostics);
  return result;
}

mlir::Location Builder::Impl::loc() { return builder.getUnknownLoc(); }
mlir::Location Builder::Impl::loc(const parser::Node &node) {
  int line = node.range.start.line > 0 ? node.range.start.line : 1;
  int column = node.range.start.column >= 0 ? node.range.start.column : 0;
  return mlir::FileLineColLoc::get(&context, moduleName,
                                   static_cast<unsigned>(line),
                                   static_cast<unsigned>(column));
}

Builder::Impl::NameBindingSnapshot
Builder::Impl::snapshotNameBinding(llvm::StringRef name) const {
  NameBindingSnapshot snapshot;
  std::string key = name.str();
  if (auto found = symbols.find(key); found != symbols.end())
    snapshot.symbol = found->second;
  if (auto found = callableAliases.find(key); found != callableAliases.end())
    snapshot.callableAlias = found->second;
  if (auto found = primitiveConstants.find(key);
      found != primitiveConstants.end())
    snapshot.primitiveConstant = found->second;
  return snapshot;
}

void Builder::Impl::restoreNameBinding(llvm::StringRef name,
                                       NameBindingSnapshot snapshot) {
  std::string key = name.str();
  if (snapshot.symbol)
    symbols[key] = *snapshot.symbol;
  else
    symbols.erase(key);

  if (snapshot.callableAlias)
    callableAliases[key] = *snapshot.callableAlias;
  else
    callableAliases.erase(key);

  if (snapshot.primitiveConstant)
    primitiveConstants[key] = *snapshot.primitiveConstant;
  else
    primitiveConstants.erase(key);
}

void Builder::Impl::bindTemporaryName(llvm::StringRef name, Value value) {
  std::string key = name.str();
  symbols[key] = value;
  callableAliases.erase(key);
  primitiveConstants.erase(key);
}

EmitResult Builder::Impl::finish() {
  EmitResult result;
  result.diagnostics = std::move(diagnostics);
  return result;
}

void Builder::Impl::error(const parser::Node &node, std::string message) {
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, node.range.start, std::move(message)});
}

mlir::ArrayAttr
Builder::Impl::stringArrayAttr(llvm::ArrayRef<std::string> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (const std::string &value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

mlir::ArrayAttr
Builder::Impl::typeArrayAttr(llvm::ArrayRef<mlir::Type> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (mlir::Type value : values)
    attrs.push_back(mlir::TypeAttr::get(value));
  return builder.getArrayAttr(attrs);
}

bool Builder::Impl::isBuiltinExceptionClass(llvm::StringRef name) const {
  return name == "BaseException" || name == "Exception" ||
         name == "RuntimeError" || name == "TypeError" ||
         name == "ValueError" || name == "KeyError" || name == "IndexError" ||
         name == "AssertionError";
}

bool Builder::Impl::isNativeDecorator(const parser::Node &node) {
  const parser::Node *call = &node;
  if (node.kind == "Name") {
    std::optional<std::string> name = lyrtBuiltinName(node);
    return name && *name == "native";
  }
  if (node.kind != "Call")
    return false;

  const parser::NodePtr *func = nodeField(node, "func");
  if (!func || !*func)
    return false;
  std::optional<std::string> name = lyrtBuiltinName(**func);
  if (!name || *name != "native")
    return false;

  const std::vector<parser::NodePtr> *args = nodeListField(*call, "args");
  if (args && !args->empty()) {
    error(node, "@native positional arguments are not supported");
    return true;
  }

  const std::vector<parser::NodePtr> *keywords =
      nodeListField(*call, "keywords");
  if (!keywords)
    return true;
  for (const parser::NodePtr &keyword : *keywords) {
    if (!keyword)
      continue;
    const std::string *arg = stringField(*keyword, "arg");
    const parser::NodePtr *value = nodeField(*keyword, "value");
    if (!arg || *arg != "gc" || !value || !*value ||
        (*value)->kind != "Constant") {
      error(*keyword, "@native currently accepts only gc=\"none\"");
      continue;
    }
    const parser::FieldValue *fieldValue = valueField(**value, "value");
    const auto *text =
        fieldValue ? std::get_if<std::string>(fieldValue) : nullptr;
    if (!text || *text != "none")
      error(**value, "@native currently accepts only gc=\"none\"");
  }
  return true;
}

py::FuncSignatureType Builder::Impl::functionSignatureType(
    llvm::ArrayRef<mlir::Type> argTypes, mlir::Type resultType,
    mlir::Type varargType, llvm::ArrayRef<mlir::Type> kwonlyTypes,
    mlir::Type kwargType) {
  llvm::SmallVector<mlir::Type> results{resultType};
  return py::FuncSignatureType::get(&context, argTypes, kwonlyTypes, varargType,
                                    kwargType, results);
}

mlir::FunctionType
Builder::Impl::asyncFunctionType(llvm::ArrayRef<mlir::Type> argTypes,
                                 mlir::Type resultType) {
  llvm::SmallVector<mlir::Type> inputs(argTypes.begin(), argTypes.end());
  inputs.push_back(exceptionCellType());
  mlir::Type asyncResult = mlir::async::ValueType::get(resultType);
  return mlir::FunctionType::get(&context, inputs, {asyncResult});
}

py::ClassOp Builder::Impl::createClass(llvm::StringRef name,
                                       mlir::ArrayAttr baseNames) {
  auto op = builder.create<py::ClassOp>(loc(), name, baseNames,
                                        mlir::ArrayAttr{}, mlir::ArrayAttr{});
  op.getBody().emplaceBlock();
  return op;
}

py::FuncOp Builder::Impl::createFunc(llvm::StringRef name,
                                     py::FuncSignatureType signature,
                                     mlir::ArrayAttr argNames, bool hasVararg,
                                     bool hasKwarg, bool mayThrow,
                                     mlir::ArrayAttr kwonlyNames,
                                     mlir::ArrayAttr closureTypes) {
  return builder.create<py::FuncOp>(
      loc(), name, signature,
      hasVararg ? builder.getUnitAttr() : mlir::UnitAttr{},
      hasKwarg ? builder.getUnitAttr() : mlir::UnitAttr{}, argNames,
      kwonlyNames, closureTypes, mlir::UnitAttr{},
      mayThrow ? builder.getUnitAttr() : mlir::UnitAttr{},
      mayThrow ? mlir::UnitAttr{} : builder.getUnitAttr());
}

void Builder::Impl::addEntryBlock(py::FuncOp func,
                                  llvm::ArrayRef<mlir::Type> argTypes) {
  mlir::Block *block = new mlir::Block();
  func.getBody().push_back(block);
  for (mlir::Type type : argTypes)
    block->addArgument(type, loc());
  builder.setInsertionPointToStart(block);
}

void Builder::Impl::addEntryBlock(mlir::func::FuncOp func) {
  mlir::Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);
}

void Builder::Impl::addAsyncEntryBlock(mlir::Operation *func,
                                       llvm::ArrayRef<mlir::Type> argTypes) {
  mlir::Region &body = func->getRegion(0);
  mlir::Block *block = new mlir::Block();
  body.push_back(block);
  for (mlir::Type type : argTypes)
    block->addArgument(type, loc());
  block->addArgument(exceptionCellType(), loc());
  builder.setInsertionPointToStart(block);
}

void Builder::Impl::emitPrelude() {
  createClass("BaseException");
  createClass("Exception", stringArrayAttr({"BaseException"}));
  for (llvm::StringRef name : {"RuntimeError", "TypeError", "ValueError",
                               "KeyError", "IndexError", "AssertionError"})
    createClass(name, stringArrayAttr({"Exception"}));

  py::TupleType varargType = py::TupleType::get(&context, {strType()});
  py::FuncSignatureType signature =
      functionSignatureType({}, noneType(), varargType);
  for (llvm::StringRef name : {"__builtin_print", "__builtin_print_raw"}) {
    builder.setInsertionPointToEnd(module->getBody());
    py::FuncOp builtin = createFunc(name, signature, {}, /*hasVararg=*/true,
                                    /*hasKwarg=*/false);
    addEntryBlock(builtin, {varargType});
    mlir::Value none = builder.create<py::NoneOp>(loc(), noneType());
    builder.create<py::ReturnOp>(loc(), mlir::ValueRange{none});
  }

  builder.setInsertionPointToEnd(module->getBody());
}

} // namespace lython::emitter
