#include "EmitterCore.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h" // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <utility>

#include "PyDialect.h.inc"

namespace lython::emitter {

ModuleEmitter::ModuleEmitter(const parser::Node &moduleNode,
                             mlir::MLIRContext &context, std::string moduleName,
                             std::string sourceName)
    : moduleNode(moduleNode), context(context),
      moduleName(std::move(moduleName)), sourceName(std::move(sourceName)),
      builder(&context), types(context) {
  if (this->sourceName.empty())
    this->sourceName = this->moduleName;
}

EmitResult ModuleEmitter::emit() {
  context.loadDialect<py::PyDialect, mlir::arith::ArithDialect,
                      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                      mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                      mlir::tensor::TensorDialect>();
  types.seedBuiltins();

  module = mlir::ModuleOp::create(builder.getUnknownLoc());
  module.setName(moduleName);
  llvm::SmallVector<std::string, 8> staticAttrNames;
  llvm::SmallVector<mlir::Attribute, 8> staticAttrValues;
  collectStaticModuleAssignments(moduleNode, staticAttrNames, staticAttrValues);
  if (!staticAttrNames.empty()) {
    module->setAttr("ly.module_static_attr_names",
                    stringArray(builder, staticAttrNames));
    module->setAttr("ly.module_static_attr_values",
                    builder.getArrayAttr(staticAttrValues));
  }
  builder.setInsertionPointToEnd(module.getBody());

  predeclareTopLevel();
  emitTopLevelDeclarations();

  auto mainType = builder.getFunctionType({}, {});
  auto main =
      builder.create<mlir::func::FuncOp>(loc(moduleNode), "__main__", mainType);
  mlir::Block *entry = main.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  emitStatements(ast::nodeList(moduleNode, "body"), /*skipDeclarations=*/true);
  if (!insertionBlockTerminated(builder))
    builder.create<mlir::func::ReturnOp>(loc(moduleNode));

  EmitResult result;
  result.diagnostics = std::move(diagnostics);
  result.module = mlir::OwningOpRef<mlir::ModuleOp>(module);
  return result;
}

mlir::Location ModuleEmitter::loc(const parser::Node &node) const {
  mlir::Location start = mlir::FileLineColLoc::get(
      &context, sourceName, node.range.start.line, node.range.start.column);
  mlir::Builder attrBuilder(&context);
  llvm::SmallVector<mlir::NamedAttribute, 4> rangeAttrs;
  rangeAttrs.push_back(attrBuilder.getNamedAttr(
      "lython.source.start_line",
      attrBuilder.getI32IntegerAttr(node.range.start.line)));
  rangeAttrs.push_back(attrBuilder.getNamedAttr(
      "lython.source.start_col",
      attrBuilder.getI32IntegerAttr(node.range.start.column)));
  rangeAttrs.push_back(attrBuilder.getNamedAttr(
      "lython.source.end_line",
      attrBuilder.getI32IntegerAttr(node.range.end.line)));
  rangeAttrs.push_back(attrBuilder.getNamedAttr(
      "lython.source.end_col",
      attrBuilder.getI32IntegerAttr(node.range.end.column)));
  return mlir::FusedLoc::get(&context, {start},
                             attrBuilder.getDictionaryAttr(rangeAttrs));
}

mlir::Type ModuleEmitter::callableProtocol() const {
  return types.protocol("Callable");
}

mlir::Type ModuleEmitter::callProtocolFor(mlir::Type calleeType) const {
  if (calleeType && py::isPyProtocolType(calleeType))
    return calleeType;
  return callableProtocol();
}

mlir::Type ModuleEmitter::callProtocolFor(const CallInferenceResult &inference,
                                          mlir::Type fallback) const {
  if (inference.evidence.callableContract &&
      py::isPyProtocolType(inference.evidence.callableContract))
    return inference.evidence.callableContract;
  return callProtocolFor(fallback);
}

mlir::Type ModuleEmitter::boolProtocol() const {
  return types.protocol("Callable");
}

mlir::Type ModuleEmitter::coroutineType(mlir::Type resultType) const {
  return types.contract("types.CoroutineType",
                        {types.object(), types.object(),
                         resultType ? resultType : types.object()});
}

FunctionSignature
ModuleEmitter::asyncPublicSignature(FunctionSignature sig) const {
  sig.resultType = coroutineType(sig.resultType);
  types.refreshCallable(sig);
  return sig;
}

} // namespace lython::emitter
