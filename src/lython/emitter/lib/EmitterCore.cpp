#include "EmitterCore.h"
#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h" // IWYU pragma: keep
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
                             std::string sourceName, EmitOptions options)
    : moduleNode(moduleNode), context(context),
      moduleName(std::move(moduleName)), sourceName(std::move(sourceName)),
      activePackageName(options.mainPackageName), options(options),
      builder(&context), types(context) {
  types.setTargetTriple(this->options.targetTriple);
  if (this->sourceName.empty())
    this->sourceName = this->moduleName;
}

EmitResult ModuleEmitter::emit() {
  context.loadDialect<py::PyDialect, mlir::arith::ArithDialect,
                      mlir::bufferization::BufferizationDialect,
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

  predeclareSourceModules();
  predeclareTopLevel();
  // After class/import predeclaration (signatures may reference user classes
  // and imported names), before any body is typed or emitted.
  types.registerModule(moduleNode);

  // Register module globals after the top-level classes are predeclared (a
  // global's annotation may name a user class) but before any function body
  // is emitted so their reads resolve. Publish their names/types for
  // runtime storage lowering.
  collectModuleGlobals(moduleNode);
  if (!moduleGlobals.empty()) {
    llvm::SmallVector<std::string, 4> globalNames;
    llvm::SmallVector<mlir::Type, 4> globalTypes;
    for (const auto &entry : moduleGlobals) {
      globalNames.push_back(entry.first().str());
      globalTypes.push_back(entry.second);
    }
    module->setAttr("ly.module_global_names", stringArray(builder, globalNames));
    module->setAttr("ly.module_global_types", typeArray(builder, globalTypes));
  }

  emitSourceModuleDeclarations();
  emitTopLevelDeclarations();

  auto mainType = builder.getFunctionType({}, {});
  auto main = mlir::func::FuncOp::create(builder, loc(moduleNode), "__main__",
                                         mainType);
  mlir::Block *entry = main.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  atModuleScope = true;
  emitStatements(ast::nodeList(moduleNode, "body"), /*skipDeclarations=*/true);
  atModuleScope = false;
  if (!insertionBlockTerminated(builder))
    mlir::func::ReturnOp::create(builder, loc(moduleNode));

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
      "ly.source.start_line",
      attrBuilder.getI32IntegerAttr(node.range.start.line)));
  rangeAttrs.push_back(attrBuilder.getNamedAttr(
      "ly.source.start_col",
      attrBuilder.getI32IntegerAttr(node.range.start.column)));
  rangeAttrs.push_back(attrBuilder.getNamedAttr(
      "ly.source.end_line",
      attrBuilder.getI32IntegerAttr(node.range.end.line)));
  rangeAttrs.push_back(attrBuilder.getNamedAttr(
      "ly.source.end_col",
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

bool ModuleEmitter::requireStaticEvidence(
    const parser::Node &anchor, const CallInferenceResult &inference) {
  if (inference)
    return true;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      inference.failureReason.empty()
          ? "operation requires manifest-backed static evidence"
          : inference.failureReason});
  return false;
}

bool ModuleEmitter::requireStaticEvidence(
    const parser::Node &anchor, const AwaitInferenceResult &inference) {
  if (inference)
    return true;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      inference.failureReason.empty()
          ? "await expression requires manifest-backed Awaitable evidence"
          : inference.failureReason});
  return false;
}

bool ModuleEmitter::requireStaticEvidence(
    const parser::Node &anchor, const YieldFromInferenceResult &inference) {
  if (inference)
    return true;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      inference.failureReason.empty()
          ? "yield from requires manifest-backed iterable evidence"
          : inference.failureReason});
  return false;
}

bool ModuleEmitter::requireStaticEvidence(
    const parser::Node &anchor,
    const AsyncIterationInferenceResult &inference) {
  if (inference)
    return true;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      inference.failureReason.empty()
          ? "async for requires manifest-backed AsyncIterable evidence"
          : inference.failureReason});
  return false;
}

bool ModuleEmitter::requireStaticEvidence(
    const parser::Node &anchor,
    const AsyncContextMethodInferenceResult &inference) {
  if (inference)
    return true;
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, anchor.range.start,
      inference.failureReason.empty()
          ? "async context manager operation requires manifest-backed evidence"
          : inference.failureReason});
  return false;
}

mlir::Type ModuleEmitter::boolProtocol() const {
  return types.protocol("Callable");
}

} // namespace lython::emitter
