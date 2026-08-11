#include "EmitterCore.h"
#include "EmitterSupport.h"
#include "PyProtocols.h"

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
  types.setGenericClassResolver(
      [this](llvm::StringRef baseName, mlir::ArrayRef<mlir::Type> arguments) {
        return ensureGenericClassSpecialization(baseName, arguments);
      });

  module = mlir::ModuleOp::create(builder.getUnknownLoc());
  module.setName(moduleName);
  // Enum desugaring rewrites the parsed tree, so it must run before anything
  // reads the module's shape (static module attributes below already do).
  desugarEnumClasses(moduleNode);
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

  // Before predeclaration: the top-level `def`/`class` spellings decide which
  // builtin fast paths may fire and which symbols the declarations are emitted
  // under, and predeclareTopLevel already binds imports and classes.
  collectTopLevelBindings();
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

  // Generic class instantiations may now be emitted as they are demanded.
  // Everything registerModule's fixpoint allocated (a parameter annotated
  // `C[int]`) waited in the queue: the fixpoint reruns, so emitting from it
  // would duplicate, and no top-level environment existed yet.
  genericClassEmissionReady = true;
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
  // Annotation resolution runs from const contexts (TypeSystem), so its
  // diagnostics (rejected string forward references) surface here.
  for (parser::Diagnostic &diagnostic : types.takeAnnotationDiagnostics())
    diagnostics.push_back(std::move(diagnostic));
  result.diagnostics = std::move(diagnostics);
  result.module = mlir::OwningOpRef<mlir::ModuleOp>(module);
  return result;
}

void ModuleEmitter::collectTopLevelBindings() {
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;
  const py::protocols::Table &table = py::protocols::Table::get(context);
  for (const parser::NodePtr &statement : *body) {
    if (!statement)
      continue;
    bool isFunction = statement->kind == "FunctionDef" ||
                      statement->kind == "AsyncFunctionDef";
    if (!isFunction && statement->kind != "ClassDef")
      continue;
    auto name = ast::string(*statement, "name");
    if (!name)
      continue;
    if (!isFunction) {
      moduleClassNames.insert(*name);
      // ⭐ The hierarchy, recorded BEFORE anything is emitted. The maps the
      // class emission fills are built as each ClassDef is reached, so a
      // question asked from a function body above a subclass got the answer
      // "no subclass" and the override guard let a silent wrong dispatch
      // through -- moving `class B` up flipped the same program to a refusal.
      // Whether a hierarchy has an override is a property of the module, not
      // of where in it the question is asked.
      auto &bases = declaredClassBases[*name];
      if (const auto *baseNodes = ast::nodeList(*statement, "bases"))
        for (const parser::NodePtr &base : *baseNodes)
          if (base && base->kind == "Name")
            bases.push_back(std::string(ast::nameSpelling(*base)));
      auto &methods = declaredClassMethods[*name];
      auto &attributes = declaredClassAttributes[*name];
      if (const auto *classBody = ast::nodeList(*statement, "body"))
        for (const parser::NodePtr &member : *classBody) {
          if (!member)
            continue;
          if (member->kind == "FunctionDef" ||
              member->kind == "AsyncFunctionDef") {
            if (auto methodName = ast::string(*member, "name"))
              methods.insert(*methodName);
            continue;
          }
          // A class-level binding is shadowed by a subclass exactly the way a
          // method is overridden, and reading it through a base-typed
          // reference is the same unresolvable dispatch.
          if (member->kind == "AnnAssign") {
            if (const parser::Node *target = ast::node(*member, "target"))
              if (target->kind == "Name")
                attributes.insert(ast::nameSpelling(*target));
            continue;
          }
          if (member->kind == "Assign")
            if (const auto *targets = ast::nodeList(*member, "targets"))
              for (const parser::NodePtr &target : *targets)
                if (target && target->kind == "Name")
                  attributes.insert(ast::nameSpelling(*target));
        }
      continue;
    }
    moduleFunctionNames.insert(*name);
    // The manifest is the authority on which spellings it owns as builtin
    // bindings: asking it, rather than carrying a hand-written list, keeps the
    // set from drifting when a builtin is added to or removed from
    // runtime/modules/*.mlir.
    if (table.freeFunctionContract((llvm::Twine("builtins.") + *name).str()))
      shadowedBuiltinSymbols[*name] = (llvm::Twine(*name) + "$user").str();
  }
}

bool ModuleEmitter::programBindsName(llvm::StringRef name) const {
  return values.find(name) != values.end() || moduleFunctionNames.count(name) ||
         moduleClassNames.count(name);
}

llvm::StringRef
ModuleEmitter::topLevelFunctionSymbol(llvm::StringRef name) const {
  auto found = shadowedBuiltinSymbols.find(name);
  if (found == shadowedBuiltinSymbols.end())
    return name;
  return found->second;
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
