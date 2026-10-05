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

#include "EmitterPyOps.h"

namespace lython::emitter {

ModuleEmitter::ModuleEmitter(const parser::Node &moduleNode,
                             mlir::MLIRContext &context, std::string moduleName,
                             std::string sourceName, EmitOptions options)
    : moduleNode(moduleNode), context(context),
      moduleName(std::move(moduleName)), sourceName(std::move(sourceName)),
      activePackageName(options.mainPackageName), options(options),
      builder(&context), types(context) {
  types.setTargetTriple(this->options.targetTriple);
  types.setJsHost(this->options.jsHost);
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
  //
  // ⭐ AND ON EVERY IMPORTED MODULE, FIRST. It used to run on the main module
  // alone, so `class Color(Enum)` in a library reached the dialect verifier as
  // "'py.class' op unknown base class 'Enum'" -- the compiler's own sentence
  // for a class CPython has no trouble with, and the same class in the main
  // module works. Imported modules go first so the main module's use rewrite
  // sees their members as well as its own.
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (source.moduleNode && !source.isStub)
      desugarEnumClasses(*source.moduleNode);
  desugarEnumClasses(moduleNode);
  // Before every binder: the parameters this moves into `type_params` are what
  // decides whether a class or a def is generic at all.
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (source.moduleNode && !source.isStub)
      desugarClassicGenerics(*source.moduleNode);
  desugarClassicGenerics(moduleNode);
  // Before any binder reads an imported module's top level: a container
  // constant there is a module GLOBAL, and the binders below hand out its
  // canonical name.
  collectImportedModuleGlobals();
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
  // ⭐ AND THE BASES THAT ARE NOT BARE NAMES. `collectTopLevelBindings` runs
  // before the imports are bound -- it has to, because the spellings it reads
  // decide which symbols the declarations are emitted under -- so a base
  // written `shapes.Shape` could not be resolved there and was recorded as
  // nothing at all. The override guard then saw a class with NO bases:
  //
  //     class Local(shapes.Shape):
  //         def name(self) -> str: return "local"
  //     print(describe(Local(6)))   # printed shape6; CPython prints local6
  //
  // A silent wrong answer -- `Shape.describe` calls `self.name()` and the
  // guard, told the hierarchy had no override, inlined Shape's. Filled here,
  // where the aliases resolve and still before anything is emitted, so the
  // answer stays a property of the module rather than of the position asked.
  resolveTopLevelBaseSpellings();
  // After class/import predeclaration (signatures may reference user classes
  // and imported names), before any body is typed or emitted.
  types.registerModule(moduleNode);

  // Register module globals after the top-level classes are predeclared (a
  // global's annotation may name a user class) but before any function body
  // is emitted so their reads resolve.
  collectModuleGlobals(moduleNode);

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
  // ⭐ The module dunders, which are compile-time constants here. A program
  // compiled by `lyc` IS the main module, so `__name__` is "__main__" and
  // `if __name__ == "__main__":` -- the most common line in Python -- folds
  // to a taken branch. It used to be refused: "unresolved name '__name__'".
  //
  // `__file__` is the source this walk was handed. `__doc__` is not bound:
  // it is the module's docstring, which this walk does not keep, and a wrong
  // constant is worse than an unresolved name.
  auto bindModuleDunder = [&](llvm::StringRef name, llvm::StringRef text) {
    mlir::Type literalType = types.literal(("\"" + text + "\"").str());
    auto constant = py::StrConstantOp::create(builder, loc(moduleNode),
                                              literalType,
                                              builder.getStringAttr(text));
    Value value{constant.getResult(), literalType};
    values[name] = value;
    types.bindSymbol(name, literalType);
  };
  bindModuleDunder("__name__", "__main__");
  bindModuleDunder("__file__", sourceName);
  // Before the main module's first statement, because that is where CPython
  // runs an imported module's class bodies: `import lib` executes lib before
  // the importer continues.
  // Module-level constants first: a class attribute initializer may read one,
  // and CPython runs the module body top to bottom.
  emitImportedModuleGlobalInitializers();
  emitImportedClassAttrInitializers();
  emitStatements(ast::nodeList(moduleNode, "body"), /*skipDeclarations=*/true);
  atModuleScope = false;
  if (!insertionBlockTerminated(builder)) {
    // CPython clears the main module's names at shutdown in the order they
    // were first bound -- not reversed, as a frame's are.
    if (options.keepLocalsAlive)
      for (const std::string &name : frameLocalOrder(moduleNode)) {
        auto bound = values.find(name);
        if (bound != values.end())
          emitKeepAlive(moduleNode, bound->second);
        if (moduleGlobals.count(name))
          emitGlobalClear(moduleNode, name);
      }
    mlir::func::ReturnOp::create(builder, loc(moduleNode));
  }

  EmitResult result;
  // Annotation resolution runs from const contexts (TypeSystem), so its
  // diagnostics (rejected string forward references) surface here.
  for (parser::Diagnostic &diagnostic : types.takeAnnotationDiagnostics())
    diagnostics.push_back(std::move(diagnostic));
  result.diagnostics = std::move(diagnostics);
  result.module = mlir::OwningOpRef<mlir::ModuleOp>(module);
  return result;
}

void ModuleEmitter::resolveTopLevelBaseSpellings() {
  const auto *body = ast::nodeList(moduleNode, "body");
  if (!body)
    return;
  for (const parser::NodePtr &statement : *body) {
    if (!statement || statement->kind != "ClassDef")
      continue;
    auto name = ast::string(*statement, "name");
    if (!name)
      continue;
    const auto *baseNodes = ast::nodeList(*statement, "bases");
    if (!baseNodes)
      continue;
    auto &bases = declaredClassBases[*name];
    bool changed = false;
    // ⭐ AND A BARE NAME THAT IS AN IMPORT, which the dotted repair below left
    // behind: the spelling pass records what the source wrote, so
    // `from shapes import Shape` records "Shape" while the receiver's contract
    // is "shapes.Shape" -- the same hierarchy, invisible to every question
    // keyed on the contract name. `class Square(shapes.Shape)` printed
    // [0, 4] and `from shapes import Shape` / `class Square(Shape)` printed
    // [0, 0] for the same program, with no diagnostic on either.
    for (std::string &recorded : bases) {
      std::string canonical = canonicalClassName(recorded);
      if (canonical.empty() || canonical == recorded ||
          llvm::is_contained(bases, canonical))
        continue;
      recorded = canonical;
      changed = true;
    }
    for (const parser::NodePtr &base : *baseNodes) {
      if (!base || base->kind == "Name")
        continue; // canonicalized in place above
      std::string qualified = ast::qualifiedName(base.get());
      if (qualified.empty())
        continue;
      std::string canonical = canonicalClassName(qualified);
      if (canonical.empty() || llvm::is_contained(bases, canonical))
        continue;
      bases.push_back(canonical);
      changed = true;
    }
    if (changed)
      types.bindDeclaredBases(*name, bases);
  }
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
      // The same hierarchy, where the SUBTYPE questions are asked from: the
      // isinstance analysis reads class ops that do not exist yet for a class
      // declared further down.
      types.bindDeclaredBases(*name, bases);
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

// ⭐ AN IMPORT BINDS THE NAME TOO, and this asked only about the three things
// the module writes itself. `from helpers import range` -- or `len`, `str`,
// `sum`, `max`, `min`, `repr`, `int` -- left every builtin fast path visible,
// so the BUILTIN ran and the imported function was never called:
//
//     # helpers.py: def range(n): return [n, n + 1]
//     from helpers import range
//     for v in range(7): print(v)     # counted 0..6; CPython prints 7, 8
//
// Eight builtins measured wrong the same way, all silently, and the one-file
// spelling of every one of them was already right -- which is what says the
// import binding is the gap and not the shadowing rule.
//
// A canonical binding is what an import leaves behind (`bindCanonicalSymbol`),
// and nothing else in a main module makes one.
bool ModuleEmitter::programBindsName(llvm::StringRef name) const {
  return values.find(name) != values.end() || moduleFunctionNames.count(name) ||
         moduleClassNames.count(name) ||
         types.lookupCanonicalBinding(name).has_value();
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
  if (anchorlessCall == &node)
    rangeAttrs.push_back(attrBuilder.getNamedAttr("ly.source.no_anchor",
                                                  attrBuilder.getUnitAttr()));
  if (!inlineFrames.empty()) {
    rangeAttrs.push_back(attrBuilder.getNamedAttr(
        "ly.source.function",
        attrBuilder.getStringAttr(inlineFrames.back().calleeName)));
    llvm::SmallVector<mlir::Attribute, 4> frames;
    for (const InlineFrame &frame : llvm::reverse(inlineFrames)) {
      llvm::SmallVector<mlir::NamedAttribute, 6> entry;
      entry.push_back(attrBuilder.getNamedAttr(
          "function", attrBuilder.getStringAttr(frame.callerName)));
      entry.push_back(attrBuilder.getNamedAttr(
          "start_line", attrBuilder.getI32IntegerAttr(frame.line)));
      entry.push_back(attrBuilder.getNamedAttr(
          "start_col", attrBuilder.getI32IntegerAttr(frame.column)));
      entry.push_back(attrBuilder.getNamedAttr(
          "end_line", attrBuilder.getI32IntegerAttr(frame.endLine)));
      entry.push_back(attrBuilder.getNamedAttr(
          "end_col", attrBuilder.getI32IntegerAttr(frame.endColumn)));
      if (frame.noAnchor)
        entry.push_back(
            attrBuilder.getNamedAttr("no_anchor", attrBuilder.getUnitAttr()));
      frames.push_back(attrBuilder.getDictionaryAttr(entry));
    }
    rangeAttrs.push_back(attrBuilder.getNamedAttr(
        "ly.source.inline_at", attrBuilder.getArrayAttr(frames)));
  }
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

} // namespace lython::emitter
