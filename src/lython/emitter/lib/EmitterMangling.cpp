#include "EmitterCore.h"
#include "EmitterPyOps.h"

#include "AstAccess.h"
#include "Mangling.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringMap.h"

#include <functional>

namespace lython::emitter {

// CPython's `__qualname__` is lexical: a def or a class is named by the
// scopes written around it -- `C.m` in a class body, `f.<locals>.g` in a
// function's -- and a lambda is `<lambda>` in whichever of those it is
// written. What a def's decorators, defaults and annotations evaluate in is
// the scope AROUND it, so only its body takes the new prefix.
// ⛔ A comprehension adds no scope (PEP 709 inlines it); a generator
// expression does, as `<genexpr>`.
void ModuleEmitter::recordPythonDefinitionNames(const parser::Node &moduleNode,
                                                llvm::StringRef moduleName) {
  std::function<void(const parser::Node &, const std::string &)> visit;
  // Every child in `outer`, but the field `innerField` in `inner`.
  auto visitChildren = [&](const parser::Node &node, const std::string &outer,
                           const std::string &inner,
                           llvm::StringRef innerField) {
    for (const parser::Field &field : node.fields) {
      const std::string &prefix = field.name == innerField ? inner : outer;
      if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (*child)
          visit(**child, prefix);
        continue;
      }
      if (const auto *children =
              std::get_if<std::vector<parser::NodePtr>>(&field.value))
        for (const parser::NodePtr &child : *children)
          if (child)
            visit(*child, prefix);
    }
  };
  visit = [&](const parser::Node &node, const std::string &prefix) {
    llvm::StringRef kind = node.kind;
    bool function = kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
                    kind == "Lambda";
    if (function || kind == "ClassDef") {
      std::string name;
      if (kind == "Lambda")
        name = "<lambda>";
      else if (auto override = pythonNameOverrides.find(&node);
               override != pythonNameOverrides.end())
        name = override->second;
      else
        name = std::string(ast::string(node, "name").value_or(""));
      // A class written in a function is named where it was written.
      std::string qualname = prefix + name;
      if (auto hoisted = hoistedClassPrefixes.find(&node);
          hoisted != hoistedClassPrefixes.end())
        qualname = hoisted->second + name;
      pythonDefinitionNames[&node] =
          PythonDefinitionName{moduleName.str(), qualname, name};
      visitChildren(
          node, hoistedClassPrefixes.count(&node) ? std::string() : prefix,
          function ? qualname + ".<locals>." : qualname + ".", "body");
      return;
    }
    if (kind == "GeneratorExp") {
      // Its first iterable is evaluated outside it, everything else inside.
      std::string inner = prefix + "<genexpr>.<locals>.";
      if (const parser::Node *element = ast::node(node, "elt"))
        visit(*element, inner);
      if (const auto *generators = ast::nodeList(node, "generators"))
        for (auto [index, generator] : llvm::enumerate(*generators))
          if (generator)
            visitChildren(*generator, inner, index == 0 ? prefix : inner,
                          "iter");
      return;
    }
    visitChildren(node, prefix, prefix, "");
  };
  visit(moduleNode, "");
}

// Every function emitted for a definition of the program takes its mangled
// symbol (common/Mangling.h), and every reference to it the new name: a
// binding names a function by its symbol, a `@native` body calls one, a class
// lists its methods' symbols, and a default's provider is named in its
// function's defaults.
// ⛔ AT THE END, and not where each symbol is chosen: the emitter names
// functions in a dozen places and reads those names back in more, and none of
// that changes -- only what the module says when it is handed on.
// ⛔ Not by MLIR's symbol-use walk: a py op names the MANIFEST method it
// stands for with a symbol reference (`target = @__add__`), so a program
// function called `__add__` would take every addition with it. The program
// names its own functions only by the strings rewritten here.
// ⛔ Not a runtime-internal module: its functions ARE the C entry points the
// runtime calls by name (`LyRt_InstallStackGuard`).
void ModuleEmitter::manglePythonDefinitionSymbols() {
  if (options.runtimeInternal || emittedPythonDefinitions.empty())
    return;
  llvm::StringMap<std::string> renamed;
  llvm::StringMap<unsigned> definitionsSoFar;
  for (const EmittedPythonDefinition &definition : emittedPythonDefinitions) {
    if (renamed.count(definition.symbol))
      continue;
    std::string symbol =
        py::mangling::pythonDefinitionSymbol(definition.module,
                                             definition.qualname);
    unsigned ordinal = definitionsSoFar[symbol]++;
    renamed[definition.symbol] =
        ordinal == 0 ? symbol
                     : py::mangling::pythonDefinitionSymbol(
                           definition.module, definition.qualname, ordinal);
  }
  auto renamedTo = [&](llvm::StringRef symbol) -> std::optional<std::string> {
    auto found = renamed.find(symbol);
    if (found == renamed.end())
      return std::nullopt;
    return found->second;
  };
  mlir::Builder attributes(&context);
  module.walk([&](mlir::Operation *op) {
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
      if (std::optional<std::string> symbol =
              renamedTo(function.getSymName()))
        function.setSymName(*symbol);
      // A default whose provider is a definition (a lambda written as the
      // default) is named in the defaults of the function that has it.
      if (auto defaults =
              function->getAttrOfType<mlir::ArrayAttr>("callable_default_values")) {
        llvm::SmallVector<mlir::Attribute, 4> rewritten;
        bool changed = false;
        for (mlir::Attribute entry : defaults) {
          auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
          auto kind = dict ? dict.getAs<mlir::StringAttr>("kind")
                           : mlir::StringAttr();
          auto value = dict ? dict.getAs<mlir::StringAttr>("value")
                            : mlir::StringAttr();
          std::optional<std::string> symbol;
          if (kind && kind.getValue() == "provider" && value)
            symbol = renamedTo(value.getValue());
          if (!symbol) {
            rewritten.push_back(entry);
            continue;
          }
          llvm::SmallVector<mlir::NamedAttribute, 4> fields(dict.begin(),
                                                            dict.end());
          for (mlir::NamedAttribute &field : fields)
            if (field.getName() == "value")
              field.setValue(attributes.getStringAttr(*symbol));
          rewritten.push_back(attributes.getDictionaryAttr(fields));
          changed = true;
        }
        if (changed)
          function->setAttr("callable_default_values",
                            attributes.getArrayAttr(rewritten));
      }
      return;
    }
    // A `@native` body calls its peers directly.
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      if (std::optional<std::string> symbol = renamedTo(call.getCallee()))
        call.setCallee(*symbol);
      return;
    }
    if (auto binding = mlir::dyn_cast<py::BindingRefOp>(op)) {
      if (std::optional<std::string> symbol =
              renamedTo(binding.getBinding()))
        binding.setBindingAttr(attributes.getStringAttr(*symbol));
      return;
    }
    // A function-docstring dispatcher's test names its candidate the same way.
    if (auto test = mlir::dyn_cast<py::TargetIsOp>(op)) {
      if (std::optional<std::string> symbol = renamedTo(test.getTarget()))
        test.setTargetAttr(attributes.getStringAttr(*symbol));
      return;
    }
    if (auto classOp = mlir::dyn_cast<py::ClassOp>(op))
      if (auto symbols =
              classOp->getAttrOfType<mlir::ArrayAttr>("method_symbols")) {
        llvm::SmallVector<mlir::Attribute, 8> rewritten;
        bool changed = false;
        for (mlir::Attribute entry : symbols) {
          auto symbol = mlir::dyn_cast<mlir::StringAttr>(entry);
          std::optional<std::string> to =
              symbol ? renamedTo(symbol.getValue()) : std::nullopt;
          rewritten.push_back(to ? attributes.getStringAttr(*to) : entry);
          changed = changed || to.has_value();
        }
        if (changed)
          classOp->setAttr("method_symbols",
                           attributes.getArrayAttr(rewritten));
      }
  });
}

} // namespace lython::emitter
