#include "EmitterSupport.h"

#include "AstAccess.h"
#include "EmitterPyOps.h"

#include "ExceptionTaxonomy.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <utility>

namespace lython::emitter {
extern const llvm::StringLiteral kCallableVarargValueTypeAttr{
    "callable_vararg_value_type"};
extern const llvm::StringLiteral kCallableKwargValueTypeAttr{
    "callable_kwarg_value_type"};
extern const llvm::StringLiteral kPackUnpackedOperandsAttr{
    "ly.unpack_operands"};

bool isPrimitiveOnlyCallable(py::CallableType callable) {
  if (!callable || callable.hasVararg() || callable.hasKwarg() ||
      callable.getResultTypes().size() != 1)
    return false;
  auto isPrimitive = [](mlir::Type type) {
    return type && !py::isPyType(type);
  };
  return llvm::all_of(callable.getPositionalTypes(), isPrimitive) &&
         llvm::all_of(callable.getKwOnlyTypes(), isPrimitive) &&
         isPrimitive(callable.getResultTypes().front());
}

bool isSourceDefinedContract(mlir::Type type) {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(type);
  if (!contract)
    return false;
  llvm::StringRef name = contract.getContractName();
  return !name.contains('.') && !name.starts_with("$");
}

bool isAssignableWithStaticEvidence(mlir::Type actual, mlir::Type expected,
                                    mlir::Operation *from) {
  if (from && isSourceDefinedContract(actual) &&
      isSourceDefinedContract(expected))
    return py::isAssignableTo(actual, expected, from);
  return py::isAssignableTo(actual, expected);
}

mlir::ArrayAttr stringArray(mlir::Builder &builder,
                            llvm::ArrayRef<std::string> values) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  for (const std::string &value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

mlir::ArrayAttr stringArray(mlir::Builder &builder,
                            llvm::ArrayRef<llvm::StringRef> values) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  for (llvm::StringRef value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

mlir::ArrayAttr typeArray(mlir::Builder &builder,
                          llvm::ArrayRef<mlir::Type> values) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  for (mlir::Type value : values)
    attrs.push_back(mlir::TypeAttr::get(value));
  return builder.getArrayAttr(attrs);
}

mlir::ArrayAttr boolArray(mlir::Builder &builder, llvm::ArrayRef<char> values) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  attrs.reserve(values.size());
  for (char value : values)
    attrs.push_back(builder.getBoolAttr(value != 0));
  return builder.getArrayAttr(attrs);
}

mlir::Type replaceSelfType(mlir::Type type, mlir::Type selfType) {
  if (!type || !selfType)
    return type;
  return py::mapPyTypeStructure(
      type, [&](mlir::Type node) -> std::optional<mlir::Type> {
        if (mlir::isa<py::SelfType>(node))
          return selfType;
        return std::nullopt;
      });
}

void replaceSelfInSignature(FunctionSignature &sig, mlir::Type selfType,
                            TypeSystem &types) {
  for (mlir::Type &type : sig.positionalTypes)
    type = replaceSelfType(type, selfType);
  for (mlir::Type &type : sig.kwOnlyTypes)
    type = replaceSelfType(type, selfType);
  sig.varargType = replaceSelfType(sig.varargType, selfType);
  sig.callableVarargType = replaceSelfType(sig.callableVarargType, selfType);
  sig.kwargType = replaceSelfType(sig.kwargType, selfType);
  sig.resultType = replaceSelfType(sig.resultType, selfType);
  types.refreshCallable(sig);
}

bool anyTrue(llvm::ArrayRef<char> values) {
  return llvm::any_of(values, [](char value) { return value != 0; });
}

std::string methodKind(const parser::Node &function) {
  if (const auto *decorators = ast::nodeList(function, "decorator_list")) {
    for (const parser::NodePtr &decorator : *decorators) {
      llvm::StringRef name = ast::nameSpelling(*decorator);
      if (name == "staticmethod")
        return "static";
      if (name == "classmethod")
        return "class";
    }
  }
  return "instance";
}

bool isTopLevelDecl(const parser::Node &node) {
  return node.kind == "FunctionDef" || node.kind == "AsyncFunctionDef" ||
         node.kind == "ClassDef";
}

std::string importBindingName(std::string_view module,
                              std::optional<std::string_view> asname) {
  if (asname)
    return std::string(*asname);
  std::string_view::size_type dot = module.find('.');
  return std::string(module.substr(0, dot));
}

mlir::Attribute defaultValueAttr(mlir::Builder &builder,
                                 const parser::Node *node) {
  if (!node)
    return builder.getUnitAttr();

  auto dict = [&](llvm::StringRef kind, mlir::Attribute value = {}) {
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
    attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr(kind)));
    if (value)
      attrs.push_back(builder.getNamedAttr("value", value));
    return builder.getDictionaryAttr(attrs);
  };

  if (node->kind != "Constant")
    return dict("unsupported", builder.getStringAttr(node->kind));
  if (ast::isNoneField(*node, "value"))
    return dict("none");
  if (auto value = ast::boolean(*node, "value"))
    return dict("bool", builder.getBoolAttr(*value));
  if (auto value = ast::integer(*node, "value"))
    return dict("int", builder.getStringAttr(std::to_string(*value)));
  if (auto value = ast::floating(*node, "value"))
    return dict("float", builder.getF64FloatAttr(*value));
  if (auto value = ast::string(*node, "value"))
    return dict("str", builder.getStringAttr(*value));
  if (const auto *value = ast::bytes(*node, "value"))
    return dict("bytes",
                builder.getStringAttr(llvm::StringRef(
                    reinterpret_cast<const char *>(value->data()),
                    value->size())));
  if (const auto *fieldValue = ast::field(*node, "value"))
    if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
      return dict("int", builder.getStringAttr(big->decimal));
  return dict("unsupported", builder.getStringAttr("Constant"));
}

llvm::SmallVector<const parser::Node *, 8>
positionalArgumentNodes(const parser::Node &arguments) {
  llvm::SmallVector<const parser::Node *, 8> result;
  if (const auto *posOnly = ast::nodeList(arguments, "posonlyargs"))
    for (const parser::NodePtr &arg : *posOnly)
      if (arg)
        result.push_back(arg.get());
  if (const auto *args = ast::nodeList(arguments, "args"))
    for (const parser::NodePtr &arg : *args)
      if (arg)
        result.push_back(arg.get());
  return result;
}

bool blockHasTerminator(mlir::Block &block) {
  return !block.empty() && block.back().hasTrait<mlir::OpTrait::IsTerminator>();
}

mlir::Operation *blockTerminator(mlir::Block &block) {
  return blockHasTerminator(block) ? &block.back() : nullptr;
}

void setInsertionBeforeTerminator(mlir::OpBuilder &builder,
                                  mlir::Block &block) {
  if (mlir::Operation *terminator = blockTerminator(block)) {
    builder.setInsertionPoint(terminator);
    return;
  }
  builder.setInsertionPointToEnd(&block);
}

bool insertionBlockTerminated(const mlir::OpBuilder &builder) {
  mlir::Block *block = builder.getInsertionBlock();
  if (!block)
    return false;
  auto insertionPoint = builder.getInsertionPoint();
  if (insertionPoint == block->begin())
    return false;
  auto previous = insertionPoint;
  --previous;
  return previous->hasTrait<mlir::OpTrait::IsTerminator>();
}

// ⭐ ONE WALK FOR EVERY "does this subtree contain such a statement" QUESTION.
// Four of them had grown -- return, break-or-continue, continue, and the loop
// pair in EmitterLoops -- and they disagreed about where to look: two walked
// the node's FIELDS, two walked `body`/`orelse`/`finalbody`/`handlers` by
// name. The named-field walk misses a `match` case, whose statements hang off
// `cases`, so a `break` inside a match inside a try inside a loop was not seen
// by the guard that exists to catch it and reached the lowering as "reference
// to block defined in another region".
//
// So the walk is over every field, and the two axes the callers actually
// differ on are parameters: which kinds count, and whether a nested LOOP ends
// the search (it does for break/continue, which target that loop, and does not
// for return).
bool containsStatementKind(const parser::Node *node,
                           llvm::ArrayRef<llvm::StringRef> kinds,
                           bool stopAtLoops) {
  if (!node)
    return false;
  if (llvm::is_contained(kinds, llvm::StringRef(node->kind)))
    return true;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return false;
  if (stopAtLoops && (node->kind == "For" || node->kind == "AsyncFor" ||
                      node->kind == "While"))
    return false;
  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (containsStatementKind(child->get(), kinds, stopAtLoops))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &item : *children)
        if (containsStatementKind(item.get(), kinds, stopAtLoops))
          return true;
    }
  }
  return false;
}

bool containsStatementKind(const std::vector<parser::NodePtr> *statements,
                           llvm::ArrayRef<llvm::StringRef> kinds,
                           bool stopAtLoops) {
  if (!statements)
    return false;
  for (const parser::NodePtr &statement : *statements)
    if (containsStatementKind(statement.get(), kinds, stopAtLoops))
      return true;
  return false;
}

bool containsReturnStatement(const std::vector<parser::NodePtr> *statements) {
  llvm::StringRef kinds[] = {"Return"};
  return containsStatementKind(statements, kinds, /*stopAtLoops=*/false);
}

bool containsBreakOrContinueStatement(
    const std::vector<parser::NodePtr> *statements) {
  llvm::StringRef kinds[] = {"Break", "Continue"};
  return containsStatementKind(statements, kinds, /*stopAtLoops=*/true);
}

bool containsContinueStatement(
    const std::vector<parser::NodePtr> *statements) {
  llvm::StringRef kinds[] = {"Continue"};
  return containsStatementKind(statements, kinds, /*stopAtLoops=*/true);
}

bool containsObjectTop(mlir::Type type, const TypeSystem &types) {
  if (!type)
    return true;
  if (type == types.object())
    return true;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    if (contract.getContractName() == "typing.Any")
      return true;
    for (mlir::Type arg : contract.getArguments())
      if (containsObjectTop(arg, types))
        return true;
    return false;
  }
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type)) {
    for (mlir::Type arg : protocol.getArguments())
      if (containsObjectTop(arg, types))
        return true;
    return false;
  }
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
    for (mlir::Type member : unionType.getMemberTypes())
      if (containsObjectTop(member, types))
        return true;
    return false;
  }
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type))
    return containsObjectTop(typeType.getInstanceType(), types);
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type)) {
    for (mlir::Type arg : callable.getPositionalTypes())
      if (containsObjectTop(arg, types))
        return true;
    for (mlir::Type arg : callable.getKwOnlyTypes())
      if (containsObjectTop(arg, types))
        return true;
    for (mlir::Type result : callable.getResultTypes())
      if (containsObjectTop(result, types))
        return true;
    if (callable.hasVararg() &&
        containsObjectTop(callable.getVarargType(), types))
      return true;
    if (callable.hasKwarg() &&
        containsObjectTop(callable.getKwargType(), types))
      return true;
  }
  return false;
}

bool isNoneTypeLike(mlir::Type type) {
  if (auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type))
    return literal.getSpelling() == "None";
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
    return contract.getContractName() == "types.NoneType";
  return false;
}

mlir::Type removeNoneFromType(mlir::Type type, TypeSystem &types) {
  if (!type || isNoneTypeLike(type))
    return {};
  auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type);
  if (!unionType)
    return {};

  bool sawNone = false;
  llvm::SmallVector<mlir::Type, 4> payloads;
  for (mlir::Type member : unionType.getMemberTypes()) {
    member = types.widenLiteral(member);
    if (isNoneTypeLike(member)) {
      sawNone = true;
      continue;
    }
    payloads.push_back(member);
  }
  return sawNone ? types.join(payloads) : mlir::Type{};
}

std::optional<mlir::Type> isinstanceTargetType(const parser::Node *node,
                                               TypeSystem &types) {
  if (!node)
    return std::nullopt;
  mlir::Type inferred = types.inferExpr(node);
  // ⛔ THE GENERIC BUILTINS ARE NOT BOUND AS CLASS OBJECTS. `int` and `str`
  // infer to `!py.type<...>` and `list`/`dict`/`tuple`/`set` infer to plain
  // `object`, so `isinstance(o, list)` was refused as "not a statically
  // resolved class type" while `isinstance(o, str)` was not. The annotation
  // resolver knows every builtin class spelling; asking it is what makes the
  // two spellings agree.
  //
  // Only from `object`, which says nothing: any other inference is a binding
  // the program made, and that is what the name means.
  if (inferred == types.object() && node->kind == "Name")
    if (mlir::Type annotated =
            types.annotationType(node))
      if (auto contract = mlir::dyn_cast<py::ContractType>(annotated))
        if (annotated != types.object()) {
          // ⛔ A BARE CONTAINER SPELLING IS THE CONTAINER OF `object`. The
          // class id compares the same either way, but the NARROWED value has
          // to be a type with methods on it: an argument-less
          // `!py.contract<"builtins.list">` answered "does not provide
          // '__len__'" on the arm that had just proved it was a list. What the
          // test proves about the elements is nothing, and `object` is how
          // that is spelled.
          if (contract.getArguments().empty()) {
            llvm::StringRef name = contract.getContractName();
            if (name == "builtins.list")
              return types.listOf(types.object());
            if (name == "builtins.set")
              return types.contract("builtins.set", {types.object()});
            if (name == "builtins.tuple")
              return types.tupleOf(types.object());
            if (name == "builtins.dict")
              return types.dictOf(types.object(), types.object());
          }
          return annotated;
        }
  auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(inferred);
  if (!typeObject)
    return std::nullopt;
  mlir::Type instance = types.widenLiteral(typeObject.getInstanceType());
  if (!instance || !mlir::isa<py::ContractType>(instance))
    return std::nullopt;
  return instance;
}

std::optional<llvm::SmallVector<mlir::Type, 4>>
isinstanceTargetTypes(const parser::Node *node, TypeSystem &types) {
  // `isinstance(x, (int, str))` is CPython's own spelling for "any of these",
  // and it was refused as "not a statically resolved class type" -- the tuple is
  // not a class, and nothing looked inside it. Each element still has to be one.
  llvm::SmallVector<mlir::Type, 4> targets;
  if (node && node->kind == "Tuple") {
    const auto *elements = ast::nodeList(*node, "elts");
    if (!elements || elements->empty())
      return std::nullopt;
    for (const parser::NodePtr &element : *elements) {
      std::optional<mlir::Type> target =
          isinstanceTargetType(element.get(), types);
      if (!target)
        return std::nullopt;
      targets.push_back(*target);
    }
    return targets;
  }
  if (std::optional<mlir::Type> single = isinstanceTargetType(node, types)) {
    targets.push_back(*single);
    return targets;
  }
  return std::nullopt;
}

// The subclass relation from the module PRE-PASS, for the two source classes
// whose class ops the subtype walk may not have seen yet.
//
// ⛔ Not a replacement for `isAssignableTo`: it knows only what the source
// wrote, so it answers nothing about manifest contracts, generics or
// protocols. It is consulted where the alternative is `setAlwaysFalse`, which
// is a WRONG answer rather than a missing one -- `isinstance(a, B)` inside a
// function defined above `class B(A)` folded to False and ran the else branch,
// printing 1 where CPython prints 2.
bool declaredSubclassOfType(mlir::Type sub, mlir::Type super,
                            TypeSystem &types) {
  auto subContract = mlir::dyn_cast_if_present<py::ContractType>(sub);
  auto superContract = mlir::dyn_cast_if_present<py::ContractType>(super);
  if (!subContract || !superContract)
    return false;
  // ⭐ THE DECLARED-BASES MAP IS THE TEST, not the name's shape.
  // `isSourceDefinedContract` reads "has no dot", which is how a manifest
  // contract is told from a program's class -- and an IMPORTED module's class
  // is dotted too (`shapes.Base`), so every hierarchy question about one
  // answered no:
  //
  //     # shapes.py: class Base: ... / class Derived(Base): ...
  //     def show(b: "shapes.Base") -> str:
  //         if isinstance(b, shapes.Derived):
  //             return "D"
  //         return "B"
  //     print(show(shapes.Derived(2)))   # printed B; CPython prints D
  //
  // A silent wrong answer, and the same classes written in ONE file are right.
  // The map holds exactly the classes this compiler DECLARED, which is what
  // the heuristic was approximating: a manifest contract is never in it, so
  // asking it directly is both narrower and correct.
  return types.declaredSubclassOf(subContract.getContractName(),
                                  superContract.getContractName());
}

bool pythonSubclassOf(mlir::Type sub, mlir::Type super, TypeSystem &types,
                      mlir::Operation *from) {
  if (declaredSubclassOfType(sub, super, types))
    return true;
  if (isAssignableWithStaticEvidence(sub, super, from))
    return true;
  // ⭐ Python's CLASS hierarchy, which is not this compiler's assignability.
  // `bool` is a subclass of `int` there -- `isinstance(True, int)` and
  // `issubclass(bool, int)` are both True -- while the ABI keeps them apart on
  // purpose: a bool is one truth bit and an int is a three-value bundle, so a
  // bool VALUE cannot be stored where an int is expected without the conversion
  // emitIntFromBool exists to make. Both predicates asked the question through
  // assignability and answered False, silently, for the one pair where CPython
  // says True.
  //
  // ⛔ Only that rung, and the numeric tower is NOT the rule: `issubclass(int,
  // float)` is False in CPython too, even though the tower converts one to the
  // other. What decides is the class hierarchy, and int's bases do not include
  // float.
  auto subContract =
      mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(sub));
  auto superContract =
      mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(super));
  return subContract && superContract &&
         subContract.getContractName() == "builtins.bool" &&
         superContract.getContractName() == "builtins.int";
}

// ⛔ A SOURCE CLASS UNDER A MANIFEST BASE, which `declaredSubclassOfType`
// cannot answer: it requires both sides to be source-defined, and the base a
// `class MyErr(Exception)` records is the bare name it was written with. So
// `isinstance(e, MyErr)` on an `Exception`-typed value folded to AlwaysFalse
// and compiled to `return "other"` with no test in it at all -- silently, for
// the shape every user-defined exception is caught by.
static bool sourceClassUnderManifestBase(mlir::Type sub, mlir::Type super,
                                         TypeSystem &types) {
  auto subContract = mlir::dyn_cast_if_present<py::ContractType>(sub);
  auto superContract = mlir::dyn_cast_if_present<py::ContractType>(super);
  if (!subContract || !superContract)
    return false;
  if (!isSourceDefinedContract(sub) || isSourceDefinedContract(super))
    return false;
  llvm::StringRef superName = superContract.getContractName();
  llvm::StringRef superLeaf = superName.rsplit('.').second;
  if (superLeaf.empty())
    superLeaf = superName;
  if (types.declaredSubclassOf(subContract.getContractName(), superLeaf))
    return true;
  // Written under a base that is itself under the target: `class E(ValueError)`
  // asked about `Exception`. The taxonomy is the only place that edge exists.
  for (const py::exceptions::BuiltinExceptionInfo &entry :
       py::exceptions::kBuiltinExceptions)
    if (py::exceptions::isBuiltinExceptionSubclassName(entry.name, superLeaf) &&
        types.declaredSubclassOf(subContract.getContractName(), entry.name))
      return true;
  return false;
}

// ⭐ THE ARGUMENTS OF AN `isinstance` TARGET ARE NOT PART OF THE QUESTION.
// A bare class name resolves to its contract with the top filled in --
// `tuple` becomes `tuple[object]` -- so `pythonSubclassOf(tuple[int, int],
// tuple[object])` compared two DIFFERENT contracts and answered False. On a
// `list[int] | tuple[int, int]` value that made `isinstance(value, tuple)`
// fold to AlwaysFalse: the true arm was dropped and the `else` still held the
// whole union, so `for item in value` was refused for a method the tuple arm
// was the only one lacking. The same union with a `str` member worked, because
// `str` carries no arguments to disagree about.
//
// Why NOT relax `pythonSubclassOf` itself: it answers `issubclass` too, where
// the arguments DO carry meaning, and it is the assignability check that the
// narrowing then trusts to place a value. Only the isinstance target is known
// to be an unsubscripted name -- CPython raises TypeError for a parameterised
// generic there, so arguments on it can only be the filler.
static bool sameClassIgnoringArguments(mlir::Type member, mlir::Type target) {
  auto memberContract = mlir::dyn_cast_if_present<py::ContractType>(member);
  auto targetContract = mlir::dyn_cast_if_present<py::ContractType>(target);
  if (!memberContract || !targetContract ||
      memberContract.getContractName() != targetContract.getContractName())
    return false;
  return llvm::all_of(targetContract.getArguments(), [](mlir::Type argument) {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(argument);
    return contract && contract.getContractName() == "builtins.object" &&
           contract.getArguments().empty();
  });
}

static bool isinstanceClassMatch(mlir::Type member, mlir::Type target,
                                 TypeSystem &types, mlir::Operation *from) {
  return sameClassIgnoringArguments(member, target) ||
         pythonSubclassOf(member, target, types, from);
}

IsInstanceAnalysis analyzeIsInstance(mlir::Type sourceType,
                                     mlir::Type targetType, TypeSystem &types,
                                     mlir::Operation *from) {
  IsInstanceAnalysis analysis;
  analysis.sourceType = types.widenLiteral(sourceType);
  analysis.targetType = types.widenLiteral(targetType);
  if (!analysis.sourceType || !analysis.targetType ||
      !mlir::isa<py::ContractType>(analysis.targetType)) {
    analysis.failureReason =
        "isinstance requires a statically resolved class target";
    return analysis;
  }

  auto setAlwaysTrue = [&]() {
    analysis.kind = IsInstanceAnalysis::Kind::AlwaysTrue;
    analysis.trueType = analysis.sourceType;
  };
  auto setAlwaysFalse = [&]() {
    analysis.kind = IsInstanceAnalysis::Kind::AlwaysFalse;
    analysis.falseType = analysis.sourceType;
  };

  if (isinstanceClassMatch(analysis.sourceType, analysis.targetType, types,
                           from)) {
    setAlwaysTrue();
    return analysis;
  }

  if (auto unionType = mlir::dyn_cast<py::UnionType>(analysis.sourceType)) {
    llvm::SmallVector<mlir::Type, 4> remaining;
    llvm::SmallVector<mlir::Type, 2> runtimeClassTestMembers;
    bool sawUnsupportedMember = false;
    for (mlir::Type rawMember : unionType.getMemberTypes()) {
      mlir::Type member = types.widenLiteral(rawMember);
      if (isinstanceClassMatch(member, analysis.targetType, types, from)) {
        analysis.unionMembers.push_back(rawMember);
      } else if (containsObjectTop(member, types)) {
        sawUnsupportedMember = true;
      } else if (isAssignableWithStaticEvidence(analysis.targetType, member,
                                                from)) {
        runtimeClassTestMembers.push_back(rawMember);
      } else {
        remaining.push_back(rawMember);
      }
    }

    if (analysis.unionMembers.size() == unionType.getMemberTypes().size()) {
      // ⛔ AND THE TRUE ARM IS STILL THE UNION when every member is the SAME
      // class differently parameterised -- `list[list[int]] | list[int]` under
      // `isinstance(v, list)`. The test is genuinely always true, so there is
      // nothing here to narrow WITH; what the arm needs is for the union
      // itself to answer `__iter__`, which it does not: the same annotation
      // without any isinstance at all is refused identically. Measured on
      // b6 (`for row in value` over that union) -- so this is the union member
      // lookup, not the narrowing, and fixing it here would only move the
      // refusal. Before this rule the fold went the other way and the loop
      // body was DEAD, printing 0 where CPython prints 3.
      setAlwaysTrue();
      return analysis;
    }
    if (analysis.unionMembers.empty()) {
      if (!sawUnsupportedMember && runtimeClassTestMembers.size() == 1) {
        analysis.kind = IsInstanceAnalysis::Kind::UnionClassTest;
        analysis.unionMembers.push_back(runtimeClassTestMembers.front());
        analysis.trueType = analysis.targetType;
        return analysis;
      }
      if (sawUnsupportedMember || !runtimeClassTestMembers.empty()) {
        analysis.failureReason =
            "isinstance over this union would require an unsupported dynamic "
            "class test inside a union member";
        return analysis;
      }
      setAlwaysFalse();
      return analysis;
    }

    analysis.kind = IsInstanceAnalysis::Kind::UnionTest;
    if (analysis.unionMembers.size() == 1)
      analysis.trueType = analysis.unionMembers.front();
    // ⭐ WHAT IS LEFT, EVEN WHEN IT IS STILL SEVERAL. The false arm used to
    // narrow only down to a single member, so a union of four narrowed to
    // nothing after the first elimination and every guard after it started
    // over from the full set. `join` of one member is that member, so the
    // single-member case is unchanged; the value is not re-tagged either way
    // -- `applyBranchNarrowing` spends a multi-member result on the NAME only.
    if (!remaining.empty() && !sawUnsupportedMember)
      analysis.falseType = types.join(remaining);
    if (!runtimeClassTestMembers.empty()) {
      analysis.kind = IsInstanceAnalysis::Kind::Unsupported;
      analysis.failureReason =
          "isinstance cannot yet combine exact union member tests with dynamic "
          "class tests";
    }
    return analysis;
  }

  if (containsObjectTop(analysis.sourceType, types)) {
    // ⭐ `object` ITSELF IS A CLASS TEST, not dynamic inspection. The value is
    // a handle and word 1 of every header is the class id, so
    // `isinstance(o, A)` is the SAME load-and-compare a source-class receiver
    // gets -- the target names a statically closed set of ids, and nothing is
    // asked of the value that its header does not already say.
    //
    // ⛔ Only when the source is `object` at the TOP. `list[object]` reaches
    // here too, and there the question is about the container, whose own class
    // is known; the members are not what the test names.
    auto sourceContract =
        mlir::dyn_cast<py::ContractType>(analysis.sourceType);
    if (sourceContract &&
        sourceContract.getContractName() == "builtins.object" &&
        mlir::isa<py::ContractType>(analysis.targetType)) {
      // ⛔ THE SUBCLASSES ARE ENUMERATED HERE AND NOWHERE LOWER. The test is an
      // exact class-id compare, so `isinstance(o, Exception)` has to name every
      // class that answers yes -- and the taxonomy that says which those are is
      // `py.class`, which the emitter consumes: no phase after this one has a
      // single one of them left to walk. Leaving it to the lowering answered
      // False for `isinstance(True, int)` and for every caught exception.
      auto addSubclass = [&](mlir::Type candidate) {
        if (!candidate || candidate == analysis.targetType ||
            llvm::is_contained(analysis.classTestTypes, candidate))
          return;
        if (pythonSubclassOf(candidate, analysis.targetType, types, from))
          analysis.classTestTypes.push_back(candidate);
      };
      // Source classes: the module has a `py.class` for each, which is what
      // `pythonSubclassOf` reads to answer about them.
      if (auto enclosing = from->getParentOfType<mlir::ModuleOp>())
        enclosing.walk([&](py::ClassOp classOp) {
          if (std::optional<mlir::Type> classType =
                  types.lookupClass(classOp.getSymName()))
            addSubclass(*classType);
        });
      // ⛔ THE BUILTINS ARE NOT IN THE MODULE. A manifest class has no
      // `py.class` here -- the walk above finds `class MyErr(Exception)` and
      // nothing else -- so `bool` under `int`, the one builtin subclass edge
      // outside the exception taxonomy, is named from the rule that defines it.
      //
      // Why NOT the exceptions too: the test compares ids through
      // `LyEH_ClassIdMatches`, which walks that taxonomy at RUNTIME and covers
      // user exception classes as well. Naming them here would be 80 redundant
      // compares in front of a walk that already answers.
      mlir::MLIRContext *context = analysis.targetType.getContext();
      addSubclass(py::ContractType::get(context, "builtins.bool"));
      analysis.kind = IsInstanceAnalysis::Kind::ClassTest;
      // ⛔ THE TEST NARROWS EVERY TARGET BUT `int` AND `bool`. Narrowing hands
      // the branch a VIEW of the box's entity, which is right exactly when
      // every class the test accepts has the target's runtime layout.
      //
      // `bool` has no entity at all -- its runtime shape is `i1`, and the
      // branch failed to compile with "builtins.bool has no statically sized
      // entity lane". `int` accepts bool, because Python's `bool` IS an `int`
      // and the test says so; but a boxed bool is `LyBool_Box`'s three-word
      // immortal singleton and an int is not, so viewing one as the other read
      // `True + 1` as 1 and `False + 1` as a pointer-shaped integer. The test
      // keeps answering CPython's answer; only the view is withheld.
      llvm::StringRef targetName =
          mlir::cast<py::ContractType>(analysis.targetType).getContractName();
      if (targetName != "builtins.bool" && targetName != "builtins.int")
        analysis.trueType = analysis.targetType;
      return analysis;
    }
    analysis.failureReason =
        "isinstance on an object-typed value requires dynamic object "
        "inspection, which is excluded from the static evidence kernel";
    return analysis;
  }

  if (mlir::isa<py::ContractType>(analysis.sourceType) &&
      (isAssignableWithStaticEvidence(analysis.targetType, analysis.sourceType,
                                      from) ||
       declaredSubclassOfType(analysis.targetType, analysis.sourceType,
                              types) ||
       sourceClassUnderManifestBase(analysis.targetType, analysis.sourceType,
                                    types))) {
    analysis.kind = IsInstanceAnalysis::Kind::ClassTest;
    analysis.trueType = analysis.targetType;
    return analysis;
  }

  // ⭐ TWO CLASSES WITH NO EDGE BETWEEN THEM ARE NOT DISJOINT: a THIRD class
  // may derive from both.
  //
  //     class A: pass
  //     class M:
  //         def only_m(self) -> int: return 5
  //     class B(A, M): pass
  //     a: A = B()
  //     print(isinstance(a, M))    # printed False; CPython prints True
  //
  // Neither type is assignable to the other and neither declares the other as
  // a base, which is what everything above reads as "cannot be" -- and a mixin
  // is exactly the shape that makes it wrong, silently, in the direction that
  // drops the branch a program wrote.
  //
  // ⛔ The ids the test names are the classes that derive from BOTH, not the
  // target's subclasses: only those can reach this point, and naming the rest
  // would be compares that cannot fire.
  //
  // ⛔ And the branch narrows only when there is exactly ONE such class. The
  // narrowing hands the branch a VIEW with that class's layout, which is sound
  // for the single candidate and a guess for a join of several -- so several
  // answer the test and narrow nothing, which is a refusal inside the branch
  // rather than a wrong layout.
  if (mlir::isa<py::ContractType>(analysis.sourceType) &&
      mlir::isa<py::ContractType>(analysis.targetType) && from) {
    llvm::SmallVector<mlir::Type, 4> both;
    // ⛔ `from` IS the module at top level, and `getParentOfType` looks only at
    // ANCESTORS -- so the walk found no class at all and the arm answered as if
    // the program had none.
    mlir::ModuleOp enclosing = mlir::dyn_cast<mlir::ModuleOp>(from);
    if (!enclosing)
      enclosing = from->getParentOfType<mlir::ModuleOp>();
    if (enclosing)
      enclosing.walk([&](py::ClassOp classOp) {
        std::optional<mlir::Type> classType =
            types.lookupClass(classOp.getSymName());
        if (!classType || llvm::is_contained(both, *classType))
          return;
        if (pythonSubclassOf(*classType, analysis.targetType, types, from) &&
            pythonSubclassOf(*classType, analysis.sourceType, types, from))
          both.push_back(*classType);
      });
    if (!both.empty()) {
      analysis.kind = IsInstanceAnalysis::Kind::ClassTest;
      analysis.targetType = both.front();
      analysis.classTestTypes.assign(std::next(both.begin()), both.end());
      if (both.size() == 1)
        analysis.trueType = both.front();
      return analysis;
    }
  }

  setAlwaysFalse();
  return analysis;
}

IsInstanceAnalysis analyzeIsInstanceAny(mlir::Type sourceType,
                                        llvm::ArrayRef<mlir::Type> targetTypes,
                                        TypeSystem &types,
                                        mlir::Operation *from) {
  // One target is the whole existing question, and answering it through the
  // merge below would change the answer for every program that has one: the
  // ClassTest and UnionClassTest kinds carry a runtime test the merge cannot
  // combine, so they are only reachable on this path.
  if (targetTypes.size() == 1)
    return analyzeIsInstance(sourceType, targetTypes.front(), types, from);

  IsInstanceAnalysis merged;
  merged.sourceType = types.widenLiteral(sourceType);

  // ⛔ EVERY ELEMENT A CLASS TEST IS ONE MERGED CLASS TEST, which is what a
  // type-erased subject always produces: the tests are independent compares of
  // the same class-id word, so their OR is the answer. Refusing them together
  // meant `isinstance(o, (list, tuple))` had to be written as two `if`s.
  //
  // ⛔ NO NARROWING out of a merged test. The subject is one of several
  // classes on the true arm and there is no single type to view it as; the
  // single-target path above is where a narrowing comes from.
  {
    llvm::SmallVector<mlir::Type, 4> classTargets;
    bool allClassTests = true;
    for (mlir::Type target : targetTypes) {
      IsInstanceAnalysis one =
          analyzeIsInstance(sourceType, target, types, from);
      if (one.kind == IsInstanceAnalysis::Kind::AlwaysFalse)
        continue;
      if (one.kind != IsInstanceAnalysis::Kind::ClassTest) {
        allClassTests = false;
        break;
      }
      if (!llvm::is_contained(classTargets, one.targetType))
        classTargets.push_back(one.targetType);
      for (mlir::Type extra : one.classTestTypes)
        if (!llvm::is_contained(classTargets, extra))
          classTargets.push_back(extra);
    }
    if (allClassTests && !classTargets.empty()) {
      merged.kind = IsInstanceAnalysis::Kind::ClassTest;
      merged.targetType = classTargets.front();
      merged.classTestTypes.assign(std::next(classTargets.begin()),
                                   classTargets.end());
      return merged;
    }
  }

  llvm::SmallVector<mlir::Type, 4> selected;
  for (mlir::Type target : targetTypes) {
    IsInstanceAnalysis one = analyzeIsInstance(sourceType, target, types, from);
    if (one.kind == IsInstanceAnalysis::Kind::AlwaysTrue) {
      merged.kind = IsInstanceAnalysis::Kind::AlwaysTrue;
      merged.trueType = merged.sourceType;
      return merged;
    }
    if (one.kind == IsInstanceAnalysis::Kind::AlwaysFalse)
      continue;
    if (one.kind != IsInstanceAnalysis::Kind::UnionTest) {
      // ⛔ A tuple element that needs a RUNTIME class test is refused rather
      // than merged: the tests are per-member ops over one union value, and a
      // class test is not one of them. Splitting the tuple by hand still works.
      merged.kind = IsInstanceAnalysis::Kind::Unsupported;
      merged.failureReason =
          one.failureReason.empty()
              ? "isinstance over a tuple of classes cannot combine a dynamic "
                "class test"
              : one.failureReason;
      return merged;
    }
    for (mlir::Type member : one.unionMembers)
      if (!llvm::is_contained(selected, member))
        selected.push_back(member);
  }
  if (selected.empty()) {
    merged.kind = IsInstanceAnalysis::Kind::AlwaysFalse;
    merged.falseType = merged.sourceType;
    return merged;
  }
  auto unionType = mlir::dyn_cast<py::UnionType>(merged.sourceType);
  if (unionType && selected.size() == unionType.getMemberTypes().size()) {
    merged.kind = IsInstanceAnalysis::Kind::AlwaysTrue;
    merged.trueType = merged.sourceType;
    return merged;
  }
  merged.kind = IsInstanceAnalysis::Kind::UnionTest;
  merged.unionMembers.assign(selected.begin(), selected.end());
  if (selected.size() == 1)
    merged.trueType = selected.front();
  if (unionType) {
    llvm::SmallVector<mlir::Type, 4> remaining;
    for (mlir::Type member : unionType.getMemberTypes())
      if (!llvm::is_contained(selected, member))
        remaining.push_back(member);
    if (remaining.size() == 1)
      merged.falseType = remaining.front();
  }
  return merged;
}

struct IsInstanceBranchAnalysis {
  std::string name;
  IsInstanceAnalysis analysis;
};

static std::optional<IsInstanceBranchAnalysis>
optionalIsInstanceBranchAnalysis(const parser::Node &test, TypeSystem &types,
                                 mlir::Operation *from) {
  if (test.kind != "Call")
    return std::nullopt;
  const parser::Node *callee = ast::node(test, "func");
  if (!callee || callee->kind != "Name" ||
      ast::nameSpelling(*callee) != "isinstance")
    return std::nullopt;
  const auto *keywords = ast::nodeList(test, "keywords");
  if (keywords && !keywords->empty())
    return std::nullopt;
  const auto *args = ast::nodeList(test, "args");
  if (!args || args->size() != 2 || !args->front() ||
      args->front()->kind != "Name")
    return std::nullopt;

  llvm::StringRef name = ast::nameSpelling(*args->front());
  std::optional<mlir::Type> sourceType = types.lookupSymbol(name);
  std::optional<llvm::SmallVector<mlir::Type, 4>> targetTypes =
      isinstanceTargetTypes((*args)[1].get(), types);
  if (!sourceType || !targetTypes)
    return std::nullopt;

  return IsInstanceBranchAnalysis{
      name.str(),
      analyzeIsInstanceAny(*sourceType, *targetTypes, types, from)};
}

const parser::Node *nameComparedWithNone(const parser::Node *left,
                                         const parser::Node *right) {
  if (!left || !right)
    return nullptr;
  if (left->kind == "Name" && right->kind == "Constant" &&
      ast::isNoneField(*right, "value"))
    return left;
  if (right->kind == "Name" && left->kind == "Constant" &&
      ast::isNoneField(*left, "value"))
    return right;
  return nullptr;
}

std::optional<NoneComparisonNarrowing>
optionalNoneComparison(const parser::Node &test, TypeSystem &types) {
  if (test.kind != "Compare")
    return std::nullopt;
  const auto *comparators = ast::nodeList(test, "comparators");
  const auto *ops = ast::nodeList(test, "ops");
  if (!comparators || comparators->size() != 1 || !ops || ops->size() != 1)
    return std::nullopt;

  const parser::Node *op = ops->front().get();
  bool trueBranchIsNone = true;
  if (ast::isOperator(op, "Is")) {
    trueBranchIsNone = true;
  } else if (ast::isOperator(op, "IsNot")) {
    trueBranchIsNone = false;
  } else {
    return std::nullopt;
  }

  const parser::Node *name =
      nameComparedWithNone(ast::node(test, "left"), comparators->front().get());
  if (!name)
    return std::nullopt;
  llvm::StringRef spelling = ast::nameSpelling(*name);
  std::optional<mlir::Type> currentType = types.lookupSymbol(spelling);
  mlir::Type payloadType =
      currentType ? removeNoneFromType(*currentType, types) : mlir::Type{};
  if (!payloadType)
    return std::nullopt;
  return NoneComparisonNarrowing{spelling.str(), trueBranchIsNone, payloadType};
}

std::optional<BranchTypeNarrowing>
optionalBranchTypeNarrowing(const parser::Node &test, TypeSystem &types,
                            mlir::Operation *from) {
  if (test.kind == "UnaryOp") {
    const parser::Node *op = ast::node(test, "op");
    if (!ast::isOperator(op, "Not"))
      return std::nullopt;
    const parser::Node *operand = ast::node(test, "operand");
    if (!operand)
      return std::nullopt;
    std::optional<BranchTypeNarrowing> inner =
        optionalBranchTypeNarrowing(*operand, types, from);
    if (!inner)
      return std::nullopt;
    std::swap(inner->trueType, inner->falseType);
    std::swap(inner->trueSourceType, inner->falseSourceType);
    return inner;
  }

  // ⭐ `A and B` PROVES A ON ITS TRUE SIDE, `A or B` PROVES A ON ITS FALSE ONE.
  // The operands are emitted with this already (the short-circuit path in
  // `emitExpr` applies each proof to the ones after it); this is the same fact
  // asked by the STATEMENT, so `if s is not None and ...:` narrows its body and
  // `while s is not None and ...:` narrows the loop's.
  //
  // ⛔ Without it the body read the union: `while n is not None and n.v < 1: n =
  // n.nxt` typed `n.nxt` off an un-narrowed `Node | None`, which infers as
  // `object`, and the back edge then carried an `object` into a header expecting
  // the union -- "type mismatch for bb argument #0 of successor #0".
  //
  // ⛔ ONE NAME here, the first operand that proves anything; the caller that
  // wants them all asks `branchTypeNarrowings`, which walks the same operands.
  if (test.kind == "BoolOp") {
    const parser::Node *op = ast::node(test, "op");
    const bool isAnd = op && op->kind == "And";
    const bool isOr = op && op->kind == "Or";
    if (!isAnd && !isOr)
      return std::nullopt;
    const auto *operands = ast::nodeList(test, "values");
    if (!operands)
      return std::nullopt;
    for (const parser::NodePtr &operand : *operands) {
      if (!operand)
        continue;
      std::optional<BranchTypeNarrowing> inner =
          optionalBranchTypeNarrowing(*operand, types, from);
      if (!inner)
        continue;
      if (isAnd) {
        inner->falseType = mlir::Type();
        inner->falseSourceType = mlir::Type();
        if (!inner->trueType)
          continue;
      } else {
        inner->trueType = mlir::Type();
        inner->trueSourceType = mlir::Type();
        if (!inner->falseType)
          continue;
      }
      return inner;
    }
    return std::nullopt;
  }

  if (std::optional<NoneComparisonNarrowing> none =
          optionalNoneComparison(test, types)) {
    BranchTypeNarrowing narrowing;
    narrowing.name = none->name;
    narrowing.trueType =
        none->trueBranchIsNone ? types.none() : none->payloadType;
    narrowing.falseType =
        none->trueBranchIsNone ? none->payloadType : types.none();
    return narrowing;
  }

  // Bare-name truthiness over an Optional: truthy implies not-None, so the
  // true branch narrows to the payload. The false branch stays un-narrowed —
  // a falsy payload (empty container) is indistinguishable from None there.
  if (test.kind == "Name") {
    llvm::StringRef spelling = ast::nameSpelling(test);
    std::optional<mlir::Type> currentType = types.lookupSymbol(spelling);
    mlir::Type payload =
        currentType ? removeNoneFromType(*currentType, types) : mlir::Type{};
    if (payload) {
      BranchTypeNarrowing narrowing;
      narrowing.name = spelling.str();
      narrowing.trueType = payload;
      return narrowing;
    }
  }

  std::optional<IsInstanceBranchAnalysis> analyzed =
      optionalIsInstanceBranchAnalysis(test, types, from);
  if (!analyzed)
    return std::nullopt;

  const IsInstanceAnalysis &analysis = analyzed->analysis;
  if (analysis.kind == IsInstanceAnalysis::Kind::Unsupported)
    return std::nullopt;

  BranchTypeNarrowing narrowing;
  narrowing.name = analyzed->name;
  narrowing.trueType = analysis.trueType;
  narrowing.falseType = analysis.falseType;
  if (analysis.kind == IsInstanceAnalysis::Kind::UnionClassTest &&
      analysis.unionMembers.size() == 1)
    narrowing.trueSourceType = analysis.unionMembers.front();
  if (!narrowing.trueType && !narrowing.falseType)
    return std::nullopt;
  return narrowing;
}

// ⭐ A CONJUNCTION PROVES ONE THING PER OPERAND, and the body gets to keep all
// of them:
//
//     if isinstance(x, B) and isinstance(y, B):
//         return x.n + y.n      # 'n' is overridden by a subclass of 'A'
//
// `x` narrowed and `y` did not, because a single `BranchTypeNarrowing` carries
// one name and the arm above returns the first operand that proves anything.
// Inside the CONDITION the later operands are already narrowed (each proof is
// applied to the operands after it), which is why `and x.only()` worked and
// the body did not.
//
// ⛔ ONE FACT PER NAME, the first. Two facts about the SAME name would have to
// be INTERSECTED -- `v is None or isinstance(v, str)` proves `int` on its false
// side, not `int | None` -- and applying the second after the first replaces
// rather than refines. `applyBranchNarrowing` makes that a safe no-op today, so
// dropping the duplicate here says what is meant instead of relying on it.
llvm::SmallVector<BranchTypeNarrowing, 2>
branchTypeNarrowings(const parser::Node &test, TypeSystem &types,
                     mlir::Operation *from) {
  llvm::SmallVector<BranchTypeNarrowing, 2> facts;
  auto add = [&](std::optional<BranchTypeNarrowing> fact) {
    if (!fact)
      return;
    for (const BranchTypeNarrowing &seen : facts)
      if (seen.name == fact->name)
        return;
    facts.push_back(std::move(*fact));
  };
  const parser::Node *op =
      test.kind == "BoolOp" ? ast::node(test, "op") : nullptr;
  const bool isAnd = op && op->kind == "And";
  const bool isOr = op && op->kind == "Or";
  if (!isAnd && !isOr) {
    add(optionalBranchTypeNarrowing(test, types, from));
    return facts;
  }
  const auto *operands = ast::nodeList(test, "values");
  if (!operands)
    return facts;
  for (const parser::NodePtr &operand : *operands) {
    if (!operand)
      continue;
    std::optional<BranchTypeNarrowing> inner =
        optionalBranchTypeNarrowing(*operand, types, from);
    if (!inner)
      continue;
    if (isAnd) {
      inner->falseType = mlir::Type();
      inner->falseSourceType = mlir::Type();
      if (!inner->trueType)
        continue;
    } else {
      inner->trueType = mlir::Type();
      inner->trueSourceType = mlir::Type();
      if (!inner->falseType)
        continue;
    }
    add(std::move(inner));
  }
  return facts;
}

std::optional<bool> optionalStaticBranchTruth(const parser::Node &test,
                                              TypeSystem &types,
                                              mlir::Operation *from) {
  if (test.kind == "Constant") {
    if (std::optional<bool> value = ast::boolean(test, "value"))
      return *value;
  }

  if (test.kind == "UnaryOp") {
    const parser::Node *op = ast::node(test, "op");
    if (!ast::isOperator(op, "Not"))
      return std::nullopt;
    const parser::Node *operand = ast::node(test, "operand");
    if (!operand)
      return std::nullopt;
    if (std::optional<bool> value =
            optionalStaticBranchTruth(*operand, types, from))
      return !*value;
    return std::nullopt;
  }

  // `<string literal> == <string literal>` (and !=) folds statically: the
  // platform constants (sys.platform, os.name, platform.system()) type as
  // string literals, so this is the compile-time platform switch idiom --
  // runtime lib modules rely on it so foreign platforms' libc symbols
  // never reach the artifact.
  if (test.kind == "Compare") {
    const auto *comparators = ast::nodeList(test, "comparators");
    const auto *ops = ast::nodeList(test, "ops");
    if (comparators && comparators->size() == 1 && ops && ops->size() == 1) {
      const parser::Node *op = ops->front().get();
      bool isEq = ast::isOperator(op, "Eq");
      bool isNe = ast::isOperator(op, "NotEq");
      if (isEq || isNe) {
        auto stringLiteralSpelling =
            [&](const parser::Node *node) -> std::optional<std::string> {
          if (!node)
            return std::nullopt;
          auto literal = mlir::dyn_cast_if_present<py::LiteralType>(
              types.inferExpr(node));
          if (!literal)
            return std::nullopt;
          llvm::StringRef spelling = literal.getSpelling();
          if (spelling.size() >= 2 && spelling.front() == '"' &&
              spelling.back() == '"')
            return spelling.str();
          return std::nullopt;
        };
        std::optional<std::string> left =
            stringLiteralSpelling(ast::node(test, "left"));
        std::optional<std::string> right =
            stringLiteralSpelling(comparators->front().get());
        if (left && right)
          return isEq ? (*left == *right) : (*left != *right);
      }
    }
  }

  std::optional<IsInstanceBranchAnalysis> analyzed =
      optionalIsInstanceBranchAnalysis(test, types, from);
  if (!analyzed)
    return std::nullopt;
  if (analyzed->analysis.kind == IsInstanceAnalysis::Kind::AlwaysTrue)
    return true;
  if (analyzed->analysis.kind == IsInstanceAnalysis::Kind::AlwaysFalse)
    return false;
  return std::nullopt;
}

mlir::Type widenInferredLiterals(mlir::Type type, const TypeSystem &types) {
  return py::mapPyTypeStructure(
      type, [&](mlir::Type node) -> std::optional<mlir::Type> {
        mlir::Type widened = types.widenLiteral(node);
        if (widened != node)
          return widened;
        return std::nullopt;
      });
}

bool hasUnexpectedObjectTop(mlir::Type actual, mlir::Type expected,
                            const TypeSystem &types) {
  if (!actual)
    return true;
  if (actual == types.object()) {
    if (!expected)
      return true;
    if (expected == types.object())
      return false;
    if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(expected))
      return contract.getContractName() != "typing.Any";
    return true;
  }
  if (!expected)
    return containsObjectTop(actual, types);

  if (auto actualCallable =
          mlir::dyn_cast_if_present<py::CallableType>(actual)) {
    auto expectedCallable =
        mlir::dyn_cast_if_present<py::CallableType>(expected);
    if (!expectedCallable)
      return containsObjectTop(actual, types);
    for (auto [actualArg, expectedArg] :
         llvm::zip(actualCallable.getPositionalTypes(),
                   expectedCallable.getPositionalTypes()))
      if (hasUnexpectedObjectTop(actualArg, expectedArg, types))
        return true;
    for (auto [actualArg, expectedArg] :
         llvm::zip(actualCallable.getKwOnlyTypes(),
                   expectedCallable.getKwOnlyTypes()))
      if (hasUnexpectedObjectTop(actualArg, expectedArg, types))
        return true;
    for (auto [actualResult, expectedResult] :
         llvm::zip(actualCallable.getResultTypes(),
                   expectedCallable.getResultTypes()))
      if (hasUnexpectedObjectTop(actualResult, expectedResult, types))
        return true;
    if (actualCallable.hasVararg()) {
      mlir::Type expectedVararg = expectedCallable.hasVararg()
                                      ? expectedCallable.getVarargType()
                                      : mlir::Type();
      if (hasUnexpectedObjectTop(actualCallable.getVarargType(), expectedVararg,
                                 types))
        return true;
    }
    if (actualCallable.hasKwarg()) {
      mlir::Type expectedKwarg = expectedCallable.hasKwarg()
                                     ? expectedCallable.getKwargType()
                                     : mlir::Type();
      if (hasUnexpectedObjectTop(actualCallable.getKwargType(), expectedKwarg,
                                 types))
        return true;
    }
    return false;
  }

  if (auto actualContract =
          mlir::dyn_cast_if_present<py::ContractType>(actual)) {
    auto expectedContract =
        mlir::dyn_cast_if_present<py::ContractType>(expected);
    if (!expectedContract ||
        actualContract.getContractName() != expectedContract.getContractName())
      return containsObjectTop(actual, types);
    for (auto [actualArg, expectedArg] : llvm::zip(
             actualContract.getArguments(), expectedContract.getArguments()))
      if (hasUnexpectedObjectTop(actualArg, expectedArg, types))
        return true;
    return false;
  }

  if (auto actualProtocol =
          mlir::dyn_cast_if_present<py::ProtocolType>(actual)) {
    auto expectedProtocol =
        mlir::dyn_cast_if_present<py::ProtocolType>(expected);
    if (!expectedProtocol ||
        actualProtocol.getProtocolName() != expectedProtocol.getProtocolName())
      return containsObjectTop(actual, types);
    for (auto [actualArg, expectedArg] : llvm::zip(
             actualProtocol.getArguments(), expectedProtocol.getArguments()))
      if (hasUnexpectedObjectTop(actualArg, expectedArg, types))
        return true;
    return false;
  }

  if (auto actualType = mlir::dyn_cast_if_present<py::TypeType>(actual)) {
    auto expectedType = mlir::dyn_cast_if_present<py::TypeType>(expected);
    return hasUnexpectedObjectTop(
        actualType.getInstanceType(),
        expectedType ? expectedType.getInstanceType() : mlir::Type(), types);
  }

  if (auto actualUnion = mlir::dyn_cast_if_present<py::UnionType>(actual)) {
    auto expectedUnion = mlir::dyn_cast_if_present<py::UnionType>(expected);
    for (mlir::Type actualMember : actualUnion.getMemberTypes()) {
      mlir::Type expectedMember =
          expectedUnion && expectedUnion.getMemberTypes().size() ==
                               actualUnion.getMemberTypes().size()
              ? expectedUnion.getMemberTypes().front()
              : expected;
      if (hasUnexpectedObjectTop(actualMember, expectedMember, types))
        return true;
    }
  }

  return false;
}


namespace {

// The carried local was replaced through a structural-mutation call chain
// (`ly.structural_mutation`), which already CONSUMED (transferred) the
// previous representation into the call; releasing the previous value again
// would double-consume the ownership token.
bool derivesViaStructuralMutationImpl(
    mlir::Value current, mlir::Value previous,
    llvm::SmallPtrSetImpl<void *> &visited, unsigned depth) {
  if (depth > 32)
    return false;
  while (current && current != previous) {
    // Any structural-mutation op (py.call append, py.setitem, ...) exposes
    // the rebound receiver as its LAST result and the receiver as operand 0.
    if (mlir::Operation *definition = current.getDefiningOp()) {
      if (!definition->hasAttr("ly.structural_mutation") ||
          definition->getNumResults() < 1 ||
          current != definition->getResult(definition->getNumResults() - 1) ||
          definition->getNumOperands() < 1)
        return false;
      current = definition->getOperand(0);
      continue;
    }
    // A merge or loop-header block argument derives via structural mutation
    // when EVERY incoming edge forwards the previous value itself (identity
    // path: the token is forwarded, not consumed), a mutation chain over it,
    // or — coinductively — a chain rooted at an already-visited argument
    // (loop back-edges: assume the invariant holds and check the remaining
    // edges). Skipping the replacement release is sound on all path kinds.
    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current);
    if (!blockArg)
      return false;
    if (!visited.insert(current.getAsOpaquePointer()).second)
      return true;
    mlir::Block *block = blockArg.getOwner();
    if (block->hasNoPredecessors())
      return false;
    for (mlir::Block *pred : block->getPredecessors()) {
      mlir::Operation *terminator = pred->getTerminator();
      auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
      if (!branch)
        return false;
      for (auto [index, successor] :
           llvm::enumerate(terminator->getSuccessors())) {
        if (successor != block)
          continue;
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(
            static_cast<unsigned>(index));
        mlir::Value incoming = operands[blockArg.getArgNumber()];
        if (!incoming)
          return false;
        if (incoming != previous &&
            !derivesViaStructuralMutationImpl(incoming, previous, visited,
                                              depth + 1))
          return false;
      }
    }
    return true;
  }
  return current == previous;
}

} // namespace

void collectAssignedNameTargets(const parser::Node *node,
                                llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "Name") {
    names.insert(ast::nameSpelling(*node));
    return;
  }
  if (node->kind == "Tuple" || node->kind == "List") {
    if (const auto *elts = ast::nodeList(*node, "elts"))
      for (const parser::NodePtr &elt : *elts)
        collectAssignedNameTargets(elt.get(), names);
  }
}

// Names a statement subtree binds through a NAME target: assignment,
// annotated/augmented assignment, walrus, `for`, `with ... as`. A nested
// `def`/`class` is a boundary either way; `bindsNestedDefinitions` says
// whether the boundary's own name counts as a binding here (it does for the
// enclosing function's locals, and does not when the question is which names
// a region rebinds).
void collectNameBindings(const parser::Node *node, llvm::StringSet<> &names,
                         bool bindsNestedDefinitions) {
  ast::walk(node, [&](const parser::Node &current) {
    if (current.kind == "FunctionDef" || current.kind == "AsyncFunctionDef" ||
        current.kind == "ClassDef") {
      if (bindsNestedDefinitions)
        if (auto name = ast::string(current, "name"))
          names.insert(*name);
      return ast::Walk::SkipChildren;
    }
    if (current.kind == "Lambda")
      return ast::Walk::SkipChildren;
    if (current.kind == "Assign") {
      if (const auto *targets = ast::nodeList(current, "targets"))
        for (const parser::NodePtr &target : *targets)
          collectAssignedNameTargets(target.get(), names);
    } else if (current.kind == "AnnAssign" || current.kind == "AugAssign" ||
               current.kind == "NamedExpr") {
      collectAssignedNameTargets(ast::node(current, "target"), names);
    } else if (current.kind == "For" || current.kind == "AsyncFor") {
      collectAssignedNameTargets(ast::node(current, "target"), names);
    } else if (current.kind == "With" || current.kind == "AsyncWith") {
      if (const auto *items = ast::nodeList(current, "items"))
        for (const parser::NodePtr &item : *items)
          collectAssignedNameTargets(ast::node(*item, "optional_vars"), names);
    }
    return ast::Walk::Continue;
  });
}

bool derivesViaStructuralMutation(mlir::Value current, mlir::Value previous) {
  llvm::SmallPtrSet<void *, 8> visited;
  return derivesViaStructuralMutationImpl(current, previous, visited,
                                          /*depth=*/0);
}

void collectAssignedNames(const parser::Node *node, llvm::StringSet<> &names) {
  if (!node)
    return;
  if (node->kind == "FunctionDef" || node->kind == "AsyncFunctionDef" ||
      node->kind == "ClassDef" || node->kind == "Lambda")
    return;
  if (node->kind == "Call") {
    // Structural-mutation candidates (`x.append(...)`) rebind `x` at the
    // emitter when the manifest declares the method a structural mutator.
    // This syntactic pre-pass only over-approximates which locals may be
    // reassigned; threading a local that ends up not reassigned is a benign
    // identity forward.
    if (const parser::Node *func = ast::node(*node, "func")) {
      if (func->kind == "Attribute") {
        // The full manifest structural-mutator surface, not just
        // append/add: `xs.extend(...)` / `d.update(...)` (also the `|=`
        // desugar) inside a try silently kept the pre-try value because the
        // rebound receiver never joined the post-try lanes.
        if (auto attr = ast::string(*func, "attr");
            attr && (*attr == "append" || *attr == "add" ||
                     *attr == "extend" || *attr == "insert" ||
                     *attr == "update" ||
                     *attr == "intersection_update" ||
                     *attr == "difference_update" ||
                     *attr == "symmetric_difference_update")) {
          if (const parser::Node *value = ast::node(*func, "value"))
            if (value->kind == "Name")
              names.insert(ast::nameSpelling(*value));
        }
      }
    }
  }
  if (node->kind == "Assign") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets) {
        collectAssignedNameTargets(target.get(), names);
        // `x[k] = v` may structurally mutate (and rebind) `x`; threading a
        // local that ends up not reassigned is a benign identity forward.
        if (target && target->kind == "Subscript")
          if (const parser::Node *container = ast::node(*target, "value"))
            if (container->kind == "Name")
              names.insert(ast::nameSpelling(*container));
      }
  } else if (node->kind == "AnnAssign" || node->kind == "AugAssign" ||
             node->kind == "NamedExpr") {
    collectAssignedNameTargets(ast::node(*node, "target"), names);
  } else if (node->kind == "For" || node->kind == "AsyncFor") {
    collectAssignedNameTargets(ast::node(*node, "target"), names);
  } else if (node->kind == "With" || node->kind == "AsyncWith") {
    if (const auto *items = ast::nodeList(*node, "items"))
      for (const parser::NodePtr &item : *items)
        collectAssignedNameTargets(ast::node(*item, "optional_vars"), names);
  }

  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child)
        collectAssignedNames(child->get(), names);
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child)
          collectAssignedNames(child.get(), names);
    }
  }
}

// Does anything after this statement in the same suite READ `name`? A store
// is not a read: `x[i] = v` reads `x`, `x = v` does not.
bool containsNameLoad(const parser::Node *node, llvm::StringRef name) {
  if (!node)
    return false;
  if (node->kind == "Name")
    return llvm::StringRef(ast::nameSpelling(*node)) == name;
  llvm::SmallPtrSet<const parser::Node *, 4> stores;
  auto noteStoreTarget = [&](const parser::Node *target) {
    if (target && target->kind == "Name")
      stores.insert(target);
  };
  if (node->kind == "Assign") {
    if (const auto *targets = ast::nodeList(*node, "targets"))
      for (const parser::NodePtr &target : *targets)
        noteStoreTarget(target.get());
  } else if (node->kind == "AnnAssign" || node->kind == "NamedExpr" ||
             node->kind == "For" || node->kind == "AsyncFor") {
    noteStoreTarget(ast::node(*node, "target"));
  } else if (node->kind == "With" || node->kind == "AsyncWith") {
    if (const auto *items = ast::nodeList(*node, "items"))
      for (const parser::NodePtr &item : *items)
        noteStoreTarget(ast::node(*item, "optional_vars"));
  }

  for (const parser::Field &field : node->fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && !stores.contains(child->get()) &&
          containsNameLoad(child->get(), name))
        return true;
    } else if (const auto *children =
                   std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
      for (const parser::NodePtr &child : *children)
        if (child && !stores.contains(child.get()) &&
            containsNameLoad(child.get(), name))
          return true;
    }
  }
  return false;
}

void collectAssignedNames(const std::vector<parser::NodePtr> *statements,
                          llvm::StringSet<> &names) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements)
    collectAssignedNames(statement.get(), names);
}

} // namespace lython::emitter
