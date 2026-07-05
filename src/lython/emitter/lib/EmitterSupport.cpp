#include "EmitterSupport.h"

#include "AstAccess.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/STLExtras.h"

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
  mlir::MLIRContext *context = type.getContext();
  if (mlir::isa<py::SelfType>(type))
    return selfType;
  if (auto contract = mlir::dyn_cast<py::ContractType>(type)) {
    llvm::SmallVector<mlir::Type, 4> arguments;
    bool changed = false;
    for (mlir::Type argument : contract.getArguments()) {
      mlir::Type replaced = replaceSelfType(argument, selfType);
      changed |= replaced != argument;
      arguments.push_back(replaced);
    }
    if (changed)
      return py::ContractType::get(context, contract.getContractName(),
                                   arguments);
    return type;
  }
  if (auto protocol = mlir::dyn_cast<py::ProtocolType>(type)) {
    llvm::SmallVector<mlir::Type, 4> arguments;
    bool changed = false;
    for (mlir::Type argument : protocol.getArguments()) {
      mlir::Type replaced = replaceSelfType(argument, selfType);
      changed |= replaced != argument;
      arguments.push_back(replaced);
    }
    if (changed)
      return py::ProtocolType::get(context, protocol.getProtocolName(),
                                   arguments);
    return type;
  }
  if (auto callable = mlir::dyn_cast<py::CallableType>(type)) {
    llvm::SmallVector<mlir::Type, 8> positional;
    llvm::SmallVector<mlir::Type, 4> kwonly;
    llvm::SmallVector<mlir::Type, 1> results;
    for (mlir::Type argument : callable.getPositionalTypes())
      positional.push_back(replaceSelfType(argument, selfType));
    for (mlir::Type argument : callable.getKwOnlyTypes())
      kwonly.push_back(replaceSelfType(argument, selfType));
    for (mlir::Type result : callable.getResultTypes())
      results.push_back(replaceSelfType(result, selfType));
    mlir::Type vararg =
        callable.hasVararg()
            ? replaceSelfType(callable.getVarargType(), selfType)
            : mlir::Type();
    mlir::Type kwarg = callable.hasKwarg()
                           ? replaceSelfType(callable.getKwargType(), selfType)
                           : mlir::Type();
    return py::CallableType::get(
        context, positional, kwonly, vararg, kwarg, results,
        callable.getPositionalNames(), callable.getKwOnlyNames(),
        callable.getPositionalDefaults(), callable.getKwOnlyDefaults(),
        callable.getVarargName(), callable.getKwargName(),
        callable.getPositionalOnlyCount());
  }
  if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
    llvm::SmallVector<mlir::Type, 4> members;
    bool changed = false;
    for (mlir::Type member : unionType.getMemberTypes()) {
      mlir::Type replaced = replaceSelfType(member, selfType);
      changed |= replaced != member;
      members.push_back(replaced);
    }
    return changed ? py::UnionType::getNormalized(context, members) : type;
  }
  if (auto typeObject = mlir::dyn_cast<py::TypeType>(type)) {
    mlir::Type instance =
        replaceSelfType(typeObject.getInstanceType(), selfType);
    return instance != typeObject.getInstanceType()
               ? py::TypeType::get(context, instance)
               : type;
  }
  if (auto unpack = mlir::dyn_cast<py::UnpackType>(type)) {
    mlir::Type packed = replaceSelfType(unpack.getPackedType(), selfType);
    return packed != unpack.getPackedType()
               ? py::UnpackType::get(context, packed)
               : type;
  }
  return type;
}

void replaceSelfInSignature(FunctionSignature &sig, mlir::Type selfType,
                            AlgorithmM &types) {
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

mlir::Type elementType(mlir::Type type, AlgorithmM &types) {
  if (auto contract = mlir::dyn_cast_or_null<py::ContractType>(type)) {
    if (contract.getContractName() == "builtins.str")
      return types.strType();
    if (!contract.getArguments().empty())
      return contract.getArguments().front();
  }
  if (auto literal = mlir::dyn_cast_or_null<py::LiteralType>(type)) {
    llvm::StringRef spelling = literal.getSpelling();
    if (spelling.starts_with("\"") && spelling.ends_with("\""))
      return types.strType();
  }
  if (auto protocol = mlir::dyn_cast_or_null<py::ProtocolType>(type)) {
    if (!protocol.getArguments().empty())
      return protocol.getArguments().front();
  }
  return types.object();
}

void appendStarredArgumentTypes(mlir::Type type, AlgorithmM &types,
                                llvm::SmallVectorImpl<mlir::Type> &out) {
  type = types.widenLiteral(type);
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    if (contract.getContractName() == "builtins.tuple") {
      llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
      if (arguments.empty())
        out.push_back(types.object());
      else if (arguments.size() == 1)
        out.push_back(arguments.front());
      else
        out.append(arguments.begin(), arguments.end());
      return;
    }
  }
  out.push_back(types.object());
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
  if (const auto *fieldValue = ast::field(*node, "value"))
    if (const auto *big = std::get_if<parser::BigInteger>(fieldValue))
      return dict("int", builder.getStringAttr(big->decimal));
  return dict("unsupported", builder.getStringAttr("Constant"));
}

mlir::ArrayAttr callableDefaultValues(mlir::Builder &builder,
                                      const parser::Node &function,
                                      const FunctionSignature &sig) {
  unsigned positionalCount = static_cast<unsigned>(sig.positionalTypes.size());
  llvm::SmallVector<mlir::Attribute, 8> values(
      positionalCount + sig.kwOnlyTypes.size(), builder.getUnitAttr());
  const parser::Node *arguments = ast::node(function, "args");
  const auto *defaults =
      arguments ? ast::nodeList(*arguments, "defaults") : nullptr;
  if (defaults && !defaults->empty()) {
    unsigned firstDefault = positionalCount - defaults->size();
    for (auto [index, value] : llvm::enumerate(*defaults))
      if (firstDefault + index < values.size())
        values[firstDefault + index] = defaultValueAttr(builder, value.get());
  }
  const auto *kwDefaults =
      arguments ? ast::nodeList(*arguments, "kw_defaults") : nullptr;
  if (kwDefaults) {
    for (auto [index, value] : llvm::enumerate(*kwDefaults)) {
      unsigned slot = positionalCount + static_cast<unsigned>(index);
      if (slot < values.size())
        values[slot] = defaultValueAttr(builder, value.get());
    }
  }
  return builder.getArrayAttr(values);
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

void ensureYield(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Block &block) {
  if (!blockHasTerminator(block)) {
    builder.setInsertionPointToEnd(&block);
    builder.create<mlir::scf::YieldOp>(loc);
  }
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

bool containsReturnStatement(const std::vector<parser::NodePtr> *statements) {
  if (!statements)
    return false;
  for (const parser::NodePtr &statement : *statements) {
    if (!statement)
      continue;
    if (statement->kind == "Return")
      return true;
    if (statement->kind == "FunctionDef" ||
        statement->kind == "AsyncFunctionDef" || statement->kind == "ClassDef")
      continue;
    if (containsReturnStatement(ast::nodeList(*statement, "body")) ||
        containsReturnStatement(ast::nodeList(*statement, "orelse")) ||
        containsReturnStatement(ast::nodeList(*statement, "finalbody")))
      return true;
    if (const auto *handlers = ast::nodeList(*statement, "handlers"))
      for (const parser::NodePtr &handler : *handlers)
        if (handler && containsReturnStatement(ast::nodeList(*handler, "body")))
          return true;
  }
  return false;
}

bool containsObjectTop(mlir::Type type, const AlgorithmM &types) {
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

mlir::Type removeNoneFromType(mlir::Type type, AlgorithmM &types) {
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
optionalNoneComparison(const parser::Node &test, AlgorithmM &types) {
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

mlir::Type widenInferredLiterals(mlir::Type type, const AlgorithmM &types) {
  if (!type)
    return type;
  mlir::Type widened = types.widenLiteral(type);
  if (widened != type)
    return widened;
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type)) {
    llvm::SmallVector<mlir::Type, 4> args;
    for (mlir::Type arg : contract.getArguments())
      args.push_back(widenInferredLiterals(arg, types));
    return py::ContractType::get(type.getContext(), contract.getContractName(),
                                 args);
  }
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type)) {
    llvm::SmallVector<mlir::Type, 4> args;
    for (mlir::Type arg : protocol.getArguments())
      args.push_back(widenInferredLiterals(arg, types));
    return py::ProtocolType::get(type.getContext(), protocol.getProtocolName(),
                                 args);
  }
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type)) {
    return py::TypeType::get(
        type.getContext(),
        widenInferredLiterals(typeType.getInstanceType(), types));
  }
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type)) {
    llvm::SmallVector<mlir::Type, 4> members;
    for (mlir::Type member : unionType.getMemberTypes())
      members.push_back(widenInferredLiterals(member, types));
    return py::UnionType::getNormalized(type.getContext(), members);
  }
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type)) {
    llvm::SmallVector<mlir::Type, 8> positional;
    llvm::SmallVector<mlir::Type, 4> kwonly;
    llvm::SmallVector<mlir::Type, 1> results;
    for (mlir::Type arg : callable.getPositionalTypes())
      positional.push_back(widenInferredLiterals(arg, types));
    for (mlir::Type arg : callable.getKwOnlyTypes())
      kwonly.push_back(widenInferredLiterals(arg, types));
    for (mlir::Type result : callable.getResultTypes())
      results.push_back(widenInferredLiterals(result, types));
    mlir::Type vararg =
        callable.hasVararg()
            ? widenInferredLiterals(callable.getVarargType(), types)
            : mlir::Type();
    mlir::Type kwarg =
        callable.hasKwarg()
            ? widenInferredLiterals(callable.getKwargType(), types)
            : mlir::Type();
    return py::CallableType::get(
        type.getContext(), positional, kwonly, vararg, kwarg, results,
        callable.getPositionalNames(), callable.getKwOnlyNames(),
        callable.getPositionalDefaults(), callable.getKwOnlyDefaults(),
        callable.getVarargName(), callable.getKwargName(),
        callable.getPositionalOnlyCount());
  }
  return type;
}

bool hasUnexpectedObjectTop(mlir::Type actual, mlir::Type expected,
                            const AlgorithmM &types) {
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

} // namespace lython::emitter
