#include "EmitterCore.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"

#include "cpp/PyProtocols.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include "PyDialect.h.inc"
#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace lython::emitter {
namespace {

constexpr llvm::StringLiteral kCallableVarargValueTypeAttr{
    "callable_vararg_value_type"};
constexpr llvm::StringLiteral kCallableKwargValueTypeAttr{
    "callable_kwarg_value_type"};
constexpr llvm::StringLiteral kPackUnpackedOperandsAttr{"ly.unpack_operands"};

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
  if (auto list = mlir::dyn_cast<py::ListType>(type)) {
    mlir::Type element = replaceSelfType(list.getElementType(), selfType);
    return element != list.getElementType()
               ? py::ListType::get(context, element)
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
    if (contract.getContractName() == "builtins.tuple" &&
        !contract.getArguments().empty()) {
      out.push_back(contract.getArguments().front());
      return;
    }
  }
  if (auto tuple = mlir::dyn_cast_if_present<py::TupleType>(type)) {
    out.append(tuple.getElementTypes().begin(), tuple.getElementTypes().end());
    return;
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
  if (mlir::isa_and_nonnull<py::NoneType>(type))
    return true;
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

struct NoneComparisonNarrowing {
  std::string name;
  bool trueBranchIsNone = true;
  mlir::Type payloadType;
};

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

} // namespace

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
                      mlir::scf::SCFDialect>();
  types.seedBuiltins();

  module = mlir::ModuleOp::create(builder.getUnknownLoc());
  module.setName(moduleName);
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

void ModuleEmitter::predeclareTopLevel() {
  if (const auto *body = ast::nodeList(moduleNode, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "Import" || statement->kind == "ImportFrom") {
        bindImportStatement(*statement, /*diagnoseUnsupported=*/false);
        continue;
      }
      if (statement->kind == "ClassDef")
        if (auto name = ast::string(*statement, "name"))
          types.bindClass(*name, types.contract(*name));
    }
  }
}

bool ModuleEmitter::bindImportStatement(const parser::Node &statement,
                                        bool diagnoseUnsupported) {
  if (statement.kind == "Import") {
    const auto *names = ast::nodeList(statement, "names");
    if (!names)
      return true;
    for (const parser::NodePtr &alias : *names) {
      if (!alias)
        continue;
      std::optional<std::string_view> name = ast::string(*alias, "name");
      if (!name)
        continue;
      std::optional<std::string_view> asname = ast::string(*alias, "asname");
      std::string local = importBindingName(*name, asname);
      if (!types.bindImportedModule(llvm::StringRef(*name),
                                    llvm::StringRef(local)) &&
          diagnoseUnsupported) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, alias->range.start,
            "unsupported import '" + std::string(*name) + "'"});
      }
    }
    return true;
  }

  if (statement.kind != "ImportFrom")
    return false;

  std::int64_t level = ast::integer(statement, "level").value_or(0);
  std::optional<std::string_view> module = ast::string(statement, "module");
  if (level != 0 || !module) {
    if (diagnoseUnsupported)
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, statement.range.start,
          "relative import is not supported by the static emitter"});
    return true;
  }
  const auto *names = ast::nodeList(statement, "names");
  if (!names)
    return true;
  for (const parser::NodePtr &alias : *names) {
    if (!alias)
      continue;
    std::optional<std::string_view> name = ast::string(*alias, "name");
    if (!name || *name == "*") {
      if (diagnoseUnsupported)
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, alias->range.start,
            "star import is not supported by the static emitter"});
      continue;
    }
    std::optional<std::string_view> asname = ast::string(*alias, "asname");
    llvm::StringRef local =
        asname ? llvm::StringRef(*asname) : llvm::StringRef(*name);
    if (!types.bindImportedName(llvm::StringRef(*module),
                                llvm::StringRef(*name), local) &&
        diagnoseUnsupported) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, alias->range.start,
                             "unsupported import '" + std::string(*module) +
                                 "." + std::string(*name) + "'"});
    }
  }
  return true;
}

void ModuleEmitter::emitTopLevelDeclarations() {
  if (const auto *body = ast::nodeList(moduleNode, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement)
        continue;
      if (statement->kind == "FunctionDef" ||
          statement->kind == "AsyncFunctionDef")
        emitFunctionDecl(*statement);
      else if (statement->kind == "ClassDef")
        emitClassContract(*statement);
    }
  }
}

void ModuleEmitter::emitFunctionDecl(const parser::Node &function) {
  auto name = ast::string(function, "name");
  if (!name)
    return;
  FunctionSignature sig = types.functionSignature(function);
  emitCallableFunction(function, *name, sig, {}, /*isLambda=*/false);
  FunctionSignature publicSig =
      function.kind == "AsyncFunctionDef" ? asyncPublicSignature(sig) : sig;
  types.bindSymbol(*name, publicSig.callable);
}

void ModuleEmitter::emitCallableFunction(const parser::Node &callable,
                                         llvm::StringRef symbolName,
                                         const FunctionSignature &sig,
                                         llvm::ArrayRef<Capture> captures,
                                         bool isLambda) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  llvm::SmallVector<mlir::Type, 8> logicalInputs(sig.positionalTypes.begin(),
                                                 sig.positionalTypes.end());
  logicalInputs.append(sig.kwOnlyTypes.begin(), sig.kwOnlyTypes.end());
  if (sig.varargType)
    logicalInputs.push_back(sig.varargType);
  if (sig.kwargType)
    logicalInputs.push_back(sig.kwargType);
  for (const Capture &capture : captures)
    logicalInputs.push_back(capture.value.type);

  auto funcType =
      builder.getFunctionType(logicalInputs, mlir::TypeRange{sig.resultType});
  auto func =
      builder.create<mlir::func::FuncOp>(loc(callable), symbolName, funcType);
  func.setPrivate();
  func->setAttr("callable_type", mlir::TypeAttr::get(sig.callable));
  if (sig.varargType)
    func->setAttr(kCallableVarargValueTypeAttr,
                  mlir::TypeAttr::get(sig.varargType));
  if (sig.kwargType)
    func->setAttr(kCallableKwargValueTypeAttr,
                  mlir::TypeAttr::get(sig.kwargType));
  func->setAttr("callable_default_values",
                callableDefaultValues(builder, callable, sig));
  if (callable.kind == "AsyncFunctionDef")
    func->setAttr("ly.async.body_result", mlir::TypeAttr::get(sig.resultType));
  if (!captures.empty()) {
    llvm::SmallVector<std::string, 4> captureNames;
    llvm::SmallVector<mlir::Type, 4> captureTypes;
    for (const Capture &capture : captures) {
      captureNames.push_back(capture.name);
      captureTypes.push_back(capture.value.type);
    }
    func->setAttr("closure_names", stringArray(builder, captureNames));
    func->setAttr("closure_types", typeArray(builder, captureTypes));
  }

  llvm::StringMap<Value> savedValues = values;
  mlir::Type savedReturnType = currentReturnType;
  std::string savedFunctionPrefix = currentFunctionPrefix;
  auto typeScope = types.pushScope();

  mlir::Block *entry = func.addEntryBlock();
  values.clear();
  currentReturnType = sig.resultType;
  currentFunctionPrefix = symbolName.str();
  types.bindSymbol(symbolName, sig.callable);
  if (const parser::Node *arguments = ast::node(callable, "args")) {
    llvm::SmallVector<const parser::Node *, 8> positional =
        positionalArgumentNodes(*arguments);
    for (auto [index, argument] : llvm::enumerate(positional)) {
      if (index >= sig.positionalTypes.size() ||
          index >= entry->getNumArguments())
        break;
      llvm::StringRef name = ast::nameSpelling(*argument);
      values[name] =
          Value{entry->getArgument(index), sig.positionalTypes[index]};
      types.bindSymbol(name, sig.positionalTypes[index]);
    }
    if (const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs")) {
      unsigned offset = static_cast<unsigned>(sig.positionalTypes.size());
      for (auto [index, argument] : llvm::enumerate(*kwonly)) {
        if (!argument || index >= sig.kwOnlyTypes.size() ||
            offset + index >= entry->getNumArguments())
          break;
        llvm::StringRef name = ast::nameSpelling(*argument);
        values[name] =
            Value{entry->getArgument(offset + index), sig.kwOnlyTypes[index]};
        types.bindSymbol(name, sig.kwOnlyTypes[index]);
      }
    }
    unsigned variadicOffset = static_cast<unsigned>(sig.positionalTypes.size() +
                                                    sig.kwOnlyTypes.size());
    if (sig.varargType) {
      if (sig.varargName && variadicOffset < entry->getNumArguments()) {
        values[*sig.varargName] =
            Value{entry->getArgument(variadicOffset), sig.varargType};
        types.bindSymbol(*sig.varargName, sig.varargType);
      }
      ++variadicOffset;
    }
    if (sig.kwargType) {
      if (sig.kwargName && variadicOffset < entry->getNumArguments()) {
        values[*sig.kwargName] =
            Value{entry->getArgument(variadicOffset), sig.kwargType};
        types.bindSymbol(*sig.kwargName, sig.kwargType);
      }
    }
  }
  unsigned captureOffset = static_cast<unsigned>(
      sig.positionalTypes.size() + sig.kwOnlyTypes.size() +
      (sig.varargType ? 1 : 0) + (sig.kwargType ? 1 : 0));
  for (auto [index, capture] : llvm::enumerate(captures)) {
    values[capture.name] =
        Value{entry->getArgument(captureOffset + index), capture.value.type};
    types.bindSymbol(capture.name, capture.value.type);
  }

  builder.setInsertionPointToStart(entry);
  if (isLambda) {
    Value body = coerceValue(emitExpr(ast::node(callable, "body")),
                             currentReturnType, callable);
    builder.create<mlir::func::ReturnOp>(loc(callable), body.value);
  } else {
    emitStatements(ast::nodeList(callable, "body"));
  }
  if (!insertionBlockTerminated(builder)) {
    Value none = emitNone(callable);
    Value result = coerceValue(none, currentReturnType, callable);
    builder.create<mlir::func::ReturnOp>(loc(callable), result.value);
  }

  values = std::move(savedValues);
  currentReturnType = savedReturnType;
  currentFunctionPrefix = std::move(savedFunctionPrefix);
}

std::optional<ModuleEmitter::MethodBinding>
ModuleEmitter::lookupClassMethod(mlir::Type receiverType,
                                 llvm::StringRef methodName) const {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  auto classMethods = classMethodBindings.find(contract.getContractName());
  if (classMethods == classMethodBindings.end())
    return std::nullopt;
  auto method = classMethods->second.find(methodName);
  if (method == classMethods->second.end())
    return std::nullopt;
  return method->second;
}

std::optional<mlir::Type>
ModuleEmitter::lookupClassField(mlir::Type receiverType,
                                llvm::StringRef fieldName) const {
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(receiverType);
  if (!contract)
    return std::nullopt;
  auto classFields = classFieldBindings.find(contract.getContractName());
  if (classFields == classFieldBindings.end())
    return std::nullopt;
  auto field = classFields->second.find(fieldName);
  if (field == classFields->second.end())
    return std::nullopt;
  return field->second;
}

ModuleEmitter::Value
ModuleEmitter::emitNestedFunctionDecl(const parser::Node &function) {
  auto name = ast::string(function, "name");
  if (!name)
    return emitNone(function);

  llvm::SmallVector<Capture, 4> captures;
  for (const std::string &captureName : lexicalCaptureNames(function)) {
    auto found = values.find(captureName);
    if (found != values.end())
      captures.push_back(Capture{captureName, found->second});
  }

  FunctionSignature sig = types.functionSignature(function);
  FunctionSignature publicSig =
      function.kind == "AsyncFunctionDef" ? asyncPublicSignature(sig) : sig;
  std::string symbolName =
      (llvm::Twine(currentFunctionPrefix.empty() ? "__main__"
                                                 : currentFunctionPrefix) +
       "$" + sanitizedSymbolPart(*name) + "$" +
       llvm::Twine(++syntheticFunctionCounter) + "$" +
       llvm::Twine(function.range.start.line) + "_" +
       llvm::Twine(function.range.start.column))
          .str();
  emitCallableFunction(function, symbolName, sig, captures, /*isLambda=*/false);
  return emitFunctionObject(function, symbolName, publicSig.callable, captures);
}

ModuleEmitter::Value ModuleEmitter::emitLambda(const parser::Node &expr,
                                               py::CallableType expected) {
  llvm::SmallVector<Capture, 4> captures;
  for (const std::string &captureName : lexicalCaptureNames(expr)) {
    auto found = values.find(captureName);
    if (found != values.end())
      captures.push_back(Capture{captureName, found->second});
  }

  FunctionSignature sig = types.functionSignature(expr, std::nullopt, expected);
  if (expected) {
    if (sig.positionalTypes.size() != expected.getPositionalTypes().size() ||
        sig.kwOnlyTypes.size() != expected.getKwOnlyTypes().size() ||
        expected.hasVararg() || expected.hasKwarg()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "lambda does not match its Callable annotation shape"});
    }
    bool unresolvedUnknown =
        hasUnexpectedObjectTop(sig.callable, expected, types);
    if (unresolvedUnknown) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "lambda Callable annotation does not resolve all Unknown types"});
    }
    if (!unresolvedUnknown &&
        !py::isAssignableTo(widenInferredLiterals(sig.callable, types),
                            expected)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "lambda body is not compatible with its Callable annotation"});
    }
  } else if (containsObjectTop(sig.callable, types)) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lambda requires a Callable annotation because its type contains "
        "unresolved Unknown"});
  }

  std::string symbolName =
      (llvm::Twine(currentFunctionPrefix.empty() ? "__main__"
                                                 : currentFunctionPrefix) +
       "$lambda$" + llvm::Twine(++syntheticFunctionCounter) + "$" +
       llvm::Twine(expr.range.start.line) + "_" +
       llvm::Twine(expr.range.start.column))
          .str();
  emitCallableFunction(expr, symbolName, sig, captures, /*isLambda=*/true);
  return emitFunctionObject(expr, symbolName, sig.callable, captures);
}

void ModuleEmitter::emitClassContract(const parser::Node &classDef) {
  auto name = ast::string(classDef, "name");
  if (!name)
    return;

  llvm::SmallVector<llvm::StringRef, 4> bases;
  if (const auto *baseNodes = ast::nodeList(classDef, "bases")) {
    for (const parser::NodePtr &base : *baseNodes)
      bases.push_back(ast::nameSpelling(*base));
  }

  llvm::StringMap<mlir::Type> fieldMap;
  collectClassFields(classDef, fieldMap);
  llvm::SmallVector<std::string, 8> fieldNames;
  llvm::SmallVector<mlir::Type, 8> fieldTypes;
  llvm::StringMap<mlir::Type> &registeredFields = classFieldBindings[*name];
  registeredFields.clear();
  for (const auto &entry : fieldMap) {
    fieldNames.push_back(entry.getKey().str());
    fieldTypes.push_back(entry.getValue());
    registeredFields[entry.getKey()] = entry.getValue();
  }

  llvm::SmallVector<std::string, 8> methodNames;
  llvm::SmallVector<std::string, 8> methodKinds;
  llvm::SmallVector<mlir::Type, 8> methodContracts;
  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &statement : *body) {
      if (!statement || (statement->kind != "FunctionDef" &&
                         statement->kind != "AsyncFunctionDef"))
        continue;
      auto methodName = ast::string(*statement, "name");
      if (!methodName)
        continue;
      std::string kind = methodKind(*statement);
      FunctionSignature sig = types.functionSignature(
          *statement, kind == "static"
                          ? std::optional<llvm::StringRef>()
                          : std::optional<llvm::StringRef>("self"));
      if (kind == "instance")
        replaceSelfInSignature(sig, types.contract(*name), types);
      else if (kind == "class")
        replaceSelfInSignature(sig, types.typeObject(types.contract(*name)),
                               types);
      if (statement->kind == "AsyncFunctionDef")
        sig = asyncPublicSignature(sig);
      methodNames.push_back(std::string(*methodName));
      methodKinds.push_back(std::move(kind));
      methodContracts.push_back(sig.callable);

      classMethodBindings[*name][*methodName] =
          MethodBinding{statement.get(), sig};
    }
  }

  mlir::OperationState state(loc(classDef), py::ClassOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(*name));
  state.addAttribute("base_names", stringArray(builder, bases));
  state.addAttribute("field_names", stringArray(builder, fieldNames));
  state.addAttribute("field_types", typeArray(builder, fieldTypes));
  state.addAttribute("field_contract_types", typeArray(builder, fieldTypes));
  state.addAttribute("method_names", stringArray(builder, methodNames));
  state.addAttribute("method_contracts", typeArray(builder, methodContracts));
  state.addAttribute("method_kinds", stringArray(builder, methodKinds));
  state.addRegion();
  mlir::Operation *op = builder.create(state);
  op->getRegion(0).push_back(new mlir::Block);
}

void ModuleEmitter::collectClassFields(
    const parser::Node &classDef, llvm::StringMap<mlir::Type> &fields) const {
  auto collectInitArgTypes = [&](const parser::Node &method,
                                 llvm::StringMap<mlir::Type> &argTypes) {
    const parser::Node *arguments = ast::node(method, "args");
    if (!arguments)
      return;
    auto collectArgs = [&](llvm::StringRef fieldName) {
      if (const auto *args = ast::nodeList(*arguments, fieldName)) {
        for (const parser::NodePtr &arg : *args) {
          if (!arg)
            continue;
          llvm::StringRef name = ast::nameSpelling(*arg);
          if (name == "self")
            continue;
          argTypes[name] = types.annotationType(ast::node(*arg, "annotation"));
        }
      }
    };
    collectArgs("posonlyargs");
    collectArgs("args");
  };

  auto collectTarget = [&](const parser::Node &target, mlir::Type type) {
    if (target.kind != "Attribute")
      return;
    const parser::Node *object = ast::node(target, "value");
    if (!object || !ast::isName(*object, "self"))
      return;
    if (auto attr = ast::string(target, "attr"))
      fields[*attr] = type ? type : types.object();
  };

  if (const auto *body = ast::nodeList(classDef, "body")) {
    for (const parser::NodePtr &method : *body) {
      if (!method || ast::nameSpelling(*method) != "__init__")
        continue;
      llvm::StringMap<mlir::Type> initArgTypes;
      collectInitArgTypes(*method, initArgTypes);
      if (const auto *stmts = ast::nodeList(*method, "body")) {
        for (const parser::NodePtr &stmt : *stmts) {
          if (!stmt)
            continue;
          if (stmt->kind == "AnnAssign") {
            collectTarget(*ast::node(*stmt, "target"),
                          types.annotationType(ast::node(*stmt, "annotation")));
          } else if (stmt->kind == "Assign") {
            const parser::Node *value = ast::node(*stmt, "value");
            mlir::Type valueType = types.inferExpr(value);
            if (value && value->kind == "Name") {
              auto found = initArgTypes.find(ast::nameSpelling(*value));
              if (found != initArgTypes.end())
                valueType = found->second;
            }
            if (const auto *targets = ast::nodeList(*stmt, "targets"))
              for (const parser::NodePtr &target : *targets)
                collectTarget(*target, valueType);
          }
        }
      }
    }
  }
}

void ModuleEmitter::emitStatements(
    const std::vector<parser::NodePtr> *statements, bool skipDeclarations) {
  if (!statements)
    return;
  for (const parser::NodePtr &statement : *statements) {
    if (insertionBlockTerminated(builder))
      break;
    if (statement && (!skipDeclarations || !isTopLevelDecl(*statement)))
      emitStatement(*statement);
  }
}

void ModuleEmitter::emitStatement(const parser::Node &statement) {
  if (statement.kind == "Expr") {
    emitExpr(ast::node(statement, "value"));
  } else if (statement.kind == "Import") {
    bindImportStatement(statement, /*diagnoseUnsupported=*/true);
  } else if (statement.kind == "ImportFrom") {
    bindImportStatement(statement, /*diagnoseUnsupported=*/true);
  } else if (statement.kind == "Assign") {
    const parser::Node *rhs = ast::node(statement, "value");
    Value value{{}, {}};
    bool emittedWithContext = false;
    if (rhs && rhs->kind == "Lambda") {
      if (const auto *targets = ast::nodeList(statement, "targets")) {
        if (targets->size() == 1 && targets->front() &&
            targets->front()->kind == "Name") {
          llvm::StringRef name = ast::nameSpelling(*targets->front());
          if (auto expectedType = types.lookupSymbol(name)) {
            if (auto expectedCallable =
                    mlir::dyn_cast_if_present<py::CallableType>(
                        *expectedType)) {
              value = emitLambda(*rhs, expectedCallable);
              emittedWithContext = true;
            }
          }
        }
      }
    }
    if (!emittedWithContext)
      value = emitExpr(rhs);
    if (const auto *targets = ast::nodeList(statement, "targets"))
      for (const parser::NodePtr &target : *targets)
        emitAssignTarget(*target, value);
  } else if (statement.kind == "AnnAssign") {
    mlir::Type annotated =
        types.annotationType(ast::node(statement, "annotation"));
    if (const parser::Node *rhs = ast::node(statement, "value")) {
      Value raw =
          rhs->kind == "Lambda"
              ? emitLambda(*rhs, mlir::dyn_cast_if_present<py::CallableType>(
                                     annotated))
              : emitExpr(rhs);
      Value value = coerceValue(raw, annotated, statement);
      emitAssignTarget(*ast::node(statement, "target"), value);
      return;
    }
    const parser::Node *target = ast::node(statement, "target");
    if (target && target->kind == "Name")
      types.bindSymbol(ast::nameSpelling(*target), annotated);
  } else if (statement.kind == "AugAssign") {
    Value lhs = emitExpr(ast::node(statement, "target"));
    Value rhs = emitExpr(ast::node(statement, "value"));
    Value value = emitBinarySpecial<py::AddOp>(
        statement, "__add__", lhs, rhs,
        types.widenLiteral(types.join({lhs.type, rhs.type})));
    emitAssignTarget(*ast::node(statement, "target"), value);
  } else if (statement.kind == "If") {
    emitIf(statement);
  } else if (statement.kind == "For") {
    emitFor(statement);
  } else if (statement.kind == "AsyncFor") {
    emitAsyncFor(statement);
  } else if (statement.kind == "With") {
    emitWith(statement, false);
  } else if (statement.kind == "AsyncWith") {
    emitWith(statement, true);
  } else if (statement.kind == "Raise") {
    if (const parser::Node *exception = ast::node(statement, "exc")) {
      Value value = emitExpr(exception);
      builder.create<py::RaiseOp>(loc(statement), value.value);
    } else {
      builder.create<py::RaiseCurrentOp>(loc(statement));
    }
  } else if (statement.kind == "FunctionDef" ||
             statement.kind == "AsyncFunctionDef") {
    Value function = emitNestedFunctionDecl(statement);
    if (auto name = ast::string(statement, "name")) {
      values[*name] = function;
      types.bindSymbol(*name, function.type);
    }
  } else if (statement.kind == "Return") {
    const parser::Node *returnValue = ast::node(statement, "value");
    Value value = returnValue && returnValue->kind == "Lambda"
                      ? emitLambda(*returnValue,
                                   mlir::dyn_cast_if_present<py::CallableType>(
                                       currentReturnType))
                      : emitExpr(returnValue);
    if (!inlineReturnContexts.empty()) {
      InlineReturnContext &ctx = inlineReturnContexts.back();
      Value result = ctx.resultType
                         ? coerceValue(value, ctx.resultType, statement)
                         : value;
      emitActiveCleanups(statement);
      builder.create<mlir::cf::BranchOp>(loc(statement), ctx.target,
                                         result.value);
      return;
    }
    if (currentReturnType) {
      Value result = coerceValue(value, currentReturnType, statement);
      emitActiveCleanups(statement);
      builder.create<mlir::func::ReturnOp>(loc(statement), result.value);
    }
  } else if (statement.kind == "Pass") {
    return;
  } else if (statement.kind == "Try") {
    emitTry(statement);
  } else if (statement.kind == "TryStar") {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "except* requires exception-group-aware py.try lowering"});
  } else {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, statement.range.start,
        "unsupported statement kind '" + statement.kind + "'"});
  }
}

void ModuleEmitter::emitAssignTarget(const parser::Node &target, Value value) {
  if (target.kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(target);
    values[name] = value;
    types.bindSymbol(name, value.type);
    return;
  }
  if (target.kind == "Attribute") {
    Value object = emitExpr(ast::node(target, "value"));
    if (auto attr = ast::string(target, "attr"))
      builder.create<py::AttrSetOp>(loc(target), object.value, *attr,
                                    value.value);
    return;
  }
  if (target.kind == "Subscript") {
    Value container = emitExpr(ast::node(target, "value"));
    Value index = emitExpr(ast::node(target, "slice"));
    CallInferenceResult inference = types.inferMethodCallWithEvidence(
        container.type, "__setitem__", {index.type, value.type});
    builder.create<py::SetItemOp>(
        loc(target), mlir::FlatSymbolRefAttr::get(&context, "__setitem__"),
        callProtocolFor(inference), container.value, index.value, value.value);
    return;
  }
  if (target.kind == "Tuple" || target.kind == "List") {
    if (const auto *elts = ast::nodeList(target, "elts")) {
      for (auto [index, elt] : llvm::enumerate(*elts)) {
        Value indexValue{builder
                             .create<py::IntConstantOp>(
                                 loc(*elt),
                                 types.literal(std::to_string(index)),
                                 builder.getStringAttr(std::to_string(index)))
                             .getResult(),
                         types.literal(std::to_string(index))};
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            value.type, "__getitem__", {indexValue.type});
        auto getItem = builder.create<py::GetItemOp>(
            loc(*elt), types.object(),
            mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
            callProtocolFor(inference), value.value, indexValue.value);
        Value item{getItem.getResult(), types.object()};
        emitAssignTarget(*elt, item);
      }
    }
  }
}

void ModuleEmitter::emitIf(const parser::Node &statement) {
  const parser::Node *test = ast::node(statement, "test");
  std::optional<NoneComparisonNarrowing> narrowing =
      test ? optionalNoneComparison(*test, types) : std::nullopt;
  auto applyNarrowing = [&](const NoneComparisonNarrowing &fact,
                            bool conditionIsTrue) {
    mlir::Type narrowed = conditionIsTrue == fact.trueBranchIsNone
                              ? types.none()
                              : fact.payloadType;
    if (!narrowed)
      return;
    auto found = values.find(fact.name);
    if (found != values.end()) {
      if (mlir::isa<py::UnionType>(found->second.value.getType()) &&
          found->second.value.getType() != narrowed) {
        auto unwrap = builder.create<py::UnionUnwrapOp>(
            loc(statement), narrowed, found->second.value);
        found->second.value = unwrap.getResult();
      }
      found->second.type = narrowed;
    }
    types.bindSymbol(fact.name, narrowed);
  };

  mlir::Value condition = emitBoolValue(emitExpr(test), statement);
  const auto *orelse = ast::nodeList(statement, "orelse");
  bool hasElse = orelse && !orelse->empty();
  mlir::Block *entry = builder.getInsertionBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(builder.getInsertionPoint());
  mlir::Block *thenBlock =
      builder.createBlock(region, continuation->getIterator());
  mlir::Block *elseBlock =
      hasElse ? builder.createBlock(region, continuation->getIterator())
              : continuation;

  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::cf::CondBranchOp>(loc(statement), condition, thenBlock,
                                         mlir::ValueRange{}, elseBlock,
                                         mlir::ValueRange{});

  llvm::StringMap<Value> saved = values;
  builder.setInsertionPointToStart(thenBlock);
  {
    auto typeScope = types.pushScope();
    if (narrowing)
      applyNarrowing(*narrowing, /*conditionIsTrue=*/true);
    emitStatements(ast::nodeList(statement, "body"));
  }
  bool thenTerminates = insertionBlockTerminated(builder);
  if (!thenTerminates)
    builder.create<mlir::cf::BranchOp>(loc(statement), continuation);

  values = saved;
  bool elseTerminates = false;
  if (hasElse) {
    builder.setInsertionPointToStart(elseBlock);
    {
      auto typeScope = types.pushScope();
      if (narrowing)
        applyNarrowing(*narrowing, /*conditionIsTrue=*/false);
      emitStatements(orelse);
    }
    elseTerminates = insertionBlockTerminated(builder);
    if (!elseTerminates)
      builder.create<mlir::cf::BranchOp>(loc(statement), continuation);
  }
  values = saved;
  setInsertionBeforeTerminator(builder, *continuation);
  if (narrowing && thenTerminates && !elseTerminates)
    applyNarrowing(*narrowing, /*conditionIsTrue=*/false);
  else if (narrowing && hasElse && elseTerminates && !thenTerminates)
    applyNarrowing(*narrowing, /*conditionIsTrue=*/true);
}

void ModuleEmitter::emitFor(const parser::Node &statement) {
  Value iterable = emitExpr(ast::node(statement, "iter"));
  mlir::Type elem = elementType(iterable.type, types);
  mlir::Type iteratorType = types.iteratorOf(elem);
  CallInferenceResult iterInference =
      types.inferMethodCallWithEvidence(iterable.type, "__iter__", {});
  if (iterInference)
    iteratorType = iterInference.resultType;
  mlir::UnitAttr returnedSelf =
      iteratorType == iterable.type ? builder.getUnitAttr() : mlir::UnitAttr();
  auto iterator = builder.create<py::IterOp>(
      loc(statement), iteratorType, "__iter__", callProtocolFor(iterInference),
      iterable.value, returnedSelf);
  auto whileOp = builder.create<mlir::scf::WhileOp>(
      loc(statement), mlir::TypeRange{}, mlir::ValueRange{});

  llvm::StringMap<Value> saved = values;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(before);
    CallInferenceResult nextInference =
        types.inferMethodCallWithEvidence(iteratorType, "__next__", {});
    if (nextInference)
      elem = nextInference.resultType;
    auto next = builder.create<py::NextOp>(
        loc(statement), elem, builder.getI1Type(), iteratorType, "__next__",
        callProtocolFor(nextInference), iterator.getResult());

    auto bodyIf =
        builder.create<mlir::scf::IfOp>(loc(statement), next.getValid(), false);
    mlir::Block &thenBlock = bodyIf.getThenRegion().front();
    setInsertionBeforeTerminator(builder, thenBlock);
    values = saved;
    {
      auto typeScope = types.pushScope();
      emitAssignTarget(*ast::node(statement, "target"),
                       Value{next.getElement(), elem});
      emitStatements(ast::nodeList(statement, "body"));
    }
    ensureYield(builder, loc(statement), thenBlock);

    values = saved;
    builder.setInsertionPointAfter(bodyIf);
    builder.create<mlir::scf::ConditionOp>(loc(statement), next.getValid(),
                                           mlir::ValueRange{});
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);
    builder.create<mlir::scf::YieldOp>(loc(statement));
  }

  values = saved;
  builder.setInsertionPointAfter(whileOp);
}

void ModuleEmitter::emitAsyncFor(const parser::Node &statement) {
  if (const auto *orelse = ast::nodeList(statement, "orelse")) {
    if (!orelse->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "async for/else is not implemented yet"});
      return;
    }
  }
  if (containsReturnStatement(ast::nodeList(statement, "body"))) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "return inside async for is not implemented yet"});
    return;
  }

  Value iterable = emitExpr(ast::node(statement, "iter"));
  mlir::Type iteratorType = types.protocol("AsyncIterator", {types.object()});
  CallInferenceResult iterInference =
      types.inferMethodCallWithEvidence(iterable.type, "__aiter__", {});
  if (iterInference)
    iteratorType = iterInference.resultType;
  mlir::UnitAttr returnedSelf =
      iteratorType == iterable.type ? builder.getUnitAttr() : mlir::UnitAttr();
  auto iterator = builder.create<py::AIterOp>(
      loc(statement), iteratorType, "__aiter__", callProtocolFor(iterInference),
      iterable.value, returnedSelf);

  auto whileOp = builder.create<mlir::scf::WhileOp>(
      loc(statement), mlir::TypeRange{}, mlir::ValueRange{});

  llvm::StringMap<Value> saved = values;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(before);

    mlir::OperationState tryState(loc(statement),
                                  py::TryOp::getOperationName());
    tryState.addTypes(types.boolType());
    tryState.addRegion();
    tryState.addRegion();
    tryState.addRegion();
    auto tryOp = mlir::cast<py::TryOp>(builder.create(tryState));

    mlir::Block *tryBlock = new mlir::Block;
    tryOp.getTryRegion().push_back(tryBlock);
    builder.setInsertionPointToStart(tryBlock);
    CallInferenceResult nextInference =
        types.inferMethodCallWithEvidence(iteratorType, "__anext__", {});
    mlir::Type awaitableType =
        nextInference ? nextInference.resultType
                      : types.protocol("Awaitable", {types.object()});
    auto next = builder.create<py::ANextOp>(
        loc(statement), awaitableType, "__anext__",
        callProtocolFor(nextInference), iterator.getResult());
    Value item =
        emitAwaitValue(statement, Value{next.getAwaitable(), awaitableType});
    values = saved;
    {
      auto typeScope = types.pushScope();
      emitAssignTarget(*ast::node(statement, "target"), item);
      emitStatements(ast::nodeList(statement, "body"));
    }
    if (!blockHasTerminator(*tryBlock)) {
      mlir::Type trueType = types.literal("True");
      Value trueValue{builder
                          .create<py::BoolConstantOp>(loc(statement), trueType,
                                                      builder.getBoolAttr(true))
                          .getResult(),
                      trueType};
      mlir::Value keepGoing =
          coerceValue(trueValue, types.boolType(), statement).value;
      builder.create<py::TryYieldOp>(loc(statement),
                                     mlir::ValueRange{keepGoing});
    }

    mlir::Block *checkBlock = new mlir::Block;
    mlir::Block *stopBlock = new mlir::Block;
    mlir::Block *rethrowBlock = new mlir::Block;
    tryOp.getExceptRegion().push_back(checkBlock);
    tryOp.getExceptRegion().push_back(stopBlock);
    tryOp.getExceptRegion().push_back(rethrowBlock);

    builder.setInsertionPointToStart(checkBlock);
    mlir::Type stopAsyncIteration =
        types.typeObject(types.contract("builtins.StopAsyncIteration"));
    mlir::OperationState matchState(
        loc(statement), py::ExceptCurrentMatchOp::getOperationName());
    matchState.addTypes(builder.getI1Type());
    matchState.addAttribute("handler", mlir::TypeAttr::get(stopAsyncIteration));
    auto match =
        mlir::cast<py::ExceptCurrentMatchOp>(builder.create(matchState));
    builder.create<mlir::cf::CondBranchOp>(loc(statement), match.getResult(),
                                           stopBlock, mlir::ValueRange{},
                                           rethrowBlock, mlir::ValueRange{});

    builder.setInsertionPointToStart(stopBlock);
    mlir::Type falseType = types.literal("False");
    Value falseValue{builder
                         .create<py::BoolConstantOp>(loc(statement), falseType,
                                                     builder.getBoolAttr(false))
                         .getResult(),
                     falseType};
    mlir::Value stop =
        coerceValue(falseValue, types.boolType(), statement).value;
    builder.create<py::ExceptYieldOp>(loc(statement), mlir::ValueRange{stop});

    builder.setInsertionPointToStart(rethrowBlock);
    builder.create<py::RaiseCurrentOp>(loc(statement));

    builder.setInsertionPointAfter(tryOp);
    auto keepGoing = builder.create<py::BoolOp>(
        loc(statement), builder.getI1Type(),
        mlir::FlatSymbolRefAttr::get(&context, "__bool__"), boolProtocol(),
        tryOp.getResult(0));
    builder.create<mlir::scf::ConditionOp>(
        loc(statement), keepGoing.getResult(), mlir::ValueRange{});
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);
    builder.create<mlir::scf::YieldOp>(loc(statement));
  }

  values = saved;
  builder.setInsertionPointAfter(whileOp);
}

void ModuleEmitter::emitTry(const parser::Node &statement) {
  const auto *handlers = ast::nodeList(statement, "handlers");
  if (!handlers || handlers->empty()) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "try without except is not implemented yet"});
    return;
  }
  if (const auto *orelse = ast::nodeList(statement, "orelse")) {
    if (!orelse->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "try/else is not implemented yet"});
      return;
    }
  }
  if (const auto *finalbody = ast::nodeList(statement, "finalbody")) {
    if (!finalbody->empty()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, statement.range.start,
                             "try/finally is not implemented yet"});
      return;
    }
  }
  if (containsReturnStatement(ast::nodeList(statement, "body"))) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, statement.range.start,
                           "return inside try is not implemented yet"});
    return;
  }
  if (const auto *handlersForReturn = ast::nodeList(statement, "handlers")) {
    for (const parser::NodePtr &handler : *handlersForReturn) {
      if (handler && containsReturnStatement(ast::nodeList(*handler, "body"))) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, handler->range.start,
            "return inside except handler is not implemented yet"});
        return;
      }
    }
  }

  mlir::OperationState state(loc(statement), py::TryOp::getOperationName());
  state.addRegion();
  state.addRegion();
  state.addRegion();
  mlir::Operation *rawTry = builder.create(state);
  auto tryOp = mlir::cast<py::TryOp>(rawTry);

  llvm::StringMap<Value> saved = values;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *tryBlock = new mlir::Block;
    tryOp.getTryRegion().push_back(tryBlock);
    builder.setInsertionPointToStart(tryBlock);
    values = saved;
    {
      auto typeScope = types.pushScope();
      emitStatements(ast::nodeList(statement, "body"));
    }
    if (!blockHasTerminator(*tryBlock))
      builder.create<py::TryYieldOp>(loc(statement), mlir::ValueRange{});
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    llvm::SmallVector<mlir::Block *, 8> checkBlocks;
    llvm::SmallVector<mlir::Block *, 8> bodyBlocks;
    checkBlocks.reserve(handlers->size());
    bodyBlocks.reserve(handlers->size());
    for (std::size_t index = 0; index < handlers->size(); ++index) {
      checkBlocks.push_back(new mlir::Block);
      bodyBlocks.push_back(new mlir::Block);
      tryOp.getExceptRegion().push_back(checkBlocks.back());
      tryOp.getExceptRegion().push_back(bodyBlocks.back());
    }
    mlir::Block *rethrowBlock = new mlir::Block;
    tryOp.getExceptRegion().push_back(rethrowBlock);

    for (auto [index, handlerPtr] : llvm::enumerate(*handlers)) {
      const parser::Node &handler = *handlerPtr;
      if (auto name = ast::string(handler, "name")) {
        diagnostics.push_back(
            parser::Diagnostic{parser::Severity::Error, handler.range.start,
                               "except-as binding is not implemented yet"});
        continue;
      }

      const parser::Node *typeNode = ast::node(handler, "type");
      if (!typeNode && index + 1 != handlers->size()) {
        diagnostics.push_back(
            parser::Diagnostic{parser::Severity::Error, handler.range.start,
                               "bare except must be the last handler"});
        continue;
      }
      if (typeNode && typeNode->kind == "Tuple") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, typeNode->range.start,
            "tuple exception handlers are not implemented yet"});
        continue;
      }

      mlir::Type handlerType =
          typeNode ? types.inferExpr(typeNode)
                   : types.typeObject(types.contract("builtins.BaseException"));
      if (!mlir::isa_and_nonnull<py::TypeType>(handlerType)) {
        handlerType = types.typeObject(types.annotationType(typeNode));
      }
      if (!mlir::isa_and_nonnull<py::TypeType>(handlerType)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error,
            typeNode ? typeNode->range.start : handler.range.start,
            "except handler must resolve to a Python type object"});
        continue;
      }

      builder.setInsertionPointToStart(checkBlocks[index]);
      mlir::OperationState matchState(
          loc(handler), py::ExceptCurrentMatchOp::getOperationName());
      matchState.addTypes(builder.getI1Type());
      matchState.addAttribute("handler", mlir::TypeAttr::get(handlerType));
      auto match =
          mlir::cast<py::ExceptCurrentMatchOp>(builder.create(matchState));
      mlir::Block *miss =
          index + 1 == handlers->size() ? rethrowBlock : checkBlocks[index + 1];
      builder.create<mlir::cf::CondBranchOp>(
          loc(handler), match.getResult(), bodyBlocks[index],
          mlir::ValueRange{}, miss, mlir::ValueRange{});

      builder.setInsertionPointToStart(bodyBlocks[index]);
      values = saved;
      {
        auto typeScope = types.pushScope();
        emitStatements(ast::nodeList(handler, "body"));
      }
      if (!blockHasTerminator(*bodyBlocks[index]))
        builder.create<py::ExceptYieldOp>(loc(handler), mlir::ValueRange{});
    }

    builder.setInsertionPointToStart(rethrowBlock);
    builder.create<py::RaiseCurrentOp>(loc(statement));
  }

  values = saved;
  builder.setInsertionPointAfter(tryOp);
}

ModuleEmitter::Value ModuleEmitter::emitExpr(const parser::Node *expr) {
  if (!expr)
    return {builder.create<py::NoneOp>(builder.getUnknownLoc(), types.none())
                .getResult(),
            types.none()};
  if (expr->kind == "Constant")
    return emitConstant(*expr);
  if (expr->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*expr);
    auto found = values.find(name);
    if (found != values.end())
      return found->second;
    mlir::Type type = types.lookupSymbol(name).value_or(types.object());
    if (auto cls = types.lookupClass(name)) {
      mlir::Type typeType = types.typeObject(*cls);
      auto op = builder.create<py::TypeObjectOp>(loc(*expr), typeType, *cls);
      return {op.getResult(), typeType};
    }
    std::string binding = std::string(name);
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      binding = *canonical;
    return emitBindingRef(*expr, binding, type);
  }
  if (expr->kind == "Call")
    return emitCall(*expr);
  if (expr->kind == "UnaryOp")
    return emitUnary(*expr);
  if (expr->kind == "BinOp")
    return emitBinary(*expr);
  if (expr->kind == "Compare")
    return emitCompare(*expr);
  if (expr->kind == "Subscript")
    return emitSubscript(*expr);
  if (expr->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(expr);
    if (!qualified.empty())
      if (auto cls = types.lookupClass(qualified)) {
        mlir::Type typeType = types.typeObject(*cls);
        auto op = builder.create<py::TypeObjectOp>(loc(*expr), typeType, *cls);
        return {op.getResult(), typeType};
      }
    if (!qualified.empty())
      if (auto symbol = types.lookupSymbol(qualified)) {
        std::string binding = qualified;
        if (std::optional<std::string> canonical =
                types.lookupCanonicalBinding(qualified))
          binding = *canonical;
        return emitBindingRef(*expr, binding, *symbol);
      }
    return emitAttribute(*expr);
  }
  if (expr->kind == "Await")
    return emitAwait(*expr);
  if (expr->kind == "List" || expr->kind == "Tuple" || expr->kind == "Dict")
    return emitContainerLiteral(*expr);
  if (expr->kind == "IfExp") {
    const parser::Node *bodyNode = ast::node(*expr, "body");
    const parser::Node *elseNode = ast::node(*expr, "orelse");
    mlir::Type resultType =
        types.join({types.inferExpr(bodyNode), types.inferExpr(elseNode)});
    mlir::Value condition =
        emitBoolValue(emitExpr(ast::node(*expr, "test")), *expr);
    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc(*expr), mlir::TypeRange{resultType}, condition, true);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value body = coerceValue(emitExpr(bodyNode), resultType, *expr);
      builder.create<mlir::scf::YieldOp>(loc(*expr), body.value);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value other = coerceValue(emitExpr(elseNode), resultType, *expr);
      builder.create<mlir::scf::YieldOp>(loc(*expr), other.value);
    }

    builder.setInsertionPointAfter(ifOp);
    return {ifOp.getResult(0), resultType};
  }
  if (expr->kind == "Lambda")
    return emitLambda(*expr);
  diagnostics.push_back(
      parser::Diagnostic{parser::Severity::Error, expr->range.start,
                         "unsupported expression kind '" + expr->kind + "'"});
  return emitNone(*expr);
}

ModuleEmitter::Value ModuleEmitter::emitConstant(const parser::Node &expr) {
  if (ast::isNoneField(expr, "value")) {
    auto op = builder.create<py::NoneOp>(loc(expr), types.none());
    return {op.getResult(), types.none()};
  }
  if (auto value = ast::boolean(expr, "value")) {
    mlir::Type type = types.literal(*value ? "True" : "False");
    auto op = builder.create<py::BoolConstantOp>(loc(expr), type,
                                                 builder.getBoolAttr(*value));
    return {op.getResult(), type};
  }
  if (auto value = ast::integer(expr, "value")) {
    std::string text = std::to_string(*value);
    mlir::Type type = types.literal(text);
    auto op = builder.create<py::IntConstantOp>(loc(expr), type,
                                                builder.getStringAttr(text));
    return {op.getResult(), type};
  }
  if (auto value = ast::floating(expr, "value")) {
    auto op = builder.create<py::FloatConstantOp>(
        loc(expr), types.floatType(), builder.getF64FloatAttr(*value));
    return {op.getResult(), types.floatType()};
  }
  if (auto value = ast::string(expr, "value")) {
    mlir::Type type = types.literal("\"" + std::string(*value) + "\"");
    auto op = builder.create<py::StrConstantOp>(loc(expr), type,
                                                builder.getStringAttr(*value));
    return {op.getResult(), type};
  }
  if (const auto *fieldValue = ast::field(expr, "value")) {
    if (const auto *big = std::get_if<parser::BigInteger>(fieldValue)) {
      mlir::Type type = types.literal(big->decimal);
      auto op = builder.create<py::IntConstantOp>(
          loc(expr), type, builder.getStringAttr(big->decimal));
      return {op.getResult(), type};
    }
  }
  diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                           expr.range.start,
                                           "unsupported constant literal"});
  return emitNone(expr);
}

ModuleEmitter::Value ModuleEmitter::emitCall(const parser::Node &expr) {
  const parser::Node *calleeNode = ast::node(expr, "func");
  std::string calleeQualified = ast::qualifiedName(calleeNode);
  if (!calleeQualified.empty())
    if (auto cls = types.lookupClass(calleeQualified))
      return emitClassInstantiation(expr, llvm::StringRef(calleeQualified),
                                    *cls);

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    if (auto cls = types.lookupClass(name))
      return emitClassInstantiation(expr, name, *cls);
    if (name == "len") {
      const auto *args = ast::nodeList(expr, "args");
      if (args && args->size() == 1) {
        Value input = emitExpr(args->front().get());
        CallInferenceResult inference =
            types.inferMethodCallWithEvidence(input.type, "__len__", {});
        mlir::Type resultType =
            inference ? inference.resultType : types.intType();
        auto op = builder.create<py::LenOp>(
            loc(expr), resultType,
            mlir::FlatSymbolRefAttr::get(&context, "__len__"),
            callProtocolFor(inference), input.value);
        return {op.getResult(), resultType};
      }
    }
    if (name == "round") {
      const auto *args = ast::nodeList(expr, "args");
      if (args && (args->size() == 1 || args->size() == 2)) {
        llvm::SmallVector<mlir::Value, 2> inputs;
        llvm::SmallVector<mlir::Type, 1> extraTypes;
        Value receiver = emitExpr(args->front().get());
        inputs.push_back(receiver.value);
        if (args->size() == 2) {
          Value ndigits = emitExpr((*args)[1].get());
          inputs.push_back(ndigits.value);
          extraTypes.push_back(ndigits.type);
        }
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            receiver.type, "__round__", extraTypes);
        mlir::Type resultType =
            inference ? inference.resultType : types.inferExpr(&expr);
        auto op =
            builder.create<py::RoundOp>(loc(expr), resultType, "__round__",
                                        callProtocolFor(inference), inputs);
        return {op.getResult(), resultType};
      }
    }
  }

  auto emitDirectCallableCall =
      [&](Value callee, mlir::Type resultOverride = {},
          llvm::ArrayRef<Value> leadingPositional = {},
          bool includeAstArguments = true) -> Value {
    llvm::SmallVector<Value, 8> positional;
    llvm::SmallVector<char, 8> positionalUnpacked;
    llvm::SmallVector<mlir::Type, 8> positionalTypes;
    for (Value value : leadingPositional) {
      positional.push_back(value);
      positionalUnpacked.push_back(0);
      positionalTypes.push_back(value.type);
    }
    if (includeAstArguments) {
      if (const auto *args = ast::nodeList(expr, "args"))
        for (const parser::NodePtr &arg : *args) {
          bool unpacked = arg && arg->kind == "Starred";
          const parser::Node *valueNode =
              unpacked ? ast::node(*arg, "value") : arg.get();
          Value value = emitExpr(valueNode);
          positional.push_back(value);
          positionalUnpacked.push_back(unpacked ? 1 : 0);
          if (unpacked)
            appendStarredArgumentTypes(value.type, types, positionalTypes);
          else
            positionalTypes.push_back(value.type);
        }
    }

    llvm::SmallVector<Value, 4> kwNames;
    llvm::SmallVector<Value, 4> kwValues;
    llvm::SmallVector<CallKeywordType, 4> keywordTypes;
    if (includeAstArguments) {
      if (const auto *keywords = ast::nodeList(expr, "keywords")) {
        for (const parser::NodePtr &keyword : *keywords) {
          if (auto name = ast::string(*keyword, "arg")) {
            mlir::Type literal =
                types.literal("\"" + std::string(*name) + "\"");
            auto stringOp = builder.create<py::StrConstantOp>(
                loc(*keyword), literal, builder.getStringAttr(*name));
            kwNames.push_back({stringOp.getResult(), literal});
            Value keywordValue = emitExpr(ast::node(*keyword, "value"));
            kwValues.push_back(keywordValue);
            keywordTypes.push_back(
                CallKeywordType{std::string(*name), keywordValue.type});
            continue;
          }
          kwValues.push_back(emitExpr(ast::node(*keyword, "value")));
        }
      }
    }

    Value posPack = emitPack(positional, positionalUnpacked);
    Value namePack = emitPack(kwNames);
    Value valuePack = emitPack(kwValues);
    CallInferenceResult inference =
        types.inferCallWithEvidence(callee.type, positionalTypes, keywordTypes);
    mlir::Type resultType =
        resultOverride ? resultOverride
                       : (inference ? inference.resultType : types.object());
    auto op = builder.create<py::CallOp>(
        loc(expr), mlir::TypeRange{resultType},
        callProtocolFor(inference, callee.type), callee.value, posPack.value,
        namePack.value, valuePack.value);
    return {op.getResults().front(), resultType};
  };

  auto emitBuiltinBinding = [&](llvm::StringRef name) -> Value {
    mlir::Type type = types.lookupSymbol(name).value_or(types.object());
    std::string binding = name.str();
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      binding = *canonical;
    return emitBindingRef(*calleeNode, binding, type);
  };

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    const auto *args = ast::nodeList(expr, "args");
    const auto *keywords = ast::nodeList(expr, "keywords");
    bool builtinVisible = values.find(name) == values.end();
    bool hasKeywords = keywords && !keywords->empty();
    if (builtinVisible && args && args->size() == 1 && !hasKeywords &&
        (name == "repr" || name == "print")) {
      mlir::Type argumentType = types.inferExpr(args->front().get());
      if (std::optional<MethodBinding> method =
              lookupClassMethod(argumentType, "__repr__")) {
        Value argument = emitExpr(args->front().get());
        llvm::StringMap<Value> emptyKeywords;
        Value repr =
            emitInlineMethodBody(expr, argument, *method, {}, emptyKeywords);
        if (name == "repr")
          return repr;
        return emitDirectCallableCall(emitBuiltinBinding(name), types.none(),
                                      {repr}, /*includeAstArguments=*/false);
      }
    }
  }

  if (calleeNode && calleeNode->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*calleeNode);
    if (values.find(name) == values.end())
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(name))
        if (*canonical == "asyncio.sleep")
          if (auto symbol = types.lookupSymbol(name))
            return emitDirectCallableCall(
                emitBindingRef(*calleeNode, *canonical, *symbol),
                types.inferExpr(&expr));
  }

  if (calleeNode && calleeNode->kind == "Attribute" &&
      !calleeQualified.empty()) {
    if (auto symbol = types.lookupSymbol(calleeQualified)) {
      std::string binding = calleeQualified;
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(calleeQualified))
        binding = *canonical;
      mlir::Type resultOverride =
          binding == "asyncio.sleep" ? types.inferExpr(&expr) : mlir::Type();
      return emitDirectCallableCall(
          emitBindingRef(*calleeNode, binding, *symbol), resultOverride);
    }
  }

  if (calleeNode && calleeNode->kind == "Attribute") {
    if (const parser::Node *receiverNode = ast::node(*calleeNode, "value")) {
      if (auto methodName = ast::string(*calleeNode, "attr")) {
        Value receiver = emitExpr(receiverNode);
        if (std::optional<MethodBinding> method =
                lookupClassMethod(receiver.type, *methodName)) {
          return emitInlineMethodCall(expr, receiver, *method);
        }

        llvm::SmallVector<Value, 8> positional;
        llvm::SmallVector<char, 8> positionalUnpacked;
        llvm::SmallVector<mlir::Type, 8> positionalTypes;
        if (const auto *args = ast::nodeList(expr, "args"))
          for (const parser::NodePtr &arg : *args) {
            bool unpacked = arg && arg->kind == "Starred";
            const parser::Node *valueNode =
                unpacked ? ast::node(*arg, "value") : arg.get();
            Value value = emitExpr(valueNode);
            positional.push_back(value);
            positionalUnpacked.push_back(unpacked ? 1 : 0);
            if (unpacked)
              appendStarredArgumentTypes(value.type, types, positionalTypes);
            else
              positionalTypes.push_back(value.type);
          }

        llvm::SmallVector<Value, 4> kwNames;
        llvm::SmallVector<Value, 4> kwValues;
        llvm::SmallVector<CallKeywordType, 4> keywordTypes;
        if (const auto *keywords = ast::nodeList(expr, "keywords")) {
          for (const parser::NodePtr &keyword : *keywords) {
            if (auto name = ast::string(*keyword, "arg")) {
              mlir::Type literal =
                  types.literal("\"" + std::string(*name) + "\"");
              auto stringOp = builder.create<py::StrConstantOp>(
                  loc(*keyword), literal, builder.getStringAttr(*name));
              kwNames.push_back({stringOp.getResult(), literal});
              Value keywordValue = emitExpr(ast::node(*keyword, "value"));
              kwValues.push_back(keywordValue);
              keywordTypes.push_back(
                  CallKeywordType{std::string(*name), keywordValue.type});
              continue;
            }
            kwValues.push_back(emitExpr(ast::node(*keyword, "value")));
          }
        }

        Value posPack = emitPack(positional, positionalUnpacked);
        Value namePack = emitPack(kwNames);
        Value valuePack = emitPack(kwValues);
        CallInferenceResult inference = types.inferMethodCallWithEvidence(
            receiver.type, *methodName, positionalTypes, keywordTypes);
        mlir::Type resultType =
            inference ? inference.resultType : types.inferExpr(&expr);
        auto op = builder.create<py::CallOp>(
            loc(expr), mlir::TypeRange{resultType}, callProtocolFor(inference),
            receiver.value, posPack.value, namePack.value, valuePack.value);
        op->setAttr("ly.bound_method", builder.getStringAttr(*methodName));
        return {op.getResults().front(), resultType};
      }
    }
  }

  return emitDirectCallableCall(emitExpr(calleeNode));
}

ModuleEmitter::Value
ModuleEmitter::emitInlineMethodCall(const parser::Node &expr, Value receiver,
                                    const MethodBinding &method) {
  if (!method.method)
    return emitNone(expr);

  llvm::SmallVector<Value, 8> positional;
  if (const auto *args = ast::nodeList(expr, "args")) {
    for (const parser::NodePtr &arg : *args) {
      if (arg && arg->kind == "Starred") {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, arg->range.start,
            "starred arguments are not supported for inlined class methods"});
        continue;
      }
      positional.push_back(emitExpr(arg.get()));
    }
  }

  llvm::StringMap<Value> keywords;
  if (const auto *keywordNodes = ast::nodeList(expr, "keywords")) {
    for (const parser::NodePtr &keyword : *keywordNodes) {
      if (auto name = ast::string(*keyword, "arg")) {
        keywords[*name] = emitExpr(ast::node(*keyword, "value"));
        continue;
      }
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, keyword->range.start,
          "variadic keyword arguments are not supported for inlined class "
          "methods"});
    }
  }

  return emitInlineMethodBody(expr, receiver, method, positional, keywords);
}

ModuleEmitter::Value ModuleEmitter::emitInlineMethodBody(
    const parser::Node &anchor, Value receiver, const MethodBinding &method,
    llvm::ArrayRef<Value> positional, const llvm::StringMap<Value> &keywords) {
  if (!method.method)
    return emitNone(anchor);
  const FunctionSignature &sig = method.signature;
  const auto *body = ast::nodeList(*method.method, "body");
  mlir::Type resultType = sig.resultType ? sig.resultType : types.none();

  llvm::StringMap<Value> savedValues = values;
  auto scope = types.pushScope();
  llvm::StringSet<> bound;
  auto bind = [&](llvm::StringRef name, Value value) {
    values[name] = value;
    types.bindSymbol(name, value.type);
    bound.insert(name);
  };

  unsigned parameterIndex = 0;
  if (!sig.positionalNames.empty()) {
    bind(sig.positionalNames.front(), receiver);
    parameterIndex = 1;
  }

  for (Value argument : positional) {
    if (parameterIndex >= sig.positionalNames.size()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, anchor.range.start,
          "too many positional arguments for inlined class method"});
      break;
    }
    bind(sig.positionalNames[parameterIndex++], argument);
  }

  auto bindKeyword = [&](llvm::StringRef name, Value value) {
    for (llvm::StringRef positionalName : sig.positionalNames) {
      if (positionalName != name)
        continue;
      if (bound.contains(name)) {
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, anchor.range.start,
            "multiple values for inlined class method argument '" + name.str() +
                "'"});
        return;
      }
      bind(name, value);
      return;
    }
    for (llvm::StringRef kwOnlyName : sig.kwOnlyNames) {
      if (kwOnlyName != name)
        continue;
      bind(name, value);
      return;
    }
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unexpected keyword argument '" + name.str() +
                               "' for inlined class method"});
  };
  for (auto &entry : keywords)
    bindKeyword(entry.getKey(), entry.getValue());

  mlir::Block *entryBlock = builder.getInsertionBlock();
  mlir::Region *region = entryBlock ? entryBlock->getParent() : nullptr;
  if (!region) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "inlined class method call requires an active insertion region"});
    values = std::move(savedValues);
    return emitNone(anchor);
  }
  mlir::Block *continuation =
      entryBlock->splitBlock(builder.getInsertionPoint());
  continuation->addArgument(resultType, loc(anchor));
  mlir::Block *bodyBlock =
      builder.createBlock(region, continuation->getIterator());

  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::cf::BranchOp>(loc(anchor), bodyBlock);
  builder.setInsertionPointToStart(bodyBlock);
  inlineReturnContexts.push_back(InlineReturnContext{continuation, resultType});
  emitStatements(body);
  inlineReturnContexts.pop_back();
  if (!insertionBlockTerminated(builder)) {
    if (resultType != types.none()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, method.method->range.start,
          "inlined class method can fall through without returning a value"});
    }
    Value none = emitNone(anchor);
    Value result = coerceValue(none, resultType, anchor);
    builder.create<mlir::cf::BranchOp>(loc(anchor), continuation, result.value);
  }
  builder.setInsertionPointToStart(continuation);
  values = std::move(savedValues);
  return {continuation->getArgument(0), resultType};
}

ModuleEmitter::Value ModuleEmitter::emitClassInstantiation(
    const parser::Node &expr, llvm::StringRef name, mlir::Type instanceType) {
  llvm::SmallVector<Value, 8> positional;
  if (const auto *args = ast::nodeList(expr, "args"))
    for (const parser::NodePtr &arg : *args)
      positional.push_back(emitExpr(arg.get()));
  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  positionalTypes.reserve(positional.size());
  for (const Value &arg : positional)
    positionalTypes.push_back(arg.type);

  mlir::Type inferredInstanceType =
      types.inferClassInstantiation(instanceType, positionalTypes, {});
  mlir::Type classType = types.typeObject(inferredInstanceType);
  auto classObject = builder.create<py::TypeObjectOp>(loc(expr), classType,
                                                      inferredInstanceType);
  Value posPack = emitPack(positional);
  Value emptyNames = emitPack({});
  Value emptyValues = emitPack({});

  auto newOp = builder.create<py::NewOp>(
      loc(expr), inferredInstanceType,
      mlir::FlatSymbolRefAttr::get(&context, "__new__"), callableProtocol(),
      classObject.getResult(), posPack.value, emptyNames.value,
      emptyValues.value);
  builder.create<py::InitOp>(
      loc(expr), types.none(),
      mlir::FlatSymbolRefAttr::get(&context, "__init__"), callableProtocol(),
      newOp.getInstance(), posPack.value, emptyNames.value, emptyValues.value);
  (void)name;
  return {newOp.getInstance(), inferredInstanceType};
}

ModuleEmitter::Value ModuleEmitter::emitUnary(const parser::Node &expr) {
  const parser::Node *op = ast::node(expr, "op");
  const parser::Node *operandNode = ast::node(expr, "operand");

  if (ast::isOperator(op, "USub") && operandNode &&
      operandNode->kind == "Constant") {
    if (auto value = ast::integer(*operandNode, "value")) {
      std::string text = "-" + std::to_string(*value);
      mlir::Type type = types.literal(text);
      auto constOp = builder.create<py::IntConstantOp>(
          loc(expr), type, builder.getStringAttr(text));
      return {constOp.getResult(), type};
    }
    if (auto value = ast::floating(*operandNode, "value")) {
      auto constOp = builder.create<py::FloatConstantOp>(
          loc(expr), types.floatType(), builder.getF64FloatAttr(-*value));
      return {constOp.getResult(), types.floatType()};
    }
    if (const auto *fieldValue = ast::field(*operandNode, "value")) {
      if (const auto *big = std::get_if<parser::BigInteger>(fieldValue)) {
        std::string text = "-" + big->decimal;
        mlir::Type type = types.literal(text);
        auto constOp = builder.create<py::IntConstantOp>(
            loc(expr), type, builder.getStringAttr(text));
        return {constOp.getResult(), type};
      }
    }
  }

  Value operand = emitExpr(operandNode);
  mlir::Type result = types.widenLiteral(types.inferExpr(&expr));
  if (ast::isOperator(op, "USub"))
    return emitUnarySpecial<py::NegOp>(expr, "__neg__", operand, result);
  if (ast::isOperator(op, "UAdd"))
    return emitUnarySpecial<py::PosOp>(expr, "__pos__", operand, result);
  if (ast::isOperator(op, "Invert"))
    return emitUnarySpecial<py::InvertOp>(expr, "__invert__", operand, result);
  if (ast::isOperator(op, "Not")) {
    mlir::Value truth = emitBoolValue(operand, expr);
    auto one = builder.create<mlir::arith::ConstantIntOp>(loc(expr), 1, 1);
    auto inverted = builder.create<mlir::arith::XOrIOp>(loc(expr), truth, one);
    auto pyBool = builder.create<py::CastFromPrimOp>(
        loc(expr), types.boolType(), inverted.getResult());
    return {pyBool.getResult(), types.boolType()};
  }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, expr.range.start, "unsupported unary operator"});
  return emitNone(expr);
}

ModuleEmitter::Value ModuleEmitter::emitBinary(const parser::Node &expr) {
  Value lhs = emitExpr(ast::node(expr, "left"));
  Value rhs = emitExpr(ast::node(expr, "right"));
  const parser::Node *op = ast::node(expr, "op");
  mlir::Type left = types.widenLiteral(lhs.type);
  mlir::Type right = types.widenLiteral(rhs.type);
  mlir::Type result = types.join({left, right});
  if (left == types.strType() && right == types.strType()) {
    result = types.strType();
  } else if (ast::isOperator(op, "Div") &&
             (left == types.intType() || left == types.floatType()) &&
             (right == types.intType() || right == types.floatType())) {
    result = types.floatType();
  } else if (left == types.floatType() || right == types.floatType()) {
    result = types.floatType();
  } else if (left == types.intType() && right == types.intType()) {
    result = types.intType();
  }
  if (ast::isOperator(op, "Sub"))
    return emitBinarySpecial<py::SubOp>(expr, "__sub__", lhs, rhs, result);
  if (ast::isOperator(op, "Mult"))
    return emitBinarySpecial<py::MulOp>(expr, "__mul__", lhs, rhs, result);
  if (ast::isOperator(op, "Div"))
    return emitBinarySpecial<py::DivOp>(expr, "__truediv__", lhs, rhs, result);
  if (ast::isOperator(op, "FloorDiv"))
    return emitBinarySpecial<py::FloorDivOp>(expr, "__floordiv__", lhs, rhs,
                                             result);
  if (ast::isOperator(op, "Mod"))
    return emitBinarySpecial<py::ModOp>(expr, "__mod__", lhs, rhs, result);
  if (ast::isOperator(op, "LShift"))
    return emitBinarySpecial<py::LShiftOp>(expr, "__lshift__", lhs, rhs,
                                           result);
  if (ast::isOperator(op, "RShift"))
    return emitBinarySpecial<py::RShiftOp>(expr, "__rshift__", lhs, rhs,
                                           result);
  if (ast::isOperator(op, "BitAnd"))
    return emitBinarySpecial<py::BitAndOp>(expr, "__and__", lhs, rhs, result);
  if (ast::isOperator(op, "BitOr"))
    return emitBinarySpecial<py::BitOrOp>(expr, "__or__", lhs, rhs, result);
  if (ast::isOperator(op, "BitXor"))
    return emitBinarySpecial<py::BitXorOp>(expr, "__xor__", lhs, rhs, result);
  return emitBinarySpecial<py::AddOp>(expr, "__add__", lhs, rhs, result);
}

ModuleEmitter::Value ModuleEmitter::emitCompare(const parser::Node &expr) {
  Value lhs = emitExpr(ast::node(expr, "left"));
  const auto *comparators = ast::nodeList(expr, "comparators");
  const auto *ops = ast::nodeList(expr, "ops");
  if (!comparators || comparators->empty()) {
    auto op = builder.create<py::BoolConstantOp>(
        loc(expr), types.literal("False"), builder.getBoolAttr(false));
    return {op.getResult(), types.literal("False")};
  }
  Value rhs = emitExpr(comparators->front().get());
  const parser::Node *op = ops && !ops->empty() ? ops->front().get() : nullptr;
  auto emitNoneIdentityTest = [&](Value candidate,
                                  Value other) -> std::optional<Value> {
    auto unionType = mlir::dyn_cast_if_present<py::UnionType>(candidate.type);
    if (!unionType || !unionType.hasMember(types.none()) ||
        !isNoneTypeLike(other.type))
      return std::nullopt;

    auto test = builder.create<py::UnionTestOp>(
        loc(expr), builder.getI1Type(), candidate.value,
        mlir::TypeAttr::get(types.none()));
    mlir::Value bit = test.getResult();
    if (ast::isOperator(op, "IsNot")) {
      auto one = builder.create<mlir::arith::ConstantIntOp>(loc(expr), 1, 1);
      bit = builder.create<mlir::arith::XOrIOp>(loc(expr), bit, one);
    }
    auto pyBool =
        builder.create<py::CastFromPrimOp>(loc(expr), types.boolType(), bit);
    return Value{pyBool.getResult(), types.boolType()};
  };
  if (ast::isOperator(op, "Is") || ast::isOperator(op, "IsNot")) {
    if (auto narrowed = emitNoneIdentityTest(lhs, rhs))
      return *narrowed;
    if (auto narrowed = emitNoneIdentityTest(rhs, lhs))
      return *narrowed;
  }
  if (ast::isOperator(op, "NotEq") || ast::isOperator(op, "IsNot"))
    return emitBinarySpecial<py::NeOp>(expr, "__ne__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "Lt"))
    return emitBinarySpecial<py::LtOp>(expr, "__lt__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "LtE"))
    return emitBinarySpecial<py::LeOp>(expr, "__le__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "Gt"))
    return emitBinarySpecial<py::GtOp>(expr, "__gt__", lhs, rhs,
                                       types.boolType());
  if (ast::isOperator(op, "GtE"))
    return emitBinarySpecial<py::GeOp>(expr, "__ge__", lhs, rhs,
                                       types.boolType());
  return emitBinarySpecial<py::EqOp>(expr, "__eq__", lhs, rhs,
                                     types.boolType());
}

ModuleEmitter::Value ModuleEmitter::emitSubscript(const parser::Node &expr) {
  Value container = emitExpr(ast::node(expr, "value"));
  Value index = emitExpr(ast::node(expr, "slice"));
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      container.type, "__getitem__", {index.type});
  mlir::Type result = inference ? inference.resultType : types.inferExpr(&expr);
  auto op = builder.create<py::GetItemOp>(
      loc(expr), result, mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
      callProtocolFor(inference), container.value, index.value);
  return {op.getResult(), result};
}

ModuleEmitter::Value ModuleEmitter::emitAttribute(const parser::Node &expr) {
  Value object = emitExpr(ast::node(expr, "value"));
  mlir::Type result = types.inferExpr(&expr);
  if (auto attr = ast::string(expr, "attr"))
    if (std::optional<mlir::Type> field = lookupClassField(object.type, *attr))
      result = *field;
  auto op = builder.create<py::AttrGetOp>(loc(expr), result, object.value,
                                          *ast::string(expr, "attr"));
  return {op.getResult(), result};
}

ModuleEmitter::Value ModuleEmitter::emitAwait(const parser::Node &expr) {
  Value awaitable = emitExpr(ast::node(expr, "value"));
  return emitAwaitValue(expr, awaitable);
}

ModuleEmitter::Value ModuleEmitter::emitAwaitValue(const parser::Node &anchor,
                                                   Value awaitable) {
  const py::protocols::Table &table = py::protocols::Table::get(context);
  std::optional<py::protocols::AwaitableResolution> resolution =
      table.resolveAwaitableWithEvidence(types.widenLiteral(awaitable.type));
  mlir::Type result = resolution ? resolution->payloadType : types.object();
  mlir::Type awaitContract = callableProtocol();
  if (resolution && resolution->awaitContract)
    awaitContract =
        callProtocolFor(resolution->awaitContract->method.signature);
  if (!resolution) {
    std::string typeText;
    llvm::raw_string_ostream typeStream(typeText);
    typeStream << awaitable.type;
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "await expression requires an Awaitable value, got " +
            typeStream.str()});
  }

  auto op = builder.create<py::AwaitOp>(loc(anchor), result, awaitContract,
                                        awaitable.value);
  return {op.getResult(), result};
}

ModuleEmitter::Value
ModuleEmitter::emitContainerLiteral(const parser::Node &expr) {
  llvm::SmallVector<Value, 8> valuesToPack;
  if (const auto *elts = ast::nodeList(expr, "elts"))
    for (const parser::NodePtr &elt : *elts)
      valuesToPack.push_back(emitExpr(elt.get()));
  if (const auto *keys = ast::nodeList(expr, "keys")) {
    const auto *vals = ast::nodeList(expr, "values");
    for (auto [index, key] : llvm::enumerate(*keys)) {
      if (key)
        valuesToPack.push_back(emitExpr(key.get()));
      if (vals && index < vals->size())
        valuesToPack.push_back(emitExpr((*vals)[index].get()));
    }
  }
  mlir::Type resultType = types.inferExpr(&expr);
  llvm::SmallVector<mlir::Value, 8> operands;
  for (Value value : valuesToPack)
    operands.push_back(value.value);
  auto op = builder.create<py::PackOp>(loc(expr), resultType, operands);
  return {op.getResult(), resultType};
}

ModuleEmitter::Value
ModuleEmitter::emitBindingRef(const parser::Node &anchor,
                              llvm::StringRef binding, mlir::Type type,
                              llvm::ArrayRef<Value> captures) {
  mlir::Type resultType = type ? type : types.object();
  llvm::SmallVector<mlir::Value, 4> captureValues;
  for (Value capture : captures)
    captureValues.push_back(capture.value);
  auto op = builder.create<py::BindingRefOp>(
      loc(anchor), resultType, builder.getStringAttr(binding), captureValues);
  return {op.getResult(), resultType};
}

ModuleEmitter::Value
ModuleEmitter::emitFunctionObject(const parser::Node &anchor,
                                  llvm::StringRef symbolName, mlir::Type type,
                                  llvm::ArrayRef<Capture> captures) {
  llvm::SmallVector<Value, 4> captureValues;
  for (const Capture &capture : captures)
    captureValues.push_back(capture.value);
  return emitBindingRef(anchor, symbolName, type, captureValues);
}

ModuleEmitter::Value ModuleEmitter::emitNone(const parser::Node &anchor) {
  auto op = builder.create<py::NoneOp>(loc(anchor), types.none());
  return {op.getResult(), types.none()};
}

ModuleEmitter::Value ModuleEmitter::emitPack(mlir::ArrayRef<Value> valuesIn,
                                             llvm::ArrayRef<char> unpacked) {
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<mlir::Type, 8> elementTypes;
  for (Value value : valuesIn) {
    operands.push_back(value.value);
    elementTypes.push_back(value.type);
  }
  mlir::Type element =
      elementTypes.empty() ? types.object() : types.join(elementTypes);
  mlir::Type resultType = types.tupleOf(element);
  auto op =
      builder.create<py::PackOp>(builder.getUnknownLoc(), resultType, operands);
  if (!unpacked.empty() && anyTrue(unpacked)) {
    if (unpacked.size() != valuesIn.size()) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error,
                             {},
                             "internal pack unpack metadata mismatch"});
    } else {
      op->setAttr(kPackUnpackedOperandsAttr, boolArray(builder, unpacked));
    }
  }
  return {op.getResult(), resultType};
}

ModuleEmitter::Value ModuleEmitter::coerceValue(Value value,
                                                mlir::Type targetType,
                                                const parser::Node &anchor) {
  if (!targetType || value.type == targetType)
    return value;
  if (auto unionType = mlir::dyn_cast<py::UnionType>(targetType)) {
    if (unionType.hasMember(value.type)) {
      auto op =
          builder.create<py::UnionWrapOp>(loc(anchor), targetType, value.value);
      return {op.getResult(), targetType};
    }
  }
  if (mlir::isa<py::ProtocolType>(targetType)) {
    auto op = builder.create<py::ProtocolViewOp>(loc(anchor), targetType,
                                                 value.value);
    return {op.getResult(), targetType};
  }
  if (mlir::isa<py::ContractType, py::LiteralType, py::CallableType,
                py::TypeType, py::SelfType, py::TypeVarType, py::ParamSpecType>(
          targetType)) {
    auto op =
        builder.create<py::ClassUpcastOp>(loc(anchor), targetType, value.value);
    return {op.getResult(), targetType};
  }
  return value;
}

mlir::Value ModuleEmitter::emitBoolValue(Value value,
                                         const parser::Node &anchor) {
  if (value.value && value.value.getType().isInteger(1))
    return value.value;
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(value.type, "__bool__", {});
  auto op = builder.create<py::BoolOp>(
      loc(anchor), builder.getI1Type(),
      mlir::FlatSymbolRefAttr::get(&context, "__bool__"),
      callProtocolFor(inference), value.value);
  return op.getResult();
}

template <typename Op>
ModuleEmitter::Value
ModuleEmitter::emitBinarySpecial(const parser::Node &anchor,
                                 llvm::StringRef method, Value lhs, Value rhs,
                                 mlir::Type resultType) {
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(lhs.type, method, {rhs.type});
  if (inference)
    resultType = inference.resultType;
  auto op = builder.create<Op>(
      loc(anchor), resultType, mlir::FlatSymbolRefAttr::get(&context, method),
      method, callProtocolFor(inference), lhs.value, rhs.value);
  return {op.getResult(), resultType};
}

template <typename Op>
ModuleEmitter::Value ModuleEmitter::emitUnarySpecial(const parser::Node &anchor,
                                                     llvm::StringRef method,
                                                     Value input,
                                                     mlir::Type resultType) {
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(input.type, method, {});
  if (inference)
    resultType = inference.resultType;
  auto op = builder.create<Op>(loc(anchor), resultType,
                               mlir::FlatSymbolRefAttr::get(&context, method),
                               method, callProtocolFor(inference), input.value);
  return {op.getResult(), resultType};
}

} // namespace lython::emitter
