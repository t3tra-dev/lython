#include "AstSynth.h"

#include <utility>

namespace lython::emitter::synth {

namespace {

NodePtr operatorNode(llvm::StringRef kind, SourceRange range) {
  return parser::makeNode(std::string(kind), range);
}

} // namespace

NodePtr name(llvm::StringRef id, SourceRange range) {
  NodePtr node = parser::makeNode("Name", range);
  parser::addField(*node, "id", std::string(id));
  return node;
}

NodePtr intConstant(std::int64_t value, SourceRange range) {
  NodePtr node = parser::makeNode("Constant", range);
  parser::addField(*node, "value", value);
  return node;
}

NodePtr strConstant(llvm::StringRef text, SourceRange range) {
  NodePtr node = parser::makeNode("Constant", range);
  parser::addField(*node, "value", std::string(text));
  return node;
}

NodePtr boolConstant(bool value, SourceRange range) {
  NodePtr node = parser::makeNode("Constant", range);
  parser::addField(*node, "value", value);
  return node;
}

NodePtr noneConstant(SourceRange range) {
  NodePtr node = parser::makeNode("Constant", range);
  parser::addField(*node, "value", std::monostate{});
  return node;
}

NodePtr attribute(NodePtr value, llvm::StringRef attr, SourceRange range) {
  NodePtr node = parser::makeNode("Attribute", range);
  parser::addField(*node, "value", std::move(value));
  parser::addField(*node, "attr", std::string(attr));
  return node;
}

NodePtr selfAttribute(llvm::StringRef receiver, llvm::StringRef attr,
                      SourceRange range) {
  return attribute(name(receiver, range), attr, range);
}

NodePtr subscript(NodePtr value, NodePtr slice, SourceRange range) {
  NodePtr node = parser::makeNode("Subscript", range);
  parser::addField(*node, "value", std::move(value));
  parser::addField(*node, "slice", std::move(slice));
  return node;
}

NodePtr tuple(std::vector<NodePtr> elts, SourceRange range) {
  NodePtr node = parser::makeNode("Tuple", range);
  parser::addField(*node, "elts", std::move(elts));
  return node;
}

NodePtr call(NodePtr func, std::vector<NodePtr> args, SourceRange range) {
  NodePtr node = parser::makeNode("Call", range);
  parser::addField(*node, "func", std::move(func));
  parser::addField(*node, "args", std::move(args));
  parser::addField(*node, "keywords", std::vector<NodePtr>{});
  return node;
}

NodePtr methodCall(NodePtr receiver, llvm::StringRef method,
                   std::vector<NodePtr> args, SourceRange range) {
  return call(attribute(std::move(receiver), method, range), std::move(args),
              range);
}

NodePtr lenCall(NodePtr value, SourceRange range) {
  return call(name("len", range), {std::move(value)}, range);
}

NodePtr reprCall(NodePtr value, SourceRange range) {
  return call(name("repr", range), {std::move(value)}, range);
}

NodePtr binOp(NodePtr left, llvm::StringRef opKind, NodePtr right,
              SourceRange range) {
  NodePtr node = parser::makeNode("BinOp", range);
  parser::addField(*node, "left", std::move(left));
  parser::addField(*node, "op", operatorNode(opKind, range));
  parser::addField(*node, "right", std::move(right));
  return node;
}

NodePtr compare(NodePtr left, llvm::StringRef opKind, NodePtr right,
                SourceRange range) {
  NodePtr node = parser::makeNode("Compare", range);
  parser::addField(*node, "left", std::move(left));
  parser::addField(*node, "ops",
                   std::vector<NodePtr>{operatorNode(opKind, range)});
  parser::addField(*node, "comparators", std::vector<NodePtr>{std::move(right)});
  return node;
}

NodePtr notOp(NodePtr operand, SourceRange range) {
  NodePtr node = parser::makeNode("UnaryOp", range);
  parser::addField(*node, "op", operatorNode("Not", range));
  parser::addField(*node, "operand", std::move(operand));
  return node;
}

NodePtr boolOp(llvm::StringRef opKind, std::vector<NodePtr> values,
               SourceRange range) {
  if (values.size() == 1)
    return std::move(values.front());
  NodePtr node = parser::makeNode("BoolOp", range);
  parser::addField(*node, "op", operatorNode(opKind, range));
  parser::addField(*node, "values", std::move(values));
  return node;
}

NodePtr ifExp(NodePtr test, NodePtr body, NodePtr orelse, SourceRange range) {
  NodePtr node = parser::makeNode("IfExp", range);
  parser::addField(*node, "test", std::move(test));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "orelse", std::move(orelse));
  return node;
}

NodePtr compareIn(NodePtr left, NodePtr right, SourceRange range) {
  return compare(std::move(left), "In", std::move(right), range);
}

NodePtr orChain(std::vector<NodePtr> values, SourceRange range) {
  return boolOp("Or", std::move(values), range);
}

NodePtr assign(NodePtr target, NodePtr value, SourceRange range) {
  NodePtr node = parser::makeNode("Assign", range);
  parser::addField(*node, "targets", std::vector<NodePtr>{std::move(target)});
  parser::addField(*node, "value", std::move(value));
  return node;
}

NodePtr annAssign(NodePtr target, NodePtr annotation, NodePtr value,
                  SourceRange range) {
  NodePtr node = parser::makeNode("AnnAssign", range);
  parser::addField(*node, "target", std::move(target));
  parser::addField(*node, "annotation", std::move(annotation));
  parser::addField(*node, "value", std::move(value));
  parser::addField(*node, "simple", std::int64_t{1});
  return node;
}

NodePtr returnStmt(NodePtr value, SourceRange range) {
  NodePtr node = parser::makeNode("Return", range);
  parser::addField(*node, "value", std::move(value));
  return node;
}

NodePtr raiseStmt(NodePtr exc, SourceRange range) {
  NodePtr node = parser::makeNode("Raise", range);
  parser::addField(*node, "exc", std::move(exc));
  return node;
}

NodePtr raiseCall(llvm::StringRef exception, llvm::StringRef message,
                  SourceRange range) {
  return raiseStmt(call(name(exception, range),
                        {strConstant(message, range)}, range),
                   range);
}

NodePtr raiseValueError(llvm::StringRef message, SourceRange range) {
  return raiseCall("ValueError", message, range);
}

NodePtr ifStmt(NodePtr test, std::vector<NodePtr> body,
               std::vector<NodePtr> orelse, SourceRange range) {
  NodePtr node = parser::makeNode("If", range);
  parser::addField(*node, "test", std::move(test));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "orelse", std::move(orelse));
  return node;
}

NodePtr forStmt(NodePtr target, NodePtr iter, std::vector<NodePtr> body,
                std::vector<NodePtr> orelse, SourceRange range) {
  NodePtr node = parser::makeNode("For", range);
  parser::addField(*node, "target", std::move(target));
  parser::addField(*node, "iter", std::move(iter));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "orelse", std::move(orelse));
  return node;
}

NodePtr whileStmt(NodePtr test, std::vector<NodePtr> body,
                  std::vector<NodePtr> orelse, SourceRange range) {
  NodePtr node = parser::makeNode("While", range);
  parser::addField(*node, "test", std::move(test));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "orelse", std::move(orelse));
  return node;
}

NodePtr breakStmt(SourceRange range) {
  return parser::makeNode("Break", range);
}

NodePtr continueStmt(SourceRange range) {
  return parser::makeNode("Continue", range);
}

NodePtr yieldStmt(NodePtr value, SourceRange range) {
  NodePtr yield = parser::makeNode("Yield", range);
  parser::addField(*yield, "value", std::move(value));
  NodePtr statement = parser::makeNode("Expr", range);
  parser::addField(*statement, "value", std::move(yield));
  return statement;
}

NodePtr arg(llvm::StringRef name, NodePtr annotation, SourceRange range) {
  NodePtr node = parser::makeNode("arg", range);
  parser::addField(*node, "arg", std::string(name));
  if (annotation)
    parser::addField(*node, "annotation", std::move(annotation));
  return node;
}

NodePtr functionDef(llvm::StringRef name, llvm::ArrayRef<Param> params,
                    std::vector<NodePtr> defaults, std::vector<NodePtr> body,
                    NodePtr returns, llvm::ArrayRef<llvm::StringRef> decorators,
                    SourceRange range,
                    llvm::SmallVectorImpl<const parser::Node *> *paramNodesOut) {
  NodePtr arguments = parser::makeNode("arguments", range);
  std::vector<NodePtr> args;
  args.reserve(params.size());
  for (const Param &param : params) {
    NodePtr node = arg(param.name, param.annotation, range);
    if (paramNodesOut)
      paramNodesOut->push_back(node.get());
    args.push_back(std::move(node));
  }
  parser::addField(*arguments, "posonlyargs", std::vector<NodePtr>{});
  parser::addField(*arguments, "args", std::move(args));
  parser::addField(*arguments, "kwonlyargs", std::vector<NodePtr>{});
  parser::addField(*arguments, "kw_defaults", std::vector<NodePtr>{});
  parser::addField(*arguments, "defaults", std::move(defaults));

  std::vector<NodePtr> decoratorNodes;
  decoratorNodes.reserve(decorators.size());
  for (llvm::StringRef decorator : decorators)
    decoratorNodes.push_back(synth::name(decorator, range));

  NodePtr node = parser::makeNode("FunctionDef", range);
  parser::addField(*node, "name", std::string(name));
  parser::addField(*node, "args", std::move(arguments));
  parser::addField(*node, "body", std::move(body));
  parser::addField(*node, "decorator_list", std::move(decoratorNodes));
  if (returns)
    parser::addField(*node, "returns", std::move(returns));
  return node;
}

} // namespace lython::emitter::synth
