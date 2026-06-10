#include "BuilderImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <set>

namespace lython::emitter {

namespace {

void collectAssignedTargetNames(const parser::Node &target,
                                std::set<std::string> &out) {
  if (target.kind == "Name") {
    if (const std::string *name = stringField(target, "id"))
      out.insert(*name);
    return;
  }
  if (target.kind != "Tuple" && target.kind != "List")
    return;
  const std::vector<parser::NodePtr> *elements = nodeListField(target, "elts");
  if (!elements)
    return;
  for (const parser::NodePtr &element : *elements)
    if (element)
      collectAssignedTargetNames(*element, out);
}

} // namespace

void collectAssignedNames(const parser::Node &stmt,
                          std::set<std::string> &out) {
  if (stmt.kind == "Assign") {
    const std::vector<parser::NodePtr> *targets =
        nodeListField(stmt, "targets");
    if (!targets)
      return;
    for (const parser::NodePtr &target : *targets)
      if (target)
        collectAssignedTargetNames(*target, out);
    return;
  }
  if (stmt.kind == "AnnAssign" || stmt.kind == "AugAssign") {
    const parser::NodePtr *target = nodeField(stmt, "target");
    if (target && *target)
      collectAssignedTargetNames(**target, out);
    return;
  }
  if (stmt.kind == "For") {
    const parser::NodePtr *target = nodeField(stmt, "target");
    if (target && *target)
      collectAssignedTargetNames(**target, out);
    if (const std::vector<parser::NodePtr> *body =
            nodeListField(stmt, "body")) {
      for (const parser::NodePtr &child : *body)
        if (child)
          collectAssignedNames(*child, out);
    }
    if (const std::vector<parser::NodePtr> *orelse =
            nodeListField(stmt, "orelse")) {
      for (const parser::NodePtr &child : *orelse)
        if (child)
          collectAssignedNames(*child, out);
    }
    return;
  }
  if (stmt.kind == "If" || stmt.kind == "While") {
    if (const std::vector<parser::NodePtr> *body =
            nodeListField(stmt, "body")) {
      for (const parser::NodePtr &child : *body)
        if (child)
          collectAssignedNames(*child, out);
    }
    if (const std::vector<parser::NodePtr> *orelse =
            nodeListField(stmt, "orelse")) {
      for (const parser::NodePtr &child : *orelse)
        if (child)
          collectAssignedNames(*child, out);
    }
    return;
  }
  if (stmt.kind == "Try") {
    for (llvm::StringRef fieldName : {"body", "orelse", "finalbody"}) {
      if (const std::vector<parser::NodePtr> *items =
              nodeListField(stmt, fieldName)) {
        for (const parser::NodePtr &child : *items)
          if (child)
            collectAssignedNames(*child, out);
      }
    }
    if (const std::vector<parser::NodePtr> *handlers =
            nodeListField(stmt, "handlers")) {
      for (const parser::NodePtr &handler : *handlers) {
        if (!handler)
          continue;
        if (const std::vector<parser::NodePtr> *items =
                nodeListField(*handler, "body")) {
          for (const parser::NodePtr &child : *items)
            if (child)
              collectAssignedNames(*child, out);
        }
      }
    }
    return;
  }
  if (stmt.kind == "Match") {
    if (const std::vector<parser::NodePtr> *cases =
            nodeListField(stmt, "cases")) {
      for (const parser::NodePtr &matchCase : *cases) {
        if (!matchCase)
          continue;
        if (const std::vector<parser::NodePtr> *items =
                nodeListField(*matchCase, "body")) {
          for (const parser::NodePtr &child : *items)
            if (child)
              collectAssignedNames(*child, out);
        }
      }
    }
  }
}

namespace {

llvm::SmallVector<mlir::Value>
currentCarriedValues(const std::map<std::string, Value> &symbols,
                     llvm::ArrayRef<std::string> carriedNames) {
  llvm::SmallVector<mlir::Value> values;
  values.reserve(carriedNames.size());
  for (const std::string &name : carriedNames)
    values.push_back(symbols.at(name).value);
  return values;
}

std::optional<std::int64_t>
staticIntegerValue(const parser::Node &node,
                   const std::map<std::string, PrimitiveConstant> &constants) {
  if (node.kind == "Constant") {
    const parser::FieldValue *value = valueField(node, "value");
    const auto *integer = value ? std::get_if<std::int64_t>(value) : nullptr;
    if (integer)
      return *integer;
    return std::nullopt;
  }
  if (node.kind == "Name") {
    const std::string *name = stringField(node, "id");
    if (!name)
      return std::nullopt;
    auto found = constants.find(*name);
    if (found == constants.end())
      return std::nullopt;
    if (!mlir::isa<mlir::IntegerType>(found->second.type))
      return std::nullopt;
    return found->second.integerValue;
  }
  if (node.kind == "UnaryOp") {
    std::optional<std::string> op = symbolField(node, "op");
    const parser::NodePtr *operand = nodeField(node, "operand");
    if (!op || !operand || !*operand)
      return std::nullopt;
    std::optional<std::int64_t> value =
        staticIntegerValue(**operand, constants);
    if (!value)
      return std::nullopt;
    if (*op == "+")
      return *value;
    if (*op == "-")
      return -*value;
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

void Builder::Impl::emitFor(const parser::Node &stmt) {
  const parser::NodePtr *target = nodeField(stmt, "target");
  const parser::NodePtr *iter = nodeField(stmt, "iter");
  const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
  const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
  if (!target || !*target || !iter || !*iter || !body || !orelse) {
    error(stmt, "For.target, For.iter, For.body, or For.orelse is missing");
    return;
  }
  if ((*target)->kind != "Name") {
    error(**target, "for target must be a primitive local name for now");
    return;
  }
  const std::string *targetName = stringField(**target, "id");
  if (!targetName) {
    error(**target, "for target name is missing");
    return;
  }

  if ((*iter)->kind != "Call") {
    error(**iter, "for iter currently supports only range(...)");
    return;
  }
  const parser::NodePtr *func = nodeField(**iter, "func");
  const std::vector<parser::NodePtr> *args = nodeListField(**iter, "args");
  const std::vector<parser::NodePtr> *keywords =
      nodeListField(**iter, "keywords");
  if (!func || !*func || (*func)->kind != "Name" || !args) {
    error(**iter, "malformed range(...) call in for statement");
    return;
  }
  const std::string *callee = stringField(**func, "id");
  if (!callee || *callee != "range") {
    error(**iter, "for iter currently supports only range(...)");
    return;
  }
  if (keywords && !keywords->empty()) {
    error(**iter, "range(...) keyword arguments are not supported");
    return;
  }
  if (args->empty() || args->size() > 3) {
    error(**iter, "range(...) expects 1 to 3 arguments");
    return;
  }
  for (const parser::NodePtr &arg : *args) {
    if (!arg) {
      error(**iter, "range(...) argument is missing");
      return;
    }
  }

  auto existingTarget = symbols.find(*targetName);
  mlir::Type defaultRangeType =
      existingTarget != symbols.end() &&
              mlir::isa<mlir::IntegerType>(existingTarget->second.type)
          ? existingTarget->second.type
          : builder.getI64Type();

  auto emitRangeBound = [&](const parser::Node &node,
                            mlir::Type preferredType) -> Value {
    if (std::optional<PrimitiveConstant> constant =
            primitiveIntConstructorConstant(node))
      return emitPrimitiveIntConstructor(node, *constant);

    std::optional<std::int64_t> staticValue =
        staticIntegerValue(node, primitiveConstants);
    if (staticValue) {
      auto intTy = mlir::cast<mlir::IntegerType>(preferredType);
      mlir::Value value = builder.create<mlir::arith::ConstantIntOp>(
          loc(node), *staticValue, intTy.getWidth());
      return Value{value, preferredType};
    }

    Value value = emitExpression(node);
    if (!value.value)
      return value;
    if (value.type == intType()) {
      if (std::optional<std::int64_t> staticValue =
              staticPyIntValue(value.value)) {
        auto intTy = mlir::cast<mlir::IntegerType>(preferredType);
        mlir::Value primitive = builder.create<mlir::arith::ConstantIntOp>(
            loc(node), *staticValue, intTy.getWidth());
        return Value{primitive, preferredType};
      }
    }
    if (!mlir::isa<mlir::IntegerType>(value.type)) {
      error(node, "range(...) bounds must be primitive integers or static "
                  "integer literals");
      return Value{{}, preferredType};
    }
    return value;
  };

  std::set<std::string> assignedNames;
  for (const parser::NodePtr &child : *body)
    if (child)
      collectAssignedNames(*child, assignedNames);
  for (const parser::NodePtr &child : *orelse)
    if (child)
      collectAssignedNames(*child, assignedNames);
  if (assignedNames.count(*targetName)) {
    error(**target, "assignment to the for target inside the loop body or "
                    "else block is not supported yet");
    return;
  }

  Value start;
  Value stop;
  Value step;
  std::int64_t staticStepValue = 1;
  if (args->size() == 1) {
    stop = emitRangeBound(*args->front(), defaultRangeType);
    if (!stop.value)
      return;
    auto intTy = mlir::dyn_cast<mlir::IntegerType>(stop.type);
    if (!intTy) {
      error(*args->front(),
            "range(...) bounds must be primitive integer values");
      return;
    }
    mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
        loc(stmt), 0, intTy.getWidth());
    mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
        loc(stmt), 1, intTy.getWidth());
    start = Value{zero, stop.type};
    step = Value{one, stop.type};
  } else {
    start = emitRangeBound(*(*args)[0], defaultRangeType);
    stop =
        emitRangeBound(*(*args)[1], start.type ? start.type : defaultRangeType);
    if (!start.value || !stop.value)
      return;
    if (start.type != stop.type || !mlir::isa<mlir::IntegerType>(start.type)) {
      error(**iter,
            "range(...) start and stop must have the same primitive integer "
            "type");
      return;
    }
    auto intTy = mlir::cast<mlir::IntegerType>(start.type);
    if (args->size() == 2) {
      mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
          loc(stmt), 1, intTy.getWidth());
      step = Value{one, start.type};
    } else {
      std::optional<std::int64_t> staticStep;
      if (std::optional<PrimitiveConstant> constant =
              primitiveIntConstructorConstant(*(*args)[2]))
        staticStep = constant->integerValue;
      else
        staticStep = staticIntegerValue(*(*args)[2], primitiveConstants);
      if (!staticStep || *staticStep == 0) {
        error(*(*args)[2],
              "range(...) step must currently be a non-zero static primitive "
              "integer");
        return;
      }
      staticStepValue = *staticStep;
      step = emitRangeBound(*(*args)[2], start.type);
      if (!step.value)
        return;
      if (step.type != start.type) {
        error(*(*args)[2],
              "range(...) step must have the same primitive integer type as "
              "start/stop");
        return;
      }
    }
  }

  const bool reuseTarget = existingTarget != symbols.end();
  if (reuseTarget) {
    if (py::isPyType(existingTarget->second.type)) {
      error(**target, "reusing a Python object variable as a for target is not "
                      "supported yet; use a primitive static type");
      return;
    }
    if (existingTarget->second.type != start.type) {
      error(**target,
            "for target type must match range(...) bounds: expected " +
                typeString(start.type) + ", got " +
                typeString(existingTarget->second.type));
      return;
    }
  }

  std::vector<std::string> carriedNames;
  llvm::SmallVector<mlir::Type> carriedTypes;
  for (const std::string &name : assignedNames) {
    auto found = symbols.find(name);
    if (found == symbols.end())
      continue;
    if (py::isPyType(found->second.type)) {
      error(stmt, "for loop-carried Python object variable '" + name +
                      "' is not supported yet; use a primitive static type");
      return;
    }
    carriedNames.push_back(name);
    carriedTypes.push_back(found->second.type);
  }
  if (reuseTarget) {
    carriedNames.push_back(*targetName);
    carriedTypes.push_back(start.type);
  }

  mlir::Region *region = builder.getBlock()->getParent();
  mlir::Block *condBlock = new mlir::Block();
  mlir::Block *bodyBlock = new mlir::Block();
  mlir::Block *stepBlock = new mlir::Block();
  mlir::Block *elseBlock = orelse->empty() ? nullptr : new mlir::Block();
  mlir::Block *afterBlock = new mlir::Block();
  region->push_back(condBlock);
  region->push_back(bodyBlock);
  region->push_back(stepBlock);
  if (elseBlock)
    region->push_back(elseBlock);
  region->push_back(afterBlock);
  for (mlir::Type type : carriedTypes) {
    condBlock->addArgument(type, loc(stmt));
    bodyBlock->addArgument(type, loc(stmt));
    stepBlock->addArgument(type, loc(stmt));
    if (elseBlock)
      elseBlock->addArgument(type, loc(stmt));
    afterBlock->addArgument(type, loc(stmt));
  }
  condBlock->addArgument(start.type, loc(stmt));
  bodyBlock->addArgument(start.type, loc(stmt));
  stepBlock->addArgument(start.type, loc(stmt));

  llvm::SmallVector<mlir::Value> initialValues =
      currentCarriedValues(symbols, carriedNames);
  initialValues.push_back(start.value);
  builder.create<mlir::cf::BranchOp>(loc(stmt), condBlock, initialValues);
  blockTerminated = true;

  std::map<std::string, Value> outerSymbols = symbols;
  std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;
  builder.setInsertionPointToStart(condBlock);
  symbols = outerSymbols;
  callableAliases = outerCallableAliases;
  for (auto [index, name] : llvm::enumerate(carriedNames))
    symbols[name] = Value{condBlock->getArgument(index), carriedTypes[index]};
  mlir::Value condIv = condBlock->getArgument(carriedTypes.size());
  mlir::arith::CmpIPredicate predicate = staticStepValue > 0
                                             ? mlir::arith::CmpIPredicate::slt
                                             : mlir::arith::CmpIPredicate::sgt;
  mlir::Value condition = builder.create<mlir::arith::CmpIOp>(
      loc(stmt), predicate, condIv, stop.value);
  llvm::SmallVector<mlir::Value> condValues =
      currentCarriedValues(symbols, carriedNames);
  condValues.push_back(condIv);
  llvm::SmallVector<mlir::Value> exitValues =
      currentCarriedValues(symbols, carriedNames);
  mlir::Block *normalExitBlock = elseBlock ? elseBlock : afterBlock;
  builder.create<mlir::cf::CondBranchOp>(
      loc(stmt), condition, bodyBlock, condValues, normalExitBlock, exitValues);

  builder.setInsertionPointToStart(bodyBlock);
  symbols = outerSymbols;
  callableAliases = outerCallableAliases;
  for (auto [index, name] : llvm::enumerate(carriedNames))
    symbols[name] = Value{bodyBlock->getArgument(index), carriedTypes[index]};
  symbols[*targetName] =
      Value{bodyBlock->getArgument(carriedTypes.size()), start.type};
  std::vector<std::string> continueNames = carriedNames;
  continueNames.push_back(*targetName);
  LoopTarget targetInfo{afterBlock, stepBlock, carriedNames, continueNames};
  loopStack.push_back(targetInfo);
  blockTerminated = false;
  for (const parser::NodePtr &child : *body) {
    if (child && !blockTerminated)
      emitStatement(*child);
  }
  if (!blockTerminated) {
    llvm::SmallVector<mlir::Value> nextValues =
        currentCarriedValues(symbols, continueNames);
    builder.create<mlir::cf::BranchOp>(loc(stmt), stepBlock, nextValues);
  }
  loopStack.pop_back();

  builder.setInsertionPointToStart(stepBlock);
  symbols = outerSymbols;
  callableAliases = outerCallableAliases;
  for (auto [index, name] : llvm::enumerate(carriedNames))
    symbols[name] = Value{stepBlock->getArgument(index), carriedTypes[index]};
  mlir::Value stepIv = stepBlock->getArgument(carriedTypes.size());
  mlir::Value nextIv =
      builder.create<mlir::arith::AddIOp>(loc(stmt), stepIv, step.value);
  llvm::SmallVector<mlir::Value> stepValues =
      currentCarriedValues(symbols, carriedNames);
  stepValues.push_back(nextIv);
  builder.create<mlir::cf::BranchOp>(loc(stmt), condBlock, stepValues);

  if (elseBlock) {
    builder.setInsertionPointToStart(elseBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    for (auto [index, name] : llvm::enumerate(carriedNames))
      symbols[name] = Value{elseBlock->getArgument(index), carriedTypes[index]};
    blockTerminated = false;
    for (const parser::NodePtr &child : *orelse) {
      if (child && !blockTerminated)
        emitStatement(*child);
    }
    if (!blockTerminated) {
      llvm::SmallVector<mlir::Value> elseValues =
          currentCarriedValues(symbols, carriedNames);
      builder.create<mlir::cf::BranchOp>(loc(stmt), afterBlock, elseValues);
    }
  }

  symbols = std::move(outerSymbols);
  callableAliases = std::move(outerCallableAliases);
  builder.setInsertionPointToStart(afterBlock);
  for (auto [index, name] : llvm::enumerate(carriedNames))
    symbols[name] = Value{afterBlock->getArgument(index), carriedTypes[index]};
  blockTerminated = false;
}

void Builder::Impl::emitWhile(const parser::Node &stmt) {
  const parser::NodePtr *test = nodeField(stmt, "test");
  const std::vector<parser::NodePtr> *body = nodeListField(stmt, "body");
  const std::vector<parser::NodePtr> *orelse = nodeListField(stmt, "orelse");
  if (!test || !*test || !body || !orelse) {
    error(stmt, "While.test, While.body, or While.orelse is missing");
    return;
  }

  std::set<std::string> assignedNames;
  for (const parser::NodePtr &child : *body)
    if (child)
      collectAssignedNames(*child, assignedNames);
  for (const parser::NodePtr &child : *orelse)
    if (child)
      collectAssignedNames(*child, assignedNames);

  std::vector<std::string> carriedNames;
  llvm::SmallVector<mlir::Type> carriedTypes;
  for (const std::string &name : assignedNames) {
    auto found = symbols.find(name);
    if (found == symbols.end())
      continue;
    carriedNames.push_back(name);
    carriedTypes.push_back(found->second.type);
  }

  mlir::Region *region = builder.getBlock()->getParent();
  mlir::Block *condBlock = new mlir::Block();
  mlir::Block *bodyBlock = new mlir::Block();
  mlir::Block *elseBlock = orelse->empty() ? nullptr : new mlir::Block();
  mlir::Block *afterBlock = new mlir::Block();
  region->push_back(condBlock);
  region->push_back(bodyBlock);
  if (elseBlock)
    region->push_back(elseBlock);
  region->push_back(afterBlock);
  for (mlir::Type type : carriedTypes) {
    condBlock->addArgument(type, loc(stmt));
    bodyBlock->addArgument(type, loc(stmt));
    if (elseBlock)
      elseBlock->addArgument(type, loc(stmt));
    afterBlock->addArgument(type, loc(stmt));
  }

  llvm::SmallVector<mlir::Value> initialValues =
      currentCarriedValues(symbols, carriedNames);
  builder.create<mlir::cf::BranchOp>(loc(stmt), condBlock, initialValues);
  blockTerminated = true;

  std::map<std::string, Value> outerSymbols = symbols;
  std::map<std::string, FunctionInfo> outerCallableAliases = callableAliases;
  builder.setInsertionPointToStart(condBlock);
  symbols = outerSymbols;
  callableAliases = outerCallableAliases;
  for (auto [index, name] : llvm::enumerate(carriedNames)) {
    symbols[name] = Value{condBlock->getArgument(index), carriedTypes[index]};
  }
  Value condition = emitCondition(**test);
  llvm::SmallVector<mlir::Value> condValues =
      currentCarriedValues(symbols, carriedNames);
  mlir::Block *normalExitBlock = elseBlock ? elseBlock : afterBlock;
  builder.create<mlir::cf::CondBranchOp>(loc(stmt), condition.value, bodyBlock,
                                         condValues, normalExitBlock,
                                         condValues);

  builder.setInsertionPointToStart(bodyBlock);
  symbols = outerSymbols;
  callableAliases = outerCallableAliases;
  for (auto [index, name] : llvm::enumerate(carriedNames)) {
    symbols[name] = Value{bodyBlock->getArgument(index), carriedTypes[index]};
  }
  LoopTarget target{afterBlock, condBlock, carriedNames, carriedNames};
  loopStack.push_back(target);
  blockTerminated = false;
  for (const parser::NodePtr &child : *body) {
    if (child && !blockTerminated)
      emitStatement(*child);
  }
  if (!blockTerminated) {
    llvm::SmallVector<mlir::Value> nextValues =
        currentCarriedValues(symbols, carriedNames);
    builder.create<mlir::cf::BranchOp>(loc(stmt), condBlock, nextValues);
  }
  loopStack.pop_back();

  if (elseBlock) {
    builder.setInsertionPointToStart(elseBlock);
    symbols = outerSymbols;
    callableAliases = outerCallableAliases;
    for (auto [index, name] : llvm::enumerate(carriedNames)) {
      symbols[name] = Value{elseBlock->getArgument(index), carriedTypes[index]};
    }
    blockTerminated = false;
    for (const parser::NodePtr &child : *orelse) {
      if (child && !blockTerminated)
        emitStatement(*child);
    }
    if (!blockTerminated) {
      llvm::SmallVector<mlir::Value> elseValues =
          currentCarriedValues(symbols, carriedNames);
      builder.create<mlir::cf::BranchOp>(loc(stmt), afterBlock, elseValues);
    }
  }

  symbols = std::move(outerSymbols);
  callableAliases = std::move(outerCallableAliases);
  builder.setInsertionPointToStart(afterBlock);
  for (auto [index, name] : llvm::enumerate(carriedNames)) {
    symbols[name] = Value{afterBlock->getArgument(index), carriedTypes[index]};
  }
  blockTerminated = false;
}

void Builder::Impl::emitBreak(const parser::Node &stmt) {
  if (loopStack.empty()) {
    error(stmt, "'break' outside loop");
    return;
  }
  const LoopTarget &target = loopStack.back();
  llvm::SmallVector<mlir::Value> carried =
      currentCarriedValues(symbols, target.breakNames);
  builder.create<mlir::cf::BranchOp>(loc(stmt), target.breakBlock, carried);
  blockTerminated = true;
}

void Builder::Impl::emitContinue(const parser::Node &stmt) {
  if (loopStack.empty()) {
    error(stmt, "'continue' outside loop");
    return;
  }
  const LoopTarget &target = loopStack.back();
  llvm::SmallVector<mlir::Value> carried =
      currentCarriedValues(symbols, target.continueNames);
  builder.create<mlir::cf::BranchOp>(loc(stmt), target.continueBlock, carried);
  blockTerminated = true;
}

} // namespace lython::emitter
