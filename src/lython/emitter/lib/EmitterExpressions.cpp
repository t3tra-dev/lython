#include "AstSynth.h"
#include "EmitterCore.h"

#include "EmitterOps.h" // IWYU pragma: keep
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "PlatformConstants.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"

#include "llvm/ADT/ScopeExit.h"
#include "PyProtocols.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <complex>
#include <functional>
#include <optional>
#include <string>

namespace lython::emitter {

// origin -> then/else -> merge(block argument) with cf branches: the one
// shape every two-armed value merge in the emitter uses (a conditional whose
// arms may themselves open new blocks, so each arm branches from wherever
// its value became available).
mlir::Value ModuleEmitter::emitValueDiamond(
    mlir::Location location, mlir::Value condition, mlir::Type resultType,
    llvm::function_ref<mlir::Value()> emitThen,
    llvm::function_ref<mlir::Value()> emitElse) {
  mlir::Block *origin = builder.getInsertionBlock();
  mlir::Region *region = origin->getParent();
  mlir::Block *thenBlock =
      builder.createBlock(region, std::next(origin->getIterator()));
  mlir::Block *elseBlock =
      builder.createBlock(region, std::next(thenBlock->getIterator()));
  mlir::Block *merge =
      builder.createBlock(region, std::next(elseBlock->getIterator()));
  mlir::BlockArgument result = merge->addArgument(resultType, location);

  builder.setInsertionPointToEnd(origin);
  mlir::cf::CondBranchOp::create(builder, location, condition, thenBlock,
                                 mlir::ValueRange{}, elseBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToStart(thenBlock);
  mlir::Value thenValue = emitThen();
  mlir::cf::BranchOp::create(builder, location, merge,
                             mlir::ValueRange{thenValue});
  builder.setInsertionPointToStart(elseBlock);
  mlir::Value elseValue = emitElse();
  mlir::cf::BranchOp::create(builder, location, merge,
                             mlir::ValueRange{elseValue});

  builder.setInsertionPointToStart(merge);
  return result;
}

Value ModuleEmitter::emitExpr(const parser::Node *expr) {
  if (!expr)
    return {py::NoneOp::create(builder, builder.getUnknownLoc(), types.none())
                .getResult(),
            types.none()};
  // ⭐ A subexpression a desugaring must EMIT ONCE and reference twice. The
  // AST has no spelling for an already-emitted value, so a rewrite that puts
  // one subtree in two places -- `a[f()] += 1` becomes a load and a store of
  // `a[f()]` -- ran its side effects twice and stored at the second index.
  // The other half: emit the child ONCE, remember it in the slot, and hand it
  // back. A rewrite that needs a subexpression in two places wraps the first
  // occurrence in this and the rest in `LyValueRef`.
  if (expr->kind == "LyValueCapture") {
    const parser::Field *slot = parser::findField(*expr, "slot");
    Value value = emitExpr(ast::node(*expr, "value"));
    if (slot && std::holds_alternative<std::int64_t>(slot->value)) {
      auto index = static_cast<std::size_t>(std::get<std::int64_t>(slot->value));
      if (index < pendingValueRefs.size())
        pendingValueRefs[index] = value;
    }
    return value;
  }
  if (expr->kind == "LyValueRef") {
    const parser::Field *slot = parser::findField(*expr, "slot");
    if (slot && std::holds_alternative<std::int64_t>(slot->value)) {
      auto index = static_cast<std::size_t>(std::get<std::int64_t>(slot->value));
      if (index < pendingValueRefs.size())
        return pendingValueRefs[index];
    }
    return emitNone(*expr);
  }
  if (expr->kind == "Constant")
    return emitConstant(*expr);
  if (expr->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*expr);
    auto found = values.find(name);
    if (found != values.end()) {
      // A boxed (nonlocal-shared) local binds to its cell instance; the
      // expression value is the cell's current content.
      if (isCellContract(found->second.type))
        return emitCellLoad(*expr, found->second);
      return found->second;
    }
    if (isModuleGlobalRead(name)) {
      mlir::Type type = moduleGlobals.lookup(name);
      auto op = py::GlobalGetOp::create(builder, loc(*expr), type,
                                        builder.getStringAttr(name));
      markBoxedModuleGlobal(op);
      return {op.getResult(), type};
    }
    if (auto literal = moduleConstantBindings.find(name);
        literal != moduleConstantBindings.end())
      return emitConstant(*literal->second);
    auto primitiveConstant = primitiveConstants.find(name);
    if (primitiveConstant != primitiveConstants.end())
      return emitPrimitiveConstant(*expr, primitiveConstant->second);
    std::optional<mlir::Type> symbolType = types.lookupSymbol(name);
    // A top-level `def int` outranks the builtin class object of that
    // spelling, exactly as it outranks the constructor path in emitCall.
    // Without this the reference materializes a py.type_object and the call
    // fails in lowering as "calling a type object held in a value", which is
    // loud but describes the compiler's confusion rather than the program.
    if (auto cls = moduleFunctionNames.count(name)
                       ? std::optional<mlir::Type>()
                       : types.lookupClass(name)) {
      // A monomorphized generic has no single class object to materialize:
      // there is one contract per instantiation. Reject here rather than let
      // the factless generic contract flow on and fail as an erased object.
      if (std::optional<Value> v = rejectGenericClassObject(*expr, *cls))
        return *v;
      mlir::Type typeType = types.typeObject(*cls);
      auto op = py::TypeObjectOp::create(builder, loc(*expr), typeType, *cls);
      return {op.getResult(), typeType};
    }
    std::string binding = std::string(name);
    if (std::optional<std::string> canonical =
            types.lookupCanonicalBinding(name))
      binding = *canonical;
    if (!symbolType) {
      diagnostics.push_back(
          parser::Diagnostic{parser::Severity::Error, expr->range.start,
                             "unresolved name '" + std::string(name) + "'"});
      return emitNone(*expr);
    }
    if (std::optional<Value> constant =
            emitStaticStringConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant = emitStaticIntConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant =
            emitManifestFloatConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant =
            emitManifestIntConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> constant =
            emitManifestStrConstant(*expr, binding))
      return *constant;
    if (std::optional<Value> literal =
            emitLiteralTypeConstant(*expr, *symbolType))
      return *literal;
    if (genericFunctions.count(name) || genericFunctions.count(binding)) {
      // No ground context reached this reference (calls and expected-typed
      // uses are intercepted earlier), so there is no instantiation to
      // materialize; emitting the type-parameterized contract would only
      // defer the failure to the ABI check.
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr->range.start,
          "reference to generic function '" + std::string(name) +
              "' requires a call or an annotated Callable context to "
              "determine its type arguments"});
      return emitNone(*expr);
    }
    return emitBindingRef(*expr, binding, *symbolType);
  }
  if (expr->kind == "Call")
    return emitCall(*expr);
  if (expr->kind == "UnaryOp")
    return emitUnary(*expr);
  if (expr->kind == "BinOp")
    return emitBinary(*expr);
  if (expr->kind == "Compare")
    return emitCompare(*expr);
  if (expr->kind == "Subscript") {
    // `C[int]` denotes the instantiation's class object, so it is a value in
    // its own right (`C[int].attr`, `isinstance(x, C[int])`), not a
    // __getitem__ on the generic name.
    if (mlir::Type instantiated = types.genericClassSubscript(expr)) {
      mlir::Type typeType = types.typeObject(instantiated);
      auto op = py::TypeObjectOp::create(builder, loc(*expr), typeType,
                                         instantiated);
      return {op.getResult(), typeType};
    }
    return emitSubscript(*expr);
  }
  if (expr->kind == "Attribute") {
    std::string qualified = ast::qualifiedName(expr);
    if (!qualified.empty())
      if (auto cls = types.lookupClass(qualified)) {
        if (std::optional<Value> v = rejectGenericClassObject(*expr, *cls))
          return *v;
        mlir::Type typeType = types.typeObject(*cls);
        auto op = py::TypeObjectOp::create(builder, loc(*expr), typeType, *cls);
        return {op.getResult(), typeType};
      }
    if (!qualified.empty()) {
      std::string binding = qualified;
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(qualified))
        binding = *canonical;
      // Platform constants fold whether or not the module attribute is a
      // known symbol (inference already typed them as target literals).
      if (std::optional<Value> constant =
              emitStaticStringConstant(*expr, binding))
        return *constant;
      if (auto symbol = types.lookupSymbol(qualified)) {
        if (std::optional<Value> constant =
                emitStaticIntConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> constant =
                emitManifestFloatConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> constant =
                emitManifestIntConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> constant =
                emitManifestStrConstant(*expr, binding))
          return *constant;
        if (std::optional<Value> literal =
                emitLiteralTypeConstant(*expr, *symbol))
          return *literal;
        if (genericFunctions.count(binding)) {
          // Same rule as the Name case: a bare qualified reference to an
          // imported generic has no instantiation to materialize.
          diagnostics.push_back(parser::Diagnostic{
              parser::Severity::Error, expr->range.start,
              "reference to generic function '" + qualified +
                  "' requires a call or an annotated Callable context to "
                  "determine its type arguments"});
          return emitNone(*expr);
        }
        return emitBindingRef(*expr, binding, *symbol);
      }
    }
    return emitAttribute(*expr);
  }
  if (expr->kind == "Await")
    return emitAwait(*expr);
  if (expr->kind == "Yield") {
    const parser::Node *valueNode = ast::node(*expr, "value");
    Value yielded = valueNode ? emitExpr(valueNode) : emitNone(*expr);
    mlir::Type sentType =
        currentGeneratorSendType ? currentGeneratorSendType : types.none();
    auto op =
        py::YieldValueOp::create(builder, loc(*expr), sentType, yielded.value);
    return {op.getSent(), sentType};
  }
  if (expr->kind == "YieldFrom") {
    const parser::Node *source = ast::node(*expr, "value");
    if (source && (source->kind == "List" || source->kind == "Tuple")) {
      if (const auto *elts = ast::nodeList(*source, "elts"))
        for (const parser::NodePtr &element : *elts) {
          Value yielded = emitExpr(element.get());
          mlir::Type sentType = currentGeneratorSendType
                                    ? currentGeneratorSendType
                                    : types.none();
          py::YieldValueOp::create(builder, loc(*expr), sentType,
                                   yielded.value);
        }
      return emitNone(*expr);
    }
    // ⭐ `yield from X` IS `for v in X: yield v`, and writing it that way is
    // what makes it compile for anything but a literal:
    //
    //     def g() -> Iterator[int]:
    //         yield from range(2)
    //     # source generator next lowering currently supports yields whose ...
    //
    // A range, a parameter's list and a str all landed there, while the loop
    // spelling of each has always worked -- so the gap was `py.yield.from` in the
    // state machine, not the iteration. The literal arm above stays: it unrolls
    // into one yield per element and needs no loop at all.
    //
    // ⛔ Exact for an ITERABLE, which is what these operands are: `yield from`
    // over one evaluates to None (there is no StopIteration value to forward),
    // and that is what the loop leaves behind. Delegating to a SUB-GENERATOR is
    // more than this -- `send` and `throw` pass through it, and `res = yield from
    // g()` takes the return value -- and that shape does not reach here: a
    // generator operand is refused earlier, by the resume-target rule.
    // ⛔ NOT for a sub-GENERATOR: `py.yield.from` is what the state machine
    // implements for delegation, and rewriting those to a loop turned two
    // passing goldens into "a generator returned out of a function cannot be
    // resumed" -- the loop iterates a generator VALUE, which is a different
    // (and refused) shape. The rewrite is for the iterables that had no path.
    auto sourceContract = mlir::dyn_cast_if_present<py::ContractType>(
        types.widenLiteral(types.inferExpr(source)));
    bool delegatesToGenerator =
        sourceContract &&
        sourceContract.getContractName() == "types.GeneratorType";
    if (source && !delegatesToGenerator) {
      std::string element = "__ly_yieldfrom_" +
                            std::to_string(syntheticFunctionCounter++);
      parser::NodePtr yielded = parser::makeNode("Yield", expr->range);
      parser::addField(*yielded, "value",
                       synth::name(element, expr->range));
      parser::NodePtr loop = synth::forStmt(
          synth::name(element, expr->range),
          parser::NodePtr(const_cast<parser::Node *>(source),
                          [](parser::Node *) {}),
          std::vector<parser::NodePtr>{
              synth::exprStmt(std::move(yielded), expr->range)},
          {}, expr->range);
      runWithScratchNames({element}, [&] { emitStatement(*loop); });
      return emitNone(*expr);
    }
    Value delegated = emitExpr(source);
    YieldFromInferenceResult yieldFromInference =
        types.inferYieldFromWithEvidence(delegated.type);
    if (!requireStaticEvidence(*expr, yieldFromInference))
      return emitNone(*expr);
    auto op = py::YieldFromOp::create(
        builder, loc(*expr), yieldFromInference.completionType,
        yieldFromInference.protocolContract, delegated.value);
    return {op.getResult(), yieldFromInference.completionType};
  }
  if (expr->kind == "List" || expr->kind == "Tuple" || expr->kind == "Dict")
    return emitContainerLiteral(*expr);
  if (expr->kind == "Set")
    return emitSetLiteral(*expr);
  if (expr->kind == "NamedExpr") {
    // Walrus: the value of `(name := E)` is E's value; the binding lands in
    // the enclosing function scope like an ordinary assignment (CPython
    // scoping — the emitter has no expression-local scopes to leak from).
    Value value = emitExpr(ast::node(*expr, "value"));
    if (const parser::Node *target = ast::node(*expr, "target"))
      emitAssignTarget(*target, value);
    return value;
  }
  if (expr->kind == "ListComp")
    return emitListComp(*expr);
  if (expr->kind == "SetComp")
    return emitComprehension(*expr, /*isDict=*/false, /*isSet=*/true);
  if (expr->kind == "DictComp")
    return emitDictComp(*expr);
  if (expr->kind == "IfExp") {
    // Conditional expression via the same cf-block merge the if STATEMENT
    // uses (one merge mechanism; region-based scf.if results are invisible
    // to the runtime bundle machinery and the affine ownership verifier).
    const parser::Node *bodyNode = ast::node(*expr, "body");
    const parser::Node *elseNode = ast::node(*expr, "orelse");
    const parser::Node *testNode = ast::node(*expr, "test");
    // ⭐ Each arm sees the narrowing its side of the test proves, the same
    // fact the if STATEMENT applies to its branches. Without it
    // `n if n is not None else 0` typed the kept arm `int | None` and the
    // join stayed a union -- the one spelling of the Optional idiom that has
    // no statement to hang the narrowing on.
    std::optional<BranchTypeNarrowing> narrowing =
        testNode ? optionalBranchTypeNarrowing(*testNode, types, module)
                 : std::nullopt;
    auto armType = [&](const parser::Node *arm, bool conditionIsTrue) {
      if (!narrowing)
        return types.widenLiteral(types.inferExpr(arm));
      mlir::Type narrowed =
          conditionIsTrue ? narrowing->trueType : narrowing->falseType;
      if (!narrowed)
        return types.widenLiteral(types.inferExpr(arm));
      auto scope = types.pushScope();
      types.bindLocalSymbol(narrowing->name, narrowed);
      return types.widenLiteral(types.inferExpr(arm));
    };
    mlir::Type resultType = types.join(
        {armType(bodyNode, /*conditionIsTrue=*/true),
         armType(elseNode, /*conditionIsTrue=*/false)});
    mlir::Value condition = emitBoolValue(emitExpr(testNode), *expr);

    auto emitArm = [&](const parser::Node *arm, bool conditionIsTrue) {
      if (!narrowing)
        return coerceValue(emitExpr(arm), resultType, *expr).value;
      mlir::Type narrowed =
          conditionIsTrue ? narrowing->trueType : narrowing->falseType;
      auto found = values.find(narrowing->name);
      if (!narrowed || found == values.end() ||
          !mlir::isa<py::UnionType>(found->second.value.getType()) ||
          !mlir::cast<py::UnionType>(found->second.value.getType())
               .hasMember(narrowed))
        return coerceValue(emitExpr(arm), resultType, *expr).value;
      Value saved = found->second;
      std::optional<mlir::Type> savedSymbol = types.lookupSymbol(narrowing->name);
      auto unwrap = py::UnionUnwrapOp::create(builder, loc(*expr), narrowed,
                                              saved.value);
      found->second = Value{unwrap.getResult(), narrowed};
      types.bindSymbol(narrowing->name, narrowed);
      mlir::Value armValue = coerceValue(emitExpr(arm), resultType, *expr).value;
      values[narrowing->name] = saved;
      if (savedSymbol)
        types.bindSymbol(narrowing->name, *savedSymbol);
      return armValue;
    };
    mlir::Value result = emitValueDiamond(
        loc(*expr), condition, resultType,
        [&] { return emitArm(bodyNode, /*conditionIsTrue=*/true); },
        [&] { return emitArm(elseNode, /*conditionIsTrue=*/false); });
    return {result, resultType};
  }
  if (expr->kind == "BoolOp") {
    // Short-circuit `and`/`or` over BOOL-typed operands via the same
    // cf-block merge as IfExp: later operands are only evaluated when the
    // accumulated truth requires them. Restricted to bool operands because
    // CPython's BoolOp returns the deciding OPERAND VALUE — for bools the
    // truth bit IS the value, so typing the result `bool` is exact; for
    // other operand types (`x or default`) it would not be.
    const parser::Node *operatorNode = ast::node(*expr, "op");
    const auto *operandNodes = ast::nodeList(*expr, "values");
    bool isAnd = operatorNode && operatorNode->kind == "And";
    bool isOr = operatorNode && operatorNode->kind == "Or";
    if ((isAnd || isOr) && operandNodes && operandNodes->size() >= 2) {
      auto boolTyped = [&](const parser::Node *operand) {
        mlir::Type type = types.inferExpr(operand);
        if (type == types.boolType())
          return true;
        auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
        return literal && (literal.getSpelling() == "True" ||
                           literal.getSpelling() == "False");
      };
      bool allBool = true;
      for (const parser::NodePtr &operand : *operandNodes)
        if (!operand || !boolTyped(operand.get()))
          allBool = false;
      if (allBool) {
        mlir::Value accumulated =
            emitBoolValue(emitExpr(operandNodes->front().get()), *expr);
        for (unsigned index = 1; index < operandNodes->size(); ++index) {
          mlir::Block *origin = builder.getInsertionBlock();
          mlir::Region *region = origin->getParent();
          mlir::Block *evalBlock =
              builder.createBlock(region, std::next(origin->getIterator()));
          mlir::Block *merge =
              builder.createBlock(region, std::next(evalBlock->getIterator()));
          mlir::BlockArgument joined =
              merge->addArgument(builder.getI1Type(), loc(*expr));

          builder.setInsertionPointToEnd(origin);
          mlir::Value decided = mlir::arith::ConstantIntOp::create(
              builder, loc(*expr), isOr ? 1 : 0, 1);
          // and: true -> evaluate next, false -> decided(false)
          // or:  true -> decided(true), false -> evaluate next
          if (isAnd)
            mlir::cf::CondBranchOp::create(builder, loc(*expr), accumulated,
                                           evalBlock, mlir::ValueRange{},
                                           merge, mlir::ValueRange{decided});
          else
            mlir::cf::CondBranchOp::create(builder, loc(*expr), accumulated,
                                           merge, mlir::ValueRange{decided},
                                           evalBlock, mlir::ValueRange{});
          builder.setInsertionPointToStart(evalBlock);
          mlir::Value next =
              emitBoolValue(emitExpr((*operandNodes)[index].get()), *expr);
          mlir::cf::BranchOp::create(builder, loc(*expr), merge,
                                     mlir::ValueRange{next});
          builder.setInsertionPointToStart(merge);
          accumulated = joined;
        }
        auto boxed = py::CastFromPrimOp::create(builder, loc(*expr),
                                                types.boolType(), accumulated);
        return {boxed.getResult(), types.boolType()};
      }
      return emitBoolOpValue(*expr, isAnd, *operandNodes);
    }
  }
  if (expr->kind == "Lambda")
    return emitLambda(*expr);
  if (expr->kind == "JoinedStr")
    return emitJoinedStr(*expr);
  if (expr->kind == "FormattedValue")
    return emitFormattedValue(*expr);
  diagnostics.push_back(
      parser::Diagnostic{parser::Severity::Error, expr->range.start,
                         "unsupported expression kind '" + expr->kind + "'"});
  return emitNone(*expr);
}

Value ModuleEmitter::emitConstant(const parser::Node &expr) {
  if (ast::isNoneField(expr, "value")) {
    auto op = py::NoneOp::create(builder, loc(expr), types.none());
    return {op.getResult(), types.none()};
  }
  if (auto value = ast::boolean(expr, "value")) {
    mlir::Type type = types.literal(*value ? "True" : "False");
    auto op = py::BoolConstantOp::create(builder, loc(expr), type,
                                         builder.getBoolAttr(*value));
    return {op.getResult(), type};
  }
  if (auto value = ast::integer(expr, "value")) {
    std::string text = std::to_string(*value);
    mlir::Type type = types.literal(text);
    auto op = py::IntConstantOp::create(builder, loc(expr), type,
                                        builder.getStringAttr(text));
    return {op.getResult(), type};
  }
  if (auto value = ast::floating(expr, "value")) {
    auto op = py::FloatConstantOp::create(builder, loc(expr), types.floatType(),
                                          builder.getF64FloatAttr(*value));
    return {op.getResult(), types.floatType()};
  }
  if (auto value = ast::string(expr, "value")) {
    mlir::Type type = types.literal("\"" + std::string(*value) + "\"");
    auto op = py::StrConstantOp::create(builder, loc(expr), type,
                                        builder.getStringAttr(*value));
    return {op.getResult(), type};
  }
  if (const auto *value = ast::bytes(expr, "value")) {
    mlir::Type type = types.contract("builtins.bytes");
    auto op = py::BytesConstantOp::create(
        builder, loc(expr), type,
        builder.getStringAttr(llvm::StringRef(
            reinterpret_cast<const char *>(value->data()), value->size())));
    return {op.getResult(), type};
  }
  if (const auto *fieldValue = ast::field(expr, "value")) {
    if (const auto *big = std::get_if<parser::BigInteger>(fieldValue)) {
      mlir::Type type = types.literal(big->decimal);
      auto op = py::IntConstantOp::create(builder, loc(expr), type,
                                          builder.getStringAttr(big->decimal));
      return {op.getResult(), type};
    }
    if (const auto *complexValue =
            std::get_if<std::complex<double>>(fieldValue)) {
      mlir::Type type = types.contract("builtins.complex");
      auto op = py::ComplexConstantOp::create(
          builder, loc(expr), type,
          builder.getF64FloatAttr(complexValue->real()),
          builder.getF64FloatAttr(complexValue->imag()));
      return {op.getResult(), type};
    }
  }
  diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                           expr.range.start,
                                           "unsupported constant literal"});
  return emitNone(expr);
}

Value ModuleEmitter::emitUnary(const parser::Node &expr) {
  const parser::Node *op = ast::node(expr, "op");
  const parser::Node *operandNode = ast::node(expr, "operand");

  if (ast::isOperator(op, "USub") && operandNode &&
      operandNode->kind == "Constant") {
    if (auto value = ast::integer(*operandNode, "value")) {
      std::string text = "-" + std::to_string(*value);
      mlir::Type type = types.literal(text);
      auto constOp = py::IntConstantOp::create(builder, loc(expr), type,
                                               builder.getStringAttr(text));
      return {constOp.getResult(), type};
    }
    if (auto value = ast::floating(*operandNode, "value")) {
      auto constOp =
          py::FloatConstantOp::create(builder, loc(expr), types.floatType(),
                                      builder.getF64FloatAttr(-*value));
      return {constOp.getResult(), types.floatType()};
    }
    if (const auto *fieldValue = ast::field(*operandNode, "value")) {
      if (const auto *big = std::get_if<parser::BigInteger>(fieldValue)) {
        std::string text = "-" + big->decimal;
        mlir::Type type = types.literal(text);
        auto constOp = py::IntConstantOp::create(builder, loc(expr), type,
                                                 builder.getStringAttr(text));
        return {constOp.getResult(), type};
      }
      if (const auto *complexValue =
              std::get_if<std::complex<double>>(fieldValue)) {
        mlir::Type type = types.contract("builtins.complex");
        auto constOp = py::ComplexConstantOp::create(
            builder, loc(expr), type,
            builder.getF64FloatAttr(-complexValue->real()),
            builder.getF64FloatAttr(-complexValue->imag()));
        return {constOp.getResult(), type};
      }
    }
  }

  Value operand = emitExpr(operandNode);
  mlir::Type result = types.widenLiteral(types.inferExpr(&expr));
  // Complex unary results stay complex regardless of the (complex-unaware)
  // expression inference.
  if (types.widenLiteral(operand.type) == types.contract("builtins.complex") &&
      (ast::isOperator(op, "USub") || ast::isOperator(op, "UAdd")))
    result = types.contract("builtins.complex");
  // ⭐ A SOURCE class's unary dunder is called, not looked up in the manifest.
  // py.neg/py.pos/py.invert resolve their target against the runtime
  // manifest, so `-v` over a class that defines __neg__ died in the lowering
  // as "runtime manifest has no V.__neg__ method" -- while __len__ and
  // __bool__ on the same class both dispatch. This is the same repair the
  // for-loop's __iter__ needed.
  for (auto [unaryOp, method] :
       {std::pair<const char *, const char *>{"USub", "__neg__"},
        {"UAdd", "__pos__"},
        {"Invert", "__invert__"},
        {"Abs", "__abs__"}})
    if (ast::isOperator(op, unaryOp))
      if (std::optional<Value> applied =
              tryEmitClassDunder(expr, operand, method))
        return *applied;
  // ⭐ The numeric tower's bottom rung, on the one operand there is. `-True` is
  // -1 in CPython because bool inherits int's arithmetic; here it was refused
  // with "builtins.bool.__neg__ ... has no implementation".
  //
  // ⛔ `not` is excluded and stays below: it is the only unary operator whose
  // answer is a BOOL, and it already lowers through the truth bit without
  // touching int at all.
  if (types.widenLiteral(operand.type) == types.boolType() &&
      (ast::isOperator(op, "USub") || ast::isOperator(op, "UAdd") ||
       ast::isOperator(op, "Invert") || ast::isOperator(op, "Abs"))) {
    operand = emitIntFromBool(expr, operand);
    result = types.intType();
  }
  if (ast::isOperator(op, "USub"))
    return emitUnarySpecial<py::NegOp>(expr, "__neg__", operand, result);
  if (ast::isOperator(op, "UAdd"))
    return emitUnarySpecial<py::PosOp>(expr, "__pos__", operand, result);
  if (ast::isOperator(op, "Invert"))
    return emitUnarySpecial<py::InvertOp>(expr, "__invert__", operand, result);
  if (ast::isOperator(op, "Not")) {
    mlir::Value truth = emitBoolValue(operand, expr);
    auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
    auto inverted = mlir::arith::XOrIOp::create(builder, loc(expr), truth, one);
    auto pyBool = py::CastFromPrimOp::create(
        builder, loc(expr), types.boolType(), inverted.getResult());
    return {pyBool.getResult(), types.boolType()};
  }
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, expr.range.start, "unsupported unary operator"});
  return emitNone(expr);
}

Value ModuleEmitter::emitBinary(const parser::Node &expr) {
  // str % args is printf-style formatting, not a manifest __mod__; it needs
  // the unevaluated right-hand AST (tuple literals supply the arguments), so
  // intercept before the operands are emitted.
  if (ast::isOperator(ast::node(expr, "op"), "Mod")) {
    const parser::Node *leftNode = ast::node(expr, "left");
    if (leftNode && types.widenLiteral(types.inferExpr(leftNode)) ==
                        types.contract("builtins.str"))
      return emitPercentFormat(expr);
  }
  // ⭐ A set operator answers the LEFT operand's type: CPython runs the
  // left's __or__, which builds its own kind, so `frozenset | set` is a
  // frozenset and `set | frozenset` is a set. The two kinds are different
  // headers here (11 words against 13), so the mixed pair had no operator at
  // all -- `sorted(f | {3})` was refused for a value that is an ordinary
  // frozenset at run time. Converting the right operand to the left's kind
  // is the same result, computed by the operator that already exists.
  if (const parser::Node *setOp = ast::node(expr, "op");
      ast::isOperator(setOp, "BitOr") || ast::isOperator(setOp, "BitAnd") ||
      ast::isOperator(setOp, "BitXor") || ast::isOperator(setOp, "Sub")) {
    auto setKind = [&](const parser::Node *side) -> llvm::StringRef {
      auto contract = mlir::dyn_cast_if_present<py::ContractType>(
          types.widenLiteral(types.inferExpr(side)));
      if (!contract)
        return {};
      llvm::StringRef name = contract.getContractName();
      return name == "builtins.set" || name == "builtins.frozenset"
                 ? name
                 : llvm::StringRef();
    };
    const parser::Node *leftNode = ast::node(expr, "left");
    const parser::Node *rightNode = ast::node(expr, "right");
    llvm::StringRef leftKind = setKind(leftNode);
    llvm::StringRef rightKind = setKind(rightNode);
    if (!leftKind.empty() && !rightKind.empty() && leftKind != rightKind) {
      const parser::Field *rightField = parser::findField(expr, "right");
      if (rightField &&
          std::holds_alternative<parser::NodePtr>(rightField->value)) {
        parser::NodePtr converter = synth::name(leftKind.rsplit('.').second.str(), expr.range);
        parser::NodePtr call = synth::call(std::move(converter), std::vector<parser::NodePtr>{
                             std::get<parser::NodePtr>(rightField->value)}, expr.range);
        parser::NodePtr rewritten = parser::makeNode("BinOp", expr.range);
        if (const parser::Field *leftField = parser::findField(expr, "left"))
          rewritten->fields.push_back(*leftField);
        if (const parser::Field *opField = parser::findField(expr, "op"))
          rewritten->fields.push_back(*opField);
        parser::addField(*rewritten, "right", std::move(call));
        return emitBinary(*rewritten);
      }
    }
  }
  Value lhs = emitExpr(ast::node(expr, "left"));
  Value rhs = emitExpr(ast::node(expr, "right"));
  const parser::Node *op = ast::node(expr, "op");
  if (std::optional<Value> primitive = emitPrimitiveBinary(expr, lhs, rhs, op))
    return *primitive;
  // int ** compile-time negative int is a float in CPython (decision log):
  // desugar to float(base) ** float(exponent) so the manifest float.__pow__
  // carries the runtime semantics (0 ** -n raises ZeroDivisionError there).
  if (ast::isOperator(op, "Pow")) {
    auto literalType = mlir::dyn_cast_if_present<py::LiteralType>(rhs.type);
    llvm::StringRef spelling =
        literalType ? literalType.getSpelling() : llvm::StringRef();
    bool negativeIntLiteral =
        spelling.size() > 1 && spelling.front() == '-' &&
        llvm::all_of(spelling.drop_front(),
                     [](char c) { return c >= '0' && c <= '9'; });
    if (negativeIntLiteral &&
        types.widenLiteral(lhs.type) == types.intType()) {
      llvm::APFloat exponent(llvm::APFloat::IEEEdouble());
      llvm::Expected<llvm::APFloat::opStatus> status =
          exponent.convertFromString(spelling,
                                     llvm::APFloat::rmNearestTiesToEven);
      if (!status) {
        llvm::consumeError(status.takeError());
      } else if (!exponent.isInfinity()) {
        // Exponents beyond the double range stay on the int path; its
        // runtime rejection mirrors CPython's OverflowError there.
        Value baseFloat = emitFloatFromInt(expr, lhs);
        auto exponentConst = py::FloatConstantOp::create(
            builder, loc(expr), types.floatType(),
            builder.getF64FloatAttr(exponent.convertToDouble()));
        Value exponentValue{exponentConst.getResult(), types.floatType()};
        return emitBinarySpecial<py::PowOp>(expr, "__pow__", baseFloat,
                                            exponentValue, types.floatType());
      }
    }
  }
  if (std::optional<Value> complexResult = emitComplexBinary(expr, lhs, rhs, op))
    return *complexResult;
  mlir::Type left = types.widenLiteral(lhs.type);
  mlir::Type right = types.widenLiteral(rhs.type);
  // ⭐ THE BOTTOM RUNG OF THE NUMERIC TOWER, at the operands, exactly like the
  // int -> float promotion below. bool IS an int in CPython and inherits int's
  // arithmetic; here the bool contract declares the operators (typeshed shows
  // what it inherits) and implements none, so `True + 1` was refused with
  // "builtins.bool.__add__ is declared by the standard-library contract but
  // has no implementation".
  //
  // ⛔ EXCEPT the three bitwise operators over TWO bools, where CPython's
  // answer is a bool and not an int -- `True | False` is `True`, and
  // `True | 1` is `1`. Those already work through bool's own operators, and
  // promoting them would change a correct answer's type.
  if (left == types.boolType() || right == types.boolType()) {
    bool bitwiseOverTwoBools =
        left == types.boolType() && right == types.boolType() &&
        (ast::isOperator(op, "BitOr") || ast::isOperator(op, "BitAnd") ||
         ast::isOperator(op, "BitXor"));
    auto numeric = [&](mlir::Type type) {
      return type == types.boolType() || type == types.intType() ||
             type == types.floatType();
    };
    if (!bitwiseOverTwoBools && numeric(left) && numeric(right)) {
      if (left == types.boolType()) {
        lhs = emitIntFromBool(expr, lhs);
        left = types.intType();
      }
      if (right == types.boolType()) {
        rhs = emitIntFromBool(expr, rhs);
        right = types.intType();
      }
    }
  }
  // ⭐ int * sequence is sequence * int. CPython gets there by returning
  // NotImplemented from int.__mul__ and running the sequence's __rmul__,
  // which for str/list/tuple/bytes IS __mul__ with the operands swapped --
  // so the swap is the whole of the reflected operation for these four, and
  // it is decidable here because both types are known. `2 * "ab"` was
  // "builtins.int does not provide manifest method '__mul__'" while
  // `"ab" * 2` worked.
  //
  // ⛔ Not a general reflected-dunder search: that one has to try the left
  // operand, observe NotImplemented, and try the right -- a runtime protocol
  // this compiler does not have. These four are the cases where the reflected
  // method is the same method, so no protocol is needed to know the answer.
  if (ast::isOperator(op, "Mult") && left == types.intType()) {
    static constexpr llvm::StringLiteral kRepeatable[] = {
        "builtins.str", "builtins.list", "builtins.tuple", "builtins.bytes"};
    if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(right))
      if (llvm::is_contained(kRepeatable, contract.getContractName())) {
        std::swap(lhs, rhs);
        std::swap(left, right);
      }
  }
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
  // ⭐ CPython's numeric tower, at the OPERANDS and not only at the result.
  // The promotion above answered `int + float` with `builtins.float` and then
  // dispatched on the unpromoted int, which has no `__add__` taking a float:
  //
  //     print(1 + 2.0)
  //     # static type builtins.int does not provide manifest method '__add__'
  //
  // Every mixed arithmetic and every mixed comparison was refused, including
  // `1 / 2` -- CPython's answer there is 0.5, which is float division of two
  // promoted operands. In CPython the promotion happens because int.__add__
  // returns NotImplemented and float.__radd__ converts; the conversion is the
  // same one, decided statically here because both types are known.
  //
  // ⛔ Only when the two operands DISAGREE. `int / int` also answers float,
  // but CPython computes it by scaling the two integers rather than by
  // converting each to a double first, so `10**400 / 10**399` is 10.0 where
  // converting the operands raises OverflowError -- which is what promoting
  // on the RESULT type did to `int_float_conversion`. The int/int true
  // division already has that path; what was missing is only the mixed pair.
  if (left == types.intType() && right == types.floatType())
    lhs = emitFloatFromInt(expr, lhs);
  else if (left == types.floatType() && right == types.intType())
    rhs = emitFloatFromInt(expr, rhs);
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
  if (ast::isOperator(op, "Pow"))
    return emitBinarySpecial<py::PowOp>(expr, "__pow__", lhs, rhs, result);
  if (ast::isOperator(op, "Add"))
    return emitBinarySpecial<py::AddOp>(expr, "__add__", lhs, rhs, result);
  // A fall-through to __add__ here would silently mis-execute unhandled
  // operators (`a @ b` on non-tensors used to become an addition).
  std::string spelling = op ? op->kind : std::string("<missing>");
  diagnostics.push_back(parser::Diagnostic{
      parser::Severity::Error, expr.range.start,
      "binary operator '" + spelling + "' is not supported for these operand "
      "types yet"});
  return emitNone(expr);
}

std::optional<Value> ModuleEmitter::emitComplexBinary(const parser::Node &expr,
                                                      Value lhs, Value rhs,
                                                      const parser::Node *op) {
  mlir::Type complexType = types.contract("builtins.complex");
  auto isComplex = [&](const Value &value) {
    return types.widenLiteral(value.type) == complexType;
  };
  if (!isComplex(lhs) && !isComplex(rhs))
    return std::nullopt;

  bool isAdd = ast::isOperator(op, "Add");
  bool isSub = ast::isOperator(op, "Sub");
  bool isMul = ast::isOperator(op, "Mult");
  bool isDiv = ast::isOperator(op, "Div");
  if (!isAdd && !isSub && !isMul && !isDiv) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "complex supports only +, -, *, / (CPython raises TypeError for the "
        "other operators)"});
    return emitNone(expr);
  }

  // Compile-time parts of a constant operand (complex, float, or small int).
  auto constantParts =
      [&](const Value &value) -> std::optional<std::complex<double>> {
    if (!value.value)
      return std::nullopt;
    if (auto constant = value.value.getDefiningOp<py::ComplexConstantOp>())
      return std::complex<double>(constant.getReal().convertToDouble(),
                                  constant.getImag().convertToDouble());
    if (auto constant = value.value.getDefiningOp<py::FloatConstantOp>())
      return std::complex<double>(constant.getValue().convertToDouble(), 0.0);
    if (auto constant = value.value.getDefiningOp<py::IntConstantOp>()) {
      long long parsed = 0;
      if (!llvm::StringRef(constant.getValue()).getAsInteger(10, parsed))
        return std::complex<double>(static_cast<double>(parsed), 0.0);
    }
    return std::nullopt;
  };
  std::optional<std::complex<double>> lhsParts = constantParts(lhs);
  std::optional<std::complex<double>> rhsParts = constantParts(rhs);

  auto materialize = [&](std::complex<double> parts) -> Value {
    auto constant = py::ComplexConstantOp::create(
        builder, loc(expr), complexType, builder.getF64FloatAttr(parts.real()),
        builder.getF64FloatAttr(parts.imag()));
    return {constant.getResult(), complexType};
  };

  // Both constant: fold (`1 + 2j` IS a BinOp over two constants). Constant
  // division by zero stays a runtime raise.
  if (lhsParts && rhsParts &&
      !(isDiv && rhsParts->real() == 0.0 && rhsParts->imag() == 0.0)) {
    std::complex<double> folded =
        isAdd   ? *lhsParts + *rhsParts
        : isSub ? *lhsParts - *rhsParts
        : isMul ? *lhsParts * *rhsParts
                : *lhsParts / *rhsParts;
    return materialize(folded);
  }

  auto promote = [&](Value value,
                     std::optional<std::complex<double>> parts)
      -> std::optional<Value> {
    if (isComplex(value))
      return value;
    if (parts)
      return materialize(*parts);
    return std::nullopt;
  };
  std::optional<Value> promotedLhs = promote(lhs, lhsParts);
  std::optional<Value> promotedRhs = promote(rhs, rhsParts);
  if (!promotedLhs || !promotedRhs) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "complex arithmetic with a non-constant int/float operand is not "
        "supported yet; both operands must be complex (or numeric constants)"});
    return emitNone(expr);
  }
  if (isAdd)
    return emitBinarySpecial<py::AddOp>(expr, "__add__", *promotedLhs,
                                        *promotedRhs, complexType);
  if (isSub)
    return emitBinarySpecial<py::SubOp>(expr, "__sub__", *promotedLhs,
                                        *promotedRhs, complexType);
  if (isMul)
    return emitBinarySpecial<py::MulOp>(expr, "__mul__", *promotedLhs,
                                        *promotedRhs, complexType);
  return emitBinarySpecial<py::DivOp>(expr, "__truediv__", *promotedLhs,
                                      *promotedRhs, complexType);
}

Value ModuleEmitter::emitCompare(const parser::Node &expr) {
  // Membership against dict views rewrites before operand emission — the
  // views have no runtime object (EmitterIterators.cpp).
  if (std::optional<Value> view = tryEmitDictViewMembership(expr))
    return *view;
  Value lhs = emitExpr(ast::node(expr, "left"));
  const auto *comparators = ast::nodeList(expr, "comparators");
  const auto *ops = ast::nodeList(expr, "ops");
  if (!comparators || comparators->empty()) {
    auto op = py::BoolConstantOp::create(
        builder, loc(expr), types.literal("False"), builder.getBoolAttr(false));
    return {op.getResult(), types.literal("False")};
  }
  // A chained comparison (`lo <= x <= hi`) is the conjunction of its adjacent
  // pairs, with each operand evaluated ONCE — the middle operand is the rhs of
  // one pair and the lhs of the next. Only the first pair used to be emitted,
  // so `48 <= ord(c) <= 57` silently answered `48 <= ord(c)`.
  //
  // ⭐ It is also SHORT-CIRCUITING: `1 > 2 > s()` never calls `s`. The pairs
  // used to be emitted eagerly and their truth bits ANDed, because the AST
  // rewrite to `a op b and b op c` duplicates the middle operand and would
  // evaluate it twice -- a worse break than not short-circuiting.
  //
  // Both properties are available now that a rewrite can name an
  // already-emitted value: each middle operand is emitted ONCE inside the
  // pair that first needs it (`LyValueCapture`) and referred to from the next
  // pair (`LyValueRef`), and the pairs go through the existing
  // short-circuiting BoolOp path.
  if (comparators->size() > 1 && ops && ops->size() == comparators->size()) {
    std::size_t refStart = pendingValueRefs.size();
    pendingValueRefs.resize(refStart + comparators->size() - 1);
    auto releaseRefs = llvm::make_scope_exit(
        [&] { pendingValueRefs.resize(refStart); });
    auto refNode = [&](std::size_t slot) {
      parser::NodePtr node = parser::makeNode("LyValueRef", expr.range);
      parser::addField(*node, "slot", static_cast<std::int64_t>(slot));
      return node;
    };
    const parser::Field *leftField = parser::findField(expr, "left");
    parser::NodePtr leftNode =
        leftField && std::holds_alternative<parser::NodePtr>(leftField->value)
            ? std::get<parser::NodePtr>(leftField->value)
            : nullptr;
    // ⛔ RIGHT-nested, not left. A capture is an SSA value defined in the arm
    // that made it, and a left-nested `((p1 and p2) and p3)` puts p3 after the
    // merge -- where p2's capture does not dominate it ("operand #0 does not
    // dominate this use" on any four-operand chain). Right-nesting emits each
    // pair INSIDE the arm of the one before it, which is also the shape the
    // evaluation order describes.
    llvm::SmallVector<parser::NodePtr, 4> pairs;
    for (std::size_t index = 0; index < comparators->size(); ++index) {
      parser::NodePtr pair = parser::makeNode("Compare", expr.range);
      parser::addField(*pair, "left",
                       index == 0 ? leftNode : refNode(refStart + index - 1));
      parser::NodePtr rhs = (*comparators)[index];
      if (index + 1 < comparators->size()) {
        parser::NodePtr capture =
            parser::makeNode("LyValueCapture", expr.range);
        parser::addField(*capture, "slot",
                         static_cast<std::int64_t>(refStart + index));
        parser::addField(*capture, "value", std::move(rhs));
        rhs = std::move(capture);
      }
      parser::addField(*pair, "comparators",
                       std::vector<parser::NodePtr>{std::move(rhs)});
      parser::addField(*pair, "ops",
                       std::vector<parser::NodePtr>{(*ops)[index]});
      pairs.push_back(std::move(pair));
    }
    parser::NodePtr conjunction;
    for (std::size_t back = pairs.size(); back > 0; --back) {
      parser::NodePtr pair = std::move(pairs[back - 1]);
      if (!conjunction) {
        conjunction = std::move(pair);
        continue;
      }
      parser::NodePtr both = parser::makeNode("BoolOp", expr.range);
      parser::addField(*both, "op", parser::makeNode("And", expr.range));
      parser::addField(*both, "values",
                       std::vector<parser::NodePtr>{std::move(pair),
                                                    std::move(conjunction)});
      conjunction = std::move(both);
    }
    if (conjunction)
      return emitExpr(conjunction.get());
  }
  Value result{};
  for (std::size_t index = 0; index < comparators->size(); ++index) {
    Value rhs = emitExpr((*comparators)[index].get());
    const parser::Node *op =
        ops && index < ops->size() ? (*ops)[index].get() : nullptr;
    std::optional<Value> optional = emitOptionalCompare(expr, lhs, rhs, op);
    Value pairwise = optional ? *optional : emitScalarCompare(expr, lhs, rhs, op);
    if (index == 0) {
      result = pairwise;
    } else {
      mlir::Value carried = emitBoolValue(result, expr);
      mlir::Value current = emitBoolValue(pairwise, expr);
      mlir::Value both =
          mlir::arith::AndIOp::create(builder, loc(expr), carried, current)
              .getResult();
      auto pyBool =
          py::CastFromPrimOp::create(builder, loc(expr), types.boolType(), both);
      result = Value{pyBool.getResult(), types.boolType()};
    }
    lhs = rhs;
  }
  return result;
}

Value ModuleEmitter::emitScalarCompare(const parser::Node &expr, Value lhs,
                                       Value rhs, const parser::Node *op) {
  if (std::optional<Value> primitive = emitPrimitiveCompare(expr, lhs, rhs, op))
    return *primitive;
  // bool-vs-bool equality lowers through the values' truth bits: the runtime
  // has no `bool.__eq__` manifest method, and two bools are equal exactly when
  // their truth values agree.
  auto isBoolLike = [&](mlir::Type type) {
    if (type == types.boolType())
      return true;
    auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
    return literal && (literal.getSpelling() == "True" ||
                       literal.getSpelling() == "False");
  };
  if ((ast::isOperator(op, "Eq") || ast::isOperator(op, "NotEq") ||
       ast::isOperator(op, "Is") || ast::isOperator(op, "IsNot")) &&
      isBoolLike(lhs.type) && isBoolLike(rhs.type)) {
    mlir::Value lhsBit = emitBoolValue(lhs, expr);
    mlir::Value rhsBit = emitBoolValue(rhs, expr);
    bool negated =
        ast::isOperator(op, "NotEq") || ast::isOperator(op, "IsNot");
    auto compared = mlir::arith::CmpIOp::create(
        builder, loc(expr),
        negated ? mlir::arith::CmpIPredicate::ne : mlir::arith::CmpIPredicate::eq,
        lhsBit, rhsBit);
    auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                             types.boolType(),
                                             compared.getResult());
    return Value{pyBool.getResult(), types.boolType()};
  }
  // ⭐ The same bottom rung of the numeric tower emitBinary applies, applied to
  // the comparison operands: bool IS an int, so the truth bit is widened and
  // int's comparison runs. `True < 2` was refused with "builtins.bool.__lt__ is
  // declared by the standard-library contract but has no implementation", and
  // so were `==`, `!=` and the other three orderings against a number.
  //
  // ⛔ AFTER the bool-vs-bool equality above, not before: two bools compare
  // through their truth bits with no boxing at all, and that is both correct
  // and cheaper. This rung is for the pairs that reach int's operators.
  //
  // ⛔ And not for `is` / `is not`, which R6 below rejects on value types --
  // promoting first would replace that diagnostic with an int identity test,
  // which is the answer R6 exists to refuse to give.
  {
    auto numericOperand = [&](mlir::Type type) {
      mlir::Type widened = types.widenLiteral(type);
      return widened == types.boolType() || widened == types.intType() ||
             widened == types.floatType();
    };
    bool comparesNumerically =
        ast::isOperator(op, "Lt") || ast::isOperator(op, "LtE") ||
        ast::isOperator(op, "Gt") || ast::isOperator(op, "GtE") ||
        ast::isOperator(op, "Eq") || ast::isOperator(op, "NotEq");
    if (comparesNumerically && (isBoolLike(lhs.type) || isBoolLike(rhs.type)) &&
        numericOperand(lhs.type) && numericOperand(rhs.type)) {
      if (isBoolLike(lhs.type))
        lhs = emitIntFromBool(expr, lhs);
      if (isBoolLike(rhs.type))
        rhs = emitIntFromBool(expr, rhs);
    }
  }
  // ⭐ `type(a) == type(b)` IS `type(a) is type(b)`: a class has exactly one
  // type object, so equality on two of them is which classes they name, and the
  // `is` spelling already folds that way. Without this the == reached the
  // manifest dispatch and reported "!py.type<...> does not provide manifest
  // method '__eq__'", which is true and useless -- the two spellings are the
  // same question and a reader picks either.
  if (ast::isOperator(op, "Eq") || ast::isOperator(op, "NotEq")) {
    if (auto lhsType = mlir::dyn_cast_if_present<py::TypeType>(lhs.type))
      if (auto rhsType = mlir::dyn_cast_if_present<py::TypeType>(rhs.type)) {
        bool same = lhsType.getInstanceType() == rhsType.getInstanceType();
        bool truth = ast::isOperator(op, "NotEq") ? !same : same;
        mlir::Type literalType = types.literal(truth ? "True" : "False");
        auto constant = py::BoolConstantOp::create(
            builder, loc(expr), literalType, builder.getBoolAttr(truth));
        return Value{constant.getResult(), literalType};
      }
  }
  auto emitNoneIdentityTest = [&](Value candidate,
                                  Value other) -> std::optional<Value> {
    auto unionType = mlir::dyn_cast_if_present<py::UnionType>(candidate.type);
    if (!unionType || !unionType.hasMember(types.none()) ||
        !isNoneTypeLike(other.type))
      return std::nullopt;

    auto test = py::UnionTestOp::create(builder, loc(expr), builder.getI1Type(),
                                        candidate.value,
                                        mlir::TypeAttr::get(types.none()));
    mlir::Value bit = test.getResult();
    if (ast::isOperator(op, "IsNot")) {
      auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
      bit = mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
    }
    auto pyBool =
        py::CastFromPrimOp::create(builder, loc(expr), types.boolType(), bit);
    return Value{pyBool.getResult(), types.boolType()};
  };
  if (ast::isOperator(op, "Is") || ast::isOperator(op, "IsNot")) {
    if (auto narrowed = emitNoneIdentityTest(lhs, rhs))
      return *narrowed;
    if (auto narrowed = emitNoneIdentityTest(rhs, lhs))
      return *narrowed;
    // Identity against None is static once no union is involved: None is a
    // singleton, so `concrete is None` folds to False (True under `is not`).
    bool negatedIdentity = ast::isOperator(op, "IsNot");
    bool lhsIsNone = isNoneTypeLike(lhs.type);
    bool rhsIsNone = isNoneTypeLike(rhs.type);
    if (lhsIsNone || rhsIsNone) {
      bool identical = lhsIsNone && rhsIsNone;
      bool truth = negatedIdentity ? !identical : identical;
      mlir::Type literalType = types.literal(truth ? "True" : "False");
      auto constant = py::BoolConstantOp::create(
          builder, loc(expr), literalType, builder.getBoolAttr(truth));
      return Value{constant.getResult(), literalType};
    }
    // ⭐ TWO TYPE OBJECTS COMPARE BY CONTRACT, at compile time. A class has
    // exactly one type object in CPython, so `type(x) is C` and `C is C` are
    // decided by which classes they name -- and both were refused as "no stable
    // identity", which took the standard exact-class test with them. This is
    // the same fold `C.__name__` gets, and for the same reason: the answer
    // cannot depend on anything the program does at run time.
    if (auto lhsType = mlir::dyn_cast_if_present<py::TypeType>(lhs.type))
      if (auto rhsType = mlir::dyn_cast_if_present<py::TypeType>(rhs.type)) {
        bool same = lhsType.getInstanceType() == rhsType.getInstanceType();
        bool truth = negatedIdentity ? !same : same;
        mlir::Type literalType = types.literal(truth ? "True" : "False");
        auto constant = py::BoolConstantOp::create(
            builder, loc(expr), literalType, builder.getBoolAttr(truth));
        return Value{constant.getResult(), literalType};
      }
    // R6: identity on value types has no stable meaning (interning is an
    // implementation detail even in CPython); require the equality operator.
    auto isValueType = [&](mlir::Type type) {
      mlir::Type widened = types.widenLiteral(type);
      return widened == types.intType() || widened == types.floatType() ||
             widened == types.strType() ||
             widened == types.contract("builtins.bytes") ||
             widened == types.contract("builtins.complex");
    };
    if (isValueType(lhs.type) || isValueType(rhs.type)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          std::string("`is` on int/str/float/bytes/complex operands is "
                      "rejected (identity of value types is an implementation "
                      "detail); use `") +
              (negatedIdentity ? "!=" : "==") + "` instead"});
      return emitNone(expr);
    }
    // Reference identity between header-carrying contracts (user-class
    // instances, containers) is an address comparison; dispatching the
    // fall-through __eq__/__ne__ here would silently turn `is` into `==`.
    mlir::Type lhsWidened = types.widenLiteral(lhs.type);
    mlir::Type rhsWidened = types.widenLiteral(rhs.type);
    if (mlir::isa<py::ContractType>(lhsWidened) &&
        mlir::isa<py::ContractType>(rhsWidened)) {
      Value lhsRef = coerceValue(lhs, lhsWidened, expr);
      Value rhsRef = coerceValue(rhs, rhsWidened, expr);
      auto identity = py::IsOp::create(builder, loc(expr), builder.getI1Type(),
                                       lhsRef.value, rhsRef.value);
      mlir::Value bit = identity.getResult();
      if (negatedIdentity) {
        auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
        bit = mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
      }
      auto pyBool =
          py::CastFromPrimOp::create(builder, loc(expr), types.boolType(), bit);
      return Value{pyBool.getResult(), types.boolType()};
    }
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "`is` requires reference-typed operands that resolve statically; "
        "this operand combination has no stable identity"});
    return emitNone(expr);
  }
  if (ast::isOperator(op, "In") || ast::isOperator(op, "NotIn")) {
    if (std::optional<Value> contains =
            tryEmitClassDunder(expr, rhs, "__contains__", {lhs})) {
      Value membership = *contains;
      if (ast::isOperator(op, "In"))
        return membership;
      mlir::Value bit = emitBoolValue(membership, expr);
      auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
      mlir::Value flipped =
          mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
      auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                               types.boolType(), flipped);
      return Value{pyBool.getResult(), types.boolType()};
    }
    CallInferenceResult inference =
        types.inferMethodCallWithEvidence(rhs.type, "__contains__", {lhs.type});
    if (!requireStaticEvidence(expr, inference))
      return emitNone(expr);
    auto contains = py::ContainsOp::create(
        builder, loc(expr), builder.getI1Type(),
        mlir::FlatSymbolRefAttr::get(&context, "__contains__"),
        callProtocolFor(inference), rhs.value, lhs.value);
    mlir::Value bit = contains.getResult();
    if (ast::isOperator(op, "NotIn")) {
      auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
      bit = mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
    }
    auto pyBool =
        py::CastFromPrimOp::create(builder, loc(expr), types.boolType(), bit);
    return Value{pyBool.getResult(), types.boolType()};
  }
  // ⭐ A generated `__eq__` compares fields; CPython's compares CLASSES first.
  //
  //     @dataclass
  //     class A: x: int
  //     @dataclass
  //     class B: x: int
  //     A(1) == B(1)      # printed True; CPython prints False
  //
  // `dataclasses` opens its `__eq__` with `if other.__class__ is
  // self.__class__`, returning `NotImplemented` otherwise -- so `==` between
  // unrelated classes, and between a base and its subclass, is False. The
  // synthesized body here has no such guard, and its `other` parameter is
  // typed as the class itself, so the field comparison ran on operands of two
  // different classes and answered on the fields alone.
  //
  // Folded at the comparison instead of guarding inside the body: both classes
  // are known here, which is the whole question, and CPython's answer for a
  // class mismatch is a constant.
  // ⭐ `==` ACROSS TWO FAMILIES IS A CONSTANT, not a call that does not exist.
  // CPython answers it with the NotImplemented rule -- neither side's __eq__
  // accepts the other, so the comparison falls back to identity and is False --
  // and Lython went looking for a runtime method with the other side's shape:
  //
  //     print("a" == 1)
  //     # cannot adapt builtins.int to runtime input 2 of builtins.str.__eq__
  //
  // The manifest is RIGHT to have only `str.__eq__(str, str)`; there is no
  // runtime question here. Both types are known at this point, which is the
  // whole question, exactly as for the dataclass fold below.
  //
  // ⛔ Only families whose cross-family answer is unconditionally False. NOT
  // container kinds: `{1} == frozenset({1})` is True, and a set and a
  // frozenset are as different as a str and an int look from here. NOT source
  // classes: a hand-written __eq__ answers whatever it likes, which is the
  // measurement recorded on the dataclass fold below.
  if (ast::isOperator(op, "Eq") || ast::isOperator(op, "NotEq")) {
    auto valueFamily = [&](mlir::Type type) -> llvm::StringRef {
      // None is a family of one. `x == None` for a concrete x is False, and
      // under a union's tag it is the arm the other members leave: without it
      // the int arm of `int | None` went looking for `int.__eq__(NoneType)`.
      if (isNoneTypeLike(type))
        return "none";
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(type));
      if (!contract)
        return {};
      llvm::StringRef name = contract.getContractName();
      if (name == "builtins.bool" || name == "builtins.int" ||
          name == "builtins.float" || name == "builtins.complex")
        return "number";
      if (name == "builtins.str")
        return "text";
      if (name == "builtins.bytes" || name == "builtins.bytearray")
        return "binary";
      return {};
    };
    llvm::StringRef lhsFamily = valueFamily(lhs.type);
    llvm::StringRef rhsFamily = valueFamily(rhs.type);
    // Two Nones are the one WITHIN-family pair that is also a constant: the
    // type has a single inhabitant, so identity settles it. Without this the
    // pair reached `types.NoneType.__eq__`, "declared by the standard-library
    // contract but has no implementation in Lython".
    bool bothNone = lhsFamily == "none" && rhsFamily == "none";
    if (!lhsFamily.empty() && !rhsFamily.empty() &&
        (lhsFamily != rhsFamily || bothNone)) {
      bool equal = bothNone;
      mlir::Value bit = mlir::arith::ConstantIntOp::create(
                            builder, loc(expr),
                            ast::isOperator(op, "NotEq") ? !equal : equal, 1)
                            .getResult();
      auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                               types.boolType(), bit);
      return Value{pyBool.getResult(), types.boolType()};
    }
  }

  // ⭐ AND A UNION COMPARES PER MEMBER, decided by the tag. `emitOptionalCompare`
  // above answers the two-way shape (None plus exactly one member) and declines
  // everything else, so `rec["age"] == 30` on a `str | int` record went to the
  // manifest -- "static type !py.union<int, str> does not provide manifest
  // method '__eq__'". Under the tag each member is concrete, and the arms whose
  // member is a different family from the other operand fold to the constant
  // above rather than needing a runtime method that does not exist.
  if (ast::isOperator(op, "Eq") || ast::isOperator(op, "NotEq")) {
    auto lhsUnion = mlir::dyn_cast_if_present<py::UnionType>(lhs.type);
    auto rhsUnion = mlir::dyn_cast_if_present<py::UnionType>(rhs.type);
    // One side at a time: the recursion below hands the other union straight
    // back to this arm with a concrete operand on the near side.
    if (lhsUnion || rhsUnion) {
      bool onLeft = static_cast<bool>(lhsUnion);
      py::UnionType unionType = onLeft ? lhsUnion : rhsUnion;
      llvm::ArrayRef<mlir::Type> members = unionType.getMemberTypes();
      mlir::Type resultType = types.boolType();
      Value subject = onLeft ? lhs : rhs;
      auto compareMember = [&](mlir::Type member) -> mlir::Value {
        // None has no header to unwrap; against a concrete operand it is
        // unequal, and against another None `emitOptionalCompare` has already
        // answered.
        Value projected;
        if (isNoneTypeLike(member)) {
          projected = Value{mlir::Value{}, member};
        } else {
          auto unwrap = py::UnionUnwrapOp::create(builder, loc(expr), member,
                                                  subject.value);
          projected = Value{unwrap.getResult(), member};
        }
        if (!projected.value) {
          bool unequal = !isNoneTypeLike(onLeft ? rhs.type : lhs.type);
          bool bit = ast::isOperator(op, "NotEq") ? unequal : !unequal;
          mlir::Value constant =
              mlir::arith::ConstantIntOp::create(builder, loc(expr), bit, 1)
                  .getResult();
          return py::CastFromPrimOp::create(builder, loc(expr), resultType,
                                            constant)
              .getResult();
        }
        Value compared = onLeft ? emitScalarCompare(expr, projected, rhs, op)
                                : emitScalarCompare(expr, lhs, projected, op);
        return coerceValue(compared, resultType, expr).value;
      };
      auto dispatch = [&](unsigned index, auto &&recurse) -> mlir::Value {
        if (index + 1 >= members.size())
          return compareMember(members[index]);
        auto test = py::UnionTestOp::create(
            builder, loc(expr), builder.getI1Type(), subject.value,
            mlir::TypeAttr::get(members[index]));
        return emitValueDiamond(
            loc(expr), test.getResult(), resultType,
            [&] { return compareMember(members[index]); },
            [&] { return recurse(index + 1, recurse); });
      };
      return Value{dispatch(0, dispatch), resultType};
    }
  }

  if (ast::isOperator(op, "Eq") || ast::isOperator(op, "NotEq")) {
    auto lhsContract =
        mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(lhs.type));
    auto rhsContract =
        mlir::dyn_cast_if_present<py::ContractType>(types.widenLiteral(rhs.type));
    // ⛔ ONLY when both sides carry the SYNTHESIZED dataclass `__eq__`. The
    // fold reasoned that two distinct classes compare unequal because each
    // has its own `__eq__` -- which says nothing about what that `__eq__`
    // answers. Two shapes it got wrong:
    //
    //     class A(NamedTuple): v: int
    //     class B(NamedTuple): v: int
    //     print(A(1) == B(1))      # printed False; CPython prints True
    //
    //     class X:
    //         def __eq__(self, o: object) -> bool: return True
    //     class Y:
    //         def __eq__(self, o: object) -> bool: return True
    //     print(X() == Y())        # printed False; CPython prints True
    //
    // A NamedTuple's `__eq__` is tuple's and compares by contents across
    // classes; a hand-written one answers whatever it likes. Only the
    // synthesized dataclass comparison has the class guard the fold assumes,
    // so only that one may be folded.
    auto classGuardedEq = [&](py::ContractType contract) {
      return classesWithClassGuardedEq.contains(contract.getContractName());
    };
    if (lhsContract && rhsContract && lhsContract != rhsContract &&
        classGuardedEq(lhsContract) && classGuardedEq(rhsContract))
    {
      mlir::Value bit = mlir::arith::ConstantIntOp::create(
                            builder, loc(expr),
                            ast::isOperator(op, "NotEq") ? 1 : 0, 1)
                            .getResult();
      auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                               types.boolType(), bit);
      return Value{pyBool.getResult(), types.boolType()};
    }
  }
  if (ast::isOperator(op, "NotEq") || ast::isOperator(op, "IsNot")) {
    // CPython's object.__ne__ negates whatever __eq__ RESOLVED to, so a class
    // that supplies only __eq__ (a user definition, a dataclass or an enum)
    // gets != for free. Derived here rather than left to the boxed __ne__:
    // that dispatcher reaches a source __eq__ only through the uniform
    // class-id hook, which admits at most five memref operands — a two-field
    // dataclass's __eq__ takes six, so it was missed and `!=` silently
    // answered identity (`Point(1,2) != Point(1,2)` was True).
    if (!lookupClassMethod(lhs.type, "__ne__") &&
        lookupClassMethod(lhs.type, "__eq__")) {
      Value equal = emitBinarySpecial<py::EqOp>(expr, "__eq__", lhs, rhs,
                                               types.boolType());
      mlir::Value bit = emitBoolValue(equal, expr);
      auto one = mlir::arith::ConstantIntOp::create(builder, loc(expr), 1, 1);
      mlir::Value flipped =
          mlir::arith::XOrIOp::create(builder, loc(expr), bit, one);
      auto pyBool = py::CastFromPrimOp::create(builder, loc(expr),
                                               types.boolType(), flipped);
      return Value{pyBool.getResult(), types.boolType()};
    }
    return emitBinarySpecial<py::NeOp>(expr, "__ne__", lhs, rhs,
                                       types.boolType());
  }
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

std::optional<Value> ModuleEmitter::emitOptionalCompare(
    const parser::Node &expr, Value lhs, Value rhs, const parser::Node *op) {
  // Only `==` / `!=` have well-defined semantics against `None`; ordering
  // comparisons on an Optional are a TypeError in CPython, so we let those
  // fall through to the manifest path (which rejects them).
  bool negated = ast::isOperator(op, "NotEq");
  if (!ast::isOperator(op, "Eq") && !negated)
    return std::nullopt;

  // An `Optional[T]` operand is a union carrying `None` plus exactly one
  // concrete member T. The single-member restriction keeps the active-member
  // dispatch two-way (None vs T); richer unions stay unsupported.
  auto singleMemberOptional = [&](mlir::Type type) -> mlir::Type {
    auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type);
    if (!unionType || !unionType.hasMember(types.none()))
      return nullptr;
    mlir::Type member;
    for (mlir::Type candidate : unionType.getMemberTypes()) {
      if (isNoneTypeLike(candidate))
        continue;
      if (member)
        return nullptr;
      member = candidate;
    }
    return member;
  };

  auto concrete = [&](Value value) {
    return !mlir::isa<py::UnionType>(value.type) && !isNoneTypeLike(value.type);
  };
  mlir::Type lhsMember = singleMemberOptional(lhs.type);
  mlir::Type rhsMember = singleMemberOptional(rhs.type);
  bool bothOptional = lhsMember && rhsMember;
  bool lhsOptional = lhsMember && concrete(rhs);
  bool rhsOptional = rhsMember && concrete(lhs);
  if (!bothOptional && !lhsOptional && !rhsOptional)
    return std::nullopt;

  mlir::Type resultType = types.boolType();
  mlir::Location location = loc(expr);
  auto testNone = [&](Value value) {
    return py::UnionTestOp::create(builder, location, builder.getI1Type(),
                                   value.value,
                                   mlir::TypeAttr::get(types.none()))
        .getResult();
  };
  auto negate = [&](mlir::Value bit) {
    mlir::Value one =
        mlir::arith::ConstantIntOp::create(builder, location, 1, 1);
    return mlir::arith::XOrIOp::create(builder, location, bit, one).getResult();
  };

  // `present` (computed in the origin block) selects the member-compare branch;
  // its complement is the "absent" branch whose value `absentBit` supplies. For
  // one Optional operand the member is always present iff that union is not
  // None; for two, both must be present, and when they are not, equality holds
  // exactly when both are None.
  mlir::Value present;
  mlir::Value absentBit;
  if (bothOptional) {
    mlir::Value lhsNone = testNone(lhs);
    mlir::Value rhsNone = testNone(rhs);
    present = mlir::arith::AndIOp::create(builder, location, negate(lhsNone),
                                          negate(rhsNone));
    absentBit =
        mlir::arith::AndIOp::create(builder, location, lhsNone, rhsNone);
  } else {
    Value optional = lhsOptional ? lhs : rhs;
    present = negate(testNone(optional));
    // `None == concrete` is False (True under `!=`); the negation below flips it.
    absentBit = mlir::arith::ConstantIntOp::create(builder, location, 0, 1);
  }
  if (negated)
    absentBit = negate(absentBit);

  // cf-block merge, mirroring the IfExp lowering (region-based scf.if results
  // are invisible to the runtime bundle machinery and the ownership verifier).
  mlir::Value result = emitValueDiamond(
      location, present, resultType,
      // Present branch: project the concrete member(s) and re-enter scalar
      // dispatch.
      [&] {
        auto unwrap = [&](Value value, mlir::Type member) -> Value {
          auto op =
              py::UnionUnwrapOp::create(builder, location, member, value.value);
          return Value{op.getResult(), member};
        };
        Value presentLhs = lhsMember ? unwrap(lhs, lhsMember) : lhs;
        Value presentRhs = rhsMember ? unwrap(rhs, rhsMember) : rhs;
        Value compared = emitScalarCompare(expr, presentLhs, presentRhs, op);
        return coerceValue(compared, resultType, expr).value;
      },
      // Absent branch: the statically known equality bit for the None case(s).
      [&] {
        return py::CastFromPrimOp::create(builder, location, resultType,
                                          absentBit)
            .getResult();
      });
  return Value{result, resultType};
}

Value ModuleEmitter::emitSubscript(const parser::Node &expr) {
  // ⭐ A NamedTuple's literal subscript IS the field at that position: the
  // instance is a tuple whose members are the declared fields, and the index
  // is right here. `p[0]` was "contract 'P' does not provide manifest method
  // '__getitem__'" while `p.x` worked. Only a literal index folds -- a
  // computed one would need a real tuple to index, and the fields have
  // different types anyway.
  if (const parser::Node *indexNode = ast::node(expr, "slice")) {
    // A negative literal is UnaryOp(USub, Constant), not a Constant.
    std::optional<std::int64_t> literalIndex;
    if (indexNode->kind == "Constant")
      literalIndex = ast::integer(*indexNode, "value");
    else if (indexNode->kind == "UnaryOp" &&
             ast::isOperator(ast::node(*indexNode, "op"), "USub"))
      if (const parser::Node *operand = ast::node(*indexNode, "operand");
          operand && operand->kind == "Constant")
        if (std::optional<std::int64_t> magnitude =
                ast::integer(*operand, "value"))
          literalIndex = -*magnitude;
    if (std::optional<std::int64_t> position = literalIndex) {
      const parser::Node *receiverNode = ast::node(expr, "value");
      auto contract = mlir::dyn_cast_if_present<py::ContractType>(
          types.widenLiteral(types.inferExpr(receiverNode)));
      if (contract && namedTupleContracts.count(contract.getContractName())) {
        llvm::ArrayRef<std::string> order =
            classFieldOrders[contract.getContractName()];
        std::int64_t at = *position < 0
                              ? *position + static_cast<std::int64_t>(order.size())
                              : *position;
        if (at >= 0 && at < static_cast<std::int64_t>(order.size())) {
          const parser::Field *receiverField = parser::findField(expr, "value");
          if (receiverField &&
              std::holds_alternative<parser::NodePtr>(receiverField->value)) {
            parser::NodePtr attribute = synth::attribute(std::get<parser::NodePtr>(receiverField->value), order[at], expr.range);
            return emitExpr(attribute.get());
          }
        }
      }
    }
  }
  Value container = emitExpr(ast::node(expr, "value"));
  // Shaped primitives are indexed before the slice is emitted: their indices
  // are static shape coordinates, not values a manifest __getitem__ receives.
  if (container.value &&
      mlir::isa<mlir::RankedTensorType>(container.value.getType())) {
    if (std::optional<Value> element = emitPrimitiveTensorGetItem(
            expr, container, ast::node(expr, "slice")))
      return element->value ? *element : emitNone(expr);
  }
  if (const parser::Node *sliceNode = ast::node(expr, "slice");
      sliceNode && sliceNode->kind == "Slice")
    return emitSliceSubscript(expr, container, *sliceNode);
  Value index = emitExpr(ast::node(expr, "slice"));
  if (std::optional<Value> item =
          tryEmitClassDunder(expr, container, "__getitem__", {index}))
    return *item;
  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      container.type, "__getitem__", {index.type});
  if (!requireStaticEvidence(expr, inference))
    return emitNone(expr);
  mlir::Type result = inference ? inference.resultType : types.inferExpr(&expr);
  auto op = py::GetItemOp::create(
      builder, loc(expr), result,
      mlir::FlatSymbolRefAttr::get(&context, "__getitem__"),
      callProtocolFor(inference), container.value, index.value);
  return {op.getResult(), result};
}

Value ModuleEmitter::emitSliceSubscript(const parser::Node &expr,
                                        Value container,
                                        const parser::Node &sliceNode) {
  const parser::Node *lower = ast::node(sliceNode, "lower");
  const parser::Node *upper = ast::node(sliceNode, "upper");
  const parser::Node *step = ast::node(sliceNode, "step");

  auto intConstant = [&](long long value) -> Value {
    std::string text = std::to_string(value);
    mlir::Type type = types.literal(text);
    auto op = py::IntConstantOp::create(builder, loc(expr), type,
                                        builder.getStringAttr(text));
    return {op.getResult(), type};
  };
  // Absent bounds keep placeholder zeros; the mask tells the runtime which
  // ones are real (their defaults depend on the step's sign at runtime).
  Value startValue = lower ? emitExpr(lower) : intConstant(0);
  Value stopValue = upper ? emitExpr(upper) : intConstant(0);
  Value stepValue = step ? emitExpr(step) : intConstant(1);
  long long maskBits = (lower ? 1 : 0) | (upper ? 2 : 0);
  Value maskValue = intConstant(maskBits);

  CallInferenceResult inference = types.inferMethodCallWithEvidence(
      container.type, "__getslice__",
      {startValue.type, stopValue.type, stepValue.type, maskValue.type});
  if (!inference) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "slicing is not supported for this receiver type (str/list/tuple/"
        "bytes provide `__getslice__`)"});
    return emitNone(expr);
  }
  if (!requireStaticEvidence(expr, inference))
    return emitNone(expr);
  Value posPack =
      emitPack({startValue, stopValue, stepValue, maskValue});
  Value namePack = emitPack({});
  Value valuePack = emitPack({});
  mlir::Type resultType = inference.resultType;
  auto op = py::CallOp::create(builder, loc(expr), mlir::TypeRange{resultType},
                               callProtocolFor(inference), container.value,
                               posPack.value, namePack.value, valuePack.value);
  op->setAttr("ly.bound_method", builder.getStringAttr("__getslice__"));
  return {op.getResults().front(), resultType};
}

Value ModuleEmitter::emitMethodObject(const parser::Node &anchor, Value object,
                                      const MethodBinding &methodBinding) {
  if (methodBinding.symbolName.empty())
    return emitNone(anchor);

  bool bindReceiver = methodBindingBindsReceiver(methodBinding);
  if (methodBinding.kind == "instance" && mlir::isa<py::TypeType>(object.type))
    bindReceiver = false;
  if (!bindReceiver)
    return emitFunctionObject(anchor, methodBinding.symbolName,
                              methodBinding.signature.publicCallable, {});

  if (!methodBinding.method ||
      methodBinding.signature.positionalTypes.empty() ||
      methodBinding.signature.positionalNames.empty() ||
      methodBinding.bodySignature.positionalTypes.empty() ||
      methodBinding.bodySignature.positionalNames.empty())
    return emitNone(anchor);

  auto bindSignature = [&](FunctionSignature sig) {
    sig.positionalTypes.erase(sig.positionalTypes.begin());
    sig.positionalNames.erase(sig.positionalNames.begin());
    if (!sig.positionalDefaults.empty())
      sig.positionalDefaults.erase(sig.positionalDefaults.begin());
    if (sig.positionalOnlyCount > 0)
      --sig.positionalOnlyCount;
    types.refreshCallable(sig);
    return sig;
  };

  FunctionSignature boundPublicSig = bindSignature(methodBinding.signature);
  FunctionSignature boundBodySig = bindSignature(methodBinding.bodySignature);

  llvm::SmallVector<Capture, 1> captures;
  mlir::Type preboundTypeObject;
  if (methodBinding.kind == "class" || methodBinding.kind == "classmethod") {
    preboundTypeObject = object.type;
    if (auto typeObject = mlir::dyn_cast<py::TypeType>(object.type))
      preboundTypeObject = typeObject.getInstanceType();
  } else {
    Value descriptorReceiver =
        emitDescriptorReceiver(anchor, object, methodBinding);
    captures.push_back(
        Capture{methodBinding.bodySignature.positionalNames.front(),
                descriptorReceiver});
  }
  std::string symbolName = (llvm::Twine(methodBinding.symbolName) + "$bound$" +
                            llvm::Twine(++syntheticFunctionCounter) + "$" +
                            llvm::Twine(anchor.range.start.line) + "_" +
                            llvm::Twine(anchor.range.start.column))
                               .str();
  bool pushedSuperContext = methodBinding.kind == "instance" &&
                            !methodBinding.definingClass.empty() &&
                            !methodBinding.bodySignature.positionalNames.empty();
  if (pushedSuperContext)
    superContexts.push_back(
        {methodBinding.definingClass,
         methodBinding.bodySignature.positionalNames.front()});
  emitCallableFunction(*methodBinding.method, symbolName, boundBodySig,
                       captures, /*isLambda=*/false,
                       /*positionalNodeOffset=*/1, preboundTypeObject);
  if (pushedSuperContext)
    superContexts.pop_back();
  return emitFunctionObject(anchor, symbolName, boundPublicSig.publicCallable,
                            captures);
}

// ⭐ `type(e).__name__` WHERE THE STATIC CLASS IS NOT THE ANSWER. In an except
// handler the static class is the one CAUGHT and CPython prints the one RAISED,
// so the fold `type(x)` uses is unavailable -- and refusing the whole idiom left
// the commonest use of type() with no spelling at all. An exception instance
// carries its dynamic class id in its header (it is what the traceback and the
// repr already read), so this one case has a runtime answer.
//
// ⛔ Intercepted BEFORE the receiver is emitted, because `type(...)` on a
// subclassed static class is refused by tryEmitTypeCall and the refusal would
// happen first.
std::optional<Value>
ModuleEmitter::tryEmitDynamicClassName(const parser::Node &expr) {
  const parser::Node *valueNode = ast::node(expr, "value");
  if (!valueNode)
    return std::nullopt;
  // ⭐ `x.__class__` IS `type(x)` -- CPython's two spellings of one question --
  // so both reach here, and the one written as an attribute is not a field
  // lookup ("class C has no field '__class__'").
  std::vector<parser::NodePtr> classAttrArgument;
  const std::vector<parser::NodePtr> *args = nullptr;
  if (valueNode->kind == "Attribute") {
    std::optional<std::string_view> attribute = ast::string(*valueNode, "attr");
    if (!attribute || *attribute != "__class__")
      return std::nullopt;
    if (const parser::Field *field = parser::findField(*valueNode, "value"))
      if (const auto *held = std::get_if<parser::NodePtr>(&field->value);
          held && *held)
        classAttrArgument.push_back(*held);
    if (classAttrArgument.empty())
      return std::nullopt;
    args = &classAttrArgument;
  } else if (valueNode->kind == "Call") {
    const parser::Node *callee = ast::node(*valueNode, "func");
    if (!callee || callee->kind != "Name" ||
        ast::nameSpelling(*callee) != "type" || programBindsName("type"))
      return std::nullopt;
    args = ast::nodeList(*valueNode, "args");
    const auto *keywords = ast::nodeList(*valueNode, "keywords");
    if (!args || args->size() != 1 || !args->front() ||
        args->front()->kind == "Starred" || (keywords && !keywords->empty()))
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  if (!args || args->size() != 1 || !args->front())
    return std::nullopt;
  mlir::Type subject = types.widenLiteral(types.inferExpr(args->front().get()));
  // ⭐ A UNION KNOWS ITS MEMBERS, and the value carries which one it is -- so
  // `type(v).__name__` over `int | str` is the member's name selected by the
  // same tag test `isinstance` uses. It was refused as "needs a statically
  // resolved class", which is true of the union and false of the value.
  //
  // ⛔ Written as the conditional expression a reader would have written, so
  // the tag tests, the narrowing and the string constants all come from paths
  // that already exist. That is also why the subject must be a NAME: the chain
  // mentions it once per member, and re-evaluating a call would run it N times.
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(subject)) {
    const parser::Node *subjectNode = args->front().get();
    if (subjectNode->kind != "Name")
      return std::nullopt;
    llvm::SmallVector<std::pair<mlir::Type, std::string>, 4> members;
    for (mlir::Type rawMember : unionType.getMemberTypes()) {
      mlir::Type member = types.widenLiteral(rawMember);
      if (isNoneTypeLike(member)) {
        members.push_back({member, "NoneType"});
        continue;
      }
      auto memberContract = mlir::dyn_cast_if_present<py::ContractType>(member);
      if (!memberContract)
        return std::nullopt;
      llvm::StringRef qualified = memberContract.getContractName();
      // A member with a subclass would answer the static name, which is the
      // same wrong answer tryEmitTypeCall refuses for a bare value.
      for (const auto &entry : classMros)
        if (entry.getKey() != qualified &&
            llvm::is_contained(entry.second, qualified))
          return std::nullopt;
      llvm::StringRef simple = qualified;
      if (auto dot = qualified.rfind('.'); dot != llvm::StringRef::npos)
        simple = qualified.drop_front(dot + 1);
      members.push_back({member, simple.str()});
    }
    if (members.size() < 2)
      return std::nullopt;
    parser::SourceRange range = expr.range;
    parser::NodePtr chain =
        synth::strConstant(members.back().second, range);
    for (auto it = members.rbegin() + 1; it != members.rend(); ++it) {
      parser::NodePtr test;
      if (isNoneTypeLike(it->first)) {
        test = synth::compare(synth::name(std::string(ast::nameSpelling(*subjectNode)),
                                          range),
                              "Is", synth::noneConstant(range), range);
      } else {
        auto memberContract = mlir::cast<py::ContractType>(it->first);
        llvm::StringRef qualified = memberContract.getContractName();
        llvm::StringRef simple = qualified;
        if (auto dot = qualified.rfind('.'); dot != llvm::StringRef::npos)
          simple = qualified.drop_front(dot + 1);
        test = synth::call(
            synth::name(std::string("isinstance"), range),
            std::vector<parser::NodePtr>{
                synth::name(std::string(ast::nameSpelling(*subjectNode)), range),
                synth::name(simple.str(), range)},
            range);
      }
      chain = synth::ifExp(std::move(test),
                           synth::strConstant(it->second, range),
                           std::move(chain), range);
    }
    synthesizedIteratorDefs.push_back(chain);
    return emitExpr(chain.get());
  }
  auto contract = mlir::dyn_cast_if_present<py::ContractType>(subject);
  if (!contract)
    return std::nullopt;
  // ⛔ A SOURCE CLASS TOO, not only an exception: its instances carry the same
  // class-id word, and this is the only answer available when the static class
  // has subclasses -- which is exactly when tryEmitTypeCall refuses to fold.
  // A manifest contract keeps the fold: `type(5)` is int by construction, and
  // an int's header word 1 is not a class id.
  if (!isExceptionContractType(subject) &&
      !isExceptionBackedClass(contract.getContractName()) &&
      !classMros.contains(contract.getContractName()))
    return std::nullopt;
  Value receiver = emitExpr(args->front().get());
  mlir::Type strType = types.contract("builtins.str");
  mlir::Type calleeContract =
      py::CallableType::get(&context, {subject}, {}, {}, {}, {strType});
  auto op = py::ClassNameOp::create(
      builder, loc(expr), strType,
      mlir::FlatSymbolRefAttr::get(&context, "__class_name__"),
      mlir::TypeAttr::get(calleeContract), receiver.value);
  return Value{op.getResult(), strType};
}

Value ModuleEmitter::emitAttribute(const parser::Node &expr) {
  if (std::optional<std::string_view> dynamicName = ast::string(expr, "attr");
      dynamicName && *dynamicName == "__name__")
    if (std::optional<Value> answered = tryEmitDynamicClassName(expr))
      return *answered;
  // `x.__class__` is `type(x)`, so it takes the same road: the fold when the
  // static class is exact, and the same refusal when it is not. Reaching the
  // field lookup instead reported "class C has no field '__class__'", which is
  // true of the storage and beside the point.
  if (std::optional<std::string_view> attribute = ast::string(expr, "attr");
      attribute && *attribute == "__class__" && !programBindsName("type"))
    if (const parser::Field *field = parser::findField(expr, "value"))
      if (const auto *subject = std::get_if<parser::NodePtr>(&field->value);
          subject && *subject) {
        parser::NodePtr call =
            synth::call(synth::name(std::string("type"), expr.range),
                        std::vector<parser::NodePtr>{*subject}, expr.range);
        synthesizedIteratorDefs.push_back(call);
        return emitExpr(call.get());
      }
  Value object = emitExpr(ast::node(expr, "value"));
  auto attr = ast::string(expr, "attr");
  if (!attr)
    return emitNone(expr);
  // ⭐ `C.__name__` IS A COMPILE-TIME STRING. It was "attr.get type object has
  // no static runtime attribute '__name__'", for a user class and for `int`
  // alike, and the name is the one thing a type object cannot fail to know.
  //
  // ⛔ The last dotted component, not the contract name: the contract is
  // `builtins.int` and CPython's answer is `int`. A user class has no dot and is
  // its own answer.
  //
  // ⛔ Folded here rather than given a runtime attribute, because there is no
  // type-object surface to hang it on -- `print(int)`, `int is int` and
  // `C.__class__` are all still refused, and this fold deliberately does not
  // pretend otherwise.
  if (*attr == "__name__")
    if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(object.type))
      if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
              typeObject.getInstanceType())) {
        llvm::StringRef qualified = contract.getContractName();
        llvm::StringRef simple = qualified;
        if (auto dot = qualified.rfind('.'); dot != llvm::StringRef::npos)
          simple = qualified.drop_front(dot + 1);
        mlir::Type type = types.literal("\"" + simple.str() + "\"");
        auto constant = py::StrConstantOp::create(
            builder, loc(expr), type, builder.getStringAttr(simple));
        return Value{constant.getResult(), type};
      }
  // Property reads inline the getter (before general attribute inference,
  // which knows nothing about accessor bindings).
  if (std::optional<MethodBinding> property =
          lookupClassMethod(object.type, *attr);
      property && property->kind == "property") {
    // A getter is a method, and reading `a.v` through a base-typed `a` is the
    // same unresolvable dispatch as calling `a.v()` would be.
    if (refuseUnresolvableDispatch(expr, object, *attr,
                                   ast::node(expr, "value")))
      return emitNone(expr);
    if (mlir::isa<py::TypeType>(object.type)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "accessing a property object through the class is not supported"});
      return emitNone(expr);
    }
    return emitInlineMethodBody(expr, object, /*bindDescriptorReceiver=*/true,
                                *property, {}, {});
  }
  mlir::Type result = types.inferExpr(&expr);
  std::optional<mlir::Type> field = lookupClassField(object.type, *attr);
  if (field)
    result = *field;

  // Mutable class attributes read from the defining class's global cell
  // (instance receivers too, unless an instance field shadows the name).
  if (!field) {
    mlir::Type receiverInstance = object.type;
    if (auto typeObject =
            mlir::dyn_cast_if_present<py::TypeType>(receiverInstance))
      receiverInstance = typeObject.getInstanceType();
    if (auto contract =
            mlir::dyn_cast_if_present<py::ContractType>(receiverInstance)) {
      // ⭐ A class attribute a subclass redeclares is read from the DEFINING
      // class's cell, so a base-typed receiver got the base's value:
      //
      //     class A: kind: int = 1
      //     class B(A): kind: int = 2
      //     x: A = B()
      //     print(x.kind)      # printed 1; CPython prints 2
      //
      // The same unresolvable dispatch the method gate refuses, reached
      // through a binding instead of a call, so it goes through the same gate
      // -- including its exemptions for a constructed receiver and for `self`
      // inside a standalone method body.
      if (subclassShadowsAttribute(contract.getContractName(), *attr) &&
          refuseUnresolvableDispatch(expr, object, *attr,
                                     ast::node(expr, "value")))
        return emitNone(expr);
      if (std::optional<std::pair<llvm::StringRef, mlir::Type>> slot =
              resolveClassAttrSlot(contract.getContractName(), *attr)) {
        std::string cellName =
            (llvm::Twine(slot->first) + "." + *attr).str();
        auto op = py::GlobalGetOp::create(builder, loc(expr), slot->second,
                                          builder.getStringAttr(cellName));
        return {op.getResult(), slot->second};
      }
    }
  }

  // An instance field shadows the class attribute of the same name.
  std::optional<mlir::Type> staticAttr =
      field ? std::nullopt : lookupClassStaticAttr(object.type, *attr);
  if (staticAttr)
    result = *staticAttr;

  std::optional<MethodBinding> methodBinding =
      lookupClassMethod(object.type, *attr);
  if (methodBinding && !methodBinding->symbolName.empty())
    return emitMethodObject(expr, object, *methodBinding);

  auto op =
      py::AttrGetOp::create(builder, loc(expr), result, object.value, *attr);
  if (field)
    op->setAttr("ly.attr.kind", builder.getStringAttr("field"));
  else if (staticAttr)
    op->setAttr("ly.attr.kind", builder.getStringAttr("static"));
  else if (methodBinding)
    op->setAttr("ly.attr.kind", builder.getStringAttr(methodBinding->kind));
  if (auto typeObject = mlir::dyn_cast_if_present<py::TypeType>(object.type)) {
    if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(
            typeObject.getInstanceType()))
      op->setAttr("ly.attr.owner",
                  builder.getStringAttr(contract.getContractName()));
  } else if (auto contract =
                 mlir::dyn_cast_if_present<py::ContractType>(object.type)) {
    op->setAttr("ly.attr.owner",
                builder.getStringAttr(contract.getContractName()));
  }
  Value value{op.getResult(), result};
  if (methodBinding)
    value.boundMethod = std::make_shared<BoundMethodValue>(
        BoundMethodValue{object, *methodBinding});
  return value;
}

Value ModuleEmitter::emitAwait(const parser::Node &expr) {
  Value awaitable = emitExpr(ast::node(expr, "value"));
  return emitAwaitValue(expr, awaitable);
}

// `asyncio.run(coro)` drives the coroutine to completion. The accepted subset
// executes awaited chains eagerly (see the top-level-await dispatch), so it
// desugars to awaiting the argument; await inference then types the result
// from the coroutine's evidence instead of the manifest contract's Any.
Value ModuleEmitter::emitAsyncioRunCall(const parser::Node &expr) {
  const auto *args = ast::nodeList(expr, "args");
  const auto *keywords = ast::nodeList(expr, "keywords");
  if (!args || args->size() != 1 || !args->front() ||
      (keywords && !keywords->empty())) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "asyncio.run supports exactly one coroutine argument (the debug "
        "keyword is not supported)"});
    return emitNone(expr);
  }
  Value awaitable = emitExpr(args->front().get());
  return emitAwaitValue(expr, awaitable);
}

Value ModuleEmitter::emitAwaitValue(const parser::Node &anchor,
                                    Value awaitable) {
  AwaitInferenceResult inference = types.inferAwaitWithEvidence(awaitable.type);
  return emitAwaitValue(anchor, awaitable, inference);
}

Value ModuleEmitter::emitAwaitValue(const parser::Node &anchor, Value awaitable,
                                    const AwaitInferenceResult &inference) {
  if (!requireStaticEvidence(anchor, inference))
    return emitNone(anchor);

  auto op = py::AwaitOp::create(builder, loc(anchor), inference.resultType,
                                inference.awaitContract, awaitable.value);
  return {op.getResult(), inference.resultType};
}

// `[expr for x in it]` desugars to a runtime-list build loop over the
// structural-mutation machinery:
//   __listcomp<N>: list[T] = []   (T = element expression type)
//   for x in it: __listcomp<N>.append(expr)
// The synthetic For statement shares the original target/iter/element
// subtrees, so iteration typing, scoping diagnostics, and loop-carried
// threading match the handwritten loop exactly. Neither the temp list nor
// the loop target leaks into the enclosing scope.
Value ModuleEmitter::emitListComp(const parser::Node &expr) {
  return emitComprehension(expr, /*isDict=*/false);
}

Value ModuleEmitter::emitDictComp(const parser::Node &expr) {
  return emitComprehension(expr, /*isDict=*/true);
}

Value ModuleEmitter::emitComprehension(const parser::Node &expr,
                                       bool isDict, bool isSet) {
  auto reject = [&](const std::string &message) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, expr.range.start, message});
    return emitNone(expr);
  };
  auto sharedSubtree = [&](llvm::StringRef name) -> parser::NodePtr {
    const parser::Field *field = parser::findField(expr, name);
    if (!field || !std::holds_alternative<parser::NodePtr>(field->value))
      return nullptr;
    return std::get<parser::NodePtr>(field->value);
  };
  const auto *generators = ast::nodeList(expr, "generators");
  if (!generators || generators->empty())
    return reject("malformed comprehension");
  parser::NodePtr elt, keyExpr, valueExpr;
  if (isDict) {
    keyExpr = sharedSubtree("key");
    valueExpr = sharedSubtree("value");
    if (!keyExpr || !valueExpr)
      return reject("malformed dict comprehension");
  } else {
    elt = sharedSubtree("elt");
    if (!elt)
      return reject("malformed list comprehension");
  }

  struct CompGenerator {
    parser::NodePtr target;
    parser::NodePtr iter;
    llvm::SmallVector<parser::NodePtr, 2> filters;
    llvm::SmallVector<llvm::StringRef, 2> targetNames;
    bool tupleTarget = false;
  };
  llvm::SmallVector<CompGenerator, 2> chain;
  for (const parser::NodePtr &generator : *generators) {
    if (!generator)
      return reject("malformed list comprehension");
    if (ast::integer(*generator, "is_async").value_or(0))
      return reject("async comprehensions are not supported");
    const parser::Field *targetField = parser::findField(*generator, "target");
    const parser::Field *iterField = parser::findField(*generator, "iter");
    if (!targetField ||
        !std::holds_alternative<parser::NodePtr>(targetField->value) ||
        !iterField ||
        !std::holds_alternative<parser::NodePtr>(iterField->value))
      return reject("malformed list comprehension generator");
    CompGenerator entry;
    entry.target = std::get<parser::NodePtr>(targetField->value);
    entry.iter = std::get<parser::NodePtr>(iterField->value);
    if (!entry.target || !entry.iter)
      return reject("list comprehension target must be a simple name");
    // `for k in d.keys()` iterates the dict itself: the keys view is a
    // phantom with no runtime representation, and dict iteration IS key
    // iteration. This unlocks the value-position uses that materialize
    // through a synthesized comprehension (sorted(d.keys()), list(d.keys())).
    if (entry.iter->kind == "Call") {
      const parser::Node *viewFunc = ast::node(*entry.iter, "func");
      const auto *viewArgs = ast::nodeList(*entry.iter, "args");
      if (viewFunc && viewFunc->kind == "Attribute" &&
          ast::string(*viewFunc, "attr").value_or("") == "keys" &&
          (!viewArgs || viewArgs->empty())) {
        if (const parser::Field *receiverField =
                parser::findField(*viewFunc, "value"))
          if (std::holds_alternative<parser::NodePtr>(receiverField->value)) {
            parser::NodePtr receiver =
                std::get<parser::NodePtr>(receiverField->value);
            auto receiverContract =
                mlir::dyn_cast_if_present<py::ContractType>(
                    types.widenLiteral(types.inferExpr(receiver.get())));
            if (receiver && receiverContract &&
                receiverContract.getContractName() == "builtins.dict")
              entry.iter = receiver;
          }
      }
    }
    if (entry.target->kind == "Name") {
      entry.targetNames.push_back(ast::nameSpelling(*entry.target));
    } else if (entry.target->kind == "Tuple") {
      // `for k, v in ...`: names only (nested unpack stays rejected).
      entry.tupleTarget = true;
      const auto *elts = ast::nodeList(*entry.target, "elts");
      if (!elts || elts->empty())
        return reject("malformed comprehension tuple target");
      for (const parser::NodePtr &element : *elts) {
        if (!element || element->kind != "Name")
          return reject(
              "comprehension tuple targets must unpack to simple names");
        entry.targetNames.push_back(ast::nameSpelling(*element));
      }
    } else {
      return reject("list comprehension target must be a simple name");
    }
    if (const auto *ifs = ast::nodeList(*generator, "ifs"))
      entry.filters.append(ifs->begin(), ifs->end());
    chain.push_back(std::move(entry));
  }

  // Element type: bind each generator's target to its iteration element type
  // ⭐ The FIRST iterable may be a call whose type only the emission knows:
  // the lazy builtin iterators (zip/enumerate/map/filter/reversed) synthesize
  // their generator when emitted, so `zip(a, b)` is `builtins.object` to
  // inferExpr. Evaluating it into a scratch binding first is what the
  // programmer's own workaround looks like -- `z = zip(a, b); [p for p in z]`
  // worked while `[p for p in zip(a, b)]` was "builtins.object does not
  // provide __iter__". Only the first: a later iterable may reference an
  // earlier target, so it cannot be hoisted out of the loop that binds it.
  std::string hoistedSource;
  std::optional<Value> hoistedPrior;
  auto restoreHoisted = llvm::make_scope_exit([&] {
    if (hoistedSource.empty())
      return;
    if (hoistedPrior)
      values[hoistedSource] = *hoistedPrior;
    else
      values.erase(hoistedSource);
  });
  if (!chain.empty() && chain.front().iter &&
      chain.front().iter->kind == "Call" &&
      !types.inferMethodCallWithEvidence(
          types.widenLiteral(types.inferExpr(chain.front().iter.get())),
          "__iter__", {})) {
    std::string name = "__lycompsrc" + std::to_string(++listCompCounter);
    parser::NodePtr target = synth::name(name, expr.range);
    parser::NodePtr assign = synth::assign(target, chain.front().iter, expr.range);
    if (auto found = values.find(name); found != values.end())
      hoistedPrior = found->second;
    emitStatement(*assign);
    if (values.find(name) != values.end()) {
      hoistedSource = name;
      parser::NodePtr bound = synth::name(name, expr.range);
      chain.front().iter = std::move(bound);
    }
  }

  // (later iterables may reference earlier targets), then infer the element
  // expression under those bindings.
  mlir::Type elementType, keyType, valueType;
  {
    auto scope = types.pushScope();
    for (const CompGenerator &entry : chain) {
      mlir::Type iterableType = types.inferExpr(entry.iter.get());
      CallInferenceResult iterInference =
          types.inferMethodCallWithEvidence(iterableType, "__iter__", {});
      // ⭐ THE SEQUENCE PROTOCOL, the same fallback the for statement takes:
      // `__len__` + `__getitem__` with no `__iter__` is iterable, and the
      // element is what the subscript answers. `types.iterationElementType`
      // knows this rule; this walk asks the two questions itself because it
      // needs the iterator type as well, so it needs the fallback too.
      mlir::Type iterationElement;
      if (!iterInference) {
        if (types.inferMethodCallWithEvidence(types.widenLiteral(iterableType),
                                              "__len__", {}))
          if (CallInferenceResult indexed = types.inferMethodCallWithEvidence(
                  types.widenLiteral(iterableType), "__getitem__",
                  {types.intType()}))
            iterationElement = types.widenLiteral(indexed.resultType);
      }
      if (!iterationElement) {
        if (!requireStaticEvidence(expr, iterInference))
          return emitNone(expr);
        CallInferenceResult nextInference = types.inferMethodCallWithEvidence(
            iterInference.resultType, "__next__", {});
        if (!requireStaticEvidence(expr, nextInference))
          return emitNone(expr);
        iterationElement = types.widenLiteral(nextInference.resultType);
      }
      if (!entry.tupleTarget) {
        types.bindLocalSymbol(entry.targetNames.front(), iterationElement);
      } else {
        for (auto [position, name] : llvm::enumerate(entry.targetNames)) {
          CallInferenceResult itemInference =
              types.inferMethodCallWithEvidence(
                  iterationElement, "__getitem__",
                  {types.literal(std::to_string(position))});
          if (!itemInference)
            return reject(
                "cannot infer the comprehension tuple target element types");
          types.bindLocalSymbol(name,
                                types.widenLiteral(itemInference.resultType));
        }
      }
    }
    if (isDict) {
      keyType = types.widenLiteral(types.inferExpr(keyExpr.get()));
      valueType = types.widenLiteral(types.inferExpr(valueExpr.get()));
    } else {
      elementType = types.widenLiteral(types.inferExpr(elt.get()));
    }
  }
  if (isDict ? (!keyType || !valueType) : !elementType)
    return reject("cannot infer the comprehension element type");

  // Temp result container, bound as a local so the build loop threads it.
  std::string tmp =
      (isDict ? "__dictcomp" : (isSet ? "__setcomp" : "__listcomp")) +
      std::to_string(++listCompCounter);
  mlir::Type containerType =
      isDict ? py::ContractType::get(builder.getContext(), "builtins.dict",
                                     {keyType, valueType})
             : py::ContractType::get(builder.getContext(),
                                     isSet ? "builtins.set" : "builtins.list",
                                     {elementType});
  auto pack =
      py::PackOp::create(builder, loc(expr), containerType, mlir::ValueRange{});
  values[tmp] = Value{pack.getResult(), containerType};

  // list: for <target> in <iter>: <tmp>.append(<elt>)
  // dict: for <target> in <iter>: <tmp>[<key>] = <value>
  parser::NodePtr tmpName = synth::name(tmp, expr.range);
  parser::NodePtr statement;
  if (isDict) {
    parser::NodePtr subscript = parser::makeNode("Subscript", expr.range);
    parser::addField(*subscript, "value", tmpName);
    parser::addField(*subscript, "slice", keyExpr);
    statement = parser::makeNode("Assign", expr.range);
    parser::addField(*statement, "targets",
                     std::vector<parser::NodePtr>{subscript});
    parser::addField(*statement, "value", valueExpr);
  } else {
    parser::NodePtr appendAttr = synth::attribute(tmpName, std::string(isSet ? "add" : "append"), expr.range);
    parser::NodePtr appendCall = synth::call(appendAttr, std::vector<parser::NodePtr>{elt}, expr.range);
    statement = parser::makeNode("Expr", expr.range);
    parser::addField(*statement, "value", appendCall);
  }
  // Build inside-out: each generator's filters wrap the current statement
  // (`if c1: if c2: ...`), then its For wraps that.
  for (const CompGenerator &entry : llvm::reverse(chain)) {
    for (const parser::NodePtr &filter : llvm::reverse(entry.filters)) {
      parser::NodePtr guard = parser::makeNode("If", expr.range);
      parser::addField(*guard, "test", filter);
      parser::addField(*guard, "body",
                       std::vector<parser::NodePtr>{statement});
      parser::addField(*guard, "orelse", std::vector<parser::NodePtr>{});
      statement = guard;
    }
    parser::NodePtr loop = parser::makeNode("For", expr.range);
    parser::addField(*loop, "target", entry.target);
    parser::addField(*loop, "iter", entry.iter);
    parser::addField(*loop, "body", std::vector<parser::NodePtr>{statement});
    parser::addField(*loop, "orelse", std::vector<parser::NodePtr>{});
    statement = loop;
  }

  llvm::SmallVector<std::pair<llvm::StringRef, std::optional<Value>>, 2>
      priorTargets;
  for (const CompGenerator &entry : chain)
    for (llvm::StringRef name : entry.targetNames) {
      std::optional<Value> prior;
      if (auto found = values.find(name); found != values.end())
        prior = found->second;
      priorTargets.push_back({name, prior});
    }
  emitFor(*statement);

  auto built = values.find(tmp);
  Value result = built != values.end()
                     ? built->second
                     : Value{pack.getResult(), containerType};
  values.erase(tmp);
  for (auto &[name, prior] : priorTargets) {
    if (prior)
      values[name] = *prior;
    else
      values.erase(name);
  }
  return result;
}

// `{e0, e1, ...}` desugars to the set-build the SetComp lowering already
// uses — an empty set pack plus one `add` per element — because a direct
// element pack would bypass `add`'s deduplication (`{1, 1}` must have one
// element).
Value ModuleEmitter::emitSetLiteral(const parser::Node &expr,
                                    mlir::Type expected) {
  auto reject = [&](const std::string &message) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, expr.range.start, message});
    return emitNone(expr);
  };
  const auto *elts = ast::nodeList(expr, "elts");
  if (!elts || elts->empty())
    return reject("malformed set literal");
  for (const parser::NodePtr &element : *elts)
    if (!element || element->kind == "Starred")
      return reject("starred elements in a set literal are not supported yet");

  mlir::Type elementType;
  if (auto expectedContract =
          mlir::dyn_cast_if_present<py::ContractType>(expected))
    if (expectedContract.getContractName() == "builtins.set" &&
        expectedContract.getArguments().size() == 1)
      elementType = expectedContract.getArguments().front();
  if (!elementType) {
    llvm::SmallVector<mlir::Type, 8> parts;
    for (const parser::NodePtr &element : *elts)
      parts.push_back(types.widenLiteral(types.inferExpr(element.get())));
    elementType = types.join(parts);
  }
  if (!elementType || containsObjectTop(elementType, types))
    return reject("cannot infer the set literal element type");

  std::string tmp = "__setlit" + std::to_string(++listCompCounter);
  mlir::Type containerType =
      py::ContractType::get(builder.getContext(), "builtins.set",
                            {elementType});
  auto pack =
      py::PackOp::create(builder, loc(expr), containerType, mlir::ValueRange{});
  std::optional<Value> priorBinding;
  if (auto found = values.find(tmp); found != values.end())
    priorBinding = found->second;
  values[tmp] = Value{pack.getResult(), containerType};

  for (const parser::NodePtr &element : *elts) {
    parser::NodePtr tmpName = synth::name(tmp, expr.range);
    parser::NodePtr addAttr = synth::attribute(tmpName, std::string("add"), expr.range);
    parser::NodePtr addCall = synth::call(addAttr, std::vector<parser::NodePtr>{element}, expr.range);
    parser::NodePtr statement = synth::exprStmt(addCall, expr.range);
    emitStatement(*statement);
  }

  auto built = values.find(tmp);
  Value result = built != values.end()
                     ? built->second
                     : Value{pack.getResult(), containerType};
  if (priorBinding)
    values[tmp] = *priorBinding;
  else
    values.erase(tmp);
  return result;
}

// The value-returning `and`/`or` (R1): the deciding operand's VALUE flows
// out. Each non-final operand contributes its decided-side type (`or` keeps a
// truthy operand, so an Optional contributes its present member; `and` keeps
// a falsy operand, so a container-or-bool-membered Optional contributes the
// whole union while an always-truthy member narrows to None); the final
// operand flows through unchanged. The join of those contributions must be
// statically representable, otherwise the combination is rejected.
Value ModuleEmitter::emitBoolOpValue(const parser::Node &expr, bool isAnd,
                                     const std::vector<parser::NodePtr> &operands) {
  auto reject = [&](const std::string &message) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, expr.range.start, message});
    return emitNone(expr);
  };
  auto singleMemberOptional = [&](mlir::Type type) -> mlir::Type {
    auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type);
    if (!unionType || !unionType.hasMember(types.none()))
      return nullptr;
    mlir::Type member;
    for (mlir::Type candidate : unionType.getMemberTypes()) {
      if (isNoneTypeLike(candidate))
        continue;
      if (member)
        return nullptr;
      member = candidate;
    }
    return member;
  };
  auto canBeFalsyWhenPresent = [&](mlir::Type member) {
    mlir::Type widened = types.widenLiteral(member);
    if (widened == types.boolType())
      return true;
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(widened);
    if (!contract)
      return true; // conservative: keep the whole union in the falsy arm
    llvm::StringRef name = contract.getContractName();
    return name == "builtins.list" || name == "builtins.dict" ||
           name == "builtins.set" || name == "builtins.tuple" ||
           name == "builtins.str" || name == "builtins.bytes" ||
           name == "builtins.int" || name == "builtins.float";
  };

  llvm::SmallVector<mlir::Type, 4> parts;
  for (auto [index, operand] : llvm::enumerate(operands)) {
    if (!operand)
      return reject("malformed boolean operation");
    mlir::Type operandType =
        types.widenLiteral(types.inferExpr(operand.get()));
    if (index + 1 == operands.size()) {
      parts.push_back(operandType);
      break;
    }
    mlir::Type member = singleMemberOptional(operandType);
    if (!isAnd) {
      // `or` keeps a TRUTHY non-final operand: an Optional's kept value is
      // its present member.
      parts.push_back(member ? types.widenLiteral(member) : operandType);
    } else if (member && !canBeFalsyWhenPresent(member)) {
      // `and` keeps a FALSY non-final operand: with an always-truthy member
      // the only falsy value is None.
      parts.push_back(types.none());
    } else {
      parts.push_back(operandType);
    }
  }
  mlir::Type resultType = types.join(parts);
  if (!resultType || containsObjectTop(resultType, types))
    return reject(
        "`and`/`or` over these operand types has no statically representable "
        "result type; wrap the operands in explicit comparisons or a "
        "conditional expression");

  // Flat N-way merge: every deciding value reaches the merge through exactly
  // one block-argument hop (nesting diamonds per operand builds multi-hop
  // merge chains the affine-ownership planner cannot balance).
  mlir::Location location = loc(expr);
  mlir::Block *origin = builder.getInsertionBlock();
  mlir::Region *region = origin->getParent();
  mlir::Block *merge =
      builder.createBlock(region, std::next(origin->getIterator()));
  mlir::BlockArgument result = merge->addArgument(resultType, location);
  builder.setInsertionPointToEnd(origin);

  for (unsigned index = 0, count = operands.size(); index < count; ++index) {
    Value current = emitExpr(operands[index].get());
    if (index + 1 == count) {
      mlir::Value last = coerceValue(current, resultType, expr).value;
      mlir::cf::BranchOp::create(builder, location, merge,
                                 mlir::ValueRange{last});
      break;
    }
    mlir::Value condition = emitBoolValue(current, expr);
    mlir::Type member =
        singleMemberOptional(types.widenLiteral(current.type));
    // The truth test may open blocks of its own (the Optional diamond): the
    // deciding branch leaves from wherever the test's emission landed.
    mlir::Block *decide = builder.getInsertionBlock();
    mlir::Block *keepBlock = builder.createBlock(region, merge->getIterator());
    mlir::Block *nextBlock = builder.createBlock(region, merge->getIterator());
    builder.setInsertionPointToEnd(decide);
    if (isAnd)
      mlir::cf::CondBranchOp::create(builder, location, condition, nextBlock,
                                     mlir::ValueRange{}, keepBlock,
                                     mlir::ValueRange{});
    else
      mlir::cf::CondBranchOp::create(builder, location, condition, keepBlock,
                                     mlir::ValueRange{}, nextBlock,
                                     mlir::ValueRange{});

    builder.setInsertionPointToStart(keepBlock);
    mlir::Value kept;
    if (!isAnd && member) {
      // Truthy implies not-None: project the member before joining.
      auto unwrap = py::UnionUnwrapOp::create(builder, location, member,
                                              current.value);
      kept = coerceValue(Value{unwrap.getResult(), member}, resultType, expr)
                 .value;
    } else if (isAnd && member && !canBeFalsyWhenPresent(member)) {
      // Falsy with an always-truthy member means the value IS None.
      kept = coerceValue(emitNone(expr), resultType, expr).value;
    } else {
      kept = coerceValue(current, resultType, expr).value;
    }
    mlir::cf::BranchOp::create(builder, location, merge,
                               mlir::ValueRange{kept});

    builder.setInsertionPointToStart(nextBlock);
  }

  builder.setInsertionPointToStart(merge);
  return Value{result, resultType};
}

Value ModuleEmitter::emitExprExpected(const parser::Node *expr,
                                      mlir::Type expected) {
  if (!expr || !expected)
    return emitExpr(expr);
  if (expr->kind == "Lambda")
    if (auto expectedCallable =
            mlir::dyn_cast_if_present<py::CallableType>(expected))
      return emitLambda(*expr, expectedCallable);
  // ⭐ A container literal under a UNION expectation takes the member of its
  // own kind. `xs: list[int] | None` then `xs = []` typed the literal
  // `list[object]` (the union is not a container contract, so the
  // expectation was dropped) and the branch join became
  // `list[int] | list[object] | None`, which nothing accepts -- while the
  // same assignment under the bare `list[int]` expectation was exact.
  if (auto unionExpected = mlir::dyn_cast<py::UnionType>(expected)) {
    llvm::StringRef wanted;
    if (expr->kind == "List")
      wanted = "builtins.list";
    else if (expr->kind == "Tuple")
      wanted = "builtins.tuple";
    else if (expr->kind == "Dict")
      wanted = "builtins.dict";
    else if (expr->kind == "Set")
      wanted = "builtins.set";
    if (!wanted.empty()) {
      mlir::Type only;
      for (mlir::Type member : unionExpected.getMemberTypes()) {
        auto contract = mlir::dyn_cast<py::ContractType>(member);
        if (!contract || contract.getContractName() != wanted)
          continue;
        if (only) {
          only = mlir::Type();
          break;
        }
        only = member;
      }
      if (only)
        expected = only;
    }
  }
  if (expr->kind == "List" || expr->kind == "Tuple" || expr->kind == "Dict")
    return emitContainerLiteral(*expr, expected);
  if (expr->kind == "Set")
    return emitSetLiteral(*expr, expected);
  // `b: Box[int] = Box(5)`: a bare generic-class construction takes its
  // instantiation from the expectation. The specialization is already
  // allocated (the annotation spelled it), so this only has to redirect the
  // construction at the specialized contract instead of the generic name.
  if (expr->kind == "Call")
    if (mlir::Type specialized =
            expectedGenericClassInstantiation(*expr, expected))
      return emitClassInstantiation(
          *expr,
          mlir::cast<py::ContractType>(specialized).getContractName(),
          specialized);
  GenericFunctionInfo *generic = nullptr;
  if (expr->kind == "Name") {
    llvm::StringRef name = ast::nameSpelling(*expr);
    if (values.find(name) == values.end())
      generic = lookupGenericFunction(name);
  } else if (expr->kind == "Attribute") {
    // Qualified references to imported generics (module.fn) take the same
    // expected-callable instantiation path as bare names.
    std::string qualified = ast::qualifiedName(expr);
    if (!qualified.empty())
      if (std::optional<std::string> canonical =
              types.lookupCanonicalBinding(qualified)) {
        auto found = genericFunctions.find(*canonical);
        if (found != genericFunctions.end())
          generic = &found->second;
      }
  }
  if (generic) {
    // A ground expected callable determines the instantiation, so a
    // first-class reference to a generic function materializes as a
    // reference to the matching specialization.
    auto expectedCallable =
        mlir::dyn_cast_if_present<py::CallableType>(expected);
    if (!expectedCallable || unboundStaticParameterCount(expectedCallable) != 0)
      return emitExpr(expr);
    std::optional<std::pair<std::string, py::CallableType>> specialization =
        ensureGenericSpecialization(*expr, *generic, expectedCallable);
    if (!specialization)
      return emitNone(*expr);
    return emitBindingRef(*expr, specialization->first,
                          specialization->second);
  }
  return emitExpr(expr);
}

// ⭐ An EMPTY container literal inside another literal takes its element type
// from its SIBLINGS. `inferExpr` already ignores it when joining them (see
// `joinIgnoringEmptyLiterals`), so the literal's own inferred type is the
// answer -- but the element still had to be EMITTED at it, or the recorded
// evidence stays `list[object]` while the container says `list[int]`:
//
//     buckets = {"a": [1], "b": []}
//     buckets["b"].append(2)
//     # dict __getitem__ evidence contract 'list[object]' is not assignable
//     # to result 'list[int]'
//
// ⛔ Only for an element that IS an empty container literal, and only when no
// expectation reached it. Handing every element the literal's joined type
// would retype the ones that decided it -- `[1, 2.5]` would emit its `1` as a
// float -- which is a different change and a wrong one.
mlir::Type ModuleEmitter::siblingExpectationFor(const parser::Node &literal,
                                                const parser::Node *element,
                                                bool forKey) {
  if (!element)
    return {};
  bool elementIsEmptyLiteral =
      (element->kind == "List" || element->kind == "Tuple" ||
       element->kind == "Set" || element->kind == "Dict") &&
      [&] {
        const auto *elts = ast::nodeList(*element, "elts");
        const auto *keys = ast::nodeList(*element, "keys");
        return (!elts || elts->empty()) && (!keys || keys->empty());
      }();
  if (!elementIsEmptyLiteral)
    return {};
  auto contract =
      mlir::dyn_cast_if_present<py::ContractType>(types.inferExpr(&literal));
  if (!contract)
    return {};
  llvm::ArrayRef<mlir::Type> arguments = contract.getArguments();
  if (literal.kind == "Dict")
    return arguments.size() == 2 ? arguments[forKey ? 0 : 1] : mlir::Type();
  return arguments.size() == 1 ? arguments.front() : mlir::Type();
}

Value ModuleEmitter::emitContainerLiteral(const parser::Node &expr,
                                          mlir::Type expected) {
  // The expectation only distributes when its container class matches the
  // literal's node kind; a mismatched expectation falls back to synthesis so
  // the caller's contract check reports it at the right place.
  auto expectedContract = mlir::dyn_cast_if_present<py::ContractType>(expected);
  llvm::StringRef expectedName =
      expectedContract ? expectedContract.getContractName() : llvm::StringRef();
  llvm::ArrayRef<mlir::Type> expectedArgs =
      expectedContract ? expectedContract.getArguments()
                       : llvm::ArrayRef<mlir::Type>();
  bool container = false;

  llvm::SmallVector<Value, 8> valuesToPack;
  bool empty = true;
  if (const auto *elts = ast::nodeList(expr, "elts")) {
    mlir::Type elementExpected;
    llvm::ArrayRef<mlir::Type> positionalExpected;
    if (expr.kind == "List" && expectedName == "builtins.list" &&
        expectedArgs.size() == 1) {
      container = true;
      elementExpected = expectedArgs.front();
    } else if (expr.kind == "Tuple" && expectedName == "builtins.tuple") {
      if (expectedArgs.size() == elts->size()) {
        container = true;
        positionalExpected = expectedArgs;
      } else if (expectedArgs.size() == 1) {
        container = true;
        elementExpected = expectedArgs.front();
      }
    }
    for (auto [index, elt] : llvm::enumerate(*elts)) {
      empty = false;
      mlir::Type eltExpected = index < positionalExpected.size()
                                   ? positionalExpected[index]
                                   : elementExpected;
      if (!eltExpected)
        eltExpected = siblingExpectationFor(expr, elt.get(), /*forKey=*/false);
      valuesToPack.push_back(emitExprExpected(elt.get(), eltExpected));
    }
  }
  if (const auto *keys = ast::nodeList(expr, "keys")) {
    mlir::Type keyExpected;
    mlir::Type valueExpected;
    if (expr.kind == "Dict" && expectedName == "builtins.dict" &&
        expectedArgs.size() == 2) {
      container = true;
      keyExpected = expectedArgs[0];
      valueExpected = expectedArgs[1];
    }
    const auto *vals = ast::nodeList(expr, "values");
    for (auto [index, key] : llvm::enumerate(*keys)) {
      empty = false;
      if (key)
        valuesToPack.push_back(emitExprExpected(key.get(), keyExpected));
      if (vals && index < vals->size()) {
        mlir::Type thisValueExpected = valueExpected;
        if (!thisValueExpected)
          thisValueExpected =
              siblingExpectationFor(expr, (*vals)[index].get(),
                                    /*forKey=*/false);
        valuesToPack.push_back(
            emitExprExpected((*vals)[index].get(), thisValueExpected));
      }
    }
  }
  // An empty literal synthesizes its top-erased element type, which the
  // stricter lowering contract match later rejects against a concrete
  // formal; adopting the expectation types the pack correctly from the
  // start. Non-empty literals keep the synthesized type: their elements
  // determine it, and the caller's coercion validates the expectation.
  mlir::Type resultType =
      (empty && container) ? expected : types.inferExpr(&expr);
  llvm::SmallVector<mlir::Value, 8> operands;
  for (Value value : valuesToPack)
    operands.push_back(value.value);
  auto op = py::PackOp::create(builder, loc(expr), resultType, operands);
  return {op.getResult(), resultType};
}

// A literal-typed symbol's value is fully determined by its type, so a
// reference materializes the constant directly instead of requiring a runtime
// binding (used for imported module-level literal constants).
std::optional<Value>
ModuleEmitter::emitLiteralTypeConstant(const parser::Node &anchor,
                                       mlir::Type type) {
  auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type);
  if (!literal)
    return std::nullopt;
  llvm::StringRef spelling = literal.getSpelling();
  if (spelling == "None")
    return emitNone(anchor);
  if (spelling == "True" || spelling == "False") {
    auto op = py::BoolConstantOp::create(builder, loc(anchor), type,
                                         builder.getBoolAttr(spelling == "True"));
    return Value{op.getResult(), type};
  }
  llvm::StringRef digits = spelling;
  if (digits.consume_front("-") ? !digits.empty() : !digits.empty()) {
    bool allDigits = llvm::all_of(digits, [](char c) {
      return c >= '0' && c <= '9';
    });
    if (allDigits) {
      auto op = py::IntConstantOp::create(builder, loc(anchor), type,
                                          builder.getStringAttr(spelling));
      return Value{op.getResult(), type};
    }
  }
  if (spelling.size() >= 2 && spelling.front() == '"' &&
      spelling.back() == '"') {
    auto op = py::StrConstantOp::create(
        builder, loc(anchor), type,
        builder.getStringAttr(spelling.drop_front().drop_back()));
    return Value{op.getResult(), type};
  }
  return std::nullopt;
}

Value ModuleEmitter::emitBindingRef(const parser::Node &anchor,
                                    llvm::StringRef binding, mlir::Type type,
                                    llvm::ArrayRef<Value> captures) {
  // A binding reference materializes a solver-resolved Python object whose
  // later dispatch (call, attribute access) observes its type. Fabricating an
  // `object` top here would let unresolved evidence reach lowering as a
  // dynamic receiver, which the static contract model forbids. A missing type
  // is an internal solver invariant violation: fail explicitly at this static
  // boundary instead of erasing the receiver.
  if (!type) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "internal: binding reference '" + binding.str() +
            "' has no resolved type; refusing to erase to object"});
    return emitNone(anchor);
  }
  // The binding string is a NAME, and the runtime lowering resolves it against
  // the manifest before it looks for a user func.func of the same name. A
  // top-level `def len` is emitted under a renamed symbol precisely so the two
  // stay distinguishable, so every reference to it must name that symbol.
  auto shadowed = shadowedBuiltinSymbols.find(binding);
  if (shadowed != shadowedBuiltinSymbols.end())
    binding = shadowed->second;
  mlir::Type resultType = type;
  llvm::SmallVector<mlir::Value, 4> captureValues;
  for (Value capture : captures)
    captureValues.push_back(capture.value);
  auto op =
      py::BindingRefOp::create(builder, loc(anchor), resultType,
                               builder.getStringAttr(binding), captureValues);
  return {op.getResult(), resultType};
}

std::optional<Value> ModuleEmitter::emitManifestFloatConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  const py::protocols::Table &table =
      py::protocols::Table::get(context);
  std::optional<double> value = table.moduleFloatConstant(binding);
  if (!value)
    return std::nullopt;
  mlir::Type type = types.floatType();
  auto op = py::FloatConstantOp::create(builder, loc(anchor), type,
                                        builder.getF64FloatAttr(*value));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitManifestIntConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  const py::protocols::Table &table =
      py::protocols::Table::get(context);
  std::optional<long long> value = table.moduleIntConstant(binding);
  if (!value)
    return std::nullopt;
  std::string spelling = std::to_string(*value);
  mlir::Type type = types.literal(spelling);
  auto op = py::IntConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(spelling));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitManifestStrConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  const py::protocols::Table &table =
      py::protocols::Table::get(context);
  std::optional<std::string> value = table.moduleStrConstant(binding);
  if (!value)
    return std::nullopt;
  mlir::Type type = types.literal("\"" + *value + "\"");
  auto op = py::StrConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(*value));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitStaticIntConstant(
    const parser::Node &anchor, llvm::StringRef binding) {
  if (!py::platform_constants::isStaticIntBinding(binding))
    return std::nullopt;
  std::optional<long long> value =
      py::platform_constants::staticIntValue(binding, options.targetTriple);
  if (!value) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unsupported target platform for static constant '" +
                               std::string(binding) + "'"});
    return emitNone(anchor);
  }
  std::string spelling = std::to_string(*value);
  mlir::Type type = types.literal(spelling);
  auto op = py::IntConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(spelling));
  return Value{op.getResult(), type};
}

std::optional<Value> ModuleEmitter::emitStaticStringConstant(
    const parser::Node &anchor, llvm::StringRef binding, bool allowCallable) {
  bool isStaticBinding = py::platform_constants::isStaticStringBinding(binding);
  bool isStaticCallable =
      allowCallable && py::platform_constants::isStaticStringCallable(binding);
  if (!isStaticBinding && !isStaticCallable)
    return std::nullopt;
  std::optional<std::string> value =
      py::platform_constants::staticStringValue(binding, options.targetTriple);
  if (!value) {
    diagnostics.push_back(
        parser::Diagnostic{parser::Severity::Error, anchor.range.start,
                           "unsupported target platform for static constant '" +
                               std::string(binding) + "'"});
    return emitNone(anchor);
  }
  mlir::Type type = types.literal("\"" + *value + "\"");
  auto op = py::StrConstantOp::create(builder, loc(anchor), type,
                                      builder.getStringAttr(*value));
  return Value{op.getResult(), type};
}

Value ModuleEmitter::emitFunctionObject(const parser::Node &anchor,
                                        llvm::StringRef symbolName,
                                        mlir::Type type,
                                        llvm::ArrayRef<Capture> captures) {
  llvm::SmallVector<Value, 4> captureValues;
  for (const Capture &capture : captures)
    captureValues.push_back(capture.value);
  return emitBindingRef(anchor, symbolName, type, captureValues);
}

Value ModuleEmitter::emitNone(const parser::Node &anchor) {
  auto op = py::NoneOp::create(builder, loc(anchor), types.none());
  return {op.getResult(), types.none()};
}

Value ModuleEmitter::emitPack(mlir::ArrayRef<Value> valuesIn,
                              llvm::ArrayRef<char> unpacked) {
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<mlir::Type, 8> elementTypes;
  for (Value value : valuesIn) {
    operands.push_back(value.value);
    elementTypes.push_back(value.type);
  }
  // An empty pack has no elements, so no element is ever retrieved and the
  // placeholder element type can never be observed by later dispatch. This is
  // the AGENTS.md-sanctioned empty-container placeholder, not an erased
  // dynamic receiver: a non-empty pack always joins concrete element evidence.
  mlir::Type element =
      elementTypes.empty() ? types.object() : types.join(elementTypes);
  mlir::Type resultType = types.tupleOf(element);
  auto op = py::PackOp::create(builder, builder.getUnknownLoc(), resultType,
                               operands);
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

bool ModuleEmitter::isNumericPrimitiveContract(mlir::Type type) const {
  return type && (type == types.boolType() || type == types.intType() ||
                  type == types.floatType());
}

Value ModuleEmitter::coerceValue(Value value, mlir::Type targetType,
                                 const parser::Node &anchor) {
  if (!targetType || value.type == targetType)
    return value;
  if (auto unionType = mlir::dyn_cast<py::UnionType>(targetType)) {
    if (unionType.hasMember(value.type)) {
      auto op = py::UnionWrapOp::create(builder, loc(anchor), targetType,
                                        value.value);
      return {op.getResult(), targetType};
    }
    // ⭐ A NARROWER UNION INJECTS INTO A WIDER ONE. `hasMember` asks whether
    // the whole source type is one member, and a union never is; the member
    // loop below then asks whether it is assignable TO a member, which it also
    // is not. So the coercion returned the value untouched and the mismatch
    // surfaced where the value was used:
    //
    //     doc = {"id": 1, "name": "x"}
    //     print(doc.get("id"))
    //     # type mismatch for bb argument #0 of successor #0
    //
    // `dict.get` merges the present arm (`int | str`) with a None into
    // `int | str | None`, and only the None arm was wrapped. `py.union.wrap`
    // already performs the injection -- it remaps the source tag member by
    // member -- so this is the emitter asking for what the lowering can do.
    if (auto sourceUnion =
            mlir::dyn_cast_if_present<py::UnionType>(value.type)) {
      if (llvm::all_of(sourceUnion.getMemberTypes(), [&](mlir::Type member) {
            return unionType.hasMember(member);
          })) {
        auto op = py::UnionWrapOp::create(builder, loc(anchor), targetType,
                                          value.value);
        return {op.getResult(), targetType};
      }
    }
    mlir::Type actual = types.widenLiteral(value.type);
    for (mlir::Type member : unionType.getMemberTypes()) {
      if (!isAssignableWithStaticEvidence(actual, member, module))
        continue;
      Value memberValue = coerceValue(value, member, anchor);
      if (memberValue.type != member)
        continue;
      auto op = py::UnionWrapOp::create(builder, loc(anchor), targetType,
                                        memberValue.value);
      return {op.getResult(), targetType};
    }
  }
  if (mlir::isa<py::ProtocolType>(targetType)) {
    auto op = py::ProtocolViewOp::create(builder, loc(anchor), targetType,
                                         value.value);
    return {op.getResult(), targetType};
  }
  // ⛔ Why NOT upcast between the numeric contracts, which is what the branch
  // below did: ClassUpcastOp is a RETYPING, right for Derived -> Base where the
  // object handle is unchanged, and int, float and bool share no
  // representation. `x: float = 3` emitted
  //
  //     %0 = py.int.constant "3" : !py.literal<3>
  //     %1 = py.class.upcast %0 : !py.literal<3> -> !py.contract<"builtins.float">
  //
  // with the int's three lanes still underneath, and the lie surfaced one use
  // later -- at module scope as "module global 'x' assignment value group has 3
  // values, expected 1", and in a function as "cannot adapt builtins.float to
  // runtime input 3 of builtins.int.__add__" the moment x was added to.
  //
  // ⛔ And why NOT convert instead: CPython does not convert at an annotation
  // either, so `x: float = 3; print(x)` prints 3 there. Converting would print
  // 3.0 -- see tests/probe/wb_argument_boundary_numeric_tower.py, where the
  // same measurement rejects it at a parameter boundary. The annotation is a
  // constraint that an int already satisfies, and the mixed int/float
  // arithmetic path carries the value from here.
  if (mlir::Type widened = types.widenLiteral(value.type);
      widened != targetType && isNumericPrimitiveContract(widened) &&
      isNumericPrimitiveContract(targetType))
    return value;
  // ⭐ AND THE SAME LIE ONE LEVEL IN. A container's ELEMENT type is its storage,
  // so retyping `list[int]` to `list[float]` leaves int boxes in float slots:
  //
  //     xs: list[float] = [1]
  //     print(xs[0])                      # printed 5e-324; CPython prints 1
  //     t: tuple[float, float] = (1, 2)
  //     print(t[0] + 0.5)                 # printed 0.5; CPython prints 1.5
  //     def f() -> list[float]: return [1]
  //     print(sum(f()))                   # printed 5e-324; CPython prints 1
  //
  // The upcast is what made those compile. Every shape that printed CPython's
  // answer did so because nothing had decoded an element yet -- `print(t)` gave
  // `(1, 2)` and `t[0] + 0.5` gave 0.5 in the same program -- so declining the
  // retyping gives back no working ground; it turns silent wrong answers into a
  // mismatch the store or the call reports.
  //
  // ⛔ Same-name containers only, argument-wise, and only between two NUMERIC
  // element contracts. `list[Derived]` to `list[Base]` is a real retyping with
  // one representation, and this must not touch it.
  if (auto targetContract = mlir::dyn_cast<py::ContractType>(targetType))
    if (auto valueContract = mlir::dyn_cast_if_present<py::ContractType>(
            types.widenLiteral(value.type));
        valueContract && targetContract.getContractName() ==
                             valueContract.getContractName()) {
      llvm::ArrayRef<mlir::Type> targetArgs = targetContract.getArguments();
      llvm::ArrayRef<mlir::Type> valueArgs = valueContract.getArguments();
      if (!targetArgs.empty() && targetArgs.size() == valueArgs.size())
        for (auto [targetArg, valueArg] : llvm::zip(targetArgs, valueArgs)) {
          mlir::Type targetElement = types.widenLiteral(targetArg);
          mlir::Type valueElement = types.widenLiteral(valueArg);
          if (targetElement != valueElement &&
              isNumericPrimitiveContract(targetElement) &&
              isNumericPrimitiveContract(valueElement))
            return value;
        }
    }
  if (mlir::isa<py::ContractType, py::LiteralType, py::CallableType,
                py::TypeType, py::SelfType, py::TypeVarType, py::ParamSpecType>(
          targetType)) {
    auto op = py::ClassUpcastOp::create(builder, loc(anchor), targetType,
                                        value.value);
    return {op.getResult(), targetType};
  }
  return value;
}

mlir::Value ModuleEmitter::emitBoolValue(Value value,
                                         const parser::Node &anchor) {
  if (value.value && value.value.getType().isInteger(1))
    return value.value;
  mlir::Type widened = types.widenLiteral(value.type);
  // None is the falsy singleton: its truth value is static.
  if (isNoneTypeLike(widened))
    return mlir::arith::ConstantIntOp::create(builder, loc(anchor), 0, 1)
        .getResult();
  // R1: implicit numeric truthiness is rejected (deliberate deviation from
  // CPython — `if n:` over int/float hides the comparison); bool stays exempt
  // because its truth bit IS the value.
  if (widened == types.intType() || widened == types.floatType()) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "implicit truthiness of " +
            std::string(widened == types.intType() ? "int" : "float") +
            " is rejected (Lython deviation from CPython); write an explicit "
            "comparison such as `x != 0`"});
    return mlir::arith::ConstantIntOp::create(builder, loc(anchor), 1, 1)
        .getResult();
  }
  // ⭐ A UNION IS TRUE OR FALSE PER MEMBER, decided by the tag. Optional[T] was
  // the only shape handled here -- None falsy, the single present member
  // re-entering truthiness under the not-None guard -- and every other union
  // reached the manifest evidence below, which has no answer:
  //
  //     cfg = {"debug": True, "level": 3, "name": "app"}
  //     if cfg["debug"]:
  //     # static type !py.union<bool, int, str> does not provide manifest
  //     # method '__bool__'
  //
  // A union has no class and no manifest contract of its own, which is the
  // same reason `emitStringifyValue` renders one through its tag rather than
  // through the dunder ladder. This is that dispatch for truthiness, and
  // Optional[T] is now one case of it rather than a rule of its own.
  //
  // ⛔ A NUMERIC MEMBER RE-ENTERS AS `!= 0` rather than being rejected the way
  // a bare numeric is above. The deviation is about a numeric the writer could
  // have compared explicitly; there is no comparison that covers `bool | int |
  // str`, so refusing here would only make the record literal unusable in a
  // condition. Under the tag the member is known, and `x != 0` is exactly
  // CPython's bool(x) -- the argument the Optional arm already made.
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(widened)) {
    llvm::ArrayRef<mlir::Type> members = unionType.getMemberTypes();
    auto memberAnswersTruth = [&](mlir::Type member) {
      mlir::Type widenedMember = types.widenLiteral(member);
      if (isNoneTypeLike(member) || widenedMember == types.intType() ||
          widenedMember == types.floatType() ||
          widenedMember == types.boolType())
        return true;
      // Asked of the MEMBER, not of the union: this is the question the code
      // below asks of a concrete type, and it is the one that has an answer.
      return static_cast<bool>(
                 types.inferMethodCallWithEvidence(member, "__bool__", {})) ||
             static_cast<bool>(
                 types.inferMethodCallWithEvidence(member, "__len__", {}));
    };
    if (!members.empty() && llvm::all_of(members, memberAnswersTruth)) {
      auto truthOfMember = [&](mlir::Type member) -> mlir::Value {
        // None has no header to unwrap and its truth is a constant.
        if (isNoneTypeLike(member))
          return mlir::arith::ConstantIntOp::create(builder, loc(anchor), 0, 1)
              .getResult();
        auto unwrap = py::UnionUnwrapOp::create(builder, loc(anchor), member,
                                                value.value);
        Value unwrapped{unwrap.getResult(), member};
        mlir::Type widenedMember = types.widenLiteral(member);
        if (widenedMember == types.intType() ||
            widenedMember == types.floatType()) {
          Value zero;
          if (widenedMember == types.intType()) {
            mlir::Type zeroType = types.literal("0");
            zero = Value{py::IntConstantOp::create(builder, loc(anchor),
                                                   zeroType,
                                                   builder.getStringAttr("0"))
                             .getResult(),
                         zeroType};
          } else {
            zero = Value{py::FloatConstantOp::create(builder, loc(anchor),
                                                     types.floatType(),
                                                     builder.getF64FloatAttr(0.0))
                             .getResult(),
                         types.floatType()};
          }
          Value nonzero = emitBinarySpecial<py::NeOp>(
              anchor, "__ne__", unwrapped, zero, types.boolType());
          return emitBoolValue(nonzero, anchor);
        }
        return emitBoolValue(unwrapped, anchor);
      };
      auto dispatch = [&](unsigned index, auto &&recurse) -> mlir::Value {
        if (index + 1 >= members.size())
          return truthOfMember(members[index]);
        auto test = py::UnionTestOp::create(
            builder, loc(anchor), builder.getI1Type(), value.value,
            mlir::TypeAttr::get(members[index]));
        return emitValueDiamond(
            loc(anchor), test.getResult(), builder.getI1Type(),
            [&] { return truthOfMember(members[index]); },
            [&] { return recurse(index + 1, recurse); });
      };
      return dispatch(0, dispatch);
    }
  }
  // Source-class truthiness walks CPython's ladder — __bool__, then __len__,
  // then object's default (always true) — here rather than through the
  // manifest evidence below, for two reasons: the evidence path cannot see
  // source methods (so a class with __bool__ reached lowering as a manifest
  // call that does not exist), and object's default has no runtime
  // implementation to dispatch to, because "the truth of an erased object" is
  // not a runtime question — it is answered by the static class.
  if (std::optional<Value> truth = tryEmitClassDunder(anchor, value, "__bool__"))
    return emitBoolValue(*truth, anchor);
  if (std::optional<Value> length =
          tryEmitClassDunder(anchor, value, "__len__")) {
    Value count = *length;
    mlir::Type zeroType = types.literal("0");
    Value zero{py::IntConstantOp::create(builder, loc(anchor), zeroType,
                                         builder.getStringAttr("0"))
                   .getResult(),
               zeroType};
    Value nonEmpty = emitBinarySpecial<py::NeOp>(anchor, "__ne__", count, zero,
                                                 types.boolType());
    return emitBoolValue(nonEmpty, anchor);
  }
  if (inheritsObjectDefaultDunder(widened, "__bool__"))
    return mlir::arith::ConstantIntOp::create(builder, loc(anchor), 1, 1)
        .getResult();
  CallInferenceResult inference =
      types.inferMethodCallWithEvidence(value.type, "__bool__", {});
  if (!requireStaticEvidence(anchor, inference)) {
    auto fallback =
        mlir::arith::ConstantIntOp::create(builder, loc(anchor), 1, 1);
    return fallback.getResult();
  }
  auto op =
      py::BoolOp::create(builder, loc(anchor), builder.getI1Type(),
                         mlir::FlatSymbolRefAttr::get(&context, "__bool__"),
                         callProtocolFor(inference), value.value);
  return op.getResult();
}

} // namespace lython::emitter
