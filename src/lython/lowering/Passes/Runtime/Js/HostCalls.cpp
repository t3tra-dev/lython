// Member access on the JavaScript host's values: a global, an attribute read
// or write, a method call. Each crosses by the static types the emitter
// resolved against the `js` stub -- the operands' going in, the result's
// coming out -- through the `_js.JsProxy` primitives (runtime/modules/_js.mlir).

#include "Runtime/Core/Lowerer.h"
#include "Runtime/ABI/ConstantData.h"

#include "JsHost.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::lowering {
namespace {

// The stub spells a JavaScript member that is a Python keyword with a
// trailing underscore (`console.assert_`), and the host knows it as `assert`.
std::string hostMemberName(llvm::StringRef pythonName) {
  static constexpr llvm::StringLiteral kKeywords[] = {
      "False",  "None",   "True",    "and",      "as",       "assert",
      "async",  "await",  "break",   "class",    "continue", "def",
      "del",    "elif",   "else",    "except",   "finally",  "for",
      "from",   "global", "if",      "import",   "in",       "is",
      "lambda", "nonlocal", "not",   "or",       "pass",     "raise",
      "return", "try",    "while",   "with",     "yield"};
  if (pythonName.ends_with("_") &&
      llvm::is_contained(kKeywords, pythonName.drop_back()))
    return pythonName.drop_back().str();
  return pythonName.str();
}

llvm::SmallVector<mlir::Value, 3> receiverOperands(mlir::ValueRange proxy,
                                                   mlir::Value name,
                                                   mlir::Value length) {
  llvm::SmallVector<mlir::Value, 3> operands(proxy.begin(), proxy.end());
  operands.push_back(name);
  operands.push_back(length);
  return operands;
}

} // namespace

bool RuntimeBundleLowerer::isJsProxyBundle(const RuntimeBundle &bundle) const {
  return bundle.contractName() == py::kJsProxyContract;
}

mlir::FailureOr<mlir::func::CallOp>
RuntimeBundleLowerer::callJsPrimitive(mlir::Operation *op, llvm::StringRef name,
                                      mlir::ValueRange operands) {
  std::optional<RuntimeSymbol> symbol =
      manifest.primitive(py::kJsProxyContract, name);
  if (!symbol)
    return op->emitError() << "runtime manifest has no " << py::kJsProxyContract
                           << " primitive '" << name << "'";
  return RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *symbol,
                                                 operands);
}

// The member's name as the host spells it: a read-only block and its length.
std::pair<mlir::Value, mlir::Value>
RuntimeBundleLowerer::jsMemberNameOperands(mlir::Operation *op,
                                           llvm::StringRef pythonName) {
  std::string name = hostMemberName(pythonName);
  llvm::SmallVector<int8_t, 32> bytes(name.begin(), name.end());
  auto type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(bytes.size())}, builder.getI8Type());
  mlir::Value block = constant_data::internReadOnlyBlock(
      module, builder, op->getLoc(), "js_name", name,
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<int8_t>(bytes)));
  mlir::Value length = mlir::arith::ConstantIntOp::create(
      builder, op->getLoc(), static_cast<int64_t>(bytes.size()), 64);
  return {block, length};
}

// One value onto the host's argument stack, by its static type.
mlir::LogicalResult
RuntimeBundleLowerer::pushJsValue(mlir::Operation *op,
                                  const RuntimeBundle &source) {
  // A union pushes the member it holds: one branch per member on the tag,
  // each pushing that member's lanes as the member.
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(
          source.objectValue.contract)) {
    mlir::ValueRange lanes = source.physicalValues();
    if (lanes.empty())
      return op->emitError() << "JavaScript argument union has no tag";
    mlir::Value tag = lanes.front();
    for (auto [index, member] : llvm::enumerate(unionType.getMemberTypes())) {
      mlir::FailureOr<unsigned> offset =
          RuntimeBundleLowerer::unionMemberValueOffset(
              op, unionType, static_cast<unsigned>(index),
              "JavaScript argument");
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, member,
                                                     "JavaScript argument");
      if (mlir::failed(offset) || mlir::failed(memberTypes))
        return mlir::failure();
      mlir::Value expected = mlir::arith::ConstantIntOp::create(
          builder, op->getLoc(), static_cast<std::int64_t>(index), 64);
      mlir::Value holds = mlir::arith::CmpIOp::create(
          builder, op->getLoc(), mlir::arith::CmpIPredicate::eq, tag, expected);
      auto branch = mlir::scf::IfOp::create(builder, op->getLoc(), holds,
                                            /*withElseRegion=*/false);
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(branch.thenBlock()->getTerminator());
      // Borrowed: the union keeps its member, and a push reads without
      // taking.
      RuntimeBundle held = RuntimeBundle::objectWithOwnership(
          member, lanes.slice(*offset, memberTypes->size()),
          ownership::OwnershipKind::Borrow);
      if (mlir::failed(pushJsValue(op, held)))
        return mlir::failure();
    }
    return mlir::success();
  }
  std::string contract = source.contractName();
  if (contract == "types.NoneType")
    return callJsPrimitive(op, "push.none", {});
  // A float lane crosses as the number it is, with no object made for it.
  if (contract == "builtins.float" && source.primitiveF64)
    return callJsPrimitive(op, "push.f64", source.primitiveF64->value);
  mlir::FailureOr<RuntimeBundle> materialized =
      RuntimeBundleLowerer::materializeObjectBundleForStorage(
          op, source, runtimeContractType(context, contract),
          "JavaScript argument");
  if (mlir::failed(materialized))
    return mlir::failure();
  mlir::ValueRange values = materialized->physicalValues();
  if (contract == py::kJsProxyContract)
    return callJsPrimitive(op, "push.proxy", values);
  if (contract == "builtins.str")
    return callJsPrimitive(op, "push.str", values);
  if (contract == "builtins.int")
    return callJsPrimitive(op, "push.int", values);
  if (contract == "builtins.bool")
    return callJsPrimitive(op, "push.bool", values);
  if (contract == "builtins.float") {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive("builtins.float", "unbox.f64");
    if (!unbox)
      return op->emitError() << "runtime manifest has no float unbox.f64";
    mlir::func::CallOp unboxed =
        RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *unbox, values);
    return callJsPrimitive(op, "push.f64", unboxed.getResult(0));
  }
  return op->emitError()
         << "a " << contract
         << " cannot be passed to JavaScript: the crossing converts None, "
            "bool, int, float, str and JavaScript values";
}

// The host value `handle` as `result` declares it; the handle is consumed.
mlir::LogicalResult RuntimeBundleLowerer::takeJsResult(mlir::Operation *op,
                                                       mlir::Value result,
                                                       mlir::Value handle) {
  mlir::Type type = result.getType();
  std::string contract = runtimeContractName(type);
  auto take = [&](llvm::StringRef primitive)
      -> mlir::FailureOr<mlir::func::CallOp> {
    return callJsPrimitive(op, primitive, handle);
  };
  if (contract == py::kJsProxyContract) {
    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::initializeObjectFromRawValues(
            op, runtimeContractType(context, py::kJsProxyContract), handle,
            bundle)))
      return mlir::failure();
    valueBundles[result] = std::move(bundle);
    return mlir::success();
  }
  if (contract == "types.NoneType") {
    if (mlir::failed(take("take.none")))
      return mlir::failure();
    return RuntimeBundleLowerer::assignObjectBundle(
        op, result, runtimeContractType(context, "types.NoneType"), {});
  }
  if (contract == "builtins.bool") {
    mlir::FailureOr<mlir::func::CallOp> truth = take("take.bool");
    if (mlir::failed(truth))
      return mlir::failure();
    return RuntimeBundleLowerer::assignObjectBundle(
        op, result, runtimeContractType(context, "builtins.bool"),
        truth->getResults());
  }
  if (contract == "builtins.float" || contract == "builtins.int") {
    mlir::FailureOr<mlir::func::CallOp> raw =
        take(contract == "builtins.float" ? "take.f64" : "take.i64");
    if (mlir::failed(raw))
      return mlir::failure();
    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::initializeObjectFromRawValues(
            op, runtimeContractType(context, contract), raw->getResults(),
            bundle)))
      return mlir::failure();
    valueBundles[result] = std::move(bundle);
    return mlir::success();
  }
  if (contract == "builtins.str") {
    mlir::FailureOr<mlir::func::CallOp> text = take("take.str");
    if (mlir::failed(text))
      return mlir::failure();
    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, runtimeContractType(context, "builtins.str"),
            text->getResults(), bundle)))
      return mlir::failure();
    valueBundles[result] = std::move(bundle);
    return mlir::success();
  }
  return op->emitError()
         << "a JavaScript value cannot be read as " << type
         << ": the crossing converts to None, bool, int, float, str and "
            "JavaScript values";
}

bool RuntimeBundleLowerer::isJsHostGlobal(py::GlobalGetOp op) const {
  return module->hasAttr(py::kJsHostModuleAttr) &&
         op.getName().starts_with((py::kJsHostModule + ".").str());
}

mlir::LogicalResult RuntimeBundleLowerer::lowerJsGlobalGet(py::GlobalGetOp op) {
  builder.setInsertionPoint(op);
  llvm::StringRef name =
      op.getName().drop_front(py::kJsHostModule.size() + 1);
  auto [block, length] = jsMemberNameOperands(op, name);
  mlir::FailureOr<mlir::func::CallOp> global =
      callJsPrimitive(op, "global", {block, length});
  if (mlir::failed(global) ||
      mlir::failed(takeJsResult(op, op.getResult(), global->getResult(0))))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerJsAttrGet(py::AttrGetOp op,
                                     const RuntimeBundle &object) {
  builder.setInsertionPoint(op);
  auto [block, length] = jsMemberNameOperands(op, op.getName());
  mlir::FailureOr<mlir::func::CallOp> member = callJsPrimitive(
      op, "get", receiverOperands(object.physicalValues(), block, length));
  if (mlir::failed(member) ||
      mlir::failed(takeJsResult(op, op.getResult(), member->getResult(0))))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerJsAttrSet(py::AttrSetOp op, RuntimeBundle object) {
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (!value)
    return op.emitError() << "attribute value has no lowered runtime bundle";
  RuntimeBundle valueCopy = *value;
  builder.setInsertionPoint(op);
  if (mlir::failed(pushJsValue(op, valueCopy)))
    return mlir::failure();
  auto [block, length] = jsMemberNameOperands(op, op.getName());
  if (mlir::failed(callJsPrimitive(
          op, "set", receiverOperands(object.physicalValues(), block, length))))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerJsMethodCall(py::CallOp op, RuntimeBundle receiver,
                                        llvm::StringRef methodName) {
  // A union read's dispatch (ModuleEmitter::adaptJsHostResult): a test of the
  // value's kind, or a conversion of a second handle to it.
  if (methodName.starts_with("__ly_js_")) {
    builder.setInsertionPoint(op);
    static const llvm::StringMap<int> kTests = {
        {"__ly_js_is_float__", 1}, {"__ly_js_is_bool__", 2},
        {"__ly_js_is_str__", 3},   {"__ly_js_is_none__", 4},
        {"__ly_js_is_int__", 5}};
    if (auto test = kTests.find(methodName); test != kTests.end()) {
      llvm::SmallVector<mlir::Value, 2> operands(
          receiver.physicalValues().begin(), receiver.physicalValues().end());
      operands.push_back(mlir::arith::ConstantIntOp::create(
          builder, op.getLoc(), test->second, 32));
      mlir::FailureOr<mlir::func::CallOp> is =
          callJsPrimitive(op, "is", operands);
      if (mlir::failed(is) ||
          mlir::failed(RuntimeBundleLowerer::assignObjectBundle(
              op, op.getResult(0),
              runtimeContractType(context, "builtins.bool"),
              is->getResults())))
        return mlir::failure();
      erase.push_back(op);
      return mlir::success();
    }
    if (methodName == "__ly_js_instanceof__") {
      llvm::SmallVector<const RuntimeBundle *, 1> sources;
      if (mlir::failed(collectPackedObjectSources(
              op, op.getPosargs(), "isinstance constructor", sources)) ||
          sources.size() != 1 || !sources.front())
        return op.emitError() << "isinstance against a JavaScript class needs "
                                 "the constructor";
      llvm::SmallVector<mlir::Value, 2> operands(
          receiver.physicalValues().begin(), receiver.physicalValues().end());
      operands.append(sources.front()->physicalValues().begin(),
                      sources.front()->physicalValues().end());
      mlir::FailureOr<mlir::func::CallOp> is =
          callJsPrimitive(op, "instanceof", operands);
      if (mlir::failed(is) ||
          mlir::failed(RuntimeBundleLowerer::assignObjectBundle(
              op, op.getResult(0),
              runtimeContractType(context, "builtins.bool"), is->getResults())))
        return mlir::failure();
      erase.push_back(op);
      return mlir::success();
    }
    mlir::FailureOr<mlir::func::CallOp> copy =
        callJsPrimitive(op, "duplicate", receiver.physicalValues());
    if (mlir::failed(copy) ||
        mlir::failed(takeJsResult(op, op.getResult(0), copy->getResult(0))))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }
  // ⛔ JavaScript has no keyword arguments: a stub parameter is positional
  // there whatever Python lets the program write, and passing it by name
  // would have to guess the position the host's function takes it at.
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(),
                                         "JavaScript call keyword names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(),
                                         "JavaScript call keyword values")))
    return mlir::failure();
  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpacked;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "JavaScript call arguments", sources,
          &unpacked)))
    return mlir::failure();
  llvm::SmallVector<RuntimeBundle, 4> arguments;
  for (const RuntimeBundle *source : sources) {
    if (!source)
      return op.emitError() << "JavaScript call argument has no evidence";
    arguments.push_back(*source);
  }
  builder.setInsertionPoint(op);
  for (const RuntimeBundle &argument : arguments)
    if (mlir::failed(pushJsValue(op, argument)))
      return mlir::failure();
  // `X.new(...)` is the stub's spelling of `new X(...)` (Pyodide's too).
  mlir::FailureOr<mlir::func::CallOp> call;
  if (methodName == "new") {
    call = callJsPrimitive(op, "construct", receiver.physicalValues());
  } else {
    auto [block, length] = jsMemberNameOperands(op, methodName);
    call = callJsPrimitive(
        op, "call_method",
        receiverOperands(receiver.physicalValues(), block, length));
  }
  if (mlir::failed(call))
    return mlir::failure();
  if (op.getNumResults() != 1)
    return op.emitError() << "JavaScript call expects one Python result";
  if (mlir::failed(takeJsResult(op, op.getResult(0), call->getResult(0))))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
