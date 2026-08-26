#include "Runtime/Core/Lowerer.h"

#include "llvm/ADT/ScopeExit.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::lowering {
namespace {

bool isPrimitiveOnlyCallable(py::CallableType callable) {
  if (!callable || callable.hasVararg() || callable.hasKwarg())
    return false;
  auto isRuntimePrimitive = [](mlir::Type type) {
    return type && !py::isPyType(type);
  };
  return llvm::all_of(callable.getPositionalTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getKwOnlyTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getResultTypes(), isRuntimePrimitive);
}

bool hasProtocolArgumentOverride(llvm::ArrayRef<mlir::Type> types) {
  return llvm::any_of(types,
                      [](mlir::Type type) { return static_cast<bool>(type); });
}

bool sameProtocolArgumentOverrides(llvm::ArrayRef<mlir::Type> lhs,
                                   llvm::ArrayRef<mlir::Type> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto entry) {
           return std::get<0>(entry) == std::get<1>(entry);
         });
}

std::string protocolSpecializationName(llvm::StringRef originalName,
                                       unsigned ordinal) {
  return (llvm::Twine(originalName) + "__lyrt_proto_" + llvm::Twine(ordinal))
      .str();
}

} // namespace

const RuntimeValueShape *
RuntimeBundleLowerer::runtimeValueShapeFor(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef purpose) const {
  std::string contract = runtimeShapeContractName(type);
  if (contract.empty()) {
    op->emitError() << purpose << " has no concrete runtime contract: " << type;
    return nullptr;
  }
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  if (!shape)
    op->emitError() << "runtime manifest has no ABI shape for " << contract
                    << " " << purpose;
  return shape;
}

bool RuntimeBundleLowerer::contractIsSubclassed(
    llvm::StringRef contract) const {
  if (!subclassedContracts) {
    subclassedContracts.emplace();
    mlir::ModuleOp mutableModule =
        const_cast<RuntimeBundleLowerer *>(this)->module;
    mutableModule.walk([&](py::ClassOp classOp) {
      auto bases = classOp->getAttrOfType<mlir::ArrayAttr>("base_names");
      if (!bases)
        return;
      for (mlir::Attribute base : bases)
        if (auto name = mlir::dyn_cast<mlir::StringAttr>(base))
          subclassedContracts->insert(name.getValue());
    });
  }
  // A base is named as written, which may be the bare class name or the
  // qualified contract; both spellings are recorded, so both are asked.
  if (subclassedContracts->contains(contract))
    return true;
  auto dot = contract.rfind('.');
  return dot != llvm::StringRef::npos &&
         subclassedContracts->contains(contract.substr(dot + 1));
}

py::ClassOp RuntimeBundleLowerer::classForContract(mlir::Type type) const {
  std::string contract = runtimeContractName(type);
  if (contract.empty())
    return {};
  mlir::ModuleOp mutableModule =
      const_cast<RuntimeBundleLowerer *>(this)->module;
  auto lookup = [&](llvm::StringRef name) -> py::ClassOp {
    return mlir::dyn_cast_or_null<py::ClassOp>(
        mlir::SymbolTable::lookupSymbolIn(mutableModule.getOperation(), name));
  };
  if (py::ClassOp classOp = lookup(contract))
    return classOp;

  llvm::StringRef shortName = llvm::StringRef(contract).rsplit('.').second;
  if (!shortName.empty() && shortName != contract)
    return lookup(shortName);
  return {};
}

bool RuntimeBundleLowerer::classDefinesMethod(mlir::Type type,
                                              llvm::StringRef name) const {
  py::ClassOp classOp = RuntimeBundleLowerer::classForContract(type);
  if (!classOp)
    return false;
  auto methodNames = classOp->getAttrOfType<mlir::ArrayAttr>("method_names");
  if (!methodNames)
    return false;
  for (mlir::Attribute attr : methodNames) {
    auto methodName = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (methodName && methodName.getValue() == name)
      return true;
  }
  return false;
}

std::optional<std::string>
RuntimeBundleLowerer::classMethodSymbol(py::ClassOp classOp,
                                        llvm::StringRef name) const {
  if (!classOp)
    return std::nullopt;
  auto methodNames = classOp->getAttrOfType<mlir::ArrayAttr>("method_names");
  auto methodSymbols =
      classOp->getAttrOfType<mlir::ArrayAttr>("method_symbols");
  if (!methodNames || !methodSymbols ||
      methodNames.size() != methodSymbols.size())
    return std::nullopt;
  for (auto [nameAttr, symbolAttr] : llvm::zip(methodNames, methodSymbols)) {
    auto methodName = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    auto symbol = mlir::dyn_cast<mlir::StringAttr>(symbolAttr);
    if (methodName && symbol && methodName.getValue() == name)
      return symbol.getValue().str();
  }
  // ⭐ Then the bases, the way attribute lookup does.
  //
  //     class Base:
  //         def __repr__(self) -> str: return "Base()"
  //     class Kid(Base): pass
  //     print([Kid()])
  //
  // aborted with "repr: boxed element has no conforming __repr__". `Kid`
  // declares no methods of its own, so the boxed-method dispatch had no entry
  // for its class id and the container's repr found nothing to call --
  // `repr(Kid())` written directly was fine, because that path resolves
  // through the emitter's MRO walk rather than through this one.
  //
  // The manifest's exception subclasses already had a rescue for the same gap
  // (`shareExceptionSubclasses`, which hands them BaseException's callee);
  // this is that rule for source classes, stated where the lookup happens
  // instead of as a second special case at the call site.
  auto baseNames = classOp->getAttrOfType<mlir::ArrayAttr>("base_names");
  if (!baseNames)
    return std::nullopt;
  for (mlir::Attribute baseAttr : baseNames) {
    auto baseName = mlir::dyn_cast<mlir::StringAttr>(baseAttr);
    if (!baseName)
      continue;
    py::ClassOp base = RuntimeBundleLowerer::classForContract(
        runtimeContractType(classOp->getContext(), baseName.getValue()));
    if (!base || base == classOp)
      continue;
    if (std::optional<std::string> inherited =
            RuntimeBundleLowerer::classMethodSymbol(base, name))
      return inherited;
  }
  return std::nullopt;
}

llvm::SmallVector<mlir::Type, 8>
RuntimeBundleLowerer::classFieldContractTypes(py::ClassOp classOp) const {
  llvm::SmallVector<mlir::Type, 8> types;
  auto attrs = classOp->getAttrOfType<mlir::ArrayAttr>("field_contract_types");
  if (!attrs)
    attrs = classOp->getAttrOfType<mlir::ArrayAttr>("field_types");
  if (!attrs)
    return types;
  types.reserve(attrs.size());
  for (mlir::Attribute attr : attrs) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr)
      return {};
    types.push_back(typeAttr.getValue());
  }
  return types;
}

mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
RuntimeBundleLowerer::runtimeValueTypesFor(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef purpose) const {
  // ⭐ `type[X]` IS ITS OWN VALUE. A type object carries nothing a program can
  // observe beyond WHICH class it is, and that is in the type -- so its
  // physical shape is EMPTY, and a parameter, a field, a return and a
  // suspension lane all carry it for free. Constructing through it stays
  // statically resolved, which is the whole point: an i64 class id in a
  // parameter would make `t(3)` a dispatch on a runtime value, and this
  // compiler has no dynamic dispatch to fall back to.
  //
  // ⛔ Why NOT the i64 class id, which is what three earlier attempts built
  // and reverted: each moved the refusal one layer down (no contract -> bad
  // object header -> no unbox.i64 primitive -> the bundle is not a
  // TypeObject) and the fourth layer is where it stops being a wiring
  // question. `t(3)` on a parameter-held class id has to pick a constructor
  // at run time, and every ancestor of that decision -- the ABI, the field
  // slot, the operand -- was being built to support a dispatch that would
  // then have to be refused anyway.
  //
  // The soundness condition is that the class is DETERMINED by the type, so a
  // class with subclasses is refused here rather than silently constructing
  // the base. Reaching those is the argument specializer's job: `make(Base)`
  // and `make(Derived)` get one body each, and inside each the parameter's
  // type names exactly one class.
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type)) {
    mlir::Type instance = typeType.getInstanceType();
    std::string instanceName = runtimeContractName(instance);
    if (instanceName.empty()) {
      op->emitError() << purpose << " has no concrete runtime contract: "
                      << type;
      return mlir::failure();
    }
    if (contractIsSubclassed(instanceName)) {
      op->emitError()
          << purpose << " is " << type
          << ", whose class is subclassed in this program, so which class it "
             "names is not decided by its type; pass the exact class (a "
             "`type[" << instanceName
          << "]` parameter reached with a subclass specializes per call) or "
             "annotate the exact one";
      return mlir::failure();
    }
    return llvm::SmallVector<mlir::Type, 8>{};
  }
  // ⭐ A layout cannot contain itself. A union of two OBJECTS stays INLINE (see
  // `classFieldStoredBoxed`), so a class reachable from its own field through
  // one expanded forever and the COMPILER died with SIGILL and not one byte of
  // diagnostic. A crash with no message is the worst answer a compiler can
  // give, so the cycle is reported where it is entered.
  //
  // ⛔ `T | None` no longer arrives here at all: it is stored as a box whose
  // empty state IS None, so `nxt: Optional["Node"]` -- the shape every linked
  // structure is written in -- terminates. What remains is the union of two
  // things that are not the same object, and for that the message can only
  // name the spellings that are boxed.
  if (!expandingContracts.insert(type).second)
    return op->emitError()
           << "class layout for " << type
           << " contains itself through a union-typed field of two object "
              "types, which is stored inline and so has no finite layout; a "
              "field typed with the class itself, or with a union of it and "
              "None, is stored as a reference and terminates";
  auto expanding = llvm::make_scope_exit([&] { expandingContracts.erase(type); });
  if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
    llvm::SmallVector<mlir::Type, 8> types{mlir::IntegerType::get(context, 64)};
    for (mlir::Type member : unionType.getMemberTypes()) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, member, purpose);
      if (mlir::failed(memberTypes))
        return mlir::failure();
      types.append(memberTypes->begin(), memberTypes->end());
    }
    return types;
  }

  std::string contract = runtimeShapeContractName(type);
  if (!contract.empty()) {
    if (const RuntimeValueShape *shape = manifest.valueShape(contract))
      return llvm::SmallVector<mlir::Type, 8>(shape->valueTypes.begin(),
                                              shape->valueTypes.end());
  }

  if (py::ClassOp classOp = RuntimeBundleLowerer::classForContract(type)) {
    // Exception-backed source classes use the runtime exception object's
    // shape (their identity lives in the header's class id, not the layout),
    // so raise/borrow/str flow through the builtin exception machinery.
    if (std::optional<std::string> ancestor =
            RuntimeBundleLowerer::exceptionAncestorContract(classOp)) {
      const RuntimeValueShape *shape = manifest.valueShape(*ancestor);
      if (!shape)
        shape = manifest.valueShape("builtins.BaseException");
      if (!shape)
        return op->emitError()
               << "runtime manifest has no exception ABI shape for "
               << purpose;
      return llvm::SmallVector<mlir::Type, 8>(shape->valueTypes.begin(),
                                              shape->valueTypes.end());
    }
    const RuntimeValueShape *objectShape =
        manifest.valueShape("builtins.object");
    if (!objectShape)
      return op->emitError()
             << "runtime manifest has no builtins.object ABI shape for "
             << purpose;
    llvm::SmallVector<mlir::Type, 8> types(objectShape->valueTypes.begin(),
                                           objectShape->valueTypes.end());
    for (auto [fieldIndex, fieldType] : llvm::enumerate(
             RuntimeBundleLowerer::classFieldContractTypes(classOp))) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldTypes =
          RuntimeBundleLowerer::classFieldStorageValueTypes(
              op, fieldType, static_cast<unsigned>(fieldIndex), purpose);
      if (mlir::failed(fieldTypes))
        return mlir::failure();
      types.append(fieldTypes->begin(), fieldTypes->end());
    }
    return types;
  }

  if (!contract.empty()) {
    const RuntimeValueShape *objectShape =
        manifest.valueShape("builtins.object");
    if (!objectShape)
      return op->emitError()
             << "runtime manifest has no builtins.object ABI shape for "
             << purpose;
    return llvm::SmallVector<mlir::Type, 8>(objectShape->valueTypes.begin(),
                                            objectShape->valueTypes.end());
  }

  const RuntimeValueShape *shape =
      RuntimeBundleLowerer::runtimeValueShapeFor(op, type, purpose);
  if (!shape)
    return mlir::failure();
  return llvm::SmallVector<mlir::Type, 8>(shape->valueTypes.begin(),
                                          shape->valueTypes.end());
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeValueTypes(
    mlir::Operation *op, mlir::Type type,
    llvm::SmallVectorImpl<mlir::Type> &types) const {
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, type, "callable ABI type");
  if (mlir::failed(valueTypes))
    return mlir::failure();
  types.append(valueTypes->begin(), valueTypes->end());
  return mlir::success();
}

bool RuntimeBundleLowerer::hasPrimitiveI64ABI(mlir::Type type) const {
  return runtimeContractName(type) == "builtins.int";
}

void RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(
    mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &types) const {
  if (!RuntimeBundleLowerer::hasPrimitiveI64ABI(type))
    return;
  types.push_back(mlir::IntegerType::get(context, 64));
  types.push_back(mlir::IntegerType::get(context, 1));
}

llvm::SmallVector<mlir::Type, 4>
RuntimeBundleLowerer::callableClosureTypes(mlir::func::FuncOp function) const {
  llvm::SmallVector<mlir::Type, 4> types;
  auto attr = function->getAttrOfType<mlir::ArrayAttr>("closure_types");
  if (!attr)
    return types;
  for (mlir::Attribute entry : attr) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(entry);
    if (!typeAttr)
      return {};
    types.push_back(typeAttr.getValue());
  }
  return types;
}

mlir::Type
RuntimeBundleLowerer::callableVarargValueType(mlir::func::FuncOp function,
                                              py::CallableType callable) const {
  if (auto attr =
          function->getAttrOfType<mlir::TypeAttr>(kCallableVarargValueTypeAttr))
    return attr.getValue();
  return callable.hasVararg() ? callable.getVarargType() : mlir::Type();
}

mlir::Type
RuntimeBundleLowerer::callableKwargValueType(mlir::func::FuncOp function,
                                             py::CallableType callable) const {
  if (auto attr =
          function->getAttrOfType<mlir::TypeAttr>(kCallableKwargValueTypeAttr))
    return attr.getValue();
  return callable.hasKwarg() ? callable.getKwargType() : mlir::Type();
}

llvm::SmallVector<mlir::Type, 8>
RuntimeBundleLowerer::callableLogicalInputTypes(
    mlir::func::FuncOp function, py::CallableType callable) const {
  llvm::SmallVector<mlir::Type, 8> logicalInputTypes(
      callable.getPositionalTypes().begin(),
      callable.getPositionalTypes().end());
  logicalInputTypes.append(callable.getKwOnlyTypes().begin(),
                           callable.getKwOnlyTypes().end());
  if (callable.hasVararg())
    logicalInputTypes.push_back(callableVarargValueType(function, callable));
  if (callable.hasKwarg())
    logicalInputTypes.push_back(callableKwargValueType(function, callable));
  llvm::SmallVector<mlir::Type, 4> closureTypes =
      callableClosureTypes(function);
  logicalInputTypes.append(closureTypes.begin(), closureTypes.end());
  return logicalInputTypes;
}

bool RuntimeBundleLowerer::isPrimitiveI64CallableClone(
    mlir::func::FuncOp function) const {
  return function && function->hasAttr(kPrimitiveI64CloneAttr);
}

bool RuntimeBundleLowerer::isCallableProtocolTemplate(
    mlir::func::FuncOp function) const {
  return function && function->hasAttr(kProtocolTemplateAttr);
}

// The clone's return ABI is (i64 raw, i1 valid) with no boxed lane, so a step
// that needs a boxed fallback cannot take one: it can only record that the raw
// lane stopped tracking the true Python value. A stack slot (rather than an
// extra block argument threaded through every block) is used because the
// clone's cf blocks are already built by the time a step discovers it must
// poison, and re-signaturing them mid-lowering dangles the saved insertion
// iterators the surrounding walk depends on. mem2reg promotes it away.
void RuntimeBundleLowerer::refusePrimitiveI64Clone(mlir::func::FuncOp clone) {
  if (clone && RuntimeBundleLowerer::isPrimitiveI64CallableClone(clone))
    refusedPrimitiveI64Clones.insert(clone.getOperation());
}

// The clone-local `memref<1xi64>` holding 1 while every decision the body has
// made so far came from a lane that really was the Python value, 0 once one
// did not. A comparison is where a lane's validity would otherwise be lost --
// an i1 has nowhere to carry it -- so the bit is parked here and AND-ed into
// whatever the clone returns.
//
// ⛔ NOT a refusal, which is what a stale comparison used to force: a `while`
// whose counter overflows is exactly this shape, and refusing it left every
// integer loop on the boxed path. The wrong branch it may take is bounded --
// the guard at each clone call means a stale lane never ENTERS a clone, so it
// cannot drive unbounded recursion, and `and(valid, compared)` reads false,
// which is the exiting direction for a loop.
void RuntimeBundleLowerer::parkPrimitiveI64CloneDecision(mlir::Operation *op,
                                                         mlir::Value stillValid) {
  auto clone = op->getParentOfType<mlir::func::FuncOp>();
  if (!clone || !RuntimeBundleLowerer::isPrimitiveI64CallableClone(clone))
    return;

  mlir::Location loc = op->getLoc();
  mlir::Value &slot = primitiveI64CloneDecisionFlags[clone.getOperation()];
  if (!slot) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&clone.getBody().front());
    slot = mlir::memref::AllocaOp::create(
               builder, loc, mlir::MemRefType::get({1}, builder.getI64Type()))
               .getResult();
    mlir::memref::StoreOp::create(builder, loc,
                                  constantI64(builder, loc, 1), slot,
                                  constantIndex(builder, loc, 0));
  }

  mlir::Value zero = constantIndex(builder, loc, 0);
  mlir::Value current =
      mlir::memref::LoadOp::create(builder, loc, slot, zero).getResult();
  mlir::Value widened =
      mlir::arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                   stillValid)
          .getResult();
  mlir::memref::StoreOp::create(
      builder, loc,
      mlir::arith::AndIOp::create(builder, loc, current, widened).getResult(),
      slot, zero);
}

mlir::Value RuntimeBundleLowerer::primitiveI64CloneDecisionsIntact(
    mlir::Operation *op, mlir::func::FuncOp clone) {
  auto found = primitiveI64CloneDecisionFlags.find(clone.getOperation());
  if (found == primitiveI64CloneDecisionFlags.end() || !found->second)
    return {};
  mlir::Location loc = op->getLoc();
  mlir::Value current =
      mlir::memref::LoadOp::create(builder, loc, found->second,
                                   constantIndex(builder, loc, 0))
          .getResult();
  return mlir::arith::CmpIOp::create(builder, loc,
                                     mlir::arith::CmpIPredicate::ne, current,
                                     constantI64(builder, loc, 0))
      .getResult();
}

bool RuntimeBundleLowerer::isUsablePrimitiveI64Clone(
    mlir::func::FuncOp clone) const {
  return clone && !refusedPrimitiveI64Clones.contains(clone.getOperation());
}

std::optional<std::string>
RuntimeBundleLowerer::callableProtocolSpecializationFor(
    llvm::StringRef target,
    llvm::ArrayRef<const RuntimeBundle *> sources) const {
  auto found = callableProtocolSpecializations.find(target);
  if (found == callableProtocolSpecializations.end())
    return std::nullopt;

  for (const CallableProtocolSpecialization &specialization :
       found->second) {
    bool matches = true;
    for (auto [index, expected] :
         llvm::enumerate(specialization.argumentTypes)) {
      if (!expected)
        continue;
      if (index >= sources.size() || !sources[index]) {
        matches = false;
        break;
      }
      mlir::Type actual = sources[index]->contract;
      if (actual == expected)
        continue;
      if (py::isAssignableTo(actual, expected))
        continue;
      matches = false;
      break;
    }
    if (matches)
      return specialization.cloneName;
  }
  return std::nullopt;
}

mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::selectCallableProtocolSpecialization(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (std::optional<std::string> cloneName =
          RuntimeBundleLowerer::callableProtocolSpecializationFor(targetName,
                                                                  sources)) {
    if (mlir::func::FuncOp clone =
            module.lookupSymbol<mlir::func::FuncOp>(*cloneName))
      return clone;
    return op.emitError() << "protocol specialization clone @" << *cloneName
                          << " for callable target " << targetName
                          << " is not defined";
  }

  if (RuntimeBundleLowerer::isCallableProtocolTemplate(target))
    return op.emitError()
           << "protocol-typed callable target " << targetName
           << " has no static specialization for these argument contracts";
  return target;
}

std::optional<std::string>
RuntimeBundleLowerer::primitiveI64CloneFor(llvm::StringRef target) const {
  auto found = primitiveI64CallableClones.find(target);
  if (found == primitiveI64CallableClones.end())
    return std::nullopt;
  return found->second;
}

bool RuntimeBundleLowerer::isPrimitiveI64CallableEligible(
    mlir::func::FuncOp function) const {
  if (!function || function.isDeclaration() ||
      RuntimeBundleLowerer::isPrimitiveI64CallableClone(function) ||
      RuntimeBundleLowerer::isCallableProtocolTemplate(function))
    return false;
  if (!RuntimeBundleLowerer::callableClosureTypes(function).empty())
    return false;

  py::CallableType callable = callableTypeOf(function);
  if (!callable || callable.getResultTypes().size() != 1 ||
      callable.hasVararg() || callable.hasKwarg() ||
      !callable.getKwOnlyTypes().empty() ||
      llvm::any_of(callable.getPositionalDefaults(),
                   [](mlir::BoolAttr attr) { return attr && attr.getValue(); }))
    return false;
  if (runtimeContractName(callable.getResultTypes().front()) != "builtins.int")
    return false;
  if (!llvm::all_of(callable.getPositionalTypes(), [](mlir::Type type) {
        return runtimeContractName(type) == "builtins.int";
      }))
    return false;
  // A return value from a control-flow MERGE (if/loop) is a boxed object,
  // which the unboxed-i64 clone return ABI cannot represent.
  //
  // ⛔ The test is "not the entry block", not "is a BlockArgument". A
  // function's parameters are the entry block's arguments, and inside a clone
  // they ARE the raw lane -- so `return n` is exactly what the clone ABI
  // represents best. Testing `isa<BlockArgument>` refused every function that
  // returns a parameter, which is `fib`, and with it every recursive integer
  // function: measured, no source function in the tree ever got a clone.
  return true;
}

namespace {

// Nothing a clone does before it reports valid=false may be observable: the
// call site answers a false by running the boxed original, so the clone's work
// has to be a rehearsal. Pure i64 arithmetic and calls to other rehearsable
// clones qualify; a runtime call, a store or an allocation does not.
//
// Only the plain (i64, i1) shape is judged -- the wider resume ABIs of
// generator clones are never speculated on.
bool isReplaySafeBody(mlir::func::FuncOp clone,
                      llvm::function_ref<bool(mlir::func::FuncOp)> calleeSafe) {
  bool safe = true;
  clone.walk([&](mlir::Operation *op) {
    if (op == clone.getOperation())
      return mlir::WalkResult::advance();
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          clone, call.getCalleeAttr());
      if (!callee || !calleeSafe(callee)) {
        safe = false;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "arith" || dialect == "cf" || dialect == "scf" ||
        dialect == "func")
      return mlir::WalkResult::advance();
    // The decision flag: a slot the clone allocates, writes and reads itself,
    // dead the moment it returns. Nothing outside can see it, which is the
    // whole question here.
    if (mlir::isa<mlir::memref::AllocaOp>(op))
      return mlir::WalkResult::advance();
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      if (load.getMemRef().getDefiningOp<mlir::memref::AllocaOp>())
        return mlir::WalkResult::advance();
    }
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      if (store.getMemRef().getDefiningOp<mlir::memref::AllocaOp>())
        return mlir::WalkResult::advance();
    }
    safe = false;
    return mlir::WalkResult::interrupt();
  });
  return safe;
}

} // namespace

// Phase-order-independent repair of the clone speculation shape emitted by
// emitPrimitiveI64CloneFallbackResult. The call sites are lowered before the
// clone bodies' returns are, so they cannot tell whether the clone they are
// speculating on can fail; this runs once every clone is final and drops the
// speculation wherever failure -- and therefore a second, observable run of
// the original body -- is possible.
//
// WHY NOT the alternatives: giving the clone a boxed return lane would make it
// allocate, and the ctypes callback thunks call clones directly from signal
// handlers (verifyCallbackSignalSafety rejects allocation there). Restricting
// which functions get a clone at all would take the stackguard handlers' clones
// away, and Ctypes/Objects.cpp hard-errors when a callback target has none.
// Unboxing at the clone's return to force valid=1 would truncate bigints.
// Folding here touches neither the clone set nor the clone ABI: it only stops
// callers from betting on a clone that can decline to answer.
// A clone may be speculated on when its raw lane drives only exact branches
// (it was never refused), its ABI is the plain (i64, i1) pair, and its body is
// a rehearsal -- transitively, since a clone calling a refused clone inherits
// that clone's wrong branches. The walk is optimistic about the cycle a
// recursive clone forms with itself, which is what lets `fib` speculate at all.
bool RuntimeBundleLowerer::isSpeculablePrimitiveI64Clone(
    mlir::func::FuncOp clone) {
  if (!clone || clone.isDeclaration() ||
      !RuntimeBundleLowerer::isPrimitiveI64CallableClone(clone))
    return false;
  mlir::FunctionType type = clone.getFunctionType();
  if (type.getNumResults() != 2 || !type.getResult(0).isInteger(64) ||
      !type.getResult(1).isInteger(1))
    return false;

  llvm::DenseSet<mlir::Operation *> assumedSafe;
  llvm::SmallVector<mlir::func::FuncOp, 4> pending{clone};
  assumedSafe.insert(clone.getOperation());
  while (!pending.empty()) {
    mlir::func::FuncOp current = pending.pop_back_val();
    if (!RuntimeBundleLowerer::isUsablePrimitiveI64Clone(current))
      return false;
    if (!isReplaySafeBody(current, [&](mlir::func::FuncOp callee) {
          if (!RuntimeBundleLowerer::isPrimitiveI64CallableClone(callee) ||
              callee.isDeclaration())
            return false;
          if (assumedSafe.insert(callee.getOperation()).second)
            pending.push_back(callee);
          return true;
        }))
      return false;
  }
  return true;
}

mlir::LogicalResult
RuntimeBundleLowerer::foldUnprovenPrimitiveI64Speculations() {
  llvm::SmallVector<mlir::scf::IfOp, 8> speculations;
  module.walk([&](mlir::scf::IfOp ifOp) {
    if (ifOp->hasAttr(kPrimitiveI64SpeculationAttr))
      speculations.push_back(ifOp);
  });

  for (mlir::scf::IfOp ifOp : speculations) {
    auto cloneRef = ifOp->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kPrimitiveI64SpeculationAttr);
    ifOp->removeAttr(kPrimitiveI64SpeculationAttr);
    // A marker without a resolvable clone cannot be shown safe, so it folds.
    mlir::func::FuncOp clone;
    if (cloneRef)
      clone = module.lookupSymbol<mlir::func::FuncOp>(cloneRef.getAttr());
    if (isSpeculablePrimitiveI64Clone(clone))
      continue;

    // The condition is the clone's validity answer; capture what produced it
    // before the only uses of it disappear. It is the call itself when the
    // arguments were provably valid, and otherwise the `scf.if` that guards
    // the call -- and THAT one has to go too: it still contains the call, and
    // a clone that is not replay-safe is one whose side effects would then be
    // observed twice. `int_clone_speculation_effects` caught exactly that.
    mlir::Operation *cloneCall = ifOp.getCondition().getDefiningOp();
    mlir::Block *elseBlock = &ifOp.getElseRegion().front();
    auto yield = mlir::cast<mlir::scf::YieldOp>(elseBlock->getTerminator());
    llvm::SmallVector<mlir::Value, 8> results(yield.getOperands());

    // Everything the else region computes only reads values defined outside
    // the scf.if, so hoisting it to just before the op preserves dominance.
    mlir::Block *parent = ifOp->getBlock();
    parent->getOperations().splice(mlir::Block::iterator(ifOp),
                                   elseBlock->getOperations(),
                                   elseBlock->begin(),
                                   mlir::Block::iterator(yield));
    ifOp.getOperation()->replaceAllUsesWith(results);
    ifOp.erase();
    if (cloneCall && cloneCall->use_empty())
      cloneCall->erase();
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::buildPrimitiveI64CallableClones() {
  llvm::SmallVector<mlir::func::FuncOp, 8> originals;
  module.walk([&](mlir::func::FuncOp function) {
    if (RuntimeBundleLowerer::isPrimitiveI64CallableEligible(function))
      originals.push_back(function);
  });

  for (mlir::func::FuncOp original : originals) {
    std::string originalName = original.getSymName().str();
    std::string cloneName = (original.getSymName() + "__lyrt_prim_i64").str();
    if (module.lookupSymbol<mlir::func::FuncOp>(cloneName)) {
      primitiveI64CallableClones[originalName] = cloneName;
      continue;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(original);
    mlir::func::FuncOp clone = original.clone();
    clone.setSymName(cloneName);
    clone->setAttr(kPrimitiveI64CloneAttr,
                   builder.getStringAttr(original.getSymName()));
    mlir::SymbolTable::setSymbolVisibility(
        clone, mlir::SymbolTable::Visibility::Private);
    builder.insert(clone);
    primitiveI64CallableClones[originalName] = cloneName;
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::seedPrimitiveI64CallableEntryArgumentBundles(
    mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes) {
  if (function.isDeclaration())
    return mlir::success();
  mlir::Block &entry = function.getBody().front();
  if (entry.getNumArguments() != logicalTypes.size())
    return function.emitError()
           << "primitive i64 callable clone entry argument count does not "
              "match callable_type";

  unsigned logicalArgCount = entry.getNumArguments();
  for (auto [index, logicalType] : llvm::enumerate(logicalTypes)) {
    if (runtimeContractName(logicalType) != "builtins.int")
      return function.emitError()
             << "primitive i64 callable clone argument " << index
             << " must be builtins.int, got " << logicalType;
    mlir::BlockArgument logicalArg = entry.getArgument(index);
    mlir::BlockArgument raw = entry.addArgument(
        mlir::IntegerType::get(context, 64), logicalArg.getLoc());
    // ⭐ VALIDITY IS THE CALLER'S OBLIGATION, NOT AN ARGUMENT. It used to ride
    // in as an i1 the body then AND-ed into its branches, which meant a clone
    // entered with valid=false took the wrong arm of every test -- `fib(93)`,
    // whose i64 lane overflows, recursed until the stack guard fired. With the
    // branch moved to the call site the lane is true here by contract, so the
    // body's branches are exactly the original's.
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&entry);
    mlir::Value valid = constantBool(builder, logicalArg.getLoc(), true);

    RuntimeBundle bundle = RuntimeBundle::objectWithOwnership(
        logicalType, mlir::ValueRange{},
        ownership::logicalOwnershipKind(logicalType,
                                                /*ownsObject=*/false));
    bundle.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
    valueBundles[logicalArg] = std::move(bundle);
  }
  callableLogicalEntryArgCounts.push_back(
      CallableLogicalEntryArgs{function, logicalArgCount});
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::buildCallableProtocolArgumentABIs() {
  struct Accumulator {
    llvm::SmallVector<llvm::SmallVector<mlir::Type, 8>, 4> specializations;
  };

  llvm::StringMap<Accumulator> accumulators;
  module.walk([&](py::CallOp call) {
    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return mlir::WalkResult::advance();

    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!target || target.isDeclaration())
      return mlir::WalkResult::advance();

    py::CallableType callable = callableTypeOf(target);
    if (!callable)
      return mlir::WalkResult::advance();

    llvm::SmallVector<mlir::Type, 8> logicalTypes =
        callableLogicalInputTypes(target, callable);
    std::optional<llvm::SmallVector<mlir::Type, 4>> sourceTypes =
        RuntimeBundleLowerer::collectCallableArgumentSourceTypes(call,
                                                                 callable);
    if (!sourceTypes || sourceTypes->size() > logicalTypes.size())
      return mlir::WalkResult::advance();

    llvm::SmallVector<mlir::Type, 8> argumentTypes(logicalTypes.size());
    for (auto [index, sourceType] : llvm::enumerate(*sourceTypes)) {
      mlir::Type logicalType = logicalTypes[index];
      if (!mlir::isa<py::ProtocolType>(logicalType) ||
          !runtimeContractName(logicalType).empty())
        continue;

      std::string sourceContract = runtimeContractName(sourceType);
      if (sourceContract.empty())
        continue;
      argumentTypes[index] = sourceType;
    }
    if (!hasProtocolArgumentOverride(argumentTypes))
      return mlir::WalkResult::advance();

    Accumulator &acc = accumulators[target.getSymName()];
    if (llvm::none_of(acc.specializations, [&](llvm::ArrayRef<mlir::Type> item) {
          return sameProtocolArgumentOverrides(item, argumentTypes);
        }))
      acc.specializations.push_back(std::move(argumentTypes));
    return mlir::WalkResult::advance();
  });

  for (auto &entry : accumulators) {
    mlir::func::FuncOp original =
        module.lookupSymbol<mlir::func::FuncOp>(entry.getKey());
    if (!original || original.isDeclaration())
      continue;

    llvm::SmallVector<CallableProtocolSpecialization, 4> &specializations =
        callableProtocolSpecializations[entry.getKey()];
    for (auto [ordinal, argumentTypes] :
         llvm::enumerate(entry.getValue().specializations)) {
      std::string cloneName =
          protocolSpecializationName(entry.getKey(), ordinal);
      mlir::func::FuncOp clone =
          module.lookupSymbol<mlir::func::FuncOp>(cloneName);
      if (!clone) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(original);
        clone = original.clone();
        clone.setSymName(cloneName);
        clone->setAttr(kProtocolSpecializationAttr,
                       builder.getStringAttr(original.getSymName()));
        mlir::SymbolTable::setSymbolVisibility(
            clone, mlir::SymbolTable::Visibility::Private);
        builder.insert(clone);
      }
      callableProtocolArgumentABIs[cloneName] = argumentTypes;
      if (auto returnedValue = returnedValueSummaries.find(entry.getKey());
          returnedValue != returnedValueSummaries.end())
        returnedValueSummaries[cloneName] = returnedValue->second;
      if (auto returnedCallable =
              returnedCallableSummaries.find(entry.getKey());
          returnedCallable != returnedCallableSummaries.end())
        returnedCallableSummaries[cloneName] = returnedCallable->second;
      specializations.push_back(CallableProtocolSpecialization{
          cloneName, llvm::SmallVector<mlir::Type, 8>(argumentTypes)});
    }
    if (!specializations.empty())
      original->setAttr(kProtocolTemplateAttr, builder.getUnitAttr());
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::prepareCallableFunctionABIs() {
  mlir::LogicalResult result = mlir::success();
  // Collected once: the walk below asks, per union-typed result, which of its
  // members are entities the caller has to release. Only a contract with a
  // release interface is one -- `int | str | None` has two and `int | None`
  // has one, and a member that is merely NAMED (a bool, a None) has none.
  llvm::SmallVector<ownership::RuntimeDeallocator, 8> deallocators =
      ownership::collectRuntimeDeallocators(module);
  auto memberOwnsALane = [&](mlir::Type member) {
    if (py::isPyNoneType(member))
      return false;
    std::string contract = runtimeContractName(member);
    if (contract.empty())
      return false;
    return llvm::any_of(deallocators,
                        [&](const ownership::RuntimeDeallocator &candidate) {
                          return candidate.contractName == contract;
                        });
  };
  module.walk([&](mlir::func::FuncOp function) {
    auto callableType =
        function->getAttrOfType<mlir::TypeAttr>("callable_type");
    if (!callableType)
      return mlir::WalkResult::advance();
    auto callable = mlir::dyn_cast<py::CallableType>(callableType.getValue());
    if (!callable) {
      function.emitError() << "callable_type must be Callable";
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    if (RuntimeBundleLowerer::isCallableProtocolTemplate(function))
      return mlir::WalkResult::advance();
    if (isPrimitiveOnlyCallable(callable))
      return mlir::WalkResult::advance();
    llvm::SmallVector<mlir::Type, 8> logicalInputTypes =
        callableLogicalInputTypes(function, callable);
    if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(function)) {
      GeneratorResumeInfo *generatorArgInfo =
          RuntimeBundleLowerer::generatorResumeInfoForClone(function);
      unsigned generatorControlCount =
          generatorArgInfo ? generatorArgInfo->argumentCount + 3 : 0;
      llvm::SmallVector<mlir::Type, 8> inputTypes;
      llvm::SmallVector<std::int64_t, 8> generatorTransferArgs;
      llvm::SmallVector<mlir::Attribute, 8> generatorResumeArgLanes;
      llvm::SmallVector<mlir::Attribute, 8> generatorBorrowedArgLanes;
      for (auto [inputIndex, logicalType] :
           llvm::enumerate(logicalInputTypes)) {
        // Generator ARGUMENT lanes (indices below argumentCount) with an
        // object contract are borrows: the generator object retained them at
        // creation (released by its drop finalizer), so the resume drivers
        // re-pass the span each call against that frame-held reference —
        // no ownership crosses here (rfc/memory-safety-proof.md: a borrow
        // against a frame-owned rho, not an Own introduction).
        if (generatorArgInfo && inputIndex < generatorArgInfo->argumentCount &&
            inputIndex < generatorArgInfo->argumentLanes.size() &&
            !generatorArgInfo->argumentLanes[inputIndex].isInt &&
            !generatorArgInfo->argumentLanes[inputIndex].isControl()) {
          const GeneratorResumeLane &lane =
              generatorArgInfo->argumentLanes[inputIndex];
          llvm::SmallVector<mlir::Type, 6> laneTypes =
              RuntimeBundleLowerer::generatorArgumentPhysicalTypes(lane);
          std::int64_t begin = static_cast<std::int64_t>(inputTypes.size());
          generatorBorrowedArgLanes.push_back(builder.getDictionaryAttr({
              builder.getNamedAttr("contract",
                                   builder.getStringAttr(lane.contract)),
              builder.getNamedAttr("begin", builder.getI64IntegerAttr(begin)),
              builder.getNamedAttr(
                  "size", builder.getI64IntegerAttr(
                              static_cast<std::int64_t>(laneTypes.size()))),
          }));
          inputTypes.append(laneTypes.begin(), laneTypes.end());
          continue;
        }
        // Generator frame lanes carry their object-family span; the frame's
        // token transfers INTO the clone (ly.ownership.transfer_args), which
        // the continuation claims re-root as tracked resources.
        if (generatorArgInfo && inputIndex >= generatorControlCount &&
            inputIndex - generatorControlCount <
                generatorArgInfo->frameLanes.size()) {
          const GeneratorResumeLane &lane =
              generatorArgInfo->frameLanes[inputIndex - generatorControlCount];
          if (!lane.isControl()) {
            std::int64_t begin = static_cast<std::int64_t>(inputTypes.size());
            llvm::SmallVector<mlir::Type, 6> laneTypes =
                RuntimeBundleLowerer::generatorLanePhysicalTypes(lane);
            // The transfer is anchored at the lane's header (group offset);
            // the remaining parts are the entity's interior views.
            if (lane.physicalCount > 0)
              generatorTransferArgs.push_back(begin);
            generatorResumeArgLanes.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("contract",
                                     builder.getStringAttr(lane.contract)),
                builder.getNamedAttr("begin",
                                     builder.getI64IntegerAttr(begin)),
                builder.getNamedAttr(
                    "size", builder.getI64IntegerAttr(
                                static_cast<std::int64_t>(laneTypes.size()))),
            }));
            inputTypes.append(laneTypes.begin(), laneTypes.end());
            continue;
          }
        }
        if (runtimeContractName(logicalType) != "builtins.int") {
          // Naming the contract and the reason, not just the expected type: the
          // parameter this rejects is usually one the user never wrote (a
          // generator's captured receiver, or a closure cell), so "must be
          // builtins.int" alone points at nothing they can act on.
          std::string contract = runtimeContractName(logicalType);
          mlir::InFlightDiagnostic diagnostic = function.emitError();
          if (generatorArgInfo)
            diagnostic << "a generator cannot carry a value of contract '"
                       << (contract.empty() ? "<none>" : contract)
                       << "' across a suspension yet: only builtins.int and "
                          "manifest contracts with a rank-1 physical shape "
                          "have a resume lane, and a user class has neither. "
                          "Read the value into an int local before the first "
                          "yield, or move the generator out of the class and "
                          "pass the fields it needs";
          else
            diagnostic << "primitive i64 callable clone parameter must be "
                          "builtins.int, but this one has contract '"
                       << (contract.empty() ? "<none>" : contract) << "'";
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        // ⛔ The raw word ONLY -- no validity bit. Generator resume clones
        // keep the pair, because a resumed frame's lane really can arrive
        // stale; a plain clone is entered under the caller's branch, so a
        // second copy of a fact already established would just invite the
        // body to branch on it (see seedPrimitiveI64CallableEntryArgumentBundles).
        if (generatorArgInfo)
          RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(logicalType,
                                                                inputTypes);
        else
          inputTypes.push_back(mlir::IntegerType::get(context, 64));
      }
      if (!generatorTransferArgs.empty())
        function->setAttr(ownership::kTransferArgsAttr,
                          builder.getI64ArrayAttr(generatorTransferArgs));
      if (!generatorResumeArgLanes.empty())
        function->setAttr("ly.generator.resume_args",
                          builder.getArrayAttr(generatorResumeArgLanes));
      if (!generatorBorrowedArgLanes.empty())
        function->setAttr("ly.generator.borrowed_args",
                          builder.getArrayAttr(generatorBorrowedArgLanes));
      llvm::SmallVector<std::int64_t, 8> generatorHeaderArgs =
          generatorTransferArgs;
      // Generator resume clones widen the yielded-value result lane to an
      // object-family span; every other lane stays on the primitive pair.
      // The lane's ownership crosses the suspension boundary through the
      // materialized owned-results contract (consumed by the affine
      // verifier's generator-frame rule and the refcount inserter).
      GeneratorResumeInfo *generatorInfo =
          RuntimeBundleLowerer::generatorResumeInfoForClone(function);
      if (callable.getResultTypes().empty() ||
          !llvm::all_of(
              llvm::enumerate(callable.getResultTypes()), [&](auto indexed) {
                if (generatorInfo &&
                    (indexed.index() == 2 ||
                     (indexed.index() >= 5 &&
                      indexed.index() - 5 < generatorInfo->frameLanes.size())))
                  return true;
                return runtimeContractName(indexed.value()) == "builtins.int";
              })) {
        function.emitError()
            << "primitive i64 callable clone results must be builtins.int";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      llvm::SmallVector<mlir::Type, 8> resultTypes;
      llvm::SmallVector<std::int64_t, 2> generatorOwnedOffsets;
      llvm::SmallVector<mlir::Attribute, 2> generatorOwnedContracts;
      llvm::SmallVector<mlir::Attribute, 2> generatorSuspendLanes;
      for (auto [resultIndex, resultType] :
           llvm::enumerate(callable.getResultTypes())) {
        const GeneratorResumeLane *suspendLane = nullptr;
        if (generatorInfo && resultIndex == 2 &&
            !generatorInfo->valueLane.isControl())
          suspendLane = &generatorInfo->valueLane;
        else if (generatorInfo && resultIndex >= 5 &&
                 resultIndex - 5 < generatorInfo->frameLanes.size() &&
                 !generatorInfo->frameLanes[resultIndex - 5].isControl())
          suspendLane = &generatorInfo->frameLanes[resultIndex - 5];
        if (suspendLane) {
          const GeneratorResumeLane &lane = *suspendLane;
          llvm::SmallVector<mlir::Type, 6> laneTypes =
              RuntimeBundleLowerer::generatorLanePhysicalTypes(lane);
          std::int64_t begin = static_cast<std::int64_t>(resultTypes.size());
          if (!lane.isNone) {
            generatorOwnedOffsets.push_back(begin);
            generatorOwnedContracts.push_back(
                builder.getStringAttr(lane.contract));
          }
          generatorSuspendLanes.push_back(builder.getDictionaryAttr({
              builder.getNamedAttr("contract",
                                   builder.getStringAttr(lane.contract)),
              builder.getNamedAttr("begin", builder.getI64IntegerAttr(begin)),
              builder.getNamedAttr(
                  "size", builder.getI64IntegerAttr(
                              static_cast<std::int64_t>(laneTypes.size()))),
          }));
          resultTypes.append(laneTypes.begin(), laneTypes.end());
          continue;
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(resultType,
                                                              resultTypes);
      }
      if (!generatorOwnedOffsets.empty()) {
        function->setAttr(
            ownership::kOwnedResultsAttr,
            mlir::DenseI64ArrayAttr::get(context, generatorOwnedOffsets));
        function->setAttr(ownership::kOwnedResultContractsAttr,
                          builder.getArrayAttr(generatorOwnedContracts));
      }
      if (!generatorSuspendLanes.empty())
        function->setAttr("ly.generator.suspend_lanes",
                          builder.getArrayAttr(generatorSuspendLanes));
      if (!function.isDeclaration()) {
        mlir::LogicalResult seeded =
            generatorInfo
                ? seedGeneratorResumeCloneEntry(function, logicalInputTypes,
                                                *generatorInfo)
                : seedPrimitiveI64CallableEntryArgumentBundles(
                      function, logicalInputTypes);
        if (mlir::failed(seeded)) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
      }
      function.setFunctionType(
          mlir::FunctionType::get(context, inputTypes, resultTypes));
      // Header arg markers for the transferred frame lanes (the transfer
      // verifier requires the anchor to be an object header). Set after the
      // final function type so the indices are in range.
      for (std::int64_t headerIndex : generatorHeaderArgs)
        function.setArgAttr(static_cast<unsigned>(headerIndex),
                            ownership::kObjectHeaderAttr,
                            builder.getUnitAttr());
      return mlir::WalkResult::advance();
    }

    llvm::SmallVector<mlir::Type, 8> abiInputTypes = logicalInputTypes;
    auto protocolEvidence =
        callableProtocolArgumentABIs.find(function.getSymName());
    if (protocolEvidence != callableProtocolArgumentABIs.end()) {
      llvm::SmallVector<mlir::Type, 8> &evidence = protocolEvidence->second;
      for (auto [index, type] : llvm::enumerate(evidence))
        if (index < abiInputTypes.size() && type)
          abiInputTypes[index] = type;
    }

    llvm::SmallVector<mlir::Type, 8> inputTypes;
    for (mlir::Type inputType : abiInputTypes) {
      if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
              function, inputType, inputTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                            inputTypes);
    }
    const CallableArgumentEvidenceABI *argumentEvidence = nullptr;
    auto argumentEvidenceIt =
        callableArgumentEvidenceABIs.find(function.getSymName());
    if (argumentEvidenceIt != callableArgumentEvidenceABIs.end()) {
      argumentEvidence = &argumentEvidenceIt->second;
      for (const RuntimeArgumentEvidenceSet &evidenceSet :
           argumentEvidence->logicalArguments) {
        for (const RuntimeArgumentEvidence &evidence :
             evidenceSet.alternatives) {
          for (mlir::Type inputType : evidence.closureValueTypes) {
            if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                    function, inputType, inputTypes))) {
              result = mlir::failure();
              return mlir::WalkResult::interrupt();
            }
            RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                                  inputTypes);
          }
          for (mlir::Type inputType : evidence.coroutineSourceTypes) {
            if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                    function, inputType, inputTypes))) {
              result = mlir::failure();
              return mlir::WalkResult::interrupt();
            }
            RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                                  inputTypes);
          }
        }
      }
    }
    const CallableAggregateEvidenceABI *aggregateEvidence = nullptr;
    auto evidence = callableAggregateEvidenceABIs.find(function.getSymName());
    if (evidence != callableAggregateEvidenceABIs.end()) {
      aggregateEvidence = &evidence->second;
      for (mlir::Type inputType : aggregateEvidence->varargElementTypes) {
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, inputType, inputTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                              inputTypes);
      }
      for (mlir::Type inputType : aggregateEvidence->kwargValueTypes) {
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, inputType, inputTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                              inputTypes);
        if (aggregateEvidence->kwargIsFull)
          inputTypes.push_back(builder.getI1Type());
      }
    }

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    llvm::SmallVector<std::int64_t, 4> ownedResultOffsets;
    llvm::SmallVector<mlir::Attribute, 4> ownedResultContracts;
    auto returnedCoroutine =
        returnedCoroutineSummaries.find(function.getSymName());
    auto returnedObjectEvidence =
        returnedObjectEvidenceSummaries.find(function.getSymName());
    auto returnedStaticObject =
        returnedStaticObjectSummaries.find(function.getSymName());
    for (auto [logicalResultIndex, resultType] :
         llvm::enumerate(callable.getResultTypes())) {
      mlir::Type abiResultType = resultType;
      if (returnedCoroutine != returnedCoroutineSummaries.end() &&
          isCoroutineLikeResultType(resultType)) {
        if (mlir::func::FuncOp target =
                module.lookupSymbol<mlir::func::FuncOp>(
                    returnedCoroutine->second.target)) {
          if (mlir::Type concrete =
                  concreteCoroutineTypeForTarget(context, target))
            abiResultType = concrete;
        }
      }
      bool protocolPrimaryOwnsResult = false;
      if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(
              resultType))
        protocolPrimaryOwnsResult =
            runtimeShapeContractName(resultType) == "builtins.object" &&
            ((returnedStaticObject != returnedStaticObjectSummaries.end() &&
              returnedStaticObject->second.resultIndex == logicalResultIndex) ||
             protocol.getProtocolName() == "Generator");
      if (protocolPrimaryOwnsResult) {
        ownedResultOffsets.push_back(
            static_cast<std::int64_t>(resultTypes.size()));
        ownedResultContracts.push_back(builder.getStringAttr("builtins.object"));
      }
      // ⭐ A UNION OWNS ITS OWN MEMBER LANES, HOWEVER MANY OWN ONE. The layout
      // already lays each member out after the tag, so the lanes to name are
      // there -- what was missing was naming them.
      //
      // ⛔ Why NOT extend the static-object summary to a list of contracts
      // instead, which is the shape the attribute already takes: that appends
      // a DUPLICATE lane per member, and the caller would then need one
      // conditionally owned bundle per duplicate while `RuntimeBundle` has a
      // single `boxedObject` slot. The union's own lanes need no second bundle
      // -- `collectTypedResourceGroups` already walks them and already stamps
      // each with its `OwnershipCondition{tag, memberIndex}`, so the whole
      // conditional machinery is reached by declaring the offsets.
      //
      // ⛔ AND ONE OWNING MEMBER IS NOT A DIFFERENT CASE. It used to be: with
      // one, this was skipped and the static-object summary below appended a
      // second copy of that member and marked THAT owned -- unconditionally,
      // because the summary has no tag. `T | None` has exactly one owning
      // member, so EVERY optional took that path: `pick() -> "Node | None"`
      // came out as three lanes with the owned one at offset 2, which is the
      // duplicate rather than the union's own, and the tag never reached the
      // resource. That is why an optional carried across a loop's back edge
      // reported "owned resource ... without release" where a two-member union
      // reported "conditionally owned ... without tag-conditioned release":
      // one obligation was being tracked as if it could not be absent.
      if (auto unionResult =
              mlir::dyn_cast_if_present<py::UnionType>(abiResultType)) {
        llvm::SmallVector<std::pair<std::int64_t, std::string>, 2> memberLanes;
        std::int64_t memberOffset =
            static_cast<std::int64_t>(resultTypes.size()) + 1;
        bool laid = true;
        for (mlir::Type member : unionResult.getMemberTypes()) {
          mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
              RuntimeBundleLowerer::runtimeValueTypesFor(
                  function, member, "union result member lane");
          if (mlir::failed(memberTypes)) {
            laid = false;
            break;
          }
          if (memberOwnsALane(member))
            memberLanes.emplace_back(memberOffset, runtimeContractName(member));
          memberOffset += static_cast<std::int64_t>(memberTypes->size());
        }
        if (laid && !memberLanes.empty()) {
          for (const auto &[offset, contract] : memberLanes) {
            ownedResultOffsets.push_back(offset);
            ownedResultContracts.push_back(builder.getStringAttr(contract));
          }
        }
      }
      if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
              function, abiResultType, resultTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(abiResultType,
                                                            resultTypes);
      if (returnedStaticObject != returnedStaticObjectSummaries.end() &&
          returnedStaticObject->second.resultIndex == logicalResultIndex) {
        mlir::Type objectContract =
            returnedStaticObject->second.objectContract;
        std::string objectContractName = runtimeContractName(objectContract);
        if (objectContractName.empty()) {
          result = function.emitError()
                   << "static returned object evidence has no runtime "
                      "contract: "
                   << objectContract;
          return mlir::WalkResult::interrupt();
        }
        ownedResultOffsets.push_back(
            static_cast<std::int64_t>(resultTypes.size()));
        ownedResultContracts.push_back(
            builder.getStringAttr(objectContractName));
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, objectContract, resultTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(objectContract,
                                                              resultTypes);
      }
      if (returnedCoroutine != returnedCoroutineSummaries.end() &&
          (isCoroutineLikeResultType(resultType) ||
           isAwaitIteratorLikeResultType(resultType))) {
        for (mlir::Type sourceType :
             returnedCoroutine->second.sourceContracts) {
          if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                  function, sourceType, resultTypes))) {
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
          RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(sourceType,
                                                                resultTypes);
        }
      }
      if (returnedObjectEvidence == returnedObjectEvidenceSummaries.end() ||
          returnedObjectEvidence->second.resultIndex != logicalResultIndex)
        continue;
      for (const ReturnedObjectEvidenceSlot &slot :
           returnedObjectEvidence->second.slots) {
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, slot.sourceContract, resultTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(
            slot.sourceContract, resultTypes);
      }
    }
    // Returned-closure LOCAL captures ride out as trailing owned result
    // lanes (the nonlocal cell escaping with its closure). Their layout is
    // fixed per function, which the summary pass guarantees by requiring a
    // single alternative for lane captures.
    if (auto returnedCallable =
            returnedCallableSummaries.find(function.getSymName());
        returnedCallable != returnedCallableSummaries.end() &&
        returnedCallable->second.alternatives.size() == 1 &&
        returnedCallable->second.alternatives.front().hasLaneCaptures()) {
      if (returnedObjectEvidence != returnedObjectEvidenceSummaries.end()) {
        function.emitError()
            << "returned closure lane captures cannot combine with returned "
               "object evidence yet";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      for (const ReturnedCallableCapture &capture :
           returnedCallable->second.alternatives.front().captures) {
        if (!capture.laneContract)
          continue;
        std::string laneContractName =
            runtimeContractName(capture.laneContract);
        if (laneContractName.empty()) {
          function.emitError() << "returned closure capture lane has no "
                                  "runtime contract";
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        ownedResultOffsets.push_back(
            static_cast<std::int64_t>(resultTypes.size()));
        ownedResultContracts.push_back(
            builder.getStringAttr(laneContractName));
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, capture.laneContract, resultTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
      }
    }
    if (!function.isDeclaration()) {
      if (mlir::failed(seedCallableEntryArgumentBundles(
              function, logicalInputTypes, abiInputTypes, aggregateEvidence))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
    }
    function.setFunctionType(
        mlir::FunctionType::get(context, inputTypes, resultTypes));
    if (!ownedResultOffsets.empty())
      function->setAttr(
          ownership::kOwnedResultsAttr,
          mlir::DenseI64ArrayAttr::get(context, ownedResultOffsets));
    if (!ownedResultContracts.empty())
      function->setAttr(ownership::kOwnedResultContractsAttr,
                        builder.getArrayAttr(ownedResultContracts));
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::LogicalResult RuntimeBundleLowerer::seedCallableEntryArgumentBundles(
    mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes,
    mlir::ArrayRef<mlir::Type> abiTypes,
    const CallableAggregateEvidenceABI *aggregateEvidence) {
  if (function.isDeclaration())
    return mlir::success();
  mlir::Block &entry = function.getBody().front();
  if (entry.getNumArguments() != logicalTypes.size())
    return function.emitError()
           << "callable function entry argument count does not match "
              "callable_type";
  if (abiTypes.size() != logicalTypes.size())
    return function.emitError()
           << "callable ABI type count does not match callable_type";

  auto seedHiddenPrimitiveI64Evidence =
      [&](mlir::Type abiType, RuntimeBundle &bundle,
          mlir::Location loc) -> mlir::LogicalResult {
    if (!RuntimeBundleLowerer::hasPrimitiveI64ABI(abiType))
      return mlir::success();
    mlir::BlockArgument raw =
        entry.addArgument(mlir::IntegerType::get(context, 64), loc);
    mlir::BlockArgument valid =
        entry.addArgument(mlir::IntegerType::get(context, 1), loc);
    bundle.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
    return mlir::success();
  };

  unsigned logicalArgCount = entry.getNumArguments();
  for (auto [index, logicalType] : llvm::enumerate(logicalTypes)) {
    mlir::Type abiType = abiTypes[index];
    mlir::BlockArgument logicalArg = entry.getArgument(index);
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(function, abiType,
                                                   "callable parameter ABI");
    if (mlir::failed(valueTypes))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> physicalArgs;
    for (mlir::Type physicalType : *valueTypes)
      physicalArgs.push_back(
          entry.addArgument(physicalType, logicalArg.getLoc()));

    // A `type[X]` parameter took no ABI input, so the bundle is rebuilt from
    // the parameter's own type. This is the fourth layer an earlier attempt
    // stopped at -- `lowerTypeObject` makes a TypeObject bundle for a
    // `py.type.object` op and nothing made one for an entry argument, so an
    // attr.set on a class-valued parameter saw a bundle with no kind at all.
    if (auto typeType = mlir::dyn_cast<py::TypeType>(abiType)) {
      valueBundles[logicalArg] =
          RuntimeBundle::typeObject(abiType, typeType.getInstanceType());
      continue;
    }
    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            function, abiType, physicalArgs, bundle,
            /*ownsObject=*/false)))
      return mlir::failure();
    if (mlir::failed(seedHiddenPrimitiveI64Evidence(abiType, bundle,
                                                    logicalArg.getLoc())))
      return mlir::failure();
    valueBundles[logicalArg] = std::move(bundle);
  }
  auto appendHiddenObject =
      [&](mlir::Type logicalType) -> mlir::FailureOr<RuntimeValue> {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(
            function, logicalType, "callable aggregate evidence ABI");
    if (mlir::failed(valueTypes))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> physicalArgs;
    for (mlir::Type physicalType : *valueTypes)
      physicalArgs.push_back(
          entry.addArgument(physicalType, function.getLoc()));

    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            function, logicalType, physicalArgs, bundle,
            /*ownsObject=*/false)))
      return mlir::failure();
    if (mlir::failed(seedHiddenPrimitiveI64Evidence(logicalType, bundle,
                                                    function.getLoc())))
      return mlir::failure();
    return bundle.objectValue;
  };

  auto argumentEvidence =
      callableArgumentEvidenceABIs.find(function.getSymName());
  if (argumentEvidence != callableArgumentEvidenceABIs.end()) {
    for (auto [logicalIndex, evidenceSet] :
         llvm::enumerate(argumentEvidence->second.logicalArguments)) {
      if (logicalIndex >= logicalTypes.size() || evidenceSet.empty())
        continue;
      mlir::BlockArgument logicalArg = entry.getArgument(logicalIndex);
      auto found = valueBundles.find(logicalArg);
      if (found == valueBundles.end() ||
          found->second.kind != RuntimeBundle::Kind::Object)
        return function.emitError()
               << "argument evidence logical argument has no object runtime "
                  "bundle";
      RuntimeBundle &bundle = found->second;
      if (evidenceSet.alternatives.size() == 1) {
        const RuntimeArgumentEvidence &evidence =
            evidenceSet.alternatives.front();
        if (!evidence.functionTarget.empty())
          bundle.functionTarget = evidence.functionTarget;
        if (!evidence.coroutineTarget.empty())
          bundle.coroutineTarget = evidence.coroutineTarget;
      }
      for (const RuntimeArgumentEvidence &evidence : evidenceSet.alternatives) {
        if (!evidence.functionTarget.empty() ||
            !evidence.closureValueTypes.empty()) {
          RuntimeCallableAlternative alternative;
          alternative.functionTarget = evidence.functionTarget;
          for (mlir::Type closureType : evidence.closureValueTypes) {
            mlir::FailureOr<RuntimeValue> closure =
                appendHiddenObject(closureType);
            if (mlir::failed(closure))
              return mlir::failure();
            alternative.closureValues.push_back(*closure);
          }
          if (evidenceSet.alternatives.size() == 1)
            bundle.closureValues = alternative.closureValues;
          bundle.callableAlternatives.push_back(std::move(alternative));
        }
        if (!evidence.coroutineTarget.empty()) {
          llvm::SmallVector<RuntimeValue, 4> coroutineSources;
          llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 4>
              coroutineSourceBundles;
          for (mlir::Type sourceType : evidence.coroutineSourceTypes) {
            mlir::FailureOr<RuntimeValue> source =
                appendHiddenObject(sourceType);
            if (mlir::failed(source))
              return mlir::failure();
            coroutineSources.push_back(*source);
            coroutineSourceBundles.push_back(std::make_shared<RuntimeBundle>(
                RuntimeBundle::object(source->contract, source->values)));
          }
          if (evidenceSet.alternatives.size() == 1) {
            bundle.coroutineSources = std::move(coroutineSources);
            bundle.coroutineSourceBundles =
                std::move(coroutineSourceBundles);
          }
        }
      }
    }
  }

  if (aggregateEvidence && aggregateEvidence->varargLogicalIndex) {
    unsigned logicalIndex = *aggregateEvidence->varargLogicalIndex;
    if (logicalIndex >= logicalTypes.size())
      return function.emitError()
             << "vararg aggregate evidence ABI references logical argument "
             << logicalIndex << ", but function has only "
             << logicalTypes.size() << " logical inputs";
    mlir::BlockArgument logicalArg = entry.getArgument(logicalIndex);
    auto found = valueBundles.find(logicalArg);
    if (found == valueBundles.end() ||
        found->second.kind != RuntimeBundle::Kind::Object)
      return function.emitError()
             << "vararg aggregate evidence logical argument has no object "
                "runtime bundle";
    RuntimeBundle &bundle = found->second;
    bundle.sequenceIndices = aggregateEvidence->varargElementIndices;
    for (mlir::Type elementType : aggregateEvidence->varargElementTypes) {
      mlir::FailureOr<RuntimeValue> element = appendHiddenObject(elementType);
      if (mlir::failed(element))
        return mlir::failure();
      bundle.sequenceElements.push_back(*element);
    }
  }

  if (aggregateEvidence && aggregateEvidence->kwargLogicalIndex) {
    unsigned logicalIndex = *aggregateEvidence->kwargLogicalIndex;
    if (logicalIndex >= logicalTypes.size())
      return function.emitError()
             << "kwarg aggregate evidence ABI references logical argument "
             << logicalIndex << ", but function has only "
             << logicalTypes.size() << " logical inputs";
    mlir::BlockArgument logicalArg = entry.getArgument(logicalIndex);
    auto found = valueBundles.find(logicalArg);
    if (found == valueBundles.end() ||
        found->second.kind != RuntimeBundle::Kind::Object)
      return function.emitError()
             << "kwarg aggregate evidence logical argument has no object "
                "runtime bundle";
    RuntimeBundle &bundle = found->second;
    bundle.mappingKeys = aggregateEvidence->kwargKeys;
    for (mlir::Type valueType : aggregateEvidence->kwargValueTypes) {
      mlir::FailureOr<RuntimeValue> value = appendHiddenObject(valueType);
      if (mlir::failed(value))
        return mlir::failure();
      bundle.mappingValues.push_back(*value);
      if (aggregateEvidence->kwargIsFull) {
        mlir::BlockArgument present =
            entry.addArgument(builder.getI1Type(), function.getLoc());
        bundle.mappingPresent.push_back(present);
      }
    }
  }
  callableLogicalEntryArgCounts.push_back(
      CallableLogicalEntryArgs{function, logicalArgCount});
  return mlir::success();
}

} // namespace py::lowering
