#include "Runtime/Core/Lowerer.h"

#include "mlir/IR/Dominance.h"

#include "Runtime/ABI/CollectionPayload.h"
#include "Runtime/ABI/BoxLayout.h"
#include "Runtime/Evidence/Callable.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace py::lowering {
namespace {

using callable_evidence::integerLiteralFromValue;

bool isEvidenceCollection(llvm::StringRef contract) {
  return contract == "builtins.list" || contract == "builtins.tuple" ||
         contract == "builtins.dict";
}

std::optional<mlir::Value> knownEvidenceEquality(mlir::Operation *op,
                                                 mlir::OpBuilder &builder,
                                                 const RuntimeBundle &lhs,
                                                 const RuntimeBundle &rhs) {
  if (lhs.kind != RuntimeBundle::Kind::Object ||
      rhs.kind != RuntimeBundle::Kind::Object)
    return std::nullopt;
  mlir::Location loc = op->getLoc();
  if (lhs.literalText && rhs.literalText)
    return constantBool(builder, loc, *lhs.literalText == *rhs.literalText);
  if (lhs.contractName() == "builtins.int" &&
      rhs.contractName() == "builtins.int" && lhs.primitiveI64 &&
      rhs.primitiveI64 && lhs.primitiveI64->value && rhs.primitiveI64->value &&
      lhs.primitiveI64->valid && rhs.primitiveI64->valid) {
    mlir::Value sameValue = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, lhs.primitiveI64->value,
        rhs.primitiveI64->value);
    mlir::Value bothValid = mlir::arith::AndIOp::create(
        builder, loc, lhs.primitiveI64->valid, rhs.primitiveI64->valid);
    return mlir::arith::AndIOp::create(builder, loc, bothValid, sameValue)
        .getResult();
  }
  if (sameRuntimeValueIdentity(lhs.objectValue, rhs.objectValue))
    return constantBool(builder, loc, true);
  return std::nullopt;
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::pinProbeOperandLiveness(
    mlir::Operation *op, const RuntimeBundle &payload,
    const RuntimeBundle *source) {
  if (payload.objectValue.ownership == ownership::OwnershipKind::Own) {
    // Why the release is gated on aliasing: an owned payload that ALIASES the
    // py-level operand is not a probe-local temporary — the ownership pass
    // places that value's release after its real last use, and releasing it
    // here frees a key the program may still read (a dict lookup would
    // corrupt its own key variable). Only boxes freshly created by
    // materializePayloadObjectBundle are invisible to the ownership pass and
    // must be consumed at the probe.
    bool aliasesSource = false;
    if (source) {
      const RuntimeBundle *sourceConcrete =
          RuntimeBundleLowerer::concreteObjectForOwnership(*source);
      aliasesSource = sourceConcrete &&
                      llvm::equal(sourceConcrete->physicalValues(),
                                  payload.physicalValues());
    }
    if (!aliasesSource)
      return RuntimeBundleLowerer::releaseAggregateSlot(op, payload,
                                                        "probe.operand");
  }
  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(payload);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return mlir::success();
  for (llvm::StringRef candidate :
       {llvm::StringRef("__len__"), llvm::StringRef("__hash__"),
        llvm::StringRef("__int__"), llvm::StringRef("__bool__")}) {
    for (RuntimeSymbol symbol :
         manifest.methodCandidates(concrete->contractName(), candidate)) {
      mlir::FunctionType type = symbol.function.getFunctionType();
      llvm::ArrayRef<mlir::Value> physicals = concrete->physicalValues();
      if (type.getNumInputs() != physicals.size())
        continue;
      bool matches = true;
      for (auto [input, physical] : llvm::zip(type.getInputs(), physicals))
        if (physical.getType() != input) {
          matches = false;
          break;
        }
      if (!matches)
        continue;
      RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), symbol,
          llvm::SmallVector<mlir::Value, 4>(physicals.begin(),
                                            physicals.end()));
      return mlir::success();
    }
  }
  // No conforming neutral manifest use (source classes have none): a real
  // call is still required as the ownership pass's liveness anchor — without
  // one the pass places the payload's release BEFORE the raw-word probe
  // call and the probe reads freed storage. Synthesize an empty private
  // per-shape function and call it with the payload's physical values.
  {
    llvm::SmallVector<mlir::Type, 4> inputTypes;
    for (mlir::Value physical : concrete->physicalValues())
      inputTypes.push_back(physical.getType());
    std::string sanitized = concrete->contractName();
    for (char &c : sanitized)
      if (!llvm::isAlnum(c))
        c = '_';
    std::string pinName =
        ("__ly_probe_pin$" + sanitized + "$" + llvm::Twine(inputTypes.size()))
            .str();
    auto pinFn = module.lookupSymbol<mlir::func::FuncOp>(pinName);
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (!pinFn) {
      mlir::OpBuilder moduleBuilder(context);
      moduleBuilder.setInsertionPointToEnd(module.getBody());
      pinFn = mlir::func::FuncOp::create(
          moduleBuilder, op->getLoc(), pinName,
          moduleBuilder.getFunctionType(inputTypes, {}));
      pinFn.setPrivate();
      mlir::Block *body = pinFn.addEntryBlock();
      moduleBuilder.setInsertionPointToStart(body);
      mlir::func::ReturnOp::create(moduleBuilder, op->getLoc());
    }
    if (pinFn.getFunctionType() ==
        mlir::FunctionType::get(context, inputTypes, {}))
      mlir::func::CallOp::create(
          builder, op->getLoc(), pinFn,
          llvm::SmallVector<mlir::Value, 4>(concrete->physicalValues().begin(),
                                            concrete->physicalValues().end()));
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerReceiverMethodResult(
    mlir::Operation *op, mlir::Value receiverValue, mlir::Value resultValue,
    llvm::StringRef missingSubject, llvm::StringRef methodName,
    bool preferManifestObjectResult) {
  const RuntimeBundle *receiver =
      RuntimeBundleLowerer::bundleFor(receiverValue);
  if (!receiver)
    return op->emitError() << missingSubject
                           << " has no lowered runtime bundle";

  llvm::SmallVector<const RuntimeBundle *, 1> sources{receiver};
  RuntimeBundle methodReceiver = *receiver;
  if (mlir::isa<py::ProtocolType>(receiverValue.getType()) &&
      mlir::isa<py::ProtocolType>(receiver->contract)) {
    methodReceiver.contract = receiverValue.getType();
    methodReceiver.objectValue.contract = receiverValue.getType();
  }
  if (py::ClassOp classOp =
          RuntimeBundleLowerer::classForContract(methodReceiver.contract)) {
    if (std::optional<std::string> methodSymbol =
            RuntimeBundleLowerer::classMethodSymbol(classOp, methodName)) {
      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(*methodSymbol);
      if (!target)
        return op->emitError()
               << "source class method @" << *methodSymbol << " is not defined";
      if (target->hasAttr("ly.async.body_result")) {
        RuntimeBundle result;
        if (mlir::failed(
                RuntimeBundleLowerer::emitAsyncFunctionTargetCallResult(
                    op, resultValue, target, *methodSymbol, sources, result)))
          return mlir::failure();
        valueBundles[resultValue] = std::move(result);
        erase.push_back(op);
        return mlir::success();
      }
    }
  }
  if (mlir::failed(lowerManifestMethodResult(
          op, resultValue, methodReceiver, methodName, sources,
          /*allowUnusedSources=*/false, preferManifestObjectResult)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

// ⭐ A SOURCE exception class has no manifest methods of its own, and the class
// name it must answer is not its static one anyway: the instance header carries
// the id, and the ancestor's `__class_name__` reads it. This is the same
// retyping the print path does for `__str__` -- for the same reason and with
// the same effect on the answer, which is that a user exception keeps its own
// name.
mlir::LogicalResult RuntimeBundleLowerer::lowerClassName(py::ClassNameOp op) {
  const RuntimeBundle *receiver = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!receiver)
    return op.emitError() << "class name operand has no lowered runtime bundle";
  RuntimeBundle methodReceiver = *receiver;
  if (!manifest.method(methodReceiver.contractName(), "__class_name__")) {
    if (std::optional<std::string> ancestor =
            RuntimeBundleLowerer::exceptionAncestorContractFor(
                methodReceiver.contract)) {
      mlir::Type ancestorType = runtimeContractType(context, *ancestor);
      methodReceiver.contract = ancestorType;
      methodReceiver.objectValue.contract = ancestorType;
    } else if (!manifest.primitive(methodReceiver.contractName(), "raise")) {
      // ⭐ NOT AN EXCEPTION: read the class id out of the instance header --
      // word 1, the word `isinstance` reads -- and look the name up in the
      // per-program table. This is the only answer available for a value whose
      // static class has subclasses, which is exactly when the emitter cannot
      // fold `type(x).__name__`.
      std::optional<RuntimeSymbol> lookup =
          manifest.primitive("builtins.object", "class_name_from_id");
      if (!lookup)
        return op.emitError() << "runtime manifest has no builtins.object "
                                 "class_name_from_id primitive";
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();
      // ⛔ NOT word 1 unconditionally. A BOX carries the payload's word 1, and
      // for an exception that is the shared LAYOUT -- `type(e).__name__` over a
      // type-erased ValueError answered "BaseException". The shared read knows
      // where each kind keeps its exact class.
      mlir::FailureOr<mlir::Value> exact =
          RuntimeBundleLowerer::exactRuntimeClassId(op, methodReceiver);
      if (mlir::failed(exact))
        return mlir::failure();
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          loc, *lookup, mlir::ValueRange{*exact});
      RuntimeBundle named;
      if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
              op, runtimeContractType(context, "builtins.str"), call, named)))
        return mlir::failure();
      valueBundles[op.getResult()] = std::move(named);
      erase.push_back(op);
      return mlir::success();
    }
    // ⛔ A manifest EXCEPTION contract (the taxonomy) keeps the manifest path:
    // it has no exception ANCESTOR to redirect to because it is one, and its
    // header carries the class id in a different word than a source instance's
    // -- reading word 1 there looked up an id nothing declares and printed
    // "object" for a caught ValueError.
  }
  llvm::SmallVector<const RuntimeBundle *, 1> sources{&methodReceiver};
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), methodReceiver, "__class_name__", sources,
          /*allowUnusedSources=*/false, /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBool(py::BoolOp op) {
  const RuntimeBundle *input = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!input)
    return op.emitError() << "bool input has no lowered runtime bundle";
  llvm::ArrayRef<mlir::Value> inputValues = input->physicalValues();
  if (input->contractName() == "builtins.bool" && inputValues.size() == 1 &&
      inputValues.front().getType().isInteger(1)) {
    op.getResult().replaceAllUsesWith(inputValues.front());
    erase.push_back(op);
    return mlir::success();
  }
  if (input->kind == RuntimeBundle::Kind::Object &&
      isEvidenceCollection(input->contractName())) {
    builder.setInsertionPoint(op);
    mlir::FailureOr<mlir::Value> length =
        RuntimeBundleLowerer::loadContainerLength(op, *input, "bool");
    if (mlir::failed(length))
      return mlir::failure();
    mlir::Value zero =
        mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64);
    mlir::Value nonEmpty = mlir::arith::CmpIOp::create(
        builder, op.getLoc(), mlir::arith::CmpIPredicate::ne, *length, zero);
    op.getResult().replaceAllUsesWith(nonEmpty);
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{input};
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__bool__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestI1MethodResult(op, op.getResult(), *input,
                                               *methodName, sources,
                                               /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerLen(py::LenOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__len__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getInput(), op.getResult(), "len input", *methodName);
}

// A raw i64 for a runtime-normalized sequence index: constant literal,
// primitive-int evidence, or an unbox call, in that preference order (the
// same ladder the runtime-mode __getitem__ walks). The caller positions the
// builder.
mlir::FailureOr<mlir::Value> RuntimeBundleLowerer::rawSequenceIndexValue(
    mlir::Operation *op, mlir::Value indexValue, const RuntimeBundle &index) {
  // `indexValue` is optional: bound-method callers pass the index through a
  // pack, which has no per-argument SSA value when it is starred.
  if (indexValue)
    if (std::optional<std::int64_t> literal =
            integerLiteralFromValue(indexValue))
      return mlir::arith::ConstantIntOp::create(builder, op->getLoc(), *literal,
                                                64)
          .getResult();
  if (primitiveI64LaneKnownValid(index.primitiveI64) &&
      index.primitiveI64->value.getType().isInteger(64))
    return index.primitiveI64->value;
  std::optional<RuntimeSymbol> unbox =
      manifest.primitive(index.contractName(), "unbox.i64");
  if (unbox &&
      unbox->function.getNumArguments() == index.physicalValues().size()) {
    mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
        op->getLoc(), *unbox, index.physicalValues());
    return call.getResult(0);
  }
  // No boxed payload to fall back to (primitive-i64 clone lanes carry only
  // the (value, valid) pair): the lane is the sole carrier.
  if (index.primitiveI64 && index.primitiveI64->value &&
      index.primitiveI64->value.getType().isInteger(64))
    return index.primitiveI64->value;
  return op->emitError()
         << "sequence index of contract " << index.contractName()
         << " has no statically unboxable integer value";
}

// True when `op` sits in a DIFFERENT block than the one defining the
// container's physical storage. Evidence recorded in the defining block names
// SSA values that join-dominated uses cannot reference, and a merge keeps only
// one predecessor's version — so the evidence tier is unusable there.
bool RuntimeBundleLowerer::crossesStorageDefiningBlock(
    mlir::Operation *op, const RuntimeBundle &bundle) {
  if (bundle.physicalValues().empty())
    return false;
  mlir::Value anchor = bundle.physicalValues().front();
  mlir::Block *defBlock = nullptr;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(anchor))
    defBlock = argument.getOwner();
  else if (mlir::Operation *defOp = anchor.getDefiningOp())
    defBlock = defOp->getBlock();
  // Same block: appended evidence dominates every later same-function use.
  return defBlock && defBlock != op->getBlock();
}

// Compile-time contents evidence describes a mutable container AS OF THE BLOCK
// THAT DEFINES ITS STORAGE, so it may only be consulted there. The physical
// payload stays authoritative everywhere else — the same truth the evidence
// mirrored — and every other block reads and writes through it.
//
// ⭐ This used to fire at MUTATIONS only, one variant per container kind, and
// the walk's own order made that unsound. A read is lowered before a store
// that appears after it in the block, and a back edge makes that store run
// FIRST:
//
//     xs: list[int] = [0]
//     for i in range(4):
//         xs[0] += 1
//     print(xs[0])      # printed 1; CPython prints 4
//
// Every iteration answered `xs[0]` from the literal's element map and stored
// `0 + 1`; the stores could not see each other. Demoting at the store was too
// late for the read in the same iteration. `d["a"] += "x"` accumulated one
// character for the same reason, and with a computed index the retain and the
// release landed on different objects -- a string reached count zero while
// still referenced and the next retain aborted with "Ly_IncRef observed
// non-positive refcount".
//
// Asking instead where the op IS makes the answer independent of walk order,
// so a back edge cannot smuggle a mutation past a read. It also removes the
// need to know which ops mutate: the two callers that did know disagreed
// (the dict variant never mirrored a field alias), and any op the enumeration
// missed was a silent wrong answer rather than a slower one.
//
// Only a mutable container, and only one whose payload exists: a tuple's
// contents cannot change, and for an evidence-only container the evidence is
// the sole description of its contents, so dropping it would lose them.
//
// ⛔ Why NOT writeBackFieldAlias for the field-alias mirror: the writeback
// re-roots the owner's owned-local marker at this op, and a root created
// inside a branch is itself the non-dominating value this demotion removes.
// ⭐ A CONTAINER NOTHING CAN MUTATE DESCRIBES THE SAME CONTENTS IN EVERY BLOCK.
// The demotion above is stated as "a different block" because it must not
// depend on the walk's order, and a mutation-site rule did. A rule over the
// container's WHOLE USE LIST has that property too: it reads the same before
// and after any op is lowered, and a back edge cannot smuggle anything past it
// because there is nothing to smuggle.
//
// It has to be a whitelist rather than a list of mutators. The two callers that
// enumerated mutators disagreed, and an op the enumeration missed was a silent
// wrong answer; an op this list has not heard of is merely a slower read.
// Storing the container anywhere, passing it to a call, or forwarding it to a
// block argument are all uses that are not on the list, so an alias that could
// be mutated behind the walk's back cannot form.
//
// Why this is worth a rule of its own: for a container whose element type is a
// UNION the runtime tier cannot answer the read at all -- there is no
// `builtins.list.__getitem__` that returns a union -- so the demotion turns a
// slower answer into a refusal:
//
//     xs = [1, "a"]
//     print(xs[0])
//     print(xs[1])   # runtime manifest has no builtins.list.__getitem__ method
//
// Printing a union branches on the tag, so the second read is in a successor
// block and lost evidence it could still have used. A record literal --
// `{"name": "ann", "age": 30}` read field by field -- is the same program.
bool RuntimeBundleLowerer::containerContentsAreUnreachableByMutation(
    mlir::Operation *op, mlir::Value containerValue,
    const RuntimeBundle &bundle) {
  bool mapping = !bundle.mappingKeys.empty();
  for (mlir::Operation *user : containerValue.getUsers()) {
    // `py.iter` is how `cfg.keys()` and `for k in cfg` both arrive, and it
    // hands out an iterator rather than the container -- there is no path from
    // the iterator back to a mutation of this value, so the description
    // survives it.
    //
    // ⛔ `py.repr` is NOT here even though printing a container plainly does
    // not change it. The exemption keeps the evidence, and an evidence read
    // hands back a BORROW of the slot where the runtime accessor hands back an
    // owned reference -- so a later mutation THROUGH such a read is refused:
    // `print(rows, ...)` followed by `rows[1].append(9)` became "list.append on
    // a field or borrowed list is not supported inside a branch or loop body".
    // The repr of a list expands into a loop, which is what put the append in
    // another block to begin with. Admitting repr means teaching the mutation
    // path to take an owned handle from an evidence element, which is a
    // different change.
    if (mlir::isa<py::LenOp, py::ContainsOp, py::IterOp>(user))
      continue;
    auto read = mlir::dyn_cast<py::GetItemOp>(user);
    if (!read || read.getContainer() != containerValue)
      return false;
    // ⛔ And every read must be one the evidence can ANSWER: a literal index
    // or key, and one that hits.
    //
    // A computed index has a dynamic evidence arm, and admitting it was tried
    // and reverted: for a heterogeneous container that arm cannot select at
    // all -- "dict __getitem__ evidence candidate 1 has a different physical
    // ABI shape" -- because the candidates would each have to be widened to
    // the union's lanes before the select chain. That is the same widening the
    // mutated-container read needs, and it is unbuilt.
    //
    // A miss raises, and the raise the evidence
    // tier emits is not the runtime tier's: it is spliced into the block the
    // read is in, which is what the demotion used to keep it out of. In
    // `i, j = [1]` the arity check raises first and `[1][1]` is dead code, but
    // the walk still lowers it -- and an IndexError raise inside a `try` left
    // the repr of the never-reached print released twice on one path.
    if (mapping) {
      std::optional<std::string> key =
          RuntimeBundleLowerer::keywordNameFromValue(read.getIndex());
      if (!key || !llvm::is_contained(bundle.mappingKeys, *key))
        return false;
      continue;
    }
    std::optional<std::int64_t> literal =
        integerLiteralFromValue(read.getIndex());
    if (!literal)
      return false;
    if (!bundle.sequenceIndices.empty()) {
      if (!llvm::is_contained(bundle.sequenceIndices, *literal))
        return false;
      continue;
    }
    std::int64_t size = static_cast<std::int64_t>(bundle.sequenceElements.size());
    std::int64_t normalized = *literal < 0 ? *literal + size : *literal;
    if (normalized < 0 || normalized >= size)
      return false;
  }

  // `containerValue` is an operand of `op`, so its defining block dominates
  // op's -- and the two blocks differ, or the caller would not be asking. Every
  // op of a strictly dominating block therefore dominates `op`, which is why
  // "defined in that block" is the whole test and no DominanceInfo is built.
  mlir::Value anchor = bundle.physicalValues().front();
  mlir::Block *defBlock = nullptr;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(anchor))
    defBlock = argument.getOwner();
  else if (mlir::Operation *definition = anchor.getDefiningOp())
    defBlock = definition->getBlock();
  if (!defBlock)
    return false;
  auto reaches = [&](mlir::Value value) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
      return argument.getOwner() == defBlock;
    mlir::Operation *definition = value.getDefiningOp();
    return definition && definition->getBlock() == defBlock;
  };
  auto valuesReach = [&](llvm::ArrayRef<RuntimeValue> values) {
    return llvm::all_of(values, [&](const RuntimeValue &value) {
      return llvm::all_of(value.values, reaches);
    });
  };
  auto bundlesReach =
      [&](llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> bundles) {
        return llvm::all_of(
            bundles, [&](const std::shared_ptr<RuntimeBundle> &element) {
              return !element || llvm::all_of(element->physicalValues(), reaches);
            });
      };
  return valuesReach(bundle.sequenceElements) &&
         bundlesReach(bundle.sequenceElementBundles) &&
         valuesReach(bundle.mappingValues) &&
         bundlesReach(bundle.mappingKeyBundles) &&
         bundlesReach(bundle.mappingValueBundles) &&
         llvm::all_of(bundle.mappingPresent, reaches);
}

bool RuntimeBundleLowerer::demoteCrossBlockContainerEvidence(
    mlir::Operation *op, mlir::Value containerValue) {
  RuntimeBundle *bundle = nullptr;
  if (auto found = valueBundles.find(containerValue);
      found != valueBundles.end())
    bundle = &found->second;
  if (!bundle || bundle->kind != RuntimeBundle::Kind::Object ||
      !RuntimeBundleLowerer::isMutableContainerContractName(
          bundle->contractName()) ||
      !RuntimeBundleLowerer::containerHasRuntimePayload(*bundle))
    return false;
  bool describesContents =
      bundle->sequenceEvidenceBacked || !bundle->sequenceElements.empty() ||
      bundle->mappingEvidenceBacked || !bundle->mappingKeys.empty();
  if (!describesContents)
    return false;
  if (!RuntimeBundleLowerer::crossesStorageDefiningBlock(op, *bundle))
    return false;
  if (RuntimeBundleLowerer::containerContentsAreUnreachableByMutation(
          op, containerValue, *bundle))
    return false;
  RuntimeBundleLowerer::demoteMutableContainerEvidence(*bundle);
  if (bundle->fieldAliasOwner && !bundle->fieldAliasName.empty()) {
    if (auto owner = valueBundles.find(bundle->fieldAliasOwner);
        owner != valueBundles.end()) {
      auto entry = owner->second.fieldBundles.find(bundle->fieldAliasName);
      if (entry != owner->second.fieldBundles.end() && entry->second) {
        auto demoted = std::make_shared<RuntimeBundle>(*entry->second);
        RuntimeBundleLowerer::demoteMutableContainerEvidence(*demoted);
        entry->second = std::move(demoted);
      }
    }
  }
  return true;
}

void RuntimeBundleLowerer::demoteCrossBlockContainerOperandEvidence(
    mlir::Operation *op) {
  for (mlir::Value operand : op->getOperands())
    RuntimeBundleLowerer::demoteCrossBlockContainerEvidence(op, operand);
}

// ⭐ A mutable container this op puts INTO a slot now has a second holder that
// can mutate it, so its own mutations may no longer take the evidence arm --
// which stores at the compile-time element count while taking the new length
// from the runtime word. See `sharedWithHolder`.
//
// ⛔ Only MARKED, never demoted. Dropping the contents evidence here was
// written and measured three ways (over pack/setitem/attrset, over container
// packs, over container packs whose source survives) and takes 145-146 tests
// down each time, aborting at runtime: the evidence is still the right answer
// for a READ, and something past the absorption depends on it.
void RuntimeBundleLowerer::markAbsorbedContainerAsShared(mlir::Operation *op) {
  mlir::Value absorbed;
  if (auto setItem = mlir::dyn_cast<py::SetItemOp>(op))
    absorbed = setItem.getValue();
  else if (auto attrSet = mlir::dyn_cast<py::AttrSetOp>(op))
    absorbed = attrSet.getValue();
  else
    return;
  auto found = valueBundles.find(absorbed);
  if (found == valueBundles.end())
    return;
  RuntimeBundle &bundle = found->second;
  if (bundle.kind != RuntimeBundle::Kind::Object ||
      !RuntimeBundleLowerer::isMutableContainerContractName(
          bundle.contractName()))
    return;
  bundle.sharedWithHolder = true;
}

// FIXED, by the walk to the root at the top of `lowerSetItem` below. A
// container read out of another container used to keep that other one's
// description of it:
//
//     grid: list[list[int]] = [[1, 2], [3, 4]]
//     grid[1][0] = 9
//     print(grid)          # [[1, 2], [9, 4]] -- the write landed
//     print(grid[1][0])    # 3                -- the read did not see it
//
// Re-measured 2026-08-14 against CPython 3.14: 9, and so does the aliased
// spelling (`first = data[0]; first["n"] = 5; data[0]["n"]`) and a store at a
// non-last outer index. The description below is what the walk is FOR.
//
// The store updates the element evidence of the container it stores INTO
// -- here the inner list. The outer list still holds the inner list's pre-store
// description in `sequenceElementBundles`, and the next `grid[1]` is answered
// from that. A FIELD receiver has a write-back for exactly this
// (`fieldAliasOwner`, a few lines down); an element receiver has no such link.
// Flat lists and dict values are unaffected -- they have no outer description
// to go stale.
//
// Why NOT repair the outer container when the receiver came from a `GetItemOp`:
// five attempts, every one of them leaks. `sequenceElementBundles` books the
// elements' references as well as describing them, and nothing tried keeps both
// halves:
//
//   - clear the parent's element evidence            -> 52 B
//   - clear only the element that was read           -> 52 B
//   - replace that element with the post-store bundle -> 93 B (and correct
//     values: 9, 7 and "z" all print right, walking the chain to the root
//     handles three levels too)
//   - the same, carrying the slot's ownership label over  -> 93 B
//   - the same, copying only the sequence evidence onto the entry the parent
//     already holds                                  -> 52 B / 41 B
//
// AND THE 52 B WAS ALREADY THERE. Measured on the unmodified compiler:
//
//     grid: list[list[int]] = [[1, 2], [3, 4]]
//     grid[1][0] = 9          # 52 B, no read of the value at all
//
// `grid[0][0] = 9` does not leak, nor does a one-element outer list, nor
// indexing without a store -- it is a store into an inner list at a NON-ZERO
// outer index, and it leaks with `inner = grid[1]` written out too. So the
// five repair attempts above were each landing on top of a leak they did not
// cause, and the figures that looked like their cost were partly this. That
// has to be fixed first; the read-back repair cannot be judged until the
// baseline is clean.
//
// Localised, for whoever picks it up. It is a store into the LAST element of
// the outer list -- index 1 of two, index 2 of three; index 1 of three is
// clean -- and the IR says which release goes astray. Both shapes allocate
// three inner lists and emit three `LyList_DecRef`s, but the clean one
// releases each of the three once, while the leaking one releases the
// setitem's receiver twice and the first inner list never:
//
//     clean:   DecRef %0  DecRef %21  DecRef %43    (three distinct)
//     leaking: DecRef %0  DecRef %21  DecRef %42    with %42 released again
//                                                   at the exit beside %43
//
// and the leaking one carries one `Ly_IncRef` the clean one does not, emitted
// immediately before that release so the pair cancels -- leaving the outer
// list's slot holding a reference nobody counted. Both come out of
// `refcount-insertion` (`runtime-lowering` emits eight retains for both
// programs; the pass makes it ten for both), so the placement differs, not
// the count.
//
// Attempted, and it made things worse. The move is recorded where the literal
// absorbs the source (`Core/CollectionPayload.cpp`), the frame token is minted
// afterwards by `Ops/GetItemOps.cpp`, and suppressing that token's release
// when the value was already absorbed took the leak from 52 B to 8420 B --
// the same token is what releases the value on the paths where the literal
// did NOT take it over, so suppressing it unconditionally drops those too.
// The condition has to distinguish the absorbed reference from the token's
// other duties, which the moved-value set alone does not.
//
// So the values are reachable and the ledger is what resists. The entry is not
// a description the parent keeps beside a reference; it IS how the reference is
// held, and every republication so far has been read as a second owner. The
// next attempt should change who releases the slot, not what the slot says --
// or move the element cache out of the ownership ledger so the two can be
// written independently.
mlir::LogicalResult RuntimeBundleLowerer::lowerSetItem(py::SetItemOp op) {
  // ⭐ A store through a name for a container ELEMENT ages the container it
  // came out of, whichever name the store is written through.
  //
  //     data: list[dict[str, int]] = [{"n": 1}]
  //     first: dict[str, int] = data[0]
  //     first["n"] = 5
  //     print(data[0]["n"])      # printed 1; CPython prints 5
  //
  // Reading through `first` was already right, so the store landed; the outer
  // list just kept describing element 0 as it was before. Same shape as the
  // one `demoteMutableContainerArgumentEvidence` handles for a value handed to
  // a callee -- there the boundary is the call, here it is the alias -- so the
  // repair is the same walk to the root.
  for (mlir::Value outer = op.getContainer(); outer;) {
    auto read = outer.getDefiningOp<py::GetItemOp>();
    if (!read)
      break;
    outer = read.getContainer();
    RuntimeBundleLowerer::demoteMutableContainerEvidenceFor(outer);
  }
  RuntimeBundleLowerer::demoteCrossBlockContainerEvidence(op.getOperation(),
                                                          op.getContainer());
  llvm::SmallVector<mlir::Value, 3> inputs{op.getContainer(), op.getIndex(),
                                           op.getValue()};
  llvm::SmallVector<const RuntimeBundle *, 3> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "setitem operands need runtime bundles", sources)))
    return mlir::failure();
  RuntimeBundleLowerer::demoteMappingEvidenceForDynamicKey(
      op.getContainer(), *sources[0], op.getIndex(), *sources[1]);
  const RuntimeBundle &container = *sources[0];
  const RuntimeBundle &index = *sources[1];
  const RuntimeBundle &value = *sources[2];
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.list") {
    std::optional<std::int64_t> rawIndex =
        integerLiteralFromValue(op.getIndex());
    // The compile-time element evidence is authoritative only for
    // evidence-backed lists. A list that crossed a function boundary
    // (closure capture, parameter) or was built in a loop carries NO
    // element evidence; treating its absent evidence as "length 0" here
    // mis-raised IndexError for every store, so those lists go through the
    // runtime payload path below instead.
    // ⭐ AND ONLY WHEN IT RECORDS ELEMENTS. `sequenceEvidenceBacked` is a flag
    // about the container's KIND, not a promise that this bundle describes the
    // contents, and a field read deliberately strips the contents
    // (`bindRetainedEvidenceBundle` does not let them cross the read):
    //
    //     class Box:
    //         def __init__(self) -> None:
    //             self.items: list[int] = []
    //     b = Box()
    //     b.items.append(1)
    //     b.items[0] = 9
    //     # IndexError: list assignment index out of range
    //     del b.items[0]        # the same mis-raise on the delete path
    //
    // The store read "backed, zero elements" as length zero and raised for an
    // index that is in range -- the mis-raise the paragraph above was written
    // against, one case further in: not a list with NO evidence, but a list whose
    // evidence says nothing about its contents. The read side already draws the
    // line here ("an evidence-BACKED container with no recorded elements ... reads
    // through the payload too", lowerRuntimeSequenceGetItem).
    //
    // ⭐ AND NOT THROUGH A FIELD AT ALL, which the empty case only hinted at. A
    // field seeded with `[0]` and grown by one append still describes one element,
    // and `items[1] = 9` then took the evidence arm and double-booked the slot
    // ("owned resource ... is released or transferred more than once"). The
    // evidence tier is only sound where the walk sees EVERY mutation of the
    // container, and it cannot see them through a field: the read strips the
    // contents it did know, and each read builds a fresh bundle from the owner.
    // So an interior view stores through the payload, which is authoritative.
    if (rawIndex && container.sequenceEvidenceBacked &&
        !container.sequenceElementBundles.empty() &&
        !container.fieldAliasOwner) {
      builder.setInsertionPoint(op);
      if (mlir::failed(RuntimeBundleLowerer::touchContainerEvidenceUse(
              op, container, "list setitem")))
        return mlir::failure();
      std::int64_t size =
          static_cast<std::int64_t>(container.sequenceElementBundles.size());
      std::int64_t normalized = *rawIndex;
      if (normalized < 0)
        normalized += size;
      if (normalized < 0 || normalized >= size) {
        builder.setInsertionPoint(op);
        if (mlir::failed(emitRuntimeException(op, "builtins.IndexError",
                                              "list assignment index out of "
                                              "range")))
          return mlir::failure();
      } else {
        RuntimeBundle updated = container;
        unsigned position = static_cast<unsigned>(normalized);
        const RuntimeBundle *oldBundle = nullptr;
        if (position < updated.sequenceElementBundles.size())
          oldBundle = updated.sequenceElementBundles[position].get();
        mlir::Type oldType = value.objectValue.contract;
        mlir::ValueRange oldValues;
        if (position < updated.sequenceElements.size()) {
          oldType = updated.sequenceElements[position].contract;
          oldValues = updated.sequenceElements[position].values;
        }
        mlir::FailureOr<RuntimeBundle> payload =
            RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
        if (mlir::failed(payload))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::replaceAggregateSlot(
                op, oldType, oldValues, oldBundle,
                payload->objectValue.contract, *payload, "list.setitem")))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
                op, updated, position, *payload)))
          return mlir::failure();
        // FIXED 2026-08-14, and the diagnosis above it was wrong about where.
        //
        //     holder: list[list[int]] = [[9]]
        //     holder[0] = [7]      # was 3 allocations, 8316 B, per execution
        //
        // `replaceAggregateSlot` above was never the problem: it releases the
        // slot's reference, correctly, and that release plus the nested
        // literal's `sequence.literal.source` release made TWO consumes of one
        // entity. `releaseOwnedGroupByLiveness` counted one reference in hand
        // against them and inserted an unfold retain nothing discharges. The
        // element's second reference is the literal's `aggregate_retain`, so
        // counting those makes it two against two and the retain goes away
        // (Passes/Ownership.cpp, "WHAT IS IN HAND IS NOT ALWAYS ONE";
        // tests/probe/wb_aggregate_slot_unfold_retain_leak.py).
        //
        // ⭐ It is the same defect as `del a[0]` and as `grid[1][0] = 9`,
        // which were recorded separately at 41/52 B and 52 B. What made them
        // look like three was that each needs a SECOND aggregate release to
        // show; with one release the old arithmetic reached zero by accident.
        RuntimeBundle stored =
            payload->withObjectOwnership(ownership::logicalOwnershipKind(
                payload->objectValue.contract, /*ownsObject=*/false));
        if (position < updated.sequenceElements.size())
          updated.sequenceElements[position] = stored.objectValue;
        if (position < updated.sequenceElementBundles.size())
          updated.sequenceElementBundles[position] =
              std::make_shared<RuntimeBundle>(std::move(stored));
        if (mlir::failed(
                RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
          return mlir::failure();
        valueBundles[op.getContainer()] = std::move(updated);
      }
      erase.push_back(op);
      return mlir::success();
    }
    if (RuntimeBundleLowerer::containerHasRuntimePayload(container) &&
        value.kind == RuntimeBundle::Kind::Object) {
      // Runtime-mode store (or a dynamic index on an evidence-backed list):
      // normalize/bounds-check against the runtime length and swap the
      // payload box in place. In-place: the receiver stays borrowed, so the
      // store is legal on captured and parameter lists whose owner is the
      // caller.
      std::optional<RuntimeSymbol> setItemBox =
          manifest.primitive("builtins.list", "setitem_box");
      if (!setItemBox)
        return op.emitError()
               << "runtime manifest has no list setitem_box primitive";
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();
      mlir::FailureOr<mlir::Value> raw =
          RuntimeBundleLowerer::rawSequenceIndexValue(op.getOperation(),
                                                      op.getIndex(), index);
      if (mlir::failed(raw))
        return mlir::failure();
      mlir::FailureOr<RuntimeBundle> payload =
          RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
      if (mlir::failed(payload))
        return mlir::failure();
      if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
              op, *payload, "list.setitem")))
        return mlir::failure();
      mlir::FailureOr<mlir::Value> box =
          RuntimeBundleLowerer::transientPayloadBox(op, *payload,
                                                    /*ownsPayload=*/true);
      if (mlir::failed(box))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 6> operands(
          container.physicalValues().begin(), container.physicalValues().end());
      operands.push_back(*raw);
      operands.push_back(*box);
      RuntimeBundleLowerer::createRuntimeCall(loc, *setItemBox, operands);
      // Pin the receiver past the raw-word call (mirrors the other *_box
      // container methods).
      if (std::optional<RuntimeSymbol> lenPin =
              manifest.method("builtins.list", "__len__")) {
        llvm::SmallVector<const RuntimeBundle *, 1> pinSources{&container};
        llvm::SmallVector<mlir::Value, 4> pinOperands;
        if (mlir::failed(buildRuntimeCallOperands(op, *lenPin, pinSources,
                                                  pinOperands,
                                                  /*allowUnusedSources=*/false)))
          return mlir::failure();
        RuntimeBundleLowerer::createRuntimeCall(loc, *lenPin, pinOperands);
      }
      // The runtime store invalidates whatever partial compile-time element
      // facts the bundle still carried.
      RuntimeBundle demoted = container;
      demoted.sequenceElements.clear();
      demoted.sequenceElementBundles.clear();
      demoted.sequenceIndices.clear();
      demoted.sequenceEvidenceBacked = false;
      if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op, demoted)))
        return mlir::failure();
      for (mlir::Value result : op->getResults())
        valueBundles[result] = demoted;
      valueBundles[op.getContainer()] = std::move(demoted);
      erase.push_back(op);
      return mlir::success();
    }
  }
  bool structuralMutation =
      op->hasAttr("ly.structural_mutation") && op.getNumResults() == 1;
  // NOTE(wave15 integration): the iter track carried a second runtime list
  // store lowering here (rebind convention, inline bounds guard). The
  // closure track's in-place LyList_SetItemBox path above subsumes it —
  // it also covers constant indexes and borrowed receivers — so the
  // duplicate was dropped rather than kept unreachable.
  bool runtimeDictInsert =
      container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict" &&
      !container.mappingEvidenceBacked && container.mappingKeys.empty() &&
      RuntimeBundleLowerer::containerHasRuntimePayload(container) &&
      index.kind == RuntimeBundle::Kind::Object &&
      value.kind == RuntimeBundle::Kind::Object;
  if ((structuralMutation || op.getNumResults() == 0) && runtimeDictInsert) {
    // Runtime-mode dict (loop-built): contents are only known to the
    // runtime. Insert through the runtime probe; the rebind result carries
    // the (possibly reallocated) representation. Keys may be any hashable
    // class — the probe hashes the transient box and raises TypeError for
    // unhashable ones.
    std::optional<RuntimeSymbol> setItemBox =
        manifest.primitive("builtins.dict", "setitem_box");
    if (!setItemBox)
      return op.emitError()
             << "runtime manifest has no dict setitem_box primitive";

    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<RuntimeBundle> payloadKey =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, index);
    if (mlir::failed(payloadKey))
      return mlir::failure();
    mlir::FailureOr<RuntimeBundle> payloadValue =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
    if (mlir::failed(payloadValue))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadKey, "dict.setitem.key")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadValue, "dict.setitem")))
      return mlir::failure();

    auto transientBox =
        [&](const RuntimeBundle &bundle) -> mlir::FailureOr<mlir::Value> {
      return RuntimeBundleLowerer::transientPayloadBox(op, bundle,
                                                       /*ownsPayload=*/true);
    };
    if (mlir::failed(RuntimeBundleLowerer::promoteInteriorViewForTransfer(
            op, container, "dict.setitem.receiver", setItemBox->function)))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> keyBox = transientBox(*payloadKey);
    if (mlir::failed(keyBox))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> valueBox = transientBox(*payloadValue);
    if (mlir::failed(valueBox))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 8> operands(
        container.physicalValues().begin(), container.physicalValues().end());
    operands.push_back(*keyBox);
    operands.push_back(*valueBox);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *setItemBox, operands);

    RuntimeBundle updated;
    if (mlir::failed(RuntimeBundleLowerer::rebindMutatedContainer(
            op, container, call.getResults(), updated)))
      return mlir::failure();
    if (structuralMutation) {
      valueBundles[op.getResult(0)] = std::move(updated);
    } else {
      // Non-rebind form (`self.data[k] = v`): there is no local to reassign, so
      // the container VALUE names the re-description. The write-back into the
      // field slot already happened in rebindMutatedContainer.
      //
      // No retain/release pair here any more. It used to compensate for the
      // read arriving without a reference of its own -- the transfer consumed
      // the SLOT's reference and the pair put one back -- and it counted a
      // reference the read now holds, which the affine verifier reads as one
      // resource released twice.
      valueBundles[op.getContainer()] = std::move(updated);
    }
    erase.push_back(op);
    return mlir::success();
  }
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict" &&
      index.kind == RuntimeBundle::Kind::Object &&
      value.kind == RuntimeBundle::Kind::Object) {
    std::optional<std::string> key =
        RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
    if (!key && index.literalText)
      key = *index.literalText;
    if (key) {
      builder.setInsertionPoint(op);
      if (mlir::failed(RuntimeBundleLowerer::touchContainerEvidenceUse(
              op, container, "dict setitem")))
        return mlir::failure();
      RuntimeBundle updated = container;
      auto found = llvm::find(updated.mappingKeys, *key);
      mlir::FailureOr<RuntimeBundle> payloadKey =
          RuntimeBundleLowerer::materializePayloadObjectBundle(op, index);
      if (mlir::failed(payloadKey))
        return mlir::failure();
      mlir::FailureOr<RuntimeBundle> payloadValue =
          RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
      if (mlir::failed(payloadValue))
        return mlir::failure();
      if (found == updated.mappingKeys.end()) {
        if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
                op, *payloadKey, "dict.setitem.key")))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
                op, *payloadValue, "dict.setitem")))
          return mlir::failure();
        unsigned position = static_cast<unsigned>(updated.mappingValues.size());
        if (mlir::failed(RuntimeBundleLowerer::storeDictKeyPayload(
                op, updated, position, *payloadKey)))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
                op, updated, position, *payloadValue)))
          return mlir::failure();
        // `updated`, not `container`: storeDictValuePayload may have grown the
        // dict, and for a handle-fronted contract the length lives in the
        // handle the growth wrote through.
        if (mlir::failed(RuntimeBundleLowerer::adjustContainerLength(
                op, updated, +1, "dict setitem")))
          return mlir::failure();
        RuntimeBundle storedKey =
            payloadKey->withObjectOwnership(ownership::logicalOwnershipKind(
                payloadKey->objectValue.contract, /*ownsObject=*/false));
        RuntimeBundle storedValue =
            payloadValue->withObjectOwnership(ownership::logicalOwnershipKind(
                payloadValue->objectValue.contract, /*ownsObject=*/false));
        updated.mappingKeys.push_back(*key);
        updated.mappingKeyBundles.push_back(
            std::make_shared<RuntimeBundle>(storedKey));
        updated.mappingValues.push_back(storedValue.objectValue);
        updated.mappingValueBundles.push_back(
            std::make_shared<RuntimeBundle>(std::move(storedValue)));
        if (!updated.mappingPresent.empty())
          updated.mappingPresent.push_back(
              constantBool(builder, op.getLoc(), true));
      } else {
        unsigned position =
            static_cast<unsigned>(found - updated.mappingKeys.begin());
        mlir::Type oldType = value.objectValue.contract;
        mlir::ValueRange oldValues;
        if (position < updated.mappingValues.size()) {
          oldType = updated.mappingValues[position].contract;
          oldValues = updated.mappingValues[position].values;
        }
        const RuntimeBundle *oldValueBundle = nullptr;
        if (position < updated.mappingValueBundles.size())
          oldValueBundle = updated.mappingValueBundles[position].get();
        if (mlir::failed(RuntimeBundleLowerer::replaceAggregateSlot(
                op, oldType, oldValues, oldValueBundle,
                payloadValue->objectValue.contract, *payloadValue,
                "dict.setitem")))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
                op, updated, position, *payloadValue)))
          return mlir::failure();
        RuntimeBundle storedValue =
            payloadValue->withObjectOwnership(ownership::logicalOwnershipKind(
                payloadValue->objectValue.contract, /*ownsObject=*/false));
        if (position < updated.mappingValues.size())
          updated.mappingValues[position] = storedValue.objectValue;
        if (position < updated.mappingValueBundles.size())
          updated.mappingValueBundles[position] =
              std::make_shared<RuntimeBundle>(std::move(storedValue));
        if (position < updated.mappingPresent.size()) {
          builder.setInsertionPoint(op);
          updated.mappingPresent[position] =
              constantBool(builder, op.getLoc(), true);
        }
      }
      if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
        return mlir::failure();
      // The structural-mutation rebind form: downstream uses read the RESULT
      // SSA value (the emitter rebound the local to it), so the updated
      // evidence must be visible under that name too.
      if (op.getNumResults() == 1)
        valueBundles[op.getResult(0)] = updated;
      valueBundles[op.getContainer()] = std::move(updated);
      erase.push_back(op);
      return mlir::success();
    }
  }
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__setitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestVoidMethod(op, *sources.front(), *methodName,
                                           sources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerDelItem(py::DelItemOp op) {
  RuntimeBundleLowerer::demoteCrossBlockContainerEvidence(op.getOperation(),
                                                          op.getContainer());
  llvm::SmallVector<mlir::Value, 2> inputs{op.getContainer(), op.getIndex()};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "delitem operands need runtime bundles", sources)))
    return mlir::failure();
  RuntimeBundleLowerer::demoteMappingEvidenceForDynamicKey(
      op.getContainer(), *sources[0], op.getIndex(), *sources[1]);
  const RuntimeBundle &container = *sources[0];
  const RuntimeBundle &index = *sources[1];
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict" &&
      !container.mappingEvidenceBacked && container.mappingKeys.empty() &&
      RuntimeBundleLowerer::containerHasRuntimePayload(container) &&
      index.kind == RuntimeBundle::Kind::Object) {
    // Runtime-mode dict delete: hash probe with a BORROWED transient key
    // box; the runtime raises KeyError (with the key's repr) on a miss and
    // compacts the dense entries in place, so the container's SSA
    // representation is unchanged.
    std::optional<RuntimeSymbol> delItemBox =
        manifest.primitive("builtins.dict", "delitem_box");
    if (!delItemBox)
      return op.emitError()
             << "runtime manifest has no dict delitem_box primitive";
    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<RuntimeBundle> payload =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, index);
    if (mlir::failed(payload))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> box =
        RuntimeBundleLowerer::transientPayloadBox(op, *payload,
                                                  /*ownsPayload=*/false);
    if (mlir::failed(box))
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 8> operands(
        container.physicalValues().begin(), container.physicalValues().end());
    operands.push_back(*box);
    RuntimeBundleLowerer::createRuntimeCall(loc, *delItemBox, operands);
    if (mlir::failed(RuntimeBundleLowerer::pinProbeOperandLiveness(
            op, *payload, &index)))
      return mlir::failure();
    for (mlir::Value result : op->getResults())
      valueBundles[result] = container;
    erase.push_back(op);
    return mlir::success();
  }
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.list") {
    std::optional<std::int64_t> rawIndex =
        integerLiteralFromValue(op.getIndex());
    // Same evidence-authority rule as lowerSetItem: absent evidence is not
    // an empty list, so only evidence-backed lists take the compile-time
    // path.
    // ⭐ AND ONLY WHEN IT RECORDS ELEMENTS. `sequenceEvidenceBacked` is a flag
    // about the container's KIND, not a promise that this bundle describes the
    // contents, and a field read deliberately strips the contents
    // (`bindRetainedEvidenceBundle` does not let them cross the read):
    //
    //     class Box:
    //         def __init__(self) -> None:
    //             self.items: list[int] = []
    //     b = Box()
    //     b.items.append(1)
    //     b.items[0] = 9
    //     # IndexError: list assignment index out of range
    //     del b.items[0]        # the same mis-raise on the delete path
    //
    // The store read "backed, zero elements" as length zero and raised for an
    // index that is in range -- the mis-raise the paragraph above was written
    // against, one case further in: not a list with NO evidence, but a list whose
    // evidence says nothing about its contents. The read side already draws the
    // line here ("an evidence-BACKED container with no recorded elements ... reads
    // through the payload too", lowerRuntimeSequenceGetItem).
    //
    // ⭐ AND NOT THROUGH A FIELD AT ALL: see `lowerSetItem` above for the case
    // that showed why (a seeded field grown by one append double-booked the slot).
    if (rawIndex && container.sequenceEvidenceBacked &&
        !container.sequenceElementBundles.empty() &&
        !container.fieldAliasOwner) {
      RuntimeBundle updated = container;
      std::int64_t size =
          static_cast<std::int64_t>(updated.sequenceElementBundles.size());
      std::int64_t normalized = *rawIndex;
      if (normalized < 0)
        normalized += size;
      builder.setInsertionPoint(op);
      if (normalized < 0 || normalized >= size) {
        if (mlir::failed(emitRuntimeException(op, "builtins.IndexError",
                                              "list assignment index out of "
                                              "range")))
          return mlir::failure();
      } else {
        mlir::FailureOr<mlir::Value> length =
            RuntimeBundleLowerer::loadContainerLength(op, container,
                                                      "list delitem");
        if (mlir::failed(length))
          return mlir::failure();
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 64);
        mlir::Value next =
            mlir::arith::SubIOp::create(builder, op.getLoc(), *length, one);
        if (mlir::failed(RuntimeBundleLowerer::storeContainerLength(
                op, container, next, "list delitem")))
          return mlir::failure();
        unsigned position = static_cast<unsigned>(normalized);
        unsigned oldSize =
            static_cast<unsigned>(updated.sequenceElementBundles.size());
        if (position < updated.sequenceElementBundles.size() &&
            updated.sequenceElementBundles[position]) {
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, *updated.sequenceElementBundles[position],
                  "list.delitem")))
            return mlir::failure();
        } else if (position < updated.sequenceElements.size()) {
          const RuntimeValue &oldElement = updated.sequenceElements[position];
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, oldElement.contract, oldElement.values, "list.delitem")))
            return mlir::failure();
        }
        if (position < updated.sequenceElements.size())
          updated.sequenceElements.erase(updated.sequenceElements.begin() +
                                         position);
        if (position < updated.sequenceElementBundles.size())
          updated.sequenceElementBundles.erase(
              updated.sequenceElementBundles.begin() + position);
        for (unsigned rewrite = position,
                      end = static_cast<unsigned>(
                          updated.sequenceElementBundles.size());
             rewrite < end; ++rewrite) {
          if (updated.sequenceElementBundles[rewrite] &&
              mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
                  op, updated, rewrite,
                  *updated.sequenceElementBundles[rewrite])))
            return mlir::failure();
        }
        if (oldSize > 0 &&
            mlir::failed(RuntimeBundleLowerer::clearSequencePayloadElement(
                op, updated, oldSize - 1)))
          return mlir::failure();
        if (mlir::failed(
                RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
          return mlir::failure();
        valueBundles[op.getContainer()] = std::move(updated);
      }
      erase.push_back(op);
      return mlir::success();
    }
    if (RuntimeBundleLowerer::containerHasRuntimePayload(container)) {
      // Runtime-mode delete: bounds-check against the runtime length,
      // release the slot and compact in place (borrowed receiver).
      std::optional<RuntimeSymbol> delItem =
          manifest.primitive("builtins.list", "delitem_index");
      if (!delItem)
        return op.emitError()
               << "runtime manifest has no list delitem_index primitive";
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();
      mlir::FailureOr<mlir::Value> raw =
          RuntimeBundleLowerer::rawSequenceIndexValue(op.getOperation(),
                                                      op.getIndex(), index);
      if (mlir::failed(raw))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 4> operands(
          container.physicalValues().begin(), container.physicalValues().end());
      operands.push_back(*raw);
      RuntimeBundleLowerer::createRuntimeCall(loc, *delItem, operands);
      if (std::optional<RuntimeSymbol> lenPin =
              manifest.method("builtins.list", "__len__")) {
        llvm::SmallVector<const RuntimeBundle *, 1> pinSources{&container};
        llvm::SmallVector<mlir::Value, 4> pinOperands;
        if (mlir::failed(buildRuntimeCallOperands(op, *lenPin, pinSources,
                                                  pinOperands,
                                                  /*allowUnusedSources=*/false)))
          return mlir::failure();
        RuntimeBundleLowerer::createRuntimeCall(loc, *lenPin, pinOperands);
      }
      RuntimeBundle demoted = container;
      demoted.sequenceElements.clear();
      demoted.sequenceElementBundles.clear();
      demoted.sequenceIndices.clear();
      demoted.sequenceEvidenceBacked = false;
      if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op, demoted)))
        return mlir::failure();
      for (mlir::Value result : op->getResults())
        valueBundles[result] = demoted;
      valueBundles[op.getContainer()] = std::move(demoted);
      erase.push_back(op);
      return mlir::success();
    }
  }
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict") {
    std::optional<std::string> key =
        RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
    if (!key && index.literalText)
      key = *index.literalText;
    if (key) {
      RuntimeBundle updated = container;
      auto found = llvm::find(updated.mappingKeys, *key);
      builder.setInsertionPoint(op);
      if (found == updated.mappingKeys.end()) {
        // Raw key: the KeyError __init__ stores repr(message).
        if (mlir::failed(
                emitRuntimeException(op, "builtins.KeyError", *key)))
          return mlir::failure();
      } else {
        mlir::FailureOr<mlir::Value> length =
            RuntimeBundleLowerer::loadContainerLength(op, container,
                                                      "dict delitem");
        if (mlir::failed(length))
          return mlir::failure();
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 64);
        mlir::Value next =
            mlir::arith::SubIOp::create(builder, op.getLoc(), *length, one);
        if (mlir::failed(RuntimeBundleLowerer::storeContainerLength(
                op, container, next, "dict delitem")))
          return mlir::failure();
        unsigned position =
            static_cast<unsigned>(found - updated.mappingKeys.begin());
        unsigned oldSize = static_cast<unsigned>(updated.mappingKeys.size());
        if (position < updated.mappingKeyBundles.size() &&
            updated.mappingKeyBundles[position]) {
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, *updated.mappingKeyBundles[position],
                  "dict.delitem.key")))
            return mlir::failure();
        }
        if (position < updated.mappingValues.size()) {
          if (position < updated.mappingValueBundles.size() &&
              updated.mappingValueBundles[position]) {
            if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                    op, *updated.mappingValueBundles[position],
                    "dict.delitem")))
              return mlir::failure();
          } else {
            const RuntimeValue &oldValue = updated.mappingValues[position];
            if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                    op, oldValue.contract, oldValue.values, "dict.delitem")))
              return mlir::failure();
          }
        }
        updated.mappingKeys.erase(updated.mappingKeys.begin() + position);
        if (position < updated.mappingKeyBundles.size())
          updated.mappingKeyBundles.erase(updated.mappingKeyBundles.begin() +
                                          position);
        if (position < updated.mappingValues.size())
          updated.mappingValues.erase(updated.mappingValues.begin() + position);
        if (position < updated.mappingValueBundles.size())
          updated.mappingValueBundles.erase(
              updated.mappingValueBundles.begin() + position);
        if (position < updated.mappingPresent.size())
          updated.mappingPresent.erase(updated.mappingPresent.begin() +
                                       position);
        for (unsigned rewrite = position, end = static_cast<unsigned>(
                                              updated.mappingKeyBundles.size());
             rewrite < end; ++rewrite) {
          if (rewrite >= updated.mappingValueBundles.size() ||
              !updated.mappingKeyBundles[rewrite] ||
              !updated.mappingValueBundles[rewrite])
            return op.emitError()
                   << "dict delitem needs key/value payload evidence to "
                      "compact storage";
          if (mlir::failed(RuntimeBundleLowerer::storeDictKeyPayload(
                  op, updated, rewrite, *updated.mappingKeyBundles[rewrite])))
            return mlir::failure();
          if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
                  op, updated, rewrite, *updated.mappingValueBundles[rewrite])))
            return mlir::failure();
        }
        if (oldSize > 0 &&
            mlir::failed(RuntimeBundleLowerer::clearDictPayloadEntry(
                op, updated, oldSize - 1)))
          return mlir::failure();
        if (mlir::failed(
                RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
          return mlir::failure();
        valueBundles[op.getContainer()] = std::move(updated);
      }
      erase.push_back(op);
      return mlir::success();
    }
  }
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__delitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestVoidMethod(op, *sources.front(), *methodName,
                                           sources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerIs(py::IsOp op) {
  llvm::SmallVector<mlir::Value, 2> inputs{op.getLhs(), op.getRhs()};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "identity operands need runtime bundles", sources)))
    return mlir::failure();
  // Identity is the address of the leading object header; an operand whose
  // bundle carries no header memref (an evidence-only aggregate) has no
  // runtime identity to compare, and guessing equality either way would
  // silently mis-execute.
  auto headerOf = [](const RuntimeBundle &bundle) -> mlir::Value {
    if (bundle.physicalValues().empty())
      return {};
    mlir::Value first = bundle.physicalValues().front();
    return mlir::isa<mlir::MemRefType>(first.getType()) ? first
                                                        : mlir::Value();
  };
  mlir::Value lhsHeader = headerOf(*sources.front());
  mlir::Value rhsHeader = headerOf(*sources.back());
  if (!lhsHeader || !rhsHeader)
    return op.emitError()
           << "`is` operand has no runtime object header (evidence-only "
              "value); use `==` instead";
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value lhsAddress = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
      builder, loc, lhsHeader);
  mlir::Value rhsAddress = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
      builder, loc, rhsHeader);
  mlir::Value same = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lhsAddress, rhsAddress);
  op.getResult().replaceAllUsesWith(same);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerContains(py::ContainsOp op) {
  llvm::SmallVector<mlir::Value, 2> inputs{op.getContainer(), op.getItem()};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "contains operands need runtime bundles", sources)))
    return mlir::failure();
  const RuntimeBundle container = *sources.front();
  const RuntimeBundle item = *sources.back();
  bool runtimeSetProbe = container.kind == RuntimeBundle::Kind::Object &&
                         (container.contractName() == "builtins.set" ||
                          container.contractName() == "builtins.frozenset") &&
                         RuntimeBundleLowerer::containerHasRuntimePayload(
                             container);
  // Evidence-backed dicts probe the payload too — a runtime key (int
  // variable, frozenset) has no literal-key evidence to consult.
  bool runtimeDictProbe = container.kind == RuntimeBundle::Kind::Object &&
                          container.contractName() == "builtins.dict" &&
                          RuntimeBundleLowerer::containerHasRuntimePayload(
                              container);
  // Membership probe with a BORROWED transient element box (hash-based for
  // set/dict, identity-or-equality scan for list/tuple), then pin both the
  // container and the probed item past the call (the box holds raw pointer
  // words the liveness cannot see).
  auto emitBoxProbe = [&]() -> mlir::LogicalResult {
    std::optional<RuntimeSymbol> containsBox =
        manifest.primitive(container.contractName(), "contains_box");
    if (!containsBox)
      return op.emitError() << "runtime manifest has no "
                            << container.contractName()
                            << " contains_box primitive";
    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<RuntimeBundle> payload =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, item);
    if (mlir::failed(payload))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> box =
        RuntimeBundleLowerer::transientPayloadBox(op, *payload,
                                                  /*ownsPayload=*/false);
    if (mlir::failed(box))
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 8> operands(
        container.physicalValues().begin(), container.physicalValues().end());
    operands.push_back(*box);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *containsBox, operands);
    if (mlir::failed(RuntimeBundleLowerer::pinProbeOperandLiveness(
            op, *payload, &item)))
      return mlir::failure();
    auto pinObject = [&](const RuntimeBundle &object,
                         llvm::StringRef pinMethod) -> mlir::LogicalResult {
      std::optional<RuntimeSymbol> method =
          manifest.method(object.contractName(), pinMethod);
      if (!method)
        return op.emitError() << "membership probe needs "
                              << object.contractName() << "." << pinMethod
                              << " to pin its operand";
      llvm::SmallVector<const RuntimeBundle *, 1> pinSources{&object};
      llvm::SmallVector<mlir::Value, 4> pinOperands;
      if (mlir::failed(buildRuntimeCallOperands(op, *method, pinSources,
                                                pinOperands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      RuntimeBundleLowerer::createRuntimeCall(loc, *method, pinOperands);
      return mlir::success();
    };
    if (mlir::failed(pinObject(container, "__len__")))
      return mlir::failure();
    op.getResult().replaceAllUsesWith(call.getResult(0));
    erase.push_back(op);
    return mlir::success();
  };
  if ((runtimeSetProbe || runtimeDictProbe) &&
      item.kind == RuntimeBundle::Kind::Object)
    return emitBoxProbe();
  bool sequenceContainer = container.kind == RuntimeBundle::Kind::Object &&
                           (container.contractName() == "builtins.list" ||
                            container.contractName() == "builtins.tuple");
  // ⭐ A SECOND HOLDER'S MUTATION IS NOT IN THE EVIDENCE, so a shared sequence
  // probes the payload like a runtime one. `b = Bag(seed); b.xs[0] = 9;
  // print(9 in seed, 3 in seed)` answered `False True` -- the pre-store slots,
  // constant-folded. The `[i]` read side of this is in GetItemOps.cpp, with the
  // measurements and the reason the evidence is kept rather than dropped.
  if (sequenceContainer && !container.sequenceElementBundles.empty() &&
      !container.sharedWithHolder) {
    // Constant-fold evidence membership when every element compares
    // statically; otherwise probe the published payload at runtime.
    bool allKnown = true;
    {
      mlir::OpBuilder::InsertionGuard probeGuard(builder);
      builder.setInsertionPoint(op);
      mlir::Block *scratch = new mlir::Block();
      builder.setInsertionPointToStart(scratch);
      for (const std::shared_ptr<RuntimeBundle> &element :
           container.sequenceElementBundles) {
        if (!element ||
            !knownEvidenceEquality(op, builder, *element, item)) {
          allKnown = false;
          break;
        }
      }
      scratch->dropAllReferences();
      delete scratch;
    }
    if (allKnown) {
      builder.setInsertionPoint(op);
      if (mlir::failed(RuntimeBundleLowerer::touchContainerEvidenceUse(
              op, container, "sequence contains")))
        return mlir::failure();
      mlir::Location loc = op.getLoc();
      mlir::Value result = constantBool(builder, loc, false);
      for (const std::shared_ptr<RuntimeBundle> &element :
           container.sequenceElementBundles) {
        std::optional<mlir::Value> equal =
            knownEvidenceEquality(op, builder, *element, item);
        if (!equal)
          return mlir::failure();
        result = mlir::arith::OrIOp::create(builder, loc, result, *equal);
      }
      op.getResult().replaceAllUsesWith(result);
      erase.push_back(op);
      return mlir::success();
    }
  }
  if (sequenceContainer &&
      RuntimeBundleLowerer::containerHasRuntimePayload(container) &&
      item.kind == RuntimeBundle::Kind::Object)
    return emitBoxProbe();
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict" &&
      !container.mappingKeys.empty() && item.contractName() == "builtins.str") {
    builder.setInsertionPoint(op);
    if (mlir::failed(RuntimeBundleLowerer::touchContainerEvidenceUse(
            op, container, "dict contains")))
      return mlir::failure();
    mlir::Location loc = op.getLoc();
    mlir::Value result = constantBool(builder, loc, false);
    if (item.literalText) {
      for (auto [index, key] : llvm::enumerate(container.mappingKeys)) {
        mlir::Value match = constantBool(builder, loc, key == *item.literalText);
        if (index < container.mappingPresent.size())
          match = mlir::arith::AndIOp::create(builder, loc, match,
                                              container.mappingPresent[index]);
        result = mlir::arith::OrIOp::create(builder, loc, result, match);
      }
      op.getResult().replaceAllUsesWith(result);
      erase.push_back(op);
      return mlir::success();
    }

    std::optional<RuntimeSymbol> eq = manifest.method("builtins.str", "__eq__");
    if (!eq)
      return op.emitError() << "dict evidence contains needs str.__eq__";
    for (auto [index, key] : llvm::enumerate(container.mappingKeys)) {
      RuntimeBundle keyObject;
      if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
              op, key, keyObject)))
        return mlir::failure();
      llvm::SmallVector<const RuntimeBundle *, 2> eqSources{&item, &keyObject};
      llvm::SmallVector<mlir::Value, 4> eqOperands;
      if (mlir::failed(buildRuntimeCallOperands(op, *eq, eqSources, eqOperands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      mlir::func::CallOp eqCall =
          RuntimeBundleLowerer::createRuntimeCall(loc, *eq, eqOperands);
      if (eqCall.getNumResults() != 1 ||
          !eqCall.getResult(0).getType().isInteger(1))
        return eq->function.emitError()
               << "str.__eq__ evidence method must return one i1";
      mlir::Value match = eqCall.getResult(0);
      if (index < container.mappingPresent.size())
        match = mlir::arith::AndIOp::create(builder, loc, match,
                                            container.mappingPresent[index]);
      result = mlir::arith::OrIOp::create(builder, loc, result, match);
    }
    op.getResult().replaceAllUsesWith(result);
    erase.push_back(op);
    return mlir::success();
  }
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__contains__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestI1MethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerIter(py::IterOp op) {
  if (op.getReturnedSelf())
    return RuntimeBundleLowerer::lowerAliasView(op, op.getIterable(),
                                                op.getResult());

  // The iterable may be a block argument whose ABI expansion has not run
  // yet (walk order does not guarantee it): materialize the bundle first,
  // or the container paths below silently fall through to the manifest
  // fallback.
  if (mlir::failed(
          RuntimeBundleLowerer::ensureValueBundle(op, op.getIterable())))
    return mlir::failure();

  // Statically evidenced list iteration: there is no runtime `list.__iter__`
  // object; iterate the compile-time element evidence through a hoisted
  // position cell instead. The cell is alloca'd once per function (so nested
  // re-creation of the iterator reuses the slot) and reset to zero here.
  if (const RuntimeBundle *iterable =
          RuntimeBundleLowerer::bundleFor(op.getIterable())) {
    // Tuples with compile-time element evidence iterate exactly like
    // evidence lists (the evidence-next path is contract-agnostic).
    bool evidenceListIterable = (iterable->contractName() == "builtins.list" ||
                                 iterable->contractName() ==
                                     "builtins.tuple") &&
                                !iterable->sequenceElements.empty() &&
                                iterable->sequenceIndices.empty() &&
                                !iterable->evidenceIteratorCell;
    // Runtime-mode list (no compile-time element evidence): iterate the
    // runtime payload through the same hoisted position cell; `py.next`
    // rebuilds each element from its payload box words. An evidence-BACKED
    // list with no recorded elements (an annotated empty literal grown by
    // loop appends) qualifies too: its payload is the only truth.
    bool runtimeListIterable = iterable->contractName() == "builtins.list" &&
                               iterable->sequenceElements.empty() &&
                               !iterable->evidenceIteratorCell &&
                               RuntimeBundleLowerer::
                                   containerHasRuntimePayload(*iterable);
    // Dict key iteration: the key boxes sit at the same offsets in the handle
    // that the list uses for its items (meta at words 2/3, the primary array's
    // base at word 4 — see ContainerLayout.h), so the runtime-list next path
    // applies verbatim. Evidence-backed dicts qualify too — their payload
    // arrays are materialized alongside the evidence (initializeDictPayload /
    // the evidence mutators keep them in sync), and iterating the live payload
    // keeps the mutation guard and insertion order identical to the runtime
    // tier.
    //
    // Why the four comments here name word offsets and not `[1]`/`[2]`: those
    // were lane indices into physicalValues(), and all five containers are now
    // one lane, so physicalValues() has size 1 and the indices named nothing.
    // The claims themselves were never wrong -- all five share words 0..7 of
    // one layout -- and the code kept working because containerInteriorView and
    // containerHasRuntimePayload are the single place that knows which
    // representation is live.
    bool runtimeDictIterable = iterable->contractName() == "builtins.dict" &&
                               iterable->sequenceElements.empty() &&
                               !iterable->evidenceIteratorCell &&
                               RuntimeBundleLowerer::
                                   containerHasRuntimePayload(*iterable);
    // Runtime sets (and frozensets — identical layout) share the list's
    // physical layout exactly (meta at words 2/3, boxed slots reached through
    // word 4), so the runtime-list next path applies verbatim.
    bool runtimeSetIterable = (iterable->contractName() == "builtins.set" ||
                               iterable->contractName() ==
                                   "builtins.frozenset") &&
                              iterable->sequenceElements.empty() &&
                              !iterable->evidenceIteratorCell &&
                              RuntimeBundleLowerer::
                                  containerHasRuntimePayload(*iterable);
    // Runtime tuples (eg.exceptions, str.partition results, tuple(xs))
    // share the list's physical layout exactly (meta at words 2/3, boxed slots
    // reached through word 4); immutable, so no mutation guard.
    bool runtimeTupleIterable = iterable->contractName() == "builtins.tuple" &&
                                iterable->sequenceElements.empty() &&
                                !iterable->evidenceIteratorCell &&
                                RuntimeBundleLowerer::
                                   containerHasRuntimePayload(*iterable);
    if (evidenceListIterable || runtimeListIterable || runtimeDictIterable ||
        runtimeSetIterable || runtimeTupleIterable) {
      mlir::func::FuncOp function = op->getParentOfType<mlir::func::FuncOp>();
      if (!function)
        return op.emitError() << "list iteration requires a function context";
      // dict/set iterators additionally remember the container's size at
      // creation (cell word 1): CPython raises RuntimeError when the size
      // changes during iteration, while list iteration legally re-checks the
      // live length each step.
      bool guardsMutation = runtimeDictIterable || runtimeSetIterable;
      // ⭐ A set's dense array is not in its table's order until it is asked
      // for: `__ly_set_raw_place` appends, because keeping the array sorted
      // under insertion is what made `set.add` linear. The walk below reads
      // the array straight, so the order has to be restored before it starts.
      // Once per iterator, not once per step -- CPython walks its table once
      // per iteration too.
      if (runtimeSetIterable) {
        std::optional<RuntimeSymbol> reorder =
            manifest.primitive(iterable->contractName(), "reorder");
        if (!reorder)
          return op.emitError()
                 << "set iteration needs a `reorder` primitive on "
                 << iterable->contractName();
        builder.setInsertionPoint(op);
        llvm::SmallVector<mlir::Value, 2> operands(
            iterable->physicalValues().begin(),
            iterable->physicalValues().end());
        RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *reorder,
                                                operands);
      }
      mlir::Value cell;
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&function.getBody().front());
        cell = mlir::memref::AllocaOp::create(
                   builder, op.getLoc(),
                   mlir::MemRefType::get({guardsMutation ? 2 : 1},
                                         builder.getI64Type()))
                   .getResult();
      }
      builder.setInsertionPoint(op);
      mlir::Value zero =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64);
      mlir::Value slot =
          mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0);
      mlir::memref::StoreOp::create(builder, op.getLoc(), zero, cell, slot);
      if (guardsMutation) {
        mlir::FailureOr<mlir::Value> initial =
            RuntimeBundleLowerer::loadContainerLength(op, *iterable,
                                                      "iterator guard");
        if (mlir::failed(initial))
          return mlir::failure();
        mlir::Value initialSlot =
            mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 1);
        mlir::memref::StoreOp::create(builder, op.getLoc(), *initial, cell,
                                      initialSlot);
      }
      RuntimeBundle iterator = *iterable;
      iterator.evidenceIteratorCell = cell;
      valueBundles[op.getResult()] = std::move(iterator);
      erase.push_back(op);
      return mlir::success();
    }
  }

  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__iter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getIterable(), op.getResult(), "iter iterable", *methodName,
      /*preferManifestObjectResult=*/true);
}

// `py.next` over a statically evidenced list iterator: bounds-check the
// position cell, select the element from the compile-time evidence, advance
// the cell, and pin the list's liveness with an explicit `__len__` use so the
// borrowed element stays valid throughout the loop.
mlir::LogicalResult
RuntimeBundleLowerer::lowerListEvidenceNext(py::NextOp op,
                                            RuntimeBundle iterator) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value cell = iterator.evidenceIteratorCell;
  mlir::Value slot = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value position =
      mlir::memref::LoadOp::create(builder, loc, cell, slot).getResult();
  mlir::Value size = mlir::arith::ConstantIntOp::create(
      builder, loc, static_cast<std::int64_t>(iterator.sequenceElements.size()),
      64);
  mlir::Value valid = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, position, size);
  mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
  mlir::Value advanced =
      mlir::arith::AddIOp::create(builder, loc, position, one);
  mlir::memref::StoreOp::create(builder, loc, advanced, cell, slot);

  // ⭐ AN EMPTY EVIDENCE SEQUENCE ITERATES ZERO TIMES. With no elements there is
  // nothing to select between, and the selection below reported "list iteration
  // evidence match/value count mismatch" -- an internal sentence for a loop that
  // simply does not run:
  //
  //     class R:
  //         def tag(self, prefix: str, *rest: int) -> str:
  //             for n in rest: ...
  //     R().tag("p")          # no extras, so `rest` is the empty tuple
  //
  // `valid` is already false here (position < 0 is false), so the element is
  // never read; it needs a value only because the op has a result. The same
  // program with one extra argument worked, and so does `for x in ()` at module
  // scope -- that literal takes the runtime path, which handles empty because a
  // runtime length can be zero.
  if (iterator.sequenceElements.empty()) {
    mlir::FailureOr<RuntimeValue> dead =
        RuntimeBundleLowerer::materializeDeadObjectValue(
            op, op.getElement().getType(), "empty sequence iteration");
    if (mlir::failed(dead))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::bindEvidenceObjectResult(
            op, op.getElement(), "list iteration", *dead)))
      return mlir::failure();
    op.getValid().replaceAllUsesWith(valid);
    valueBundles[op.getNext()] = iterator;
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<mlir::Value, 8> matches;
  matches.reserve(iterator.sequenceElements.size());
  for (unsigned index = 0, end = iterator.sequenceElements.size(); index < end;
       ++index) {
    mlir::Value expected = mlir::arith::ConstantIntOp::create(
        builder, loc, static_cast<std::int64_t>(index), 64);
    mlir::Value indexMatches = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, position, expected);
    matches.push_back(
        mlir::arith::AndIOp::create(builder, loc, valid, indexMatches));
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getElement(), iterator.sequenceElements, matches,
          "list iteration", "builtins.IndexError", "list iteration exhausted",
          /*raiseOnMiss=*/false);
  if (mlir::failed(selected))
    return mlir::failure();

  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(iterator.contractName(), "__len__");
  if (!lenMethod)
    return op.emitError()
           << "list iteration needs a runtime __len__ to pin the container";
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&iterator};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(loc, *lenMethod, lenOperands);

  op.getValid().replaceAllUsesWith(valid);
  valueBundles[op.getElement()] = std::move(*selected);
  valueBundles[op.getNext()] = iterator;
  erase.push_back(op);
  return mlir::success();
}

// `py.next` over a runtime-mode list iterator: bounds-check the position
// cell against the runtime length, rebuild the element's physical values from
// its payload box words (immortal dead placeholder words are selected on the
// exhausted branch, so the unconditional retain below is a no-op there),
// retain the element via its contract's `own` primitive, advance the cell,
// and pin the list's liveness with an explicit `__len__` use.
mlir::LogicalResult
RuntimeBundleLowerer::lowerListRuntimeNext(py::NextOp op,
                                           RuntimeBundle iterator) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value cell = iterator.evidenceIteratorCell;
  if (!RuntimeBundleLowerer::containerHasRuntimePayload(iterator))
    return op.emitError() << "runtime list iterator has no physical payload";

  mlir::Type elementContract = op.getElement().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> elementShapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, elementContract,
                                                 "runtime list element");
  if (mlir::failed(elementShapes))
    return mlir::failure();
  // ⭐ A UNION ELEMENT IS BUILT FROM THE BOX. Its physical form starts with a
  // TAG, an i64 and not a memref, so this check refused `for v in xs` over a
  // `list[int | str]` outright -- the read the getitem path answers the same
  // way one line below.
  auto elementUnion = mlir::dyn_cast<py::UnionType>(elementContract);
  if (!elementUnion)
    for (mlir::Type shape : *elementShapes) {
      auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
      if (!memref || memref.getRank() != 1)
        return op.emitError()
               << "iteration over a runtime-mode list of " << elementContract
               << " requires rank-1 memref physical values, got " << shape;
    }

  mlir::Value slot = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value position =
      mlir::memref::LoadOp::create(builder, loc, cell, slot).getResult();
  mlir::FailureOr<mlir::Value> lengthOr =
      RuntimeBundleLowerer::loadContainerLength(op, iterator, "iterator next");
  if (mlir::failed(lengthOr))
    return mlir::failure();
  mlir::Value length = *lengthOr;
  // dict/set iteration: raise RuntimeError when the container's size changed
  // since the iterator was created (CPython's mutation-during-iteration
  // guard; the size at creation sits in cell word 1).
  bool guardsMutation = iterator.contractName() == "builtins.dict" ||
                        iterator.contractName() == "builtins.set" ||
                        iterator.contractName() == "builtins.frozenset";
  if (guardsMutation) {
    mlir::Value initialSlot =
        mlir::arith::ConstantIndexOp::create(builder, loc, 1);
    mlir::Value initial =
        mlir::memref::LoadOp::create(builder, loc, cell, initialSlot)
            .getResult();
    mlir::Value changed = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::ne, length, initial);
    auto changedGuard = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{}, changed, /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(&changedGuard.getThenRegion().front());
      llvm::StringRef message =
          iterator.contractName() == "builtins.dict"
              ? "dictionary changed size during iteration"
              : "Set changed size during iteration";
      if (mlir::failed(
              emitRuntimeException(op, "builtins.RuntimeError", message)))
        return mlir::failure();
    }
    builder.setInsertionPointAfter(changedGuard);
  }
  mlir::Value valid = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, position, length);
  mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
  mlir::Value advanced =
      mlir::arith::AddIOp::create(builder, loc, position, one);
  mlir::memref::StoreOp::create(builder, loc, advanced, cell, slot);

  // Exhausted-branch placeholder: the payload always has at least one
  // allocated slot, so clamping keeps the loads in bounds; the loaded words
  // are replaced by the immortal placeholder's words before use.
  mlir::Value zero64 =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value safe =
      mlir::arith::SelectOp::create(builder, loc, valid, position, zero64)
          .getResult();

  if (runtimeContractName(elementContract) == "builtins.object") {
    // Erased read lane: box the slot's canonical payload handle; the
    // exhausted branch yields the None handle inside the primitive, so no
    // dead placeholder machinery is needed.
    std::optional<RuntimeSymbol> fromSlot =
        manifest.primitive("builtins.object", "from_slot");
    if (!fromSlot)
      return op.emitError()
             << "runtime manifest has no object from_slot primitive";
    mlir::FailureOr<mlir::Value> itemsView =
        RuntimeBundleLowerer::containerInteriorView(
            op, iterator, ContainerInterior::Primary, "iterator element");
    if (mlir::failed(itemsView))
      return mlir::failure();
    mlir::func::CallOp boxed = RuntimeBundleLowerer::createRuntimeCall(
        loc, *fromSlot, mlir::ValueRange{*itemsView, safe, valid});
    RuntimeValue element{elementContract,
                         {boxed.getResult(0)},
                         ownership::logicalOwnershipKind(elementContract,
                                                         /*ownsObject=*/false)};
    if (mlir::failed(bindEvidenceObjectResult(op, op.getElement(),
                                              "runtime list element", element)))
      return mlir::failure();
    std::optional<RuntimeSymbol> lenPin =
        manifest.method(iterator.contractName(), "__len__");
    if (!lenPin)
      return op.emitError()
             << "list iteration needs a runtime __len__ to pin the container";
    llvm::SmallVector<const RuntimeBundle *, 1> pinSources{&iterator};
    llvm::SmallVector<mlir::Value, 4> pinOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *lenPin, pinSources,
                                              pinOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    builder.setInsertionPoint(op);
    RuntimeBundleLowerer::createRuntimeCall(loc, *lenPin, pinOperands);
    op.getValid().replaceAllUsesWith(valid);
    valueBundles[op.getNext()] = iterator;
    erase.push_back(op);
    return mlir::success();
  }

  if (elementUnion) {
    mlir::FailureOr<mlir::Value> itemsView =
        RuntimeBundleLowerer::containerInteriorView(
            op, iterator, ContainerInterior::Primary, "iterator element");
    if (mlir::failed(itemsView))
      return mlir::failure();
    mlir::Value wordsPerSlot = mlir::arith::ConstantIntOp::create(
        builder, loc, box_abi::kWordsPerBox, 64);
    mlir::Value base =
        mlir::arith::MulIOp::create(builder, loc, safe, wordsPerSlot)
            .getResult();
    // ⛔ On the exhausted branch `safe` is 0, so these are slot 0's words --
    // in bounds by the payload's own invariant, and never used: the loop reads
    // the element only where `valid` says there is one. An EMPTY payload's
    // slot 0 is zeroed, whose class id matches no member, so every lane takes
    // its dead arm and nothing is dereferenced.
    mlir::Value classWord =
        box_abi::loadContainerBoxWord(builder, loc, *itemsView, base, 1);
    mlir::Value entityWord = box_abi::loadContainerBoxWord(
        builder, loc, *itemsView, base, box_abi::kEntityWord);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> unionValues =
        RuntimeBundleLowerer::unionValuesFromBoxWords(op, elementUnion,
                                                      classWord, entityWord);
    if (mlir::failed(unionValues))
      return mlir::failure();
    // ⛔ A REFERENCE PER MEMBER, not a borrow. The erased branch beside this
    // one mints an owned box (`from_slot`), and for the same reason: an
    // unpack's element outlives the container when the container is a
    // temporary -- `k, v = make(1)` released the tuple and then increfed a str
    // whose count had already reached zero.
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> owned =
        RuntimeBundleLowerer::retainUnionMemberValues(op, elementUnion,
                                                       *unionValues);
    if (mlir::failed(owned))
      return mlir::failure();
    RuntimeBundle element;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
            op, elementContract, *owned, element,
            ownership::logicalOwnershipKind(elementContract,
                                            /*ownsObject=*/true))))
      return mlir::failure();
    if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getElement(),
                                                      std::move(element))))
      return mlir::failure();
    // ⛔ AND THE CONTAINER IS PINNED PAST THE RETAINS. The release planner puts
    // a temporary's death after its last USE, and the union's lanes are built
    // from the box by arithmetic on an ADDRESS -- nothing the planner reads as
    // a use of the container. It placed the tuple's release between the build
    // and the retain, so the retain ran on a string already at zero.
    if (mlir::failed(pinContainerLiveness(op, iterator,
                                          /*insertAfterOp=*/true)))
      return mlir::failure();
    std::optional<RuntimeSymbol> lenPin =
        manifest.method(iterator.contractName(), "__len__");
    if (!lenPin)
      return op.emitError()
             << "list iteration needs a runtime __len__ to pin the container";
    llvm::SmallVector<const RuntimeBundle *, 1> pinSources{&iterator};
    llvm::SmallVector<mlir::Value, 4> pinOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *lenPin, pinSources,
                                              pinOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    builder.setInsertionPoint(op);
    RuntimeBundleLowerer::createRuntimeCall(loc, *lenPin, pinOperands);
    op.getValid().replaceAllUsesWith(valid);
    valueBundles[op.getNext()] = iterator;
    erase.push_back(op);
    return mlir::success();
  }

  mlir::FailureOr<RuntimeValue> dead =
      RuntimeBundleLowerer::materializeNonOwningDeadObjectValue(
          op.getOperation(), elementContract,
          "exhausted runtime list element");
  if (mlir::failed(dead))
    return mlir::failure();
  // Box-stored contracts keep boxed placeholders too, so the word-select
  // machinery below stays shape-uniform.
  if (std::optional<RuntimeSymbol> box = manifest.primitive(
          runtimeContractName(elementContract), "box")) {
    mlir::func::CallOp boxed =
        RuntimeBundleLowerer::createRuntimeCall(loc, *box, dead->values);
    dead->values.assign(boxed.getResults().begin(), boxed.getResults().end());
  }
  if (elementShapes->size() != dead->values.size())
    return op.emitError()
           << "dead placeholder for " << elementContract
           << " does not match the contract's physical value count";

  mlir::FailureOr<mlir::Value> itemsOr =
      RuntimeBundleLowerer::containerInteriorView(
          op, iterator, ContainerInterior::Primary, "iterator element");
  if (mlir::failed(itemsOr))
    return mlir::failure();
  mlir::Value items = *itemsOr;
  mlir::Value wordsPerSlot =
      mlir::arith::ConstantIntOp::create(builder, loc, box_abi::kWordsPerBox,
                                         64);
  mlir::Value base =
      mlir::arith::MulIOp::create(builder, loc, safe, wordsPerSlot)
          .getResult();
  auto loadBoxWord = [&](std::int64_t wordIndex) -> mlir::Value {
    mlir::Value offset = mlir::arith::ConstantIntOp::create(
        builder, loc, wordIndex, 64);
    mlir::Value word =
        mlir::arith::AddIOp::create(builder, loc, base, offset).getResult();
    mlir::Value index = mlir::arith::IndexCastOp::create(
                            builder, loc, builder.getIndexType(), word)
                            .getResult();
    return mlir::memref::LoadOp::create(builder, loc, items, index)
        .getResult();
  };
  auto pointerWord = [&](mlir::Value value) -> mlir::Value {
    mlir::Value pointerIndex =
        mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                             value);
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getI64Type(), pointerIndex)
        .getResult();
  };
  auto sizeWord = [&](mlir::Value value) -> mlir::Value {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!memref || memref.getRank() != 1)
      return zero64;
    if (memref.hasStaticShape())
      return mlir::arith::ConstantIntOp::create(builder, loc,
                                                memref.getDimSize(0), 64);
    mlir::Value dim =
        mlir::memref::DimOp::create(builder, loc, value, 0).getResult();
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getI64Type(), dim)
        .getResult();
  };

  // ⛔ BRANCHED AND NOT SELECTED. The lanes used to be a word-wise select
  // between the box's and the placeholder's, which reads the box either way and
  // is fine when the words are all the box carries. They are not any more: past
  // the entity they come from a call that DEREFERENCES it, and an exhausted
  // iterator's slot has nothing there. The placeholder cannot answer that call
  // either -- its lanes are separate zeroed allocations, not one entity -- so
  // the two sides have to be different code.
  llvm::SmallVector<mlir::Type, 4> laneTypes(elementShapes->begin(),
                                             elementShapes->end());
  auto pick = mlir::scf::IfOp::create(builder, loc, laneTypes, valid,
                                      /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&pick.getThenRegion().front());
    mlir::Value entityWord = loadBoxWord(box_abi::kEntityWord);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> lanes =
        RuntimeBundleLowerer::lanesFromBoxEntity(
            builder, loc, entityWord, laneTypes,
            runtimeContractName(elementContract), op);
    if (mlir::failed(lanes))
      return mlir::failure();
    mlir::scf::YieldOp::create(builder, loc, *lanes);
    builder.setInsertionPointToStart(&pick.getElseRegion().front());
    llvm::SmallVector<mlir::Value, 4> deadLanes;
    for (auto [index, deadValue] : llvm::enumerate(dead->values))
      deadLanes.push_back(memrefFromBoxWords(
          builder, loc, pointerWord(deadValue), sizeWord(deadValue),
          mlir::cast<mlir::MemRefType>((*elementShapes)[index])));
    mlir::scf::YieldOp::create(builder, loc, deadLanes);
  }
  llvm::SmallVector<mlir::Value, 4> elementValues(pick.getResults().begin(),
                                                  pick.getResults().end());

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, elementContract,
                                                   elementValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  bool valueSemantics = canonical->size() != elementValues.size() ||
                        !llvm::equal(*canonical, elementValues);
  RuntimeValue element{elementContract, *canonical,
                       ownership::logicalOwnershipKind(elementContract,
                                                       /*ownsObject=*/false)};
  if (valueSemantics) {
    // Unboxed elements are copied values; no ownership to root.
    if (mlir::failed(bindEvidenceObjectResult(op, op.getElement(),
                                              "runtime list element", element)))
      return mlir::failure();
  } else {
    std::optional<RuntimeValue> retained =
        RuntimeBundleLowerer::retainEvidenceElement(op, element);
    if (!retained)
      return op.emitError()
             << "iteration over a runtime-mode list of " << elementContract
             << " needs an own primitive in the runtime manifest";
    if (mlir::failed(bindEvidenceObjectResult(op, op.getElement(),
                                              "runtime list element",
                                              *retained)))
      return mlir::failure();
  }

  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(iterator.contractName(), "__len__");
  if (!lenMethod)
    return op.emitError()
           << "list iteration needs a runtime __len__ to pin the container";
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&iterator};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(loc, *lenMethod, lenOperands);

  op.getValid().replaceAllUsesWith(valid);
  valueBundles[op.getNext()] = iterator;
  erase.push_back(op);
  return mlir::success();
}

const RuntimeBundle *RuntimeBundleLowerer::forwardedIteratorBundleFor(
    mlir::Value value,
    llvm::function_ref<bool(const RuntimeBundle &)> carries) const {
  llvm::SmallPtrSet<mlir::Value, 8> seen;
  llvm::SmallVector<mlir::Value, 8> worklist{value};
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!current || !seen.insert(current).second)
      continue;
    if (const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(current);
        bundle && carries(*bundle))
      return bundle;
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(current);
    if (!argument)
      continue;
    mlir::Block *block = argument.getOwner();
    for (mlir::Block *predecessor : block->getPredecessors()) {
      mlir::Operation *terminator = predecessor->getTerminator();
      auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
      if (!branch)
        continue;
      for (unsigned index = 0, end = terminator->getNumSuccessors(); index < end;
           ++index) {
        if (terminator->getSuccessor(index) != block)
          continue;
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(index);
        if (argument.getArgNumber() < operands.size())
          worklist.push_back(operands[argument.getArgNumber()]);
      }
    }
  }
  return nullptr;
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNext(py::NextOp op) {
  const RuntimeBundle *iterator =
      RuntimeBundleLowerer::bundleFor(op.getIterator());
  if (!iterator)
    return op.emitError() << "next iterator has no lowered runtime bundle";
  // ⭐ AN EVIDENCE ITERATOR IS A COMPILE-TIME TOKEN AND A BLOCK ARGUMENT
  // CANNOT CARRY ONE. A list, dict or set is iterated by POSITION through a
  // cell alloca'd once per function -- there is no runtime iterator object --
  // so a block argument that forwards the token has a bundle built from its
  // TYPE, which is the bare `Iterator` protocol. Inside a generator the state
  // machine threads the loop's values through the resume function's block
  // arguments, so every generator that iterated a dict or a set was refused:
  //
  //     def keys(d: "dict[str, int]"):
  //         for k in d:
  //             yield k
  //     # protocol-typed receiver '!py.protocol<"Iterator", [builtins.str]>'
  //     # has no concrete runtime method evidence for __next__
  //
  // The cell is a function-level alloca, so it is valid in every block of the
  // function: following the forwarding edges back to the value that owns the
  // token is the whole repair. The same loop in a plain function was never
  // threaded and always worked.
  RuntimeBundle forwardedIterator;
  if (!iterator->evidenceIteratorCell)
    if (const RuntimeBundle *forwarded =
            RuntimeBundleLowerer::forwardedIteratorBundleFor(
                op.getIterator(), [](const RuntimeBundle &bundle) {
                  return static_cast<bool>(bundle.evidenceIteratorCell);
                })) {
      forwardedIterator = *forwarded;
      iterator = &forwardedIterator;
    }
  if (iterator->evidenceIteratorCell) {
    if (iterator->sequenceElements.empty() && !iterator->sequenceEvidenceBacked)
      return RuntimeBundleLowerer::lowerListRuntimeNext(op, *iterator);
    return RuntimeBundleLowerer::lowerListEvidenceNext(op, *iterator);
  }
  if (iterator->contractName() == "types.GeneratorType") {
    // ⭐ AND THE SAME FORWARDING FOR A GENERATOR'S FRAME TARGET. The state
    // machine threads a loop's iterator through the resume function's block
    // arguments, and a block argument's bundle is rebuilt from its TYPE --
    // which carries the contract and not the target. So a generator iterated
    // INSIDE another generator was refused while the same two functions with
    // the outer one not a generator worked.
    if (iterator->generatorTarget.empty())
      if (const RuntimeBundle *forwarded =
              RuntimeBundleLowerer::forwardedIteratorBundleFor(
                  op.getIterator(), [&](const RuntimeBundle &bundle) {
                    if (bundle.generatorTarget.empty())
                      return false;
                    // ⛔ AND ONLY WHERE THE RESUME'S OPERANDS REACH THIS USE.
                    // A generator's own frame values are defined once, at the
                    // call that created it; when the state machine has split
                    // the loop across resume functions they do not dominate
                    // the `py.next` in the block that reads them, and adopting
                    // the bundle anyway turns a sentence the reader can act on
                    // into "operand #0 does not dominate this use". That shape
                    // is the generator-frame work below, not this forwarding.
                    mlir::DominanceInfo dominance;
                    for (const RuntimeValue &source : bundle.generatorSources)
                      for (mlir::Value value : source.values)
                        if (value && !dominance.properlyDominates(
                                         value, op.getOperation()))
                          return false;
                    return true;
                  })) {
        forwardedIterator = *forwarded;
        iterator = &forwardedIterator;
      }
    if (!iterator->generatorTarget.empty())
      return RuntimeBundleLowerer::lowerSourceGeneratorNext(op, *iterator);
    // ⛔ A generator VALUE with no frame target that the forwarding above
    // could not reach either: the loop lives across a suspension, so the
    // resume's own operands are defined in a block that does not dominate
    // this read. Falling through to the manifest path reported "runtime
    // manifest has no types.GeneratorType.__next__ method" -- a sentence
    // about the manifest for a program that did nothing to it. Named here
    // rather than repaired: carrying the frame across a suspension is the
    // generator-frame work the resume lane note in GeneratorStateMachine.cpp
    // describes.
    //
    // ⛔ The advice has to be the advice that WORKS. `bind it to a local in
    // the same function` was in this sentence and does not help inside a
    // generator -- the local is the same value across the same suspension --
    // and it is the shape a reader most naturally tries next.
    return op.emitError()
           << "a generator returned out of a function cannot be resumed here: "
              "the frame it resumes into is not reachable from this read. "
              "Outside a generator, iterate it directly; inside one, "
              "materialize it first (`for v in list(inner())`)";
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{iterator};
  std::optional<EmittedRuntimeCall> emitted;
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__next__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(emitManifestMethodCall(op, *iterator, *methodName, sources,
                                          /*allowUnusedSources=*/false,
                                          emitted)))
    return mlir::failure();
  if (!emitted->symbol.validResultIndex)
    return op.emitError()
           << "runtime __next__ method must declare valid_result_index";

  mlir::func::CallOp call = emitted->call;
  unsigned validIndex = *emitted->symbol.validResultIndex;
  if (validIndex >= call.getNumResults())
    return op.emitError() << "runtime __next__ valid_result_index is outside "
                             "the result list";
  mlir::Value valid = call.getResult(validIndex);
  if (!valid.getType().isInteger(1))
    return op.emitError() << "runtime __next__ valid result must be an i1";

  std::string elementContract = runtimeContractName(op.getElement().getType());
  if (elementContract.empty())
    elementContract = emitted->symbol.elementContract;
  if (elementContract.empty())
    return op.emitError() << "runtime __next__ element needs a concrete "
                             "manifest element contract";

  std::string nextContract = runtimeContractName(op.getNext().getType());
  if (nextContract.empty())
    nextContract = emitted->symbol.nextContract;
  if (nextContract.empty())
    nextContract = iterator->contractName();
  if (nextContract.empty())
    return op.emitError() << "runtime __next__ next state needs a concrete "
                             "manifest contract";

  llvm::SmallVector<mlir::Value, 4> elementValues;
  for (unsigned index = 0; index < validIndex; ++index)
    elementValues.push_back(call.getResult(index));

  llvm::SmallVector<mlir::Value, 4> nextValues;
  for (unsigned index = validIndex + 1; index < call.getNumResults(); ++index)
    nextValues.push_back(call.getResult(index));

  op.getValid().replaceAllUsesWith(valid);
  if (mlir::failed(assignObjectBundle(
          op, op.getElement(), runtimeContractType(context, elementContract),
          elementValues)))
    return mlir::failure();
  if (mlir::failed(assignObjectBundle(
          op, op.getNext(), runtimeContractType(context, nextContract),
          nextValues)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

template <typename ManagerOp>
mlir::LogicalResult
RuntimeBundleLowerer::lowerContextEnter(ManagerOp op, llvm::StringRef noun,
                                       llvm::StringRef defaultMethod) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                defaultMethod);
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getManager(), op.getResult(), noun, *methodName,
      /*preferManifestObjectResult=*/true);
}

template <typename ManagerOp>
mlir::LogicalResult
RuntimeBundleLowerer::lowerContextExit(ManagerOp op, llvm::StringRef noun,
                                      llvm::StringRef defaultMethod) {
  llvm::SmallVector<mlir::Value, 4> inputs{op.getManager(), op.getExcType(),
                                           op.getExcValue(), op.getTraceback()};
  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  if (mlir::failed(collectObjectSources(op, inputs, noun, sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                defaultMethod);
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/true,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerEnter(py::EnterOp op) {
  return lowerContextEnter(op, "enter manager", "__enter__");
}

mlir::LogicalResult RuntimeBundleLowerer::lowerExit(py::ExitOp op) {
  return lowerContextExit(op, "exit operands need runtime bundles",
                          "__exit__");
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAEnter(py::AEnterOp op) {
  return lowerContextEnter(op, "aenter manager", "__aenter__");
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAExit(py::AExitOp op) {
  return lowerContextExit(op, "aexit operands need runtime bundles",
                          "__aexit__");
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAIter(py::AIterOp op) {
  if (op.getReturnedSelf())
    return RuntimeBundleLowerer::lowerAliasView(op, op.getAsyncIterable(),
                                                op.getResult());
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__aiter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getAsyncIterable(), op.getResult(), "aiter iterable", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerANext(py::ANextOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__anext__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getAsyncIterator(), op.getAwaitable(), "anext iterator",
      *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerRound(py::RoundOp op) {
  if (op.getInputs().empty())
    return op.emitError() << "round requires at least a receiver input";

  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, op.getInputs(), "round input has no lowered runtime bundle",
          sources)))
    return mlir::failure();

  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__round__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerUnarySpecial(mlir::Operation *op, mlir::Value input,
                                        llvm::StringRef methodName,
                                        mlir::Value resultValue) {
  if (methodName == "__repr__") {
    const RuntimeBundle *inputBundle = RuntimeBundleLowerer::bundleFor(input);
    if (!inputBundle)
      return op->emitError() << "repr operand has no lowered runtime bundle";
    if (RuntimeBundleLowerer::needsDefaultObjectRepr(*inputBundle)) {
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, *inputBundle, result)))
        return mlir::failure();
      valueBundles[resultValue] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }
  // `-x`, `~x` and `abs(x)` on a value that still has its machine word are the
  // three that would otherwise put a manifest call in the middle of an
  // otherwise-unboxed function -- and one such call is enough to cost the
  // function its clone, because a clone that allocates cannot be speculated on.
  if (primitiveI64UnarySpecialSupported(methodName)) {
    llvm::SmallVector<mlir::Value, 1> inputs{input};
    llvm::SmallVector<const RuntimeBundle *, 1> sources;
    if (mlir::failed(collectObjectSources(
            op, inputs, "unary special method operand needs a runtime bundle",
            sources)))
      return mlir::failure();
    if (sources.size() == 1 &&
        RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[0])) {
      if (mlir::failed(RuntimeBundleLowerer::lowerPrimitiveI64BinarySpecial(
              op, methodName, sources, resultValue)))
        return mlir::failure();
      erase.push_back(op);
      return mlir::success();
    }
  }
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, input, resultValue, "unary special method operand", methodName,
      /*preferManifestObjectResult=*/true);
}

} // namespace py::lowering
