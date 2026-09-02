#include "Runtime/Core/Lowerer.h"
#include "ArithBuilders.h"

#include "Runtime/ABI/BoxLayout.h"

#include "llvm/Support/Process.h"

#include <algorithm>
#include <iterator>

namespace py::lowering {
namespace {

using lython::common::constantI64;

bool isSequenceCollection(llvm::StringRef contract) {
  return contract == "builtins.list" || contract == "builtins.tuple";
}

bool isI64Payload(mlir::Value value) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  return memref && memref.getRank() == 1 &&
         mlir::isa<mlir::IntegerType>(memref.getElementType()) &&
         mlir::cast<mlir::IntegerType>(memref.getElementType()).getWidth() ==
             64;
}

// The identity of the container an aggregate slot store lands in, allocated on
// first ask and reused after. Unique within the enclosing function, which is the
// scope every ownership walk runs in.
//
// Why NOT a process-wide counter: the runtime lowering pass is free to visit
// functions concurrently, and a shared counter would make the emitted IR depend
// on scheduling. Per-function ids are also the only ones an ownership walk can
// check, since it never leaves the function it is verifying.
std::optional<std::int64_t> aggregateIdentityOf(mlir::Operation *parent) {
  if (!parent)
    return std::nullopt;
  if (auto existing =
          parent->getAttrOfType<mlir::IntegerAttr>(ownership::kAggregateIdAttr))
    return existing.getInt();
  auto function = parent->getParentOfType<mlir::func::FuncOp>();
  if (!function)
    return std::nullopt;
  auto i64 = mlir::IntegerType::get(parent->getContext(), 64);
  std::int64_t next = 1;
  if (auto counter = function->getAttrOfType<mlir::IntegerAttr>(
          ownership::kAggregateIdNextAttr))
    next = counter.getInt();
  function->setAttr(ownership::kAggregateIdNextAttr,
                    mlir::IntegerAttr::get(i64, next + 1));
  parent->setAttr(ownership::kAggregateIdAttr,
                  mlir::IntegerAttr::get(i64, next));
  return next;
}

} // namespace

// Charge to `container` every slot-absorption retain emitted since `anchor`, so
// an ownership walk can name the `parent` of `aggregate(parent, path)`. The
// retains are found by their existing marker rather than returned by the emitter
// because one logical slot store can emit several (a union member per active
// arm, a boxed payload plus its header).
//
// `anchor` is the op that PRECEDED the builder's insertion point before the
// retains were emitted, or null when that point was the block's beginning.
//
// Why NOT the insertion ITERATOR captured up front: an ilist insert happens
// BEFORE the iterator, so the iterator still names the same op afterwards and
// the range between the two captures is always empty. Written that way, this
// emitted the id on the container and no link on any retain; the walk read that
// as "no parent" and fell back to shipped behaviour with no diagnostic at all.
// It surfaced only as `parked=0` in the state-explosion message, which is why
// that counter is printed there.
//
// Why NOT widen `retainAggregateSlot` to return its calls: it recurses over
// union members and box layouts and there is no single call to return; the
// marker attribute already identifies them, and reading it back keeps the parent
// link in one place instead of on every recursion arm.
void chargeSlotRetainsToParent(mlir::OpBuilder &builder, mlir::Block *block,
                               mlir::Operation *anchor,
                               const RuntimeBundle &container) {
  if (!block || builder.getInsertionBlock() != block)
    return;
  llvm::ArrayRef<mlir::Value> lanes = container.physicalValues();
  if (lanes.empty())
    return;
  std::optional<std::int64_t> identity =
      aggregateIdentityOf(lanes.front().getDefiningOp());
  if (!identity)
    return;
  auto attr = mlir::IntegerAttr::get(
      mlir::IntegerType::get(builder.getContext(), 64), *identity);
  mlir::Block::iterator it =
      anchor ? std::next(anchor->getIterator()) : block->begin();
  for (mlir::Block::iterator last = builder.getInsertionPoint(); it != last;
       ++it)
    if (it->hasAttr(ownership::kAggregateRetainAttr) &&
        !it->hasAttr(ownership::kAggregateParentAttr))
      it->setAttr(ownership::kAggregateParentAttr, attr);
}

// The op the builder would insert after, or null at a block's beginning.
mlir::Operation *insertionAnchor(mlir::OpBuilder &builder) {
  mlir::Block *block = builder.getInsertionBlock();
  if (!block)
    return nullptr;
  mlir::Block::iterator point = builder.getInsertionPoint();
  if (point == block->begin())
    return nullptr;
  return &*std::prev(point);
}

namespace {

// Can `from` reach itself without passing through `barrier`? Asks whether a use
// can execute more than once per single production of the value it uses.
bool blockReachesItselfAvoiding(mlir::Block *from, mlir::Block *barrier) {
  if (!from || from == barrier)
    return false;
  llvm::SmallVector<mlir::Block *, 8> worklist(from->getSuccessors().begin(),
                                               from->getSuccessors().end());
  llvm::SmallPtrSet<mlir::Block *, 16> seen;
  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    if (block == from)
      return true;
    if (block == barrier)
      continue;
    if (!seen.insert(block).second)
      continue;
    for (mlir::Block *successor : block->getSuccessors())
      worklist.push_back(successor);
  }
  return false;
}

// Can this collection literal execute more often than `logicalSource` is
// produced? The half of the source-move rule that `valueIsConsumedOnlyBy` cannot
// answer, factored out because the sequence and dict literals need the SAME
// query: a per-container copy would let the two drift, and they already had
// drifted once -- the dict literal shipped deciding the move on the use-set
// answer alone while the sequence literal conjoined this one, which is how the
// nested-loop over-release stayed reachable through `{"a": i}` after `[i]` was
// closed.
//
// `LYTHON_ABLATE_LOOP_LEVEL_SOURCE_MOVE=1` restores the shipped predicate for
// both. Its failure mode is an over-release, so it is for bisecting a regression
// to this rule, never for production.
//
// Why NOT name the toggle after the container: one toggle answering for both is
// what makes an ablation sweep comparable across them.
bool literalMayOutrunSource(mlir::Operation *op, mlir::Value logicalSource) {
  static bool guardEnabled = [] {
    auto value =
        llvm::sys::Process::GetEnv("LYTHON_ABLATE_LOOP_LEVEL_SOURCE_MOVE");
    return !(value && !value->empty() && *value != "0");
  }();
  if (!guardEnabled || !logicalSource || !op->getBlock())
    return false;
  mlir::Block *defBlock = logicalSource.getParentBlock();
  if (mlir::Operation *defOp = logicalSource.getDefiningOp())
    defBlock = defOp->getBlock();
  if (!defBlock || defBlock->getParent() != op->getBlock()->getParent())
    return false;
  return blockReachesItselfAvoiding(op->getBlock(), defBlock);
}

} // namespace

bool RuntimeBundleLowerer::isMutableContainerContractName(
    llvm::StringRef contract) {
  return contract == "builtins.list" || contract == "builtins.dict" ||
         contract == "builtins.set";
}

bool RuntimeBundleLowerer::isSequenceLikeContractName(
    llvm::StringRef contract) {
  return contract == "builtins.list" || contract == "builtins.tuple" ||
         contract == "builtins.set" || contract == "builtins.frozenset";
}

// Compile-time contents evidence is only valid while every mutation of the
// container is visible to this walk. Once the value escapes into code the
// walk cannot see through (a user function, a closure environment), the
// runtime payload becomes the sole authority: evidence-backed lowerings keep
// payload and length in sync at every step, so dropping the evidence is
// always semantics-preserving (reads fall back to the runtime paths).
void RuntimeBundleLowerer::clearContainerEvidence(RuntimeBundle &bundle) {
  bundle.sequenceElements.clear();
  bundle.sequenceElementBundles.clear();
  bundle.sequenceIndices.clear();
  bundle.sequenceEvidenceBacked = false;
  bundle.sequenceCapacity = 0;
  bundle.mappingKeys.clear();
  bundle.mappingKeyBundles.clear();
  bundle.mappingValues.clear();
  bundle.mappingValueBundles.clear();
  bundle.mappingPresent.clear();
  bundle.mappingEvidenceBacked = false;
  bundle.mappingCapacity = 0;
}

void RuntimeBundleLowerer::demoteMutableContainerEvidence(
    RuntimeBundle &bundle) {
  if (bundle.kind != RuntimeBundle::Kind::Object ||
      !isMutableContainerContractName(bundle.contractName()))
    return;
  RuntimeBundleLowerer::clearContainerEvidence(bundle);
}

// A class instance's fieldBundles are a CACHE of what this walk last stored
// into the instance's box-fronted field slots. Every field store lands in the
// slot, so any frame holding the instance can replace a field's value without
// this walk seeing it — and then the cache names a value the program has
// already dropped. Dropping the cache costs a reload from the box words and is
// always semantics-preserving; keeping it past a boundary is a silent
// mis-execution (`def set(b): b.f = 1.5` observed by the caller).
void RuntimeBundleLowerer::dropObjectFieldEvidence(RuntimeBundle &bundle) {
  if (bundle.kind != RuntimeBundle::Kind::Object || bundle.fieldBundles.empty())
    return;
  // Only the fields whose storage IS a box slot, since only those can be
  // reloaded from it. The residual shapes (a union field, an int past the last
  // header word) still keep their value in the instance's lanes, so their
  // evidence is the only description of it and dropping it would lose the value
  // outright -- they are exactly the fields a store still re-roots, and they
  // keep the pre-4a defect along with the pre-4a cache. A header-word field
  // needs no entry either way: its load reads the word and never consults this
  // map.
  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(bundle.objectValue.contract);
  if (!classOp)
    return;
  auto fieldNames = classOp->getAttrOfType<mlir::ArrayAttr>("field_names");
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (!fieldNames || fieldNames.size() != fieldTypes.size())
    return;
  for (auto [index, nameAttr] : llvm::enumerate(fieldNames)) {
    auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    if (!name ||
        !RuntimeBundleLowerer::classFieldStoredBoxed(fieldTypes[index]))
      continue;
    bundle.fieldBundles.erase(name.getValue());
  }
}

void RuntimeBundleLowerer::demoteMutableContainerEvidenceFor(
    mlir::Value value) {
  auto found = valueBundles.find(value);
  if (found == valueBundles.end())
    return;
  demoteMutableContainerEvidence(found->second);
  dropObjectFieldEvidence(found->second);
}

void RuntimeBundleLowerer::demoteMutableContainerArgumentEvidence(
    py::CallOp op) {
  auto elementCanBeMutated = [&](mlir::Value element) {
    const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(element);
    if (!bundle || !bundle->objectValue.contract)
      return true;
    auto mutableContract = [&](mlir::Type contract) {
      return RuntimeBundleLowerer::isMutableContainerContractName(
          runtimeContractName(contract));
    };
    if (auto unionType = mlir::dyn_cast<py::UnionType>(
            bundle->objectValue.contract))
      return llvm::any_of(unionType.getMemberTypes(), mutableContract);
    return mutableContract(bundle->objectValue.contract);
  };
  auto demotePack = [&](mlir::Value packValue) {
    const RuntimeBundle *pack = RuntimeBundleLowerer::bundleFor(packValue);
    if (!pack || pack->kind != RuntimeBundle::Kind::Aggregate)
      return;
    // Copy: demotion writes into valueBundles and would invalidate `pack`.
    llvm::SmallVector<mlir::Value, 8> operands(pack->aggregateOperands.begin(),
                                               pack->aggregateOperands.end());
    for (mlir::Value operand : operands) {
      demoteMutableContainerEvidenceFor(operand);
      // ⭐ And the container it came OUT of, whose element map still describes
      // it as the callee found it.
      //
      //     def grow(v: list[int]) -> None: v.append(2)
      //     data: list[list[int]] = [[1]]
      //     grow(data[0])
      //     print(data[0][1])      # IndexError; CPython prints 2
      //
      // `len(data[0])` was already right -- the runtime length grew -- so only
      // the subscript READ was stale, answered from the outer list's cached
      // one-element description of the inner one. Demoting the argument alone
      // does not reach that copy. Binding the element to a local first was
      // correct, because then the local is what the walk tracks.
      //
      // ⛔ ONLY when the element the callee received can BE mutated. The walk
      // used to run for every subscript argument, and an immutable element
      // took its container's description with it:
      //
      //     def render(v: int | str) -> str: ...
      //     vals = [1, "b"]
      //     print(render(vals[0]))
      //     print(vals[1])   # runtime manifest has no builtins.list.__getitem__
      //
      // A callee cannot mutate an int or a str, so the outer element map is
      // still true after the call -- and for a UNION element the demotion is
      // not a slower read but a refusal, because the runtime tier has no
      // `__getitem__` that returns one. A union whose member is a mutable
      // container still counts as mutable: the tag decides which, and either
      // way the callee holds something it can change.
      if (elementCanBeMutated(operand))
        for (mlir::Value outer = operand; outer;) {
          auto read = outer.getDefiningOp<py::GetItemOp>();
          if (!read)
            break;
          outer = read.getContainer();
          demoteMutableContainerEvidenceFor(outer);
        }
    }
  };
  demotePack(op.getPosargs());
  demotePack(op.getKwvalues());
}

namespace {

constexpr unsigned kPayloadHandleWords =
    static_cast<unsigned>(box_abi::kWordsPerBox);
constexpr unsigned kPayloadOwnedFlagSlot =
    static_cast<unsigned>(box_abi::kOwnedFlagWord);
// ⭐ CPython's `list_resize` (Objects/listobject.c), and it has to be exactly
// that: this figure is what the LOWERING then believes the runtime allocated,
// and a store below a capacity the runtime does not have writes past the
// array. The manifest's LyList_EnsureCapacity computes the same expression for
// the same reason.
//
//   over-allocate mildly and pad to a multiple of 4:
//     new_allocated = (newsize + (newsize >> 3) + 6) & ~3
//
// ⛔ NOT the doubling-from-64 this used to be. A minimum of 64 with a 16-word
// element box meant every list, however short, took 8 KB -- `[i, i + 1, i + 2]`
// in a loop allocated and freed 8 KB per iteration, where CPython's PyList_New
// takes the exact size (three pointers).
std::uint64_t growCapacity(std::uint64_t current, std::uint64_t required) {
  if (current >= required && required >= current / 2)
    return current;
  return (required + (required >> 3) + 6) & ~static_cast<std::uint64_t>(3);
}


mlir::LogicalResult storePayloadWord(mlir::Operation *op,
                                     mlir::OpBuilder &builder,
                                     mlir::Value payload, unsigned index,
                                     mlir::Value word, llvm::StringRef label) {
  if (!payload || !isI64Payload(payload))
    return op->emitError() << label << " payload has invalid type "
                           << (payload ? payload.getType() : mlir::Type());
  builder.setInsertionPoint(op);
  mlir::Value slot = constantIndex(builder, op->getLoc(), index);
  mlir::memref::StoreOp::create(builder, op->getLoc(), word, payload, slot);
  return mlir::success();
}

mlir::LogicalResult storePayloadHandle(mlir::Operation *op,
                                       mlir::OpBuilder &builder,
                                       mlir::Value payload, unsigned index,
                                       llvm::ArrayRef<mlir::Value> words,
                                       llvm::StringRef label) {
  if (words.size() != kPayloadHandleWords)
    return op->emitError() << label << " payload handle must have "
                           << kPayloadHandleWords << " words";
  unsigned base = index * kPayloadHandleWords;
  for (auto [offset, word] : llvm::enumerate(words))
    if (mlir::failed(
            storePayloadWord(op, builder, payload, base + offset, word, label)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult storePayloadHandleAt(mlir::Operation *op,
                                         mlir::OpBuilder &builder,
                                         mlir::Value payload,
                                         mlir::Value logicalIndex,
                                         llvm::ArrayRef<mlir::Value> words,
                                         llvm::StringRef label) {
  if (!payload || !isI64Payload(payload))
    return op->emitError() << label << " payload has invalid type "
                           << (payload ? payload.getType() : mlir::Type());
  if (words.size() != kPayloadHandleWords)
    return op->emitError() << label << " payload handle must have "
                           << kPayloadHandleWords << " words";
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::Value wordsPerSlot =
      constantI64(builder, loc, static_cast<std::int64_t>(kPayloadHandleWords));
  mlir::Value base =
      mlir::arith::MulIOp::create(builder, loc, logicalIndex, wordsPerSlot)
          .getResult();
  mlir::Value baseIndex = mlir::arith::IndexCastOp::create(
                              builder, loc, builder.getIndexType(), base)
                              .getResult();
  for (auto [offset, word] : llvm::enumerate(words)) {
    mlir::Value slot =
        mlir::arith::AddIOp::create(
            builder, loc, baseIndex,
            constantIndex(builder, loc, static_cast<unsigned>(offset)))
            .getResult();
    mlir::memref::StoreOp::create(builder, loc, word, payload, slot);
  }
  return mlir::success();
}

mlir::LogicalResult clearPayloadHandle(mlir::Operation *op,
                                       mlir::OpBuilder &builder,
                                       mlir::Value payload, unsigned index,
                                       llvm::StringRef label) {
  mlir::Value zero = constantI64(builder, op->getLoc(), 0);
  llvm::SmallVector<mlir::Value, 4> words(kPayloadHandleWords, zero);
  return storePayloadHandle(op, builder, payload, index, words, label);
}

} // namespace

// PyList_New(size): exactly `size` slots, no over-allocation. Growth is where
// CPython over-allocates, not construction.
std::uint64_t
RuntimeBundleLowerer::collectionInitialCapacity(std::uint64_t arity) const {
  return arity;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
RuntimeBundleLowerer::objectPayloadHandleWords(mlir::Operation *op,
                                               const RuntimeBundle &value,
                                               bool ownsPayload) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::Value zero = constantI64(builder, loc, 0);
  // ⛔ WORD 0 IS A REFCOUNT EVEN WHEN THERE IS NOTHING TO COUNT. None's handle
  // is otherwise all zeros -- no class, no payload pointer, and word 14 says
  // the slot owns nothing -- but a STANDALONE box (an `object` argument, not a
  // container slot) keeps its own refcount there, and LyObject_DecRef read the
  // zero as "already dead": `def f(v: object) -> int: return 1` called with
  // None aborted in Ly_DecRef without the body touching v. A slot does not read
  // word 0 at all, and LyObject_FromSlot overwrites it with 1 on the way out,
  // so the two uses agree on 1 and disagreed only on 0.
  mlir::Value one = constantI64(builder, loc, 1);
  auto emptyHandle = [&]() {
    llvm::SmallVector<mlir::Value, 4> words(kPayloadHandleWords, zero);
    words[0] = one;
    return words;
  };

  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(value);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "collection payload element is not an object";
  if (concrete->contractName() == "types.NoneType")
    return emptyHandle();
  // ⭐ A UNION IS STORED AS THE MEMBER THE TAG NAMES. The read half rebuilds a
  // union from a slot's box (`unionValuesFromBoxWords`); this is the same
  // question in the other direction, and without it a value that came OUT of
  // one container could not go INTO another -- `config[k] = v` after
  // `k, v = parse(line)`, which is how every such table is built.
  //
  // ⛔ SELECTS, NOT BRANCHES, for the reason the optional FIELD store records:
  // a guarded store leaves the ownership walk with a group it cannot follow.
  // Every member's handle words are computed unconditionally and the TAG picks
  // which set is written. That is sound because an inactive member's lanes are
  // the immortal dead placeholder every producer of a union gives them, so
  // computing its handle reads a constant global and writes nothing.
  if (auto unionType =
          mlir::dyn_cast<py::UnionType>(concrete->objectValue.contract)) {
    if (concrete->physicalValues().empty())
      return op->emitError() << "a union payload element has no tag";
    mlir::Value tag = concrete->physicalValues().front();
    llvm::SmallVector<mlir::Value, 4> words = emptyHandle();
    for (auto [index, member] : llvm::enumerate(unionType.getMemberTypes())) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> laneTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, member,
                                                     "union member ABI");
      if (mlir::failed(laneTypes))
        return mlir::failure();
      if (laneTypes->empty())
        continue;
      mlir::FailureOr<unsigned> offset =
          RuntimeBundleLowerer::unionMemberValueOffset(op, unionType, index,
                                                       "union member ABI");
      if (mlir::failed(offset))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 4> lanes;
      appendValueSlice(concrete->physicalValues(), *offset,
                       static_cast<unsigned>(laneTypes->size()), lanes);
      RuntimeBundle memberBundle;
      if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
              op, member, lanes, memberBundle,
              concrete->objectValue.ownership)))
        return mlir::failure();
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> memberWords =
          RuntimeBundleLowerer::objectPayloadHandleWords(op, memberBundle,
                                                          ownsPayload);
      if (mlir::failed(memberWords))
        return mlir::failure();
      builder.setInsertionPoint(op);
      mlir::Value active = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, tag,
          constantI64(builder, loc, static_cast<std::int64_t>(index)));
      for (auto [slot, word] : llvm::enumerate(words))
        if (slot < memberWords->size())
          words[slot] = mlir::arith::SelectOp::create(
                            builder, loc, active, (*memberWords)[slot], word)
                            .getResult();
    }
    return words;
  }
  // Slots hold CANONICAL payload handles (word 1 = payload class, words 4+
  // = the payload's own memrefs) so hash/eq/repr dispatch reads them
  // uniformly. An opaque erased `object` (no tracked concrete payload)
  // would store a handle-of-box indirection those dispatchers cannot
  // distinguish; reject it loudly rather than mis-execute.
  if (concrete->contractName() == "builtins.object")
    return op->emitError()
           << "a type-erased `object` value cannot be stored in a runtime "
              "container slot yet; give the container a concrete element "
              "type annotation";
  if (concrete->physicalValues().empty())
    return op->emitError()
           << "collection payload element " << concrete->contract
           << " has no physical object handle; materialize it before storing";
  // ⭐ A BOX HOLDS ONE ADDRESS, so a value wider than one physical lane can be
  // stored only if its contract can rebuild the rest from that address. Storing
  // a wider one used to keep the leading lanes and lose the tail, silently, and
  // the same width knocked the class out of boxed-method dispatch, turning a
  // `__repr__` that plainly exists into a runtime abort. Reject at the box, the
  // earliest point where the width is known.
  //
  // A UNION is what reaches this now. A class instance is one lane however many
  // fields it has -- they live in its body -- unless one of them is a union,
  // whose storage is a tag plus every member's lanes: the members do not share
  // an entity, so no single address names them.
  //
  // ⛔ READING a union element out of a container works and WRITING one does
  // not, and the asymmetry is in the direction, not an oversight. A read has
  // the box and rebuilds each member's lanes from it under a tag guard
  // (`unionValuesFromBoxWords`); a write would have to pick the ACTIVE
  // member's box by the tag and there is no single address to put in the slot,
  // so it is a tag-dispatched store and not a store. What still lands here is
  // a union that came OUT of one container going INTO another --
  // `ys: list[int | str] = [xs[0]]`, and `t[1:]` on a `tuple[int, str]`,
  // whose slice element is the union of the members.
  if (concrete->physicalValues().size() > 1 &&
      !RuntimeBundleLowerer::laneWordsPrimitiveFor(concrete->contractName()))
    return op->emitError()
           << "a " << concrete->contract << " value expands to "
           << concrete->physicalValues().size()
           << " physical values and nothing can rebuild them from one address; "
              "it cannot be stored in a container slot or boxed field yet "
              "(a union keeps every member's lanes, and a field holding one "
              "keeps them in its class)";

  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectPhysicalHeader(op, concrete->objectValue);
  if (mlir::failed(header))
    return mlir::failure();
  mlir::Value classSlot = constantIndex(builder, loc, 1);
  mlir::Value payloadClass =
      mlir::memref::LoadOp::create(builder, loc, *header, classSlot)
          .getResult();
  mlir::Value pointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                           *header);
  mlir::Value payloadPointer =
      mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                       pointerIndex)
          .getResult();
  mlir::Value refcount = constantI64(builder, loc, 1);
  mlir::Value owned = constantI64(builder, loc, ownsPayload ? 1 : 0);
  // ⛔ THE LANES ARE NOT COPIED IN. They used to be, a pointer and a size word
  // each, and every reader now rebuilds them from word 2 instead
  // (`lanesFromBoxEntity`) -- so writing them would be maintaining a second
  // copy that nothing consults and that a reallocation can falsify.
  llvm::SmallVector<mlir::Value, 4> words(kPayloadHandleWords, zero);
  words[0] = refcount;
  words[1] = payloadClass;
  words[box_abi::kEntityWord] = payloadPointer;
  words[kPayloadOwnedFlagSlot] = owned;
  return words;
}

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::materializePayloadObjectBundle(
    mlir::Operation *op, const RuntimeBundle &valueRef) {
  // Copies: this function inserts into `valueBundles` and then keeps reading its
  // operand bundles, and the caller's arguments are references INTO that
  // DenseMap -- an insertion that rehashes moves the entry and every later read
  // is freed memory. Found as a live defect on `lowerBoundMethodCall`'s receiver
  // (see CallableOps.cpp); these are the rest of the same audit. Neither of
  // these keys is ever rewritten here, so the copy changes nothing else.
  RuntimeBundle value = valueRef;
  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(value);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "collection payload requires an object bundle";
  if (concrete->contractName() == "types.NoneType")
    return *concrete;
  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(*concrete)) {
    builder.setInsertionPoint(op);
    mlir::FailureOr<RuntimeValue> object =
        RuntimeBundleLowerer::materializePrimitiveI64Object(op, *concrete);
    if (mlir::failed(object))
      return mlir::failure();
    RuntimeBundle materialized =
        RuntimeBundle::object(concrete->objectValue.contract, object->values);
    materialized.copyEvidenceFrom(*concrete);
    return materialized;
  }
  // A contract with a `box` primitive stores its boxed form (e.g. bool: the
  // canonical i1 boxes to an immortal singleton header) — the slot layout
  // requires a header-fronted value group.
  if (!concrete->physicalValues().empty() &&
      !ownership::isObjectHeaderLikeType(
          concrete->physicalValues().front().getType())) {
    if (std::optional<RuntimeSymbol> box =
            manifest.primitive(concrete->contractName(), "box")) {
      builder.setInsertionPoint(op);
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), *box, concrete->physicalValues());
      RuntimeBundle materialized = RuntimeBundle::object(
          concrete->objectValue.contract, call.getResults());
      materialized.copyEvidenceFrom(*concrete);
      return materialized;
    }
  }
  if (concrete->physicalValues().empty())
    return op->emitError() << "collection payload element "
                           << concrete->contract
                           << " has no physical object handle";
  return *concrete;
}

// One growth step for either payload kind: the capacity the bundle tracks and
// the contract that owns `ensure_capacity` are the whole difference. The two
// kinds record capacity in different fields because one bundle can be both.
mlir::LogicalResult RuntimeBundleLowerer::ensurePayloadCapacity(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    llvm::StringRef label, llvm::StringRef contractName,
    std::uint64_t RuntimeBundle::*capacity) {
  if (container.*capacity && index < container.*capacity)
    return mlir::success();
  // ⛔ An unknown capacity used to be assumed to be at least 64. With the
  // allocator taking the exact size (CPython's PyList_New) that assumption is
  // false, and the store it skipped the growth for would run off the array.

  std::optional<RuntimeSymbol> ensure =
      manifest.primitive(contractName, "ensure_capacity");
  if (!ensure)
    return op->emitError() << label
                           << " payload capacity is exceeded, but the runtime "
                           << "manifest has no ensure_capacity primitive";

  builder.setInsertionPoint(op);
  mlir::Value required =
      constantI64(builder, op->getLoc(), static_cast<std::int64_t>(index) + 1);
  llvm::SmallVector<mlir::Value, 6> operands(container.physicalValues().begin(),
                                             container.physicalValues().end());
  operands.push_back(required);
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *ensure, operands);
  RuntimeBundle updated;
  // A handle-fronted contract's ensure_capacity returns nothing: it wrote the
  // new bases into the handle, so the bundle it grew is still the right one.
  if (mlir::failed(RuntimeBundleLowerer::rebindMutatedContainer(
          op, container, call.getResults(), updated)))
    return mlir::failure();
  updated.copyEvidenceFrom(container);
  // An unknown old capacity is taken as 0: the growth then depends only on
  // what is required, which is the figure the runtime also computes from it.
  updated.*capacity =
      growCapacity(container.*capacity, static_cast<std::uint64_t>(index) + 1);
  container = std::move(updated);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::ensureSequencePayloadCapacity(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    llvm::StringRef label) {
  return ensurePayloadCapacity(op, container, index, label,
                               container.contractName(),
                               &RuntimeBundle::sequenceCapacity);
}

mlir::LogicalResult RuntimeBundleLowerer::ensureDictPayloadCapacity(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  return ensurePayloadCapacity(op, container, index, "dict", "builtins.dict",
                               &RuntimeBundle::mappingCapacity);
}

// ⛔ A USE-SET FACT IS NOT A PROXY FOR AN EXECUTION-FREQUENCY FACT, and this
// predicate's only callers need the second one. "Every use of this value is this
// op" says nothing about how many times that ONE op runs per single production of
// the value. A literal nested in a loop the source is defined OUTSIDE of has
// exactly one use, satisfies this predicate, and executes N times -- so the
// source's single token would be handed to a container N times.
//
// NEITHER `initializeSequencePayload` NOR `initializeDictPayload` decides the
// move on this answer alone; both conjoin `literalMayOutrunSource` above.
//
// The dict side was labelled a KNOWN GAP here until 2026-07-28 -- measured
// rather than reasoned about, because an unmeasured shape is not an input to a
// shipping decision. The measurement found the gap real (11 of 25 enumerated
// shapes aborted, 1 was silently wrong) AND found a second defect the sequence
// side never had: no dedup of a source filling several slots. So "same code,
// same defect" would have been wrong in both directions -- the dict path was
// also missing something the sequence path already had.
//
// ⚠️ What this predicate still cannot answer, and what therefore must not be
// built on it: it says nothing about whether the container ever RELEASES what it
// retained. A literal that moves the token correctly still leaks one object per
// execution (measured 2026-07-28 on both container kinds, unbounded, pre-dating
// all of this work). That is a teardown-accounting defect, tracked separately;
// do not read a correct move decision here as a balanced one.
bool RuntimeBundleLowerer::valueIsConsumedOnlyBy(mlir::Value value,
                                                 mlir::Operation *op) {
  if (!value || !op)
    return false;
  for (mlir::OpOperand &use : value.getUses())
    if (use.getOwner() != op)
      return false;
  return true;
}

mlir::LogicalResult RuntimeBundleLowerer::initializeSequencePayload(
    mlir::Operation *op, RuntimeBundle &container,
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> elements,
    llvm::ArrayRef<mlir::Value> logicalSources) {
  if (!isSequenceCollection(container.contractName()))
    return mlir::success();
  container.sequenceCapacity =
      RuntimeBundleLowerer::collectionInitialCapacity(elements.size());
  container.sequenceEvidenceBacked = true;
  // Every slot retains its own claim. Whether the SOURCE's claim also moves
  // into the container depends on what the source is:
  //
  //   - a temporary this literal is the only user of (`["a", "b"]`) hands its
  //     token over, or nothing would ever release it (the exception path out
  //     of a later call would leak it — the affine-ownership verifier's
  //     "still owned when ... unwinds" rule);
  //   - a value that outlives the literal (`t = (s,)`, where `s` is read
  //     again) keeps its claim. Taking it used to leave the binding with no
  //     reference at all, so `s` dangled the moment the container died and
  //     the next read silently printed the empty string. The refcount pass
  //     places that release at the source's real last use instead.
  //
  // ⚠️ THE "KEEPS ITS CLAIM" BRANCH WAS UNREACHABLE INSIDE A LOOP UNTIL
  // 2026-07-28, because the single-use predicate below always moved the token
  // first. The moment the frequency query started declining those moves, two
  // things that had never run had to be repaired with it: the affine walk did not
  // converge on the retain this branch emits (fixed by charging it to the holder
  // -- verifier/runtime/AffineOwnership.cpp), and the container's contents
  // evidence became a lie (fixed at the end of this function). Neither was a new
  // defect; both were invariants that held only because something else happened
  // to hold, the same shape as the `borrowEdgeRetainIsSpellable` note in
  // Passes/Ownership.cpp.
  //
  // One value can fill several slots (`(j, j)`): every slot retains, but the
  // source hands over the ONE token it holds, so the move is deduped.
  llvm::SmallPtrSet<void *, 4> movedSources;
  bool declinedLoopLevelMove = false;
  for (auto [index, element] : llvm::enumerate(elements)) {
    if (!element)
      continue;
    mlir::FailureOr<RuntimeBundle> payload =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, *element);
    if (mlir::failed(payload))
      return mlir::failure();
    mlir::Block *retainBlock = builder.getInsertionBlock();
    mlir::Operation *retainAnchor = insertionAnchor(builder);
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payload, "sequence.literal")))
      return mlir::failure();
    chargeSlotRetainsToParent(builder, retainBlock, retainAnchor, container);
    if (mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
            op, container, static_cast<unsigned>(index), *payload)))
      return mlir::failure();
    mlir::Value logicalSource =
        index < logicalSources.size() ? logicalSources[index] : mlir::Value{};
    bool sourceIsTemporary =
        logicalSources.empty() || !logicalSource ||
        RuntimeBundleLowerer::valueIsConsumedOnlyBy(logicalSource, op);
    // A LITERAL THAT CAN RUN MORE OFTEN THAN ITS SOURCE IS PRODUCED MAY NOT TAKE
    // THE SOURCE'S TOKEN. `valueIsConsumedOnlyBy` above answers a use-SET
    // question and the move needs an execution-FREQUENCY one; the CFG query here
    // supplies the missing half. `for i in range(4): for j in range(4):
    // ys = [i, j]` has one use of `i` that runs four times per production, and
    // handing the one token over four times over-released it: shipped, it
    // alternated between `Ly_DecRef observed non-positive refcount` and silently
    // printing 0 for the same binary, and it survived `--release`. The immortal
    // small-int cache {0, 1, 2} absorbed it while the loop variable stayed in
    // that set, which is why it read as a threshold at n=4 rather than as a
    // frequency bug (tests/probe/seqlit_cache_boundary_pair.py holds the pair
    // that separates the two readings).
    //
    // `LYTHON_ABLATE_LOOP_LEVEL_SOURCE_MOVE=1` restores the shipped predicate
    // (see literalMayOutrunSource above, which both literal kinds share).
    //
    // ⛔ Why this predicate ALONE is not the repair, measured 2026-07-28: it
    // needs the parked-retain modelling in verifier/runtime/AffineOwnership.cpp
    // (or the affine walk does not converge on the no-move shape it creates) AND
    // the evidence demotion at the end of this function (or the read-back of an
    // element the container does not own becomes a silent wrong answer). Landing
    // any two of the three was measured worse than landing none.
    if (sourceIsTemporary && literalMayOutrunSource(op, logicalSource)) {
      sourceIsTemporary = false;
      declinedLoopLevelMove = true;
    }
    if (sourceIsTemporary &&
        payload->objectValue.ownership == ownership::OwnershipKind::Own) {
      // Key on the ELEMENT's physical identity, not the materialized
      // payload's: materialization may mint a fresh per-slot box view, and
      // two slots of one source must still dedupe.
      mlir::ValueRange sourceValues = element->physicalValues().empty()
                                          ? payload->physicalValues()
                                          : element->physicalValues();
      bool firstOccurrence =
          sourceValues.empty() ||
          movedSources.insert(sourceValues.front().getAsOpaquePointer()).second;
      // FIXED 2026-08-14. `grid[1][0] = 9` leaked 52 B with no read of the
      // value at all, and this release was suspected because it is the
      // source's death. It is correct: the literal owns the reference now.
      //
      // ⭐ THE COUNT WAS WRONG, NOT THIS RELEASE. The element ends up with two
      // references -- this token and the `aggregate_retain` for the slot --
      // and the store adds a second aggregate release, so two discharge two.
      // `releaseOwnedGroupByLiveness` read "one reference in hand" against two
      // consumes and inserted an unfold retain nothing pays for
      // (Passes/Ownership.cpp, "WHAT IS IN HAND IS NOT ALWAYS ONE").
      //
      // Which is why the three repairs measured here all missed. Each looked
      // for a way to suppress a RELEASE:
      //   - suppress the later read's frame token when the value was absorbed:
      //     52 B became 8420 B -- that token also releases the value on paths
      //     where the literal did NOT take it over.
      //   - treat a literal-absorbed element as borrowed in
      //     `frameKeepsOwnedSourceOf`: no effect; not on this path.
      //   - demote the SOURCE bundle's ownership here: no effect either.
      // None of them could work: there was one release too FEW for the retains
      // in hand, not one too many.
      if (firstOccurrence &&
          mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
              op, *payload, "sequence.literal.source")))
        return mlir::failure();
    }
    RuntimeBundle stored = payload->withObjectOwnership(
        ownership::logicalOwnershipKind(payload->objectValue.contract,
                                        /*ownsObject=*/false));
    if (index < container.sequenceElementBundles.size())
      container.sequenceElementBundles[index] =
          std::make_shared<RuntimeBundle>(stored);
    if (index < container.sequenceElements.size())
      container.sequenceElements[index] = stored.objectValue;
  }
  // A DECLINED MOVE INVALIDATES THE COMPILE-TIME CONTENTS EVIDENCE. Evidence
  // resolves `ys[0]` to the very SSA value that was stored, so the reader gets
  // the element with no reference of its own -- fine while the literal owned the
  // source's token (the container was then the sole owner and outlived every
  // read), and wrong the moment the source keeps its claim: binding the read to
  // a name that outlives the container leaves that name dangling.
  //
  // Measured: with the move declined and the evidence kept,
  // `for i in range(3,4): for j in range(2): ys = [i]; v = ys[0]` prints a freed
  // value with exit 0 -- a refusal on shipped turned into a SILENT WRONG ANSWER,
  // the one direction this family may never move in
  // (tests/probe/seqlit_nested_read_only_silent.py).
  //
  // Why NOT retain at the read instead: the read is lowered from the evidence in
  // another pass, and a retain there would double-count every read the evidence
  // path already resolves correctly for a container that DID take the token.
  // Dropping the evidence puts the element back behind the runtime accessor,
  // which returns a value with its own reference, and it is local to the exact
  // literal whose ownership rule just changed.
  //
  // Why NOT drop it for every literal: the evidence is what turns a literal's
  // element access into no code at all, and the declined move is a rare shape (a
  // literal in a loop its element's source is defined outside of).
  if (declinedLoopLevelMove) {
    container.sequenceElements.clear();
    container.sequenceElementBundles.clear();
    container.sequenceIndices.clear();
    container.sequenceEvidenceBacked = false;
  }
  return mlir::success();
}

// ⭐ ONE PAYLOAD SLOT STORE. The order here is the rule: grow FIRST, take the
// interior view AFTER. Growing may reallocate the array, and a view taken
// before the growth names the old one -- which is why the view is re-derived
// here and not passed in. Three callers each had this written out, and the
// reason was written on one of them.
mlir::LogicalResult RuntimeBundleLowerer::storePayloadSlot(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    const RuntimeBundle &element, ContainerInterior interior,
    llvm::StringRef label, llvm::StringRef capacityContract,
    std::uint64_t RuntimeBundle::*capacity) {
  if (mlir::failed(RuntimeBundleLowerer::ensurePayloadCapacity(
          op, container, index, label, capacityContract, capacity)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, element);
  if (mlir::failed(words))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> slots =
      RuntimeBundleLowerer::containerInteriorView(op, container, interior,
                                                  label);
  if (mlir::failed(slots))
    return mlir::failure();
  return storePayloadHandle(op, builder, *slots, index, *words, label);
}

mlir::LogicalResult RuntimeBundleLowerer::storeSequencePayloadElement(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    const RuntimeBundle &element) {
  if (!isSequenceCollection(container.contractName()))
    return mlir::success();
  return storePayloadSlot(op, container, index, element,
                          ContainerInterior::Primary, container.contractName(),
                          container.contractName(),
                          &RuntimeBundle::sequenceCapacity);
}

mlir::LogicalResult RuntimeBundleLowerer::storeSequencePayloadElementAt(
    mlir::Operation *op, RuntimeBundle &container, mlir::Value logicalIndex,
    const RuntimeBundle &element) {
  if (!isSequenceCollection(container.contractName()))
    return op->emitError() << container.contractName()
                           << " is not a sequence collection";
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, element);
  if (mlir::failed(words))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> items =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Primary,
          container.contractName());
  if (mlir::failed(items))
    return mlir::failure();
  return storePayloadHandleAt(op, builder, *items, logicalIndex, *words,
                              container.contractName());
}

mlir::LogicalResult RuntimeBundleLowerer::clearSequencePayloadElement(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (!isSequenceCollection(container.contractName()))
    return mlir::success();
  if (container.sequenceCapacity && index >= container.sequenceCapacity)
    return op->emitError() << container.contractName()
                           << " payload clear index exceeds capacity";
  mlir::FailureOr<mlir::Value> items =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Primary,
          container.contractName());
  if (mlir::failed(items))
    return mlir::failure();
  return clearPayloadHandle(op, builder, *items, index,
                            container.contractName());
}

mlir::LogicalResult RuntimeBundleLowerer::initializeDictPayload(
    mlir::Operation *op, RuntimeBundle &container,
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> keys,
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> values,
    llvm::ArrayRef<mlir::Value> logicalKeySources,
    llvm::ArrayRef<mlir::Value> logicalValueSources) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (keys.size() != values.size())
    return op->emitError() << "dict payload key/value count mismatch";
  container.mappingCapacity =
      RuntimeBundleLowerer::collectionInitialCapacity(keys.size());
  container.mappingEvidenceBacked = true;
  // ONE dedup set for both sides and all entries: the move hands over the ONE
  // token a source holds, so a source that fills several slots may only be
  // released once. Not per-side and not per-entry, because `{"a": x, "b": x}`
  // repeats across ENTRIES while the sequence literal's `(j, j)` repeats within
  // one -- the same defect reached by a different spelling.
  llvm::SmallPtrSet<void *, 4> movedSources;
  bool declinedLoopLevelMove = false;
  for (auto [index, key] : llvm::enumerate(keys)) {
    if (!key || !values[index])
      return op->emitError() << "dict payload entry has no object evidence";
    mlir::FailureOr<RuntimeBundle> payloadKey =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, *key);
    if (mlir::failed(payloadKey))
      return mlir::failure();
    mlir::FailureOr<RuntimeBundle> payloadValue =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op,
                                                             *values[index]);
    if (mlir::failed(payloadValue))
      return mlir::failure();
    mlir::Block *retainBlock = builder.getInsertionBlock();
    mlir::Operation *retainAnchor = insertionAnchor(builder);
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadKey, "dict.literal.key")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadValue, "dict.literal.value")))
      return mlir::failure();
    chargeSlotRetainsToParent(builder, retainBlock, retainAnchor, container);
    if (mlir::failed(RuntimeBundleLowerer::storeDictKeyPayload(
            op, container, static_cast<unsigned>(index), *payloadKey)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
            op, container, static_cast<unsigned>(index), *payloadValue)))
      return mlir::failure();
    // Same temporary-only move rule as initializeSequencePayload: a literal
    // may take over a temporary's token, but `d = {"k": s}` must leave the
    // local `s` its claim or `s` dangles once the dict dies.
    //
    // ⛔ THIS DECIDED THE MOVE ON `valueIsConsumedOnlyBy` ALONE UNTIL
    // 2026-07-28, so the nested-loop over-release the sequence literal was
    // repaired for stayed reachable through the dict literal. Measured on
    // bcfbbf9 over 25 enumerated shapes: 11 aborted (`Ly_DecRef observed
    // non-positive refcount` / SIGSEGV), 1 was a silent wrong answer, 1 was
    // refused by state explosion, and the cache axis reproduced exactly --
    // `range(0,3)` clean and `range(3,6)` aborting at the same trip count,
    // because the immortal small-int cache {0, 1, 2} absorbs the over-release.
    //
    // Only the VALUE side can carry arbitrary provenance. A dict literal reaches
    // this function only when every key is a `py.str_constant`
    // (PackAndBindingOps.cpp), so `{i: v}` never gets here -- one non-static key
    // sends the whole literal down the `setitem_box` probe path. The key side is
    // still routed through the same rule rather than special-cased, because
    // "constants are never Own" is a property of another pass, not of this one.
    auto moveSourceIfTemporary =
        [&](const RuntimeBundle &payload, const RuntimeBundle &element,
            llvm::ArrayRef<mlir::Value> sources,
            llvm::StringRef slot) -> mlir::LogicalResult {
      if (payload.objectValue.ownership != ownership::OwnershipKind::Own)
        return mlir::success();
      mlir::Value logicalSource =
          index < sources.size() ? sources[index] : mlir::Value{};
      if (!sources.empty() && logicalSource &&
          !RuntimeBundleLowerer::valueIsConsumedOnlyBy(logicalSource, op))
        return mlir::success();
      if (literalMayOutrunSource(op, logicalSource)) {
        declinedLoopLevelMove = true;
        return mlir::success();
      }
      // LYTHON_ABLATE_DICT_SOURCE_MOVE_DEDUP=1 restores the shipped behaviour
      // (release once per SLOT rather than once per SOURCE). Its failure mode is
      // an over-release of a source that fills two entries, which reached a
      // SILENT WRONG ANSWER, not an abort: `x = "q" + "rs"; d = {"a": x, "b": x}`
      // printed `len(d["a"])` as 0 on 5/5 reps.
      static bool dedupEnabled =
          !llvm::sys::Process::GetEnv("LYTHON_ABLATE_DICT_SOURCE_MOVE_DEDUP")
               .has_value();
      // Key on the ELEMENT's physical identity, not the materialized payload's:
      // materialization mints a fresh per-slot box view, so two slots fed by one
      // source would not compare equal through the payload.
      mlir::ValueRange sourceValues = element.physicalValues().empty()
                                          ? payload.physicalValues()
                                          : element.physicalValues();
      if (dedupEnabled && !sourceValues.empty() &&
          !movedSources.insert(sourceValues.front().getAsOpaquePointer()).second)
        return mlir::success();
      return RuntimeBundleLowerer::releaseAggregateSlot(op, payload, slot);
    };
    if (mlir::failed(moveSourceIfTemporary(*payloadKey, *key, logicalKeySources,
                                           "dict.literal.key.source")))
      return mlir::failure();
    if (mlir::failed(moveSourceIfTemporary(*payloadValue, *values[index],
                                           logicalValueSources,
                                           "dict.literal.value.source")))
      return mlir::failure();
    RuntimeBundle storedKey = payloadKey->withObjectOwnership(
        ownership::logicalOwnershipKind(payloadKey->objectValue.contract,
                                        /*ownsObject=*/false));
    RuntimeBundle storedValue =
        payloadValue->withObjectOwnership(ownership::logicalOwnershipKind(
            payloadValue->objectValue.contract, /*ownsObject=*/false));
    if (index < container.mappingKeyBundles.size())
      container.mappingKeyBundles[index] =
          std::make_shared<RuntimeBundle>(storedKey);
    if (index < container.mappingValueBundles.size())
      container.mappingValueBundles[index] =
          std::make_shared<RuntimeBundle>(storedValue);
    if (index < container.mappingValues.size())
      container.mappingValues[index] = storedValue.objectValue;
  }
  // A DECLINED MOVE INVALIDATES THE COMPILE-TIME CONTENTS EVIDENCE, for the same
  // reason it does on the sequence side (see the end of
  // initializeSequencePayload): evidence resolves `d["a"]` to the very SSA value
  // that was stored, which is only sound while the container owns the source's
  // token. With the move declined the source keeps its claim, so binding the read
  // to a name that outlives the container leaves that name dangling.
  //
  // Measured 2026-07-28 on the dict side specifically: with the frequency query
  // in and this demotion out, `for i in range(3,6): for j in range(2):
  // d = {"a": i}; acc += d["a"]` printed a wrong sum with exit 0. Dropping the
  // evidence sends the read back through lowerDictEvidenceGetItem's
  // `mappingKeys.empty()` bail-out to the runtime accessor, which returns a value
  // with a reference of its own.
  //
  // Why NOT call demoteMutableContainerEvidence: that also zeroes
  // `mappingCapacity`, which is not evidence but the PHYSICAL extent of the
  // arrays this function just wrote into -- storeDictValuePayload would then
  // re-grow an already-grown dict.
  //
  // Why NOT demote every dict literal: the evidence is what turns `d["a"]` into
  // no code at all, and a declined move is a rare shape (a literal in a loop its
  // value's source is defined outside of).
  //
  // LYTHON_ABLATE_DICT_EVIDENCE_DEMOTION=1 keeps the evidence, which is the
  // two-of-three combination the sequence side measured as WORSE than shipping
  // nothing. It exists so that combination stays reproducible from one binary.
  static bool demotionEnabled =
      !llvm::sys::Process::GetEnv("LYTHON_ABLATE_DICT_EVIDENCE_DEMOTION")
           .has_value();
  if (declinedLoopLevelMove && demotionEnabled) {
    container.mappingKeys.clear();
    container.mappingKeyBundles.clear();
    container.mappingValues.clear();
    container.mappingValueBundles.clear();
    container.mappingPresent.clear();
    container.mappingEvidenceBacked = false;
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::storeDictKeyPayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    const RuntimeBundle &key) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  return storePayloadSlot(op, container, index, key,
                          ContainerInterior::Primary, "dict keys",
                          "builtins.dict", &RuntimeBundle::mappingCapacity);
}

mlir::LogicalResult RuntimeBundleLowerer::storeDictValuePayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    const RuntimeBundle &value) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (mlir::failed(storePayloadSlot(op, container, index, value,
                                    ContainerInterior::Secondary,
                                    "dict values", "builtins.dict",
                                    &RuntimeBundle::mappingCapacity)))
    return mlir::failure();
  // The present word is what makes the slot findable; it is stored last so a
  // failed value store never leaves a slot claiming an entry it has not got.
  mlir::FailureOr<mlir::Value> present =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Present, "dict present");
  if (mlir::failed(present))
    return mlir::failure();
  return storePayloadWord(op, builder, *present, index,
                          constantI64(builder, op->getLoc(), 1),
                          "dict present");
}

mlir::LogicalResult RuntimeBundleLowerer::clearDictKeyPayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (container.mappingCapacity && index >= container.mappingCapacity)
    return op->emitError() << "dict payload clear index exceeds capacity";
  mlir::FailureOr<mlir::Value> keys =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Primary, "dict keys");
  if (mlir::failed(keys))
    return mlir::failure();
  return clearPayloadHandle(op, builder, *keys, index, "dict keys");
}

mlir::LogicalResult RuntimeBundleLowerer::clearDictValuePayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (container.mappingCapacity && index >= container.mappingCapacity)
    return op->emitError() << "dict payload clear index exceeds capacity";
  mlir::Value zero = constantI64(builder, op->getLoc(), 0);
  mlir::FailureOr<mlir::Value> valuesArray =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Secondary, "dict values");
  if (mlir::failed(valuesArray))
    return mlir::failure();
  if (mlir::failed(
          clearPayloadHandle(op, builder, *valuesArray, index, "dict values")))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> present =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Present, "dict present");
  if (mlir::failed(present))
    return mlir::failure();
  return storePayloadWord(op, builder, *present, index, zero, "dict present");
}

mlir::LogicalResult RuntimeBundleLowerer::clearDictPayloadEntry(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (mlir::failed(
          RuntimeBundleLowerer::clearDictKeyPayload(op, container, index)))
    return mlir::failure();
  return RuntimeBundleLowerer::clearDictValuePayload(op, container, index);
}

} // namespace py::lowering
