// The calling convention of a memref: what its descriptor needs, not all of
// it.
//
// MLIR's func-to-LLVM conversion passes every memref as its descriptor,
// five words for a rank-1 one (allocated and aligned pointer, offset, size,
// stride). An identity-layout memref's offset is 0 and its stride 1 by its
// type, and a static shape fixes its size too. So a static-shape one crosses
// a call as `!llvm.ptr` and a rank-1 dynamic one as `!llvm.struct<(ptr,
// iN)>` (one value, which the backend passes in two registers); each side of
// the boundary converts with a cast marked `kNarrowCastAttr`, and the LLVM
// conversion lowers it to the aligned pointer (and size) one way and to a
// descriptor rebuilt from them the other, whose allocated pointer is the
// aligned one.
//
// Which calls:
//   - a private function with a body that is only called directly, in the
//     module being lowered (a program, a pre-lowered runtime module);
//   - in a program, every declaration: what a program declares and does not
//     define is the native runtime's, which narrows its public definitions
//     the same way (`NarrowingScope`, RuntimeLibrary.cpp);
//   - the native runtime's functions that spell a descriptor out as its five
//     words, once they are rewritten to take the narrow form
//     (`narrowExpandedDescriptors`).
//
// ⭐ A DEALLOCATION FREES THE ALIGNED POINTER. What is freed is an owned
// reference, and an owned reference points at the start of its allocation
// (the proof's site-address-recovers), so the two pointers are one there; a
// rebuilt descriptor's allocated pointer is right for every deallocation.
// LYTHON_VERIFY_BARE_POINTER_ABI=1 checks that they are one at each.
//
// ⭐ THE ALLOCATED POINTER IS KEPT ONLY WHERE IT IS BOTH DIFFERENT AND READ.
// A lane of a block or an image in a table is a view whose allocated pointer
// is the block's or the table's; `OffsetAnalysis` keeps the descriptor at a
// position such a value reaches and whose allocated pointer something other
// than a deallocation reads.
//
// ⛔ Not upstream's bare-pointer convention: it is one switch for a whole
// conversion, and it refuses every function with a dynamic-shape memref.
// ⛔ Not a slot function or a slot hook: the type object's hook calls a slot
// through a pointer with the hook's own lowered type, which nothing here
// rewrites.
// ⛔ Not a thin raise wrapper: the exception lowering redirects a call to one
// to `LyEH_RecordException` only when the two types are the same.

#include "Passes/Runtime/Cleanup/Transforms.h"

#include "Common/TypeObjects.h"

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <atomic>
#include <cstdlib>
#include <string>

namespace py::lowering::runtime::cleanup {
namespace {

enum class Carrier { Descriptor, Pointer, PointerAndSize };

Carrier carrierOf(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || !memref.getLayout().isIdentity() || memref.getMemorySpace())
    return Carrier::Descriptor;
  if (memref.hasStaticShape())
    return Carrier::Pointer;
  return memref.getRank() == 1 ? Carrier::PointerAndSize : Carrier::Descriptor;
}

bool narrows(mlir::Type type) { return carrierOf(type) != Carrier::Descriptor; }

// What carries a memref across a call: the aligned pointer, and for a
// dynamic shape its size after it.
llvm::SmallVector<mlir::Type, 2>
carrierTypes(mlir::MLIRContext *context, Carrier carrier, unsigned indexBits) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(context);
  if (carrier == Carrier::Pointer)
    return {ptr};
  return {ptr, mlir::IntegerType::get(context, indexBits)};
}

bool envSet(const char *name) {
  const char *value = std::getenv(name);
  return value && *value && llvm::StringRef(value) != "0";
}

void collectSlotNames(mlir::Attribute attr, llvm::StringSet<> &names) {
  if (auto name = mlir::dyn_cast<mlir::StringAttr>(attr)) {
    names.insert(name.getValue());
  } else if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute element : array)
      collectSlotNames(element, names);
  } else if (auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
    for (mlir::NamedAttribute entry : dictionary)
      collectSlotNames(entry.getValue(), names);
  }
}

// `isThinRaiseWrapper` (EH.cpp), read at the func level.
bool isThinRaiseWrapper(mlir::func::FuncOp function) {
  if (!function.getBody().hasOneBlock())
    return false;
  mlir::Block &entry = function.getBody().front();
  auto call = mlir::dyn_cast<mlir::func::CallOp>(&entry.front());
  if (!call || call.getCallee() != "LyEH_ThrowException")
    return false;
  return llvm::equal(call.getOperands(), entry.getArguments());
}

// Where to put what a call needs first: ahead of the try markers that name
// it. ⛔ Not between them and the call: the exception lowering guards the
// call that FOLLOWS a marker, and LYTHON_VERIFY_BARE_POINTER_ABI's check is a
// call.
mlir::Operation *beforeTryMarkers(mlir::Operation *call) {
  mlir::Operation *at = call;
  while (auto marker =
             mlir::dyn_cast_or_null<mlir::func::CallOp>(at->getPrevNode())) {
    llvm::StringRef callee = marker.getCallee();
    if (callee != "LyEH_TryCallSiteMarker" && callee != "LyEH_TryCatchMarker" &&
        callee != "LyEH_TryCatchAnchor")
      break;
    at = marker;
  }
  return at;
}

mlir::UnrealizedConversionCastOp
markNarrow(mlir::OpBuilder &builder, mlir::UnrealizedConversionCastOp cast,
           bool atStart) {
  cast->setAttr(kNarrowCastAttr, builder.getUnitAttr());
  if (atStart)
    cast->setAttr(kNarrowAtStartAttr, builder.getUnitAttr());
  return cast;
}

// The carrier values of `memref`.
mlir::ValueRange toCarrier(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::TypeRange carrier, mlir::Value memref,
                           bool atStart) {
  return markNarrow(builder,
                    mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                             carrier, memref),
                    atStart)
      .getResults();
}

// The memref the carrier values stand for.
mlir::Value fromCarrier(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Type memref, mlir::ValueRange carrier) {
  return markNarrow(builder,
                    mlir::UnrealizedConversionCastOp::create(
                        builder, loc, mlir::TypeRange{memref}, carrier),
                    /*atStart=*/false)
      .getResult(0);
}

// Where a descriptor's allocated pointer can differ from its aligned one, and
// where anything reads it -- both followed through every way a descriptor
// moves in the module: casts and views, `scf.if`/`scf.for`, branches,
// selects, and the calls and returns of the functions that may change.
//
// OFFSET (forward): a view at an offset other than 0 starts one (a lane of a
// block, an image in a table), and so does whatever this does not follow --
// an unknown op, an argument of a function whose callers are not all here.
// An allocation, a global and a view at 0 do not; neither does a descriptor
// rebuilt from a narrow carrier, or one assembled from one pointer.
//
// READ (backward): whatever this does not follow reads it -- a cast to the
// descriptor struct, a return whose callers are not all here, an argument of
// a callee that keeps descriptors, an unknown op. A load, a store, a pointer
// extraction and a narrow cast read the aligned one only; a deallocation has
// become a narrow cast by now.
//
// A position may carry a pointer unless a value with an offset reaches it AND
// its allocated pointer is read there or after.
class OffsetAnalysis {
public:
  using Functions = llvm::DenseMap<mlir::StringAttr, mlir::func::FuncOp>;

  OffsetAnalysis(mlir::ModuleOp module, const Functions &followed,
                 const llvm::DenseSet<mlir::StringAttr> &forced)
      : followed(followed), forced(forced) {
    for (auto &[name, candidate] : followed) {
      mlir::func::FuncOp function = candidate;
      function.walk([&, name = name](mlir::func::ReturnOp ret) {
        returns[name].push_back(ret);
      });
    }
    module.walk([&](mlir::Operation *op) {
      for (mlir::Region &region : op->getRegions())
        for (mlir::Block &block : region)
          for (mlir::BlockArgument argument : block.getArguments())
            if (mlir::isa<mlir::MemRefType>(argument.getType()))
              classifyArgument(argument);
      for (mlir::OpResult result : op->getResults())
        if (mlir::isa<mlir::MemRefType>(result.getType()))
          classifyResult(result);
      for (mlir::OpOperand &operand : op->getOpOperands())
        if (mlir::isa<mlir::MemRefType>(operand.get().getType()))
          classifyUse(operand);
      followCallsAndBranches(op);
    });
    propagate(offset, offsetWork, flowsTo);
    propagate(read, readWork, flowsFrom);
  }

  bool hasOffset(mlir::Value value) const { return offset.contains(value); }
  bool allocatedRead(mlir::Value value) const { return read.contains(value); }

private:
  static void
  propagate(llvm::DenseSet<mlir::Value> &set,
            llvm::SmallVectorImpl<mlir::Value> &work,
            const llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>>
                &edges) {
    while (!work.empty()) {
      mlir::Value value = work.pop_back_val();
      for (mlir::Value next : edges.lookup(value))
        if (set.insert(next).second)
          work.push_back(next);
    }
  }
  void startOffset(mlir::Value value) {
    if (offset.insert(value).second)
      offsetWork.push_back(value);
  }
  void startRead(mlir::Value value) {
    if (read.insert(value).second)
      readWork.push_back(value);
  }
  void flow(mlir::Value from, mlir::Value to) {
    if (!mlir::isa<mlir::MemRefType>(to.getType()))
      return;
    flowsTo[from].push_back(to);
    flowsFrom[to].push_back(from);
  }
  bool isFollowed(mlir::func::CallOp call) const {
    return followed.contains(call.getCalleeAttr().getAttr());
  }
  bool isForced(mlir::func::CallOp call) const {
    return forced.contains(call.getCalleeAttr().getAttr());
  }

  void classifyArgument(mlir::BlockArgument argument) {
    mlir::Block *block = argument.getOwner();
    mlir::Operation *parent = block->getParentOp();
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(parent)) {
      // An entry argument hears from the calls followed below; a later
      // block's, from the branches into it. A forced definition's arrives
      // narrow, so it starts where the rebuilt descriptor does.
      if (block->isEntryBlock() &&
          !followed.contains(function.getSymNameAttr()) &&
          !(forced.contains(function.getSymNameAttr()) &&
            narrows(argument.getType())))
        startOffset(argument);
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
      if (argument.getArgNumber() == 0)
        return;
      unsigned carried = argument.getArgNumber() - 1;
      flow(loop.getInitArgs()[carried], argument);
      flow(loop.getBody()->getTerminator()->getOperand(carried), argument);
      return;
    }
    if (block->isEntryBlock())
      startOffset(argument);
  }

  void classifyResult(mlir::OpResult result) {
    mlir::Operation *op = result.getOwner();
    unsigned index = result.getResultNumber();
    if (mlir::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp,
                  mlir::memref::GetGlobalOp>(op))
      return;
    if (auto view = mlir::dyn_cast<mlir::memref::ViewOp>(op)) {
      flow(view.getSource(), result);
      if (!mlir::matchPattern(view.getByteShift(), mlir::m_Zero()))
        startOffset(result);
      return;
    }
    if (mlir::isa<mlir::memref::CastOp, mlir::memref::ReinterpretCastOp,
                  mlir::memref::SubViewOp, mlir::memref::AssumeAlignmentOp,
                  mlir::memref::CollapseShapeOp, mlir::memref::ExpandShapeOp,
                  mlir::memref::ExtractStridedMetadataOp>(op)) {
      flow(op->getOperand(0), result);
      return;
    }
    // A descriptor assembled in place (MemRef1D.h) frees what every access
    // goes through when the two pointers it was given are one.
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      if (cast->hasAttr(kNarrowCastAttr))
        return;
      mlir::Value allocated = cast->getNumOperands() == 1
                                  ? insertedAt(cast->getOperand(0), 0)
                                  : mlir::Value();
      if (!allocated || allocated != insertedAt(cast->getOperand(0), 1))
        startOffset(result);
      return;
    }
    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
      flow(select.getTrueValue(), result);
      flow(select.getFalseValue(), result);
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      for (mlir::Region *region :
           {&branch.getThenRegion(), &branch.getElseRegion()})
        if (!region->empty())
          flow(region->front().getTerminator()->getOperand(index), result);
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      flow(loop.getInitArgs()[index], result);
      flow(loop.getBody()->getTerminator()->getOperand(index), result);
      return;
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      if (isFollowed(call)) {
        for (mlir::func::ReturnOp ret :
             returns.lookup(call.getCalleeAttr().getAttr()))
          flow(ret->getOperand(index), result);
        return;
      }
      if (isForced(call) && narrows(result.getType()))
        return;
    }
    startOffset(result);
  }

  void classifyUse(mlir::OpOperand &use) {
    mlir::Operation *user = use.getOwner();
    mlir::Value value = use.get();
    // Followed as flows: the edge carries a read back.
    if (mlir::isa<mlir::memref::ViewOp, mlir::memref::CastOp,
                  mlir::memref::ReinterpretCastOp, mlir::memref::SubViewOp,
                  mlir::memref::AssumeAlignmentOp,
                  mlir::memref::CollapseShapeOp, mlir::memref::ExpandShapeOp,
                  mlir::memref::ExtractStridedMetadataOp, mlir::arith::SelectOp,
                  mlir::cf::BranchOp, mlir::cf::CondBranchOp>(user))
      return;
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(user))
      if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(yield->getParentOp()))
        return;
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(user))
      if (use.getOperandNumber() >= loop.getNumControlOperands())
        return;
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(user))
      if (isFollowed(call) || (isForced(call) && narrows(value.getType())))
        return;
    if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(user))
      if (auto function = ret->getParentOfType<mlir::func::FuncOp>())
        if (followed.contains(function.getSymNameAttr()) ||
            (forced.contains(function.getSymNameAttr()) &&
             narrows(value.getType())))
          return;
    // What reads only the aligned pointer.
    if (user->hasAttr(kNarrowCastAttr))
      return;
    if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                  mlir::memref::DimOp, mlir::memref::RankOp,
                  mlir::memref::ExtractAlignedPointerAsIndexOp,
                  mlir::memref::CopyOp, mlir::memref::AtomicRMWOp,
                  mlir::memref::GenericAtomicRMWOp, mlir::memref::PrefetchOp>(
            user))
      return;
    startRead(value);
  }

  void followCallsAndBranches(mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      auto callee = followed.find(call.getCalleeAttr().getAttr());
      if (callee != followed.end()) {
        mlir::func::FuncOp function = callee->second;
        for (auto [operand, argument] : llvm::zip(
                 call.getOperands(), function.getBody().front().getArguments()))
          flow(operand, argument);
      }
    }
    if (auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(op))
      for (unsigned index = 0; index < op->getNumSuccessors(); ++index) {
        mlir::Block *successor = op->getSuccessor(index);
        mlir::SuccessorOperands operands = branch.getSuccessorOperands(index);
        unsigned produced = operands.getProducedOperandCount();
        for (auto [position, operand] :
             llvm::enumerate(operands.getForwardedOperands()))
          flow(operand, successor->getArgument(produced + position));
      }
  }

  // The value an `llvm.insertvalue` chain put at top-level `position` of a
  // struct, if the chain says.
  static mlir::Value insertedAt(mlir::Value aggregate, std::int64_t position) {
    while (auto insert = aggregate.getDefiningOp<mlir::LLVM::InsertValueOp>()) {
      if (insert.getPosition() == llvm::ArrayRef<std::int64_t>{position})
        return insert.getValue();
      aggregate = insert.getContainer();
    }
    return {};
  }

  const Functions &followed;
  const llvm::DenseSet<mlir::StringAttr> &forced;
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::func::ReturnOp>>
      returns;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> flowsTo;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> flowsFrom;
  llvm::DenseSet<mlir::Value> offset;
  llvm::DenseSet<mlir::Value> read;
  llvm::SmallVector<mlir::Value, 64> offsetWork;
  llvm::SmallVector<mlir::Value, 64> readWork;
};

unsigned indexBitsOf(mlir::ModuleOp module) {
  return mlir::LowerToLLVMOptions(module.getContext(), mlir::DataLayout(module))
      .getIndexBitwidth();
}

} // namespace

unsigned narrowMemRefCalls(mlir::ModuleOp module, NarrowingScope scope) {
  // Only the program's own calls: what crosses to the native runtime is its
  // ABI, built into the precompiled runtime and the pre-lowered modules.
  bool ownCalls = !envSet("LYTHON_ABLATE_BARE_POINTER_ABI");
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  unsigned indexBits = indexBitsOf(module);
  auto ptr = mlir::LLVM::LLVMPointerType::get(context);
  auto carried = [&](mlir::Type type) {
    return carrierTypes(context, carrierOf(type), indexBits);
  };

  llvm::StringSet<> slots;
  if (mlir::Attribute table = module->getAttr("ly.type_slots"))
    collectSlotNames(table, slots);
  for (const type_objects::SlotHook &hook : type_objects::kSlotHooks)
    slots.insert(hook.name);
  module.walk([&](mlir::func::FuncOp function) {
    if (mlir::Attribute hooked =
            function->getAttr(type_objects::kHookSlotsAttr))
      collectSlotNames(hooked, slots);
  });

  // A symbol named anywhere but as a direct callee can be reached some other
  // way, with whatever type that way assumes.
  llvm::StringSet<> otherwiseNamed;
  std::optional<mlir::SymbolTable::UseRange> uses =
      mlir::SymbolTable::getSymbolUses(module.getOperation());
  if (!uses)
    return 0;
  for (const mlir::SymbolTable::SymbolUse &use : *uses) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getUser());
    if (call && call.getCalleeAttr() == use.getSymbolRef())
      continue;
    otherwiseNamed.insert(use.getSymbolRef().getRootReference().getValue());
  }

  OffsetAnalysis::Functions followed;
  llvm::DenseSet<mlir::StringAttr> forced;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    llvm::StringRef name = function.getSymName();
    if (slots.contains(name) || name.starts_with("__ly_slot.") ||
        otherwiseNamed.contains(name))
      continue;
    mlir::FunctionType type = function.getFunctionType();
    bool narrowable = llvm::any_of(type.getInputs(), narrows) ||
                      llvm::any_of(type.getResults(), narrows);
    if (function.isDeclaration()) {
      if (scope.declarations && narrowable)
        forced.insert(function.getSymNameAttr());
      continue;
    }
    if (!function.isPrivate()) {
      if (scope.publicDefinitions && narrowable)
        forced.insert(function.getSymNameAttr());
      continue;
    }
    if (!ownCalls || isThinRaiseWrapper(function))
      continue;
    // Every one is followed; only those with a narrowable memref change.
    followed[function.getSymNameAttr()] = function;
  }

  // Each deallocation frees the aligned pointer, as `memref.dealloc` lowers
  // to (`free`, which the finalization sends to the object allocator).
  // ⛔ Not a cast there and back around the dealloc: a canonicalizer folds
  // the pair away.
  llvm::SmallVector<mlir::memref::DeallocOp> deallocs;
  module.walk([&](mlir::memref::DeallocOp dealloc) {
    if (narrows(dealloc.getMemref().getType()))
      deallocs.push_back(dealloc);
  });
  if (!deallocs.empty()) {
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> free =
        mlir::LLVM::lookupOrCreateFreeFn(builder, module);
    if (mlir::failed(free))
      return 0;
    for (mlir::memref::DeallocOp dealloc : deallocs) {
      builder.setInsertionPoint(dealloc);
      mlir::Value pointer =
          toCarrier(builder, dealloc.getLoc(), mlir::TypeRange{ptr},
                    dealloc.getMemref(), /*atStart=*/true)[0];
      mlir::LLVM::CallOp::create(builder, dealloc.getLoc(), *free,
                                 mlir::ValueRange{pointer});
      dealloc.erase();
    }
  }

  OffsetAnalysis offsets(module, followed, forced);
  struct Narrowed {
    mlir::FunctionType wide;
    llvm::SmallVector<bool, 8> inputs;
    llvm::SmallVector<bool, 4> results;
  };
  llvm::DenseMap<mlir::StringAttr, Narrowed> narrowed;
  bool trace = envSet("LYTHON_TRACE_BARE_POINTER_ABI");

  auto decide = [&](mlir::func::FuncOp function) -> std::optional<Narrowed> {
    mlir::FunctionType type = function.getFunctionType();
    Narrowed decided{type, {}, {}};
    if (forced.contains(function.getSymNameAttr())) {
      for (mlir::Type input : type.getInputs())
        decided.inputs.push_back(narrows(input));
      for (mlir::Type result : type.getResults())
        decided.results.push_back(narrows(result));
      return decided;
    }
    if (llvm::none_of(type.getInputs(), narrows) &&
        llvm::none_of(type.getResults(), narrows))
      return std::nullopt;
    mlir::Block &entry = function.getBody().front();
    for (auto [index, input] : llvm::enumerate(type.getInputs())) {
      mlir::BlockArgument argument = entry.getArgument(index);
      decided.inputs.push_back(
          narrows(input) &&
          !(offsets.hasOffset(argument) && offsets.allocatedRead(argument)));
    }
    llvm::SmallVector<mlir::func::ReturnOp, 4> returns;
    function.walk([&](mlir::func::ReturnOp ret) { returns.push_back(ret); });
    for (auto [index, result] : llvm::enumerate(type.getResults()))
      decided.results.push_back(
          narrows(result) && llvm::none_of(returns, [&](auto ret) {
            mlir::Value value = ret->getOperand(index);
            return offsets.hasOffset(value) && offsets.allocatedRead(value);
          }));
    if (trace) {
      for (auto [index, input] : llvm::enumerate(type.getInputs()))
        if (narrows(input) && !decided.inputs[index])
          llvm::errs() << "[bare-wide] argument " << index << " of "
                       << function.getSymName() << "\n";
      for (auto [index, result] : llvm::enumerate(type.getResults()))
        if (narrows(result) && !decided.results[index])
          llvm::errs() << "[bare-wide] result " << index << " of "
                       << function.getSymName() << "\n";
    }
    if (llvm::none_of(decided.inputs, [](bool narrow) { return narrow; }) &&
        llvm::none_of(decided.results, [](bool narrow) { return narrow; }))
      return std::nullopt;
    return decided;
  };

  llvm::SmallVector<mlir::func::FuncOp, 64> changing;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    if (followed.contains(function.getSymNameAttr()) ||
        forced.contains(function.getSymNameAttr()))
      changing.push_back(function);
  for (mlir::func::FuncOp function : changing) {
    std::optional<Narrowed> decided = decide(function);
    if (!decided)
      continue;
    mlir::FunctionType type = function.getFunctionType();
    llvm::SmallVector<mlir::Type, 8> inputs;
    llvm::SmallVector<mlir::DictionaryAttr, 8> inputAttrs;
    for (auto [index, input] : llvm::enumerate(type.getInputs())) {
      if (!decided->inputs[index]) {
        inputs.push_back(input);
        inputAttrs.push_back(function.getArgAttrDict(index));
        continue;
      }
      for (mlir::Type carrier : carried(input)) {
        inputs.push_back(carrier);
        inputAttrs.push_back(mlir::DictionaryAttr::get(context));
      }
    }
    llvm::SmallVector<mlir::Type, 4> results;
    llvm::SmallVector<mlir::DictionaryAttr, 4> resultAttrs;
    for (auto [index, result] : llvm::enumerate(type.getResults())) {
      if (!decided->results[index]) {
        results.push_back(result);
        resultAttrs.push_back(function.getResultAttrDict(index));
        continue;
      }
      for (mlir::Type carrier : carried(result)) {
        results.push_back(carrier);
        resultAttrs.push_back(mlir::DictionaryAttr::get(context));
      }
    }
    if (!function.isDeclaration()) {
      mlir::Block &entry = function.getBody().front();
      mlir::Location loc = function.getLoc();
      // Last first, so the indices before each are still the wide ones.
      for (unsigned index = type.getNumInputs(); index-- > 0;) {
        if (!decided->inputs[index])
          continue;
        mlir::Type wide = type.getInput(index);
        llvm::SmallVector<mlir::Type, 2> carrier = carried(wide);
        mlir::BlockArgument old = entry.getArgument(index);
        llvm::SmallVector<mlir::Value, 2> values;
        for (auto [offset, carrierType] : llvm::enumerate(carrier))
          values.push_back(
              entry.insertArgument(index + 1 + offset, carrierType, loc));
        builder.setInsertionPointToStart(&entry);
        old.replaceAllUsesWith(fromCarrier(builder, loc, wide, values));
        entry.eraseArgument(index);
      }
      function.walk([&](mlir::func::ReturnOp ret) {
        builder.setInsertionPoint(ret);
        llvm::SmallVector<mlir::Value, 4> operands;
        for (mlir::OpOperand &operand : ret->getOpOperands()) {
          unsigned index = operand.getOperandNumber();
          if (!decided->results[index]) {
            operands.push_back(operand.get());
            continue;
          }
          llvm::append_range(
              operands,
              toCarrier(builder, ret.getLoc(), carried(type.getResult(index)),
                        operand.get(), !offsets.hasOffset(operand.get())));
        }
        ret->setOperands(operands);
      });
    }
    function.setFunctionType(mlir::FunctionType::get(context, inputs, results));
    function.setAllArgAttrs(inputAttrs);
    function.setAllResultAttrs(resultAttrs);
    narrowed[function.getSymNameAttr()] = std::move(*decided);
  }

  llvm::SmallVector<mlir::func::CallOp, 256> calls;
  module.walk([&](mlir::func::CallOp call) {
    if (narrowed.contains(call.getCalleeAttr().getAttr()))
      calls.push_back(call);
  });
  // Asked of the operands as the analysis saw them: rewriting one call
  // replaces its results, which later calls use.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<bool, 8>> atStart;
  for (mlir::func::CallOp call : calls)
    for (mlir::Value operand : call->getOperands())
      atStart[call].push_back(!offsets.hasOffset(operand));
  for (mlir::func::CallOp call : calls) {
    const Narrowed &decided =
        narrowed.find(call.getCalleeAttr().getAttr())->second;
    mlir::Location loc = call.getLoc();
    builder.setInsertionPoint(beforeTryMarkers(call));
    llvm::SmallVector<mlir::Value, 8> operands;
    for (auto [index, input] : llvm::enumerate(decided.wide.getInputs())) {
      mlir::Value operand = call->getOperand(index);
      if (!decided.inputs[index]) {
        operands.push_back(operand);
        continue;
      }
      llvm::append_range(operands, toCarrier(builder, loc, carried(input),
                                             operand, atStart[call][index]));
    }
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    for (auto [index, result] : llvm::enumerate(decided.wide.getResults())) {
      if (decided.results[index])
        llvm::append_range(resultTypes, carried(result));
      else
        resultTypes.push_back(result);
    }
    builder.setInsertionPoint(call);
    auto narrowCall = mlir::func::CallOp::create(
        builder, loc, call.getCalleeAttr(), resultTypes, operands);
    narrowCall->setAttrs(call->getAttrs());
    builder.setInsertionPointAfter(narrowCall);
    unsigned at = 0;
    for (auto [index, result] : llvm::enumerate(decided.wide.getResults())) {
      if (!decided.results[index]) {
        call->getResult(index).replaceAllUsesWith(narrowCall->getResult(at++));
        continue;
      }
      unsigned width = carried(result).size();
      call->getResult(index).replaceAllUsesWith(fromCarrier(
          builder, loc, result, narrowCall->getResults().slice(at, width)));
      at += width;
    }
    call.erase();
  }
  return narrowed.size();
}

unsigned narrowExpandedDescriptors(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  unsigned indexBits = indexBitsOf(module);
  auto index = mlir::IntegerType::get(context, indexBits);
  mlir::OpBuilder builder(context);
  unsigned changed = 0;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    auto spelled = function->getAttrOfType<mlir::DenseI64ArrayAttr>(
        kExpandedDescriptorsAttr);
    if (!spelled)
      continue;
    function->removeAttr(kExpandedDescriptorsAttr);
    if (function.isDeclaration())
      continue;
    llvm::ArrayRef<std::int64_t> pairs = spelled.asArrayRef();
    mlir::Block &entry = function.getBody().front();
    mlir::Location loc = function.getLoc();
    auto constant = [&](std::int64_t value) {
      return mlir::LLVM::ConstantOp::create(
                 builder, loc, index, builder.getIntegerAttr(index, value))
          .getResult();
    };
    // Last first, so the positions before each are still the spelled ones.
    // The five words [allocated, aligned, offset, size, stride] become the
    // aligned pointer, then the size when the shape does not fix it.
    for (std::size_t at = pairs.size(); at >= 2; at -= 2) {
      unsigned first = static_cast<unsigned>(pairs[at - 2]);
      std::int64_t staticSize = pairs[at - 1];
      builder.setInsertionPointToStart(&entry);
      mlir::Value aligned = entry.getArgument(first + 1);
      mlir::Value size = staticSize < 0
                             ? mlir::Value(entry.getArgument(first + 3))
                             : constant(staticSize);
      entry.getArgument(first).replaceAllUsesWith(aligned);
      entry.getArgument(first + 2).replaceAllUsesWith(constant(0));
      if (staticSize >= 0)
        entry.getArgument(first + 3).replaceAllUsesWith(size);
      entry.getArgument(first + 4).replaceAllUsesWith(constant(1));
      llvm::BitVector drop(entry.getNumArguments());
      drop.set(first);
      drop.set(first + 2);
      if (staticSize >= 0)
        drop.set(first + 3);
      drop.set(first + 4);
      if (mlir::failed(function.eraseArguments(drop)))
        return changed;
    }
    ++changed;
  }
  return changed;
}

namespace {

// The casts on each side of a call that carries a memref narrow: the aligned
// pointer (and size) one way, a descriptor rebuilt from them the other.
class NarrowCastLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::UnrealizedConversionCastOp> {
public:
  NarrowCastLowering(const mlir::LLVMTypeConverter &converter, bool verify)
      : ConvertOpToLLVMPattern(converter), verify(verify) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr(kNarrowCastAttr))
      return mlir::failure();
    mlir::Location loc = op.getLoc();
    mlir::ValueRange inputs = adaptor.getInputs();
    if (auto memref =
            mlir::dyn_cast<mlir::MemRefType>(op.getResult(0).getType())) {
      if (memref.hasStaticShape()) {
        rewriter.replaceOp(
            op, mlir::Value(mlir::MemRefDescriptor::fromStaticShape(
                    rewriter, loc, *getTypeConverter(), memref, inputs[0])));
        return mlir::success();
      }
      auto descriptor = mlir::MemRefDescriptor::poison(
          rewriter, loc, getTypeConverter()->convertType(memref));
      descriptor.setAllocatedPtr(rewriter, loc, inputs[0]);
      descriptor.setAlignedPtr(rewriter, loc, inputs[0]);
      descriptor.setConstantOffset(rewriter, loc, 0);
      descriptor.setSize(rewriter, loc, 0, inputs[1]);
      descriptor.setConstantStride(rewriter, loc, 0, 1);
      rewriter.replaceOp(op, mlir::Value(descriptor));
      return mlir::success();
    }
    mlir::MemRefDescriptor descriptor(inputs[0]);
    mlir::Value aligned = descriptor.alignedPtr(rewriter, loc);
    if (verify && op->hasAttr(kNarrowAtStartAttr))
      check(op, descriptor, aligned, rewriter);
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, aligned);
      return mlir::success();
    }
    rewriter.replaceOp(
        op, mlir::ValueRange{aligned, descriptor.size(rewriter, loc, 0)});
    return mlir::success();
  }

private:
  void check(mlir::Operation *op, mlir::MemRefDescriptor &descriptor,
             mlir::Value aligned,
             mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Location loc = op->getLoc();
    std::string site = "a memref passed narrow is not at the start of its "
                       "allocation: ";
    // The call may have been converted to `llvm.call` already, and the
    // function to `llvm.func`.
    mlir::Operation *user = op->getResult(0).hasOneUse()
                                ? *op->getResult(0).getUsers().begin()
                                : nullptr;
    auto call = mlir::dyn_cast_or_null<mlir::CallOpInterface>(user);
    auto callee = call ? mlir::dyn_cast_or_null<mlir::SymbolRefAttr>(
                             call.getCallableForCallee())
                       : mlir::SymbolRefAttr();
    if (callee)
      site += ("argument of " + callee.getRootReference().getValue()).str();
    else if (auto function = op->getParentOfType<mlir::FunctionOpInterface>())
      site += ("result or deallocation in " +
               mlir::SymbolTable::getSymbolName(function).getValue())
                  .str();
    mlir::Value label = mlir::LLVM::createGlobalString(
        loc, rewriter,
        ("__ly_abi_label." + llvm::Twine(labels.fetch_add(1))).str(), site,
        mlir::LLVM::Linkage::Internal);
    mlir::LLVM::CallOp::create(
        rewriter, loc, mlir::TypeRange{}, kNarrowCheck,
        mlir::ValueRange{descriptor.allocatedPtr(rewriter, loc), aligned,
                         label});
  }

  bool verify;
  mutable std::atomic<unsigned> labels{0};
};

class LowerNarrowCasts
    : public mlir::PassWrapper<LowerNarrowCasts,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerNarrowCasts)

  llvm::StringRef getArgument() const final {
    return "lython-lower-narrow-casts";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    bool verify = verifyingNarrowCalls();
    if (verify)
      defineNarrowCheck(module);
    mlir::LowerToLLVMOptions options(&getContext(), mlir::DataLayout(module));
    mlir::LLVMTypeConverter converter(&getContext(), options);
    mlir::ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    mlir::RewritePatternSet patterns(&getContext());
    populateNarrowCastLowering(converter, patterns, target, verify);
    if (mlir::failed(
            mlir::applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

bool verifyingNarrowCalls() { return envSet("LYTHON_VERIFY_BARE_POINTER_ABI"); }

void populateNarrowCastLowering(mlir::LLVMTypeConverter &converter,
                                mlir::RewritePatternSet &patterns,
                                mlir::ConversionTarget &target, bool verify) {
  patterns.add<NarrowCastLowering>(converter, verify);
  target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
      [](mlir::UnrealizedConversionCastOp op) {
        return !op->hasAttr(kNarrowCastAttr);
      });
}

void defineNarrowCheck(mlir::ModuleOp module) {
  if (module.lookupSymbol(kNarrowCheck))
    return;
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  mlir::Location loc = module.getLoc();
  auto ptr = mlir::LLVM::LLVMPointerType::get(context);
  auto voidType = mlir::LLVM::LLVMVoidType::get(context);
  builder.setInsertionPointToEnd(module.getBody());
  if (!module.lookupSymbol("abort"))
    mlir::LLVM::LLVMFuncOp::create(
        builder, loc, "abort", mlir::LLVM::LLVMFunctionType::get(voidType, {}));
  // `cf.assert`'s lowering may have declared it already, returning nothing.
  auto puts = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("puts");
  if (!puts)
    puts = mlir::LLVM::LLVMFuncOp::create(
        builder, loc, "puts",
        mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(), {ptr}));
  auto check = mlir::LLVM::LLVMFuncOp::create(
      builder, loc, kNarrowCheck,
      mlir::LLVM::LLVMFunctionType::get(voidType, {ptr, ptr, ptr}),
      mlir::LLVM::Linkage::Internal);
  mlir::Block *entry = check.addEntryBlock(builder);
  mlir::Block *held = builder.createBlock(&check.getBody());
  mlir::Block *broken = builder.createBlock(&check.getBody());
  builder.setInsertionPointToStart(entry);
  mlir::Value allocated = entry->getArgument(0);
  mlir::Value same =
      mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                 allocated, entry->getArgument(1));
  // The placeholder a global's descriptor carries (memref.get_global).
  mlir::Value placeholder = mlir::LLVM::IntToPtrOp::create(
      builder, loc, ptr,
      mlir::LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                     builder.getI64IntegerAttr(0xdeadbeef))
          .getResult());
  mlir::Value global = mlir::LLVM::ICmpOp::create(
      builder, loc, mlir::LLVM::ICmpPredicate::eq, allocated, placeholder);
  mlir::Value fine = mlir::LLVM::OrOp::create(builder, loc, same, global);
  mlir::LLVM::CondBrOp::create(builder, loc, fine, held, broken);
  builder.setInsertionPointToStart(held);
  mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
  builder.setInsertionPointToStart(broken);
  mlir::LLVM::CallOp::create(builder, loc, puts,
                             mlir::ValueRange{entry->getArgument(2)});
  mlir::LLVM::CallOp::create(builder, loc, mlir::TypeRange{}, "abort",
                             mlir::ValueRange{});
  mlir::LLVM::UnreachableOp::create(builder, loc);
}

std::unique_ptr<mlir::Pass> createLowerNarrowCastsPass() {
  return std::make_unique<LowerNarrowCasts>();
}

} // namespace py::lowering::runtime::cleanup
