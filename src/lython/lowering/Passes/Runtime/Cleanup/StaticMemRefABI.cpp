// The calling convention of a static-shape memref: one pointer.
//
// MLIR's func-to-LLVM conversion passes every memref as its descriptor,
// five words (allocated and aligned pointer, offset, size, stride) for a
// rank-1 one. For a handle whose shape is static every word but the two
// pointers is a constant, and the program spent most of its instructions
// unpacking and repacking them: an object crosses a call as several handles.
//
// Here every private function with a body that is only called directly takes
// and returns such a memref as `!llvm.ptr`, and each side of the boundary
// converts with a cast marked `kBarePointerCastAttr`. The LLVM conversion
// lowers the cast to the aligned pointer one way and to a descriptor rebuilt
// from it the other (LoweringPipeline.cpp), whose allocated pointer is the
// aligned one.
//
// ⭐ THE ALLOCATED POINTER IS KEPT ONLY WHERE IT IS BOTH DIFFERENT AND READ.
// A lane of a block or an image in a table is a view whose allocated pointer
// is the block's or the table's; `OffsetAnalysis` keeps the descriptor at a
// position such a value reaches and whose allocated pointer something reads.
// A deallocation is not such a reader: it frees through the aligned pointer
// (below), since what it frees is at the start of its allocation.
//
// ⛔ Not upstream's bare-pointer convention: it is one switch for a whole
// conversion, and it refuses every function with a dynamic-shape memref.
// ⛔ Not a declaration: its body is lowered elsewhere (the native runtime)
// with the descriptor convention.
// ⛔ Not a slot function: the type object's hook calls it through a pointer
// with the hook's own lowered type, which nothing here rewrites.
// ⛔ Not a thin raise wrapper: the exception lowering redirects a call to one
// to `LyEH_RecordException` only when the two types are the same.

#include "Passes/Runtime/Cleanup/Transforms.h"

#include "Common/TypeObjects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <cstdlib>
#include <string>

namespace py::lowering::runtime::cleanup {
namespace {

bool passesBare(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.hasStaticShape() && memref.getLayout().isIdentity() &&
         !memref.getMemorySpace();
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

mlir::Value castTo(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Type type, mlir::Value value, bool atStart = false) {
  auto cast =
      mlir::UnrealizedConversionCastOp::create(builder, loc, type, value);
  cast->setAttr(kBarePointerCastAttr, builder.getUnitAttr());
  if (atStart)
    cast->setAttr(kBarePointerAtStartAttr, builder.getUnitAttr());
  return cast.getResult(0);
}

// Where a descriptor's allocated pointer can differ from its aligned one, and
// where anything reads it -- both followed through every way a descriptor
// moves in the module: casts and views, `scf.if`/`scf.for`, branches,
// selects, and the calls and returns of the functions that may change.
//
// OFFSET (forward): a view at an offset other than 0 starts one (a lane of a
// block, an image in a table), and so does whatever this does not follow --
// an unknown op, a declaration's result, an argument of a function whose
// callers are not all here. An allocation, a global and a view at 0 do not;
// neither does `__ly_global_view_*`, built with one pointer for both
// (RuntimeSupportBuilder.cpp).
//
// READ (backward): a deallocation reads it, and so does whatever this does
// not follow -- a declaration's argument (the native runtime stores the
// exception's descriptors whole), a cast to the descriptor struct, a return
// whose callers are not all here, an unknown op. A load, a store and a
// pointer extraction read the aligned one only.
//
// A position may carry a pointer unless a value with an offset reaches it AND
// its allocated pointer is read there or after.
class OffsetAnalysis {
public:
  using Candidates = llvm::DenseMap<mlir::StringAttr, mlir::func::FuncOp>;

  OffsetAnalysis(mlir::ModuleOp module, const Candidates &candidates)
      : candidates(candidates) {
    for (auto &[name, candidate] : candidates) {
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
  // What a read reaching `value` started at, walking the flows forward.
  std::string readOrigin(mlir::Value value) const {
    llvm::DenseSet<mlir::Value> seen;
    llvm::SmallVector<mlir::Value> work{value};
    while (!work.empty()) {
      mlir::Value at = work.pop_back_val();
      if (!seen.insert(at).second)
        continue;
      auto seed = readSeed.find(at);
      if (seed != readSeed.end())
        return seed->second;
      for (mlir::Value next : flowsTo.lookup(at))
        if (read.contains(next))
          work.push_back(next);
    }
    return "?";
  }

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
  bool isCandidate(mlir::func::CallOp call) const {
    return candidates.contains(call.getCalleeAttr().getAttr());
  }

  void classifyArgument(mlir::BlockArgument argument) {
    mlir::Block *block = argument.getOwner();
    mlir::Operation *parent = block->getParentOp();
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(parent)) {
      // An entry argument hears from the calls followed below; a later
      // block's, from the branches into it.
      if (block->isEntryBlock() &&
          !candidates.contains(function.getSymNameAttr()))
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
    // goes through when the two pointers it was given are one; so does one
    // rebuilt from a pointer.
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      if (cast->hasAttr(kBarePointerCastAttr))
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
      if (isCandidate(call)) {
        for (mlir::func::ReturnOp ret :
             returns.lookup(call.getCalleeAttr().getAttr()))
          flow(ret->getOperand(index), result);
        return;
      }
      if (call.getCallee().starts_with("__ly_global_view_"))
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
      if (isCandidate(call))
        return;
    if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(user))
      if (auto function = ret->getParentOfType<mlir::func::FuncOp>())
        if (candidates.contains(function.getSymNameAttr()))
          return;
    // What reads only the aligned pointer.
    if (user->hasAttr(kBarePointerCastAttr))
      return;
    if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                  mlir::memref::DimOp, mlir::memref::RankOp,
                  mlir::memref::ExtractAlignedPointerAsIndexOp,
                  mlir::memref::CopyOp, mlir::memref::AtomicRMWOp,
                  mlir::memref::GenericAtomicRMWOp, mlir::memref::PrefetchOp>(
            user))
      return;
    if (std::getenv("LYTHON_TRACE_BARE_POINTER_ABI")) {
      std::string what = user->getName().getStringRef().str();
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(user))
        what += " " + call.getCallee().str();
      readSeed[value] = what;
    }
    startRead(value);
  }

  void followCallsAndBranches(mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      auto callee = candidates.find(call.getCalleeAttr().getAttr());
      if (callee != candidates.end()) {
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

  const Candidates &candidates;
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::func::ReturnOp>>
      returns;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> flowsTo;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> flowsFrom;
  llvm::DenseSet<mlir::Value> offset;
  llvm::DenseSet<mlir::Value> read;
  llvm::DenseMap<mlir::Value, std::string> readSeed;
  llvm::SmallVector<mlir::Value, 64> offsetWork;
  llvm::SmallVector<mlir::Value, 64> readWork;
};

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

} // namespace

unsigned passStaticMemRefsBare(mlir::ModuleOp module) {
  if (const char *ablate = std::getenv("LYTHON_ABLATE_BARE_POINTER_ABI");
      ablate && *ablate && llvm::StringRef(ablate) != "0")
    return 0;

  llvm::StringSet<> slots;
  if (mlir::Attribute table = module->getAttr("ly.type_slots"))
    collectSlotNames(table, slots);
  module.walk([&](mlir::func::FuncOp function) {
    if (mlir::Attribute hooked =
            function->getAttr(type_objects::kHookSlotsAttr))
      collectSlotNames(hooked, slots);
  });

  // A symbol named anywhere but as a direct callee can be reached some other
  // way, with whatever type that way assumes.
  llvm::StringSet<> otherwiseNamed;
  if (std::optional<mlir::SymbolTable::UseRange> uses =
          mlir::SymbolTable::getSymbolUses(module.getOperation())) {
    for (const mlir::SymbolTable::SymbolUse &use : *uses) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getUser());
      if (call && call.getCalleeAttr() == use.getSymbolRef())
        continue;
      otherwiseNamed.insert(use.getSymbolRef().getRootReference().getValue());
    }
  } else {
    return 0;
  }

  llvm::DenseMap<mlir::StringAttr, mlir::func::FuncOp> candidates;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (!function.isPrivate() || function.isDeclaration())
      continue;
    llvm::StringRef name = function.getSymName();
    if (slots.contains(name) || name.starts_with("__ly_slot.") ||
        otherwiseNamed.contains(name) || isThinRaiseWrapper(function))
      continue;
    // Every one is followed; only those with a static-shape memref change.
    candidates[function.getSymNameAttr()] = function;
  }

  mlir::MLIRContext *context = module.getContext();
  auto ptr = mlir::LLVM::LLVMPointerType::get(context);
  mlir::OpBuilder builder(context);

  // ⭐ A DEALLOCATION FREES THE ALIGNED POINTER. What is freed is an owned
  // reference, and an owned reference points at the start of its allocation
  // (the proof's site-address-recovers), so the two pointers are one there;
  // freeing through the one every access uses is what lets a pointer stand
  // for the descriptor. LYTHON_VERIFY_BARE_POINTER_ABI=1 checks that they are
  // one at every deallocation.
  module.walk([&](mlir::memref::DeallocOp dealloc) {
    mlir::Value freed = dealloc.getMemref();
    if (!passesBare(freed.getType()))
      return;
    builder.setInsertionPoint(dealloc);
    mlir::Value pointer =
        castTo(builder, dealloc.getLoc(), ptr, freed, /*atStart=*/true);
    dealloc.getMemrefMutable().assign(
        castTo(builder, dealloc.getLoc(), freed.getType(), pointer));
  });

  OffsetAnalysis offsets(module, candidates);
  struct Narrowed {
    mlir::FunctionType wide;
    llvm::SmallVector<bool, 8> inputs;
    llvm::SmallVector<bool, 4> results;
  };
  llvm::DenseMap<mlir::StringAttr, Narrowed> narrowed;

  for (auto &[name, function] : candidates) {
    mlir::FunctionType type = function.getFunctionType();
    if (llvm::none_of(type.getInputs(), passesBare) &&
        llvm::none_of(type.getResults(), passesBare))
      continue;
    mlir::Block &entry = function.getBody().front();
    Narrowed decided{type, {}, {}};
    for (auto [index, input] : llvm::enumerate(type.getInputs())) {
      mlir::BlockArgument argument = entry.getArgument(index);
      decided.inputs.push_back(
          passesBare(input) &&
          !(offsets.hasOffset(argument) && offsets.allocatedRead(argument)));
    }
    llvm::SmallVector<mlir::func::ReturnOp, 4> returns;
    function.walk([&](mlir::func::ReturnOp ret) { returns.push_back(ret); });
    for (auto [index, result] : llvm::enumerate(type.getResults()))
      decided.results.push_back(
          passesBare(result) && llvm::none_of(returns, [&](auto ret) {
            mlir::Value value = ret->getOperand(index);
            return offsets.hasOffset(value) && offsets.allocatedRead(value);
          }));
    if (std::getenv("LYTHON_TRACE_BARE_POINTER_ABI")) {
      for (auto [index, input] : llvm::enumerate(type.getInputs()))
        if (passesBare(input) && !decided.inputs[index])
          llvm::errs() << "[bare-wide] argument " << index << " of "
                       << name.getValue() << " read by "
                       << offsets.readOrigin(entry.getArgument(index)) << "\n";
      for (auto [index, result] : llvm::enumerate(type.getResults()))
        if (passesBare(result) && !decided.results[index])
          llvm::errs() << "[bare-wide] result " << index << " of "
                       << name.getValue() << "\n";
    }
    if (llvm::none_of(decided.inputs, [](bool bare) { return bare; }) &&
        llvm::none_of(decided.results, [](bool bare) { return bare; }))
      continue;

    builder.setInsertionPointToStart(&entry);
    llvm::SmallVector<mlir::Type, 8> inputs(type.getInputs());
    for (auto [index, input] : llvm::enumerate(type.getInputs())) {
      if (!decided.inputs[index])
        continue;
      mlir::BlockArgument argument = entry.getArgument(index);
      argument.setType(ptr);
      mlir::Value handle = castTo(builder, argument.getLoc(), input, argument);
      argument.replaceAllUsesExcept(handle, handle.getDefiningOp());
      inputs[index] = ptr;
      function.setArgAttrs(index, mlir::DictionaryAttr::get(context));
    }
    llvm::SmallVector<mlir::Type, 4> results(type.getResults());
    for (auto [index, result] : llvm::enumerate(type.getResults()))
      if (decided.results[index]) {
        results[index] = ptr;
        function.setResultAttrs(index, mlir::DictionaryAttr::get(context));
      }
    for (mlir::func::ReturnOp ret : returns) {
      builder.setInsertionPoint(ret);
      for (mlir::OpOperand &operand : ret->getOpOperands())
        if (decided.results[operand.getOperandNumber()])
          operand.set(castTo(builder, ret.getLoc(), ptr, operand.get(),
                             !offsets.hasOffset(operand.get())));
    }
    function.setFunctionType(mlir::FunctionType::get(context, inputs, results));
    narrowed[name] = std::move(decided);
  }

  module.walk([&](mlir::func::CallOp call) {
    auto found = narrowed.find(call.getCalleeAttr().getAttr());
    if (found == narrowed.end())
      return;
    const Narrowed &decided = found->second;
    builder.setInsertionPoint(beforeTryMarkers(call));
    for (auto [index, input] : llvm::enumerate(decided.wide.getInputs()))
      if (decided.inputs[index])
        call->setOperand(
            index, castTo(builder, call.getLoc(), ptr, call->getOperand(index),
                          !offsets.hasOffset(call->getOperand(index))));
    builder.setInsertionPointAfter(call);
    for (auto [index, result] : llvm::enumerate(decided.wide.getResults())) {
      if (!decided.results[index])
        continue;
      mlir::Value value = call->getResult(index);
      value.setType(ptr);
      mlir::Value handle = castTo(builder, call.getLoc(), result, value);
      value.replaceAllUsesExcept(handle, handle.getDefiningOp());
    }
  });
  return narrowed.size();
}

} // namespace py::lowering::runtime::cleanup
