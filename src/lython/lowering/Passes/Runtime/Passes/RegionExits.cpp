// Writes out as blocks every `scf.if` whose arm holds an owned value across a
// call that may raise, so the unwind cleanup sees the value at the call.
//
// The unwind cleanup (Ownership.cpp, `insertUnwindCleanupReleases`) releases
// what a frame holds when a call unwinds out of it. A call nested in a region
// op is wired through an anchor placed before the region's top-level ancestor,
// and the cleanup there can name only values that dominate the anchor -- so a
// value made INSIDE the arm, before the call, was never released:
//
//     t = MaterializeRead(x)        # a box made in the slow arm
//     r = original(t)               # raises: t leaks
//
// Written as blocks, the arm is function-level code, the value is a group like
// any other, and the existing model places its cleanup.
//
// ⛔ Not every `scf.if`: the int and float fast/slow diamonds are everywhere,
// and an arm that makes nothing owned before its calls (or makes it and lets it
// die first) loses nothing to the anchor. Writing those out would only add
// blocks for the ownership phases to walk.
// ⛔ Not the anchor's cleanup reaching into the region (a slot the arm writes
// the handle to): that is a second cleanup model beside the first, and every
// release path through the arm would have to clear it.

#include "Ownership.h"
#include "Common/PythonSourceRange.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

#include <cstdlib>
#include <memory>

namespace py::lowering {
namespace {

namespace own = py::ownership;

bool traceEnabled() {
  static const bool enabled =
      std::getenv("LYTHON_TRACE_REGION_FLATTEN") != nullptr;
  return enabled;
}

// A value this op makes that the frame then owns: an owned call result, or a
// value marked frame-owned where it was produced.
// ⛔ Not an immortal one (a literal's static object, the deferred-int
// stand-in): its token is owned in name only, and its release does nothing,
// so an unwind that skips it loses nothing.
bool producesOwnedValue(mlir::Operation *op, mlir::SymbolTable &symbols) {
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    auto callee = symbols.lookup<mlir::func::FuncOp>(call.getCallee());
    if (!callee)
      return false;
    if (auto primitive =
            callee->getAttrOfType<mlir::StringAttr>("ly.runtime.primitive"))
      if (primitive.getValue() == "from_static" ||
          primitive.getValue() == "deferred_stand_in")
        return false;
    for (unsigned index = 0, end = call.getNumResults(); index < end; ++index)
      if (own::callResultGroupIsOwned(callee, index))
        return true;
    return false;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op))
    return cast->hasAttr(own::kOwnedLocalObjectAttr);
  return false;
}

bool mayRaise(mlir::func::CallOp call, mlir::SymbolTable &symbols) {
  auto callee = symbols.lookup<mlir::func::FuncOp>(call.getCallee());
  return callee && (own::isRaiseLikeFunction(callee) ||
                    own::mayRaisePythonException(callee));
}

// The op of `block` that is `op` or holds it.
mlir::Operation *ancestorIn(mlir::Block *block, mlir::Operation *op) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

// Does the call `at` take this operand's reference -- one its callee's
// contract transfers? Then the value is the callee's while the callee runs,
// and an unwind out of it is the callee's to release.
bool transfersOperand(mlir::OpOperand &use, mlir::Operation *at,
                      mlir::SymbolTable &symbols) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(at);
  if (!call || use.getOwner() != at)
    return false;
  auto callee = symbols.lookup<mlir::func::FuncOp>(call.getCallee());
  auto transferred =
      callee ? callee->getAttrOfType<mlir::ArrayAttr>(own::kTransferArgsAttr)
             : mlir::ArrayAttr();
  if (!transferred)
    return false;
  for (mlir::Attribute index : transferred)
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(index))
      if (integer.getInt() == static_cast<std::int64_t>(use.getOperandNumber()))
        return true;
  return false;
}

// Does `block` make an owned value before `at` that is still used at `at` or
// after it -- a value the frame holds while `at` runs?
// A value whose last uses are `at` itself, one of which `at` takes the
// reference of, is not held: the lanes it hands `at` beside that one travel
// with it (an exception's header, payload and message lanes).
bool holdsOwnedValueAcross(mlir::Block *block, mlir::Operation *at,
                           mlir::SymbolTable &symbols) {
  for (mlir::Operation &op : *block) {
    if (&op == at)
      return false;
    if (!producesOwnedValue(&op, symbols))
      continue;
    bool usedAfter = false;
    bool usedAt = false;
    bool transferredAt = false;
    for (mlir::Value result : op.getResults())
      for (mlir::OpOperand &use : result.getUses()) {
        mlir::Operation *inBlock = ancestorIn(block, use.getOwner());
        if (!inBlock)
          continue;
        if (inBlock == at) {
          usedAt = true;
          transferredAt |= transfersOperand(use, at, symbols);
        } else if (at->isBeforeInBlock(inBlock)) {
          usedAfter = true;
        }
      }
    if (usedAfter || (usedAt && !transferredAt))
      return true;
  }
  return false;
}

// Every object `ifOp` yields is one its arm made and owns, so written out as
// blocks the merge is an owned value arriving on each edge.
// ⛔ Not an arm that yields a value it borrows (a list element read through a
// bounds check): as an if result that is a view of the operand, as a block
// argument it is a merge that needs a token of its own, and the retain for it
// cannot always be spelled where it would have to go.
bool yieldsOnlyOwnedObjects(mlir::scf::IfOp ifOp, mlir::SymbolTable &symbols) {
  for (mlir::Region *arm : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    if (arm->empty())
      continue;
    mlir::Operation *yield = arm->front().getTerminator();
    for (mlir::Value value : yield->getOperands()) {
      if (!mlir::isa<mlir::BaseMemRefType>(value.getType()))
        continue;
      mlir::Operation *producer = value.getDefiningOp();
      if (!producer || producer->getBlock() != &arm->front() ||
          !producesOwnedValue(producer, symbols))
        return false;
    }
  }
  return true;
}

// Writes `ifOp` out as blocks in the region that holds it, which must be one
// that takes several (the function's body, or an arm written out already).
void writeOutAsBlocks(mlir::scf::IfOp ifOp) {
  mlir::IRRewriter rewriter(ifOp.getContext());
  mlir::Location loc = ifOp.getLoc();
  mlir::Block *head = ifOp->getBlock();
  mlir::Block *tail = rewriter.splitBlock(head, std::next(ifOp->getIterator()));
  for (mlir::Type type : ifOp.getResultTypes())
    tail->addArgument(type, loc);
  auto inlineArm = [&](mlir::Region &arm) -> mlir::Block * {
    mlir::Block *entry = &arm.front();
    mlir::Operation *yield = arm.back().getTerminator();
    rewriter.setInsertionPoint(yield);
    mlir::cf::BranchOp::create(rewriter, yield->getLoc(), tail,
                               yield->getOperands());
    rewriter.eraseOp(yield);
    rewriter.inlineRegionBefore(arm, tail);
    return entry;
  };
  mlir::Block *thenEntry = inlineArm(ifOp.getThenRegion());
  mlir::Block *elseEntry = ifOp.getElseRegion().empty()
                               ? tail
                               : inlineArm(ifOp.getElseRegion());
  rewriter.setInsertionPointToEnd(head);
  mlir::cf::CondBranchOp::create(rewriter, loc, ifOp.getCondition(), thenEntry,
                                 mlir::ValueRange{}, elseEntry,
                                 mlir::ValueRange{});
  rewriter.replaceOp(ifOp, tail->getArguments());
}

class RegionExitFlatteningPass
    : public mlir::PassWrapper<RegionExitFlatteningPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegionExitFlatteningPass)

  llvm::StringRef getArgument() const final {
    return "lython-region-exit-flattening";
  }
  llvm::StringRef getDescription() const final {
    return "write out as blocks each scf.if whose arm holds an owned value "
           "across a call that may raise";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    // LYTHON_ABLATE_REGION_FLATTEN=1 leaves every arm nested, so the leak this
    // pass closes and its repair come from one binary.
    static const bool ablated =
        std::getenv("LYTHON_ABLATE_REGION_FLATTEN") != nullptr;
    if (ablated)
      return;
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symbols(module);
    for (auto function : module.getOps<mlir::func::FuncOp>()) {
      // The functions the unwind cleanup wires (see `ehPhaseProcessesFunction`
      // there): Python-level code, not the manifest's.
      if (function.isDeclaration() || own::isRuntimeManifestFunction(function) ||
          !findPythonSourceLoc(function.getLoc()).has_value())
        continue;
      mlir::Region *body = &function.getBody();
      llvm::SetVector<mlir::Operation *> selected;
      function.walk([&](mlir::func::CallOp call) {
        if (call->getParentRegion() == body || !mayRaise(call, symbols))
          return;
        // Every arm between the call and the function body that holds an owned
        // value across it -- and so every region op from that arm outwards.
        for (mlir::Block *block = call->getBlock();
             block && block->getParent() != body;
             block = block->getParentOp()->getBlock()) {
          mlir::Operation *at = ancestorIn(block, call);
          if (!holdsOwnedValueAcross(block, at, symbols))
            continue;
          // ⛔ All the way out or not at all: an if written out inside an op
          // whose region takes one block (a loop's) is not valid IR.
          llvm::SmallVector<mlir::Operation *, 4> chain;
          mlir::Operation *owner = block->getParentOp();
          for (; owner && owner != function.getOperation();
               owner = owner->getParentOp()) {
            auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(owner);
            if (!ifOp || !yieldsOnlyOwnedObjects(ifOp, symbols))
              break;
            chain.push_back(owner);
          }
          if (owner == function.getOperation()) {
            selected.insert(chain.begin(), chain.end());
            if (traceEnabled())
              llvm::errs() << "[region-flatten-why] " << function.getSymName()
                           << " call " << call.getCallee() << "\n";
          }
          else if (traceEnabled())
            llvm::errs() << "[region-flatten] " << function.getSymName()
                         << ": a value held across a call under "
                         << owner->getName() << " stays nested\n";
          break;
        }
      });
      if (selected.empty())
        continue;
      // Outermost first: an arm can take blocks only once its own if is out.
      llvm::SmallVector<mlir::Operation *, 16> order(selected.begin(),
                                                     selected.end());
      llvm::DenseMap<mlir::Operation *, unsigned> depth;
      for (mlir::Operation *op : order) {
        unsigned levels = 0;
        for (mlir::Operation *parent = op->getParentOp();
             parent && parent != function.getOperation();
             parent = parent->getParentOp())
          ++levels;
        depth[op] = levels;
      }
      llvm::stable_sort(order, [&](mlir::Operation *lhs, mlir::Operation *rhs) {
        return depth[lhs] < depth[rhs];
      });
      for (mlir::Operation *op : order) {
        if (traceEnabled())
          llvm::errs() << "[region-flatten] " << function.getSymName() << " "
                       << op->getLoc() << "\n";
        writeOutAsBlocks(mlir::cast<mlir::scf::IfOp>(op));
      }
    }
  }
};

} // namespace
} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRegionExitFlatteningPass() {
  return std::make_unique<lowering::RegionExitFlatteningPass>();
}

} // namespace py
