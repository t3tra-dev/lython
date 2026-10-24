// What this pins: no hand-written runtime loop keeps iterating after its answer
// is fixed.
//
// The shape, found in four places when the runtime was measured against CPython
// (2026-10-24), is an `scf.for` whose carried value reaches a state the rest of
// the iterations cannot leave:
//
//     %r = scf.for ... iter_args(%acc = %zero) {
//       %decided = arith.cmpi ne, %acc, %zero
//       %next = arith.select %decided, %acc, %this     once nonzero, stays
//       scf.yield %next
//     }
//
// `scf.for` has no break, so a first-difference search written this way reads
// both operands to the end. CPython's loops stop. The cost was not constant: the
// carry propagation of the bigint multiply ran to the end of the result on every
// row, which made `f *= i` O(n^2) per multiply (2.7 s where CPython takes 33 ms).
//
// A loop is reported when ONE carried value is absorbing -- the yield returns
// that value unchanged once a predicate of it alone holds -- and, once it holds,
// nothing else in the iteration is observable: every other carried value is
// yielded unchanged under the same predicate, and nothing in the body writes
// memory except inside an `scf.if` the predicate skips. Such a loop can stop at
// that point with the same results; that is the repair (`scf.while`).

#include "Driver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace {

std::vector<std::string> runtimeModulePaths() {
  std::vector<std::string> paths;
  for (const char *group : {"modules", "objects", "python"}) {
    llvm::SmallString<256> dir(LYTHON_RUNTIME_DIR);
    llvm::sys::path::append(dir, group);
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(dir, ec), end;
         it != end && !ec; it.increment(ec)) {
      llvm::StringRef path = it->path();
      if (llvm::sys::path::extension(path) == ".mlir")
        paths.push_back(path.str());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

// Whether `op` itself (not its regions) only reads. A call is judged by its
// callee's body; anything unknown writes.
bool onlyReads(mlir::Operation *op,
               const llvm::DenseSet<llvm::StringRef> &readOnly) {
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
    return readOnly.contains(call.getCallee());
  if (op->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> list;
  effects.getEffects(list);
  return llvm::all_of(list, [](const auto &effect) {
    return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
  });
}

// Functions whose bodies only read, by fixpoint over every module. Starts empty,
// so a recursive function is never read-only; that only loses findings.
llvm::DenseSet<llvm::StringRef>
readOnlyFunctions(llvm::ArrayRef<mlir::ModuleOp> modules) {
  llvm::DenseSet<llvm::StringRef> known;
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::ModuleOp module : modules)
      module.walk([&](mlir::func::FuncOp function) {
        if (function.isExternal() || known.contains(function.getSymName()))
          return;
        bool reads = true;
        function.walk([&](mlir::Operation *op) {
          if (op != function.getOperation() && !onlyReads(op, known))
            reads = false;
        });
        if (reads) {
          known.insert(function.getSymName());
          changed = true;
        }
      });
  }
  return known;
}

// Whether `value` is computed, inside `loop`, from `carried` and loop
// invariants only (by side-effect-free ops), and does use `carried`.
bool dependsOnlyOn(mlir::Value value, mlir::Value carried,
                   mlir::scf::ForOp loop, bool &touches) {
  if (value == carried) {
    touches = true;
    return true;
  }
  if (!loop.getRegion().isAncestor(value.getParentRegion()))
    return true;
  mlir::Operation *def = value.getDefiningOp();
  if (!def || def->getNumRegions() != 0 || !mlir::isMemoryEffectFree(def))
    return false;
  for (mlir::Value operand : def->getOperands())
    if (!dependsOnlyOn(operand, carried, loop, touches))
      return false;
  return true;
}

bool predicateOf(mlir::Value condition, mlir::Value carried,
                 mlir::scf::ForOp loop) {
  bool touches = false;
  return dependsOnlyOn(condition, carried, loop, touches) && touches;
}

// The state a carried value cannot leave: once `predicate == whenTrue`, the
// yield hands the value back unchanged. `gate` is set when the yield comes out
// of an `scf.if` on the predicate, whose other branch then runs only before it.
struct Absorbing {
  unsigned position;
  mlir::Value predicate;
  bool whenTrue;
  mlir::scf::IfOp gate;
  bool gateSkipsThen;
};

bool sameConstant(mlir::Value a, mlir::Value b) {
  auto ca = a.getDefiningOp<mlir::arith::ConstantOp>();
  auto cb = b.getDefiningOp<mlir::arith::ConstantOp>();
  return ca && cb && ca.getValue() == cb.getValue();
}

// The branch of `gate` that leaves `carried` alone yields it back, or yields the
// constant the predicate compares it with (which selects that branch again).
bool branchKeeps(mlir::Region &branch, unsigned result, mlir::Value carried,
                 mlir::Value condition) {
  mlir::Block &block = branch.front();
  for (mlir::Operation &op : block.without_terminator())
    if (!mlir::isMemoryEffectFree(&op))
      return false;
  mlir::Value yielded = block.getTerminator()->getOperand(result);
  if (yielded == carried)
    return true;
  auto compare = condition.getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare)
    return false;
  mlir::Value other = compare.getLhs() == carried   ? compare.getRhs()
                      : compare.getRhs() == carried ? compare.getLhs()
                                                    : mlir::Value();
  return other && sameConstant(yielded, other);
}

std::optional<Absorbing> absorbingAt(mlir::scf::ForOp loop, unsigned position) {
  mlir::Value carried = loop.getRegionIterArgs()[position];
  mlir::Value yielded = loop.getBody()->getTerminator()->getOperand(position);
  mlir::Operation *def = yielded.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    mlir::Value condition = select.getCondition();
    if (!predicateOf(condition, carried, loop))
      return std::nullopt;
    if (select.getTrueValue() == carried)
      return Absorbing{position, condition, true, {}, false};
    if (select.getFalseValue() == carried)
      return Absorbing{position, condition, false, {}, false};
    return std::nullopt;
  }
  if (mlir::isa<mlir::arith::AndIOp, mlir::arith::OrIOp>(def) &&
      carried.getType().isInteger(1) &&
      (def->getOperand(0) == carried || def->getOperand(1) == carried))
    return Absorbing{position, carried, mlir::isa<mlir::arith::OrIOp>(def), {},
                     false};
  if (auto gate = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
    mlir::Value condition = gate.getCondition();
    if (gate.getElseRegion().empty() ||
        !predicateOf(condition, carried, loop))
      return std::nullopt;
    unsigned result = mlir::cast<mlir::OpResult>(yielded).getResultNumber();
    if (branchKeeps(gate.getElseRegion(), result, carried, condition))
      return Absorbing{position, condition, false, gate, false};
    if (branchKeeps(gate.getThenRegion(), result, carried, condition))
      return Absorbing{position, condition, true, gate, true};
  }
  return std::nullopt;
}

// Every other carried value is yielded unchanged once `state` holds.
bool othersFreeze(mlir::scf::ForOp loop, Absorbing state) {
  auto args = loop.getRegionIterArgs();
  mlir::Operation *yield = loop.getBody()->getTerminator();
  for (unsigned j = 0; j < args.size(); ++j) {
    if (j == state.position)
      continue;
    mlir::Value carried = args[j];
    mlir::Value yielded = yield->getOperand(j);
    if (yielded == carried)
      continue;
    if (auto select = yielded.getDefiningOp<mlir::arith::SelectOp>()) {
      mlir::Value kept = state.whenTrue ? select.getTrueValue()
                                        : select.getFalseValue();
      if (select.getCondition() == state.predicate && kept == carried)
        continue;
    }
    if (state.gate && yielded.getDefiningOp() == state.gate.getOperation()) {
      unsigned result = mlir::cast<mlir::OpResult>(yielded).getResultNumber();
      mlir::Region &kept = state.gateSkipsThen ? state.gate.getThenRegion()
                                               : state.gate.getElseRegion();
      if (kept.front().getTerminator()->getOperand(result) == carried)
        continue;
    }
    return false;
  }
  return true;
}

// Nothing in the body writes, except inside the gate's branch that the
// absorbed state no longer takes.
bool bodyOnlyReads(mlir::scf::ForOp loop, Absorbing state,
                   const llvm::DenseSet<llvm::StringRef> &readOnly) {
  mlir::Region *active = nullptr;
  if (state.gate)
    active = state.gateSkipsThen ? &state.gate.getElseRegion()
                                 : &state.gate.getThenRegion();
  bool reads = true;
  loop.getBody()->walk([&](mlir::Operation *op) {
    if (active && active->isAncestor(op->getParentRegion()))
      return;
    if (!onlyReads(op, readOnly))
      reads = false;
  });
  return reads;
}

class RuntimeSources {
public:
  RuntimeSources() : context(makeRegistry()) {
    context.allowUnregisteredDialects(true);
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(const std::string &path) {
    return mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
  }

private:
  static mlir::DialectRegistry makeRegistry() {
    mlir::DialectRegistry registry;
    lython::driver::registerLythonDialects(registry);
    return registry;
  }

  mlir::MLIRContext context;
};

std::string where(mlir::Operation *op) {
  if (auto loc = mlir::dyn_cast<mlir::FileLineColLoc>(op->getLoc()))
    return llvm::sys::path::filename(loc.getFilename().getValue()).str() +
           ":" + std::to_string(loc.getLine());
  return "?";
}

// ⭐ The property, over every runtime module.
TEST(RuntimeLoopExitTest, NoLoopRunsOnAfterItsAnswerIsFixed) {
  RuntimeSources sources;
  std::vector<std::string> paths = runtimeModulePaths();
  ASSERT_GE(paths.size(), 10u) << "expected the runtime module tree at "
                               << LYTHON_RUNTIME_DIR;

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owned;
  llvm::SmallVector<mlir::ModuleOp, 32> modules;
  for (const std::string &path : paths) {
    owned.push_back(sources.parse(path));
    ASSERT_TRUE(owned.back()) << "could not parse " << path;
    modules.push_back(*owned.back());
  }
  llvm::DenseSet<llvm::StringRef> readOnly = readOnlyFunctions(modules);
  // A test whose purity analysis found nothing would only see store-free loops.
  EXPECT_TRUE(readOnly.contains("__ly_unicode_get"))
      << "the read-only fixpoint did not reach a plain code-point read";

  unsigned loops = 0;
  std::vector<std::string> report;
  for (mlir::ModuleOp module : modules)
    module.walk([&](mlir::scf::ForOp loop) {
      ++loops;
      for (unsigned position = 0; position < loop.getNumRegionIterArgs();
           ++position) {
        std::optional<Absorbing> state = absorbingAt(loop, position);
        if (!state || !othersFreeze(loop, *state) ||
            !bodyOnlyReads(loop, *state, readOnly))
          continue;
        auto function = loop->getParentOfType<mlir::func::FuncOp>();
        report.push_back(where(loop) + " in @" +
                         function.getSymName().str() + ": carried value #" +
                         std::to_string(position) + " is fixed once its " +
                         (state->gate ? "gate" : "select") + " says so");
        break;
      }
    });
  EXPECT_GE(loops, 100u);

  std::string joined;
  for (const std::string &line : report)
    joined += "\n  " + line;
  EXPECT_TRUE(report.empty())
      << "a loop keeps iterating after its result can no longer change; stop "
         "it there (scf.while):" << joined;
}

} // namespace
