// What this pins: no hand-written runtime function raises while holding an owned
// object whose only release is on the other side of the raise.
//
// That defect was found three times in one session and repaired three times --
// `__ly_io_fopen`, ten `posix.mlir` entry points, five `_random`/`_time` raise
// helpers -- and the shape was identical every time:
//
//     %obj = call @Something()            owned result, refcount 1
//     ... call @__ly_raise_something()    does not return
//     call @LyX_DecRef(%obj)              never reached
//
// It leaked whole objects on every raising path: 789 B for `stdlib_os_fs`, 467 for
// `w3_cross_os_try_rebind`, 83 for `io_file`. The comment in `__ly_io_fopen` had
// said so in prose for a release ("the encoded path leaks on that exception path")
// and prose does not fail a build.
//
// WHY A UNIT TEST AND NOT A PIPELINE VERIFIER, which is where it belongs. Two
// blocks, both measured rather than assumed:
//
//   * The affine-ownership verifier already models a raise as a function exit and
//     already reports "reaches function exit without release, transfer, or owned
//     return". It is blind here on purpose: `callInFunctionTopLevelRegion`
//     restricts unwind-exit modelling to the function's top-level region, because
//     refcount-insertion cannot emit cleanup inside a nested single-block region
//     (`scf.if` arms). Widening the model without that would reject every nested
//     slow path in USER code, which the compiler cannot currently repair -- the
//     Wave 0 hand-off. The runtime modules are different: they are hand-written,
//     so every report here is fixable by a person.
//   * Phase coverage is uneven anyway. A deliberate double release in
//     `posix.mlir` compiles and crashes at runtime, while the same shape in
//     `_io.mlir` is refused -- posix is not in the module set the runtime
//     pre-lowering phase verifies. A check that reads the FILES has no such hole.
//
// So this reads the runtime sources directly, with real parsing and real
// dominance, and covers all of them by construction.

#include "Driver.h"
#include "PyDialectTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <system_error>
#include <vector>

namespace {

// The one call that never returns by definition. Everything else is derived.
constexpr llvm::StringRef kThrower = "LyEH_ThrowException";
constexpr llvm::StringRef kOwnedResults = "ly.ownership.owned_results";
constexpr llvm::StringRef kReleaseArgs = "ly.ownership.release_args";
constexpr llvm::StringRef kTransferArgs = "ly.ownership.transfer_args";

std::vector<std::string> runtimeModulePaths() {
  std::vector<std::string> paths;
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(LYTHON_RUNTIME_MODULES_DIR, ec), end;
       it != end && !ec; it.increment(ec)) {
    llvm::StringRef path = it->path();
    if (llvm::sys::path::extension(path) == ".mlir")
      paths.push_back(path.str());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

llvm::DenseSet<unsigned> indexSet(mlir::Operation *op, llvm::StringRef name) {
  llvm::DenseSet<unsigned> out;
  auto array = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!array)
    return out;
  for (mlir::Attribute entry : array)
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(entry))
      out.insert(static_cast<unsigned>(integer.getInt()));
  return out;
}

// Functions that never return, by fixpoint from the thrower.
//
// A function is non-returning when some block that DOMINATES every `func.return`
// in it calls a non-returning function -- that is, when every return is behind a
// call that does not come back. `__ly_posix_throw` qualifies (straight line, one
// return, the call is in the same block); `__ly_io_fopen` does not (its raising
// blocks dominate nothing).
llvm::DenseSet<llvm::StringRef> nonReturning(mlir::ModuleOp module) {
  llvm::DenseSet<llvm::StringRef> known;
  known.insert(kThrower);
  bool changed = true;
  while (changed) {
    changed = false;
    module.walk([&](mlir::func::FuncOp function) {
      if (function.isExternal() || known.contains(function.getSymName()))
        return;
      llvm::SmallVector<mlir::Operation *, 4> returns;
      function.walk([&](mlir::func::ReturnOp ret) { returns.push_back(ret); });
      if (returns.empty())
        return;
      mlir::DominanceInfo dominance(function);
      bool blocked = true;
      for (mlir::Operation *ret : returns) {
        bool behindACall = false;
        function.walk([&](mlir::func::CallOp call) {
          if (behindACall || !known.contains(call.getCallee()))
            return;
          if (dominance.properlyDominates(call.getOperation(), ret))
            behindACall = true;
        });
        if (!behindACall) {
          blocked = false;
          break;
        }
      }
      if (blocked) {
        known.insert(function.getSymName());
        changed = true;
      }
    });
  }
  return known;
}

struct Finding {
  std::string function;
  std::string producer;
  std::string raise;
};

// An owned call result that reaches a non-returning call with nothing having
// released or transferred it first.
//
// "Reaches" is dominance in the other direction: the producer dominates the raise
// (so the object exists there) and no release/transfer of it dominates the raise
// (so it is still held). A release placed AFTER the raise dominates nothing, which
// is exactly why the defect was invisible to reading.
std::vector<Finding> findingsFor(mlir::ModuleOp module,
                                 const llvm::DenseSet<llvm::StringRef> &noReturn) {
  std::vector<Finding> findings;
  llvm::DenseMap<llvm::StringRef, mlir::func::FuncOp> byName;
  module.walk([&](mlir::func::FuncOp f) { byName[f.getSymName()] = f; });

  module.walk([&](mlir::func::FuncOp function) {
    if (function.isExternal() || noReturn.contains(function.getSymName()))
      return;
    mlir::DominanceInfo dominance(function);

    llvm::SmallVector<mlir::Operation *, 4> raises;
    function.walk([&](mlir::func::CallOp call) {
      if (noReturn.contains(call.getCallee()))
        raises.push_back(call.getOperation());
    });
    if (raises.empty())
      return;

    function.walk([&](mlir::func::CallOp producer) {
      auto callee = byName.find(producer.getCallee());
      if (callee == byName.end())
        return;
      llvm::DenseSet<unsigned> owned =
          indexSet(callee->second.getOperation(), kOwnedResults);
      if (owned.empty())
        return;
      for (unsigned index : owned) {
        if (index >= producer.getNumResults())
          continue;
        mlir::Value object = producer.getResult(index);
        for (mlir::Operation *raise : raises) {
          if (!dominance.properlyDominates(producer.getOperation(), raise))
            continue;
          // Handed to the raise itself: that is the repair, not the defect.
          bool handedOver = false;
          for (mlir::Value operand : raise->getOperands())
            if (operand == object)
              handedOver = true;
          if (handedOver)
            continue;
          bool consumed = false;
          for (mlir::Operation *user : object.getUsers()) {
            auto call = mlir::dyn_cast<mlir::func::CallOp>(user);
            if (!call || !dominance.properlyDominates(user, raise))
              continue;
            auto target = byName.find(call.getCallee());
            if (target == byName.end())
              continue;
            llvm::DenseSet<unsigned> release =
                indexSet(target->second.getOperation(), kReleaseArgs);
            llvm::DenseSet<unsigned> transfer =
                indexSet(target->second.getOperation(), kTransferArgs);
            for (unsigned position = 0; position < call.getNumOperands();
                 ++position) {
              if (call.getOperand(position) != object)
                continue;
              if (release.contains(position) || transfer.contains(position))
                consumed = true;
            }
          }
          if (!consumed)
            findings.push_back({function.getSymName().str(),
                                producer.getCallee().str(),
                                mlir::cast<mlir::func::CallOp>(raise)
                                    .getCallee()
                                    .str()});
        }
      }
    });
  });
  return findings;
}

class RuntimeSources {
public:
  // The compiler's own registry, not a hand-picked list: these files use the py
  // dialect and whatever the manifest reaches for, and a list maintained here
  // would drift into "test cannot parse the tree" the first time a module used
  // something new.
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

// ⭐ The property, over every runtime module.
TEST(RuntimeRaisePathTest, NoOwnedObjectIsHeldAcrossARaise) {
  RuntimeSources sources;
  std::vector<std::string> paths = runtimeModulePaths();
  // A test that parsed nothing would pass. Say so instead.
  ASSERT_GE(paths.size(), 10u) << "expected the runtime module tree at "
                               << LYTHON_RUNTIME_MODULES_DIR;

  unsigned parsed = 0;
  unsigned withRaises = 0;
  std::vector<std::string> report;
  for (const std::string &path : paths) {
    mlir::OwningOpRef<mlir::ModuleOp> module = sources.parse(path);
    ASSERT_TRUE(module) << "could not parse " << path;
    ++parsed;
    llvm::DenseSet<llvm::StringRef> noReturn = nonReturning(*module);
    // The thrower alone means nothing was derived; every module that raises has
    // at least one helper above it.
    if (noReturn.size() > 1)
      ++withRaises;
    for (const Finding &finding : findingsFor(*module, noReturn))
      report.push_back(llvm::sys::path::filename(path).str() + ": " +
                       finding.function + " holds the owned result of @" +
                       finding.producer + " across @" + finding.raise);
  }
  EXPECT_EQ(parsed, paths.size());
  // Without this the test could pass by deriving no raise helpers at all.
  EXPECT_GE(withRaises, 3u) << "no module derived a non-returning helper; the "
                               "fixpoint is not working";

  std::string joined;
  for (const std::string &line : report)
    joined += "\n  " + line;
  EXPECT_TRUE(report.empty())
      << "an owned object is held across a call that does not return, so its "
         "release is unreachable on that path:" << joined;
}

} // namespace
