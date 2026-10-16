#include "Passes/Runtime/Cleanup/Transforms.h"

#include "Common/PythonSourceRange.h"
#include "Ownership.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdlib>

namespace py::lowering::runtime::cleanup {

namespace {

enum class Verdict { Raises, Never, Body };

bool raiseSymbolName(llvm::StringRef name) {
  return name == "LyEH_ThrowException" || name == "LyEH_RethrowCurrent" ||
         name == "LyEH_StarRethrowResidual" ||
         name == "LyEH_StarRethrowSoleCollected" ||
         name == "LyEH_StarThrowCombined" || name.ends_with("_Raise");
}

// What a function is before its body is read. ⛔ A body compiled from Python
// is not read: `mayRaisePythonException` says it raises whatever it calls,
// and a primitive clone or a generator step has its own rule in Cleanup/EH.cpp
// that this one would have to mirror.
Verdict classify(mlir::FunctionOpInterface function) {
  llvm::StringRef name = function.getName();
  if (raiseSymbolName(name))
    return Verdict::Raises;
  if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(function.getOperation())) {
    if (ownership::isRaiseLikeFunction(func))
      return Verdict::Raises;
    if (ownership::isNonRaisingRuntimeSymbol(name))
      return Verdict::Never;
    if (func.isDeclaration())
      return ownership::mayRaisePythonException(func) ? Verdict::Raises
                                                      : Verdict::Never;
    if (func->hasAttr(ownership::kCallableTypeAttr) ||
        findPythonSourceLoc(func.getLoc()).has_value())
      return Verdict::Raises;
    return Verdict::Body;
  }
  if (ownership::isNonRaisingRuntimeSymbol(name))
    return Verdict::Never;
  if (function.isExternal())
    return name.starts_with("Ly") ? Verdict::Raises : Verdict::Never;
  return Verdict::Body;
}

} // namespace

// ⭐ A FUNCTION RAISES WHEN SOMETHING IT CALLS DOES; the seeds are the raise
// primitives, a call through a pointer (a hook or a callback can run Python),
// any py op left unlowered, and a declaration the name test says may raise.
// The rest is a least fixpoint over the reversed call graph: a cycle with no
// seed on it raises nothing.
//
// ⛔ Not the name test alone, which is what `mayRaisePythonException` was: every
// `Ly*` call was an unwind edge, so a literal of n strings or floats held
// n - 1 boxed elements across the n-th `LyUnicode_FromStatic` /
// `LyFloat_FromF64` -- neither of which can raise -- and the unwind cleanups
// it built were n^2 releases (400 elements: 10 s JIT, 260 s AOT).
//
// `LYTHON_ABLATE_CANNOT_RAISE=1` marks nothing.
unsigned markCannotRaiseFunctions(mlir::ModuleOp module) {
  if (std::getenv("LYTHON_ABLATE_CANNOT_RAISE"))
    return 0;
  mlir::SymbolTable symbols(module);
  llvm::SmallVector<mlir::FunctionOpInterface> functions;
  llvm::DenseMap<mlir::Operation *, unsigned> indexOf;
  for (mlir::Operation &op : module.getBody()->getOperations())
    if (auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(&op)) {
      indexOf[&op] = functions.size();
      functions.push_back(function);
    }

  llvm::SmallVector<bool> raises(functions.size(), false);
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>> callers(functions.size());
  llvm::SmallVector<unsigned> worklist;
  auto seed = [&](unsigned index) {
    if (!raises[index]) {
      raises[index] = true;
      worklist.push_back(index);
    }
  };

  for (unsigned index = 0; index < functions.size(); ++index) {
    mlir::FunctionOpInterface function = functions[index];
    Verdict verdict = classify(function);
    if (verdict == Verdict::Raises)
      seed(index);
    if (verdict != Verdict::Body)
      continue;
    bool direct = false;
    function->walk([&](mlir::Operation *op) {
      if (direct)
        return mlir::WalkResult::interrupt();
      if (op->getDialect() && op->getDialect()->getNamespace() == "py") {
        direct = true;
        return mlir::WalkResult::interrupt();
      }
      auto call = mlir::dyn_cast<mlir::CallOpInterface>(op);
      if (!call)
        return mlir::WalkResult::advance();
      auto symbol = mlir::dyn_cast_if_present<mlir::SymbolRefAttr>(
          call.getCallableForCallee());
      mlir::Operation *callee =
          symbol ? symbols.lookup(symbol.getRootReference()) : nullptr;
      auto found = callee ? indexOf.find(callee) : indexOf.end();
      if (found == indexOf.end()) {
        direct = true;
        return mlir::WalkResult::interrupt();
      }
      callers[found->second].push_back(index);
      return mlir::WalkResult::advance();
    });
    if (direct)
      seed(index);
  }

  while (!worklist.empty()) {
    unsigned callee = worklist.pop_back_val();
    for (unsigned caller : callers[callee])
      seed(caller);
  }

  unsigned marked = 0;
  for (auto [index, function] : llvm::enumerate(functions)) {
    auto func = mlir::dyn_cast<mlir::func::FuncOp>(function.getOperation());
    if (!func || raises[index] || classify(function) != Verdict::Body ||
        !ownership::mayRaisePythonException(func))
      continue;
    func->setAttr(ownership::kCannotRaiseAttr,
                  mlir::UnitAttr::get(module.getContext()));
    ++marked;
  }
  return marked;
}

// ⭐ THE SAME FACT AT THE LLVM LAYER, so Cleanup/EH.cpp's
// `mayPropagatePythonException` (which reads `doesNotThrow`) agrees with
// `mayRaisePythonException`: a call the ownership pass gave no cleanup must
// not become an invoke either. ⛔ Not set on the func.func: the upstream
// func-to-llvm conversion drops `passthrough`.
void markCannotRaiseNoUnwind(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  for (auto function : module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    if (!function->hasAttr(ownership::kCannotRaiseAttr))
      continue;
    llvm::SmallVector<mlir::Attribute, 4> passthrough;
    if (mlir::ArrayAttr existing = function.getPassthroughAttr())
      passthrough.append(existing.begin(), existing.end());
    passthrough.push_back(mlir::StringAttr::get(context, "nounwind"));
    function.setPassthroughAttr(mlir::ArrayAttr::get(context, passthrough));
  }
}

} // namespace py::lowering::runtime::cleanup
