// Which values a program can watch being released, and the keep-alives and
// module-global clears for every other value removed.
//
// The emitter keeps every local referenced to where CPython's frame lets it
// go (py.keep_alive) and clears the main module's globals at the end
// (py.global.clear), whatever the value is. Here each of those is kept only
// when its type can run code or touch the outside world on release:
//
//   - a class whose instances run `__del__` (the driver gives such a class
//     `__ly_finalize__`), or that can hold such a value in a field;
//   - a generator or coroutine, whose `finally` and `with` exits run when it
//     is closed on release;
//   - a file, which is flushed and closed on release;
//   - `object`, a protocol and a callable, which can hold any of these --
//     when the program has any of these at all;
//   - a container, union or generic of any of them.
//
// A value of a base class is watched when any subclass is, since the base
// names what the value may be.

#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#define GET_OP_CLASSES
#include "PyOps.h.inc"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <memory>

namespace py::lowering {
namespace {

class ObservableRelease {
public:
  explicit ObservableRelease(mlir::ModuleOp module) : module(module) {
    module.walk([&](py::ClassOp classOp) { classes.push_back(classOp); });
    for (py::ClassOp classOp : classes)
      if (auto bases = classOp->getAttrOfType<mlir::ArrayAttr>("base_names"))
        for (mlir::Attribute base : bases)
          if (auto name = mlir::dyn_cast<mlir::StringAttr>(base))
            if (py::ClassOp baseOp = lookup(name.getValue()))
              subclasses[baseOp.getOperation()].push_back(classOp);
    growWatchedClasses();
    // An erased value -- `object`, a protocol, a callable's captures -- can
    // only be something the program makes. A program that makes nothing
    // watched has erased values that hold nothing watched.
    // ⛔ Not "erased is always watched": every program that passes a
    // function as a value kept it, and what it captured, to the end of the
    // frame for nothing.
    llvm::DenseSet<mlir::Type> seen;
    module.walk([&](mlir::Operation *op) {
      if (erasedWatched)
        return;
      auto consider = [&](mlir::Type type) {
        if (!erasedWatched && seen.insert(type).second && watched(type))
          erasedWatched = true;
      };
      for (mlir::Type type : op->getResultTypes())
        consider(type);
      for (mlir::Region &region : op->getRegions())
        for (mlir::Block &block : region)
          for (mlir::BlockArgument argument : block.getArguments())
            consider(argument.getType());
    });
    if (erasedWatched)
      growWatchedClasses();
  }

  bool watched(mlir::Type type) {
    llvm::DenseSet<mlir::Type> visiting;
    return watched(type, visiting);
  }

private:
  // A field can hold an instance of a class declared later, and a base is
  // watched because of a subclass: both are facts about the whole set, so
  // the set grows until it stops.
  void growWatchedClasses() {
    for (bool grew = true; grew;) {
      grew = false;
      for (py::ClassOp classOp : classes)
        if (!watchedClasses.contains(classOp.getOperation()) &&
            classIsWatched(classOp)) {
          watchedClasses.insert(classOp.getOperation());
          grew = true;
        }
    }
  }

  py::ClassOp lookup(llvm::StringRef contract) {
    auto find = [&](llvm::StringRef name) -> py::ClassOp {
      return mlir::dyn_cast_or_null<py::ClassOp>(
          mlir::SymbolTable::lookupSymbolIn(module.getOperation(), name));
    };
    if (py::ClassOp classOp = find(contract))
      return classOp;
    llvm::StringRef shortName = contract.rsplit('.').second;
    if (!shortName.empty() && shortName != contract)
      return find(shortName);
    return {};
  }

  bool classIsWatched(py::ClassOp classOp) {
    if (auto methods = classOp->getAttrOfType<mlir::ArrayAttr>("method_names"))
      for (mlir::Attribute method : methods)
        if (auto name = mlir::dyn_cast<mlir::StringAttr>(method))
          if (name.getValue() == "__ly_finalize__")
            return true;
    if (auto fields = classOp->getAttrOfType<mlir::ArrayAttr>("field_types"))
      for (mlir::Attribute field : fields)
        if (auto type = mlir::dyn_cast<mlir::TypeAttr>(field))
          if (watched(type.getValue()))
            return true;
    for (py::ClassOp subclass : subclasses.lookup(classOp.getOperation()))
      if (watchedClasses.contains(subclass.getOperation()))
        return true;
    return false;
  }

  bool watched(mlir::Type type, llvm::DenseSet<mlir::Type> &visiting) {
    if (!type || !visiting.insert(type).second)
      return false;
    if (mlir::isa<py::ProtocolType, py::CallableType>(type))
      return erasedWatched;
    if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
      for (mlir::Type member : unionType.getMemberTypes())
        if (watched(member, visiting))
          return true;
      return false;
    }
    auto contract = mlir::dyn_cast<py::ContractType>(type);
    if (!contract)
      return false;
    llvm::StringRef name = contract.getContractName();
    if (name == "builtins.object")
      return erasedWatched;
    if (name == "_io.TextIOWrapper" || name == "_io.FileIO")
      return true;
    if (name.starts_with("types.") &&
        (name.contains("Generator") || name.contains("Coroutine")))
      return true;
    if (py::ClassOp classOp = lookup(name))
      if (watchedClasses.contains(classOp.getOperation()))
        return true;
    for (mlir::Type argument : contract.getArguments())
      if (watched(argument, visiting))
        return true;
    return false;
  }

  mlir::ModuleOp module;
  llvm::SmallVector<py::ClassOp, 32> classes;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<py::ClassOp, 2>>
      subclasses;
  llvm::DenseSet<mlir::Operation *> watchedClasses;
  bool erasedWatched = false;
};

class ObservableReleasePass
    : public mlir::PassWrapper<ObservableReleasePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ObservableReleasePass)

  llvm::StringRef getArgument() const final {
    return "lython-observable-release";
  }
  llvm::StringRef getDescription() const final {
    return "keep values referenced to CPython's release point only where the "
           "release can be observed";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<mlir::Operation *, 64> unwatched;
    bool any = false;
    module.walk([&](mlir::Operation *op) {
      any |= mlir::isa<py::KeepAliveOp, py::GlobalClearOp>(op);
    });
    if (!any)
      return;
    ObservableRelease observable(module);
    module.walk([&](mlir::Operation *op) {
      if (auto keep = mlir::dyn_cast<py::KeepAliveOp>(op)) {
        mlir::Type asked = keep.getObservedAs()
                               ? *keep.getObservedAs()
                               : keep.getObject().getType();
        if (!observable.watched(asked))
          unwatched.push_back(op);
      } else if (auto clear = mlir::dyn_cast<py::GlobalClearOp>(op)) {
        if (!observable.watched(clear.getType()))
          unwatched.push_back(op);
      }
    });
    for (mlir::Operation *op : unwatched)
      op->erase();
  }
};

} // namespace
} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createObservableReleasePass() {
  return std::make_unique<lowering::ObservableReleasePass>();
}

} // namespace py
