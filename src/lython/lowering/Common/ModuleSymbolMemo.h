#ifndef LYTHON_LOWERING_COMMON_MODULESYMBOLMEMO_H
#define LYTHON_LOWERING_COMMON_MODULESYMBOLMEMO_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <utility>

namespace py {

// The module symbols a lookup has found, kept while a phase that installs
// this runs: the runtime lowering asks for the same runtime functions,
// globals and type objects by name for every op it lowers. A name that was
// not found is looked up again, so a symbol made by any builder is found.
// ⛔ Not the module's symbol list on every ask, which scans every runtime
// symbol for each one: 1 s of a unittest compile.
// ⛔ Not mlir::SymbolTable, which must be told of every symbol added, and
// symbols are added through dozens of builders.
// ⛔ And not installed where a symbol can disappear or be renamed unseen:
// the installer erases one only through a path that calls `forget`.
// LYTHON_VERIFY_SYMBOL_MEMO=1 looks every remembered answer up again and
// aborts where they differ; LYTHON_ABLATE_SYMBOL_MEMO=1 remembers nothing.
class ModuleSymbolMemo {
public:
  explicit ModuleSymbolMemo(mlir::ModuleOp module)
      : module(module), previous(active),
        verify(std::getenv("LYTHON_VERIFY_SYMBOL_MEMO") != nullptr) {
    if (!std::getenv("LYTHON_ABLATE_SYMBOL_MEMO"))
      active = this;
  }
  ~ModuleSymbolMemo() {
    if (active == this)
      active = previous;
  }
  ModuleSymbolMemo(const ModuleSymbolMemo &) = delete;
  ModuleSymbolMemo &operator=(const ModuleSymbolMemo &) = delete;

  // The memo of `module`, if one is installed on this thread.
  static ModuleSymbolMemo *of(mlir::ModuleOp module) {
    return active && active->module == module ? active : nullptr;
  }

  mlir::Operation *lookup(mlir::StringAttr name) {
    auto found = symbols.find(name);
    if (found == symbols.end()) {
      mlir::Operation *symbol =
          mlir::SymbolTable::lookupSymbolIn(module.getOperation(), name);
      if (symbol)
        symbols[name] = symbol;
      return symbol;
    }
    if (verify) {
      mlir::Operation *symbol =
          mlir::SymbolTable::lookupSymbolIn(module.getOperation(), name);
      if (symbol != found->second) {
        llvm::errs() << "internal: the symbol memo answered a different op "
                        "for '"
                     << name.getValue() << "'\n";
        std::abort();
      }
    }
    return found->second;
  }
  void remember(mlir::Operation *symbol) {
    symbols[mlir::SymbolTable::getSymbolName(symbol)] = symbol;
  }
  void forget(mlir::Operation *op) {
    auto name =
        op->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName());
    if (!name)
      return;
    auto found = symbols.find(name);
    if (found != symbols.end() && found->second == op)
      symbols.erase(found);
  }

private:
  mlir::ModuleOp module;
  ModuleSymbolMemo *previous;
  bool verify;
  llvm::DenseMap<mlir::StringAttr, mlir::Operation *> symbols;
  static inline thread_local ModuleSymbolMemo *active = nullptr;
};

// `module.lookupSymbol(name)`, answered from the installed memo when there
// is one.
inline mlir::Operation *lookupModuleSymbol(mlir::ModuleOp module,
                                           llvm::StringRef name) {
  if (ModuleSymbolMemo *memo = ModuleSymbolMemo::of(module))
    return memo->lookup(mlir::StringAttr::get(module.getContext(), name));
  return module.lookupSymbol(name);
}

inline mlir::Operation *lookupModuleSymbol(mlir::ModuleOp module,
                                           mlir::StringAttr name) {
  if (ModuleSymbolMemo *memo = ModuleSymbolMemo::of(module))
    return memo->lookup(name);
  return module.lookupSymbol(name);
}

template <typename OpT, typename Name>
OpT lookupModuleSymbol(mlir::ModuleOp module, Name &&name) {
  return mlir::dyn_cast_or_null<OpT>(
      lookupModuleSymbol(module, std::forward<Name>(name)));
}

// Records a symbol just made, so the next ask does not scan for it.
inline void rememberModuleSymbol(mlir::ModuleOp module,
                                 mlir::Operation *symbol) {
  if (ModuleSymbolMemo *memo = ModuleSymbolMemo::of(module))
    memo->remember(symbol);
}

// To be called before `op` is erased from `module`.
inline void forgetModuleSymbol(mlir::ModuleOp module, mlir::Operation *op) {
  if (ModuleSymbolMemo *memo = ModuleSymbolMemo::of(module))
    memo->forget(op);
}

} // namespace py

#endif // LYTHON_LOWERING_COMMON_MODULESYMBOLMEMO_H
