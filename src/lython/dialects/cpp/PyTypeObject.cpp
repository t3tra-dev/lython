#include "cpp/PyTypeObject.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py::type_object {
namespace {

mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 4>>
basesOf(mlir::Operation *diagnostic, ClassOp op) {
  llvm::SmallVector<llvm::StringRef, 4> bases;
  mlir::ArrayAttr attrs = op.getBaseNamesAttr();
  if (!attrs)
    return bases;
  for (mlir::Attribute attr : attrs) {
    auto name = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!name) {
      diagnostic->emitOpError("base_names must contain only StringAttr values");
      return mlir::failure();
    }
    bases.push_back(name.getValue());
  }
  return bases;
}

mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>>
c3Merge(llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 8>, 8> sequences,
        mlir::Operation *diagnostic, llvm::StringRef className) {
  llvm::SmallVector<llvm::StringRef, 8> result;
  auto compact = [&]() {
    llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 8>, 8> next;
    for (auto &sequence : sequences)
      if (!sequence.empty())
        next.push_back(std::move(sequence));
    sequences = std::move(next);
  };
  compact();

  while (!sequences.empty()) {
    std::optional<llvm::StringRef> candidate;
    for (const auto &sequence : sequences) {
      llvm::StringRef head = sequence.front();
      bool appearsInTail = false;
      for (const auto &other : sequences) {
        if (llvm::is_contained(
                llvm::ArrayRef<llvm::StringRef>(other).drop_front(), head)) {
          appearsInTail = true;
          break;
        }
      }
      if (!appearsInTail) {
        candidate = head;
        break;
      }
    }
    if (!candidate) {
      diagnostic->emitOpError("inconsistent C3 MRO for class '")
          << className << "'";
      return mlir::failure();
    }

    result.push_back(*candidate);
    for (auto &sequence : sequences)
      if (!sequence.empty() && sequence.front() == *candidate)
        sequence.erase(sequence.begin());
    compact();
  }
  return result;
}

mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>>
computeMro(mlir::Operation *from, llvm::StringRef name,
           llvm::StringSet<> &visiting) {
  if (!visiting.insert(name).second) {
    from->emitOpError("class inheritance cycle through '") << name << "'";
    return mlir::failure();
  }
  ClassOp classOp = lookup(from, name);
  if (!classOp) {
    from->emitOpError("unknown class '") << name << "'";
    return mlir::failure();
  }
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 4>> bases =
      basesOf(from, classOp);
  if (mlir::failed(bases))
    return mlir::failure();

  llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 8>, 8> sequences;
  for (llvm::StringRef baseName : *bases) {
    mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>> baseMro =
        computeMro(from, baseName, visiting);
    if (mlir::failed(baseMro))
      return mlir::failure();
    sequences.push_back(std::move(*baseMro));
  }
  sequences.emplace_back(bases->begin(), bases->end());
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>> tail =
      c3Merge(std::move(sequences), from, name);
  if (mlir::failed(tail))
    return mlir::failure();

  visiting.erase(name);
  llvm::SmallVector<llvm::StringRef, 8> result{name};
  result.append(tail->begin(), tail->end());
  return result;
}

} // namespace

ClassOp lookup(mlir::Operation *from, llvm::StringRef name) {
  if (!from || name.empty())
    return nullptr;
  mlir::StringAttr symbolName = mlir::StringAttr::get(from->getContext(), name);
  for (mlir::Operation *table = from; table; table = table->getParentOp()) {
    if (!table->hasTrait<mlir::OpTrait::SymbolTable>())
      continue;
    mlir::Operation *symbol =
        mlir::SymbolTable::lookupSymbolIn(table, symbolName);
    if (auto classOp = mlir::dyn_cast_or_null<ClassOp>(symbol))
      return classOp;
  }
  return nullptr;
}

mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>>
mroNames(mlir::Operation *from, llvm::StringRef name) {
  llvm::StringSet<> visiting;
  return computeMro(from, name, visiting);
}

mlir::LogicalResult verifyBases(ClassOp op) {
  llvm::StringRef className = op.getSymNameAttr().getValue();
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 4>> bases =
      basesOf(op.getOperation(), op);
  if (mlir::failed(bases))
    return mlir::failure();

  llvm::StringSet<> seen;
  for (llvm::StringRef baseName : *bases) {
    if (baseName == className)
      return op.emitOpError("class cannot inherit from itself");
    if (!seen.insert(baseName).second)
      return op.emitOpError("duplicate base class '") << baseName << "'";
    if (!lookup(op, baseName))
      return op.emitOpError("unknown base class '") << baseName << "'";
  }

  if (className == kBaseException && !bases->empty())
    return op.emitOpError("BaseException must be the exception root");
  if (className == kException) {
    if (bases->size() != 1 || bases->front() != kBaseException)
      return op.emitOpError("Exception must inherit from BaseException");
  }

  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>> mro =
      mroNames(op, className);
  return mlir::failed(mro) ? mlir::failure() : mlir::success();
}

mlir::FailureOr<bool> isSubclassOf(mlir::Operation *from,
                                   llvm::StringRef derived,
                                   llvm::StringRef base) {
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>> mro =
      mroNames(from, derived);
  if (mlir::failed(mro))
    return mlir::failure();
  if (!lookup(from, base)) {
    from->emitOpError("unknown class '") << base << "'";
    return mlir::failure();
  }
  return llvm::is_contained(*mro, base);
}

mlir::FailureOr<bool> exceptionMatches(mlir::Operation *from,
                                       llvm::StringRef handler) {
  return isSubclassOf(from, kException, handler);
}

} // namespace py::type_object
