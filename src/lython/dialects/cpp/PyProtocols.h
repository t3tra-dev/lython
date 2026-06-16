#pragma once

#include "PyDialectTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace py::protocols {

// One base protocol instantiation in the typing manifest.
struct ProtocolBase {
  std::string name;
  std::vector<mlir::Type> arguments;
};

// One required method contract. The signature's type variable occurrences are
// spelled !py.class<"$T"> in runtime/typing.mlir.
struct ProtocolMethod {
  py::CallableType signature;
  bool mayThrow = false;
  bool noThrow = false;
};

// Optional annotation-only short form for protocols whose public generic
// spelling maps a compact argument list onto non-leading parameters.
struct ProtocolShortForm {
  std::vector<unsigned> positions;
  std::vector<mlir::Type> defaults;
};

// Internal representation for one typing manifest entry.
//
// Protocol contracts are py.class declarations marked with
// `ly.typing.protocol` and rooted at @Protocol. Concrete builtin declarations
// (list/tuple/str/dict/range) are not protocols themselves; they bind concrete
// dialect types to protocol base instantiations.
struct ProtocolInfo {
  std::vector<std::string> params;
  std::vector<std::string> paramVariance;
  std::vector<mlir::Type> paramDefaults;
  std::vector<ProtocolShortForm> shortForms;
  std::vector<ProtocolBase> bases;
  std::map<std::string, std::vector<ProtocolMethod>> methods;
  bool isProtocol = false;
  bool isAbstract = false;
  bool isFinal = false;
};

// Manifest-backed binding proof for a receiver type. This is intentionally
// small: consumers that need method contracts should ask Table with this
// receiver instead of copying the binding logic.
struct ProtocolEvidence {
  std::string manifestClass;
  std::map<std::string, mlir::Type> binding;
  const ProtocolInfo *info = nullptr;
};

// The loaded manifest. The dialect (TableGen) keeps only value-level runtime
// representations; the Protocol tower
// (Iterable/Container/Sized/Iterator/Reversible/Collection/Sequence/range) is
// constructed in the manifest and expanded here through base substitution.
class Table {
public:
  // Loads the embedded typing manifest once per MLIRContext (types are
  // owned by the context). The manifest bytecode is compiled into the
  // binary at build time.
  static const Table &get(mlir::MLIRContext &context);

  // Resolves a method on a concrete dialect type or an already-erased
  // !py.protocol receiver by binding it to its manifest class and walking the
  // instantiated bases. The returned signature has all type variables
  // substituted.
  std::optional<py::CallableType> methodOn(mlir::Type receiverType,
                                           llvm::StringRef methodName) const;
  std::vector<py::CallableType>
  methodOverloadsOn(mlir::Type receiverType, llvm::StringRef methodName) const;
  std::vector<ProtocolMethod>
  methodContractsOn(mlir::Type receiverType, llvm::StringRef methodName) const;
  std::optional<ProtocolMethod>
  resolveMethodContractOn(mlir::Type receiverType, llvm::StringRef methodName,
                          llvm::ArrayRef<mlir::Type> argumentTypes) const;
  std::optional<py::CallableType>
  resolveMethodOn(mlir::Type receiverType, llvm::StringRef methodName,
                  llvm::ArrayRef<mlir::Type> argumentTypes) const;
  std::optional<mlir::Type>
  resolveMethodResultOn(mlir::Type receiverType, llvm::StringRef methodName,
                        llvm::ArrayRef<mlir::Type> argumentTypes) const;
  std::optional<std::vector<mlir::Type>>
  protocolArgumentsFor(mlir::Type receiverType,
                       llvm::StringRef protocolName) const;
  std::optional<std::vector<mlir::Type>>
  completeProtocolArguments(llvm::StringRef protocolName,
                            llvm::ArrayRef<mlir::Type> supplied) const;
  std::optional<ProtocolEvidence> evidenceFor(mlir::Type receiverType) const;
  mlir::Type awaitablePayloadType(mlir::Type type) const;
  bool conformsTo(mlir::Type receiverType, py::ProtocolType protocol) const;

  const ProtocolInfo *lookup(llvm::StringRef name) const;
  bool isProtocol(llvm::StringRef name) const;
  bool loaded() const { return !classes.empty(); }

private:
  void
  collectMethodContractsIn(llvm::StringRef className,
                           const std::map<std::string, mlir::Type> &binding,
                           llvm::StringRef methodName, unsigned depth,
                           std::vector<ProtocolMethod> &out) const;

  std::map<std::string, ProtocolInfo> classes;
};

} // namespace py::protocols
