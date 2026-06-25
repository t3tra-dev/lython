#pragma once

#include "PyDialectTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
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

bool sameMethodContract(const ProtocolMethod &lhs, const ProtocolMethod &rhs);

std::optional<std::int64_t> normalizeFiniteTupleIndex(std::int64_t index,
                                                      std::size_t size);

// Returns the callable view used for a bound method by dropping the receiver
// parameter while preserving parameter metadata.
py::CallableType bindReceiverCallable(py::CallableType signature);

struct KeywordArgument {
  std::string name;
  mlir::Type type;
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
  std::map<std::string, mlir::Type> fields;
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

// A selected method contract plus the evidence used to select and specialize
// it. This is the manifest-level proof object that higher layers should carry
// when they need to lower a protocol operation without recomputing the same
// facts.
struct ContractResolution {
  ProtocolMethod method;
  std::string methodName;
  std::map<std::string, mlir::Type> typeBindings;
  std::optional<ProtocolEvidence> receiverEvidence;
  int score = 0;
};

// A selected field contract plus the binding/evidence used to specialize it.
// This mirrors ContractResolution for attributes so callers do not have to
// recompute or discard manifest binding facts when resolving fields.
struct FieldResolution {
  mlir::Type contractType;
  std::string fieldName;
  std::map<std::string, mlir::Type> typeBindings;
  std::optional<ProtocolEvidence> receiverEvidence;
};

// A selected awaitable contract. Native async.value<T> has no manifest method
// contract but still has a payload; protocol-backed awaitables carry the
// __await__ contract used to derive that payload so inference/lowering can keep
// the same evidence.
struct AwaitableResolution {
  mlir::Type payloadType;
  std::optional<ContractResolution> awaitContract;
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
  static Table &getMutable(mlir::MLIRContext &context);

  void registerClass(llvm::StringRef name, ProtocolInfo info);

  // Enumerates manifest method candidates on a concrete dialect type or an
  // already-erased !py.protocol receiver. Each candidate carries the receiver
  // binding/evidence used to specialize its signature; call-site selection is
  // performed by the caller's Callable matcher.
  std::vector<ContractResolution>
  methodContractCandidatesWithEvidence(mlir::Type receiverType,
                                       llvm::StringRef methodName) const;
  std::optional<FieldResolution>
  resolveFieldContractWithEvidence(mlir::Type receiverType,
                                   llvm::StringRef fieldName) const;
  std::optional<std::vector<mlir::Type>>
  protocolArgumentsFor(mlir::Type receiverType,
                       llvm::StringRef protocolName) const;
  std::optional<std::vector<mlir::Type>>
  completeProtocolArguments(llvm::StringRef protocolName,
                            llvm::ArrayRef<mlir::Type> supplied) const;
  std::optional<ProtocolEvidence> evidenceFor(mlir::Type receiverType) const;
  std::optional<AwaitableResolution>
  resolveAwaitableWithEvidence(mlir::Type type) const;
  mlir::Type awaitablePayloadType(mlir::Type type) const;

  const ProtocolInfo *lookup(llvm::StringRef name) const;
  bool isProtocol(llvm::StringRef name) const;
  bool loaded() const { return !classes.empty(); }

private:
  std::optional<FieldResolution>
  collectFieldResolutionIn(llvm::StringRef className,
                           const std::map<std::string, mlir::Type> &binding,
                           llvm::StringRef fieldName, unsigned depth) const;
  std::vector<ProtocolMethod>
  collectReceiverMethodContracts(mlir::Type receiverType,
                                 llvm::StringRef methodName) const;
  bool
  collectMethodContractsIn(llvm::StringRef className,
                           const std::map<std::string, mlir::Type> &binding,
                           llvm::StringRef methodName, unsigned depth,
                           std::vector<ProtocolMethod> &out) const;
  std::map<std::string, ProtocolInfo> classes;
};

} // namespace py::protocols
