#pragma once

#include "PyDialectTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace py::protocols {

struct ManifestSource {
  const char *name = nullptr;
  const char *data = nullptr;
  std::size_t size = 0;
};

llvm::ArrayRef<ManifestSource> manifestSources();

enum class Variance { Covariant, Contravariant, Invariant };

// One base protocol instantiation in the typing manifest.
struct ProtocolBase {
  std::string name;
  std::vector<mlir::Type> arguments;
};

// One required method contract. The signature's type variable occurrences
// are spelled !py.class<"$T"> in the module manifests.
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
// One `ly.typing.field_param_bindings` entry ("field:param:via_base"):
// assigning `type[C]` to `field` binds the class's `param` type parameter to
// the single argument of C's `via_base[X]` base (literal None binds None).
struct FieldParamBinding {
  std::string field;
  std::string param;
  std::string viaBase;
};

struct ProtocolInfo {
  std::vector<std::string> params;
  std::vector<std::string> paramVariance;
  std::vector<mlir::Type> paramDefaults;
  std::vector<ProtocolShortForm> shortForms;
  std::vector<ProtocolBase> bases;
  std::map<std::string, mlir::Type> fields;
  std::map<std::string, std::vector<ProtocolMethod>> methods;
  // Methods declared via `ly.typing.structural_mutators` that structurally
  // mutate the receiver (may reallocate its storage). Calls to these are
  // emitted with an extra receiver-typed result that rebinds the local.
  std::set<std::string> structuralMutators;
  // Ordered attribute names for positional class patterns: the class's
  // `__match_args__` tuple (user classes) or `ly.typing.match_args` (manifest).
  std::vector<std::string> matchArgs;
  std::vector<FieldParamBinding> fieldParamBindings;
  // `ly.typing.fields_spec` ("attr:via_base"): subclasses declare aggregate
  // fields via a class assignment to `attr` (e.g. ctypes' `_fields_`); each
  // field types as its declared class's `via_base[V]` base argument.
  std::string fieldsSpecAttrName;
  std::string fieldsSpecViaBase;
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
  // Loads the embedded typing manifest sources once per MLIRContext (types are
  // owned by the context). The manifest text is compiled into the binary at
  // build time; protocol facts are still derived by parsing MLIR, not by a
  // protocol-specific generator.
  static const Table &get(mlir::MLIRContext &context);
  static Table &getMutable(mlir::MLIRContext &context);

  void registerClass(llvm::StringRef name, ProtocolInfo info);

  std::optional<std::string>
  moduleClassExport(llvm::StringRef moduleName,
                    llvm::StringRef exportedName) const;
  std::optional<std::string>
  qualifiedClassExport(llvm::StringRef qualifiedName) const;
  std::optional<std::string>
  bareClassExport(llvm::StringRef exportedName) const;
  std::vector<std::pair<std::string, std::string>>
  moduleClassExports(llvm::StringRef moduleName) const;
  bool isModuleCallableExport(llvm::StringRef moduleName,
                              llvm::StringRef exportedName) const;
  std::vector<std::string>
  moduleCallableExports(llvm::StringRef moduleName) const;

  // Manifest-declared Callable contract for a free (module-level or builtin)
  // function, keyed by fully-qualified name (e.g. "ctypes.sizeof"). This is the
  // manifest source for module/builtin function signatures so imported and
  // builtin callables need not be typed by a C++ contract table.
  std::optional<mlir::Type>
  freeFunctionContract(llvm::StringRef qualifiedName) const;

  // True when the manifest declares `methodName` as a structural mutator of
  // the receiver's class (`ly.typing.structural_mutators`).
  bool isStructuralMutator(mlir::Type receiverType,
                           llvm::StringRef methodName) const;

  // Ordered `__match_args__` attribute names for positional class patterns on
  // the receiver's class; nullopt when the class declares none.
  std::optional<std::vector<std::string>>
  matchArgsFor(mlir::Type receiverType) const;

  // Manifest float constants (`ly.typing.float_constant_names`/`_values`,
  // e.g. math.pi). Keyed by fully-qualified name; the per-module list drives
  // import binding.
  std::optional<double> moduleFloatConstant(llvm::StringRef qualifiedName) const;
  std::vector<std::string>
  moduleFloatConstantExports(llvm::StringRef moduleName) const;

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
  // Manifest-driven contract refinement on field assignment
  // (ly.typing.field_param_bindings): the refined receiver contract when the
  // assignment binds one of the receiver class's type parameters.
  std::optional<mlir::Type>
  refineContractByFieldAssignment(mlir::Type receiverType,
                                  llvm::StringRef fieldName,
                                  mlir::Type valueType) const;
  // The `ly.typing.fields_spec` ("attr:via_base") declared by `className` or
  // one of its manifest bases; accepts export aliases (e.g.
  // "ctypes.Structure").
  std::optional<std::pair<std::string, std::string>>
  aggregateFieldsSpec(llvm::StringRef className) const;
  // The value type an instance of `instanceType` converts to: the single
  // argument of its `viaBase[V]` manifest base.
  std::optional<mlir::Type> conversionTypeViaBase(mlir::Type instanceType,
                                                  llvm::StringRef viaBase) const;
  std::optional<std::vector<mlir::Type>>
  protocolArgumentsFor(mlir::Type receiverType,
                       llvm::StringRef protocolName) const;
  std::optional<std::vector<mlir::Type>>
  completeProtocolArguments(llvm::StringRef protocolName,
                            llvm::ArrayRef<mlir::Type> supplied) const;
  Variance parameterVariance(llvm::StringRef protocolName,
                             unsigned index) const;
  bool isProtocolSubtypeOf(
      py::ProtocolType subtype, py::ProtocolType supertype,
      llvm::function_ref<bool(mlir::Type, mlir::Type, Variance)>
          argumentMatches) const;
  std::optional<ProtocolEvidence> evidenceFor(mlir::Type receiverType) const;
  bool isManifestSubclassOf(mlir::Type receiverType,
                            llvm::StringRef baseClassName) const;
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
  std::map<std::string, std::map<std::string, std::string>>
      classExportsByModule;
  std::map<std::string, std::vector<std::string>> callableExportsByModule;
  std::map<std::string, mlir::Type> freeFunctionContracts;
  std::map<std::string, double> floatConstants;
  std::map<std::string, std::vector<std::string>> floatConstantsByModule;
};

} // namespace py::protocols
