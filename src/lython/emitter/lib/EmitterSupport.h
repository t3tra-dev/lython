#pragma once

#include "Ast.h"
#include "PrimitiveTypes.h"
#include "TypeSystem.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lython::emitter {

extern const llvm::StringLiteral kCallableVarargValueTypeAttr;
extern const llvm::StringLiteral kCallableKwargValueTypeAttr;
extern const llvm::StringLiteral kPackUnpackedOperandsAttr;

bool isPrimitiveOnlyCallable(py::CallableType callable);

mlir::ArrayAttr stringArray(mlir::Builder &builder,
                            llvm::ArrayRef<std::string> values);
mlir::ArrayAttr stringArray(mlir::Builder &builder,
                            llvm::ArrayRef<llvm::StringRef> values);
mlir::ArrayAttr typeArray(mlir::Builder &builder,
                          llvm::ArrayRef<mlir::Type> values);
mlir::ArrayAttr boolArray(mlir::Builder &builder, llvm::ArrayRef<char> values);

mlir::Type replaceSelfType(mlir::Type type, mlir::Type selfType);
void replaceSelfInSignature(FunctionSignature &sig, mlir::Type selfType,
                            TypeSystem &types);
bool anyTrue(llvm::ArrayRef<char> values);
std::string methodKind(const parser::Node &function);
bool isTopLevelDecl(const parser::Node &node);
std::string importBindingName(std::string_view module,
                              std::optional<std::string_view> asname);
mlir::Attribute defaultValueAttr(mlir::Builder &builder,
                                 const parser::Node *node);
llvm::SmallVector<const parser::Node *, 8>
positionalArgumentNodes(const parser::Node &arguments);

bool blockHasTerminator(mlir::Block &block);
mlir::Operation *blockTerminator(mlir::Block &block);
void setInsertionBeforeTerminator(mlir::OpBuilder &builder, mlir::Block &block);
bool insertionBlockTerminated(const mlir::OpBuilder &builder);
// Statements of one of `kinds` anywhere in the subtree. A nested function,
// lambda or class always ends the search; `stopAtLoops` also ends it at a
// nested loop, which is what a break/continue question wants.
bool containsStatementKind(const parser::Node *node,
                           llvm::ArrayRef<llvm::StringRef> kinds,
                           bool stopAtLoops);
bool containsStatementKind(const std::vector<parser::NodePtr> *statements,
                           llvm::ArrayRef<llvm::StringRef> kinds,
                           bool stopAtLoops);
bool containsReturnStatement(const std::vector<parser::NodePtr> *statements);
bool containsBreakOrContinueStatement(
    const std::vector<parser::NodePtr> *statements);
// `continue` only. The break edge and the continue edge leave a try in
// different directions, and one of them scales where the other does not --
// see the nested-loop guard in EmitterExceptions.cpp.
bool containsContinueStatement(const std::vector<parser::NodePtr> *statements);
void collectNameBindings(const parser::Node *node, llvm::StringSet<> &names,
                         bool bindsNestedDefinitions);
void collectAssignedNameTargets(const parser::Node *node,
                                llvm::StringSet<> &names);
void collectAssignedNames(const parser::Node *node, llvm::StringSet<> &names);
void collectAssignedNames(const std::vector<parser::NodePtr> *statements,
                          llvm::StringSet<> &names);
bool derivesViaStructuralMutation(mlir::Value current, mlir::Value previous);

bool containsObjectTop(mlir::Type type, const TypeSystem &types);
bool isNoneTypeLike(mlir::Type type);
// Does this subtree READ `name`? A store is not a read: `x[i] = v` reads `x`,
// `x = v` does not. Shared by the two forward scans that ask it -- the slot a
// region-bound name gets, and the union a `None` seed takes.
bool containsNameLoad(const parser::Node *node, llvm::StringRef name);
mlir::Type removeNoneFromType(mlir::Type type, TypeSystem &types);

struct NoneComparisonNarrowing {
  std::string name;
  bool trueBranchIsNone = true;
  mlir::Type payloadType;
};

struct BranchTypeNarrowing {
  // A local's name, or -- when `isMemberPath` -- the dotted path of a field
  // read (`self.left`). The two are spent differently: a local is one value
  // the branch holds, a field is re-read at every use.
  std::string name;
  bool isMemberPath = false;
  mlir::Type trueType;
  mlir::Type falseType;
  mlir::Type trueSourceType;
  mlir::Type falseSourceType;
};

struct IsInstanceAnalysis {
  enum class Kind {
    AlwaysTrue,
    AlwaysFalse,
    UnionTest,
    UnionClassTest,
    ClassTest,
    Unsupported
  };

  mlir::Type sourceType;
  mlir::Type targetType;
  Kind kind = Kind::Unsupported;
  llvm::SmallVector<mlir::Type, 4> unionMembers;
  // ClassTest only, and only when the source is `object`: every class the
  // taxonomy says answers YES, because the class id the test compares is the
  // EXACT class and a subclass carries its own. Empty means the target's own
  // id is the whole answer, which the lowering derives on its own.
  llvm::SmallVector<mlir::Type, 4> classTestTypes;
  mlir::Type trueType;
  mlir::Type falseType;
  std::string failureReason;
};

const parser::Node *nameComparedWithNone(const parser::Node *left,
                                         const parser::Node *right);
std::optional<NoneComparisonNarrowing>
optionalNoneComparison(const parser::Node &test, TypeSystem &types);
// A body that is only `...`: the stub spelling Python uses for a declaration
// it does not mean to run (`def area(self) -> int: ...`). Both the function
// path and the method-INLINING path have to recognise it, and asking twice is
// what left the inlined one refusing what the other had just learned to accept.
bool isEllipsisStubBody(const std::vector<parser::NodePtr> *body);
// Every fact a test proves, one per NAME. A conjunction may prove things about
// several names at once (`isinstance(a, X) and isinstance(b, Y)`), and a single
// fact cannot carry them -- the body then narrowed only the first.
llvm::SmallVector<BranchTypeNarrowing, 2>
branchTypeNarrowings(const parser::Node &test, TypeSystem &types,
                     mlir::Operation *from);
std::optional<BranchTypeNarrowing>
optionalBranchTypeNarrowing(const parser::Node &test, TypeSystem &types,
                            mlir::Operation *from);
std::optional<bool> optionalStaticBranchTruth(const parser::Node &test,
                                              TypeSystem &types,
                                              mlir::Operation *from);
std::optional<mlir::Type> isinstanceTargetType(const parser::Node *node,
                                               TypeSystem &types);
// The same question for `isinstance(x, (A, B))`: one class, or a tuple of them.
std::optional<llvm::SmallVector<mlir::Type, 4>>
isinstanceTargetTypes(const parser::Node *node, TypeSystem &types);
// A contract this program declares (`class Err`), as opposed to one a manifest
// declares (`builtins.BaseException`). The two are told apart by the dot: only
// a manifest contract is module-qualified.
bool isSourceDefinedContract(mlir::Type type);

// The subclass relation the module PRE-PASS recorded, for two source classes
// whose class ops the subtype walk may not have created yet.
bool declaredSubclassOfType(mlir::Type sub, mlir::Type super,
                            TypeSystem &types);

bool isAssignableWithStaticEvidence(mlir::Type actual, mlir::Type expected,
                                    mlir::Operation *from);
// Does Python's class hierarchy say `sub` is a subclass of `super`? Wider than
// assignability by exactly one rung (bool < int); see the definition.
bool pythonSubclassOf(mlir::Type sub, mlir::Type super, TypeSystem &types,
                      mlir::Operation *from);

IsInstanceAnalysis analyzeIsInstance(mlir::Type sourceType,
                                     mlir::Type targetType, TypeSystem &types,
                                     mlir::Operation *from);
// "is it any of these": the tuple form, merged into one answer.
IsInstanceAnalysis analyzeIsInstanceAny(mlir::Type sourceType,
                                        llvm::ArrayRef<mlir::Type> targetTypes,
                                        TypeSystem &types,
                                        mlir::Operation *from);
mlir::Type widenInferredLiterals(mlir::Type type, const TypeSystem &types);
bool hasUnexpectedObjectTop(mlir::Type actual, mlir::Type expected,
                            const TypeSystem &types);

} // namespace lython::emitter
