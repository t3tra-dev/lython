#pragma once

#include "Ast.h"
#include "PrimitiveTypes.h"
#include "TypeSystem.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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
                            AlgorithmM &types);
bool anyTrue(llvm::ArrayRef<char> values);
std::string methodKind(const parser::Node &function);
mlir::Type elementType(mlir::Type type, AlgorithmM &types);
void appendStarredArgumentTypes(mlir::Type type, AlgorithmM &types,
                                llvm::SmallVectorImpl<mlir::Type> &out);
bool isTopLevelDecl(const parser::Node &node);
std::string importBindingName(std::string_view module,
                              std::optional<std::string_view> asname);
mlir::Attribute defaultValueAttr(mlir::Builder &builder,
                                 const parser::Node *node);
mlir::ArrayAttr callableDefaultValues(mlir::Builder &builder,
                                      const parser::Node &function,
                                      const FunctionSignature &sig);
llvm::SmallVector<const parser::Node *, 8>
positionalArgumentNodes(const parser::Node &arguments);

bool blockHasTerminator(mlir::Block &block);
mlir::Operation *blockTerminator(mlir::Block &block);
void setInsertionBeforeTerminator(mlir::OpBuilder &builder, mlir::Block &block);
void ensureYield(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Block &block);
bool insertionBlockTerminated(const mlir::OpBuilder &builder);
bool containsReturnStatement(const std::vector<parser::NodePtr> *statements);

bool containsObjectTop(mlir::Type type, const AlgorithmM &types);
bool isNoneTypeLike(mlir::Type type);
mlir::Type removeNoneFromType(mlir::Type type, AlgorithmM &types);

struct NoneComparisonNarrowing {
  std::string name;
  bool trueBranchIsNone = true;
  mlir::Type payloadType;
};

const parser::Node *nameComparedWithNone(const parser::Node *left,
                                         const parser::Node *right);
std::optional<NoneComparisonNarrowing>
optionalNoneComparison(const parser::Node &test, AlgorithmM &types);
mlir::Type widenInferredLiterals(mlir::Type type, const AlgorithmM &types);
bool hasUnexpectedObjectTop(mlir::Type actual, mlir::Type expected,
                            const AlgorithmM &types);

} // namespace lython::emitter
