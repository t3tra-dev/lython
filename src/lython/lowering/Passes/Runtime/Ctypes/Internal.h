#pragma once

#include "Runtime/Core/Lowerer.h"

#include "Native.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace py::lowering::ctypes {

struct CtypesLayout {
  enum class ABIKind {
    SignedInteger,
    UnsignedInteger,
    Floating,
    Pointer,
    Aggregate
  };

  std::uint64_t size = 0;
  std::uint64_t align = 0;
  ABIKind kind = ABIKind::SignedInteger;
};

struct CtypesFieldLayout {
  std::string name;
  std::string contract;
  mlir::Type type;
  CtypesLayout layout;
  std::uint64_t offset = 0;
};

struct CtypesAggregateLayout {
  CtypesLayout layout;
  llvm::SmallVector<CtypesFieldLayout, 8> fields;
  bool isUnion = false;
};

struct CtypesArrayType {
  mlir::Type elementType;
  std::string elementContract;
  CtypesLayout elementLayout;
  CtypesLayout layout;
  std::uint64_t count = 0;
};

using TargetPlatformFacts = py::native::TargetPlatformFacts;

llvm::StringRef stripCtypesModule(llvm::StringRef contract);
std::optional<std::string> ctypesModuleAttrContract(mlir::MLIRContext &context,
                                                    llvm::StringRef moduleName,
                                                    llvm::StringRef attr);
std::optional<std::string> ctypesBareNameContract(mlir::MLIRContext &context,
                                                  llvm::StringRef name);
std::optional<std::string>
ctypesQualifiedNameContract(mlir::MLIRContext &context, llvm::StringRef name);
bool isStaticCtypesFunctionName(mlir::MLIRContext &context,
                                llvm::StringRef name);
mlir::Type ctypesContractType(mlir::MLIRContext *context,
                              llvm::StringRef contract);
RuntimeBundle makeCtypesModuleBundle(mlir::Type resultType,
                                     llvm::StringRef moduleName);
std::optional<llvm::StringRef> ctypesFromAddressTarget(llvm::StringRef binding);
std::optional<llvm::StringRef> ctypesFromBufferTarget(llvm::StringRef binding);
std::optional<llvm::StringRef>
ctypesFromBufferCopyTarget(llvm::StringRef binding);

std::optional<TargetPlatformFacts> targetPlatformFacts(mlir::ModuleOp module);
std::optional<CtypesLayout>
ctypesLayout(llvm::StringRef contract,
             const std::optional<TargetPlatformFacts> &facts);
py::ClassOp lookupClassForContract(mlir::ModuleOp module,
                                   llvm::StringRef contract);
std::optional<std::string> ctypesContractName(mlir::Type type);
std::optional<std::string> ctypesAggregateKind(mlir::ModuleOp module,
                                               py::ClassOp classOp);
std::optional<CtypesAggregateLayout>
ctypesAggregateLayout(mlir::ModuleOp module, py::ClassOp classOp,
                      const std::optional<TargetPlatformFacts> &facts,
                      unsigned depth);
std::optional<CtypesLayout>
ctypesStaticLayout(mlir::ModuleOp module, llvm::StringRef contract,
                   const std::optional<TargetPlatformFacts> &facts,
                   unsigned depth = 0);
std::optional<CtypesLayout>
ctypesStaticLayoutForType(mlir::ModuleOp module, mlir::Type type,
                          const std::optional<TargetPlatformFacts> &facts,
                          unsigned depth = 0);
std::optional<CtypesArrayType>
ctypesArrayType(mlir::ModuleOp module, mlir::Type type,
                const std::optional<TargetPlatformFacts> &facts,
                unsigned depth = 0);
std::string targetFactsLabel(const std::optional<TargetPlatformFacts> &facts);
bool isFixedOrTargetDependentCtypesScalar(llvm::StringRef contract);
bool isCtypesIntegralLike(llvm::StringRef contract);
bool isCtypesVoidPointer(llvm::StringRef contract);
bool isCtypesPointerContract(llvm::StringRef contract);
bool isNoneBundle(const RuntimeBundle &bundle);
bool isStaticSequenceBundle(const RuntimeBundle &bundle);
std::string ctypesContractFromBundle(const RuntimeBundle &bundle);
mlir::Type ctypesTypeFromBundle(const RuntimeBundle &bundle);
std::optional<std::string> ctypesTypeObjectName(const RuntimeBundle &bundle);

void keepAliveSource(RuntimeCtypesEvidence &evidence,
                     const RuntimeBundle &source);
void keepAliveBufferSource(RuntimeBufferEvidence &evidence,
                           const RuntimeBundle &source);
mlir::Value cdataStorageAddress(const RuntimeCtypesEvidence &evidence);
mlir::Value cdataStorageAddressValid(const RuntimeCtypesEvidence &evidence);
mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                       bool value);
mlir::Value constantI64(mlir::OpBuilder &builder, mlir::Location loc,
                        std::int64_t value);
mlir::Value constantIndex(mlir::OpBuilder &builder, mlir::Location loc,
                          std::int64_t value);
std::string ctypesLibraryABI(llvm::StringRef contract);
bool isKnownTrue(mlir::Value value);
bool isIntegerScalarLayout(const CtypesLayout &layout);
bool isFloatingScalarLayout(const CtypesLayout &layout);
bool isPointerScalarLayout(const CtypesLayout &layout);
llvm::StringRef
ctypesProvenanceName(RuntimeCtypesEvidence::Provenance provenance);
llvm::StringRef ctypesLifetimeName(RuntimeCtypesEvidence::Lifetime lifetime);
std::optional<std::int64_t> knownI64Constant(mlir::Value value);
bool fitsStaticIntegerLayout(std::int64_t value, const CtypesLayout &layout);
mlir::IntegerType nativeIntegerType(mlir::Builder &builder,
                                    const CtypesLayout &layout);
mlir::IntegerType
nativePointerIntegerType(mlir::Builder &builder,
                         const std::optional<TargetPlatformFacts> &facts);
mlir::LLVM::LLVMPointerType nativePointerType(mlir::MLIRContext *context);
mlir::Value coerceNativeInteger(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value,
                                mlir::IntegerType targetType);
std::optional<mlir::Value> extractNativeIntegerArgument(
    mlir::Operation *op, mlir::OpBuilder &builder, const RuntimeBundle &source,
    llvm::StringRef expectedContract, const CtypesLayout &layout,
    const std::optional<TargetPlatformFacts> &facts);
mlir::Value
integerToNativePointer(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value value,
                       const std::optional<TargetPlatformFacts> &facts);
mlir::Value nativePointerToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value pointer);
mlir::Value addressWithOffset(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value address, std::int64_t offset,
                              const std::optional<TargetPlatformFacts> &facts);
mlir::Value
nativePointerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value address,
                         const std::optional<TargetPlatformFacts> &facts);
mlir::Value
addressOfNativeCellAlloca(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value nativeValue,
                          const std::optional<TargetPlatformFacts> &facts);
mlir::Value addressOfZeroedNativeBytesAlloca(
    mlir::OpBuilder &builder, mlir::Location loc, std::uint64_t size,
    const std::optional<TargetPlatformFacts> &facts);
mlir::Value
loadNativeIntegerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value address, mlir::IntegerType nativeType,
                             const std::optional<TargetPlatformFacts> &facts);
void storeNativeIntegerToAddress(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value address,
    mlir::Value value, mlir::IntegerType nativeType,
    const std::optional<TargetPlatformFacts> &facts);
mlir::Value widenNativeInteger(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value value, const CtypesLayout &layout);
void copyNativeBytes(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value destinationAddress, mlir::Value sourceAddress,
                     std::uint64_t byteCount,
                     const std::optional<TargetPlatformFacts> &facts);
mlir::LogicalResult storeCtypesValueToAddress(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Value destinationAddress, mlir::Type expectedType,
    llvm::StringRef expectedContract, const CtypesLayout &layout,
    const RuntimeBundle &source,
    const std::optional<TargetPlatformFacts> &facts);
RuntimeBufferEvidence
makeCtypesBufferEvidence(mlir::OpBuilder &builder, mlir::Location loc,
                         const CtypesLayout &layout, mlir::Value address,
                         mlir::Value valid, const RuntimeBundle &owner,
                         bool writable);
void attachCtypesBufferEvidence(mlir::OpBuilder &builder, mlir::Location loc,
                                RuntimeBundle &bundle,
                                RuntimeCtypesEvidence &ctypes,
                                const CtypesLayout &layout, bool writable);
mlir::FailureOr<RuntimeBundle> materializeCtypesAddressView(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Type ctype, llvm::StringRef ctypeName, const CtypesLayout &layout,
    mlir::Value storageAddress, mlir::Value storageValid,
    const RuntimeBundle &owner);
mlir::FailureOr<RuntimeBundle> materializeCtypesPythonReadResult(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Type ctype, llvm::StringRef ctypeName, const CtypesLayout &layout,
    mlir::Value storageAddress, mlir::Value storageValid,
    const RuntimeBundle &owner);
mlir::FailureOr<RuntimeBundle>
materializeCtypesCell(mlir::Operation *op, mlir::OpBuilder &builder,
                      mlir::ModuleOp module, mlir::Type ctype,
                      llvm::StringRef ctypeName,
                      llvm::ArrayRef<const RuntimeBundle *> sources);
mlir::FailureOr<RuntimeBundle>
materializeCtypesLibrary(mlir::Operation *op, mlir::ModuleOp module,
                         mlir::Type ctype, llvm::StringRef ctypeName,
                         llvm::ArrayRef<const RuntimeBundle *> sources);
std::optional<mlir::Value>
stackPointerForBorrowedScalar(mlir::Operation *op, mlir::OpBuilder &builder,
                              const RuntimeCtypesEvidence &evidence,
                              const std::optional<TargetPlatformFacts> &facts);
std::optional<mlir::Value>
extractNativePointerArgument(mlir::Operation *op, mlir::OpBuilder &builder,
                             const RuntimeBundle &source,
                             const std::optional<TargetPlatformFacts> &facts);
std::optional<mlir::Value>
extractPointerAddressInteger(mlir::Operation *op, mlir::OpBuilder &builder,
                             const RuntimeBundle &source,
                             const std::optional<TargetPlatformFacts> &facts);
mlir::FailureOr<mlir::func::FuncOp> getOrCreateNativeDeclaration(
    mlir::Operation *op, mlir::ModuleOp module, mlir::OpBuilder &builder,
    llvm::StringRef name, mlir::FunctionType type,
    llvm::ArrayRef<std::string> argTypes, llvm::StringRef resultType,
    llvm::StringRef abi, bool processLibrary, const TargetPlatformFacts &facts);
std::string describeNativeArgumentSource(const RuntimeBundle &source);

} // namespace py::lowering::ctypes
