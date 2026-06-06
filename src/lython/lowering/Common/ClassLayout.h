#pragma once

#include "PyDialectTypes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <utility>

namespace py {

class PyLLVMTypeConverter;

namespace class_layout {

struct Layout;

namespace Object {
static constexpr int64_t kHeaderIndex = 0;
static constexpr int64_t kPayloadIndex = 1;
int64_t partCount(mlir::LLVM::LLVMStructType objectType);
mlir::Type descriptorType(mlir::LLVM::LLVMStructType objectType,
                          int64_t partIndex);
mlir::Type headerDescriptorType(mlir::LLVM::LLVMStructType objectType);
mlir::Value descriptor(mlir::Location loc,
                       mlir::LLVM::LLVMStructType objectType,
                       mlir::Value object, int64_t partIndex,
                       mlir::OpBuilder &builder);
mlir::Value headerDescriptor(mlir::Location loc,
                             mlir::LLVM::LLVMStructType objectType,
                             mlir::Value object, mlir::OpBuilder &builder);
mlir::Value fromDescriptors(mlir::Location loc,
                            mlir::LLVM::LLVMStructType objectType,
                            mlir::ValueRange descriptors,
                            mlir::OpBuilder &builder);
} // namespace Object

namespace Payload {
inline int64_t lockIndex(int64_t fieldCount) { return fieldCount; }
mlir::MemRefType lockMemRefType(mlir::MLIRContext *ctx);
mlir::MemRefType fieldMemRefType(mlir::Type fieldStorageType,
                                 mlir::MLIRContext *ctx);
} // namespace Payload

namespace Header {
static constexpr int64_t kRefcountSlot = 0;
static constexpr int64_t kLayoutIdSlot = 1;

mlir::MemRefType memrefType(mlir::MLIRContext *ctx);
} // namespace Header

namespace DescriptorShape {
void mark(mlir::Operation *op, mlir::MemRefType memrefType);
void copy(mlir::Operation *source, mlir::Operation *target);
bool has(mlir::Operation *op);
bool matches(mlir::Operation *op, mlir::MemRefType memrefType);
mlir::LogicalResult verify(mlir::Operation *op, mlir::MemRefType memrefType,
                           llvm::StringRef what);
} // namespace DescriptorShape

struct FieldInfo {
  mlir::StringAttr name;
  mlir::Type logicalType;
  mlir::Type storageType;
  llvm::SmallVector<mlir::Type, 4> storageParts;
  int64_t payloadPartStart = 0;
  int64_t payloadPartCount = 0;
};

struct Layout {
  mlir::MemRefType headerType;
  mlir::LLVM::LLVMStructType storageType;
  llvm::SmallVector<mlir::MemRefType, 4> payloadPartTypes;
  mlir::LLVM::LLVMStructType objectType;
  llvm::SmallVector<FieldInfo, 8> fields;
};

namespace Payload {
llvm::ArrayRef<mlir::MemRefType> partTypes(const Layout &layout);
int64_t partCount(const Layout &layout);
mlir::MemRefType partType(const Layout &layout, int64_t partIndex);
int64_t fieldPartIndex(const Layout &layout, int64_t fieldIndex);
int64_t fieldPartCount(const Layout &layout, int64_t fieldIndex);
int64_t lockPartIndex(const Layout &layout);
mlir::MemRefType fieldPartType(const Layout &layout, int64_t fieldIndex);
mlir::LLVM::LLVMStructType storageType(const Layout &layout);
bool isStorageValue(const Layout &layout, mlir::Value value);
mlir::Type fieldStorageType(const Layout &layout, int64_t fieldIndex);
mlir::Type lockStorageType(const Layout &layout);
mlir::Value zeroStorage(mlir::Location loc, const Layout &layout,
                        mlir::OpBuilder &builder);
mlir::Value extractField(mlir::Location loc, const Layout &layout,
                         mlir::Value storage, int64_t fieldIndex,
                         mlir::OpBuilder &builder);
mlir::Value insertField(mlir::Location loc, const Layout &layout,
                        mlir::Value storage, int64_t fieldIndex,
                        mlir::Value fieldValue, mlir::OpBuilder &builder);
mlir::Value composeField(mlir::Location loc, const Layout &layout,
                         int64_t fieldIndex, mlir::ValueRange parts,
                         mlir::OpBuilder &builder);
mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
decomposeField(mlir::Location loc, const Layout &layout, int64_t fieldIndex,
               mlir::Value fieldValue, mlir::OpBuilder &builder);
} // namespace Payload

mlir::Type descriptorStorageType(mlir::MemRefType memrefType,
                                 mlir::MLIRContext *ctx);
bool isDescriptorStorageType(mlir::Type type);
bool isObjectCarrierType(mlir::Type type);
bool isObjectCarrierMemRefType(mlir::Type type);
mlir::LLVM::LLVMStructType
objectCarrierType(mlir::MLIRContext *ctx,
                  llvm::ArrayRef<mlir::MemRefType> partTypes);
mlir::LLVM::LLVMStructType objectCarrierType(mlir::Type type);
mlir::MemRefType carrierType(mlir::LLVM::LLVMStructType objectType,
                             mlir::MLIRContext *ctx);
void partsValueTypes(const Layout &layout,
                     llvm::SmallVectorImpl<mlir::Type> &types);
mlir::FailureOr<Layout> get(mlir::Operation *from, ClassType classType,
                            const PyLLVMTypeConverter &typeConverter);
mlir::Type carrierStorageType(mlir::ModuleOp module, ClassType classType,
                              const PyLLVMTypeConverter &typeConverter,
                              mlir::MLIRContext *ctx);
mlir::FailureOr<std::pair<unsigned, FieldInfo>>
lookupField(mlir::Operation *from, ClassType classType,
            const PyLLVMTypeConverter &typeConverter,
            llvm::StringRef fieldName);

} // namespace class_layout
} // namespace py
