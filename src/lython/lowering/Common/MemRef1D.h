#pragma once

// Where a rank-1 memref descriptor is assembled and taken apart.
//
// `ABI/BoxLayout.cpp` said it was "the one place a descriptor is built in the
// lowering passes". That was true when written and stopped being true when
// `buildMemRef1D` arrived for the exception triple, which builds the same five
// fields from the other side of the tree -- the fourth comment in this area
// found claiming an exclusivity nothing checked. Rather than correct the count
// again, the assembly lives here and both callers come to it.
//
// The two callers genuinely differ in context and that is why they stayed
// apart: the box path rewrites inside a pass, with a real insertion point and
// source location, while the runtime support builders generate whole functions
// from a `SupportBuilder` that carries its own builder and an unknown loc.
// Taking `OpBuilder` and `Location` is what lets one body serve both;
// `Common/SupportBuilder.h` keeps the `SupportBuilder &` spellings as
// forwarders so that layer reads the way it did.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace py::lowering {

// A rank-1 memref's descriptor members. `allocated` is the free()-able base and
// `aligned` the one every access goes through; the runtime's entities are
// single allocations, so the two coincide, but taking them apart separately is
// what keeps that a fact about the data rather than an assumption here.
struct MemRef1DParts {
  mlir::Value allocated;
  mlir::Value aligned;
  mlir::Value offset;
  mlir::Value size;
  mlir::Value stride;
};

inline mlir::Type memRef1DDescriptorType(mlir::MLIRContext *context) {
  mlir::Type i64 = mlir::IntegerType::get(context, 64);
  auto ptr = mlir::LLVM::LLVMPointerType::get(context);
  auto arrayOne = mlir::LLVM::LLVMArrayType::get(i64, 1);
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptr, ptr, i64, arrayOne, arrayOne});
}

inline mlir::Value buildMemRef1D(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Type memrefType,
                                 const MemRef1DParts &parts) {
  mlir::Value descriptor = mlir::LLVM::UndefOp::create(
      builder, loc, memRef1DDescriptorType(builder.getContext()));
  auto set = [&](mlir::Value field, llvm::ArrayRef<std::int64_t> path) {
    descriptor =
        mlir::LLVM::InsertValueOp::create(builder, loc, descriptor, field, path)
            .getResult();
  };
  set(parts.allocated, {0});
  set(parts.aligned, {1});
  set(parts.offset, {2});
  set(parts.size, {3, 0});
  set(parts.stride, {4, 0});
  return mlir::UnrealizedConversionCastOp::create(
             builder, loc, mlir::TypeRange{memrefType}, descriptor)
      .getResult(0);
}

inline MemRef1DParts explodeMemRef1D(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value memref) {
  mlir::Value descriptor =
      mlir::UnrealizedConversionCastOp::create(
          builder, loc,
          mlir::TypeRange{memRef1DDescriptorType(builder.getContext())}, memref)
          .getResult(0);
  auto member = [&](llvm::ArrayRef<std::int64_t> path) {
    return mlir::LLVM::ExtractValueOp::create(builder, loc, descriptor, path)
        .getResult();
  };
  return {member({0}), member({1}), member({2}), member({3, 0}),
          member({4, 0})};
}

} // namespace py::lowering
