#pragma once

#include "PyDialectTypes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

namespace py {

struct RuntimeSymbols {
  static constexpr llvm::StringLiteral kStrFromUtf8{"LyUnicode_FromUTF8"};
  static constexpr llvm::StringLiteral kLongFromI64{"LyLong_FromI64"};
  static constexpr llvm::StringLiteral kFloatFromDouble{"LyFloat_FromDouble"};
  static constexpr llvm::StringLiteral kTupleNew{"LyTuple_New"};
  static constexpr llvm::StringLiteral kTupleSetItem{"LyTuple_SetItem"};
  static constexpr llvm::StringLiteral kGetNone{"Ly_GetNone"};
  static constexpr llvm::StringLiteral kGetBuiltinPrint{"Ly_GetBuiltinPrint"};
  static constexpr llvm::StringLiteral kCallVectorcall{"Ly_CallVectorcall"};
  static constexpr llvm::StringLiteral kCall{"Ly_Call"};
  static constexpr llvm::StringLiteral kDictNew{"LyDict_New"};
  static constexpr llvm::StringLiteral kDictInsert{"LyDict_Insert"};
  // Generic numeric operations (with type dispatch)
  static constexpr llvm::StringLiteral kNumberAdd{"LyNumber_Add"};
  static constexpr llvm::StringLiteral kNumberSub{"LyNumber_Sub"};
  static constexpr llvm::StringLiteral kNumberLe{"LyNumber_Le"};
  // Type-specialized integer operations (inlinable fast paths)
  static constexpr llvm::StringLiteral kLongAdd{"LyLong_Add"};
  static constexpr llvm::StringLiteral kLongSub{"LyLong_Sub"};
  static constexpr llvm::StringLiteral kLongCompare{"LyLong_Compare"};
  static constexpr llvm::StringLiteral kBoolFromBool{"LyBool_FromBool"};
  static constexpr llvm::StringLiteral kBoolAsBool{"LyBool_AsBool"};
  static constexpr llvm::StringLiteral kIncRef{"Ly_IncRef"};
  static constexpr llvm::StringLiteral kDecRef{"Ly_DecRef"};
};

class PyLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  explicit PyLLVMTypeConverter(mlir::MLIRContext *ctx);

  mlir::Type getPyObjectPtrType() const { return pyObjectPtrType; }

private:
  mlir::Type pyObjectPtrType;
};

class RuntimeAPI {
public:
  RuntimeAPI(mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
             const PyLLVMTypeConverter &typeConverter);

  mlir::LLVM::CallOp call(mlir::Location loc, llvm::StringRef name,
                          mlir::Type resultType, mlir::ValueRange operands);
  mlir::Value getStringLiteral(mlir::Location loc, mlir::StringAttr literal);
  mlir::Value getI64Constant(mlir::Location loc, std::int64_t value);
  mlir::Value getF64Constant(mlir::Location loc, double value);
  mlir::Type getPyObjectPtrType() const { return pyObjectPtrType; }

private:
  mlir::ModuleOp module;
  mlir::ConversionPatternRewriter &rewriter;
  mlir::Type pyObjectPtrType;
};

void populatePyValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     mlir::RewritePatternSet &patterns);
void populatePyTupleLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     mlir::RewritePatternSet &patterns);
void populatePyDictLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns);
void populatePyCallLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns);
void populatePyRefCountLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns);

/// Creates a pass that automatically inserts py.incref/py.decref operations.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass();

} // namespace py
