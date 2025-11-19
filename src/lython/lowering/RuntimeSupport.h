#pragma once

#include "PyDialectTypes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

namespace py {

struct RuntimeSymbols {
  static constexpr llvm::StringLiteral kStrFromUtf8{"LyUnicode_FromUTF8"};
  static constexpr llvm::StringLiteral kTupleNew{"LyTuple_New"};
  static constexpr llvm::StringLiteral kTupleSetItem{"LyTuple_SetItem"};
  static constexpr llvm::StringLiteral kGetNone{"Ly_GetNone"};
  static constexpr llvm::StringLiteral kGetBuiltinPrint{"Ly_GetBuiltinPrint"};
  static constexpr llvm::StringLiteral kCallVectorcall{"Ly_CallVectorcall"};
  static constexpr llvm::StringLiteral kCall{"Ly_Call"};
  static constexpr llvm::StringLiteral kDictNew{"LyDict_New"};
  static constexpr llvm::StringLiteral kDictInsert{"LyDict_Insert"};
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

} // namespace py
