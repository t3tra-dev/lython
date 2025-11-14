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
  static constexpr llvm::StringLiteral kStrFromUtf8{"__py_str_from_utf8"};
  static constexpr llvm::StringLiteral kTupleNew{"__py_tuple_new"};
  static constexpr llvm::StringLiteral kTupleSetItem{"__py_tuple_setitem"};
  static constexpr llvm::StringLiteral kGetNone{"__py_get_none"};
  static constexpr llvm::StringLiteral kGetBuiltinPrint{
      "__py_get_builtin_print"};
  static constexpr llvm::StringLiteral kCallVectorcall{"__py_call_vectorcall"};
  static constexpr llvm::StringLiteral kCall{"__py_call"};
  static constexpr llvm::StringLiteral kDictNew{"__py_dict_new"};
  static constexpr llvm::StringLiteral kDictInsert{"__py_dict_insert"};
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
