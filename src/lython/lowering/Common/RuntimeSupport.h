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
  static constexpr llvm::StringLiteral kStrInternStaticUtf8{
      "LyUnicode_InternStaticUTF8"};
  static constexpr llvm::StringLiteral kUnicodeConcat{"LyUnicode_Concat"};
  static constexpr llvm::StringLiteral kLongFromI64{"LyLong_FromI64"};
  static constexpr llvm::StringLiteral kLongAsI64{"LyLong_AsI64"};
  static constexpr llvm::StringLiteral kLongFromString{"LyLong_FromString"};
  static constexpr llvm::StringLiteral kFloatFromDouble{"LyFloat_FromDouble"};
  static constexpr llvm::StringLiteral kFloatAsDouble{"LyFloat_AsDouble"};
  static constexpr llvm::StringLiteral kGetNone{"Ly_GetNone"};
  static constexpr llvm::StringLiteral kGetBuiltinPrint{"Ly_GetBuiltinPrint"};
  static constexpr llvm::StringLiteral kBuiltinPrintImpl{"builtin_print_impl"};
  static constexpr llvm::StringLiteral kExceptionNew{"LyException_New"};
  static constexpr llvm::StringLiteral kExceptionSetCurrent{
      "LyException_SetCurrent"};
  static constexpr llvm::StringLiteral kExceptionGetCurrent{
      "LyException_GetCurrent"};
  static constexpr llvm::StringLiteral kExceptionClear{"LyException_Clear"};
  static constexpr llvm::StringLiteral kEHThrow{"LyEH_Throw"};
  static constexpr llvm::StringLiteral kEHCapture{"LyEH_Capture"};
  static constexpr llvm::StringLiteral kEHReportUnhandled{
      "LyEH_ReportUnhandled"};
  static constexpr llvm::StringLiteral kTracebackPush{"LyTraceback_Push"};
  static constexpr llvm::StringLiteral kTracebackPop{"LyTraceback_Pop"};
  // Generic numeric operations (with type dispatch)
  static constexpr llvm::StringLiteral kNumberAdd{"LyNumber_Add"};
  static constexpr llvm::StringLiteral kNumberSub{"LyNumber_Sub"};
  static constexpr llvm::StringLiteral kNumberLt{"LyNumber_Lt"};
  static constexpr llvm::StringLiteral kNumberLe{"LyNumber_Le"};
  static constexpr llvm::StringLiteral kNumberGt{"LyNumber_Gt"};
  static constexpr llvm::StringLiteral kNumberGe{"LyNumber_Ge"};
  static constexpr llvm::StringLiteral kNumberEq{"LyNumber_Eq"};
  static constexpr llvm::StringLiteral kNumberNe{"LyNumber_Ne"};
  // Type-specialized integer operations (inlinable fast paths)
  static constexpr llvm::StringLiteral kLongAdd{"LyLong_Add"};
  static constexpr llvm::StringLiteral kLongSub{"LyLong_Sub"};
  static constexpr llvm::StringLiteral kLongCompare{"LyLong_Compare"};
  static constexpr llvm::StringLiteral kBoolFromBool{"LyBool_FromBool"};
  static constexpr llvm::StringLiteral kBoolAsBool{"LyBool_AsBool"};
  static constexpr llvm::StringLiteral kIncRef{"Ly_IncRef"};
  static constexpr llvm::StringLiteral kDecRef{"Ly_DecRef"};
  static constexpr llvm::StringLiteral kObjectRepr{"LyObject_Repr"};
  static constexpr llvm::StringLiteral kObjectEqBool{"LyObject_EqBool"};
  static constexpr llvm::StringLiteral kClassReprNamed{"LyClass_ReprNamed"};
  static constexpr llvm::StringLiteral kMemAlloc{"LyMem_Alloc"};
  static constexpr llvm::StringLiteral kMemFree{"LyMem_Free"};
  static constexpr llvm::StringLiteral kStrFromUtf8Len{"LyUnicode_FromUTF8Len"};
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

// Pattern population functions

void populatePyValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     mlir::RewritePatternSet &patterns);
void populatePyListValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns);
void populatePyNumberValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                           mlir::RewritePatternSet &patterns);
void populatePyClassValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                          mlir::RewritePatternSet &patterns);
void populatePyTupleValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                          mlir::RewritePatternSet &patterns);
void populatePyDictValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns);
void populatePyCallLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns);
void populatePyRefCountLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns);
void populatePyFuncLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns);

/// Optimization functions

/// Runs early publication preparation on Py dialect ops before refcount
/// insertion. This inserts explicit py.publish boundaries and computes
/// publication summary attributes on py.func symbols.
void runEarlyPublicationPreparation(mlir::ModuleOp module);

/// Runs pre-lowering optimizations on Py dialect ops.
/// Call this after call conversion and before value conversion.
void runPreLoweringOptimizations(mlir::ModuleOp module);

/// Runs post-lowering optimizations on LLVM dialect ops.
/// Call this after value conversion completes.
void runPostLoweringOptimizations(mlir::ModuleOp module);

/// Creates a pass that applies all Py-specific optimizations.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPyOptimizationPass();

/// Creates a pass that prepares explicit publication boundaries before
/// refcount insertion and runtime lowering.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass();

/// Creates a pass that automatically inserts py.incref/py.decref operations.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass();

/// Creates a pass that verifies @native functions do not use py.* types.
/// This enforces the modal logic separation between Primitive World and Object
/// World.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass();

} // namespace py
