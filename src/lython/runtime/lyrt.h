#pragma once

#include <cstdint>

#include "objects/bool.h"
#include "objects/exception.h"
#include "objects/float.h"
#include "objects/function.h"
#include "objects/long.h"
#include "objects/object.h"
#include "objects/unicode.h"
#include "traceback.h"

template <typename T> struct UnrankedMemRefType;

extern "C" {
LyUnicodeObject *LyUnicode_FromUTF8(const char *data, std::size_t len);
LyUnicodeObject *LyUnicode_InternStaticUTF8(const char *data, std::size_t len);
LyUnicodeObject *LyUnicode_Concat(LyObject *lhs, LyObject *rhs);

LyObject *Ly_GetNone();
LyFunctionObject *Ly_GetBuiltinPrint();
LyObject *builtin_print_impl(LyObject *module, LyObject *const *objects,
                             Ly_ssize_t objects_length, LyObject *sep,
                             LyObject *end, LyObject *file, int flush);
LyUnicodeObject *LyTensorF16_Repr(UnrankedMemRefType<std::uint16_t> *memref);
LyUnicodeObject *LyTensorF32_Repr(UnrankedMemRefType<float> *memref);
LyUnicodeObject *LyTensorF64_Repr(UnrankedMemRefType<double> *memref);
LyUnicodeObject *LyTensorF128_Repr(UnrankedMemRefType<long double> *memref);
LyUnicodeObject *
_mlir_ciface_LyTensorF16_Repr(UnrankedMemRefType<std::uint16_t> *memref);
LyUnicodeObject *
_mlir_ciface_LyTensorF32_Repr(UnrankedMemRefType<float> *memref);
LyUnicodeObject *
_mlir_ciface_LyTensorF64_Repr(UnrankedMemRefType<double> *memref);
LyUnicodeObject *
_mlir_ciface_LyTensorF128_Repr(UnrankedMemRefType<long double> *memref);
LyUnicodeObject *LyListI64_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *LyListBool_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *LyListF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *LyListPtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                                LyUnicodeObject *(*repr_item)(void *));
LyUnicodeObject *
_mlir_ciface_LyListI64_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *
_mlir_ciface_LyListBool_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *
_mlir_ciface_LyListF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *
_mlir_ciface_LyListPtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                            LyUnicodeObject *(*repr_item)(void *));
LyUnicodeObject *LyTupleI64_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *LyTupleBool_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *LyTupleF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *LyTuplePtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                                 LyUnicodeObject *(*repr_item)(void *));
LyUnicodeObject *
_mlir_ciface_LyTupleI64_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *
_mlir_ciface_LyTupleBool_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *
_mlir_ciface_LyTupleF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref);
LyUnicodeObject *
_mlir_ciface_LyTuplePtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                             LyUnicodeObject *(*repr_item)(void *));
LyUnicodeObject *LyDictPacked_Repr(UnrankedMemRefType<std::int64_t> *memref,
                                   std::int64_t key_kind,
                                   std::int64_t value_kind,
                                   LyUnicodeObject *(*repr_key)(void *),
                                   LyUnicodeObject *(*repr_value)(void *));
LyUnicodeObject *
_mlir_ciface_LyDictPacked_Repr(UnrankedMemRefType<std::int64_t> *memref,
                               std::int64_t key_kind, std::int64_t value_kind,
                               LyUnicodeObject *(*repr_key)(void *),
                               LyUnicodeObject *(*repr_value)(void *));

LyLongObject *LyLong_FromI64(std::int64_t value);
std::int64_t LyLong_AsI64(LyObject *object);
LyFloatObject *LyFloat_FromDouble(double value);
double LyFloat_AsDouble(LyObject *object);
LyObject *LyNumber_Add(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Sub(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Lt(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Le(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Gt(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Ge(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Eq(LyObject *lhs, LyObject *rhs);
LyObject *LyNumber_Ne(LyObject *lhs, LyObject *rhs);
bool LyBool_AsBool(LyObject *object);
bool LyObject_EqBool(LyObject *lhs, LyObject *rhs);
LyUnicodeObject *LyClass_ReprNamed(const char *name, const void *object);
void *LyMem_Alloc(std::size_t size);
void LyMem_Free(void *ptr);

void LyEH_Throw(LyObject *exception);

} // extern "C"
