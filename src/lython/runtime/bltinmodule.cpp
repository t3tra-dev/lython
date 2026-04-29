#include "lyrt.h"

#include "objects/function.h"
#include "objects/unicode.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

extern "C" LyObject *builtin_print_impl(LyObject *module,
                                        LyObject *const *objects,
                                        Ly_ssize_t objects_length,
                                        LyObject *sep, LyObject *end,
                                        LyObject *file, int flush);

namespace {

void emitRepr(LyObject *object) {
  LyUnicodeObject *repr = LyObject_Repr(object);
  if (!repr)
    return;
  if (repr->utf8_data)
    std::cout.write(repr->utf8_data, repr->utf8_length);
  Ly_DecRef(reinterpret_cast<LyObject *>(repr));
}

float halfToFloat(std::uint16_t value) {
  std::uint16_t sign = (value >> 15) & 0x1;
  std::uint16_t exp = (value >> 10) & 0x1F;
  std::uint16_t mant = value & 0x3FF;

  if (exp == 0) {
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    float m = static_cast<float>(mant) / 1024.0f;
    float val = std::ldexp(m, -14);
    return sign ? -val : val;
  }
  if (exp == 31) {
    if (mant == 0)
      return sign ? -INFINITY : INFINITY;
    return std::numeric_limits<float>::quiet_NaN();
  }

  float m = 1.0f + static_cast<float>(mant) / 1024.0f;
  float val = std::ldexp(m, static_cast<int>(exp) - 15);
  return sign ? -val : val;
}

template <typename T, typename Convert>
void appendTensorDynamic(std::string &out, const DynamicMemRefType<T> &memref,
                         int64_t dim, int64_t offset, Convert convert) {
  if (dim == memref.rank) {
    std::ostringstream stream;
    stream.setf(std::ios::fmtflags(0), std::ios::floatfield);
    stream << std::setprecision(8)
           << static_cast<long double>(convert(memref.data[offset]));
    out += stream.str();
    return;
  }

  out.push_back('[');
  for (int64_t i = 0; i < memref.sizes[dim]; ++i) {
    if (i != 0)
      out.append(", ");
    appendTensorDynamic(out, memref, dim + 1, offset + i * memref.strides[dim],
                        convert);
  }
  out.push_back(']');
}

std::string formatPythonFloat(double value) {
  std::ostringstream stream;
  stream.setf(std::ios::fmtflags(0), std::ios::floatfield);
  stream << std::setprecision(12) << value;
  std::string result = stream.str();
  if (std::isfinite(value) && result.find_first_of(".eE") == std::string::npos)
    result.append(".0");
  return result;
}

template <typename Format>
LyUnicodeObject *reprPackedListI64(UnrankedMemRefType<std::int64_t> *memref,
                                   Format format) {
  DynamicMemRefType<std::int64_t> dyn(*memref);
  auto loadSlot = [&](std::int64_t slot) -> std::int64_t {
    return dyn.data[dyn.offset + slot * dyn.strides[0]];
  };

  std::int64_t size = loadSlot(0);
  std::string out;
  out.reserve(2 +
              static_cast<std::size_t>(std::max<std::int64_t>(size, 0)) * 4);
  out.push_back('[');
  for (std::int64_t i = 0; i < size; ++i) {
    if (i != 0)
      out.append(", ");
    format(out, loadSlot(4 + i));
  }
  out.push_back(']');
  return LyUnicode_FromUTF8(out.c_str(), out.size());
}

template <typename Format>
LyUnicodeObject *reprPackedTupleI64(UnrankedMemRefType<std::int64_t> *memref,
                                    Format format) {
  DynamicMemRefType<std::int64_t> dyn(*memref);
  auto loadSlot = [&](std::int64_t slot) -> std::int64_t {
    return dyn.data[dyn.offset + slot * dyn.strides[0]];
  };

  std::int64_t size = loadSlot(0);
  std::string out;
  out.reserve(2 +
              static_cast<std::size_t>(std::max<std::int64_t>(size, 0)) * 4);
  out.push_back('(');
  for (std::int64_t i = 0; i < size; ++i) {
    if (i != 0)
      out.append(", ");
    format(out, loadSlot(3 + i));
  }
  if (size == 1)
    out.push_back(',');
  out.push_back(')');
  return LyUnicode_FromUTF8(out.c_str(), out.size());
}

LyObject *builtin_print_vectorcall(LyObject *callable, LyObject *const *args,
                                   std::size_t nargsf, LyObject *) {
  return builtin_print_impl(callable, args, static_cast<Ly_ssize_t>(nargsf),
                            nullptr, nullptr, nullptr, 0);
}

} // namespace

extern "C" {

LyObject *builtin_print_impl(LyObject *, LyObject *const *objects,
                             Ly_ssize_t objects_length, LyObject *, LyObject *,
                             LyObject *, int) {
  for (Ly_ssize_t i = 0; i < objects_length; ++i) {
    emitRepr(objects[i]);
    if (i + 1 < objects_length)
      std::cout << ' ';
  }
  std::cout << std::endl;
  return Ly_GetNone();
}

LyUnicodeObject *LyTensorF16_Repr(UnrankedMemRefType<std::uint16_t> *memref) {
  DynamicMemRefType<std::uint16_t> dyn(*memref);
  std::string out;
  out.reserve(64);
  appendTensorDynamic(out, dyn, 0, dyn.offset,
                      [](std::uint16_t v) { return halfToFloat(v); });
  return LyUnicode_FromUTF8(out.c_str(), out.size());
}

LyUnicodeObject *LyTensorF32_Repr(UnrankedMemRefType<float> *memref) {
  DynamicMemRefType<float> dyn(*memref);
  std::string out;
  out.reserve(64);
  appendTensorDynamic(out, dyn, 0, dyn.offset, [](float v) { return v; });
  return LyUnicode_FromUTF8(out.c_str(), out.size());
}

LyUnicodeObject *LyTensorF64_Repr(UnrankedMemRefType<double> *memref) {
  DynamicMemRefType<double> dyn(*memref);
  std::string out;
  out.reserve(64);
  appendTensorDynamic(out, dyn, 0, dyn.offset, [](double v) { return v; });
  return LyUnicode_FromUTF8(out.c_str(), out.size());
}

LyUnicodeObject *LyTensorF128_Repr(UnrankedMemRefType<long double> *memref) {
  DynamicMemRefType<long double> dyn(*memref);
  std::string out;
  out.reserve(64);
  appendTensorDynamic(out, dyn, 0, dyn.offset, [](long double v) { return v; });
  return LyUnicode_FromUTF8(out.c_str(), out.size());
}

LyUnicodeObject *
_mlir_ciface_LyTensorF16_Repr(UnrankedMemRefType<std::uint16_t> *memref) {
  return LyTensorF16_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyTensorF32_Repr(UnrankedMemRefType<float> *memref) {
  return LyTensorF32_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyTensorF64_Repr(UnrankedMemRefType<double> *memref) {
  return LyTensorF64_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyTensorF128_Repr(UnrankedMemRefType<long double> *memref) {
  return LyTensorF128_Repr(memref);
}

LyUnicodeObject *LyListI64_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return reprPackedListI64(memref, [](std::string &out, std::int64_t value) {
    out.append(std::to_string(value));
  });
}

LyUnicodeObject *LyListBool_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return reprPackedListI64(memref, [](std::string &out, std::int64_t value) {
    out.append(value ? "True" : "False");
  });
}

LyUnicodeObject *LyListF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return reprPackedListI64(memref, [](std::string &out, std::int64_t bits) {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    out.append(formatPythonFloat(value));
  });
}

LyUnicodeObject *LyListPtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                                LyUnicodeObject *(*repr_item)(void *)) {
  return reprPackedListI64(memref, [&](std::string &out, std::int64_t bits) {
    auto *object = reinterpret_cast<void *>(static_cast<std::uintptr_t>(bits));
    LyUnicodeObject *repr = repr_item ? repr_item(object) : nullptr;
    if (!repr)
      return;
    if (repr->utf8_data)
      out.append(repr->utf8_data, repr->utf8_length);
    Ly_DecRef(reinterpret_cast<LyObject *>(repr));
  });
}

LyUnicodeObject *LyTupleI64_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return reprPackedTupleI64(memref, [](std::string &out, std::int64_t value) {
    out.append(std::to_string(value));
  });
}

LyUnicodeObject *LyTupleBool_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return reprPackedTupleI64(memref, [](std::string &out, std::int64_t value) {
    out.append(value ? "True" : "False");
  });
}

LyUnicodeObject *LyTupleF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return reprPackedTupleI64(memref, [](std::string &out, std::int64_t bits) {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    out.append(formatPythonFloat(value));
  });
}

LyUnicodeObject *LyTuplePtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                                 LyUnicodeObject *(*repr_item)(void *)) {
  return reprPackedTupleI64(memref, [&](std::string &out, std::int64_t bits) {
    auto *object = reinterpret_cast<void *>(static_cast<std::uintptr_t>(bits));
    LyUnicodeObject *repr = repr_item ? repr_item(object) : nullptr;
    if (!repr)
      return;
    if (repr->utf8_data)
      out.append(repr->utf8_data, repr->utf8_length);
    Ly_DecRef(reinterpret_cast<LyObject *>(repr));
  });
}

LyUnicodeObject *
_mlir_ciface_LyListI64_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return LyListI64_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyListBool_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return LyListBool_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyListF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return LyListF64Bits_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyListPtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                            LyUnicodeObject *(*repr_item)(void *)) {
  return LyListPtr_Repr(memref, repr_item);
}

LyUnicodeObject *
_mlir_ciface_LyTupleI64_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return LyTupleI64_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyTupleBool_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return LyTupleBool_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyTupleF64Bits_Repr(UnrankedMemRefType<std::int64_t> *memref) {
  return LyTupleF64Bits_Repr(memref);
}

LyUnicodeObject *
_mlir_ciface_LyTuplePtr_Repr(UnrankedMemRefType<std::int64_t> *memref,
                             LyUnicodeObject *(*repr_item)(void *)) {
  return LyTuplePtr_Repr(memref, repr_item);
}

LyFunctionObject *Ly_GetBuiltinPrint() {
  static LyFunctionObject *builtin = [] {
    void *raw = ::operator new(sizeof(LyFunctionObject));
    std::memset(raw, 0, sizeof(LyFunctionObject));
    auto *func = reinterpret_cast<LyFunctionObject *>(raw);
    func->ob_base.ob_refcnt = 1;
    func->ob_base.ob_type = &LyFunction_Type();
    func->func_name =
        reinterpret_cast<LyObject *>(LyUnicode_FromUTF8("print", 5));
    func->vectorcall = &builtin_print_vectorcall;
    return func;
  }();
  return builtin;
}
}
