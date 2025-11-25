#include "lyrt.h"

#include "objects/function.h"
#include "objects/unicode.h"

#include <cstring>
#include <iostream>

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
