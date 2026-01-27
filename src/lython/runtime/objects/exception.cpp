#include "objects/exception.h"

#include "objects/tuple.h"

#include <cxxabi.h>
#include <csetjmp>
#include <new>

namespace {

LyTypeObject makeExceptionType() {
  LyTypeObject type{};
  type.ob_base.ob_base.ob_refcnt = 1;
  type.ob_base.ob_base.ob_type = nullptr;
  type.ob_base.ob_size = 0;
  type.tp_name = "Exception";
  type.tp_basicsize = sizeof(LyExceptionObject);
  type.tp_itemsize = 0;
  type.tp_vectorcall_offset = 0;
  type.tp_dealloc = &LyException_Dealloc;
  type.tp_repr = nullptr;
  return type;
}

thread_local LyObject *g_current_exception = nullptr;
thread_local jmp_buf *g_jump_env = nullptr;

} // namespace

LyTypeObject &LyException_Type() {
  static LyTypeObject type = makeExceptionType();
  return type;
}

void LyException_Dealloc(LyObject *object) {
  auto *exc = reinterpret_cast<LyExceptionObject *>(object);
  if (exc->message)
    Ly_DecRef(reinterpret_cast<LyObject *>(exc->message));
  ::operator delete(exc);
}

extern "C" {

LyExceptionObject *LyException_New(LyObject *type, LyUnicodeObject *message,
                                   LyTupleObject *args, LyObject *cause,
                                   LyObject *context, LyObject *traceback,
                                   LyObject *location, LyObject *extras) {
  (void)type;
  (void)args;
  (void)cause;
  (void)context;
  (void)traceback;
  (void)location;
  (void)extras;

  auto *exc = static_cast<LyExceptionObject *>(
      ::operator new(sizeof(LyExceptionObject), std::nothrow));
  if (!exc)
    return nullptr;
  exc->ob_base.ob_refcnt = 1;
  exc->ob_base.ob_type = &LyException_Type();
  exc->message = message;
  if (message)
    Ly_IncRef(reinterpret_cast<LyObject *>(message));
  return exc;
}

void LyException_SetCurrent(LyObject *exception) {
  if (exception)
    Ly_IncRef(exception);
  if (g_current_exception)
    Ly_DecRef(g_current_exception);
  g_current_exception = exception;
}

LyObject *LyException_GetCurrent() { return g_current_exception; }

void LyException_Clear() {
  if (g_current_exception)
    Ly_DecRef(g_current_exception);
  g_current_exception = nullptr;
}

void LyEH_Throw(LyObject *exception) {
  LyException_SetCurrent(exception);
  if (g_jump_env)
    longjmp(*g_jump_env, 1);
  throw exception;
}

LyObject *LyEH_Capture(void *exc) {
  LyObject *current = LyException_GetCurrent();
  if (current)
    return current;
  (void)exc;
  auto *msg = LyUnicode_FromUTF8("unhandled exception", 20);
  auto *created = LyException_New(nullptr, msg, nullptr, nullptr, nullptr,
                                  nullptr, nullptr, nullptr);
  current = reinterpret_cast<LyObject *>(created);
  LyException_SetCurrent(current);
  if (msg)
    Ly_DecRef(reinterpret_cast<LyObject *>(msg));
  return current;
}

void LyEH_SetJump(void *env) {
  g_jump_env = reinterpret_cast<jmp_buf *>(env);
}

void LyEH_ClearJump() { g_jump_env = nullptr; }
}
