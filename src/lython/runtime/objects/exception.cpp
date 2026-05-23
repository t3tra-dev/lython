#include "objects/exception.h"

#include <csetjmp>
#include <cxxabi.h>
#include <new>

namespace {

LyTypeObject makeExceptionType() {
  LyTypeObject type{};
  type.ob_base.ob_base.ob_refcnt = kLyImmortalRefcount;
  type.ob_base.ob_base.ob_type = nullptr;
  type.ob_base.ob_size = 0;
  type.tp_name = "Exception";
  type.tp_basicsize = sizeof(LyExceptionObject);
  type.tp_itemsize = 0;
  type.tp_vectorcall_offset = 0;
  type.tp_dealloc = &LyException_Dealloc;
  type.tp_repr = &LyException_Repr;
  return type;
}

thread_local LyObject *g_current_exception = nullptr;
thread_local jmp_buf *g_jump_env = nullptr;

void increfIfPresent(LyObject *object) {
  if (object)
    Ly_IncRef(object);
}

void decrefIfPresent(LyObject *object) {
  if (object)
    Ly_DecRef(object);
}

} // namespace

LyTypeObject &LyException_Type() {
  static LyTypeObject type = makeExceptionType();
  return type;
}

void LyException_Dealloc(LyObject *object) {
  auto *exc = reinterpret_cast<LyExceptionObject *>(object);
  decrefIfPresent(exc->type);
  decrefIfPresent(reinterpret_cast<LyObject *>(exc->message));
  decrefIfPresent(exc->args);
  decrefIfPresent(exc->cause);
  decrefIfPresent(exc->context);
  decrefIfPresent(exc->traceback);
  decrefIfPresent(exc->location);
  decrefIfPresent(exc->extras);
  ::operator delete(exc);
}

LyUnicodeObject *LyException_Repr(LyObject *object) {
  auto *exc = reinterpret_cast<LyExceptionObject *>(object);
  if (!exc || !exc->message)
    return LyUnicode_FromUTF8("", 0);
  Ly_IncRef(reinterpret_cast<LyObject *>(exc->message));
  return exc->message;
}

extern "C" {

LyExceptionObject *LyException_New(LyObject *type, LyUnicodeObject *message,
                                   LyObject *args, LyObject *cause,
                                   LyObject *context, LyObject *traceback,
                                   LyObject *location, LyObject *extras) {
  // The runtime type object is still the concrete Exception type. The Python
  // class payload is preserved separately once class objects become durable.
  (void)type;

  auto *exc = static_cast<LyExceptionObject *>(
      ::operator new(sizeof(LyExceptionObject), std::nothrow));
  if (!exc)
    return nullptr;
  exc->ob_base.ob_refcnt = 1;
  exc->ob_base.ob_type = &LyException_Type();
  exc->type = nullptr;
  exc->message = message;
  exc->args = args;
  exc->cause = cause;
  exc->context = context;
  exc->traceback = traceback;
  exc->location = location;
  exc->extras = extras;
  increfIfPresent(reinterpret_cast<LyObject *>(message));
  increfIfPresent(args);
  increfIfPresent(cause);
  increfIfPresent(context);
  increfIfPresent(traceback);
  increfIfPresent(location);
  increfIfPresent(extras);
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
  if (g_current_exception)
    Ly_DecRef(g_current_exception);
  g_current_exception = exception;
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
  if (current)
    Ly_DecRef(current);
  if (msg)
    Ly_DecRef(reinterpret_cast<LyObject *>(msg));
  return current;
}

void LyEH_SetJump(void *env) { g_jump_env = reinterpret_cast<jmp_buf *>(env); }

void LyEH_ClearJump() { g_jump_env = nullptr; }
}
