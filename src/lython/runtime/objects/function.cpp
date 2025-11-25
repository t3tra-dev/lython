#include "objects/function.h"
#include "objects/object.h"

void LyFunction_Dealloc(LyObject *object) {
  if (!object)
    return;

  auto *func = reinterpret_cast<LyFunctionObject *>(object);

  // Decref owned references
  // Note: func_globals is a borrowed reference (owned by module), not decref'd
  // Note: func_code will need decref when code objects are implemented
  // Note: func_weakreflist needs special handling when weak refs are
  // implemented
  Ly_DecRef(func->func_defaults);
  Ly_DecRef(func->func_kwdefaults);
  Ly_DecRef(func->func_closure);
  Ly_DecRef(func->func_doc);
  Ly_DecRef(func->func_name);
  Ly_DecRef(func->func_qualname);
  Ly_DecRef(func->func_module);
  Ly_DecRef(func->func_annotations);
  Ly_DecRef(func->func_dict);

  ::operator delete(func);
}

LyVectorcallFunc LyFunction_LoadVectorcall(LyObject *callable) {
  if (!callable || !callable->ob_type)
    return nullptr;
  LyTypeObject *type = callable->ob_type;
  if (type->tp_vectorcall_offset <= 0)
    return nullptr;
  auto *slot = reinterpret_cast<LyVectorcallFunc *>(
      reinterpret_cast<char *>(callable) + type->tp_vectorcall_offset);
  return slot ? *slot : nullptr;
}
