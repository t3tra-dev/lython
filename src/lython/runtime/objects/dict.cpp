#include "objects/dict.h"

#include <algorithm>

LyDictObject *LyDict_Cast(LyObject *object) {
  if (!object || object->ob_type != &LyDict_Type())
    return nullptr;
  return reinterpret_cast<LyDictObject *>(object);
}

extern "C" {

LyObject *LyDict_New() {
  void *raw = ::operator new(sizeof(LyDictObject));
  auto *dict = reinterpret_cast<LyDictObject *>(raw);
  dict->ob_base.ob_refcnt = 1;
  dict->ob_base.ob_type = &LyDict_Type();
  dict->entries.clear();
  return reinterpret_cast<LyObject *>(dict);
}

LyObject *LyDict_Insert(LyObject *dictObject, LyObject *key, LyObject *value) {
  LyDictObject *dict = LyDict_Cast(dictObject);
  if (!dict)
    return nullptr;

  auto it =
      std::find_if(dict->entries.begin(), dict->entries.end(),
                   [&](const LyDictEntry &entry) { return entry.key == key; });

  if (it == dict->entries.end()) {
    dict->entries.push_back({key, value});
    Ly_IncRef(key);
    Ly_IncRef(value);
  } else {
    it->value = value;
    Ly_IncRef(value);
  }

  return dictObject;
}
}
