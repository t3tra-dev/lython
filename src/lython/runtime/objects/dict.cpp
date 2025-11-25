#include "objects/dict.h"
#include "objects/unicode.h"

#include <algorithm>
#include <string>

LyDictObject *LyDict_Cast(LyObject *object) {
  if (!object || object->ob_type != &LyDict_Type())
    return nullptr;
  return reinterpret_cast<LyDictObject *>(object);
}

extern "C" {

LyObject *LyDict_New() {
  auto *dict = new LyDictObject();
  dict->ob_base.ob_refcnt = 1;
  dict->ob_base.ob_type = &LyDict_Type();
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

void LyDict_Dealloc(LyObject *object) {
  if (!object)
    return;
  auto *dict = LyDict_Cast(object);
  if (!dict)
    return;
  for (auto &entry : dict->entries) {
    Ly_DecRef(entry.key);
    Ly_DecRef(entry.value);
  }
  delete dict;
}

LyUnicodeObject *LyDict_Repr(LyObject *object) {
  auto *dict = LyDict_Cast(object);
  if (!dict)
    return LyUnicode_FromUTF8("{}", 2);
  std::string text = "{";
  for (std::size_t i = 0; i < dict->entries.size(); ++i) {
    LyUnicodeObject *keyRepr = LyObject_Repr(dict->entries[i].key);
    LyUnicodeObject *valRepr = LyObject_Repr(dict->entries[i].value);
    if (keyRepr) {
      text.append(keyRepr->utf8_data,
                  static_cast<std::size_t>(keyRepr->utf8_length));
      Ly_DecRef(reinterpret_cast<LyObject *>(keyRepr));
    }
    text += ": ";
    if (valRepr) {
      text.append(valRepr->utf8_data,
                  static_cast<std::size_t>(valRepr->utf8_length));
      Ly_DecRef(reinterpret_cast<LyObject *>(valRepr));
    }
    if (i + 1 < dict->entries.size())
      text += ", ";
  }
  text += "}";
  return LyUnicode_FromUTF8(text.c_str(), text.size());
}
