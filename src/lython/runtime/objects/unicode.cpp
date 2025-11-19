#include "objects/unicode.h"

#include <cstring>

const char *LyUnicode_AsUTF8(LyObject *object) {
  if (!object || object->ob_type != &LyUnicode_Type())
    return nullptr;
  auto *unicode = reinterpret_cast<LyUnicodeObject *>(object);
  return unicode->utf8_data;
}

extern "C" {

LyUnicodeObject *LyUnicode_FromUTF8(const char *data, std::size_t len) {
  if (!data)
    return nullptr;

  void *raw = ::operator new(sizeof(LyUnicodeObject));
  auto *unicode = reinterpret_cast<LyUnicodeObject *>(raw);
  unicode->ob_base.ob_base.ob_refcnt = 1;
  unicode->ob_base.ob_base.ob_type = &LyUnicode_Type();
  unicode->ob_base.ob_size = static_cast<Ly_ssize_t>(len);
  unicode->utf8_length = static_cast<Ly_ssize_t>(len);
  unicode->utf8_data = new char[len + 1];
  std::memcpy(unicode->utf8_data, data, len);
  unicode->utf8_data[len] = '\0';
  unicode->hash = 0;
  return unicode;
}
}
