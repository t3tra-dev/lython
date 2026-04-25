#include "objects/unicode.h"

#include <cstring>
#include <string>
#include <unordered_map>

namespace {

LyUnicodeObject *allocateUnicodeFromUtf8(const char *data, std::size_t len) {
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

struct StaticUnicodeInternTable {
  LyMutex mutex;
  std::unordered_map<std::string, LyUnicodeObject *> entries;
};

StaticUnicodeInternTable &getStaticUnicodeInternTable() {
  static StaticUnicodeInternTable table;
  return table;
}

} // namespace

const char *LyUnicode_AsUTF8(LyObject *object) {
  if (!object || object->ob_type != &LyUnicode_Type())
    return nullptr;
  auto *unicode = reinterpret_cast<LyUnicodeObject *>(object);
  return unicode->utf8_data;
}

extern "C" {

LyUnicodeObject *LyUnicode_FromUTF8(const char *data, std::size_t len) {
  return allocateUnicodeFromUtf8(data, len);
}

LyUnicodeObject *LyUnicode_InternStaticUTF8(const char *data, std::size_t len) {
  if (!data)
    return nullptr;

  auto &table = getStaticUnicodeInternTable();
  std::lock_guard<LyMutex> guard(table.mutex);

  std::string key(data, len);
  auto it = table.entries.find(key);
  if (it != table.entries.end())
    return it->second;

  auto *unicode = allocateUnicodeFromUtf8(data, len);
  if (!unicode)
    return nullptr;
  unicode->ob_base.ob_base.ob_refcnt = kLyImmortalRefcount;
  table.entries.emplace(std::move(key), unicode);
  return unicode;
}

LyUnicodeObject *LyUnicode_Concat(LyObject *lhs, LyObject *rhs) {
  if (!lhs || !rhs)
    return nullptr;
  if (lhs->ob_type != &LyUnicode_Type() || rhs->ob_type != &LyUnicode_Type())
    return nullptr;

  auto *left = reinterpret_cast<LyUnicodeObject *>(lhs);
  auto *right = reinterpret_cast<LyUnicodeObject *>(rhs);
  std::size_t leftLen = static_cast<std::size_t>(left->utf8_length);
  std::size_t rightLen = static_cast<std::size_t>(right->utf8_length);
  std::size_t totalLen = leftLen + rightLen;

  auto *result = LyUnicode_FromUTF8("", 0);
  if (!result)
    return nullptr;
  delete[] result->utf8_data;
  result->utf8_data = new char[totalLen + 1];
  std::memcpy(result->utf8_data, left->utf8_data, leftLen);
  std::memcpy(result->utf8_data + leftLen, right->utf8_data, rightLen);
  result->utf8_data[totalLen] = '\0';
  result->utf8_length = static_cast<Ly_ssize_t>(totalLen);
  result->ob_base.ob_size = static_cast<Ly_ssize_t>(totalLen);
  result->hash = 0;
  return result;
}
}

void LyUnicode_Dealloc(LyObject *object) {
  if (!object)
    return;
  auto *unicode = reinterpret_cast<LyUnicodeObject *>(object);
  delete[] unicode->utf8_data;
  ::operator delete(unicode);
}

LyUnicodeObject *LyUnicode_Repr(LyObject *object) {
  auto *unicode = reinterpret_cast<LyUnicodeObject *>(object);
  return LyUnicode_FromUTF8(unicode->utf8_data,
                            static_cast<std::size_t>(unicode->utf8_length));
}
