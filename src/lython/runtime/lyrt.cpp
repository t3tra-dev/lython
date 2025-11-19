#include "lyrt.h"

#include "objects/dict.h"
#include "objects/function.h"
#include "objects/tuple.h"

#include <vector>

extern "C" {

LyObject *Ly_CallVectorcall(LyObject *callable, LyTupleObject *posargs,
                            LyTupleObject *kwnames, LyTupleObject *kwvalues) {
  LyVectorcallFunc fn = LyFunction_LoadVectorcall(callable);
  if (!fn)
    return nullptr;

  std::size_t positionalCount =
      posargs ? static_cast<std::size_t>(posargs->ob_base.ob_size) : 0;
  std::size_t keywordCount =
      kwnames ? static_cast<std::size_t>(kwnames->ob_base.ob_size) : 0;
  if (keywordCount > 0) {
    if (!kwvalues ||
        kwvalues->ob_base.ob_size != static_cast<Ly_ssize_t>(keywordCount)) {
      return nullptr;
    }
  }

  std::vector<LyObject *> storage;
  storage.reserve(positionalCount + keywordCount);
  if (posargs) {
    for (std::size_t i = 0; i < positionalCount; ++i)
      storage.push_back(posargs->ob_item[i]);
  }
  if (kwvalues) {
    for (std::size_t i = 0; i < keywordCount; ++i)
      storage.push_back(kwvalues->ob_item[i]);
  }

  LyObject *kwnamesObj = reinterpret_cast<LyObject *>(kwnames);
  return fn(callable, storage.data(), positionalCount, kwnamesObj);
}

LyObject *Ly_Call(LyObject *callable, LyTupleObject *posargs,
                  LyObject *kwargs) {
  LyTupleObject *kwnames = nullptr;
  LyTupleObject *kwvalues = nullptr;
  if (!kwargs || kwargs == Ly_GetNone()) {
    kwnames = LyTuple_New(0);
    kwvalues = LyTuple_New(0);
  } else if (auto *dict = LyDict_Cast(kwargs)) {
    std::size_t size = dict->entries.size();
    kwnames = LyTuple_New(size);
    kwvalues = LyTuple_New(size);
    for (std::size_t i = 0; i < size; ++i) {
      LyTuple_SetItem(kwnames, i, dict->entries[i].key);
      LyTuple_SetItem(kwvalues, i, dict->entries[i].value);
    }
  } else {
    return nullptr;
  }

  return Ly_CallVectorcall(callable, posargs, kwnames, kwvalues);
}
}
