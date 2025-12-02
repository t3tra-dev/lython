#include "objects/tuple.h"
#include "objects/unicode.h"

#include <climits>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kTuplesPerBlock = 64;

std::size_t tupleByteSize(std::size_t elements) {
  constexpr std::size_t baseOffset = offsetof(LyTupleObject, ob_item);
  if (elements == 0)
    return baseOffset;
  return baseOffset + elements * sizeof(LyObject *);
}

struct TupleArena {
  std::size_t elementCount;
  std::size_t tupleBytes;
  std::vector<std::unique_ptr<char[]>> blocks;
  std::size_t nextIndex = 0;
};

TupleArena &getArena(std::size_t count) {
  static std::unordered_map<std::size_t, TupleArena> arenas;
  auto [it, inserted] =
      arenas.try_emplace(count, TupleArena{count, tupleByteSize(count), {}, 0});
  return it->second;
}

LyTupleObject *allocateTuple(std::size_t size) {
  TupleArena &arena = getArena(size);
  std::size_t blockIndex = arena.nextIndex / kTuplesPerBlock;
  std::size_t offsetIndex = arena.nextIndex % kTuplesPerBlock;
  if (blockIndex >= arena.blocks.size()) {
    arena.blocks.push_back(
        std::make_unique<char[]>(arena.tupleBytes * kTuplesPerBlock));
  }
  char *base = arena.blocks[blockIndex].get() + offsetIndex * arena.tupleBytes;
  arena.nextIndex++;
  std::memset(base, 0, arena.tupleBytes);
  auto *tuple = reinterpret_cast<LyTupleObject *>(base);
  tuple->ob_base.ob_base.ob_refcnt = 1;
  tuple->ob_base.ob_base.ob_type = &LyTuple_Type();
  tuple->ob_base.ob_size = static_cast<Ly_ssize_t>(size);
  return tuple;
}

} // namespace

// Singleton empty tuple (immortal - refcount max value)
static LyTupleObject *emptyTuple = nullptr;

extern "C" {

LyTupleObject *LyTuple_New(std::size_t size) { return allocateTuple(size); }

LyTupleObject *Ly_GetEmptyTuple() {
  if (!emptyTuple) {
    emptyTuple = allocateTuple(0);
    // Make immortal by setting max refcount
    emptyTuple->ob_base.ob_base.ob_refcnt = PTRDIFF_MAX;
  }
  return emptyTuple;
}

void LyTuple_SetItem(LyTupleObject *tuple, std::size_t index, LyObject *value) {
  if (!tuple)
    return;
  auto size = static_cast<std::size_t>(tuple->ob_base.ob_size);
  if (index >= size)
    return;
  tuple->ob_item[index] = value;
  Ly_IncRef(value);
}
}

void LyTuple_Dealloc(LyObject *object) {
  if (!object)
    return;
  auto *tuple = reinterpret_cast<LyTupleObject *>(object);
  Ly_ssize_t size = tuple->ob_base.ob_size;
  for (Ly_ssize_t i = 0; i < size; ++i) {
    Ly_DecRef(tuple->ob_item[i]);
  }
}

LyUnicodeObject *LyTuple_Repr(LyObject *object) {
  auto *tuple = reinterpret_cast<LyTupleObject *>(object);
  std::string text = "(";
  Ly_ssize_t size = tuple->ob_base.ob_size;
  for (Ly_ssize_t i = 0; i < size; ++i) {
    LyUnicodeObject *itemRepr = LyObject_Repr(tuple->ob_item[i]);
    if (itemRepr) {
      text.append(itemRepr->utf8_data,
                  static_cast<std::size_t>(itemRepr->utf8_length));
      Ly_DecRef(reinterpret_cast<LyObject *>(itemRepr));
    }
    if (i + 1 < size)
      text += ", ";
  }
  text += ")";
  return LyUnicode_FromUTF8(text.c_str(), text.size());
}
