#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace lython::common {

// ⭐ ONE C3 MERGE. The dialect linearizes over `StringRef` and reports through
// an op diagnostic, the emitter over `std::string` and reports a TypeError;
// the merge itself is the same, so it takes neither. Nullopt is the "no
// linearization exists" case and the caller owns the wording.
template <typename T>
std::optional<llvm::SmallVector<T, 8>>
c3Merge(llvm::SmallVector<llvm::SmallVector<T, 8>, 8> sequences) {
  llvm::SmallVector<T, 8> result;
  auto compact = [&]() {
    llvm::SmallVector<llvm::SmallVector<T, 8>, 8> next;
    for (auto &sequence : sequences)
      if (!sequence.empty())
        next.push_back(std::move(sequence));
    sequences = std::move(next);
  };
  compact();

  while (!sequences.empty()) {
    std::optional<T> candidate;
    for (const auto &sequence : sequences) {
      const T &head = sequence.front();
      bool appearsInTail = false;
      for (const auto &other : sequences) {
        if (llvm::is_contained(llvm::ArrayRef<T>(other).drop_front(), head)) {
          appearsInTail = true;
          break;
        }
      }
      if (!appearsInTail) {
        candidate = head;
        break;
      }
    }
    if (!candidate)
      return std::nullopt;

    result.push_back(*candidate);
    for (auto &sequence : sequences)
      if (!sequence.empty() && sequence.front() == *candidate)
        sequence.erase(sequence.begin());
    compact();
  }
  return result;
}

} // namespace lython::common
