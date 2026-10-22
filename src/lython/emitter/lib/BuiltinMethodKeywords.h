#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace lython::emitter {

// The parameter names CPython's argument clinic gives the builtin methods
// that accept a keyword at all, in position order, and how many of them are
// positional-only. A keyword spelling of one of these is the argument at that
// position: the emitter rewrites the call so (EmitterCalls.cpp), and the type
// inference reads the call so (TypeSystem.cpp) -- the two must not disagree,
// or a value is emitted that its slot was never typed for.
//
// ⛔ Only these. find/index/center/startswith and the rest are positional-only
// in CPython too, and their refusal here is the same answer.
struct BuiltinMethodParameters {
  llvm::ArrayRef<llvm::StringRef> names;
  unsigned positionalOnly = 0;
};

inline std::optional<BuiltinMethodParameters>
builtinMethodParameters(llvm::StringRef method) {
  static const llvm::StringRef kSplitNames[] = {"sep", "maxsplit"};
  static const llvm::StringRef kReplaceNames[] = {"old", "new", "count"};
  static const llvm::StringRef kSplitlinesNames[] = {"keepends"};
  static const llvm::StringRef kCodecNames[] = {"encoding", "errors"};
  // ⛔ `signed` is keyword-only in CPython and is accepted positionally
  // here: the position is how the manifest's overloads are told apart.
  static const llvm::StringRef kToBytesNames[] = {"length", "byteorder",
                                                  "signed"};
  static const llvm::StringRef kFromBytesNames[] = {"bytes", "byteorder",
                                                    "signed"};
  if (method == "split" || method == "rsplit")
    return BuiltinMethodParameters{kSplitNames, 0};
  if (method == "replace")
    return BuiltinMethodParameters{kReplaceNames, 2};
  if (method == "splitlines")
    return BuiltinMethodParameters{kSplitlinesNames, 0};
  if (method == "encode" || method == "decode")
    return BuiltinMethodParameters{kCodecNames, 0};
  if (method == "to_bytes")
    return BuiltinMethodParameters{kToBytesNames, 0};
  if (method == "from_bytes")
    return BuiltinMethodParameters{kFromBytesNames, 0};
  return std::nullopt;
}

} // namespace lython::emitter
