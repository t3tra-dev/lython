#pragma once

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace py {

class PerfScope {
public:
  explicit PerfScope(llvm::StringRef phase);
  ~PerfScope();

  PerfScope(const PerfScope &) = delete;
  PerfScope &operator=(const PerfScope &) = delete;

private:
  struct State;
  std::unique_ptr<State> state;
};

} // namespace py
