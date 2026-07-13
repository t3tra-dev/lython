#pragma once

#include "mlir/Support/LogicalResult.h"
#include "llvm/IR/PassManager.h"

#include <string>

namespace llvm {
class Triple;
namespace orc {
class LLJIT;
} // namespace orc
} // namespace llvm

namespace lython::driver {

struct SanitizerConfig {
  bool address = false;
  bool leak = false;
  bool thread = false;
  bool undefined = false;

  bool any() const { return address || leak || thread || undefined; }
  bool requiresLLVMInstrumentation() const { return address || thread; }
  bool requiresJITRuntimePreload() const { return address || leak || thread; }
};

void recordSanitizerAction(bool enable, const std::string &value);
mlir::LogicalResult buildSanitizerConfig(SanitizerConfig &config);
std::string sanitizerClangList(const SanitizerConfig &config);

void addSanitizerInstrumentationPasses(llvm::ModulePassManager &modulePM,
                                       const SanitizerConfig &sanitizers);
mlir::LogicalResult
addJITSanitizerRuntimes(llvm::orc::LLJIT &jit,
                        const SanitizerConfig &sanitizers,
                        const llvm::Triple &targetTriple);
mlir::LogicalResult
ensureJITSanitizerRuntimesPreloaded(char **argv,
                                    const SanitizerConfig &sanitizers,
                                    const llvm::Triple &targetTriple);
void callLeakSanitizerHook(const char *symbolName);

} // namespace lython::driver
