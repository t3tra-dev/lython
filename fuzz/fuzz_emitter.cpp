#include "FuzzerSupport.h"

#include "Emitter.h"
#include "Parser.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/TargetParser/Host.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

// Exercises the py dialect emitter on every parser-accepted input. The
// contract under test: the emitter either produces a module or reports
// diagnostics — it never crashes or aborts (llvm_unreachable) on accepted
// syntax.
namespace {

// Stage counters, reported at exit when LYTHON_FUZZ_STATS is set.
struct StageStats {
  std::uint64_t total = 0;
  std::uint64_t parsed = 0;

  ~StageStats() {
    if (total == 0 || std::getenv("LYTHON_FUZZ_STATS") == nullptr)
      return;
    std::fprintf(stderr, "[fuzz_emitter] total=%llu parse-ok=%llu (%.1f%%)\n",
                 static_cast<unsigned long long>(total),
                 static_cast<unsigned long long>(parsed),
                 100.0 * static_cast<double>(parsed) /
                     static_cast<double>(total));
  }
};

StageStats stats;

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > (16u << 10))
    return 0;
  ++stats.total;
  std::string_view source(reinterpret_cast<const char *>(data), size);
  lython::parser::ParseOptions parseOptions;
  parseOptions.typeComments = true;
  lython::parser::ParseResult parsed =
      lython::parser::parse(source, "<fuzz>.py", parseOptions);
  if (!parsed.ok())
    return 0;
  ++stats.parsed;

  mlir::MLIRContext context(lythonFuzzerRegistry(),
                            mlir::MLIRContext::Threading::DISABLED);
  context.getDiagEngine().registerHandler(
      [](mlir::Diagnostic &) { return mlir::success(); });

  lython::emitter::EmitOptions options;
  options.targetTriple = llvm::sys::getDefaultTargetTriple();
  (void)lython::emitter::emitModule(*parsed.tree, context, "__main__",
                                    "<fuzz>.py", options);
  return 0;
}
