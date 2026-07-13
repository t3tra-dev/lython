#include "FuzzerSupport.h"

#include "Driver.h"
#include "Parser.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

// Exercises the full static pipeline: parse -> import resolution (embedded
// stdlib only; the import base directory does not exist) -> emit -> lowering
// with verifiers enabled -> LLVM IR translation + safety metadata
// verification. No JIT execution and no filesystem writes: the contract under
// test is "reject with diagnostics or produce verified IR, never crash" —
// the never-silently-mis-execute boundary itself.

namespace {

// Stage counters, reported at exit when LYTHON_FUZZ_STATS is set. They answer
// "how much fuzz time reaches past the parser" — the cheap-reject fraction is
// pure parser territory that fuzz_parser covers at ~40x the execution rate.
struct StageStats {
  std::uint64_t total = 0;
  std::uint64_t parsed = 0;
  std::uint64_t compiled = 0;

  ~StageStats() {
    if (total == 0 || std::getenv("LYTHON_FUZZ_STATS") == nullptr)
      return;
    std::fprintf(stderr,
                 "[fuzz_pipeline] total=%llu parse-ok=%llu (%.1f%%) "
                 "compile-ok=%llu (%.1f%%)\n",
                 static_cast<unsigned long long>(total),
                 static_cast<unsigned long long>(parsed),
                 100.0 * static_cast<double>(parsed) /
                     static_cast<double>(total),
                 static_cast<unsigned long long>(compiled),
                 100.0 * static_cast<double>(compiled) /
                     static_cast<double>(total));
  }
};

StageStats stats;

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > (8u << 10))
    return 0;
  ++stats.total;
  std::string_view source(reinterpret_cast<const char *>(data), size);

  // Cheap-reject gate: most byte-level mutants die in the parser. Paying the
  // MLIRContext setup (~ms) for them would spend pipeline time on inputs that
  // can only ever add parser coverage. The accepted minority parses twice
  // (again inside the driver) — microseconds against the pipeline cost.
  {
    lython::parser::ParseOptions parseOptions;
    parseOptions.typeComments = true;
    lython::parser::ParseResult parsed =
        lython::parser::parse(source, "<fuzz>.py", parseOptions);
    if (!parsed.ok())
      return 0;
  }
  ++stats.parsed;

  mlir::MLIRContext context(lythonFuzzerRegistry(),
                            mlir::MLIRContext::Threading::DISABLED);
  context.getDiagEngine().registerHandler(
      [](mlir::Diagnostic &) { return mlir::success(); });

  lython::driver::DriverOptions options;
  lython::driver::VerifiedLLVMModule out;
  if (mlir::succeeded(lython::driver::compilePythonSourceToLLVMIR(
          llvm::StringRef(source.data(), source.size()), "<fuzz>.py",
          "<lython-fuzz-no-imports>", options, context, out, llvm::nulls())))
    ++stats.compiled;
  return 0;
}
