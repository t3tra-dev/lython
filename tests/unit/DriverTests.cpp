#include "Driver.h"

#include "embedded.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <string>

namespace {

// Same one-time process setup as lyc's main() and the fuzz harnesses.
const mlir::DialectRegistry &testRegistry() {
  static mlir::DialectRegistry *registry = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    py::runtime_library::embedded::registerPyRuntimeEmbeddedModules();
    auto *result = new mlir::DialectRegistry();
    lython::driver::registerLythonDialects(*result);
    return result;
  }();
  return *registry;
}

struct CompileResult {
  bool succeeded = false;
  lython::driver::VerifiedLLVMModule verified;
  std::string diagnostics;
};

CompileResult compileSource(llvm::StringRef source) {
  CompileResult result;
  mlir::MLIRContext context(testRegistry());
  llvm::raw_string_ostream diag(result.diagnostics);
  lython::driver::DriverOptions options;
  result.succeeded = mlir::succeeded(lython::driver::compilePythonSourceToLLVMIR(
      source, "<test>.py", "<lython-no-import-dir>", options, context,
      result.verified, diag));
  return result;
}

TEST(DriverTest, CompilesHelloToVerifiedLLVMIR) {
  CompileResult result = compileSource("print(\"hello driver\")\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  ASSERT_TRUE(result.verified.llvmModule);
  EXPECT_NE(result.verified.llvmModule->getFunction("__main__"), nullptr);
}

TEST(DriverTest, ReportsParseErrorDiagnostics) {
  CompileResult result = compileSource("def broken(:\n");
  EXPECT_FALSE(result.succeeded);
  EXPECT_NE(result.diagnostics.find("parse error"), std::string::npos)
      << result.diagnostics;
}

TEST(DriverTest, ReportsEmitErrorDiagnostics) {
  CompileResult result = compileSource("x = eval(\"1\")\n");
  EXPECT_FALSE(result.succeeded);
  EXPECT_NE(result.diagnostics.find("unresolved name 'eval'"),
            std::string::npos)
      << result.diagnostics;
}

// The embedded stdlib must resolve through the driver library itself: the
// import base directory does not exist, so `import os` can only come from
// the sources compiled into LythonDriver.
TEST(DriverTest, ResolvesEmbeddedStdlibImports) {
  CompileResult result = compileSource("import os\nprint(os.name)\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  EXPECT_NE(result.verified.llvmModule->getFunction("__main__"), nullptr);
}

TEST(DriverTest, RepeatedCompileIsStable) {
  for (int round = 0; round < 3; ++round) {
    CompileResult result = compileSource("print(40 + 2)\n");
    EXPECT_TRUE(result.succeeded) << "round " << round << ": "
                                  << result.diagnostics;
  }
}

} // namespace
