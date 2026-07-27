#include "Driver.h"

#include "embedded.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
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

CompileResult compileSource(llvm::StringRef source,
                            const lython::driver::DriverOptions &options =
                                lython::driver::DriverOptions{}) {
  CompileResult result;
  mlir::MLIRContext context(testRegistry());
  llvm::raw_string_ostream diag(result.diagnostics);
  result.succeeded = mlir::succeeded(lython::driver::compilePythonSourceToLLVMIR(
      source, "<test>.py", "<lython-no-import-dir>", options, context,
      result.verified, diag));
  return result;
}

lython::driver::DriverOptions targetOptions(llvm::StringRef triple,
                                            llvm::StringRef cpu) {
  lython::driver::DriverOptions options;
  options.targetTriple = triple.str();
  options.targetCPU = cpu.str();
  return options;
}

// The tensor constructor only takes a spelled-out nested literal, so the shape
// has to be written into the source rather than built at runtime.
std::string matrixLiteral(int outer, int inner) {
  std::string text = "[";
  for (int i = 0; i < outer; ++i) {
    text += i ? ",[" : "[";
    for (int j = 0; j < inner; ++j)
      text += (j ? "," : "") + std::to_string((i + j) % 7) + ".0";
    text += "]";
  }
  return text + "]";
}

std::string matmulSource(int m, int k, int n, llvm::StringRef element) {
  std::string type = "Float[" + element.str() + "]";
  return "from lyrt import from_prim\n"
         "from lyrt.prim import Float, Matrix\n"
         "a = Matrix[" +
         type + ", " + std::to_string(m) + ", " + std::to_string(k) + "](" +
         matrixLiteral(m, k) +
         ")\n"
         "b = Matrix[" +
         type + ", " + std::to_string(k) + ", " + std::to_string(n) + "](" +
         matrixLiteral(k, n) +
         ")\n"
         "c = a @ b\n"
         "print(from_prim(c[0, 0]))\n";
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

// Every architecture reaches the tiled GEMM path for some shape: SME declines
// contractions below its work threshold, and the other targets have no matmul
// pass of their own at all. A tile that does not divide its extent must not
// leave the trailing linalg.matmul dynamically shaped, which nothing lowers and
// the affine loop conversion rejects.
TEST(DriverTest, CompilesUnevenMatmulForEveryTensorTarget) {
  struct Target {
    const char *name;
    const char *triple;
    const char *cpu;
  };
  // Generic covers both a non-SME AArch64 host and a plain x86_64 one.
  const Target targets[] = {
      {"arm-sme", "", ""},
      {"arm-generic", "", "apple-m1"},
      {"x86-avx2-fma", "x86_64-unknown-linux-gnu", "haswell"},
      {"x86-sse42", "x86_64-unknown-linux-gnu", "nehalem"},
      {"x86-generic", "x86_64-unknown-linux-gnu", "x86-64"},
  };
  const int shapes[][3] = {{9, 9, 9}, {16, 16, 9}, {70, 8, 8}, {16, 16, 1}};

  for (const Target &target : targets) {
    for (const auto &shape : shapes) {
      CompileResult result =
          compileSource(matmulSource(shape[0], shape[1], shape[2], "32"),
                        targetOptions(target.triple, target.cpu));
      EXPECT_TRUE(result.succeeded)
          << target.name << " " << shape[0] << "x" << shape[1] << "x"
          << shape[2] << ": " << result.diagnostics;
    }
  }
}

// The f64 tiles SME needs live behind FEAT_SME_F64F64, so a target without it
// has to fall back rather than emit an FMOPA the backend cannot select.
TEST(DriverTest, CompilesF64MatmulWithAndWithoutSMEF64) {
  std::string source = matmulSource(16, 16, 16, "64");
  for (const char *cpu : {"", "apple-m1"}) {
    CompileResult result = compileSource(source, targetOptions("", cpu));
    EXPECT_TRUE(result.succeeded)
        << "cpu='" << cpu << "': " << result.diagnostics;
  }
}

// Blocks of `function` that are reachable from themselves. Such a block runs
// more than once per frame, so an alloca in it extends the frame every time.
llvm::SmallPtrSet<const llvm::BasicBlock *, 8>
selfReachableBlocks(const llvm::Function &function) {
  llvm::SmallPtrSet<const llvm::BasicBlock *, 8> result;
  for (const llvm::BasicBlock &block : function) {
    llvm::SmallPtrSet<const llvm::BasicBlock *, 32> seen;
    llvm::SmallVector<const llvm::BasicBlock *, 32> worklist(
        llvm::succ_begin(&block), llvm::succ_end(&block));
    while (!worklist.empty()) {
      const llvm::BasicBlock *next = worklist.pop_back_val();
      if (!seen.insert(next).second)
        continue;
      worklist.append(llvm::succ_begin(next), llvm::succ_end(next));
    }
    if (seen.contains(&block))
      result.insert(&block);
  }
  return result;
}

// A boxed container mutation inside a loop must not leave its 16-word payload
// box slot in the loop body: `memref.alloca` outside the entry block becomes a
// dynamic LLVM stack adjustment that nothing reclaims before the function
// returns, so the frame grew 128 bytes per iteration and the stack guard raised
// RecursionError past ~25,000 iterations. Every alloca that survives in a
// self-reachable block here is a byte buffer feeding a raise (the
// changed-size guard, the traceback file/function names), which by construction
// runs at most once per frame; assert on element type rather than on a count so
// that a NEW non-i8 loop-body slot fails by name and so that removing the last
// one also fails and asks for this test to be tightened.
TEST(DriverTest, BoxedContainerLoopKeepsPayloadSlotsOutOfTheLoopBody) {
  CompileResult result = compileSource("d: dict[int, int] = {}\n"
                                       "for i in range(4):\n"
                                       "    d[i] = i\n"
                                       "acc = 0\n"
                                       "for k in d:\n"
                                       "    d[0] = k\n"
                                       "    acc = acc + 1\n"
                                       "print(acc)\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  const llvm::Function *main = result.verified.llvmModule->getFunction("__main__");
  ASSERT_NE(main, nullptr);

  llvm::SmallPtrSet<const llvm::BasicBlock *, 8> repeated =
      selfReachableBlocks(*main);
  ASSERT_FALSE(repeated.empty()) << "the loops were compiled away; this test "
                                    "would then assert nothing";
  for (const llvm::BasicBlock &block : *main) {
    if (!repeated.contains(&block))
      continue;
    for (const llvm::Instruction &instruction : block) {
      const auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction);
      if (!alloca)
        continue;
      std::string described;
      llvm::raw_string_ostream out(described);
      alloca->print(out);
      EXPECT_TRUE(alloca->getAllocatedType()->isIntegerTy(8))
          << "alloca in a block that repeats: " << described;
    }
  }
}

TEST(DriverTest, RepeatedCompileIsStable) {
  for (int round = 0; round < 3; ++round) {
    CompileResult result = compileSource("print(40 + 2)\n");
    EXPECT_TRUE(result.succeeded) << "round " << round << ": "
                                  << result.diagnostics;
  }
}

} // namespace
