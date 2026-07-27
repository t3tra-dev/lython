#include "Driver.h"

#include "embedded.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

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

// Names every alloca of `function` that sits in a block able to reach itself,
// each rendered as its own IR text so a failure reports the offending slot by
// name instead of a count that has moved.
std::vector<std::string> allocasInRepeatedBlocks(const llvm::Function &function,
                                                 bool &sawRepeatedBlock) {
  llvm::SmallPtrSet<const llvm::BasicBlock *, 8> repeated =
      selfReachableBlocks(function);
  sawRepeatedBlock = !repeated.empty();
  std::vector<std::string> found;
  for (const llvm::BasicBlock &block : function) {
    if (!repeated.contains(&block))
      continue;
    for (const llvm::Instruction &instruction : block) {
      if (!llvm::isa<llvm::AllocaInst>(&instruction))
        continue;
      std::string described;
      llvm::raw_string_ostream out(described);
      instruction.print(out);
      found.push_back(described);
    }
  }
  return found;
}

// Is `text` the initializer of some read-only global of `module`?
//
// This is the anti-vacuity half of the literal test below: "no alloca in the
// loop body" is also what a literal that was folded away entirely would produce,
// and that would be a different (and unnoticed) change. Finding the bytes in
// read-only data proves the literal still reaches the lowering under test.
bool hasConstantBytes(const llvm::Module &module, llvm::StringRef text) {
  for (const llvm::GlobalVariable &global : module.globals()) {
    if (!global.isConstant() || !global.hasInitializer())
      continue;
    const auto *data =
        llvm::dyn_cast<llvm::ConstantDataArray>(global.getInitializer());
    if (data && data->isString() && data->getRawDataValues() == text)
      return true;
  }
  return false;
}

// A boxed container mutation inside a loop must not leave its 16-word payload
// box slot in the loop body: `memref.alloca` outside the entry block becomes a
// dynamic LLVM stack adjustment that nothing reclaims before the function
// returns, so the frame grew 128 bytes per iteration and the stack guard raised
// RecursionError past ~25,000 iterations.
//
// The assertion is a set and not a count so that a NEW loop-body slot fails by
// name. It reads "none at all" rather than the "none except i8" it was first
// written as: the three i8 buffers that used to survive here were a raise
// message and the traceback file/function names, all of them compile-time
// literals, and those are now shared read-only globals.
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

  bool sawRepeatedBlock = false;
  std::vector<std::string> found = allocasInRepeatedBlocks(*main,
                                                           sawRepeatedBlock);
  ASSERT_TRUE(sawRepeatedBlock) << "the loops were compiled away; this test "
                                  "would then assert nothing";
  for (const std::string &described : found)
    ADD_FAILURE() << "alloca in a block that repeats:" << described;
}

// A `str` or `bytes` literal in a loop body must not put its bytes on the
// frame. The buffer feeding `builtins.str.__new__` / `builtins.bytes.__new__` was
// a `memref.alloca` plus one store per byte, so the frame grew by the length of
// the literal on every iteration -- measured at 275,000 iterations of a 20-byte
// literal before RecursionError, against 4,000,000 for the same loop with an
// `int` literal in place of the `str` one.
//
// It is a shared read-only global instead of a hoisted or reused frame slot
// because the buffer is not the object's payload: both initializers allocate
// their own payload and copy out of it, so two occurrences of one literal can
// share storage and nothing can write through it.
TEST(DriverTest, StringAndBytesLiteralsInALoopStayOutOfTheFrame) {
  CompileResult result = compileSource("n = 0\n"
                                       "for i in range(4):\n"
                                       "    s = \"loop body literal\"\n"
                                       "    b = b\"loop body bytes\"\n"
                                       "    n = n + len(s) + len(b)\n"
                                       "print(n)\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  const llvm::Function *main = result.verified.llvmModule->getFunction("__main__");
  ASSERT_NE(main, nullptr);

  bool sawRepeatedBlock = false;
  std::vector<std::string> found = allocasInRepeatedBlocks(*main,
                                                           sawRepeatedBlock);
  ASSERT_TRUE(sawRepeatedBlock) << "the loop was compiled away; this test would "
                                  "then assert nothing";
  for (const std::string &described : found)
    ADD_FAILURE() << "alloca in a block that repeats:" << described;

  EXPECT_TRUE(hasConstantBytes(*result.verified.llvmModule, "loop body literal"))
      << "the str literal's bytes are in neither the frame nor read-only data, "
         "so this test is no longer looking at the lowering it was written for";
  EXPECT_TRUE(hasConstantBytes(*result.verified.llvmModule, "loop body bytes"))
      << "the bytes literal's bytes are in neither the frame nor read-only "
         "data, so this test is no longer looking at the lowering it was "
         "written for";
}

// The `int` arm of the same class. `lowerIntConstant` splits a beyond-i64 literal
// into 30-bit limbs at compile time, and the limbs used to be stored into a
// per-execution `memref.alloca<?xi32>` -- 4 bytes of frame per limb per iteration,
// RecursionError past 300,000.
//
// It is a separate test from the str/bytes one because it was a separate find:
// both literals grew the same frame, the 20-byte `str` reached the cliff at
// 275,000 and a 7-limb `int` needs 300,000, so this instance was invisible until
// the other was fixed. Two tests keep that distinction reportable.
TEST(DriverTest, BigIntLiteralInALoopStaysOutOfTheFrame) {
  CompileResult result = compileSource(
      "n = 0\n"
      "for i in range(4):\n"
      "    big = 123456789012345678901234567890123456789012345678901234567890\n"
      "    n = n + (big % 97)\n"
      "print(n)\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  const llvm::Function *main = result.verified.llvmModule->getFunction("__main__");
  ASSERT_NE(main, nullptr);

  bool sawRepeatedBlock = false;
  std::vector<std::string> found = allocasInRepeatedBlocks(*main,
                                                           sawRepeatedBlock);
  ASSERT_TRUE(sawRepeatedBlock) << "the loop was compiled away; this test would "
                                  "then assert nothing";
  for (const std::string &described : found)
    ADD_FAILURE() << "alloca in a block that repeats:" << described;

  // Anti-vacuity: the literal needs 2 limbs beyond i64 and must still be lowered
  // through the digit path, not folded to something with no block at all. 2^30
  // per limb, so 60 decimal digits is 7 limbs -- assert the limb block exists as
  // read-only data rather than asserting its exact contents, which would restate
  // the limb split rather than check it (the golden checks the values).
  bool sawLimbBlock = false;
  for (const llvm::GlobalVariable &global :
       result.verified.llvmModule->globals()) {
    if (!global.isConstant() || !global.hasInitializer())
      continue;
    if (!global.getName().starts_with("__ly_const_digits_"))
      continue;
    sawLimbBlock = true;
    break;
  }
  EXPECT_TRUE(sawLimbBlock)
      << "no read-only limb block, so the beyond-i64 literal no longer reaches "
         "the lowering this test was written for";
}

// Reading a bool out of an erased slot yields the box, and bool.__str__ takes
// the unboxed i1: the operand adapter has to unbox. It grew arms for i64 and
// f64 but not i1, so `str()` and single-argument `print()` over a boxed bool
// failed to lower. Driver-level rather than golden: what regressed was
// lowering, and the printed value is already pinned by the many cases that
// stringify an unboxed bool.
TEST(DriverTest, AdaptsBoxedBoolToUnboxedStrInput) {
  for (const char *source :
       {"t: tuple = (\"s\", True)\nprint(t[1])\n",
        "t: tuple = (\"s\", True)\nprint(str(t[1]), \"x\")\n"}) {
    CompileResult result = compileSource(source);
    EXPECT_TRUE(result.succeeded) << source << ": " << result.diagnostics;
  }
}

// Follow a released value back to the call that produced it, past the
// extractvalue chain that unpacks a multi-result runtime call.
const llvm::CallBase *definingCallOf(const llvm::Value *value) {
  while (const auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(value))
    value = extract->getAggregateOperand();
  return llvm::dyn_cast<llvm::CallBase>(value);
}

llvm::StringRef calleeNameOf(const llvm::CallBase &call) {
  const llvm::Function *callee = call.getCalledFunction();
  return callee ? callee->getName() : llvm::StringRef{};
}

// An owned result must be released by ITS OWN contract's deallocator, not by the
// deallocator of the contract whose method produced it.  `int.__repr__` returns a
// `builtins.str`, and before `ly.runtime.result_contract` was consulted when the
// owned-result group is formed, the string was released through `LyLong_DecRef`
// -- accepted only because every width-2 release body was byte-identical.
TEST(DriverTest, IntReprStringIsReleasedByStrDeallocator) {
  CompileResult result = compileSource("big = 2 ** 90 + 12345\n"
                                       "print(repr(big))\n"
                                       "print(str(big))\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  const llvm::Function *main = result.verified.llvmModule->getFunction("__main__");
  ASSERT_NE(main, nullptr);

  unsigned reprResultsReleased = 0;
  for (const llvm::BasicBlock &block : *main) {
    for (const llvm::Instruction &instruction : block) {
      const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (!call)
        continue;
      llvm::StringRef releaseName = calleeNameOf(*call);
      if (!releaseName.ends_with("_DecRef") || call->arg_empty())
        continue;
      const llvm::CallBase *producer = definingCallOf(call->getArgOperand(0));
      if (!producer)
        continue;
      llvm::StringRef producerName = calleeNameOf(*producer);
      if (producerName != "LyLong_Repr" && producerName != "LyLong_Str")
        continue;
      ++reprResultsReleased;
      EXPECT_EQ(releaseName, "LyUnicode_DecRef")
          << "the str returned by " << producerName.str()
          << " is released through " << releaseName.str();
    }
  }
  // Without this the test passes when nothing matched, which is the shape the
  // defect itself has: no group, so no release to inspect.
  ASSERT_GT(reprResultsReleased, 0u)
      << "no release of an int-to-str result was found, so this test asserted "
         "nothing";
}

TEST(DriverTest, RepeatedCompileIsStable) {
  for (int round = 0; round < 3; ++round) {
    CompileResult result = compileSource("print(40 + 2)\n");
    EXPECT_TRUE(result.succeeded) << "round " << round << ": "
                                  << result.diagnostics;
  }
}

} // namespace
