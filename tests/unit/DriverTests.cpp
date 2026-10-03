#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

#include "Driver.h"
#include "DriverCodeGen.h"

#include "Common/RuntimeLibrary.h"
#include "Common/LibcPrototypes.h"
#include "Common/RuntimeSupport.h"
#include "Common/SupportBuilder.h"
#include "Runtime/ABI/BoxLayout.h"

#include "embedded.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "Common/UnwindABI.h"
#include "PlatformConstants.h"

#include "llvm/IR/Instructions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>
#include <memory>
#include <optional>

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
  // A refusal raised by a PASS reaches the context's engine, not the driver's
  // stream, so without this handler `diagnostics` holds only what the frontend
  // wrote and a lowering refusal can be asserted on nothing but its exit
  // status -- which any other failure also produces.
  mlir::ScopedDiagnosticHandler capture(
      &context, [&](mlir::Diagnostic &diagnostic) {
        diag << diagnostic.str() << "\n";
        return mlir::failure(); // let the default handler still print it
      });
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

// A Python `def main()` still links as an executable.
//
// The AOT entry point installs a C `main`, and the user's function is lowered
// under its Python name, so the two collided and the driver refused the program
// with "symbol 'main' already exists". `def main()` is the single most ordinary
// function name in Python, and it compiled under JIT the whole time -- the two
// output modes disagreed on a valid program.
//
// Why here and not in the leak gate (the only other stage that links AOT): that
// gate reports an unbuildable subject as "could not measure", which ctest maps
// to SKIP. A regression of this would turn it green-by-omission rather than red.
TEST(DriverTest, InstallsAOTEntryPointBesideAPythonMain) {
  CompileResult result = compileSource("def main() -> None:\n"
                                       "    print(\"hi\")\n"
                                       "\n"
                                       "main()\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  ASSERT_TRUE(result.verified.llvmModule);
  llvm::Function *pythonMain = result.verified.llvmModule->getFunction("main");
  ASSERT_NE(pythonMain, nullptr);
  ASSERT_FALSE(pythonMain->isDeclaration());

  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  ASSERT_TRUE(mlir::succeeded(lython::driver::installAOTEntryPoint(
      *result.verified.llvmModule, diag)))
      << diagnostics;

  // The C entry is the one now named `main`, and it is the (i32, ptr) -> i32
  // one the linker needs -- not the Python function that used to hold the name.
  llvm::Function *entry = result.verified.llvmModule->getFunction("main");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->arg_size(), 2u);
  EXPECT_TRUE(entry->getReturnType()->isIntegerTy(32));
  EXPECT_NE(entry, pythonMain);
  // The Python function is still in the module, still called: renaming moved
  // the symbol, it did not drop the definition.
  EXPECT_FALSE(pythonMain->isDeclaration());
  EXPECT_FALSE(pythonMain->use_empty());
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

// A generator built over an object argument RETAINS that argument into its
// frame slot, so the creating function still holds the handle it started with
// and has to release it. Building the generator therefore produces two
// references (the constructor's and the aggregate retain's) against two
// obligations: the frame's, discharged by the drop finalizer, and the
// creator's, discharged here.
//
// This assertion cannot be written as a golden: the program exits 0 and prints
// the right answer whether or not the second release exists. What it costs is
// one range per generator built -- 1 root / 64 B per iteration under
// `leaks --atExit`, linear through 40000 iterations with no saturation.
//
// Counting releases of anything (rather than of the value LyRange_New
// produced) would be satisfied by the drop finalizer's own release, which is a
// different obligation in a different function -- so the search is scoped to
// the function that builds the generator, and refuses to pass if it did not
// find one.
TEST(DriverTest, GeneratorObjectArgumentIsReleasedByItsCreatorToo) {
  CompileResult result = compileSource("def f(n: int) -> int:\n"
                                       "    total = 0\n"
                                       "    i = 0\n"
                                       "    while i < n:\n"
                                       "        x = range(3)\n"
                                       "        it = iter(x)\n"
                                       "        total += next(it)\n"
                                       "        i += 1\n"
                                       "    return total\n"
                                       "print(f(4))\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  ASSERT_TRUE(result.verified.llvmModule);

  // Located by what it does, not by its name: the lowering is free to rename
  // the resume clones and the driver.
  const llvm::Function *creator = nullptr;
  unsigned creatorCount = 0;
  for (const llvm::Function &function : *result.verified.llvmModule) {
    bool buildsRange = false;
    bool buildsGenerator = false;
    for (const llvm::BasicBlock &block : function) {
      for (const llvm::Instruction &instruction : block) {
        const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        if (!call)
          continue;
        llvm::StringRef name = calleeNameOf(*call);
        buildsRange |= name == "LyRange_New";
        buildsGenerator |= name == "LyGenerator_New";
      }
    }
    if (buildsRange && buildsGenerator) {
      creator = &function;
      ++creatorCount;
    }
  }
  // Without this the test passes when the shape stopped being generated at
  // all, which is how a predicate quietly stops predicating anything.
  ASSERT_NE(creator, nullptr)
      << "no function builds both a range and a generator, so this test "
         "asserted nothing about the shape it is named for";
  ASSERT_EQ(creatorCount, 1u) << "expected exactly one generator creation site";

  unsigned rangesBuilt = 0;
  unsigned rangesRetained = 0;
  unsigned rangesReleased = 0;
  for (const llvm::BasicBlock &block : *creator) {
    for (const llvm::Instruction &instruction : block) {
      const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (!call)
        continue;
      llvm::StringRef name = calleeNameOf(*call);
      if (name == "LyRange_New")
        ++rangesBuilt;
      else if (name == "Ly_IncRef")
        ++rangesRetained;
      else if (name == "LyRange_DecRef")
        ++rangesReleased;
    }
  }
  EXPECT_EQ(rangesBuilt, 1u);
  // Not EQ: Ly_IncRef is contract-agnostic, so an unrelated retain appearing
  // in this function later would make an exact count fail for no reason. What
  // has to hold is that the frame's reference came from a retain at all.
  EXPECT_GE(rangesRetained, 1u)
      << "the frame slot's reference should come from a retain";
  // The creator produced two references (constructor + retain) and hands one
  // obligation to the frame, so exactly one release belongs here. Zero is the
  // leak this test exists for; two would be the mirror defect.
  EXPECT_EQ(rangesReleased, 1u)
      << "the generator's creator built " << rangesBuilt << " range(s) and "
      << "retained " << rangesRetained
      << " into the frame, but released " << rangesReleased
      << " -- the frame's retain and the creator's own handle are two "
         "references, and the drop finalizer discharges only one of them";
}

// The exception chain is walked with pointers, start to finish.
//
// A raise that interrupts the handling of another exception parks it in a heap
// node. The node was 21 untyped i64 words and its address was a word too, so
// the payload's three ALIGNED POINTERS were stored as integers, the links
// between nodes were integers, and every reader turned them back with an
// `inttoptr` before it could use them.
//
// The memory model documents `extract_aligned_pointer_as_index` as where
// provenance is lost and says what comes back through an integer is outside it
// by its own statement -- so those readers were building descriptors, and
// following links, that no judgment in `proof/` covers.
//
// The assertion is not a count: it is that these functions contain NO
// integer-to-pointer conversion at all. They allocate nothing and receive the
// node they work on, so there is no honest reason for one, and any that appears
// means a slot went back to holding a word.
//
// Not asserted here, because they are a different slot's problem: the star
// frame and the generator stash area still hold node addresses as words, so
// `LyEH_StarResidualParts` and `LyEH_UnstashException` each still widen one.
TEST(DriverTest, TheExceptionChainIsWalkedWithPointers) {
  CompileResult result = compileSource("try:\n"
                                       "    raise ValueError(\"outer\")\n"
                                       "except ValueError as e:\n"
                                       "    try:\n"
                                       "        raise KeyError(\"inner\")\n"
                                       "    except KeyError as k:\n"
                                       "        print(k)\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  // The EH runtime is a separate module until this link, exactly as in
  // `--emit-llvm` and the JIT; the link picks its support module by triple, so
  // the target has to be settled first.
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  ASSERT_TRUE(mlir::succeeded(lython::driver::configureLLVMModuleCodeGenTarget(
      *result.verified.llvmModule,
      lython::driver::detectTensorLoweringTarget(
          lython::driver::DriverOptions{}),
      lython::driver::DriverOptions{}, diag)))
      << diagnostics;
  ASSERT_TRUE(mlir::succeeded(py::runtime_library::linkEmbeddedNativeRuntime(
      *result.verified.llvmModule)));

  // Destruction, the traceback report, the except* frame's residual drop, and
  // the two that move the chain in and out of the process slot.
  for (const char *name :
       {"release_chain_node", "print_chain_node", "release_star_node",
        "LyEH_DiscardCurrentException", "LyEH_SetCurrentCause"}) {
    const llvm::Function *fn = result.verified.llvmModule->getFunction(name);
    ASSERT_NE(fn, nullptr)
        << name << " is gone; this test no longer looks at anything";
    for (const llvm::BasicBlock &block : *fn)
      for (const llvm::Instruction &instruction : block) {
        if (!llvm::isa<llvm::IntToPtrInst>(&instruction))
          continue;
        std::string described;
        llvm::raw_string_ostream(described) << instruction;
        ADD_FAILURE() << name << " makes a pointer out of an integer:"
                      << described;
      }
  }
}

// Every call into the EH runtime matches the definition it reaches.
//
// The exception triple crosses this boundary as three memrefs, and the two
// sides are verified as MLIR SEPARATELY -- the call sites in the lowering pass
// and the manifests, the definitions in the runtime support builder -- then
// meet only after both are LLVM IR. Nothing there compares them: checked with
// `opt -passes=verify`, which accepts a four-argument call to a two-parameter
// definition and exits 0. A drift would link, run, and read its arguments off
// the wrong registers.
//
// The types now come from one place (`Common/ExceptionABI.h`), which is what
// makes a drift unlikely; this is what makes it visible. Both were needed --
// before, the definitions hand-transcribed MLIR's descriptor layout, so the
// two sides could disagree without either being edited, just by MLIR changing
// what a memref lowers to.
TEST(DriverTest, EveryCallIntoTheEHRuntimeMatchesItsDefinition) {
  CompileResult result = compileSource("try:\n"
                                       "    raise ValueError(\"boom\")\n"
                                       "except* ValueError as eg:\n"
                                       "    print(len(eg.exceptions))\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  ASSERT_TRUE(mlir::succeeded(lython::driver::configureLLVMModuleCodeGenTarget(
      *result.verified.llvmModule,
      lython::driver::detectTensorLoweringTarget(
          lython::driver::DriverOptions{}),
      lython::driver::DriverOptions{}, diag)))
      << diagnostics;
  ASSERT_TRUE(mlir::succeeded(py::runtime_library::linkEmbeddedNativeRuntime(
      *result.verified.llvmModule)));

  // The functions that carry an exception across the boundary. Named rather
  // than discovered by prefix: a `LyEH_` symbol that stops being reachable
  // should fail here, not quietly drop out of the check.
  const char *carriers[] = {
      "LyEH_ThrowException",    "LyEH_BorrowCurrentException",
      "LyEH_StarResidualParts", "LyEH_StarApplyMatch",
      "LyEH_StarThrowCombined", "LyEH_StarDiscardSplit"};
  for (const char *name : carriers) {
    llvm::Function *fn = result.verified.llvmModule->getFunction(name);
    ASSERT_NE(fn, nullptr) << name << " is not in the linked module";
    EXPECT_FALSE(fn->isDeclaration())
        << name << " was never defined -- the runtime support module and the "
                   "call sites disagree on its name";
    for (const llvm::User *user : fn->users()) {
      const auto *call = llvm::dyn_cast<llvm::CallBase>(user);
      // ⛔ `getCalledOperand()`, NOT `getCalledFunction()`. The latter returns
      // null precisely when the call's signature disagrees with the callee's,
      // which is the case this test exists for -- filtering on it skips the
      // defect and passes. (Observed: 5 users, 0 of them "calls to fn".)
      if (!call || call->getCalledOperand() != fn)
        continue;
      EXPECT_EQ(call->getFunctionType(), fn->getFunctionType())
          << name << ": a call site's signature is not the definition's, so "
                     "the two sides read different arguments";
    }
  }
}

// A module global's pointer cell holds a pointer.
//
// A module-level object is parked in one i64 cell per stored word: a bound
// flag, then a pointer and a size per physical memref. The pointer cell held
// an INTEGER -- the store side reached it with
// `memref.extract_aligned_pointer_as_index`, which the memory model documents
// as where provenance is lost, and the read side widened it back through
// `__ly_global_view_*`. Round-tripping an owning reference through an integer
// on every read of a module-level list, dict or str.
//
// `__ly_global_view_*` itself stays: it exists so a MANIFEST body can obtain a
// descriptor through a call rather than a cast, which this pipeline rejects in
// its input. What changed is that the compiler's own path no longer needs it --
// it holds a pointer, so it builds the view where it stands.
TEST(DriverTest, AModuleGlobalsPointerCellHoldsAPointer) {
  CompileResult result = compileSource("LABEL: str = \"module scope\"\n"
                                       "\n"
                                       "def read_global() -> int:\n"
                                       "    return len(LABEL)\n"
                                       "\n"
                                       "print(read_global())\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;

  unsigned pointerCells = 0;
  for (const llvm::GlobalVariable &cell :
       result.verified.llvmModule->globals()) {
    llvm::StringRef name = cell.getName();
    if (!name.starts_with("__ly_module_global_obj_"))
      continue;
    // `_p<i>`; the `_s<i>` sizes and the `_init` flag are genuinely words.
    llvm::StringRef slot = name.rsplit('_').second;
    if (!slot.starts_with("p"))
      continue;
    ++pointerCells;
    if (cell.getValueType()->isPointerTy())
      continue;
    ADD_FAILURE() << name.str()
                  << " does not hold a pointer, so every read of this global "
                     "widens an address back into one";
  }
  // A str is two physical memrefs, so the program has two of these. Asserted
  // so that a lowering change which stopped emitting cells at all would fail
  // here rather than pass with nothing to check.
  EXPECT_EQ(pointerCells, 2u);

  const llvm::Function *reader =
      result.verified.llvmModule->getFunction("read_global");
  ASSERT_NE(reader, nullptr);
  for (const llvm::BasicBlock &block : *reader)
    for (const llvm::Instruction &instruction : block) {
      const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (!call || !call->getCalledOperand())
        continue;
      llvm::StringRef callee = call->getCalledOperand()->getName();
      if (!callee.starts_with("__ly_global_view_"))
        continue;
      ADD_FAILURE() << "reading a module global still goes through "
                    << callee.str()
                    << ", which takes the payload's address as a word";
    }
}

// A parked exception is reached by pointer, not by address.
//
// The stash cell is one slot holding a chain node while its owner is not
// running: a suspended generator's in-flight token, and an except* frame's
// residual and one per clause body that raised. It held the node's ADDRESS,
// and the cell's own address was passed as one too -- the generator side
// reached it with `memref.extract_aligned_pointer_as_index`, which the memory
// model documents as where provenance is lost.
//
// It holds a pointer now, and that is possible even though a generator's cell
// lives inside a `memref<?xi64>` (a memref cannot have a pointer element type
// -- see BoxLayout.h). Three functions own the cell and nothing else reads or
// writes one, so nothing goes through the memref: callers hand over the cell's
// ADDRESS, which the descriptor's aligned member supplies as a pointer.
//
// The except* frame is a `!py.except_star_frame` rather than an `i64`, so its
// eleven entry points have no integer to widen either. A dialect type saying
// "number" about an identity is what forced that, and there was no way to fix
// it below the dialect.
TEST(DriverTest, AParkedExceptionIsReachedByPointer) {
  CompileResult result = compileSource("def gen() -> object:\n"
                                       "    try:\n"
                                       "        yield 1\n"
                                       "    finally:\n"
                                       "        pass\n"
                                       "\n"
                                       "for v in gen():\n"
                                       "    print(v)\n"
                                       "\n"
                                       "try:\n"
                                       "    raise ValueError(\"boom\")\n"
                                       "except* ValueError as eg:\n"
                                       "    print(len(eg.exceptions))\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  ASSERT_TRUE(mlir::succeeded(lython::driver::configureLLVMModuleCodeGenTarget(
      *result.verified.llvmModule,
      lython::driver::detectTensorLoweringTarget(
          lython::driver::DriverOptions{}),
      lython::driver::DriverOptions{}, diag)))
      << diagnostics;
  ASSERT_TRUE(mlir::succeeded(py::runtime_library::linkEmbeddedNativeRuntime(
      *result.verified.llvmModule)));

  // The three that own a cell. They receive its address and touch nothing
  // else that is not already a pointer, so there is no honest reason for a
  // conversion in any of them.
  for (const char *name :
       {"LyEH_StashCurrentException", "LyEH_UnstashException",
        "LyEH_AdoptStashedAsContext"}) {
    const llvm::Function *fn = result.verified.llvmModule->getFunction(name);
    ASSERT_NE(fn, nullptr) << name << " is gone";
    for (const llvm::BasicBlock &block : *fn)
      for (const llvm::Instruction &instruction : block) {
        if (!llvm::isa<llvm::IntToPtrInst>(&instruction))
          continue;
        std::string described;
        llvm::raw_string_ostream(described) << instruction;
        ADD_FAILURE() << name << " makes a pointer out of an integer:"
                      << described;
      }
  }

  // And the except* surface, which takes the frame. It is a
  // `!py.except_star_frame` in the dialect and an `!llvm.ptr` after lowering,
  // so nothing here has an integer to widen either -- these were one apiece
  // for as long as the dialect said the frame was an `i64`.
  for (const char *name :
       {"LyEH_StarBegin", "LyEH_StarHasResidual", "LyEH_StarCollect",
        "LyEH_StarCollectedCount", "LyEH_StarNodesPtr",
        "LyEH_StarResidualParts", "LyEH_StarApplyMatch",
        "LyEH_StarThrowCombined", "LyEH_StarDiscardSplit", "LyEH_StarPop",
        "LyEH_StarRethrowResidual", "LyEH_StarRethrowSoleCollected",
        "release_star_node", "__ly_exc_star_combine"}) {
    const llvm::Function *fn = result.verified.llvmModule->getFunction(name);
    ASSERT_NE(fn, nullptr) << name << " is gone";
    for (const llvm::BasicBlock &block : *fn)
      for (const llvm::Instruction &instruction : block) {
        if (!llvm::isa<llvm::IntToPtrInst>(&instruction))
          continue;
        std::string described;
        llvm::raw_string_ostream(described) << instruction;
        ADD_FAILURE() << name << " makes a pointer out of an integer:"
                      << described;
      }
  }
}

// The word offsets builtins.mlir reads are the ones the C++ structs have.
//
// A manifest body cannot name a C++ struct, so where one reaches into a
// runtime structure it counts words: `__ly_exc_star_combine` reads a parked
// chain node at words 2, 7, 12 and 14, and the payload-box helpers stride by
// 16 and index from 4 and 9. Those numbers are a second copy of a layout whose
// first copy is a `LLVMStructType` in the support builder, and nothing joined
// them.
//
// It has already come close. The chain node was 21 untyped words until
// recently and is a struct now; the manifest kept working only because that
// change preserved every offset, which was intent and not a guarantee. A
// reordering does not fail to build -- both sides compile, link, and read
// different fields.
//
// So: compute the offsets from the type the compiler actually emits, and
// compare them against the numbers the manifest is written around. The
// duplication is the point -- a check restates the contract, which is what
// makes it a check.
TEST(DriverTest, ManifestWordOffsetsMatchTheRuntimeStructs) {
  CompileResult result = compileSource("try:\n"
                                       "    raise ValueError(\"boom\")\n"
                                       "except* ValueError as eg:\n"
                                       "    print(len(eg.exceptions))\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  ASSERT_TRUE(mlir::succeeded(lython::driver::configureLLVMModuleCodeGenTarget(
      *result.verified.llvmModule,
      lython::driver::detectTensorLoweringTarget(
          lython::driver::DriverOptions{}),
      lython::driver::DriverOptions{}, diag)))
      << diagnostics;
  ASSERT_TRUE(mlir::succeeded(py::runtime_library::linkEmbeddedNativeRuntime(
      *result.verified.llvmModule)));

  const llvm::DataLayout &layout = result.verified.llvmModule->getDataLayout();
  auto *node = llvm::StructType::getTypeByName(
      result.verified.llvmModule->getContext(), "ExceptionChainNode");
  ASSERT_NE(node, nullptr)
      << "the chain node type is gone; builtins.mlir still reads its words";

  // node -> payload -> section -> field, in bytes. The member index comes from
  // the enum rather than a literal, so that reordering the struct reports a
  // WORD mismatch -- the thing the manifest cares about -- instead of failing
  // to find a struct where it expected one.
  const unsigned payloadMember = py::runtime_library::kNodePayload;
  auto *parts = llvm::dyn_cast<llvm::StructType>(
      node->getElementType(payloadMember));
  ASSERT_NE(parts, nullptr)
      << "member " << payloadMember
      << " of the chain node is not the payload any more, and builtins.mlir "
         "still reads the payload's fields by word";
  auto fieldWord = [&](unsigned section, unsigned field) -> std::uint64_t {
    std::uint64_t offset =
        layout.getStructLayout(node)->getElementOffset(payloadMember);
    offset += layout.getStructLayout(parts)->getElementOffset(section);
    auto *view = llvm::cast<llvm::StructType>(parts->getElementType(section));
    offset += layout.getStructLayout(view)->getElementOffset(field);
    return offset / 8;
  };

  struct Read {
    const char *what;
    unsigned section;
    unsigned field;
    std::uint64_t word;
  };
  // Field 1 is the descriptor's aligned pointer, field 3 its size.
  const Read reads[] = {
      {"the exception object", 0, 1, 2},
      {"the message header", 1, 1, 7},
      {"the message bytes", 2, 1, 12},
      {"the message length", 2, 3, 14},
  };
  for (const Read &read : reads)
    EXPECT_EQ(fieldWord(read.section, read.field), read.word)
        << "__ly_exc_star_combine reads " << read.what << " at word "
        << read.word << " of a chain node, and the struct now puts it at word "
        << fieldWord(read.section, read.field);

  // ⭐ THE PAYLOAD BOX IS READ OUT OF THE MANIFEST, not restated here. What
  // stood in this place compared the C++ constants against literals -- which
  // says nothing about the manifest, and the manifest is the half that indexes
  // box words. `builtins.mlir` now states the layout once, in the
  // `__ly_box_*_word` helpers, and this reads their constants back.
  //
  // Why it matters that this is mechanical: narrowing the box is a type change
  // the verifier checks and an ARITHMETIC change it does not, so a store to the
  // wrong word of a right-sized box compiles and corrupts a refcount at run
  // time. A first attempt at the narrowing missed several sites; this test is
  // what would have named them.
  {
    std::ifstream manifest(LYTHON_SOURCE_DIR
                           "/src/lython/runtime/modules/builtins.mlir");
    ASSERT_TRUE(manifest.good()) << "cannot read builtins.mlir";
    std::stringstream buffer;
    buffer << manifest.rdbuf();
    const std::string text = buffer.str();

    auto constantIn = [&](const char *helper) -> std::int64_t {
      const std::string needle =
          std::string("func.func private @") + helper + "(";
      std::size_t at = text.find(needle);
      EXPECT_NE(at, std::string::npos)
          << helper << " is the manifest's only spelling of that offset and it "
          << "is gone";
      if (at == std::string::npos)
        return -1;
      std::size_t constant = text.find("arith.constant ", at);
      if (constant == std::string::npos)
        return -1;
      return std::strtoll(text.c_str() + constant + std::strlen("arith.constant "),
                          nullptr, 10);
    };

    EXPECT_EQ(constantIn("__ly_box_word_count"),
              py::lowering::box_abi::kWordsPerBox)
        << "the manifest strides slots by a different box width than "
           "ABI/BoxLayout.h";
    EXPECT_EQ(constantIn("__ly_box_entity_word"),
              py::lowering::box_abi::kEntityWord)
        << "the manifest reads the one address a box holds from elsewhere";
    EXPECT_EQ(constantIn("__ly_box_owned_word"),
              py::lowering::box_abi::kOwnedFlagWord)
        << "the manifest writes the owned flag elsewhere";
    EXPECT_EQ(constantIn("__ly_box_hash_word"),
              py::lowering::box_abi::kHashWord)
        << "the manifest caches the hash in a different word";
    EXPECT_EQ(py::lowering::box_abi::kHashWord,
              py::lowering::box_abi::kWordsPerBox - 1)
        << "the cached hash is the box's last word";

    // ⭐ AND NO FUNCTION MAY STRIDE BY A LITERAL AGAIN. The helpers are only
    // worth having if nothing goes around them, and a stride that does is
    // invisible to every other check: it is arithmetic on a correctly typed
    // memref. This splits the manifest into functions, collects the names each
    // one binds to the box width, and fails if any of them reaches a multiply
    // -- which is what a slot stride is.
    //
    // ⛔ The name has to be resolved per FUNCTION. `%c16` is bound in dozens of
    // them, and a check that looked for the literal on the multiply's own line
    // found nothing at all: the literal is on the binding, one line up. That
    // version passed with a stride put back by hand, which is the only reason
    // this one is written out.
    //
    // Two exemptions, and neither is a box. `LyBytes_FromHex` multiplies an
    // accumulator by sixteen per hex digit, and `%probe_scale` is the 5 in
    // CPython's `i*5 + 1 + perturb` open-addressing walk -- which the box width
    // happens to equal, so the exemption is the NAME rather than the functions,
    // and a stride that spelled itself any other way still fails.
    {
      const std::string width =
          std::to_string(py::lowering::box_abi::kWordsPerBox);
      const std::string bindIndex = " = arith.constant " + width + " : index";
      const std::string bindI64 = " = arith.constant " + width + " : i64";
      std::size_t at = 0;
      while (at < text.size()) {
        std::size_t start = text.find("  func.func ", at);
        if (start == std::string::npos)
          break;
        std::size_t stop = text.find("\n  }\n", start);
        if (stop == std::string::npos)
          stop = text.size();
        const std::string body = text.substr(start, stop - start);
        at = stop + 1;
        std::size_t sym = body.find('@');
        std::size_t open = body.find('(', sym);
        const std::string name =
            (sym == std::string::npos || open == std::string::npos)
                ? std::string("<unnamed>")
                : body.substr(sym + 1, open - sym - 1);
        if (name.rfind("__ly_box_", 0) == 0 || name == "LyBytes_FromHex")
          continue;
        for (const std::string &bind : {bindIndex, bindI64}) {
          std::size_t declared = 0;
          while ((declared = body.find(bind, declared)) != std::string::npos) {
            std::size_t nameStart = body.rfind('%', declared);
            const std::string bound =
                body.substr(nameStart, declared - nameStart);
            declared += bind.size();
            if (bound.empty() || bound == "%probe_scale")
              continue;
            std::size_t use = 0;
            while ((use = body.find("arith.muli ", use)) != std::string::npos) {
              std::size_t eol = body.find('\n', use);
              const std::string line = body.substr(use, eol - use);
              use = eol == std::string::npos ? body.size() : eol;
              if (line.find(bound + " ") != std::string::npos ||
                  line.find(bound + ",") != std::string::npos)
                ADD_FAILURE()
                    << name << " multiplies by " << bound
                    << ", a literal box width; strides go through "
                       "__ly_box_slot_base";
            }
          }
        }
      }
    }

    // ⭐ AND NO FUNCTION MAY COPY A BOX A LITERAL NUMBER OF WORDS AT A TIME.
    // The multiply check above only knows the CURRENT width, so a copy loop
    // left at the previous one is invisible to it -- which is exactly what
    // happened: `LyList_SetSlice` strode by `__ly_box_word_count()` and then
    // copied `%c16` words per slot, so every box after the first landed four
    // words short and the next element's refcount word took the tail. It reads
    // correctly, it type-checks, and the program it breaks is not the one that
    // ran the copy: `b[::2] = ...` printed an `<object object>` where an int
    // had been, and `("a", "b", "c")` aborted in `Ly_DecRef` about a third of
    // the time, depending on what the pool handed out.
    //
    // ⛔ The bound is what this looks at and not the loop body, because the
    // body is correct in every one of these: load a word, store a word. The
    // literal is the whole defect, and it is on the binding line.
    //
    // `__ly_set_raw_swap_bodies` is the one exemption and it is not a box: it
    // swaps words 2..8 of two SET HANDLES, whose width is the set's.
    {
      std::size_t at = 0;
      while (at < text.size()) {
        std::size_t start = text.find("  func.func ", at);
        if (start == std::string::npos)
          break;
        std::size_t stop = text.find("\n  }\n", start);
        if (stop == std::string::npos)
          stop = text.size();
        const std::string body = text.substr(start, stop - start);
        at = stop + 1;
        std::size_t sym = body.find('@');
        std::size_t open = body.find('(', sym);
        const std::string name =
            (sym == std::string::npos || open == std::string::npos)
                ? std::string("<unnamed>")
                : body.substr(sym + 1, open - sym - 1);
        if (name.rfind("__ly_box_", 0) == 0 ||
            name == "__ly_set_raw_swap_bodies")
          continue;
        if (body.find("memref.store") == std::string::npos ||
            body.find("memref<?xi64>") == std::string::npos)
          continue;
        for (int words = 4; words <= 64; ++words) {
          const std::string bind =
              " = arith.constant " + std::to_string(words) + " : index";
          std::size_t declared = 0;
          while ((declared = body.find(bind, declared)) != std::string::npos) {
            std::size_t nameStart = body.rfind('%', declared);
            const std::string bound =
                body.substr(nameStart, declared - nameStart);
            declared += bind.size();
            if (bound.empty())
              continue;
            std::size_t use = 0;
            while ((use = body.find("scf.for ", use)) != std::string::npos) {
              std::size_t eol = body.find('\n', use);
              const std::string line = body.substr(use, eol - use);
              use = eol == std::string::npos ? body.size() : eol;
              if (line.find(" to " + bound + " step") != std::string::npos)
                ADD_FAILURE()
                    << name << " walks " << bound
                    << " words per box; the count comes from "
                       "__ly_box_word_count";
            }
          }
        }
      }
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

// Every landing pad is a pure cleanup, a catch-ALL, or a list of Python class
// ids -- and `LyEH_Personality` reads it as exactly that. A clause naming
// anything else would be dereferenced as if its first word were a class id.
TEST(DriverTest, EveryLandingPadClauseIsAPythonClassOrCatchAll) {
  CompileResult result = compileSource("def boom() -> int:\n"
                                       "    raise ValueError('x')\n"
                                       "\n"
                                       "def run() -> int:\n"
                                       "    try:\n"
                                       "        return boom()\n"
                                       "    except ValueError:\n"
                                       "        return 1\n"
                                       "    except KeyError:\n"
                                       "        return 2\n"
                                       "\n"
                                       "print(run())\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  ASSERT_TRUE(result.verified.llvmModule);

  unsigned pads = 0;
  llvm::SmallVector<std::string, 4> named;
  for (llvm::Function &function : *result.verified.llvmModule)
    for (llvm::BasicBlock &block : function) {
      auto *pad = block.getLandingPadInst();
      if (!pad)
        continue;
      ++pads;
      EXPECT_TRUE(pad->getNumClauses() > 0 || pad->isCleanup())
          << "a landing pad in " << function.getName().str()
          << " that neither cleans up nor catches is never entered";
      for (unsigned index = 0; index < pad->getNumClauses(); ++index) {
        EXPECT_TRUE(pad->isCatch(index))
            << "a filter clause (an exception specification) in "
            << function.getName().str()
            << ": LyEH_Personality has no path for one";
        llvm::Constant *clause = pad->getClause(index);
        if (clause->isNullValue())
          continue;
        auto *global = llvm::dyn_cast<llvm::GlobalVariable>(clause);
        ASSERT_NE(global, nullptr) << "a clause that is not a global";
        EXPECT_TRUE(global->getName().starts_with("__ly_exc_type_"))
            << global->getName().str()
            << " is not a Python class id record, and the personality would "
               "read its first word as one";
        named.push_back(global->getName().str());
      }
    }
  EXPECT_GT(pads, 0u) << "the program above must produce landing pads at all";
  EXPECT_FALSE(named.empty())
      << "two named except arms and no finally must reach the type table";
}

// ⛔ The clause list is what lets the personality decide a frame is NOT entered,
// so every shape whose handled set is not exactly the list must stay a
// catch-all. `except*` matches an ExceptionGroup CONTAINING the named class
// rather than the class; a `finally` runs its body for every exception, so that
// frame really is entered by all of them.
TEST(DriverTest, ShapesThatHandleMoreThanTheyNameStayCatchAll) {
  for (llvm::StringRef program :
       {"try:\n    raise ValueError('x')\nexcept* ValueError as e:\n"
        "    print(e)\n",
        "def run() -> int:\n    try:\n        raise ValueError('x')\n"
        "    except ValueError:\n        return 1\n    finally:\n"
        "        print('d')\n\nprint(run())\n"}) {
    CompileResult result = compileSource(program);
    ASSERT_TRUE(result.succeeded) << result.diagnostics;
    for (llvm::Function &function : *result.verified.llvmModule)
      for (llvm::BasicBlock &block : function) {
        auto *pad = block.getLandingPadInst();
        if (!pad)
          continue;
        for (unsigned index = 0; index < pad->getNumClauses(); ++index)
          EXPECT_TRUE(pad->getClause(index)->isNullValue())
              << "a typed clause under:\n" << program.str();
      }
  }
}

// A bare `except` DOES get a clause, and it is BaseException -- which is not an
// exception to the rule above but the reason there is no exception to make:
// `LyEH_ClassIdMatches` answers true for everything against it, so naming it is
// the same decision a catch-all makes, reached one call earlier.
TEST(DriverTest, ABareExceptNamesBaseException) {
  CompileResult result = compileSource("def boom() -> int:\n"
                                       "    raise ValueError('x')\n"
                                       "\n"
                                       "try:\n"
                                       "    boom()\n"
                                       "except:\n"
                                       "    print('any')\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  unsigned typedClauses = 0;
  for (llvm::Function &function : *result.verified.llvmModule)
    for (llvm::BasicBlock &block : function) {
      auto *pad = block.getLandingPadInst();
      if (!pad)
        continue;
      for (unsigned index = 0; index < pad->getNumClauses(); ++index) {
        llvm::Constant *clause = pad->getClause(index);
        if (clause->isNullValue())
          continue;
        ++typedClauses;
        EXPECT_EQ(clause->getName(), "__ly_exc_type_5")
            << "a bare except that names anything narrower than BaseException "
               "drops the exceptions it does not name";
      }
    }
  EXPECT_GT(typedClauses, 0u);
}

// A tuple of classes is one arm, and every class in it has to reach the table:
// the one left out is an exception the personality walks past.
TEST(DriverTest, EveryClassOfATupleArmReachesTheTypeTable) {
  CompileResult result = compileSource("def boom() -> int:\n"
                                       "    raise KeyError('x')\n"
                                       "\n"
                                       "try:\n"
                                       "    boom()\n"
                                       "except (ValueError, KeyError):\n"
                                       "    print('caught')\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  unsigned typedClauses = 0;
  for (llvm::Function &function : *result.verified.llvmModule)
    for (llvm::BasicBlock &block : function) {
      auto *pad = block.getLandingPadInst();
      if (!pad)
        continue;
      for (unsigned index = 0; index < pad->getNumClauses(); ++index)
        if (!pad->getClause(index)->isNullValue())
          ++typedClauses;
    }
  EXPECT_EQ(typedClauses, 2u)
      << "both arms of the tuple must be clauses, or neither";
}

// The personality is chosen from the target, in one place, and both sides ask
// the same question -- the pass that names it here and the support builder that
// defines it. A target that cannot have the Python one keeps the C++ ABI's.
TEST(DriverTest, ThePersonalityIsTheOneTheTargetCanHave) {
  CompileResult result = compileSource("try:\n"
                                       "    raise ValueError('x')\n"
                                       "except ValueError:\n"
                                       "    print('caught')\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  ASSERT_TRUE(result.verified.llvmModule);

  llvm::Triple host(llvm::sys::getDefaultTargetTriple());
  llvm::StringRef expected = py::runtime_library::personalityNameFor(host);
  unsigned checked = 0;
  for (llvm::Function &function : *result.verified.llvmModule) {
    if (!function.hasPersonalityFn())
      continue;
    ++checked;
    EXPECT_EQ(function.getPersonalityFn()->getName(), expected);
  }
  EXPECT_GT(checked, 0u);
}

// WHAT: the LSDA `@TType` encoding each target this compiler codegens for
// actually emits, through BOTH factories -- the AOT one and a JIT target
// machine builder configured the way tools/CLI.cpp configures it.
// `ly_eh_lookup_site` reads ONE encoding, `DW_EH_PE_indirect | pcrel | sdata4`
// (0x9b), and aborts on anything else, so a target that emits a different one
// turns every raise into an abort inside the unwinder.
//
// It got there twice, on the two knobs the encoding depends on. ELF defaults
// to `Reloc::Static`, which emits `udata4` (0x03) -- 162 tests. Then ORC's
// default code model is Large, which emits `sdata8` (0x9c) -- the same 160
// again, JIT only, while the AOT half passed. MachO answers 0x9b whatever
// either knob says, which is why the triples here are NAMED: the bug is
// invisible on the machine that has it.
//
// The Large row is asserted too, and is not redundant: it is the measurement
// that says WHY the code model is spelled rather than left to the default.
//
// ⛔ aarch64 ELF is 0x9c under BOTH code models -- the width follows LP64
// there, not the model -- so no knob reaches it and that target still refuses.
// Recorded in tests/probe/wb_aarch64_elf_type_table_width.py.
TEST(DriverTest, EveryTargetsExceptionTableIsTheOneTheReaderReads) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  // DW_EH_PE_indirect (0x80) | DW_EH_PE_pcrel (0x10) | DW_EH_PE_sdata4 (0x0b),
  // and the same with DW_EH_PE_sdata8 (0x0c).
  constexpr unsigned kIndirectPcrelSdata4 = 0x9b;
  constexpr unsigned kIndirectPcrelSdata8 = 0x9c;

  auto encodingOf = [](llvm::TargetMachine &machine) {
    // The encoding is decided only once the object-file lowering has an
    // MCContext: `getTTypeEncoding()` answers 0 before that, which would be a
    // silent pass rather than a failure.
    llvm::MCContext context(machine.getTargetTriple(), machine.getMCAsmInfo(),
                            machine.getMCRegisterInfo(),
                            machine.getMCSubtargetInfo());
    machine.getObjFileLowering()->Initialize(context, machine);
    return machine.getObjFileLowering()->getTTypeEncoding();
  };

  const std::vector<std::pair<const char *, unsigned>> expected = {
      {"x86_64-unknown-linux-gnu", kIndirectPcrelSdata4},
      {"arm64-apple-macosx", kIndirectPcrelSdata4},
      {"x86_64-apple-macosx", kIndirectPcrelSdata4},
      {"armv7-unknown-linux-gnueabihf", kIndirectPcrelSdata4},
      {"aarch64-unknown-linux-gnu", kIndirectPcrelSdata8},
  };

  unsigned checked = 0;
  for (auto [triple, encoding] : expected) {
    lython::driver::DriverOptions options;
    options.targetTriple = triple;
    std::string diagnostics;
    llvm::raw_string_ostream diag(diagnostics);
    std::unique_ptr<llvm::TargetMachine> aot =
        lython::driver::createCodeGenTargetMachine(
            py::TensorLoweringTarget{}, options, nullptr, diag);
    if (!aot)
      continue; // this build of LLVM does not carry that backend
    ++checked;
    EXPECT_EQ(encodingOf(*aot), encoding) << triple << " (AOT)";

    // The JIT reaches codegen through ORC, so it is configured separately and
    // was wrong on its own for a whole CI round. Same two knobs, same answer.
    llvm::orc::JITTargetMachineBuilder jitBuilder{llvm::Triple(triple)};
    // The same three calls CLI.cpp makes, in the same order: the exception
    // MODEL is a knob too, and armv7 answers 0 without it (EHABI has no
    // type-table encoding to report).
    llvm::TargetOptions jitOptions = jitBuilder.getOptions();
    lython::driver::applyExceptionUnwindOptions(jitOptions,
                                                llvm::Triple(triple));
    jitBuilder.setOptions(jitOptions);
    jitBuilder.setRelocationModel(
        lython::driver::exceptionTableRelocationModel());
    jitBuilder.setCodeModel(lython::driver::exceptionTableCodeModel());
    auto jit = jitBuilder.createTargetMachine();
    ASSERT_TRUE(static_cast<bool>(jit))
        << triple << " " << llvm::toString(jit.takeError());
    EXPECT_EQ(encodingOf(**jit), encoding) << triple << " (JIT)";
  }
  EXPECT_GT(checked, 0u);

  // ⭐ AND THAT THE JIT ACTUALLY ASKS. The two arms above both configure their
  // own builder, so they agree with each other by construction and would keep
  // agreeing if the JIT stopped setting a knob -- which is the exact way this
  // broke: the AOT path was fixed, the JIT path was not, and one whole CI
  // round went into finding that out. A text check on the source is crude and
  // it is the only thing here that fails when the call goes missing.
  {
    std::ifstream cli(std::string(LYTHON_SOURCE_DIR) + "/tools/CLI.cpp");
    ASSERT_TRUE(cli.is_open());
    std::stringstream buffer;
    buffer << cli.rdbuf();
    const std::string text = buffer.str();
    EXPECT_NE(text.find("exceptionTableRelocationModel()"), std::string::npos);
    EXPECT_NE(text.find("exceptionTableCodeModel()"), std::string::npos);
  }

  // The measurement the code model is spelled FOR: leave it at ORC's default
  // and x86-64 moves to eight-byte type-table entries.
  std::string error;
  llvm::Triple x86(llvm::Triple::normalize("x86_64-unknown-linux-gnu"));
  if (const llvm::Target *target =
          llvm::TargetRegistry::lookupTarget(x86, error)) {
    llvm::TargetOptions opt;
    std::unique_ptr<llvm::TargetMachine> large(target->createTargetMachine(
        x86, "generic", "", opt,
        lython::driver::exceptionTableRelocationModel(),
        llvm::CodeModel::Large));
    ASSERT_TRUE(static_cast<bool>(large));
    EXPECT_EQ(encodingOf(*large), kIndirectPcrelSdata8);
  }
}

// WHAT: the refusal a generator gets when the state machine declines it names
// the value that made it decline, not just the tier's own limit.
//
// `prev: "int | None"` carried across the suspension lands in the single-lane
// tier and is refused for yielding a two-lane union -- and the same generator
// with a plain int prev compiles, so the yield's arity is not the whole story.
// What sent it down is the UNION being live across the yield with no frame
// lane, and a frame lane is keyed on a runtime contract. Reading the lower
// message alone sends the reader after the yield.
//
// The dict.items() spelling this used to assert is compiled now (a generator's
// dict walk goes through its keys); the probe for the union is
// tests/probe/wb_generator_carries_an_optional.py.
//
// Driver-layer and not golden: the whole behaviour is a refusal, and the
// control is the same generator without the union, which compiles.
TEST(DriverTest, ARefusedGeneratorNamesWhatSentItDown) {
  CompileResult refused =
      compileSource("def pairwise(xs: \"list[int]\"):\n"
                    "    prev: \"int | None\" = None\n"
                    "    for x in xs:\n"
                    "        if prev is not None:\n"
                    "            yield prev + x\n"
                    "        prev = x\n"
                    "print(list(pairwise([1, 2, 3])))\n");
  EXPECT_FALSE(refused.succeeded);
  EXPECT_NE(refused.diagnostics.find("declined this generator because"),
            std::string::npos)
      << refused.diagnostics;
  EXPECT_NE(refused.diagnostics.find("is live across a yield"),
            std::string::npos)
      << refused.diagnostics;

  // A `continue` before the yield leaves an `arith.constant true` live across
  // the suspension; it is rematerialized at its uses now rather than needing a
  // lane, so the whole shape compiles.
  CompileResult accepted = compileSource("def lines(text: str):\n"
                                         "    for line in text.splitlines():\n"
                                         "        if not line:\n"
                                         "            continue\n"
                                         "        yield line\n"
                                         "for line in lines(\"a\\n\\nb\"):\n"
                                         "    print(line)\n");
  EXPECT_TRUE(accepted.succeeded) << accepted.diagnostics;
}

// `T | None` is one field, not a tag and two layouts. It is stored as a BOX --
// the same box a plain class-typed field gets -- so a class that names itself
// through one has a finite layout, and its ABI is the same width as if the
// field could not be absent.
TEST(DriverTest, AnOptionalFieldIsBoxedLikeThePayloadAlone) {
  CompileResult optional = compileSource(
      "class Node:\n"
      "    v: int\n"
      "    nxt: \"Node | None\"\n"
      "    def __init__(self, v: int) -> None:\n"
      "        self.v = v\n"
      "        self.nxt = None\n"
      "\n"
      "def take(n: Node) -> int:\n"
      "    return n.v\n"
      "\n"
      "take(Node(1))\n");
  ASSERT_TRUE(optional.succeeded) << optional.diagnostics;
  CompileResult plain = compileSource("class Node:\n"
                                      "    v: int\n"
                                      "    nxt: \"Node\"\n"
                                      "    def __init__(self, v: int) -> None:\n"
                                      "        self.v = v\n"
                                      "        self.nxt = self\n"
                                      "\n"
                                      "def take(n: Node) -> int:\n"
                                      "    return n.v\n"
                                      "\n"
                                      "take(Node(1))\n");
  ASSERT_TRUE(plain.succeeded) << plain.diagnostics;
  llvm::Function *optionalTake = optional.verified.llvmModule->getFunction("take");
  llvm::Function *plainTake = plain.verified.llvmModule->getFunction("take");
  ASSERT_NE(optionalTake, nullptr);
  ASSERT_NE(plainTake, nullptr);
  // The two modules are compiled in separate LLVM contexts, so identical types
  // are distinct objects; the printed form is what can be compared.
  std::string optionalSignature;
  std::string plainSignature;
  llvm::raw_string_ostream(optionalSignature) << *optionalTake->getFunctionType();
  llvm::raw_string_ostream(plainSignature) << *plainTake->getFunctionType();
  EXPECT_EQ(optionalSignature, plainSignature)
      << "an optional field costs the same lanes as the payload alone; a tag "
         "and the member's inline lanes would widen the instance";
}

// A union of two OBJECTS still has no box to be stored in, so a class that
// names itself through one has no finite layout and is refused. The refusal is
// what keeps the expansion from recursing until the compiler dies with SIGILL
// and no diagnostic, which is what it did.
TEST(DriverTest, AUnionOfTwoObjectsCannotReachItsOwnClass) {
  CompileResult result = compileSource("class Node:\n"
                                       "    v: int\n"
                                       "    nxt: \"Node | type[Node]\"\n"
                                       "    def __init__(self, v: int) -> None:\n"
                                       "        self.v = v\n"
                                       "        self.nxt = Node\n"
                                       "\n"
                                       "print(Node(1).v)\n");
  EXPECT_FALSE(result.succeeded);
  EXPECT_NE(result.diagnostics.find("contains itself through a union-typed "
                                    "field that is stored inline"),
            std::string::npos)
      << result.diagnostics;
}

// And a union of two OBJECTS no longer does. Each member fits a box, so the
// field is one payload handle whose class word names the member -- which is
// what `Node | Leaf` needs to terminate, and what a `type[X]` member, whose
// value is empty, cannot have.
TEST(DriverTest, AUnionOfTwoObjectsReachesItsOwnClass) {
  CompileResult result = compileSource("class Leaf:\n"
                                       "    n: int\n"
                                       "    def __init__(self, n: int) -> None:\n"
                                       "        self.n = n\n"
                                       "\n"
                                       "class Node:\n"
                                       "    v: int\n"
                                       "    nxt: \"Node | Leaf\"\n"
                                       "    def __init__(self, v: int) -> None:\n"
                                       "        self.v = v\n"
                                       "        self.nxt = Leaf(0)\n"
                                       "\n"
                                       "print(Node(1).v)\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// An optional result carries its payload ONCE. `T | None` is a union with one
// arm that returns an object, and that used to send it down a different path
// from `A | B`: the static-object evidence summary appended a SECOND copy of
// the payload's lanes and marked that copy owned, with no tag to condition the
// obligation on. The duplicate is observable in the ABI -- the same value came
// back twice -- and the ownership consequence was that an optional carried
// across a loop's back edge was diagnosed as unconditionally owned.
TEST(DriverTest, AnOptionalResultCarriesItsPayloadOnce) {
  CompileResult result = compileSource("class Node:\n"
                                       "    def __init__(self) -> None:\n"
                                       "        pass\n"
                                       "\n"
                                       "def one() -> Node:\n"
                                       "    return Node()\n"
                                       "\n"
                                       "def maybe(flag: bool) -> \"Node | None\":\n"
                                       "    if flag:\n"
                                       "        return Node()\n"
                                       "    return None\n"
                                       "\n"
                                       "one()\n"
                                       "maybe(True)\n"
                                       "print('ok')\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  ASSERT_TRUE(result.verified.llvmModule);
  llvm::Function *one = result.verified.llvmModule->getFunction("one");
  llvm::Function *maybe = result.verified.llvmModule->getFunction("maybe");
  ASSERT_NE(one, nullptr);
  ASSERT_NE(maybe, nullptr);

  auto *optional = llvm::dyn_cast<llvm::StructType>(maybe->getReturnType());
  ASSERT_NE(optional, nullptr) << "an optional result is a tag plus lanes";
  EXPECT_TRUE(optional->getElementType(0)->isIntegerTy(64))
      << "the first element of an optional result is its tag";
  ASSERT_EQ(optional->getNumElements(), 2u)
      << "an optional result is its tag and ONE copy of the payload; a second "
         "copy is the static-object evidence summary treating the union as a "
         "single returned object";
  EXPECT_EQ(optional->getElementType(1), one->getReturnType())
      << "the lane after the tag is what the payload is returned as on its "
         "own";
}

// An owned value whose name is rebound by two sequential `if` chains still gets
// its release. Every edge into the first merge borrows -- the value is read
// again after the merge, so none of them can be a move -- and a merge with no
// transferring edge used to be dropped, leaving its argument neither owned nor
// lent. The forward was then read as though the token had moved, so the source
// lost its release on the paths the merge does not carry it out on.
TEST(DriverTest, AnOwnedValueSurvivesTwoRebindingIfChains) {
  CompileResult result =
      compileSource("def f(line: str, col: int, end_col: int) -> int:\n"
                    "    length = len(line)\n"
                    "    marker_end = length\n"
                    "    if end_col > col:\n"
                    "        marker_end = end_col\n"
                    "        if marker_end > length:\n"
                    "            marker_end = length\n"
                    "    if marker_end <= col:\n"
                    "        marker_end = col + 1\n"
                    "        if marker_end > length:\n"
                    "            marker_end = length\n"
                    "    return marker_end\n"
                    "\n"
                    "print(f('return a // b', 7, 13))\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// A borrowed parameter rebound into a local, twice, with a loop between the
// rebinds. Each rebind lends the parameter to a merge argument, and both
// returns were lost: the walk kept a pre-rename name verbatim across edges, so
// the release written under the loop's name for a naming taken before it was
// invisible; and a `cond_br` forwarding one value to BOTH successors' arguments
// made the candidate propagation give up, so the loop header's arguments were
// never owned and the loop-exit edge lent them instead of transferring.
TEST(DriverTest, ABorrowedParameterRebindsAcrossALoop) {
  CompileResult result =
      compileSource("def anchors(line: str, col: int, end_col: int) -> str:\n"
                    "    length = len(line)\n"
                    "    start = 0\n"
                    "    if col > 0 and col < length:\n"
                    "        start = col\n"
                    "    else:\n"
                    "        while start < length and line[start] == ' ':\n"
                    "            start += 1\n"
                    "    marker_end = length\n"
                    "    if end_col > col and end_col > 0:\n"
                    "        marker_end = end_col\n"
                    "        if marker_end > length:\n"
                    "            marker_end = length\n"
                    "    if marker_end <= start:\n"
                    "        marker_end = start + 1\n"
                    "    caret = -1\n"
                    "    split = start\n"
                    "    while split < marker_end:\n"
                    "        if line[split] == '(':\n"
                    "            caret = split\n"
                    "            break\n"
                    "        split += 1\n"
                    "    if caret < 0:\n"
                    "        caret = start\n"
                    "    out = ''\n"
                    "    mark = start\n"
                    "    while mark < marker_end:\n"
                    "        if caret <= mark:\n"
                    "            out += '^'\n"
                    "        else:\n"
                    "            out += '~'\n"
                    "        mark += 1\n"
                    "    return out\n"
                    "\n"
                    "print(anchors('return a // b', 7, 13))\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// What: a generator method that recurses into its children is refused for the
// DELEGATION limit rather than for a method the class plainly declares. The
// old message named `walk` as missing and pointed at its own `def`, because
// the yield-type inference walks the body before the method is published.
TEST(DriverTest, ARecursiveGeneratorMethodNamesTheRealLimit) {
  CompileResult refused = compileSource(
      "from typing import Iterator\n"
      "class Tree:\n"
      "    def __init__(self, value: int) -> None:\n"
      "        self.value = value\n"
      "        self.children: list[\"Tree\"] = []\n"
      "    def walk(self) -> \"Iterator[int]\":\n"
      "        yield self.value\n"
      "        for child in self.children:\n"
      "            for nested in child.walk():\n"
      "                yield nested\n"
      "print(list(Tree(1).walk()))\n");
  EXPECT_FALSE(refused.succeeded);
  EXPECT_NE(refused.diagnostics.find("recursive delegation has no static "
                                     "expansion"),
            std::string::npos)
      << refused.diagnostics;
  EXPECT_EQ(refused.diagnostics.find("does not provide manifest method"),
            std::string::npos)
      << refused.diagnostics;

  // A generator method called from a SIBLING still compiles: the publication
  // change must not disturb the case that already worked.
  CompileResult accepted = compileSource(
      "from typing import Iterator\n"
      "class Bag:\n"
      "    def __init__(self, value: int) -> None:\n"
      "        self.value = value\n"
      "    def each(self) -> \"Iterator[int]\":\n"
      "        yield self.value\n"
      "    def total(self) -> int:\n"
      "        n = 0\n"
      "        for x in self.each():\n"
      "            n += x\n"
      "        return n\n"
      "print(Bag(5).total())\n");
  EXPECT_TRUE(accepted.succeeded) << accepted.diagnostics;
}

// A bool LIVE ACROSS a yield still has no frame lane, and the tier that
// refuses it must say so: its own sentence is about int yield bodies, which is
// never why a generator arrived there. The yielded value's lane is separate and
// compiles (tests/golden/cases/a_generator_that_yields_a_bool.py).
TEST(DriverTest, ABoolLiveAcrossAYieldNamesTheRealLimit) {
  // A bool live across a yield has a frame lane now: the frame WORD accounting
  // always gave it one and the STORE side always took one word for a bare i1;
  // only the LOAD half was missing, so the gate asked with `allowBool` false
  // and this compiled to a refusal about the frame layout.
  CompileResult flag = compileSource(
      "from typing import Iterator\n"
      "def go() -> Iterator[bool]:\n"
      "    flag = True\n"
      "    for _ in range(3):\n"
      "        yield flag\n"
      "        flag = not flag\n"
      "print(list(go()))\n");
  EXPECT_TRUE(flag.succeeded) << flag.diagnostics;

  // The limit the message names is real for a UNION, which has no lane at all:
  // the sentence has to keep naming the VALUE, because the tier below refuses
  // for a reason that is never why the program came down to it.
  CompileResult refused = compileSource(
      "from typing import Iterator, Optional\n"
      "def go() -> Iterator[int]:\n"
      "    v: Optional[int] = 3\n"
      "    yield 0\n"
      "    yield 0 if v is None else v\n"
      "print(list(go()))\n");
  EXPECT_FALSE(refused.succeeded);
  EXPECT_NE(refused.diagnostics.find("is live across a yield and has no "
                                     "generator frame lane"),
            std::string::npos)
      << refused.diagnostics;
}

// A `with ... as X` target inside a generator is typed before the body walk
// reaches the yields, the same way a loop target is. Without it the generator
// was refused for its own correct annotation; the limit it really meets is the
// unwind cleanup one (tests/probe/wb_a_try_inside_a_loop_inside_a_generator.py).
TEST(DriverTest, AWithTargetInAGeneratorIsNotBlamedOnTheAnnotation) {
  CompileResult refused = compileSource(
      "from typing import Iterator\n"
      "class Ctx:\n"
      "    def __enter__(self) -> int:\n"
      "        return 5\n"
      "    def __exit__(self, a: object, b: object, c: object) -> bool:\n"
      "        return False\n"
      "def go() -> Iterator[int]:\n"
      "    with Ctx() as base:\n"
      "        yield base\n"
      "print(list(go()))\n");
  EXPECT_FALSE(refused.succeeded);
  EXPECT_EQ(refused.diagnostics.find("but yields"), std::string::npos)
      << refused.diagnostics;
  EXPECT_NE(refused.diagnostics.find("unwind cleanup cannot target a handler "
                                     "entry with block arguments"),
            std::string::npos)
      << refused.diagnostics;
}

// What: the names Emscripten answers to. CPython built for Emscripten reports
// `sys.platform == "emscripten"` and `platform.system() == "Emscripten"`, and
// both fold from the one row the target triple selects.
TEST(DriverTest, EmscriptenNamesItselfTheWayCPythonDoes) {
  EXPECT_EQ(py::platform_constants::staticStringValue(
                "sys.platform", "wasm64-unknown-emscripten"),
            std::optional<std::string>("emscripten"));
  EXPECT_EQ(py::platform_constants::staticStringValue(
                "platform.system", "wasm64-unknown-emscripten"),
            std::optional<std::string>("Emscripten"));
  EXPECT_EQ(py::platform_constants::staticIntValue("sys.maxsize",
                                                   "wasm64-unknown-emscripten"),
            std::optional<long long>(9223372036854775807LL));
}

// What: the libc facts the OS cluster reads on wasm64 Emscripten are musl's as
// measured under `emcc -m64`: WASI errno numbers, and a `struct stat` that
// leads with 32-bit dev_t and mode_t and ends with st_ino.
TEST(DriverTest, Wasm64EmscriptenReadsMuslsLayout) {
  py::runtime_library::HostTargetLayout layout =
      py::runtime_library::hostTargetLayout(
          llvm::Triple("wasm64-unknown-emscripten"));
  EXPECT_TRUE(layout.posix);
  EXPECT_EQ(layout.errnoAccessor, "__errno_location");
  EXPECT_EQ(layout.errnoNumbering, py::exceptions::ErrnoNumbering::WASI);
  EXPECT_EQ(layout.statDev[0], 0);
  EXPECT_EQ(layout.statDev[1], 4);
  EXPECT_EQ(layout.statMode[0], 4);
  EXPECT_EQ(layout.statNlink[0], 8);
  EXPECT_EQ(layout.statUid[0], 16);
  EXPECT_EQ(layout.statGid[0], 20);
  EXPECT_EQ(layout.statSize[0], 32);
  EXPECT_EQ(layout.statAtime[0], 48);
  EXPECT_EQ(layout.statMtime[0], 64);
  EXPECT_EQ(layout.statCtime[0], 80);
  EXPECT_EQ(layout.statIno[0], 96);
  EXPECT_EQ(layout.direntNameOffset, 19);
  EXPECT_EQ(layout.clockMonotonic, 1);

  int enoent = 0;
  for (const py::exceptions::OSErrorErrnoMapping &row :
       py::exceptions::kOSErrorErrnoMap)
    if (row.posixName == "ENOENT")
      enoent = row.valueFor(layout.errnoNumbering);
  EXPECT_EQ(enoent, 44);
}

namespace {

// Compiles `source` for `triple` and links the runtime into it, which is where
// the raise primitive and the libc declarations meet. The whole
// VerifiedLLVMModule comes back because it owns the LLVMContext; `llvmModule`
// is null when a step failed. With `refusal`, the libc prototype pass may
// refuse the program: its message lands there instead of failing the test.
lython::driver::VerifiedLLVMModule
compileAndLinkFor(llvm::StringRef source, llvm::StringRef triple,
                  std::string *refusal = nullptr) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  lython::driver::DriverOptions options;
  options.targetTriple = triple.str();
  CompileResult result = compileSource(source, options);
  EXPECT_TRUE(result.succeeded) << triple.str() << "\n" << result.diagnostics;
  if (!result.succeeded)
    return {};
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  if (mlir::failed(lython::driver::configureLLVMModuleCodeGenTarget(
          *result.verified.llvmModule,
          lython::driver::detectTensorLoweringTarget(options), options,
          diag))) {
    ADD_FAILURE() << triple.str() << "\n" << diagnostics;
    return {};
  }
  if (mlir::failed(py::runtime_library::linkEmbeddedNativeRuntime(
          *result.verified.llvmModule))) {
    ADD_FAILURE() << "runtime link failed for " << triple.str();
    return {};
  }
  lython::driver::redirectAllocationsToObjectAllocator(
      *result.verified.llvmModule, /*bypass=*/false);
  if (mlir::failed(py::runtime_library::declareLibcWithTargetPrototypes(
          *result.verified.llvmModule, diag))) {
    if (refusal)
      *refusal = diagnostics;
    else
      ADD_FAILURE() << triple.str() << "\n" << diagnostics;
    return {};
  }
  return std::move(result.verified);
}

bool isCalled(const llvm::Module &module, llvm::StringRef name) {
  const llvm::Function *fn = module.getFunction(name);
  if (!fn)
    return false;
  for (const llvm::User *user : fn->users())
    if (const auto *call = llvm::dyn_cast<llvm::CallBase>(user))
      if (call->getCalledOperand() == fn)
        return true;
  return false;
}

} // namespace

// What: a wasm64 module raises through `_Unwind_RaiseException` like every
// other target, and after the funclet rewrite it holds no landingpad and no
// resume, names the wasm personality, verifies, and gets through the
// WebAssembly backend's instruction selection -- which crashed on the
// landingpads before.
TEST(DriverTest, AWasm64ModuleCarriesItsPadsAsFunclets) {
  const char *source = "def fail(n: int) -> int:\n"
                       "    if n > 0:\n"
                       "        raise ValueError('x')\n"
                       "    return n\n"
                       "\n"
                       "try:\n"
                       "    try:\n"
                       "        fail(1)\n"
                       "    finally:\n"
                       "        print('cleanup')\n"
                       "except ValueError as e:\n"
                       "    print('caught', e)\n";
  lython::driver::VerifiedLLVMModule wasmResult =
      compileAndLinkFor(source, "wasm64-unknown-emscripten");
  ASSERT_TRUE(wasmResult.llvmModule);
  llvm::Module *wasm = wasmResult.llvmModule.get();
  EXPECT_TRUE(isCalled(*wasm, "_Unwind_RaiseException"));

  EXPECT_TRUE(py::convertLandingPadsToWasmFunclets(*wasm));
  unsigned catchPads = 0;
  for (llvm::Function &function : *wasm)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block) {
        EXPECT_FALSE(llvm::isa<llvm::LandingPadInst>(&instruction))
            << function.getName().str();
        EXPECT_FALSE(llvm::isa<llvm::ResumeInst>(&instruction))
            << function.getName().str();
        if (llvm::isa<llvm::CatchPadInst>(&instruction)) {
          ++catchPads;
          EXPECT_EQ(function.getPersonalityFn()->getName(),
                    py::runtime_library::kWasmPersonalityName);
        }
      }
  EXPECT_GT(catchPads, 0u);
  std::string broken;
  llvm::raw_string_ostream brokenStream(broken);
  ASSERT_FALSE(llvm::verifyModule(*wasm, &brokenStream)) << broken;

  lython::driver::DriverOptions options;
  options.targetTriple = "wasm64-unknown-emscripten";
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  llvm::InitializeAllAsmPrinters();
  std::unique_ptr<llvm::TargetMachine> machine =
      lython::driver::createCodeGenTargetMachine(
          lython::driver::detectTensorLoweringTarget(options), options, nullptr,
          diag);
  ASSERT_TRUE(machine) << diagnostics;
  llvm::SmallString<0> object;
  llvm::raw_svector_ostream objectStream(object);
  llvm::legacy::PassManager codegen;
  ASSERT_FALSE(machine->addPassesToEmitFile(codegen, objectStream, nullptr,
                                            llvm::CodeGenFileType::ObjectFile));
  codegen.run(*wasm);
  EXPECT_FALSE(object.empty());
}

// What: every `puts` the linked module calls is C's `int puts(const char *)`.
// MLIR's `cf.assert` lowering declares it returning void, and wasm-ld turns a
// call through the wrong signature into a trap.
TEST(DriverTest, PutsIsDeclaredWithItsCPrototype) {
  lython::driver::VerifiedLLVMModule wasmResult =
      compileAndLinkFor("print('hi')\n", "wasm64-unknown-emscripten");
  ASSERT_TRUE(wasmResult.llvmModule);
  llvm::Module *wasm = wasmResult.llvmModule.get();
  const llvm::Function *puts = wasm->getFunction("puts");
  ASSERT_NE(puts, nullptr)
      << "no cf.assert reaches this module any more; the test looks at nothing";
  EXPECT_TRUE(puts->getReturnType()->isIntegerTy(32));
}

// What: a 32-bit target compiles only where its libc was measured -- armv7
// glibc and wasm32 Emscripten -- and any other is refused before lowering
// rather than read through guessed struct layouts.
TEST(DriverTest, A32BitTargetCompilesOnlyWhereItsLibcWasMeasured) {
  for (const char *triple :
       {"armv7-unknown-linux-gnueabihf", "wasm32-unknown-emscripten"}) {
    lython::driver::DriverOptions options;
    options.targetTriple = triple;
    CompileResult result = compileSource("print(1)\n", options);
    EXPECT_TRUE(result.succeeded) << triple << "\n" << result.diagnostics;
  }
  for (const char *triple :
       {"i686-unknown-linux-gnu", "riscv32-unknown-linux-gnu"}) {
    lython::driver::DriverOptions options;
    options.targetTriple = triple;
    CompileResult result = compileSource("print(1)\n", options);
    EXPECT_FALSE(result.succeeded) << triple;
    EXPECT_NE(result.diagnostics.find("no measured libc layout"),
              std::string::npos)
        << triple << "\n"
        << result.diagnostics;
  }
}

// What: where the target's malloc promises less than 16-byte alignment
// (Emscripten: 8), the object allocator takes its arenas and large blocks from
// aligned_alloc; where malloc already promises 16 it keeps calling malloc.
TEST(DriverTest, TheObjectAllocatorAlignsWhereMallocDoesNot) {
  auto callsFrom = [](const llvm::Module &module, llvm::StringRef caller,
                      llvm::StringRef callee) {
    const llvm::Function *fn = module.getFunction(caller);
    if (!fn)
      return false;
    for (const llvm::BasicBlock &block : *fn)
      for (const llvm::Instruction &instruction : block)
        if (const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
          if (const llvm::Function *target = call->getCalledFunction())
            if (target->getName() == callee)
              return true;
    return false;
  };

  lython::driver::VerifiedLLVMModule wasm =
      compileAndLinkFor("print('hi')\n", "wasm64-unknown-emscripten");
  ASSERT_TRUE(wasm.llvmModule);
  EXPECT_TRUE(callsFrom(*wasm.llvmModule, "LyMem_Alloc", "aligned_alloc"));
  EXPECT_FALSE(callsFrom(*wasm.llvmModule, "LyMem_Alloc", "malloc"));

  lython::driver::VerifiedLLVMModule host =
      compileAndLinkFor("print('hi')\n", llvm::sys::getDefaultTargetTriple());
  ASSERT_TRUE(host.llvmModule);
  EXPECT_TRUE(callsFrom(*host.llvmModule, "LyMem_Alloc", "malloc"));
  EXPECT_FALSE(callsFrom(*host.llvmModule, "LyMem_Alloc", "aligned_alloc"));
}

namespace {

const char *ctypesAddressSource(bool withPrototype) {
  return withPrototype ? "import ctypes\n"
                         "libc = ctypes.CDLL(None)\n"
                         "w = libc[\"write\"]\n"
                         "w.restype = ctypes.c_long\n"
                         "w.argtypes = [ctypes.c_int, ctypes.c_void_p, "
                         "ctypes.c_long]\n"
                         "addr: int = ctypes.cast(w, ctypes.c_void_p).value\n"
                         "print(addr != 0)\n"
                       : "import ctypes\n"
                         "libc = ctypes.CDLL(None)\n"
                         "w = libc[\"write\"]\n"
                         "addr: int = ctypes.cast(w, ctypes.c_void_p).value\n"
                         "print(addr != 0)\n";
}

} // namespace

// What: a ctypes symbol whose address is taken is declared with the prototype
// its restype/argtypes name, not as `void (...)` -- on wasm the declaration is
// the import's signature, and `write` is also the runtime's own import.
TEST(DriverTest, ACtypesSymbolAddressIsDeclaredWithItsPrototype) {
  lython::driver::DriverOptions options;
  options.targetTriple = "wasm64-unknown-emscripten";
  CompileResult result = compileSource(ctypesAddressSource(true), options);
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  const llvm::Function *write = result.verified.llvmModule->getFunction("write");
  ASSERT_NE(write, nullptr);
  llvm::LLVMContext &context = result.verified.llvmModule->getContext();
  llvm::FunctionType *expected = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context),
      {llvm::Type::getInt32Ty(context), llvm::PointerType::getUnqual(context),
       llvm::Type::getInt64Ty(context)},
      /*isVarArg=*/false);
  EXPECT_EQ(write->getFunctionType(), expected);
}

// What: without a restype the prototype is unknown. A native target links the
// address by name and compiles it; wasm would import `write` with a made-up
// signature, so it is refused and the message says what to set.
TEST(DriverTest, AnUntypedCtypesSymbolAddressIsRefusedOnWasm) {
  lython::driver::DriverOptions wasm;
  wasm.targetTriple = "wasm64-unknown-emscripten";
  CompileResult refused = compileSource(ctypesAddressSource(false), wasm);
  EXPECT_FALSE(refused.succeeded);
  EXPECT_NE(refused.diagnostics.find("needs its prototype: set restype"),
            std::string::npos)
      << refused.diagnostics;

  CompileResult host = compileSource(ctypesAddressSource(false));
  EXPECT_TRUE(host.succeeded) << host.diagnostics;
}

// What: the ILP32 libc facts, as measured -- armv7 glibc 2.36 built with
// _FILE_OFFSET_BITS=64 and _TIME_BITS=64, and wasm32 Emscripten's musl.
TEST(DriverTest, ILP32TargetsReadTheirMeasuredLayouts) {
  py::runtime_library::HostTargetLayout arm =
      py::runtime_library::hostTargetLayout(
          llvm::Triple("armv7-unknown-linux-gnueabihf"));
  EXPECT_EQ(arm.errnoNumbering, py::exceptions::ErrnoNumbering::Linux);
  EXPECT_EQ(arm.mallocAlignment, 8);
  EXPECT_EQ(arm.statDev[0], 0);
  EXPECT_EQ(arm.statDev[1], 8);
  EXPECT_EQ(arm.statIno[0], 8);
  EXPECT_EQ(arm.statMode[0], 16);
  EXPECT_EQ(arm.statNlink[0], 20);
  EXPECT_EQ(arm.statNlink[1], 4);
  EXPECT_EQ(arm.statUid[0], 24);
  EXPECT_EQ(arm.statGid[0], 28);
  EXPECT_EQ(arm.statSize[0], 40);
  EXPECT_EQ(arm.statAtime[0], 64);
  EXPECT_EQ(arm.statMtime[0], 80);
  EXPECT_EQ(arm.statCtime[0], 96);
  EXPECT_EQ(arm.timespecNsec[0], 8);
  EXPECT_EQ(arm.timespecNsec[1], -4);
  EXPECT_EQ(arm.tmGmtoff[0], 36);
  EXPECT_EQ(arm.tmGmtoff[1], -4);

  py::runtime_library::HostTargetLayout wasm =
      py::runtime_library::hostTargetLayout(
          llvm::Triple("wasm32-unknown-emscripten"));
  EXPECT_EQ(wasm.errnoNumbering, py::exceptions::ErrnoNumbering::WASI);
  EXPECT_EQ(wasm.mallocAlignment, 8);
  EXPECT_EQ(wasm.statDev[1], 4);
  EXPECT_EQ(wasm.statMode[0], 4);
  EXPECT_EQ(wasm.statNlink[0], 8);
  EXPECT_EQ(wasm.statNlink[1], 4);
  EXPECT_EQ(wasm.statUid[0], 12);
  EXPECT_EQ(wasm.statGid[0], 16);
  EXPECT_EQ(wasm.statSize[0], 24);
  EXPECT_EQ(wasm.statAtime[0], 40);
  EXPECT_EQ(wasm.statMtime[0], 56);
  EXPECT_EQ(wasm.statCtime[0], 72);
  EXPECT_EQ(wasm.statIno[0], 88);
  EXPECT_EQ(wasm.tmGmtoff[0], 36);
}

// What: after the runtime link on armv7, libc is declared as C declares it
// there -- size_t and long 32-bit, the time64/LFS symbols for the struct
// readers -- the allocator still stands in for malloc, and the runtime's
// struct offsets are armv7's: nothing stores past the 96 bytes
// `ExceptionParts` occupies with 4-byte pointers.
TEST(DriverTest, AnArmv7RuntimeIsBuiltForArmv7) {
  lython::driver::VerifiedLLVMModule result =
      compileAndLinkFor("import os\n"
                        "print(os.stat('.').st_mode != 0)\n",
                        "armv7-unknown-linux-gnueabihf");
  ASSERT_TRUE(result.llvmModule);
  llvm::Module &module = *result.llvmModule;
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *ptr = llvm::PointerType::getUnqual(context);

  const llvm::Function *fwrite = module.getFunction("fwrite");
  ASSERT_NE(fwrite, nullptr);
  EXPECT_EQ(fwrite->getFunctionType(),
            llvm::FunctionType::get(i32, {ptr, i32, i32, ptr}, false));
  EXPECT_EQ(module.getFunction("stat"), nullptr);
  EXPECT_NE(module.getFunction("__stat64_time64"), nullptr);
  EXPECT_EQ(module.getFunction("clock_gettime"), nullptr);

  for (const llvm::Function &function : module) {
    if (function.isDeclaration() || function.getName().starts_with("LyMem_"))
      continue;
    for (const llvm::BasicBlock &block : function)
      for (const llvm::Instruction &instruction : block)
        if (const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
          if (const llvm::Function *callee = call->getCalledFunction())
            EXPECT_NE(callee->getName(), "malloc")
                << function.getName().str()
                << " calls libc malloc beside LyMem_Free";
  }

  const llvm::GlobalVariable *parts = module.getNamedGlobal("g_current_parts");
  ASSERT_NE(parts, nullptr);
  std::uint64_t bytes =
      module.getDataLayout().getTypeAllocSize(parts->getValueType());
  EXPECT_EQ(bytes, 96u);
  for (const llvm::User *user : parts->users()) {
    const auto *gep = llvm::dyn_cast<llvm::GEPOperator>(user);
    if (!gep)
      continue;
    llvm::APInt offset(64, 0);
    if (gep->accumulateConstantOffset(module.getDataLayout(), offset))
      EXPECT_LT(offset.getZExtValue(), bytes);
  }

  // A catch pad names no Python class under the C++ ABI's personality, which
  // would read the clause as a std::type_info.
  for (const llvm::Function &function : module)
    for (const llvm::BasicBlock &block : function)
      if (const auto *pad = block.getLandingPadInst())
        for (unsigned index = 0; index < pad->getNumClauses(); ++index)
          EXPECT_TRUE(
              llvm::isa<llvm::ConstantPointerNull>(pad->getClause(index)))
              << function.getName().str();
}

// What: wasm32-wasip1 reads wasi-libc's facts as measured under wasmtime --
// WASI errno numbers, a 16-byte malloc, a dirent name after a u64 inode and a
// u8 type, clockid_t as the address of `_CLOCK_*` -- and calls itself what
// CPython's WASI build calls itself.
TEST(DriverTest, WasiPreview1ReadsWasiLibcsLayout) {
  py::runtime_library::HostTargetLayout wasi =
      py::runtime_library::hostTargetLayout(
          llvm::Triple("wasm32-unknown-wasip1"));
  EXPECT_EQ(wasi.errnoNumbering, py::exceptions::ErrnoNumbering::WASI);
  EXPECT_EQ(wasi.mallocAlignment, 16);
  EXPECT_EQ(wasi.direntNameOffset, 9);
  EXPECT_EQ(wasi.statMode[0], 24);
  EXPECT_EQ(wasi.statSize[0], 48);
  EXPECT_EQ(wasi.timespecNsec[1], -4);
  EXPECT_EQ(wasi.tmGmtoff[0], 36);
  EXPECT_EQ(wasi.clockMonotonicGlobal, "_CLOCK_MONOTONIC");
  EXPECT_EQ(wasi.clockRealtimeGlobal, "_CLOCK_REALTIME");
  EXPECT_EQ(py::platform_constants::staticStringValue("sys.platform",
                                                      "wasm32-wasip1"),
            std::optional<std::string>("wasi"));
  EXPECT_EQ(py::platform_constants::staticStringValue("platform.system",
                                                      "wasm32-wasip1"),
            std::optional<std::string>("wasi"));
}

// What: a call wasi-libc cannot answer is refused when the program reaches
// it, naming the path from `__main__`; a program that does not reach it
// links, with clock_gettime taking wasi-libc's pointer clockid_t.
TEST(DriverTest, AWasiProgramIsRefusedOnlyWhereItReachesWhatWasiLacks) {
  std::string refused;
  lython::driver::VerifiedLLVMModule uid = compileAndLinkFor(
      "import os\nprint(os.getuid())\n", "wasm32-wasip1", &refused);
  EXPECT_FALSE(uid.llvmModule);
  EXPECT_NE(refused.find("has no 'getuid'"), std::string::npos) << refused;
  EXPECT_NE(refused.find("__main__ -> "), std::string::npos) << refused;

  lython::driver::VerifiedLLVMModule clock = compileAndLinkFor(
      "import time\nprint(time.monotonic() > 0)\n", "wasm32-wasip1");
  ASSERT_TRUE(clock.llvmModule);
  const llvm::Function *gettime =
      clock.llvmModule->getFunction("clock_gettime");
  ASSERT_NE(gettime, nullptr);
  EXPECT_TRUE(gettime->getFunctionType()->getParamType(0)->isPointerTy());
}

// What: a Python function is local to the module, so its name -- here a C
// library function's -- is never the symbol the runtime's call binds to.
TEST(DriverTest, APythonFunctionIsNotAnExternalSymbol) {
  CompileResult result = compileSource("def write(fd: int) -> int:\n"
                                       "    return fd + 1\n"
                                       "\n"
                                       "print(write(1))\n");
  ASSERT_TRUE(result.succeeded) << result.diagnostics;
  const llvm::Function *write =
      result.verified.llvmModule->getFunction("write");
  ASSERT_NE(write, nullptr);
  EXPECT_TRUE(write->hasLocalLinkage());
  const llvm::Function *main =
      result.verified.llvmModule->getFunction("__main__");
  ASSERT_NE(main, nullptr);
  EXPECT_FALSE(main->hasLocalLinkage());
}

// What: a ctypes library other than the program itself is refused where it is
// constructed, whether or not its symbols are later called -- nothing opens
// it, so its names would resolve among the program's own.
TEST(DriverTest, ACtypesLibraryOtherThanTheProgramIsRefused) {
  for (llvm::StringRef use :
       {"f = lib[\"cos\"]\n"
        "f.restype = ctypes.c_double\n"
        "f.argtypes = [ctypes.c_double]\n"
        "print(f(0.0))\n",
        "f = lib[\"cos\"]\n"
        "print(ctypes.cast(f, ctypes.c_void_p).value)\n"}) {
    CompileResult result = compileSource(
        ("import ctypes\nlib = ctypes.CDLL(\"libm.so.6\")\n" + use).str());
    EXPECT_FALSE(result.succeeded) << use.str();
    EXPECT_NE(result.diagnostics.find(
                  "ctypes.CDLL can only name the program itself (None)"),
              std::string::npos)
        << result.diagnostics;
  }
}

// What: the address of a C function the libc table does not know (`cos`),
// with a restype, links on wasm32-wasip1 -- declared with the program's
// prototype and marked as the program's, where it used to be refused as a gap
// in the table.
TEST(DriverTest, AnUntabledCtypesSymbolsAddressLinksOnWasi) {
  lython::driver::VerifiedLLVMModule linked =
      compileAndLinkFor("import ctypes\n"
                        "libc = ctypes.CDLL(None)\n"
                        "f = libc[\"cos\"]\n"
                        "f.restype = ctypes.c_double\n"
                        "f.argtypes = [ctypes.c_double]\n"
                        "address: int = ctypes.cast(f, ctypes.c_void_p).value\n"
                        "print(address != 0)\n",
                        "wasm32-wasip1");
  ASSERT_TRUE(linked.llvmModule);
  const llvm::Function *cos = linked.llvmModule->getFunction("cos");
  ASSERT_NE(cos, nullptr);
  EXPECT_TRUE(cos->getReturnType()->isDoubleTy());
  EXPECT_TRUE(
      cos->hasFnAttribute(py::runtime_library::kCtypesForeignSymbolAttr));
}

// Compiles `mainSource` to LLVM IR with `modules` (name -> source) written
// beside it, where its imports find them.
static CompileResult compileWithModules(
    llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> modules,
    llvm::StringRef mainSource) {
  CompileResult result;
  llvm::SmallString<128> dir;
  if (llvm::sys::fs::createUniqueDirectory("lython-driver-import", dir)) {
    result.diagnostics = "could not create a temporary import directory";
    return result;
  }
  for (auto [name, source] : modules) {
    llvm::SmallString<128> path(dir);
    llvm::sys::path::append(path, llvm::Twine(name) + ".py");
    std::error_code error;
    llvm::raw_fd_ostream out(path, error);
    out << source;
  }
  llvm::SmallString<128> mainPath(dir);
  llvm::sys::path::append(mainPath, "main.py");
  mlir::MLIRContext context(testRegistry());
  llvm::raw_string_ostream diag(result.diagnostics);
  mlir::ScopedDiagnosticHandler capture(&context,
                                        [&](mlir::Diagnostic &diagnostic) {
                                          diag << diagnostic.str() << "\n";
                                          return mlir::failure();
                                        });
  result.succeeded =
      mlir::succeeded(lython::driver::compilePythonSourceToLLVMIR(
          mainSource, mainPath, dir, lython::driver::DriverOptions{}, context,
          result.verified, diag));
  llvm::sys::fs::remove_directories(dir);
  return result;
}

// What: a function of an imported module that returns a class its module
// imports is typed, at the call, with that class -- not with a contract
// spelled from the bare name, whose fields nothing declares.
TEST(DriverTest, AnImportedFunctionReturnsTheClassItsModuleImports) {
  CompileResult result =
      compileWithModules({{"things", "class Thing:\n"
                                     "    def __init__(self, n: int) -> None:\n"
                                     "        self.n = n\n"},
                          {"maker", "from things import Thing\n\n\n"
                                    "def make(n: int) -> Thing:\n"
                                    "    return Thing(n)\n"}},
                         "import maker\nprint(maker.make(3).n)\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// What: an imported module may keep callables in a container global, the
// same as one callable global; the element type is resolved.
TEST(DriverTest, AnImportedModuleKeepsCallablesInAContainer) {
  CompileResult result = compileWithModules(
      {{"registry", "from typing import Callable\n\n"
                    "HANDLERS: dict[int, Callable[[], None]] = {}\n\n\n"
                    "def add(slot: int, run: Callable[[], None]) -> None:\n"
                    "    HANDLERS[slot] = run\n\n\n"
                    "def fire(slot: int) -> None:\n"
                    "    HANDLERS[slot]()\n"}},
      "import registry\n\n\ndef hello() -> None:\n    print(\"hi\")\n\n\n"
      "registry.add(1, hello)\nregistry.fire(1)\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// What: a closure that calls a captured callable returning a union of a
// class and None compiles: the edge after the dispatch-miss raise carries an
// immortal placeholder whose member header a retain can be written against.
TEST(DriverTest, AClosureCallsACapturedCallableReturningAnOptional) {
  CompileResult result =
      compileSource("from typing import Callable\n\n\n"
                    "class Box:\n"
                    "    def __init__(self, n: int) -> None:\n"
                    "        self.n = n\n\n\n"
                    "def outer(f: Callable[[int], \"Box | None\"]) -> None:\n"
                    "    def run() -> None:\n"
                    "        print(f(1) is None)\n"
                    "    run()\n\n\n"
                    "def g(n: int) -> \"Box | None\":\n"
                    "    return None if n == 1 else Box(n)\n\n\n"
                    "outer(g)\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// What: `Callable[..., R]` takes any callable returning R -- one with
// parameters as well as one without -- in the emitter's check and in the
// lowering's, which read the annotation's `Any` tail as the same "any
// arguments".
TEST(DriverTest, ACallableOfAnyArgumentsTakesAFunctionWithParameters) {
  CompileResult result =
      compileSource("from typing import Callable\n\n\n"
                    "def run(f: \"Callable[..., object]\") -> None:\n"
                    "    print(\"got\")\n\n\n"
                    "def none() -> None:\n"
                    "    pass\n\n\n"
                    "def two(a: int, b: str) -> int:\n"
                    "    return a\n\n\n"
                    "run(none)\nrun(two)\n");
  EXPECT_TRUE(result.succeeded) << result.diagnostics;
}

// A generator resumed where its creating function is not known goes through
// its frame; that is refused, naming the function, when a generator that may
// be the value there has no frame (it is not a state machine), and when the
// value is typed by the `Generator` protocol, which is not reference counted.
TEST(DriverTest, AGeneratorResumedByItsFrameNamesWhatItCannotResume) {
  CompileResult stateless =
      compileSource("from typing import Iterator\n\n\n"
                    "def many(*xs: int) -> Iterator[int]:\n"
                    "    for x in xs:\n"
                    "        yield x\n\n\n"
                    "def total(it: Iterator[int]) -> int:\n"
                    "    t = 0\n"
                    "    for v in it:\n"
                    "        t += v\n"
                    "    return t\n\n\n"
                    "print(total(many(1, 2)))\n");
  EXPECT_FALSE(stateless.succeeded);
  EXPECT_NE(stateless.diagnostics.find("'many' has none (it is not a state "
                                       "machine: it takes *args"),
            std::string::npos)
      << stateless.diagnostics;

  CompileResult protocol =
      compileSource("from typing import Generator\n\n\n"
                    "def worker(n: int) -> Generator[int, None, None]:\n"
                    "    yield n\n\n\n"
                    "tasks: list[Generator[int, None, None]] = [worker(2)]\n"
                    "t = tasks.pop(0)\n"
                    "print(next(t))\n");
  EXPECT_FALSE(protocol.succeeded);
  EXPECT_NE(protocol.diagnostics.find("is not reference counted"),
            std::string::npos)
      << protocol.diagnostics;
}

// next() and send() carry a generator's return value in the StopIteration
// they raise, as str(value); a returned instance of the program's own class
// has no runtime __str__ to render there, and next() on it is refused.
TEST(DriverTest, ANextOnAGeneratorReturningAClassIsRefused) {
  CompileResult result =
      compileSource("from typing import Generator\n\n\n"
                    "class Box:\n"
                    "    def __init__(self, v: int) -> None:\n"
                    "        self.v = v\n\n\n"
                    "def boxed() -> Generator[int, None, Box]:\n"
                    "    yield 1\n"
                    "    return Box(2)\n\n\n"
                    "g = boxed()\n"
                    "print(next(g))\n");
  EXPECT_FALSE(result.succeeded);
  EXPECT_NE(result.diagnostics.find("'Box' value has no runtime __str__"),
            std::string::npos)
      << result.diagnostics;
}
