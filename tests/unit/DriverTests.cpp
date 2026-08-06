#include "Driver.h"
#include "DriverCodeGen.h"

#include "Common/RuntimeLibrary.h"
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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
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

  // The payload box, whose layout the manifest strides through directly
  // (`%c16`, and slots counted from the pointer and size bases).
  EXPECT_EQ(py::lowering::box_abi::kWordsPerBox, 16)
      << "__ly_exc_payload_store multiplies the slot index by 16";
  EXPECT_EQ(py::lowering::box_abi::kPointerWordBase, 4)
      << "the manifest's box helpers index pointer words from 4";
  EXPECT_EQ(py::lowering::box_abi::kSizeWordBase, 9)
      << "the manifest's box helpers index size words from 9";
  EXPECT_EQ(py::lowering::box_abi::kOwnedFlagWord, 14)
      << "the manifest's box helpers write the owned flag at word 14";
}

TEST(DriverTest, RepeatedCompileIsStable) {
  for (int round = 0; round < 3; ++round) {
    CompileResult result = compileSource("print(40 + 2)\n");
    EXPECT_TRUE(result.succeeded) << "round " << round << ": "
                                  << result.diagnostics;
  }
}

} // namespace
