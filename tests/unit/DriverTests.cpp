#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

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
#include "Common/UnwindABI.h"

#include "llvm/IR/Instructions.h"
#include "llvm/TargetParser/Host.h"
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
  EXPECT_FALSE(result.succeeded);
  EXPECT_NE(result.diagnostics.find("contains itself through a union-typed "
                                    "field of two object types"),
            std::string::npos)
      << result.diagnostics;
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
