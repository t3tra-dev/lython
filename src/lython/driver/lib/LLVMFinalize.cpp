#include "DriverCodeGen.h"

#include "Common/PythonSourceRange.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <optional>
#include <string>
#include <system_error>

using namespace mlir;

namespace lython::driver {

// The box-layout helpers are the manifest's only spelling of the payload box's
// width and word offsets, so they are called from every function that touches a
// box -- including `__ly_slot_less` and the container copies. At the JIT's
// default `-jit-opt=0` LLVM inlines nothing it is not told to, so without this
// the aggregation that made the layout checkable would put a call on every box
// word. AlwaysInliner runs at every optimisation level.
//
// ⭐ AND THE STR COPY HELPERS, for a different reason: they are called PER RUN,
// and a run is short. `str.replace` moves the spans between matches through
// `__ly_unicode_copy_run`, which decides on the width and then calls
// `__ly_unicode_copy_bytes` -- two calls and a branch to move fourteen bytes,
// where CPython emits a memcpy the compiler can see through. Inlining both puts
// the width test where the caller's width is a loop invariant, so it is decided
// once rather than per run.
//
// ⛔ NOT A GENERAL "inline the hot manifest functions". Every function marked
// here is copied into every caller at every optimisation level; these two are
// ten lines each and are called from loops whose trip count is the run length.
void markBoxLayoutHelpersAlwaysInline(llvm::Module &module) {
  for (llvm::Function &function : module) {
    if (function.isDeclaration())
      continue;
    llvm::StringRef name = function.getName();
    if (name.starts_with("__ly_box_") || name == "__ly_unicode_copy_run" ||
        name == "__ly_unicode_copy_bytes" || name == "__ly_unicode_get" ||
        name == "__ly_unicode_put")
      function.addFnAttr(llvm::Attribute::AlwaysInline);
  }
}

unsigned redirectAllocationsToObjectAllocator(llvm::Module &module,
                                              bool bypass) {
  markBoxLayoutHelpersAlwaysInline(module);
  if (bypass)
    return 0;
  struct Redirect {
    const char *from;
    const char *to;
  };
  static constexpr Redirect kRedirects[] = {{"malloc", "LyMem_Alloc"},
                                            {"free", "LyMem_Free"},
                                            {"realloc", "LyMem_Realloc"}};
  unsigned moved = 0;
  for (llvm::Function &function : module) {
    // The allocator itself keeps the system allocator: it is what it is built
    // on. Nothing else in the module may reach malloc directly, or a block
    // taken from the pool would be handed to free.
    if (function.getName().starts_with("LyMem_"))
      continue;
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        if (!call)
          continue;
        llvm::Function *callee = call->getCalledFunction();
        if (!callee)
          continue;
        for (const Redirect &redirect : kRedirects) {
          if (callee->getName() != redirect.from)
            continue;
          llvm::Function *replacement = module.getFunction(redirect.to);
          if (!replacement ||
              replacement->getFunctionType() != callee->getFunctionType())
            break;
          call->setCalledFunction(replacement);
          ++moved;
          break;
        }
      }
    }
  }
  return moved;
}

void dumpLLVMForPass(const py::IRDumpConfig &config, llvm::StringRef passName,
                     llvm::Module &module) {
  if (!config.shouldDump(passName))
    return;
  llvm::errs() << "\n=== [LYTHON_IR_DUMP:" << passName << " LLVM] ===\n";
  module.print(llvm::errs(), nullptr);
  llvm::errs() << "\n";
}

LogicalResult writeLLVMIR(llvm::Module &llvmModule, StringRef outputPath,
                          llvm::raw_ostream &diag) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    diag << "Failed to open output file: " << ec.message() << "\n";
    return failure();
  }
  llvmModule.print(out, nullptr);
  return success();
}

LogicalResult installAOTEntryPoint(llvm::Module &llvmModule,
                                   llvm::raw_ostream &diag) {
  llvm::Function *pythonMain = llvmModule.getFunction("__main__");
  if (!pythonMain) {
    diag << "error: cannot build executable: missing __main__ entry\n";
    return failure();
  }
  if (!pythonMain->arg_empty() || pythonMain->isVarArg()) {
    diag << "error: cannot build executable: __main__ must not take "
            "arguments\n";
    return failure();
  }

  if (llvm::Function *existing = llvmModule.getFunction("main")) {
    if (existing->isDeclaration()) {
      existing->eraseFromParent();
    } else {
      // `def main()` IS the ordinary Python idiom, and the definition standing
      // here is the user's function, not a second C entry: the C one is created
      // below and there is nothing else in the module that could have claimed
      // the name. Move it aside instead of refusing the program.
      //
      // Renaming is safe because LLVM references are by Value*, not by symbol
      // name -- every call site keeps pointing at the same Function -- and the
      // symbol has no external meaning: a Python function is reached through
      // this module's own calls, never through the C name.
      //
      // Why NOT mangle Python function symbols in the emitter instead: the
      // clash exists only in the AOT link, since JIT installs no C `main`, so
      // mangling would rename a symbol everywhere for a conflict that arises in
      // one of the two output modes -- and that name is what `--emit-llvm`
      // output and every backtrace are read by.
      existing->setName("__ly_py_main");
    }
  }
  constexpr llvm::StringLiteral kAOTEntryThunkName = "__lython_aot_entry";
  if (llvm::Function *existing = llvmModule.getFunction(kAOTEntryThunkName)) {
    if (!existing->isDeclaration()) {
      diag << "error: cannot build executable: symbol '" << kAOTEntryThunkName
           << "' already exists\n";
      return failure();
    }
    existing->eraseFromParent();
  }

  llvm::LLVMContext &context = llvmModule.getContext();
  llvm::Type *voidTy = llvm::Type::getVoidTy(context);
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *ptr = llvm::PointerType::getUnqual(context);
  llvm::FunctionType *entryThunkType =
      llvm::FunctionType::get(voidTy, /*isVarArg=*/false);
  llvm::Function *entryThunk =
      llvm::Function::Create(entryThunkType, llvm::GlobalValue::InternalLinkage,
                             kAOTEntryThunkName, llvmModule);
  entryThunk->setUWTableKind(llvm::UWTableKind::Async);

  llvm::BasicBlock *thunkBlock =
      llvm::BasicBlock::Create(context, "entry", entryThunk);
  llvm::IRBuilder<> thunkBuilder(thunkBlock);
  thunkBuilder.CreateCall(pythonMain->getFunctionType(), pythonMain, {});
  thunkBuilder.CreateRetVoid();

  llvm::FunctionType *mainType =
      llvm::FunctionType::get(i32, {i32, ptr}, /*isVarArg=*/false);
  llvm::Function *main = llvm::Function::Create(
      mainType, llvm::GlobalValue::ExternalLinkage, "main", llvmModule);
  main->setUWTableKind(llvm::UWTableKind::Async);

  llvm::FunctionType *initArgsType =
      llvm::FunctionType::get(voidTy, {i32, ptr}, /*isVarArg=*/false);
  llvm::FunctionCallee initArgs =
      llvmModule.getOrInsertFunction("LyHost_InitArgs", initArgsType);
  llvm::FunctionType *runnerType =
      llvm::FunctionType::get(i32, {ptr}, /*isVarArg=*/false);
  llvm::FunctionCallee runner =
      llvmModule.getOrInsertFunction("LyRunPythonMain", runnerType);

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", main);
  llvm::IRBuilder<> builder(entry);
  builder.CreateCall(initArgs, {main->getArg(0), main->getArg(1)});
  llvm::CallInst *status = builder.CreateCall(runner, {entryThunk});
  builder.CreateRet(status);
  return success();
}

// Lowers LLVM coroutines and runs a standard LLVM module pipeline. MLIR-level
// passes never ran SROA/mem2reg-class cleanups on the translated IR, so
// without this the descriptor allocas of every lowered object stay in the
// frame (~2KB per object-handling call frame) and nothing is ever inlined.
void runLLVMCoroLowering(llvm::Module &llvmModule,
                         const SanitizerConfig &sanitizers,
                         llvm::TargetMachine *targetMachine,
                         llvm::OptimizationLevel optimizationLevel) {
  llvm::LoopAnalysisManager loopAM;
  llvm::FunctionAnalysisManager functionAM;
  llvm::CGSCCAnalysisManager cgsccAM;
  llvm::ModuleAnalysisManager moduleAM;
  llvm::PassBuilder passBuilder(targetMachine);
  passBuilder.registerModuleAnalyses(moduleAM);
  passBuilder.registerCGSCCAnalyses(cgsccAM);
  passBuilder.registerFunctionAnalyses(functionAM);
  passBuilder.registerLoopAnalyses(loopAM);
  passBuilder.crossRegisterProxies(loopAM, functionAM, cgsccAM, moduleAM);

  llvm::ModulePassManager modulePM =
      passBuilder.buildPerModuleDefaultPipeline(optimizationLevel);
  lython::driver::addSanitizerInstrumentationPasses(modulePM, sanitizers);
  modulePM.run(llvmModule, moduleAM);
}

static llvm::StringRef
exceptionPersonalityForTarget(const llvm::Triple &triple) {
  if (triple.isWindowsGNUEnvironment())
    return "__gxx_personality_seh0";
  return "__gxx_personality_v0";
}

void rewriteExceptionPersonalityForTarget(llvm::Module &llvmModule) {
  llvm::Triple triple(llvmModule.getTargetTriple());
  llvm::StringRef personalityName = exceptionPersonalityForTarget(triple);
  if (personalityName == "__gxx_personality_v0")
    return;

  llvm::LLVMContext &context = llvmModule.getContext();
  llvm::FunctionType *personalityType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(context), /*isVarArg=*/true);
  llvm::Function *personalityFn = llvmModule.getFunction(personalityName);
  if (!personalityFn)
    personalityFn = llvm::Function::Create(personalityType,
                                           llvm::GlobalValue::ExternalLinkage,
                                           personalityName, llvmModule);

  if (llvm::Function *itanium = llvmModule.getFunction("__gxx_personality_v0"))
    itanium->replaceAllUsesWith(personalityFn);
}

struct PythonDebugScopeCache {
  MLIRContext *context;
  llvm::StringMap<LLVM::DIFileAttr> files;
  llvm::StringMap<LLVM::DICompileUnitAttr> compileUnits;

  explicit PythonDebugScopeCache(MLIRContext *context) : context(context) {}

  LLVM::DIFileAttr fileFor(StringRef sourcePath) {
    if (auto found = files.find(sourcePath); found != files.end())
      return found->second;

    StringRef directory = llvm::sys::path::parent_path(sourcePath);
    StringRef basename = llvm::sys::path::filename(sourcePath);
    if (directory.empty())
      directory = ".";
    LLVM::DIFileAttr file = LLVM::DIFileAttr::get(context, basename, directory);
    files[sourcePath] = file;
    return file;
  }

  LLVM::DICompileUnitAttr compileUnitFor(StringRef sourcePath) {
    if (auto found = compileUnits.find(sourcePath); found != compileUnits.end())
      return found->second;

    LLVM::DICompileUnitAttr unit = LLVM::DICompileUnitAttr::get(
        DistinctAttr::create(UnitAttr::get(context)),
        llvm::dwarf::DW_LANG_Python, fileFor(sourcePath),
        StringAttr::get(context, "lython"),
        /*isOptimized=*/true, LLVM::DIEmissionKind::LineTablesOnly);
    compileUnits[sourcePath] = unit;
    return unit;
  }
};

static Location scopedPythonDebugLoc(Location loc,
                                     LLVM::DISubprogramAttr scope) {
  if (loc->findInstanceOf<FusedLocWith<LLVM::DILocalScopeAttr>>())
    return loc;
  return FusedLoc::get(loc.getContext(), {loc}, scope);
}

void attachPythonDebugInfo(ModuleOp module) {
  PythonDebugScopeCache cache(module.getContext());
  LLVM::DINullTypeAttr voidType =
      LLVM::DINullTypeAttr::get(module.getContext());
  LLVM::DISubroutineTypeAttr subroutineType = LLVM::DISubroutineTypeAttr::get(
      module.getContext(), ArrayRef<LLVM::DITypeAttr>{voidType});

  module.walk([&](LLVM::LLVMFuncOp function) {
    if (function.getLoc()
            ->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

    std::optional<FileLineColLoc> sourceLoc =
        py::findPythonSourceLoc(function.getLoc());
    if (!sourceLoc)
      return;

    StringRef sourcePath = sourceLoc->getFilename().getValue();
    LLVM::DIFileAttr file = cache.fileFor(sourcePath);
    LLVM::DICompileUnitAttr compileUnit = cache.compileUnitFor(sourcePath);
    StringRef linkageName = function.getSymName();
    StringRef displayName =
        linkageName == "__main__" ? "<module>" : linkageName;
    uint32_t flagBits =
        static_cast<uint32_t>(LLVM::DISubprogramFlags::Definition) |
        static_cast<uint32_t>(LLVM::DISubprogramFlags::Optimized);
    if (linkageName == "__main__")
      flagBits |=
          static_cast<uint32_t>(LLVM::DISubprogramFlags::MainSubprogram);
    auto flags = static_cast<LLVM::DISubprogramFlags>(flagBits);

    LLVM::DISubprogramAttr subprogram = LLVM::DISubprogramAttr::get(
        module.getContext(),
        DistinctAttr::create(UnitAttr::get(module.getContext())), compileUnit,
        compileUnit, StringAttr::get(module.getContext(), displayName),
        StringAttr::get(module.getContext(), linkageName), file,
        sourceLoc->getLine(), sourceLoc->getLine(), flags, subroutineType,
        ArrayRef<LLVM::DINodeAttr>{}, ArrayRef<LLVM::DINodeAttr>{});

    function->setLoc(scopedPythonDebugLoc(function.getLoc(), subprogram));
    function.walk([&](Operation *op) {
      if (op == function.getOperation())
        return;
      if (!py::findPythonSourceLoc(op->getLoc()))
        return;
      op->setLoc(scopedPythonDebugLoc(op->getLoc(), subprogram));
    });
  });
}

} // namespace lython::driver
