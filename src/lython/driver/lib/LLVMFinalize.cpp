#include "DriverCodeGen.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
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
    if (!existing->isDeclaration()) {
      diag << "error: cannot build executable: symbol 'main' already exists\n";
      return failure();
    }
    existing->eraseFromParent();
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

static std::optional<FileLineColLoc> findPythonSourceLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    if (fileLoc.getFilename().getValue().ends_with(".py"))
      return fileLoc;
    return std::nullopt;
  }
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return findPythonSourceLoc(nameLoc.getChildLoc());
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fused.getLocations())
      if (auto found = findPythonSourceLoc(child))
        return found;
  }
  return std::nullopt;
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
        findPythonSourceLoc(function.getLoc());
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
      if (!findPythonSourceLoc(op->getLoc()))
        return;
      op->setLoc(scopedPythonDebugLoc(op->getLoc(), subprogram));
    });
  });
}

} // namespace lython::driver
