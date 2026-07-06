#include "Common/PythonSourceRange.h"
#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py {
namespace {

struct PythonCatchTarget {
  llvm::BasicBlock *block = nullptr;
  llvm::CallInst *marker = nullptr;
};

struct PythonTryCallMarker {
  std::int64_t id = 0;
  llvm::CallInst *marker = nullptr;
  llvm::BasicBlock *catchBlock = nullptr;
};

struct PendingPythonTryCallMarker {
  std::int64_t id = 0;
  llvm::CallInst *marker = nullptr;
};

std::string tracebackFunctionName(llvm::StringRef symbolName) {
  return symbolName == "__main__" ? std::string("<module>") : symbolName.str();
}

bool isPythonFunction(mlir::LLVM::LLVMFuncOp function) {
  return static_cast<bool>(findPythonSourceLoc(function.getLoc()));
}

bool isPythonDebugFunction(const llvm::Function *function) {
  if (!function || function->isDeclaration())
    return false;
  llvm::DISubprogram *subprogram = function->getSubprogram();
  if (!subprogram)
    return false;
  llvm::DICompileUnit *unit = subprogram->getUnit();
  if (!unit)
    return false;
  llvm::DISourceLanguageName language = unit->getSourceLanguage();
  return language.getName() == llvm::dwarf::DW_LANG_Python ||
         (language.hasVersionedName() &&
          language.getName() == llvm::dwarf::DW_LNAME_Python);
}

bool mayPropagatePythonException(const llvm::Function *callee) {
  if (isPythonDebugFunction(callee))
    return true;
  if (!callee || callee->isDeclaration() || callee->isIntrinsic() ||
      callee->doesNotThrow())
    return false;
  llvm::StringRef name = callee->getName();
  if (name == "LyEH_ThrowException" || name.ends_with("_Raise") ||
      name == "LyEH_BeginCatch" || name == "LyEH_ClassIdMatches" ||
      name == "LyEH_CurrentExceptionClassId" ||
      name == "LyEH_CurrentExceptionMatches" ||
      name == "LyEH_DiscardCurrentExceptionIfMatches" ||
      name == "LyEH_DiscardCurrentException" ||
      name == "LyEH_TryCallSiteMarker" || name == "LyEH_TryCatchMarker" ||
      name == "LyEH_TryCatchAnchor" || name.starts_with("LyTraceback_"))
    return false;
  return name.starts_with("Ly");
}

bool isRuntimeMarkerCall(const llvm::CallInst &call, llvm::StringRef name) {
  const llvm::Function *callee = call.getCalledFunction();
  return callee && callee->getName() == name;
}

std::optional<std::int64_t> i64ConstantArgument(const llvm::CallInst &call,
                                                unsigned index) {
  if (index >= call.arg_size())
    return std::nullopt;
  auto *constant = llvm::dyn_cast<llvm::ConstantInt>(call.getArgOperand(index));
  if (!constant)
    return std::nullopt;
  return constant->getSExtValue();
}

bool canSkipBetweenTryMarkerAndCall(const llvm::Instruction &instruction) {
  if (llvm::isa<llvm::DbgInfoIntrinsic>(&instruction))
    return true;
  if (!instruction.mayHaveSideEffects())
    return true;
  const auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
  if (!intrinsic)
    return false;
  switch (intrinsic->getIntrinsicID()) {
  case llvm::Intrinsic::lifetime_start:
  case llvm::Intrinsic::lifetime_end:
  case llvm::Intrinsic::assume:
    return true;
  default:
    return false;
  }
}

std::string debugLocationPath(const llvm::DILocation &loc) {
  llvm::SmallString<256> path(loc.getDirectory());
  llvm::sys::path::append(path, loc.getFilename());
  return path.str().str();
}

std::string debugLocationFunctionName(const llvm::DILocation &loc) {
  if (auto *scope = llvm::dyn_cast_or_null<llvm::DILocalScope>(loc.getScope()))
    if (llvm::DISubprogram *subprogram = scope->getSubprogram()) {
      llvm::StringRef name = subprogram->getName();
      if (!name.empty())
        return name.str();
      llvm::StringRef linkage = subprogram->getLinkageName();
      if (!linkage.empty())
        return linkage.str();
    }
  return "<unknown>";
}

llvm::FunctionCallee tracebackPushCStringRange(llvm::Module &module) {
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *ptr = llvm::PointerType::getUnqual(context);
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::FunctionType *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {ptr, ptr, i32, i32, i32, i32},
      /*isVarArg=*/false);
  return module.getOrInsertFunction("LyTraceback_PushCStringRange", type);
}

llvm::FunctionCallee beginPythonCatch(llvm::Module &module) {
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *ptr = llvm::PointerType::getUnqual(context);
  llvm::FunctionType *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {ptr}, /*isVarArg=*/false);
  return module.getOrInsertFunction("LyEH_BeginCatch", type);
}

llvm::Constant *gxxPersonality(llvm::Module &module) {
  llvm::LLVMContext &context = module.getContext();
  llvm::FunctionType *type =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context),
                              /*isVarArg=*/true);
  return llvm::cast<llvm::Constant>(
      module.getOrInsertFunction("__gxx_personality_v0", type).getCallee());
}

llvm::Constant *globalCStringPtr(llvm::IRBuilder<> &builder,
                                 llvm::StringRef text, llvm::StringRef name) {
  llvm::GlobalVariable *global = builder.CreateGlobalString(text, name);
  llvm::Constant *zero =
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(builder.getContext()), 0);
  llvm::Constant *indices[] = {zero, zero};
  return llvm::ConstantExpr::getInBoundsGetElementPtr(global->getValueType(),
                                                      global, indices);
}

const PythonCallSiteRange *
matchCallSiteRange(llvm::CallInst &call,
                   llvm::ArrayRef<PythonCallSiteRange> callSites,
                   const llvm::DILocation &debugLoc) {
  llvm::Function *callee = call.getCalledFunction();
  if (!callee)
    return nullptr;

  llvm::StringRef callerName = call.getFunction()->getName();
  llvm::StringRef calleeName = callee->getName();
  const PythonCallSiteRange *lineMatch = nullptr;
  for (const PythonCallSiteRange &site : callSites) {
    if (callerName != site.caller || calleeName != site.callee)
      continue;
    if (site.line != static_cast<std::int32_t>(debugLoc.getLine()))
      continue;
    if (site.column == static_cast<std::int32_t>(debugLoc.getColumn()))
      return &site;
    if (!lineMatch)
      lineMatch = &site;
  }
  return lineMatch;
}

llvm::Value *i32Constant(llvm::IRBuilder<> &builder, std::int32_t value) {
  return llvm::ConstantInt::get(llvm::Type::getInt32Ty(builder.getContext()),
                                static_cast<std::uint32_t>(value));
}

void buildPythonCleanupBlock(llvm::CallInst &call, llvm::BasicBlock *unwindDest,
                             llvm::DILocation &debugLoc,
                             const PythonCallSiteRange *site) {
  llvm::Function *function = call.getFunction();
  llvm::Module *module = function->getParent();
  llvm::LLVMContext &context = module->getContext();
  llvm::BasicBlock *cleanup = llvm::BasicBlock::Create(
      context, "py.traceback.cleanup", function, unwindDest);
  llvm::IRBuilder<> builder(cleanup);
  llvm::StructType *landingPadType = llvm::StructType::get(
      llvm::PointerType::getUnqual(context), llvm::Type::getInt32Ty(context));
  llvm::LandingPadInst *landingPad =
      builder.CreateLandingPad(landingPadType, 0, "py.lpad");
  landingPad->setCleanup(true);

  std::string fallbackFile = debugLocationPath(debugLoc);
  std::string fallbackFunction = debugLocationFunctionName(debugLoc);
  llvm::StringRef fileName =
      site ? llvm::StringRef(site->filename) : llvm::StringRef(fallbackFile);
  llvm::StringRef functionName = site ? llvm::StringRef(site->functionName)
                                      : llvm::StringRef(fallbackFunction);
  std::int32_t line =
      site ? site->line : static_cast<std::int32_t>(debugLoc.getLine());
  std::int32_t column =
      site ? site->column : static_cast<std::int32_t>(debugLoc.getColumn());
  std::int32_t endLine = site ? site->endLine : line;
  std::int32_t endColumn = site ? site->endColumn : 0;

  llvm::Value *file = globalCStringPtr(builder, fileName, "py.tb.file");
  llvm::Value *name = globalCStringPtr(builder, functionName, "py.tb.func");
  builder.CreateCall(
      tracebackPushCStringRange(*module),
      {file, name, i32Constant(builder, line), i32Constant(builder, column),
       i32Constant(builder, endLine), i32Constant(builder, endColumn)});
  builder.CreateResume(landingPad);
}

llvm::LandingPadInst *createCatchAllLandingPad(llvm::IRBuilder<> &builder,
                                               llvm::StringRef name) {
  llvm::LLVMContext &context = builder.getContext();
  llvm::StructType *landingPadType = llvm::StructType::get(
      llvm::PointerType::getUnqual(context), llvm::Type::getInt32Ty(context));
  llvm::LandingPadInst *landingPad =
      builder.CreateLandingPad(landingPadType, 1, name);
  landingPad->addClause(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(context)));
  return landingPad;
}

void emitTracebackPush(llvm::IRBuilder<> &builder, llvm::Module &module,
                       const PythonCallSiteRange *site,
                       llvm::DILocation &debugLoc) {
  std::string fallbackFile = debugLocationPath(debugLoc);
  std::string fallbackFunction = debugLocationFunctionName(debugLoc);
  llvm::StringRef fileName =
      site ? llvm::StringRef(site->filename) : llvm::StringRef(fallbackFile);
  llvm::StringRef functionName = site ? llvm::StringRef(site->functionName)
                                      : llvm::StringRef(fallbackFunction);
  std::int32_t line =
      site ? site->line : static_cast<std::int32_t>(debugLoc.getLine());
  std::int32_t column =
      site ? site->column : static_cast<std::int32_t>(debugLoc.getColumn());
  std::int32_t endLine = site ? site->endLine : line;
  std::int32_t endColumn = site ? site->endColumn : 0;

  llvm::Value *file = globalCStringPtr(builder, fileName, "py.tb.file");
  llvm::Value *name = globalCStringPtr(builder, functionName, "py.tb.func");
  builder.CreateCall(
      tracebackPushCStringRange(module),
      {file, name, i32Constant(builder, line), i32Constant(builder, column),
       i32Constant(builder, endLine), i32Constant(builder, endColumn)});
}

llvm::BasicBlock *
buildPythonCatchDispatchBlock(llvm::CallInst &call, llvm::BasicBlock *catchDest,
                              llvm::DILocation &debugLoc,
                              const PythonCallSiteRange *site) {
  llvm::Function *function = call.getFunction();
  llvm::Module *module = function->getParent();
  llvm::LLVMContext &context = module->getContext();
  llvm::BasicBlock *landing =
      llvm::BasicBlock::Create(context, "py.try.catch", function, catchDest);
  llvm::IRBuilder<> builder(landing);
  llvm::LandingPadInst *landingPad =
      createCatchAllLandingPad(builder, "py.catch.lpad");
  llvm::Value *exceptionObject =
      builder.CreateExtractValue(landingPad, {0}, "py.catch.exception");
  builder.CreateCall(beginPythonCatch(*module), {exceptionObject});
  emitTracebackPush(builder, *module, site, debugLoc);
  builder.CreateBr(catchDest);
  return landing;
}

bool convertCallToPythonInvoke(llvm::CallInst &call,
                               llvm::ArrayRef<PythonCallSiteRange> callSites) {
  if (call.isInlineAsm() || call.getNumOperandBundles() != 0)
    return false;
  llvm::Function *callee = call.getCalledFunction();
  if (!mayPropagatePythonException(callee))
    return false;
  llvm::DebugLoc debugLocation = call.getDebugLoc();
  auto *debugLoc =
      llvm::dyn_cast_or_null<llvm::DILocation>(debugLocation.get());
  if (!debugLoc)
    return false;

  llvm::Function *function = call.getFunction();
  function->setPersonalityFn(gxxPersonality(*function->getParent()));

  auto splitPoint = call.getIterator();
  ++splitPoint;
  llvm::BasicBlock *block = call.getParent();
  llvm::BasicBlock *normalDest =
      block->splitBasicBlock(splitPoint, "py.invoke.cont");
  llvm::Instruction *oldBranch = block->getTerminator();
  buildPythonCleanupBlock(call, normalDest, *debugLoc,
                          matchCallSiteRange(call, callSites, *debugLoc));
  llvm::BasicBlock *cleanupDest = normalDest->getPrevNode();

  llvm::IRBuilder<> builder(oldBranch);
  llvm::SmallVector<llvm::Value *, 8> args(call.args());
  llvm::InvokeInst *invoke =
      builder.CreateInvoke(call.getFunctionType(), call.getCalledOperand(),
                           normalDest, cleanupDest, args, call.getName());
  invoke->setCallingConv(call.getCallingConv());
  invoke->setAttributes(call.getAttributes());
  invoke->setDebugLoc(debugLocation);

  if (!call.getType()->isVoidTy())
    call.replaceAllUsesWith(invoke);
  call.eraseFromParent();
  oldBranch->eraseFromParent();
  return true;
}

bool rewriteTryCatchAnchor(llvm::CallInst &call) {
  if (!isRuntimeMarkerCall(call, "LyEH_TryCatchAnchor"))
    return false;

  if (call.hasOneUse()) {
    if (auto *branch = llvm::dyn_cast<llvm::BranchInst>(*call.user_begin())) {
      if (branch->isConditional() && branch->getCondition() == &call) {
        llvm::BasicBlock *tryDest = branch->getSuccessor(1);
        llvm::IRBuilder<> builder(branch);
        builder.CreateBr(tryDest);
        branch->eraseFromParent();
        call.eraseFromParent();
        return true;
      }
    }
  }

  call.replaceAllUsesWith(
      llvm::ConstantInt::getFalse(call.getFunction()->getContext()));
  if (call.use_empty()) {
    call.eraseFromParent();
    return true;
  }
  return false;
}

bool convertCallToPythonTryInvoke(
    llvm::CallInst &call, const PythonTryCallMarker &marker,
    llvm::ArrayRef<PythonCallSiteRange> callSites) {
  if (call.isInlineAsm() || call.getNumOperandBundles() != 0)
    return false;
  llvm::Function *callee = call.getCalledFunction();
  if (!mayPropagatePythonException(callee))
    return false;
  llvm::DebugLoc debugLocation = call.getDebugLoc();
  auto *debugLoc =
      llvm::dyn_cast_or_null<llvm::DILocation>(debugLocation.get());
  if (!debugLoc)
    return false;

  llvm::Function *function = call.getFunction();
  function->setPersonalityFn(gxxPersonality(*function->getParent()));

  auto splitPoint = call.getIterator();
  ++splitPoint;
  llvm::BasicBlock *block = call.getParent();
  llvm::BasicBlock *normalDest =
      block->splitBasicBlock(splitPoint, "py.invoke.cont");
  llvm::Instruction *oldBranch = block->getTerminator();
  llvm::BasicBlock *catchDispatch = buildPythonCatchDispatchBlock(
      call, marker.catchBlock, *debugLoc,
      matchCallSiteRange(call, callSites, *debugLoc));

  llvm::IRBuilder<> builder(oldBranch);
  llvm::SmallVector<llvm::Value *, 8> args(call.args());
  llvm::InvokeInst *invoke =
      builder.CreateInvoke(call.getFunctionType(), call.getCalledOperand(),
                           normalDest, catchDispatch, args, call.getName());
  invoke->setCallingConv(call.getCallingConv());
  invoke->setAttributes(call.getAttributes());
  invoke->setDebugLoc(debugLocation);

  if (!call.getType()->isVoidTy())
    call.replaceAllUsesWith(invoke);
  call.eraseFromParent();
  oldBranch->eraseFromParent();
  return true;
}

} // namespace

void collectPythonCallSiteRanges(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<PythonCallSiteRange> &callSites) {
  module.walk([&](mlir::LLVM::CallOp call) {
    std::optional<llvm::StringRef> calleeName = call.getCallee();
    if (!calleeName)
      return;
    auto caller = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (!caller)
      return;
    if (!isPythonFunction(caller))
      return;
    std::optional<PythonSourceRange> source = pythonSourceRange(call.getLoc());
    if (!source)
      return;

    PythonCallSiteRange site;
    site.caller = caller.getSymName().str();
    site.callee = calleeName->str();
    site.filename = source->filename;
    site.functionName = tracebackFunctionName(caller.getSymName());
    site.line = source->line;
    site.column = source->column;
    site.endLine = source->endLine;
    site.endColumn = source->endColumn;
    callSites.push_back(std::move(site));
  });
}

bool installPythonExceptionCleanupFrames(
    llvm::Module &module, llvm::ArrayRef<PythonCallSiteRange> callSites) {
  llvm::SmallVector<llvm::CallInst *, 16> calls;
  llvm::SmallVector<llvm::CallInst *, 8> anchors;
  llvm::SmallVector<llvm::CallInst *, 16> callSiteMarkers;
  llvm::DenseMap<std::int64_t, PythonCatchTarget> catchTargets;
  llvm::DenseMap<llvm::CallInst *, PendingPythonTryCallMarker> tryCallSites;
  for (llvm::Function &function : module) {
    if (!isPythonDebugFunction(&function))
      continue;
    for (llvm::BasicBlock &block : function) {
      std::optional<PendingPythonTryCallMarker> pendingTryMarker;
      for (llvm::Instruction &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        if (!call) {
          if (pendingTryMarker && !canSkipBetweenTryMarkerAndCall(instruction))
            pendingTryMarker.reset();
          continue;
        }
        if (isRuntimeMarkerCall(*call, "LyEH_TryCatchMarker")) {
          if (std::optional<std::int64_t> id = i64ConstantArgument(*call, 0))
            catchTargets[*id] = PythonCatchTarget{&block, call};
          pendingTryMarker.reset();
          continue;
        }
        if (isRuntimeMarkerCall(*call, "LyEH_TryCallSiteMarker")) {
          callSiteMarkers.push_back(call);
          if (std::optional<std::int64_t> id = i64ConstantArgument(*call, 0))
            pendingTryMarker = PendingPythonTryCallMarker{*id, call};
          else
            pendingTryMarker.reset();
          continue;
        }
        if (isRuntimeMarkerCall(*call, "LyEH_TryCatchAnchor")) {
          anchors.push_back(call);
          pendingTryMarker.reset();
          continue;
        }
        if (pendingTryMarker) {
          if (mayPropagatePythonException(call->getCalledFunction())) {
            tryCallSites[call] = *pendingTryMarker;
            pendingTryMarker.reset();
          } else if (!canSkipBetweenTryMarkerAndCall(*call)) {
            pendingTryMarker.reset();
          }
        }
        calls.push_back(call);
      }
    }
  }

  bool changed = false;
  for (llvm::CallInst *anchor : anchors)
    changed |= rewriteTryCatchAnchor(*anchor);
  for (llvm::CallInst *call : calls) {
    auto markerInfo = tryCallSites.find(call);
    if (markerInfo != tryCallSites.end()) {
      auto target = catchTargets.find(markerInfo->second.id);
      if (target != catchTargets.end() && target->second.block) {
        PythonTryCallMarker marker{markerInfo->second.id,
                                   markerInfo->second.marker,
                                   target->second.block};
        if (convertCallToPythonTryInvoke(*call, marker, callSites)) {
          changed = true;
          continue;
        }
      }
    }
    changed |= convertCallToPythonInvoke(*call, callSites);
  }
  for (auto &entry : catchTargets)
    if (entry.second.marker)
      entry.second.marker->eraseFromParent();
  for (llvm::CallInst *marker : callSiteMarkers)
    marker->eraseFromParent();
  return changed;
}

} // namespace py
