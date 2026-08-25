#include "Common/PythonSourceRange.h"
#include "Common/RuntimeSupport.h"
#include "Common/UnwindABI.h"

#include "Ownership.h"

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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

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
  if (symbolName == "__main__")
    return "<module>";
  // Specialization and generator state-machine clones carry ABI suffixes;
  // the traceback shows the Python-level name.
  for (llvm::StringRef marker : {"__lyrt_prim", "__lyrt_gen"}) {
    std::size_t suffix = symbolName.find(marker);
    if (suffix != llvm::StringRef::npos)
      symbolName = symbolName.take_front(suffix);
  }
  return symbolName.str();
}

// The generator machinery splits one Python generator into a clone family
// (`g__lyrt_gen_resume`, `g__lyrt_gen_resume__step`); an unwind edge between
// members of the same family is state-machine plumbing, not a Python frame.
bool isGeneratorInternalEdge(const llvm::CallInst &call) {
  const llvm::Function *callee = call.getCalledFunction();
  if (!callee)
    return false;
  llvm::StringRef callerName = call.getFunction()->getName();
  llvm::StringRef calleeName = callee->getName();
  std::size_t callerMark = callerName.find("__lyrt_gen");
  std::size_t calleeMark = calleeName.find("__lyrt_gen");
  if (callerMark == llvm::StringRef::npos ||
      calleeMark == llvm::StringRef::npos)
    return false;
  // Why family-root comparison instead of caller/callee prefixing: sibling
  // drivers (`__throw` calling `__advance` calling `__step`) share the root
  // but neither name prefixes the other, and their edges are exactly the
  // plumbing this predicate must hide.
  return callerName.take_front(callerMark) ==
         calleeName.take_front(calleeMark);
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

// Refcount maintenance can never raise a Python exception; the ownership
// inserter is free to place these between a try call-site marker and the
// call it marks, so the EH pass must neither invoke-convert them nor let
// them break the marker/call adjacency.
//
// The set itself lives in common/ because the ownership pass reads it too --
// the generated source-class deallocators and unwind cleanup thunks are pure
// release compositions (DecRef family + free), and treating them as pairing
// breakers here silently unpaired a raise from its handler while treating them
// as may-raise there demanded a cleanup around the call that IS the cleanup.
bool isNonUnwindingRefcountHelper(const llvm::Function *callee) {
  return callee &&
         py::ownership::isRefcountMaintenanceSymbol(callee->getName());
}

bool mayPropagatePythonException(const llvm::Function *callee) {
  if (isPythonDebugFunction(callee))
    return true;
  if (!callee || callee->isDeclaration() || callee->isIntrinsic() ||
      callee->doesNotThrow())
    return false;
  llvm::StringRef name = callee->getName();
  // A raise primitive is where an exception STARTS; the unwind edge out of it
  // is `isPythonRuntimeRaiseCall`'s business, not a propagation.
  if (name == "LyEH_ThrowException" || name.ends_with("_Raise"))
    return false;
  if (py::ownership::isNonRaisingRuntimeSymbol(name))
    return false;
  return name.starts_with("Ly");
}

bool isPythonRuntimeRaiseCall(const llvm::Function *callee) {
  if (!callee)
    return false;
  llvm::StringRef name = callee->getName();
  return name == "LyEH_ThrowException" || name == "LyEH_RethrowCurrent" ||
         name == "LyEH_StarRethrowResidual" ||
         name == "LyEH_StarRethrowSoleCollected" ||
         name == "LyEH_StarThrowCombined" || name.ends_with("_Raise");
}

bool mayTransferToPythonTryHandler(const llvm::Function *callee) {
  return isPythonRuntimeRaiseCall(callee) || mayPropagatePythonException(callee);
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
        return tracebackFunctionName(name);
      llvm::StringRef linkage = subprogram->getLinkageName();
      if (!linkage.empty())
        return tracebackFunctionName(linkage);
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

// ⛔ Takes the triple rather than reading `module.getTargetTriple()`. This pass
// runs on the LLVM IR the moment it is translated, BEFORE the runtime support
// module is linked in and before the driver stamps the triple on it -- so
// neither the definition nor the triple is in the module to ask. Both sides
// call UnwindABI.h with the same triple instead.
llvm::Constant *pythonPersonality(llvm::Module &module,
                                  const llvm::Triple &triple) {
  llvm::LLVMContext &context = module.getContext();
  llvm::FunctionType *type =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context),
                              /*isVarArg=*/true);
  return llvm::cast<llvm::Constant>(
      module
          .getOrInsertFunction(py::runtime_library::personalityNameFor(triple),
                               type)
          .getCallee());
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

  // A raise primitive already pushed its own raise-site frame during MLIR
  // lowering; pushing the invoke site again would duplicate the frame (and
  // CPython shows no caret anchors on raise lines).
  if (!isPythonRuntimeRaiseCall(call.getCalledFunction()) &&
      !isGeneratorInternalEdge(call))
    emitTracebackPush(builder, *module, site, debugLoc);
  builder.CreateResume(landingPad);
}

// The Python classes a try's `except` arms test for, read back off the dispatch
// chain the lowering already built: a run of blocks each holding one
// `LyEH_CurrentExceptionMatches(<constant class id>)` and branching on it, ending
// in the re-raise that runs when no arm matched.
//
// ⛔ Returns nothing unless the whole chain reads that way. The clause list this
// feeds is what the personality uses to decide the frame is NOT entered -- a
// list missing one class is an exception flying past a handler that would have
// caught it. A pad with no list is a catch-all, which is what every pad used to
// be: slower, never wrong.
// The markers this pass erases before it returns. They are in the blocks the
// walk below reads, and they are not there afterwards.
bool isErasedTryMarker(const llvm::Instruction &instruction) {
  const auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
  if (!call || !call->getCalledFunction())
    return false;
  llvm::StringRef name = call->getCalledFunction()->getName();
  return name == "LyEH_TryCatchMarker" || name == "LyEH_TryCallSiteMarker" ||
         name == "LyEH_TryCatchAnchor";
}

// The only instructions a block may hold and still count as "does nothing for
// an exception it does not name".
bool holdsOnly(llvm::BasicBlock &block,
               llvm::ArrayRef<const llvm::Instruction *> allowed) {
  for (llvm::Instruction &instruction : block) {
    if (instruction.isDebugOrPseudoInst() || isErasedTryMarker(instruction))
      continue;
    if (!llvm::is_contained(allowed, &instruction))
      return false;
  }
  return true;
}

std::optional<llvm::SmallVector<std::int64_t, 4>>
handledClassIds(llvm::BasicBlock *dispatch) {
  llvm::SmallVector<std::int64_t, 4> ids;
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> seen;
  llvm::BasicBlock *block = dispatch;
  while (block && seen.insert(block).second) {
    auto *branch = llvm::dyn_cast<llvm::BranchInst>(block->getTerminator());
    // The pad lands on a forwarding block; following it is following control
    // flow, not assuming anything about what runs.
    if (branch && branch->isUnconditional() && holdsOnly(*block, {branch})) {
      block = branch->getSuccessor(0);
      continue;
    }
    if (branch && branch->isConditional()) {
      auto *test = llvm::dyn_cast<llvm::CallInst>(branch->getCondition());
      if (!test || !isRuntimeMarkerCall(*test, "LyEH_CurrentExceptionMatches") ||
          test->getParent() != block)
        return std::nullopt;
      // ⛔ And NOTHING ELSE in the block. A generator's resume step marks itself
      // dead here before testing the arms, and that store has to happen for
      // every exception, not only the ones the arms name -- a frame with work
      // in front of its dispatch is a frame that must be entered.
      if (!holdsOnly(*block, {test, branch}))
        return std::nullopt;
      auto *classId = llvm::dyn_cast<llvm::ConstantInt>(test->getArgOperand(0));
      if (!classId)
        return std::nullopt;
      ids.push_back(classId->getSExtValue());
      block = branch->getSuccessor(1);
      continue;
    }
    // The end of the chain: nothing matched, so the exception carries straight
    // on. Re-raising FIRST is what proves the frame does nothing for an
    // exception outside the list -- a `finally` puts its body here instead, and
    // that frame really is entered for everything, so it stays a catch-all.
    llvm::Instruction *tail = nullptr;
    for (llvm::Instruction &instruction : *block)
      if (!instruction.isDebugOrPseudoInst() && !isErasedTryMarker(instruction)) {
        tail = &instruction;
        break;
      }
    auto *rethrow = llvm::dyn_cast_or_null<llvm::CallBase>(tail);
    if (rethrow && rethrow->getCalledFunction() &&
        rethrow->getCalledFunction()->getName() == "LyEH_RethrowCurrent")
      return ids.empty() ? std::nullopt : std::optional(ids);
    return std::nullopt;
  }
  return std::nullopt;
}

// One global per class id, holding the id in its first word. The type table
// entries in the LSDA point at these, and the personality loads the word.
llvm::Constant *exceptionTypeGlobal(llvm::Module &module, std::int64_t classId) {
  std::string name = ("__ly_exc_type_" + llvm::Twine(classId)).str();
  if (llvm::GlobalVariable *existing = module.getNamedGlobal(name))
    return existing;
  llvm::Type *word = llvm::Type::getInt64Ty(module.getContext());
  auto *global = new llvm::GlobalVariable(
      module, word, /*isConstant=*/true, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantInt::get(word, classId), name);
  global->setAlignment(llvm::Align(8));
  return global;
}

// The pad for a try's catch dispatch. Naming the classes lets the personality
// answer during the SEARCH phase, which is what keeps a frame that does not
// handle this exception from being entered at all.
//
// ⛔ Such a pad is ALSO a cleanup, and that is not optional. A Python frame puts
// itself in the traceback by being entered; a frame skipped for not handling the
// exception would vanish from what the program prints, which is a wrong answer
// rather than a slow one. The cleanup entry brings it back in the second phase,
// where the selector tells the two apart.
llvm::LandingPadInst *createCatchLandingPad(llvm::IRBuilder<> &builder,
                                            llvm::StringRef name,
                                            llvm::BasicBlock *dispatch) {
  llvm::LLVMContext &context = builder.getContext();
  llvm::StructType *landingPadType = llvm::StructType::get(
      llvm::PointerType::getUnqual(context), llvm::Type::getInt32Ty(context));
  std::optional<llvm::SmallVector<std::int64_t, 4>> ids =
      handledClassIds(dispatch);
  llvm::LandingPadInst *landingPad =
      builder.CreateLandingPad(landingPadType, ids ? ids->size() : 1, name);
  if (!ids) {
    landingPad->addClause(
        llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(context)));
    return landingPad;
  }
  llvm::Module &module = *builder.GetInsertBlock()->getModule();
  for (std::int64_t classId : *ids)
    landingPad->addClause(exceptionTypeGlobal(module, classId));
  landingPad->setCleanup(true);
  return landingPad;
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
      createCatchLandingPad(builder, "py.catch.lpad", catchDest);
  llvm::Value *exceptionObject =
      builder.CreateExtractValue(landingPad, {0}, "py.catch.exception");
  // Same rule as the cleanup pads: raise primitives already recorded their
  // raise-site frame during MLIR lowering.
  bool recordsFrame = !isPythonRuntimeRaiseCall(call.getCalledFunction()) &&
                      !isGeneratorInternalEdge(call);

  if (landingPad->isCleanup()) {
    llvm::BasicBlock *passing = llvm::BasicBlock::Create(
        context, "py.try.passing", function, catchDest);
    llvm::BasicBlock *handling = llvm::BasicBlock::Create(
        context, "py.try.handling", function, catchDest);
    llvm::Value *selector =
        builder.CreateExtractValue(landingPad, {1}, "py.catch.selector");
    builder.CreateCondBr(
        builder.CreateICmpEQ(selector,
                             llvm::ConstantInt::get(selector->getType(), 0)),
        passing, handling);

    builder.SetInsertPoint(passing);
    if (recordsFrame)
      emitTracebackPush(builder, *module, site, debugLoc);
    builder.CreateResume(landingPad);

    builder.SetInsertPoint(handling);
  }

  builder.CreateCall(beginPythonCatch(*module), {exceptionObject});
  if (recordsFrame)
    emitTracebackPush(builder, *module, site, debugLoc);
  builder.CreateBr(catchDest);
  return landing;
}

// A call that may reach a Python handler becomes an invoke the same way in
// both directions: split after the call, build the unwind destination, and
// carry the call's convention, attributes and debug location across. The only
// difference is where it unwinds -- a cleanup that re-raises, or the enclosing
// try's catch dispatch.
bool convertCallToPythonInvoke(
    llvm::CallInst &call, const llvm::Triple &triple,
    llvm::ArrayRef<PythonCallSiteRange> callSites,
    llvm::function_ref<llvm::BasicBlock *(llvm::BasicBlock *,
                                          llvm::DILocation &)>
        buildUnwindDest) {
  if (call.isInlineAsm() || call.getNumOperandBundles() != 0)
    return false;
  llvm::Function *callee = call.getCalledFunction();
  if (!mayTransferToPythonTryHandler(callee))
    return false;
  llvm::DebugLoc debugLocation = call.getDebugLoc();
  auto *debugLoc =
      llvm::dyn_cast_or_null<llvm::DILocation>(debugLocation.get());
  if (!debugLoc)
    return false;

  llvm::Function *function = call.getFunction();
  function->setPersonalityFn(
      pythonPersonality(*function->getParent(), triple));

  auto splitPoint = call.getIterator();
  ++splitPoint;
  llvm::BasicBlock *block = call.getParent();
  llvm::BasicBlock *normalDest =
      block->splitBasicBlock(splitPoint, "py.invoke.cont");
  llvm::Instruction *oldBranch = block->getTerminator();
  llvm::BasicBlock *unwindDest = buildUnwindDest(normalDest, *debugLoc);

  llvm::IRBuilder<> builder(oldBranch);
  llvm::SmallVector<llvm::Value *, 8> args(call.args());
  llvm::InvokeInst *invoke =
      builder.CreateInvoke(call.getFunctionType(), call.getCalledOperand(),
                           normalDest, unwindDest, args, call.getName());
  invoke->setCallingConv(call.getCallingConv());
  invoke->setAttributes(call.getAttributes());
  invoke->setDebugLoc(debugLocation);

  if (!call.getType()->isVoidTy())
    call.replaceAllUsesWith(invoke);
  call.eraseFromParent();
  oldBranch->eraseFromParent();
  return true;
}

bool convertCallToPythonInvoke(llvm::CallInst &call,
                               const llvm::Triple &triple,
                               llvm::ArrayRef<PythonCallSiteRange> callSites) {
  return convertCallToPythonInvoke(
      call, triple, callSites,
      [&](llvm::BasicBlock *normalDest, llvm::DILocation &debugLoc) {
        buildPythonCleanupBlock(call, normalDest, debugLoc,
                                matchCallSiteRange(call, callSites, debugLoc));
        // buildPythonCleanupBlock inserts ahead of the continuation.
        return normalDest->getPrevNode();
      });
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
    llvm::CallInst &call, const llvm::Triple &triple,
    const PythonTryCallMarker &marker,
    llvm::ArrayRef<PythonCallSiteRange> callSites) {
  return convertCallToPythonInvoke(
      call, triple, callSites,
      [&](llvm::BasicBlock *, llvm::DILocation &debugLoc) {
        return buildPythonCatchDispatchBlock(
            call, marker.catchBlock, debugLoc,
            matchCallSiteRange(call, callSites, debugLoc));
      });
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
    llvm::Module &module, const llvm::Triple &triple,
    llvm::ArrayRef<PythonCallSiteRange> callSites) {
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
          if (mayTransferToPythonTryHandler(call->getCalledFunction())) {
            tryCallSites[call] = *pendingTryMarker;
            pendingTryMarker.reset();
          } else if (isNonUnwindingRefcountHelper(call->getCalledFunction()) ||
                     isRuntimeMarkerCall(*call, "LyTraceback_Push")) {
            // Ownership insertion may schedule releases (and the raise path
            // its traceback frame) between the marker and the marked call;
            // they cannot unwind, so the marker stays pending.
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
        if (convertCallToPythonTryInvoke(*call, triple, marker, callSites)) {
          changed = true;
          continue;
        }
      }
    }
    changed |= convertCallToPythonInvoke(*call, triple, callSites);
  }
  for (auto &entry : catchTargets)
    if (entry.second.marker)
      entry.second.marker->eraseFromParent();
  for (llvm::CallInst *marker : callSiteMarkers)
    marker->eraseFromParent();
  return changed;
}

} // namespace py
