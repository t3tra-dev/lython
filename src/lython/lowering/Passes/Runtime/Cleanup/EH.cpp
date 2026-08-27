#include "Common/PythonSourceRange.h"
#include "Common/RuntimeSupport.h"
#include "Common/UnwindABI.h"

#include "Native.h"

#include "Ownership.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
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
      llvm::Type::getVoidTy(context), {ptr, ptr, i32, i32, i32, i32, i32},
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
  // ⛔ No site means no range: `endColumn` above is 0, and carets drawn from a
  // range that was never recorded would underline something the program does
  // not say. A frame without one prints its source line alone, as CPython's
  // does for a frame whose positions it lacks.
  std::int32_t marker = site && !site->noAnchor ? 1 : 0;

  llvm::Value *file = globalCStringPtr(builder, fileName, "py.tb.file");
  llvm::Value *name = globalCStringPtr(builder, functionName, "py.tb.func");
  builder.CreateCall(
      tracebackPushCStringRange(module),
      {file, name, i32Constant(builder, line), i32Constant(builder, column),
       i32Constant(builder, endLine), i32Constant(builder, endColumn),
       i32Constant(builder, marker)});
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
  // Marked so a later pass can find the pads that CATCH, once the runtime is
  // linked and the raise primitives have bodies to read.
  landingPad->setMetadata("ly.catch",
                          llvm::MDNode::get(context, {}));
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

// ⭐ A C FUNCTION DOES NOT UNWIND, AND SAYING SO IS WHAT SHRINKS THE LSDA.
// Without it LLVM has to assume `puts` might throw, so anything calling it
// might throw, so every call to that is an invoke with a landing pad and a
// call-site entry. Half the landing pads in a program were this:
// `__gcc_except_tab` 1468 bytes -> 732.
//
// The libc declarations LLVM already recognises get this from
// `InferFunctionAttrs`; the ones it does not are the ones whose prototype here
// differs from the header's (`void puts(ptr)`, `i32 getentropy(ptr, i64)`),
// which is most of the OS cluster.
//
// ⛔ IT IS NOT A SPEEDUP, AND HERE IS WHERE THE TIME ACTUALLY IS. The
// personality reads the call-site table linearly, so a raise costs the entries
// BEFORE it: 0/10/40/80 statements in front of a `try` measure 1598/1698/2052/
// 2442 ns per raise -- 10.6 ns per preceding call site, dead straight. The
// calls this removes are the ones that cannot raise, and those were never the
// ones in front of a `try` in a loop: the 40 `str(i) + 'a'` in that measurement
// call `LyUnicode_FromBytes`, which really can raise, so the prefix is
// unchanged and so is the time (2045 ns -> 2061). Shortening the table is not
// the lever; not scanning it twice per frame would be.
//
// ⛔ NOT the `Ly*` declarations, which is the whole point of them, and not the
// ctypes symbols, which may be anything a program dlopens -- including C++ that
// really does unwind. `_Unwind_RaiseException` is the loudest counterexample of
// all: marking it would delete the raise's own unwind edge.
void markCLibraryDeclarationsNonUnwinding(
    llvm::Module &module, llvm::ArrayRef<std::string> foreignSymbols) {
  llvm::StringSet<> foreign;
  for (const std::string &symbol : foreignSymbols)
    foreign.insert(symbol);
  for (llvm::Function &function : module) {
    if (!function.isDeclaration() || function.isIntrinsic())
      continue;
    llvm::StringRef name = function.getName();
    if (name.starts_with("Ly") || name.starts_with("__ly") ||
        name.starts_with("_Unwind") || name.starts_with("__gxx_personality") ||
        foreign.contains(name))
      continue;
    function.setDoesNotThrow();
  }
}

namespace {

// The block a catch pad hands control to once it has taken the exception. The
// shape is the one `buildPythonCatchDispatchBlock` built, and nothing has run
// between: a plain branch, or the selector test whose handler arm branches.
llvm::BasicBlock *dispatchBlockOf(llvm::LandingPadInst *pad) {
  auto *branch = llvm::dyn_cast<llvm::BranchInst>(pad->getParent()->getTerminator());
  if (!branch)
    return nullptr;
  if (branch->isUnconditional())
    return branch->getSuccessor(0);
  auto *handling =
      llvm::dyn_cast<llvm::BranchInst>(branch->getSuccessor(1)->getTerminator());
  return handling && handling->isUnconditional() ? handling->getSuccessor(0)
                                                 : nullptr;
}

// A raise primitive that does nothing but hand the triple to
// `LyEH_ThrowException`. Every one in the manifests is written that way; this
// READS the body rather than trusting the convention, because a fat one would
// have work that a caller branching past the throw would skip.
bool isThinRaiseWrapper(llvm::Function &function) {
  if (function.isDeclaration() || function.size() != 1)
    return false;
  llvm::BasicBlock &entry = function.getEntryBlock();
  auto instruction = entry.begin();
  auto *call = llvm::dyn_cast<llvm::CallInst>(&*instruction);
  if (!call || !call->getCalledFunction() ||
      call->getCalledFunction()->getName() != "LyEH_ThrowException")
    return false;
  if (call->arg_size() != function.arg_size())
    return false;
  for (unsigned index = 0; index < call->arg_size(); ++index)
    if (call->getArgOperand(index) != function.getArg(index))
      return false;
  auto *ret = llvm::dyn_cast<llvm::ReturnInst>(&*std::next(instruction));
  return ret && !ret->getReturnValue();
}

} // namespace

// ⭐ A RAISE CAUGHT IN ITS OWN ACTIVATION IS A BRANCH. `try: raise ValueError`
// puts the raise and the handler in one frame, and the whole unwinder was being
// asked to find a landing pad already named in the same function -- a search
// phase and a cleanup phase, each walking every frame's call-site table, to
// arrive at a block a few instructions away.
//
// The raise becomes `LyEH_RecordException`, which is the half of a raise that
// is not the unwind, and the call returns to the dispatch chain instead of
// landing there. The chain tests the arms exactly as it would have; if none
// match it calls `LyEH_RethrowCurrent`, which raises for real.
//
// ⛔ The call STAYS AN INVOKE. Recording an exception can itself unwind (the
// interrupted exception becomes a context, and building that allocates), and
// that unwind still has to reach this frame's pad. Only the NORMAL edge moves.
//
// ⛔ And only when the raise's unwind edge is the try's catch pad. A `with` or
// a `finally` in between makes the edge a cleanup instead, and that cleanup has
// to run; the edge being the catch pad is the proof that nothing else does.
bool branchLocalRaisesToTheirHandler(llvm::Module &module) {
  llvm::Function *record = module.getFunction("LyEH_RecordException");
  if (!record)
    return false;
  bool changed = false;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function) {
      auto *invoke = llvm::dyn_cast<llvm::InvokeInst>(block.getTerminator());
      if (!invoke)
        continue;
      llvm::Function *callee = invoke->getCalledFunction();
      if (!callee || callee->getFunctionType() != record->getFunctionType())
        continue;
      if (callee != module.getFunction("LyEH_ThrowException") &&
          !(isPythonRuntimeRaiseCall(callee) && isThinRaiseWrapper(*callee)))
        continue;
      auto *pad = invoke->getUnwindDest()->getLandingPadInst();
      if (!pad || !pad->getMetadata("ly.catch"))
        continue;
      llvm::BasicBlock *dispatch = dispatchBlockOf(pad);
      // A dispatch chain reached from two places would need its incoming
      // values merged, and nothing here builds one.
      if (!dispatch || !dispatch->phis().empty() ||
          dispatch == invoke->getNormalDest())
        continue;
      invoke->getNormalDest()->removePredecessor(&block);
      invoke->setNormalDest(dispatch);
      invoke->setCalledFunction(record);
      changed = true;
    }
  return changed;
}

// ⭐ A FRAME POINTER IS WHAT MAKES A FRAME CHEAP TO LEAVE. Without one the
// prologue is sp-relative, and Darwin's compact unwind has no encoding for
// that -- 84% of a program's functions came out as `UNWIND_ARM64_MODE_DWARF`,
// which is a CFI program to interpret rather than a bitmask to read. With one
// they are `MODE_FRAME`: nine bits saying which register pairs were spilled,
// and x29 pointing at the next frame.
//
// The cost is one register and two instructions per function, measured over
// fourteen benchmarks at 1.020x mean -- and the ones long enough to be signal
// rather than startup noise at 1.001x, 1.014x and 1.019x. A cross-frame raise
// goes 2540 ns -> 1811 for it. Programs that never raise pay one to two percent.
//
// ⛔ AND THAT IS THE WHOLE OF IT. A walk of this compiler's own was built on top
// of this and is NOT HERE. It works: it reads `__unwind_info` itself (6.8 ns per
// frame against the 123 ns `_dyld_find_unwind_sections` costs, which is 60% of a
// raise), steps frames from the compact encoding, and finds the same landing pad
// the personality does -- but the step that enters the pad is hand-written
// assembly, because setting x19-x28, d8-d15, sp and the program counter at once
// is not something LLVM has a value for. That is one such routine per OS and
// architecture, none of which LLVM would check, and every one of them able to
// resume a program in a handler holding another function's registers.
//
// What settles it is not the asm but the SHAPE: a walk that sometimes runs and
// otherwise stands down leaves two unwinders to keep correct, and the one that
// runs rarely is the one that rots. Everything goes through
// `_Unwind_RaiseException`. Frame pointers are worth having on their own -- they
// make the unwinder that IS here read a bitmask instead of interpreting a CFI
// program, and a profiler can walk the stack too.
void forceFramePointers(llvm::Module &module) {
  for (llvm::Function &function : module) {
    if (function.isDeclaration())
      continue;
    function.addFnAttr("frame-pointer", "all");
  }
}

void collectCtypesForeignSymbols(mlir::ModuleOp module,
                                 llvm::SmallVectorImpl<std::string> &symbols) {
  module.walk([&](mlir::func::FuncOp function) {
    if (auto symbol = function->getAttrOfType<mlir::StringAttr>(
            py::native::kNativeSymbolAttr))
      symbols.push_back(symbol.getValue().str());
  });
}

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
    site.noAnchor = source->noAnchor;
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
