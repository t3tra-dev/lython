#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py {
namespace {

struct PythonSourceRange {
  std::string filename;
  std::int32_t line = 0;
  std::int32_t column = 0;
  std::int32_t endLine = 0;
  std::int32_t endColumn = 0;
};

std::optional<mlir::FileLineColLoc> findPythonSourceLoc(mlir::Location loc) {
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    if (fileLoc.getFilename().getValue().ends_with(".py"))
      return fileLoc;
    return std::nullopt;
  }
  if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc))
    return findPythonSourceLoc(nameLoc.getChildLoc());
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location child : fused.getLocations())
      if (auto found = findPythonSourceLoc(child))
        return found;
  }
  return std::nullopt;
}

std::optional<std::int32_t> i32Attr(mlir::DictionaryAttr dict,
                                    llvm::StringRef name) {
  auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(dict.get(name));
  if (!attr)
    return std::nullopt;
  return static_cast<std::int32_t>(attr.getInt());
}

std::optional<PythonSourceRange>
sourceRangeFromDict(mlir::DictionaryAttr dict) {
  auto startLine = i32Attr(dict, "lython.source.start_line");
  auto startCol = i32Attr(dict, "lython.source.start_col");
  auto endLine = i32Attr(dict, "lython.source.end_line");
  auto endCol = i32Attr(dict, "lython.source.end_col");
  if (!startLine || !startCol || !endLine || !endCol)
    return std::nullopt;
  PythonSourceRange range;
  range.line = *startLine;
  range.column = *startCol;
  range.endLine = *endLine;
  range.endColumn = *endCol;
  return range;
}

std::optional<PythonSourceRange> findSourceRangeMetadata(mlir::Location loc) {
  if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc))
    return findSourceRangeMetadata(nameLoc.getChildLoc());
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    if (auto dict =
            mlir::dyn_cast_or_null<mlir::DictionaryAttr>(fused.getMetadata()))
      if (auto range = sourceRangeFromDict(dict))
        return range;
    for (mlir::Location child : fused.getLocations())
      if (auto range = findSourceRangeMetadata(child))
        return range;
  }
  return std::nullopt;
}

std::optional<PythonSourceRange> pythonSourceRange(mlir::Location loc) {
  std::optional<mlir::FileLineColLoc> fileLoc = findPythonSourceLoc(loc);
  if (!fileLoc)
    return std::nullopt;

  PythonSourceRange range;
  range.filename = fileLoc->getFilename().getValue().str();
  range.line = static_cast<std::int32_t>(fileLoc->getLine());
  range.column = static_cast<std::int32_t>(fileLoc->getColumn());
  range.endLine = range.line;
  range.endColumn = 0;
  if (auto metadata = findSourceRangeMetadata(loc)) {
    metadata->filename = range.filename;
    return metadata;
  }
  return range;
}

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
  return unit && static_cast<unsigned>(unit->getSourceLanguage()) ==
                     llvm::dwarf::DW_LANG_Python;
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

bool convertCallToPythonInvoke(llvm::CallInst &call,
                               llvm::ArrayRef<PythonCallSiteRange> callSites) {
  if (call.isInlineAsm() || call.getNumOperandBundles() != 0)
    return false;
  llvm::Function *callee = call.getCalledFunction();
  if (!isPythonDebugFunction(callee))
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
    auto callee = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(*calleeName);
    if (!callee || !isPythonFunction(callee))
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
  for (llvm::Function &function : module) {
    if (!isPythonDebugFunction(&function))
      continue;
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &instruction : block) {
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
          calls.push_back(call);
      }
    }
  }

  bool changed = false;
  for (llvm::CallInst *call : calls)
    changed |= convertCallToPythonInvoke(*call, callSites);
  return changed;
}

} // namespace py
