#include "Common/RuntimeSupport.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace py {
namespace {

// The AArch64 backend rewrites memory intrinsics reached from streaming code
// into these SME ABI entry points, and it does so during codegen — after the
// last point where the module could name them. Darwin ships no implementation
// (neither libSystem nor compiler-rt define them), so the definitions have to
// travel with the module.
constexpr llvm::StringLiteral kMemcpyName{"__arm_sc_memcpy"};
constexpr llvm::StringLiteral kMemmoveName{"__arm_sc_memmove"};
constexpr llvm::StringLiteral kMemsetName{"__arm_sc_memset"};
constexpr llvm::StringLiteral kMemchrName{"__arm_sc_memchr"};

bool hasStreamingFunction(llvm::Module &module) {
  for (llvm::Function &function : module) {
    if (function.hasFnAttribute("aarch64_pstate_sm_body") ||
        function.hasFnAttribute("aarch64_pstate_sm_enabled") ||
        function.hasFnAttribute("aarch64_pstate_sm_compatible") ||
        function.hasFnAttribute("aarch64_new_za") ||
        function.hasFnAttribute("aarch64_inout_za"))
      return true;
  }
  return false;
}

llvm::Function *declareRoutine(llvm::Module &module, llvm::StringRef name,
                               llvm::FunctionType *type) {
  if (module.getFunction(name))
    return nullptr;
  auto *function = llvm::Function::Create(
      type, llvm::GlobalValue::InternalLinkage, name, &module);
  // Why sm_compatible rather than plain: these run with PSTATE.SM inherited
  // from a streaming caller, where the non-streaming instruction set the
  // backend would otherwise be free to pick is illegal.
  function->addFnAttr("aarch64_pstate_sm_compatible");
  function->addFnAttr(llvm::Attribute::NoUnwind);
  return function;
}

// Every loop below stores and loads through volatile accesses. Why: the loop
// idiom recognizer would otherwise fold each one back into the very memory
// intrinsic the backend expands into a call here, leaving the routine calling
// itself.
void defineMemset(llvm::Module &module, llvm::LLVMContext &context) {
  auto *ptrType = llvm::PointerType::getUnqual(context);
  auto *i64 = llvm::Type::getInt64Ty(context);
  auto *type = llvm::FunctionType::get(
      ptrType, {ptrType, llvm::Type::getInt32Ty(context), i64}, false);
  llvm::Function *function = declareRoutine(module, kMemsetName, type);
  if (!function)
    return;

  auto *entry = llvm::BasicBlock::Create(context, "entry", function);
  auto *header = llvm::BasicBlock::Create(context, "loop.header", function);
  auto *body = llvm::BasicBlock::Create(context, "loop.body", function);
  auto *exit = llvm::BasicBlock::Create(context, "exit", function);

  llvm::Argument *dest = function->getArg(0);
  llvm::Argument *value = function->getArg(1);
  llvm::Argument *count = function->getArg(2);

  llvm::IRBuilder<> builder(entry);
  llvm::Value *byte =
      builder.CreateTrunc(value, llvm::Type::getInt8Ty(context));
  builder.CreateBr(header);

  builder.SetInsertPoint(header);
  llvm::PHINode *index = builder.CreatePHI(i64, 2);
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);
  builder.CreateCondBr(builder.CreateICmpULT(index, count), body, exit);

  builder.SetInsertPoint(body);
  builder.CreateStore(
      byte,
      builder.CreateGEP(llvm::Type::getInt8Ty(context), dest, index),
      /*isVolatile=*/true);
  llvm::Value *next = builder.CreateAdd(index, llvm::ConstantInt::get(i64, 1));
  index->addIncoming(next, body);
  builder.CreateBr(header);

  builder.SetInsertPoint(exit);
  builder.CreateRet(dest);
}

// A single forward byte loop covers memcpy; memmove additionally needs the
// backward direction when the ranges overlap the wrong way.
void defineCopy(llvm::Module &module, llvm::LLVMContext &context,
                llvm::StringRef name, bool handleOverlap) {
  auto *ptrType = llvm::PointerType::getUnqual(context);
  auto *i8 = llvm::Type::getInt8Ty(context);
  auto *i64 = llvm::Type::getInt64Ty(context);
  auto *type = llvm::FunctionType::get(ptrType, {ptrType, ptrType, i64}, false);
  llvm::Function *function = declareRoutine(module, name, type);
  if (!function)
    return;

  auto *entry = llvm::BasicBlock::Create(context, "entry", function);
  auto *forwardHeader =
      llvm::BasicBlock::Create(context, "forward.header", function);
  auto *forwardBody =
      llvm::BasicBlock::Create(context, "forward.body", function);
  auto *exit = llvm::BasicBlock::Create(context, "exit", function);

  llvm::Argument *dest = function->getArg(0);
  llvm::Argument *source = function->getArg(1);
  llvm::Argument *count = function->getArg(2);

  llvm::IRBuilder<> builder(entry);
  llvm::BasicBlock *backwardHeader = nullptr;
  llvm::BasicBlock *backwardBody = nullptr;
  if (handleOverlap) {
    backwardHeader =
        llvm::BasicBlock::Create(context, "backward.header", function);
    backwardBody = llvm::BasicBlock::Create(context, "backward.body", function);
    // Copying backward is only required when dest lies inside [source, +count);
    // comparing the raw addresses is what decides that.
    llvm::Value *destAddress = builder.CreatePtrToInt(dest, i64);
    llvm::Value *sourceAddress = builder.CreatePtrToInt(source, i64);
    llvm::Value *overlaps = builder.CreateAnd(
        builder.CreateICmpUGT(destAddress, sourceAddress),
        builder.CreateICmpULT(destAddress,
                              builder.CreateAdd(sourceAddress, count)));
    builder.CreateCondBr(overlaps, backwardHeader, forwardHeader);
  } else {
    builder.CreateBr(forwardHeader);
  }

  builder.SetInsertPoint(forwardHeader);
  llvm::PHINode *index = builder.CreatePHI(i64, 2);
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);
  builder.CreateCondBr(builder.CreateICmpULT(index, count), forwardBody, exit);

  builder.SetInsertPoint(forwardBody);
  llvm::Value *loaded = builder.CreateLoad(
      i8, builder.CreateGEP(i8, source, index), /*isVolatile=*/true);
  builder.CreateStore(loaded, builder.CreateGEP(i8, dest, index),
                      /*isVolatile=*/true);
  llvm::Value *next = builder.CreateAdd(index, llvm::ConstantInt::get(i64, 1));
  index->addIncoming(next, forwardBody);
  builder.CreateBr(forwardHeader);

  if (handleOverlap) {
    builder.SetInsertPoint(backwardHeader);
    llvm::PHINode *remaining = builder.CreatePHI(i64, 2);
    remaining->addIncoming(count, entry);
    builder.CreateCondBr(
        builder.CreateICmpUGT(remaining, llvm::ConstantInt::get(i64, 0)),
        backwardBody, exit);

    builder.SetInsertPoint(backwardBody);
    llvm::Value *last =
        builder.CreateSub(remaining, llvm::ConstantInt::get(i64, 1));
    llvm::Value *tail = builder.CreateLoad(i8, builder.CreateGEP(i8, source, last),
                                           /*isVolatile=*/true);
    builder.CreateStore(tail, builder.CreateGEP(i8, dest, last),
                        /*isVolatile=*/true);
    remaining->addIncoming(last, backwardBody);
    builder.CreateBr(backwardHeader);
  }

  builder.SetInsertPoint(exit);
  builder.CreateRet(dest);
}

void defineMemchr(llvm::Module &module, llvm::LLVMContext &context) {
  auto *ptrType = llvm::PointerType::getUnqual(context);
  auto *i8 = llvm::Type::getInt8Ty(context);
  auto *i64 = llvm::Type::getInt64Ty(context);
  auto *type = llvm::FunctionType::get(
      ptrType, {ptrType, llvm::Type::getInt32Ty(context), i64}, false);
  llvm::Function *function = declareRoutine(module, kMemchrName, type);
  if (!function)
    return;

  auto *entry = llvm::BasicBlock::Create(context, "entry", function);
  auto *header = llvm::BasicBlock::Create(context, "loop.header", function);
  auto *body = llvm::BasicBlock::Create(context, "loop.body", function);
  auto *found = llvm::BasicBlock::Create(context, "found", function);
  auto *notFound = llvm::BasicBlock::Create(context, "not.found", function);

  llvm::Argument *source = function->getArg(0);
  llvm::Argument *value = function->getArg(1);
  llvm::Argument *count = function->getArg(2);

  llvm::IRBuilder<> builder(entry);
  llvm::Value *needle = builder.CreateTrunc(value, i8);
  builder.CreateBr(header);

  builder.SetInsertPoint(header);
  llvm::PHINode *index = builder.CreatePHI(i64, 2);
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);
  builder.CreateCondBr(builder.CreateICmpULT(index, count), body, notFound);

  builder.SetInsertPoint(body);
  llvm::Value *element = builder.CreateGEP(i8, source, index);
  llvm::Value *current = builder.CreateLoad(i8, element, /*isVolatile=*/true);
  llvm::Value *next = builder.CreateAdd(index, llvm::ConstantInt::get(i64, 1));
  index->addIncoming(next, body);
  builder.CreateCondBr(builder.CreateICmpEQ(current, needle), found, header);

  builder.SetInsertPoint(found);
  builder.CreateRet(builder.CreateGEP(i8, source, index));

  builder.SetInsertPoint(notFound);
  builder.CreateRet(llvm::ConstantPointerNull::get(ptrType));
}

} // namespace

void installArmStreamingCompatibleMemoryRoutines(llvm::Module &module) {
  // Why the streaming attributes rather than the triple: only the ArmSME
  // pipeline ever attaches them, and the module's triple is not set until
  // codegen configuration, which runs after this.
  if (!hasStreamingFunction(module))
    return;

  llvm::LLVMContext &context = module.getContext();
  defineMemset(module, context);
  defineCopy(module, context, kMemcpyName, /*handleOverlap=*/false);
  defineCopy(module, context, kMemmoveName, /*handleOverlap=*/true);
  defineMemchr(module, context);

  // Nothing references these until the backend rewrites a memory intrinsic, so
  // every earlier pass sees them as dead.
  llvm::SmallVector<llvm::GlobalValue *, 4> routines;
  for (llvm::StringRef name :
       {kMemcpyName, kMemmoveName, kMemsetName, kMemchrName}) {
    if (llvm::Function *function = module.getFunction(name))
      routines.push_back(function);
  }
  llvm::appendToCompilerUsed(module, routines);
}

} // namespace py
