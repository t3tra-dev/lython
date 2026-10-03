#include "Common/LibcPrototypes.h"

#include "Native.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <optional>
#include <string>

namespace py::runtime_library {
namespace {

// The C types a prototype here is written in. `Int64` is `time_t` and `off_t`:
// 64-bit on every target, because the 32-bit glibc spellings below are the
// time64/LFS ones.
enum class CType { Void, Int, UInt, Long, SizeT, SSizeT, Int64, Ptr, Double };

struct LibcPrototype {
  llvm::StringLiteral name;
  CType result;
  CType params[4];
  unsigned arity;
  bool isVarArg = false;
};

using C = CType;

// ⛔ Every C library function the runtime calls, and nothing it does not. A
// function missing here is refused on a target that needs exact prototypes,
// which is how this table finds out it is incomplete.
constexpr LibcPrototype kLibc[] = {
    {"_errno", C::Ptr, {}, 0},
    {"_exit", C::Void, {C::Int}, 1},
    {"__errno_location", C::Ptr, {}, 0},
    {"__error", C::Ptr, {}, 0},
    {"abort", C::Void, {}, 0},
    {"access", C::Int, {C::Ptr, C::Int}, 2},
    {"aligned_alloc", C::Ptr, {C::SizeT, C::SizeT}, 2},
    {"calloc", C::Ptr, {C::SizeT, C::SizeT}, 2},
    {"chdir", C::Int, {C::Ptr}, 1},
    {"clock_gettime", C::Int, {C::Int, C::Ptr}, 2},
    {"closedir", C::Int, {C::Ptr}, 1},
    {"dlsym", C::Ptr, {C::Ptr, C::Ptr}, 2},
    {"exit", C::Void, {C::Int}, 1},
    {"fclose", C::Int, {C::Ptr}, 1},
    {"fflush", C::Int, {C::Ptr}, 1},
    {"fgetc", C::Int, {C::Ptr}, 1},
    {"fgets", C::Ptr, {C::Ptr, C::Int, C::Ptr}, 3},
    {"fileno", C::Int, {C::Ptr}, 1},
    {"fmod", C::Double, {C::Double, C::Double}, 2},
    {"fopen", C::Ptr, {C::Ptr, C::Ptr}, 2},
    {"fread", C::SizeT, {C::Ptr, C::SizeT, C::SizeT, C::Ptr}, 4},
    {"free", C::Void, {C::Ptr}, 1},
    {"fseek", C::Int, {C::Ptr, C::Long, C::Int}, 3},
    {"ftell", C::Long, {C::Ptr}, 1},
    {"ftruncate", C::Int, {C::Int, C::Int64}, 2},
    {"fwrite", C::SizeT, {C::Ptr, C::SizeT, C::SizeT, C::Ptr}, 4},
    {"getchar", C::Int, {}, 0},
    {"getcwd", C::Ptr, {C::Ptr, C::SizeT}, 2},
    {"getegid", C::UInt, {}, 0},
    {"getentropy", C::Int, {C::Ptr, C::SizeT}, 2},
    {"getenv", C::Ptr, {C::Ptr}, 1},
    {"geteuid", C::UInt, {}, 0},
    {"getgid", C::UInt, {}, 0},
    {"getpid", C::Int, {}, 0},
    {"getppid", C::Int, {}, 0},
    {"getuid", C::UInt, {}, 0},
    {"gmtime_r", C::Ptr, {C::Ptr, C::Ptr}, 2},
    {"localtime_r", C::Ptr, {C::Ptr, C::Ptr}, 2},
    {"lstat", C::Int, {C::Ptr, C::Ptr}, 2},
    {"malloc", C::Ptr, {C::SizeT}, 1},
    {"memcpy", C::Ptr, {C::Ptr, C::Ptr, C::SizeT}, 3},
    {"memmove", C::Ptr, {C::Ptr, C::Ptr, C::SizeT}, 3},
    {"memset", C::Ptr, {C::Ptr, C::Int, C::SizeT}, 3},
    {"mkdir", C::Int, {C::Ptr, C::UInt}, 2},
    {"mktime", C::Int64, {C::Ptr}, 1},
    {"nanosleep", C::Int, {C::Ptr, C::Ptr}, 2},
    {"opendir", C::Ptr, {C::Ptr}, 1},
    // Darwin's spellings only: pthread_t is a pointer there, and the parallel
    // dispatch that calls these is Darwin-only (TensorParallel.cpp).
    {"pthread_create", C::Int, {C::Ptr, C::Ptr, C::Ptr, C::Ptr}, 4},
    {"pthread_get_stackaddr_np", C::Ptr, {C::Ptr}, 1},
    {"pthread_get_stacksize_np", C::SizeT, {C::Ptr}, 1},
    {"pthread_join", C::Int, {C::Ptr, C::Ptr}, 2},
    {"pthread_self", C::Ptr, {}, 0},
    {"puts", C::Int, {C::Ptr}, 1},
    {"raise", C::Int, {C::Int}, 1},
    {"readdir", C::Ptr, {C::Ptr}, 1},
    {"realloc", C::Ptr, {C::Ptr, C::SizeT}, 2},
    {"rename", C::Int, {C::Ptr, C::Ptr}, 2},
    {"rmdir", C::Int, {C::Ptr}, 1},
    {"setenv", C::Int, {C::Ptr, C::Ptr, C::Int}, 3},
    {"sigaction", C::Int, {C::Int, C::Ptr, C::Ptr}, 3},
    {"sigaltstack", C::Int, {C::Ptr, C::Ptr}, 2},
    {"signal", C::Ptr, {C::Int, C::Ptr}, 2},
    {"snprintf", C::Int, {C::Ptr, C::SizeT, C::Ptr}, 3, /*isVarArg=*/true},
    {"stat", C::Int, {C::Ptr, C::Ptr}, 2},
    {"strerror", C::Ptr, {C::Int}, 1},
    {"strftime", C::SizeT, {C::Ptr, C::SizeT, C::Ptr, C::Ptr}, 4},
    {"strlen", C::SizeT, {C::Ptr}, 1},
    {"strtod", C::Double, {C::Ptr, C::Ptr}, 2},
    {"strtol", C::Long, {C::Ptr, C::Ptr, C::Int}, 3},
    {"sysconf", C::Long, {C::Int}, 1},
    {"ungetc", C::Int, {C::Int, C::Ptr}, 2},
    {"unlink", C::Int, {C::Ptr}, 1},
    {"unsetenv", C::Int, {C::Ptr}, 1},
    {"write", C::SSizeT, {C::Int, C::Ptr, C::SizeT}, 3},
};

const LibcPrototype *findPrototype(llvm::StringRef name) {
  for (const LibcPrototype &prototype : kLibc)
    if (prototype.name == name)
      return &prototype;
  return nullptr;
}

// The symbol the target's own headers would have called. The struct layouts
// these spellings imply are HostTargetLayout's (SupportBuilder.h), keyed by the
// same triple.
//
// - Darwin x86_64: the 64-bit-inode `struct stat` / `struct dirent` are the
//   `$INODE64` symbols; the unsuffixed ones are the deprecated 32-bit ones.
// - 32-bit glibc: what `-D_FILE_OFFSET_BITS=64 -D_TIME_BITS=64` resolves to,
//   read off `nm -u` of a probe built that way on armv7 (glibc 2.36). With
//   them `time_t` and `off_t` are 64-bit, as on every other target here.
llvm::StringRef targetSymbolFor(llvm::StringRef name,
                                const llvm::Triple &triple) {
  if (triple.isOSDarwin() && triple.getArch() == llvm::Triple::x86_64)
    return llvm::StringSwitch<llvm::StringRef>(name)
        .Case("stat", "stat$INODE64")
        .Case("lstat", "lstat$INODE64")
        .Case("readdir", "readdir$INODE64")
        .Default(name);
  if (triple.isOSLinux() && triple.isGNUEnvironment() && triple.isArch32Bit())
    return llvm::StringSwitch<llvm::StringRef>(name)
        .Case("stat", "__stat64_time64")
        .Case("lstat", "__lstat64_time64")
        .Case("readdir", "readdir64")
        .Case("clock_gettime", "__clock_gettime64")
        .Case("nanosleep", "__nanosleep64")
        .Case("gmtime_r", "__gmtime64_r")
        .Case("localtime_r", "__localtime64_r")
        .Case("mktime", "__mktime64")
        .Case("ftruncate", "ftruncate64")
        .Default(name);
  return name;
}

struct TargetWidths {
  unsigned pointerBits;
  unsigned longBits;
};

bool isSigned(CType type) {
  return type == C::Int || type == C::Long || type == C::SSizeT ||
         type == C::Int64;
}

llvm::Type *lower(CType type, const TargetWidths &widths,
                  llvm::LLVMContext &context) {
  switch (type) {
  case C::Void:
    return llvm::Type::getVoidTy(context);
  case C::Int:
  case C::UInt:
    return llvm::Type::getInt32Ty(context);
  case C::Long:
    return llvm::Type::getIntNTy(context, widths.longBits);
  case C::SizeT:
  case C::SSizeT:
    return llvm::Type::getIntNTy(context, widths.pointerBits);
  case C::Int64:
    return llvm::Type::getInt64Ty(context);
  case C::Ptr:
    return llvm::PointerType::getUnqual(context);
  case C::Double:
    return llvm::Type::getDoubleTy(context);
  }
  llvm_unreachable("unknown C type");
}

llvm::FunctionType *lower(const LibcPrototype &prototype,
                          const TargetWidths &widths,
                          llvm::LLVMContext &context) {
  llvm::SmallVector<llvm::Type *, 4> params;
  for (unsigned index = 0; index < prototype.arity; ++index)
    params.push_back(lower(prototype.params[index], widths, context));
  return llvm::FunctionType::get(lower(prototype.result, widths, context),
                                 params, prototype.isVarArg);
}

// The runtime's value as the C parameter wants it. A narrowing is checked:
// a size beyond SIZE_MAX saturates -- no object that large exists, so the
// allocator reports failure through its usual NULL -- and any other value
// that does not fit traps rather than reaching C as a different number.
llvm::Value *toParameter(llvm::IRBuilder<> &builder, llvm::Value *value,
                         llvm::Type *want, CType type) {
  llvm::Type *have = value->getType();
  if (have == want)
    return value;
  if (have->isPointerTy() && want->isIntegerTy())
    return builder.CreatePtrToInt(value, want);
  if (have->isIntegerTy() && want->isPointerTy())
    return builder.CreateIntToPtr(value, want);
  if (!have->isIntegerTy() || !want->isIntegerTy())
    return builder.CreateBitCast(value, want);
  unsigned haveBits = have->getIntegerBitWidth();
  unsigned wantBits = want->getIntegerBitWidth();
  if (haveBits < wantBits)
    return isSigned(type) ? builder.CreateSExt(value, want)
                          : builder.CreateZExt(value, want);
  llvm::Value *narrow = builder.CreateTrunc(value, want);
  llvm::Value *back = isSigned(type) ? builder.CreateSExt(narrow, have)
                                     : builder.CreateZExt(narrow, have);
  llvm::Value *fits = builder.CreateICmpEQ(back, value);
  if (type == C::SizeT)
    return builder.CreateSelect(fits, narrow,
                                llvm::ConstantInt::getAllOnesValue(want));
  llvm::Instruction *trap = llvm::SplitBlockAndInsertIfThen(
      builder.CreateNot(fits), builder.GetInsertPoint(), /*Unreachable=*/true);
  llvm::IRBuilder<> trapBuilder(trap);
  trapBuilder.CreateIntrinsic(llvm::Intrinsic::trap, {});
  return narrow;
}

llvm::Value *toResult(llvm::IRBuilder<> &builder, llvm::Value *value,
                      llvm::Type *want, CType type) {
  llvm::Type *have = value->getType();
  if (have == want)
    return value;
  if (have->isPointerTy() && want->isIntegerTy())
    return builder.CreatePtrToInt(value, want);
  if (have->isIntegerTy() && want->isPointerTy())
    return builder.CreateIntToPtr(value, want);
  if (have->isIntegerTy() && want->isIntegerTy())
    return builder.CreateIntCast(value, want, isSigned(type));
  return builder.CreateBitCast(value, want);
}

mlir::LogicalResult rebuildCall(llvm::CallBase *call, llvm::Function *callee,
                                const LibcPrototype &prototype,
                                llvm::raw_ostream &diag) {
  llvm::FunctionType *want = callee->getFunctionType();
  if (call->arg_size() < want->getNumParams()) {
    diag << "error: a call to '" << prototype.name << "' passes "
         << call->arg_size() << " arguments; C declares "
         << want->getNumParams() << "\n";
    return mlir::failure();
  }
  llvm::IRBuilder<> builder(call);
  llvm::SmallVector<llvm::Value *, 6> args;
  for (unsigned index = 0; index < call->arg_size(); ++index) {
    llvm::Value *arg = call->getArgOperand(index);
    if (index < want->getNumParams()) {
      builder.SetInsertPoint(call);
      arg = toParameter(builder, arg, want->getParamType(index),
                        prototype.params[index]);
    }
    args.push_back(arg);
  }
  builder.SetInsertPoint(call);
  llvm::CallBase *replacement;
  if (auto *invoke = llvm::dyn_cast<llvm::InvokeInst>(call))
    replacement = builder.CreateInvoke(want, callee, invoke->getNormalDest(),
                                       invoke->getUnwindDest(), args);
  else
    replacement = builder.CreateCall(want, callee, args);
  replacement->setCallingConv(call->getCallingConv());
  replacement->setDebugLoc(call->getDebugLoc());
  replacement->setAttributes(llvm::AttributeList::get(
      call->getContext(), call->getAttributes().getFnAttrs(), {}, {}));

  if (!call->getType()->isVoidTy() && !call->use_empty()) {
    if (replacement->getType()->isVoidTy()) {
      diag << "error: a call to '" << prototype.name
           << "' uses a result C does not return\n";
      return mlir::failure();
    }
    llvm::BasicBlock::iterator after;
    if (auto *invoke = llvm::dyn_cast<llvm::InvokeInst>(replacement)) {
      llvm::BasicBlock *continuation =
          llvm::SplitEdge(invoke->getParent(), invoke->getNormalDest());
      after = continuation->getFirstInsertionPt();
    } else {
      after = std::next(replacement->getIterator());
    }
    builder.SetInsertPoint(after);
    call->replaceAllUsesWith(
        toResult(builder, replacement, call->getType(), prototype.result));
  }
  call->eraseFromParent();
  return mlir::success();
}

bool isRuntimeOwnSymbol(llvm::StringRef name) {
  return name.starts_with("Ly") || name.starts_with("__ly") ||
         name.starts_with("_Unwind") || name.starts_with("__gxx") ||
         name.starts_with("__cxa");
}

} // namespace

mlir::LogicalResult declareLibcWithTargetPrototypes(llvm::Module &module,
                                                    llvm::raw_ostream &diag) {
  llvm::Triple triple(module.getTargetTriple());
  TargetWidths widths;
  widths.pointerBits = module.getDataLayout().getPointerSizeInBits();
  widths.longBits = static_cast<unsigned>(
      py::native::expectedCLongWidth(triple.str(), widths.pointerBits));
  // Where every C prototype here would spell itself the runtime's way anyway,
  // a function the table lacks is linked by name and cannot be misread.
  bool needsExactPrototypes =
      widths.pointerBits != 64 || widths.longBits != 64 ||
      py::native::callsAreCheckedBySignature(triple.str());

  llvm::SmallVector<std::pair<llvm::Function *, const LibcPrototype *>, 32>
      declared;
  llvm::SmallVector<std::string, 4> unknown;
  for (llvm::Function &function : module) {
    if (!function.isDeclaration() || function.isIntrinsic() ||
        function.use_empty() || isRuntimeOwnSymbol(function.getName()))
      continue;
    if (const LibcPrototype *prototype = findPrototype(function.getName())) {
      declared.push_back({&function, prototype});
      continue;
    }
    if (needsExactPrototypes &&
        !function.hasFnAttribute(kCtypesForeignSymbolAttr))
      unknown.push_back(function.getName().str());
  }
  if (!unknown.empty()) {
    diag << "error: no C prototype is recorded for";
    for (const std::string &name : unknown)
      diag << " '" << name << "'";
    diag << " (lowering/Common/LibcPrototypes.cpp), and " << triple.str()
         << " cannot call a C function by guessing its widths\n";
    return mlir::failure();
  }

  llvm::LLVMContext &context = module.getContext();
  for (auto [function, prototype] : declared) {
    llvm::FunctionType *want = lower(*prototype, widths, context);
    llvm::StringRef symbol = targetSymbolFor(prototype->name, triple);
    if (function->getFunctionType() == want && function->getName() == symbol)
      continue;
    std::string symbolName = symbol.str();
    function->setName(function->getName() + ".ly.libc");
    llvm::Function *target = module.getFunction(symbolName);
    if (!target)
      target = llvm::Function::Create(want, llvm::GlobalValue::ExternalLinkage,
                                      symbolName, module);
    if (target->getFunctionType() != want) {
      diag << "error: '" << symbolName
           << "' is declared twice with different types\n";
      return mlir::failure();
    }
    target->setAttributes(llvm::AttributeList::get(
        context, function->getAttributes().getFnAttrs(), {}, {}));
    for (llvm::User *user : llvm::make_early_inc_range(function->users()))
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(user))
        if (call->getCalledOperand() == function &&
            mlir::failed(rebuildCall(call, target, *prototype, diag)))
          return mlir::failure();
    function->replaceAllUsesWith(target);
    function->eraseFromParent();
  }
  return mlir::success();
}

} // namespace py::runtime_library
