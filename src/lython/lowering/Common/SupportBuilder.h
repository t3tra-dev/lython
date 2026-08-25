#pragma once

// Shared facade for the native runtime support builders (RuntimeSupportBuilder
// and TracebackSupportBuilder compose the same module).

#include "Common/ExceptionABI.h"
#include "Common/MemRef1D.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <string>

namespace py::runtime_library {

// The rank-1 descriptor vocabulary is shared with the passes (Common/MemRef1D.h
// says why); these are the names this layer spells it with.
using py::lowering::MemRef1DParts;
using py::lowering::buildMemRef1D;
using py::lowering::explodeMemRef1D;
using py::lowering::memRef1DDescriptorType;

// The host boundary is otherwise target-independent (portable libc calls at
// the fopen/fwrite altitude). Three things about the OS cluster cannot be:
// the errno accessor's symbol name, the byte offsets inside `struct stat` and
// `struct dirent` (which differ per libc AND per arch), and Darwin's
// $INODE64 symbol variants on x86_64. They are gathered here so the OS
// cluster reads offsets by name and every per-target fact has one home.
//
// Field widths are byte counts; a negative width means the field is signed
// (dev_t is int32_t on Darwin) and is sign-extended into the i64 result.
struct HostTargetLayout {
  bool posix = true;

  // Which column of kOSErrorErrnoMap the target's errno numbers come from.
  bool bsdErrnoValues = false;

  llvm::StringRef errnoAccessor = "__errno_location";
  llvm::StringRef statSymbol = "stat";
  llvm::StringRef lstatSymbol = "lstat";
  llvm::StringRef readdirSymbol = "readdir";

  // Offset/width pairs for the os.stat_result fields, in the order
  // posix._stat_fields returns them.
  int statMode[2] = {24, 4};
  int statIno[2] = {8, 8};
  int statDev[2] = {0, 8};
  int statNlink[2] = {16, 8};
  int statUid[2] = {28, 4};
  int statGid[2] = {32, 4};
  int statSize[2] = {48, 8};
  int statAtime[2] = {72, 8};
  int statMtime[2] = {88, 8};
  int statCtime[2] = {104, 8};

  // `struct dirent`'s NUL-terminated d_name.
  int direntNameOffset = 19;

  // CLOCK_MONOTONIC's value (CLOCK_REALTIME is 0 everywhere).
  int clockMonotonic = 1;
};

// Derives the layout above from the target triple. Unknown OS/arch pairs keep
// the Linux/x86_64 defaults rather than guessing: a wrong guess would read
// the wrong words silently, so the OS cluster refuses to run at all on a
// non-POSIX target (`posix == false` makes every entry point fail loudly).
inline HostTargetLayout hostTargetLayout(const llvm::Triple &triple) {
  HostTargetLayout layout;
  if (triple.isOSWindows()) {
    layout.posix = false;
    layout.errnoAccessor = "_errno";
    return layout;
  }
  if (triple.isOSDarwin()) {
    layout.errnoAccessor = "__error";
    layout.bsdErrnoValues = true;
    // Darwin's 64-bit-inode struct stat/dirent are the default ABI on arm64
    // but a $INODE64-suffixed variant on x86_64, where the unsuffixed symbol
    // is still the deprecated 32-bit-inode one.
    if (triple.getArch() == llvm::Triple::x86_64) {
      layout.statSymbol = "stat$INODE64";
      layout.lstatSymbol = "lstat$INODE64";
      layout.readdirSymbol = "readdir$INODE64";
    }
    layout.statMode[0] = 4;
    layout.statMode[1] = 2;
    layout.statIno[0] = 8;
    layout.statDev[0] = 0;
    layout.statDev[1] = -4;
    layout.statNlink[0] = 6;
    layout.statNlink[1] = 2;
    layout.statUid[0] = 16;
    layout.statUid[1] = 4;
    layout.statGid[0] = 20;
    layout.statGid[1] = 4;
    layout.statSize[0] = 96;
    layout.statSize[1] = 8;
    layout.statAtime[0] = 32;
    layout.statMtime[0] = 48;
    layout.statCtime[0] = 64;
    layout.direntNameOffset = 21;
    layout.clockMonotonic = 6; // CLOCK_MONOTONIC
    return layout;
  }
  // Linux: the kernel struct stat is arch-specific. aarch64 packs st_mode and
  // st_nlink as two 32-bit words where x86_64 has a 64-bit st_nlink first.
  if (triple.getArch() == llvm::Triple::aarch64 ||
      triple.getArch() == llvm::Triple::aarch64_be) {
    layout.statMode[0] = 16;
    layout.statNlink[0] = 20;
    layout.statNlink[1] = 4;
    layout.statUid[0] = 24;
    layout.statGid[0] = 28;
  }
  return layout;
}

// Small builder facade that mirrors the shape of the hand-written support IR
// while keeping the C++ concise. Every runtime routine is composed from the
// high-level dialects (func/arith/math/cf/ub) so the existing native-runtime
// lowering pipeline finalizes it to LLVM.
struct SupportBuilder {
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::ModuleOp module;
  HostTargetLayout host;

  SupportBuilder(mlir::ModuleOp module, const llvm::Triple &triple)
      : builder(module.getContext()), loc(builder.getUnknownLoc()),
        module(module), host(hostTargetLayout(triple)) {}

  mlir::Type f64() { return builder.getF64Type(); }
  mlir::Type i64() { return builder.getIntegerType(64); }
  mlir::Type i32() { return builder.getIntegerType(32); }
  mlir::Type i8() { return builder.getIntegerType(8); }
  mlir::Type i1() { return builder.getIntegerType(1); }
  mlir::Type ptr() { return mlir::LLVM::LLVMPointerType::get(builder.getContext()); }

  mlir::Value intToPtr(mlir::Value address) {
    return mlir::LLVM::IntToPtrOp::create(builder, loc, ptr(), address);
  }
  // The narrowing direction: a reference handed to the refcount helpers, which
  // accept a tagged immediate as readily as a pointer. Its inverse is not a
  // pair with it -- see the note on `exceptionPartsType`.
  mlir::Value ptrToInt(mlir::Value pointer) {
    return mlir::LLVM::PtrToIntOp::create(builder, loc, i64(), pointer);
  }
  // base[index] as an i64-element GEP (index in i64 units unless noted).
  mlir::Value gepI64(mlir::Value base, mlir::Value index) {
    return mlir::LLVM::GEPOp::create(builder, loc, ptr(), i64(), base,
                                     mlir::ValueRange{index});
  }
  mlir::Value loadI64(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, i64(), pointer,
                                      /*alignment=*/8);
  }

  mlir::Value fconst(double value) {
    return mlir::arith::ConstantOp::create(builder, loc,
                                           builder.getF64FloatAttr(value));
  }
  mlir::Value iconst(std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, i64(), value);
  }
  mlir::Value iconst32(std::int32_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, i32(), value);
  }
  mlir::Value iconst8(std::int8_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, i8(), value);
  }
  mlir::Value nullPtr() {
    return mlir::LLVM::ZeroOp::create(builder, loc, ptr()).getResult();
  }
  mlir::Value addrOf(llvm::StringRef name) {
    return mlir::LLVM::AddressOfOp::create(builder, loc, ptr(), name)
        .getResult();
  }
  mlir::Value ptrEq(mlir::Value a, mlir::Value b) {
    return mlir::LLVM::ICmpOp::create(builder, loc,
                                      mlir::LLVM::ICmpPredicate::eq, a, b);
  }
  mlir::Value ptrNe(mlir::Value a, mlir::Value b) {
    return mlir::LLVM::ICmpOp::create(builder, loc,
                                      mlir::LLVM::ICmpPredicate::ne, a, b);
  }
  mlir::Value gepI8(mlir::Value base, mlir::Value index) {
    return mlir::LLVM::GEPOp::create(builder, loc, ptr(), i8(), base,
                                     mlir::ValueRange{index});
  }
  mlir::Value loadI8(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, i8(), pointer,
                                      /*alignment=*/1);
  }
  void storeI8(mlir::Value value, mlir::Value pointer) {
    mlir::LLVM::StoreOp::create(builder, loc, value, pointer, /*alignment=*/1);
  }
  mlir::Value loadPtrVal(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, ptr(), pointer,
                                      /*alignment=*/8);
  }
  mlir::Value loadI32(mlir::Value pointer) {
    return mlir::LLVM::LoadOp::create(builder, loc, i32(), pointer,
                                      /*alignment=*/4);
  }
  // Fields of a TracebackFrame slot: frame[0, index].
  mlir::Value frameField(mlir::Type frameType, mlir::Value frame,
                         std::int32_t index) {
    return mlir::LLVM::GEPOp::create(
        builder, loc, ptr(), frameType, frame,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                           mlir::LLVM::GEPArg(index)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
  }
  mlir::ValueRange call(llvm::StringRef callee, mlir::TypeRange results,
                        mlir::ValueRange args) {
    return mlir::func::CallOp::create(builder, loc, callee, results, args)
        .getResults();
  }
  // Internal constant C string global (idempotent); NUL is appended.
  void stringGlobal(llvm::StringRef name, llvm::StringRef text) {
    if (module.lookupSymbol(name))
      return;
    std::string data = text.str();
    data.push_back('\0');
    auto type = mlir::LLVM::LLVMArrayType::get(i8(), data.size());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    mlir::LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                 mlir::LLVM::Linkage::Internal, name,
                                 builder.getStringAttr(data),
                                 /*alignment=*/1);
  }
  mlir::Value cmpf(mlir::arith::CmpFPredicate pred, mlir::Value a,
                   mlir::Value b) {
    return mlir::arith::CmpFOp::create(builder, loc, pred, a, b);
  }
  mlir::Value cmpi(mlir::arith::CmpIPredicate pred, mlir::Value a,
                   mlir::Value b) {
    return mlir::arith::CmpIOp::create(builder, loc, pred, a, b);
  }
  mlir::Value orBit(mlir::Value a, mlir::Value b) {
    return mlir::arith::OrIOp::create(builder, loc, a, b);
  }

  // Declares an external libc symbol (resolved at final link).
  void declareExternal(llvm::StringRef name, mlir::FunctionType type) {
    if (module.lookupSymbol(name))
      return;
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    auto fn = mlir::func::FuncOp::create(builder, loc, name, type);
    fn.setPrivate();
  }

  mlir::func::FuncOp beginFunction(llvm::StringRef name, mlir::FunctionType type,
                                   bool isPrivate = false) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    auto fn = mlir::func::FuncOp::create(builder, loc, name, type);
    fn.setVisibility(isPrivate ? mlir::SymbolTable::Visibility::Private
                               : mlir::SymbolTable::Visibility::Public);
    return fn;
  }

  // Terminates a trap block: abort() then a poison return (abort is noreturn,
  // so the return is dead but keeps the block well-formed without dropping to
  // the llvm dialect's `unreachable`).
  void emitTrap(mlir::Type resultType) {
    mlir::func::CallOp::create(builder, loc, "abort", mlir::TypeRange{},
                               mlir::ValueRange{});
    mlir::Value poison =
        mlir::ub::PoisonOp::create(builder, loc, resultType, nullptr);
    mlir::func::ReturnOp::create(builder, loc, poison);
  }
};

// ---------------------------------------------------------------------------
// The in-flight exception, and the node an interrupted one is parked in.
//
// An exception travels as THREE rank-1 memrefs -- the exception object's
// header, its message's header, and the message's bytes. Two shapes of that
// triple exist and the difference is only how a rank-1 shape is spelled:
// `exceptionBorrowPartsType` is what a function returns (sizes and strides are
// `[1 x i64]`, the descriptor layout MLIR's memref lowering produces), and
// `exceptionPartsType` is what lives IN MEMORY (plain `i64`, since a stored
// descriptor here is only ever rank 1).
//
// ⛔ Both spell the pointer members `!llvm.ptr`, and nothing may read them as
// words. `extract_aligned_pointer_as_index` is where the memory model says
// provenance is lost, and what comes back through an integer is outside the
// model by its own statement -- so a descriptor reassembled from words is a
// descriptor the model cannot talk about. Stored and loaded as a pointer it
// stays the pointer the raise built, and the handler's view is the same
// reference rather than a look-alike.
//
// Both files that build the EH runtime need these, so they live here instead
// of being declared privately in each and drifting.
// ---------------------------------------------------------------------------

// {{allocated, aligned, offset, size, stride} x3}, 120 bytes.
mlir::Type exceptionPartsType(SupportBuilder &b);

// &parts[section][field], typed: fields 0 and 1 load as `!llvm.ptr`.
mlir::Value partsField(SupportBuilder &b, mlir::Value parts,
                       std::int32_t section, std::int32_t field);

// The whole triple as one value, so a move is a load and a store rather than a
// byte copy that erases which member was a pointer.
mlir::Value loadExceptionParts(SupportBuilder &b, mlir::Value parts);
void storeExceptionParts(SupportBuilder &b, mlir::Value parts,
                         mlir::Value value);
void clearExceptionParts(SupportBuilder &b, mlir::Value parts);

// The heap node a raise moves an interrupted exception into (also the parking
// spot for a suspended generator's token). Members, in order:
//
//   0 refcount    -- a node may be both __cause__ and __context__
//   1 payload     -- ExceptionParts; an owning reference to object + message
//   2 frames      -- TracebackFrame*, a malloc'd snapshot owning its names
//   3 frameCount
//   4 cause       -- ExceptionChainNode*, null = none
//   5 context     -- ExceptionChainNode*, null = none
//   6 suppress    -- __suppress_context__
//
// 8 + 120 + 8 + 8 + 8 + 8 + 8 = 168 bytes, which is what the malloc asks for.
enum ChainNodeMember : std::int32_t {
  kNodeRefcount = 0,
  kNodePayload = 1,
  kNodeFrames = 2,
  kNodeFrameCount = 3,
  kNodeCause = 4,
  kNodeContext = 5,
  kNodeSuppress = 6,
};

mlir::Type exceptionChainNodeType(SupportBuilder &b);
mlir::Value nodeMember(SupportBuilder &b, mlir::Value node,
                       std::int32_t member);

// Free a chain node whose caller believes it is the last owner.
//
// The node has a refcount and `release_chain_node` honours it: decrement, and
// destroy only at zero. Four callers do not go through it -- they move the
// node's members out and free the shell -- and each was correct only at one
// owner, a belief recorded in a comment and enforced nowhere. A second owner
// would not have been a diagnosed failure; it would have been that owner
// reading freed memory.
//
// The object refcount already has this check (`release_storage_raw_to_zero
// observed non-positive refcount`). The node's had no counterpart, which is
// the whole difference between a refcount and a number in a struct.
void freeSoleChainNode(SupportBuilder &b, mlir::Value node,
                       llvm::StringRef site);

// ---------------------------------------------------------------------------
// The stash cell: one slot holding a parked chain node, or null.
//
// Three functions own it -- `LyEH_StashCurrentException` fills it,
// `LyEH_UnstashException` empties it, `LyEH_AdoptStashedAsContext` moves what
// is in it -- and nothing else reads or writes one. That is why it can hold a
// POINTER even where its bytes belong to a `memref<?xi64>`: a memref cannot
// have a pointer element type, but nothing here goes through the memref. The
// callers hand over the cell's address, not its contents.
//
// The two kinds of caller are a suspended generator (its storage, and a stack
// slot for the resumer's context) and an except* frame (its residual, and one
// per clause body that raised).
inline mlir::Type stashCellType(SupportBuilder &b) { return b.ptr(); }

// ---------------------------------------------------------------------------
// The except* frame (PEP 654): the residual exception between clauses plus
// everything the clause bodies raised, each parked as a chain node.
//
//   0 residual   -- stash cell; the caught exception, null before begin
//   1 present    -- the node outlives its payload, staying on as the
//                   traceback/chain donor after the last slice matched
//   2 collected  -- how many of the array below are in use
//   3 clauses[32]-- stash cells, one per clause body that raised
//
// A `parent` word used to sit between 2 and 3, linking frames into a stack.
// The frame is an SSA value now, so nesting is lexical and the word had no
// remaining reader.
enum StarFrameMember : std::int32_t {
  kStarResidual = 0,
  kStarPresent = 1,
  kStarCollected = 2,
  kStarClauses = 3,
};

inline constexpr std::int64_t kStarClauseLimit = 32;

// `sizeof(type)` as an i64, by the GEP-on-null trick MLIR gives no op for.
// Written rather than a constant because both callers malloc exactly one of
// these and a constant that drifts from the struct is a heap overrun.
mlir::Value typeSizeBytes(SupportBuilder &b, mlir::Type type);

mlir::Type starFrameType(SupportBuilder &b);
mlir::Value starFrameMember(SupportBuilder &b, mlir::Value frame,
                            std::int32_t member);
// &frame->clauses[index]; the index is dynamic.
mlir::Value starClauseCell(SupportBuilder &b, mlir::Value frame,
                           mlir::Value index);
mlir::Value nodePartsField(SupportBuilder &b, mlir::Value node,
                           std::int32_t section, std::int32_t field);

// ---------------------------------------------------------------------------
// The exception triple in the memref world, and the bridge to the descriptor.
//
// This is the SAME triple as above, in the memref types the callers use --
// `exceptionTripleTypes` (Common/ExceptionABI.h) is what a func-level runtime
// entry point declares. The two helpers below take a value of one of those
// types apart and put it back together.
//
// ⛔ The bridge is `unrealized_conversion_cast`, NOT
// `extract_aligned_pointer_as_index`. Both get at the pointer inside a memref
// and only one of them keeps it a pointer: the index op hands back an integer,
// which the memory model documents as where provenance is lost. The cast is
// erased against the func-to-LLVM conversion's own inverse cast by
// `reconcile-unrealized-casts`, so nothing survives it at all.
// ---------------------------------------------------------------------------

// `MemRef1DParts` and the assembly itself live in Common/MemRef1D.h, shared
// with the box path in the passes. These are the `SupportBuilder &` spellings.
MemRef1DParts explodeMemRef1D(SupportBuilder &b, mlir::Value memref);
mlir::Value buildMemRef1D(SupportBuilder &b, mlir::Type memrefType,
                          const MemRef1DParts &parts);

// Three memref values into an ExceptionParts region (the process slot, or a
// chain node's payload -- `nodeMember(node, kNodePayload)` is one).
// Emits the raise of a Lython `_Unwind_Exception` at the current insertion
// point (RuntimeSupportBuilder.cpp). Inline at every raise on purpose.
void emitRaiseCarrier(SupportBuilder &b);

void storeExceptionTriple(SupportBuilder &b, mlir::Value parts,
                          mlir::ValueRange triple);

// Emits the host-boundary cluster (raw write / exit status / argv / FILE*
// and buffer wrappers, plus the OS/time cluster); implemented in
// HostSupportBuilder.cpp.
void buildHostSupport(SupportBuilder &b);

// Emits the OS/time cluster (errno, the OSError message formatter, the
// filesystem and environment calls, the clock and calendar calls) that
// modules/posix.mlir and modules/time.mlir call into; implemented in
// OsSupportBuilder.cpp.
void buildOsSupport(SupportBuilder &b);

// Emits the traceback cluster (frame stack, push/pop accounting, uncaught
// exception printer) into the module; implemented in
// TracebackSupportBuilder.cpp.
void buildTracebackSupport(SupportBuilder &b);

} // namespace py::runtime_library
