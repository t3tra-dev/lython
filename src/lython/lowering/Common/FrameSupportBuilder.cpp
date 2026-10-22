#include "Common/SupportBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// The live call stack: what `sys._getframe` would walk, for the readers that
// need the CALLER's place -- warnings' stacklevel, logging's findCaller.
//
// ⭐ ONLY THE FUNCTIONS THAT CAN REACH A READER WRITE IT. The frames are
// written at call sites by the frame pass (Passes/Runtime/Cleanup/Frames.cpp),
// and only in a function from which a call can reach `_frame`'s readers; a
// program that never asks pays nothing. A function reads the depth once at its
// entry and, before each call that can reach a reader, writes its own frame --
// and the frames of the bodies the emitter inlined into it -- at that depth
// and sets the depth just past them.
// ⛔ No pop. A return or an unwind leaves the depth where the callee put it,
// and the next write in the caller sets it back; the readers are themselves
// such calls, so a reader never sees a frame its caller has left.
// ⛔ Not the traceback stack (TracebackSupportBuilder.cpp): that one holds the
// frames an exception has passed through, is filled while it unwinds, and is
// cleared when it is handled; this one holds where calls are now.
//
// A record is four words: the file, the function name and the module name
// (each a NUL-terminated constant the pass emitted) and the line.

namespace py::runtime_library {
namespace {

constexpr std::int64_t kFrameWords = 4;

void declareFrameSupport(SupportBuilder &b) {
  b.declareExternal("realloc", b.builder.getFunctionType({b.ptr(), b.i64()},
                                                         {b.ptr()}));
  b.declareExternal("strlen",
                    b.builder.getFunctionType({b.ptr()}, {b.i64()}));
  b.declareExternal("abort", b.builder.getFunctionType({}, {}));
  mlir::OpBuilder::InsertionGuard guard(b.builder);
  b.builder.setInsertionPointToEnd(b.module.getBody());
  for (llvm::StringRef name : {"g_frame_size", "g_frame_capacity"})
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::Internal, name,
                                 b.builder.getIntegerAttr(b.i64(), 0),
                                 /*alignment=*/8);
  auto stack = mlir::LLVM::GlobalOp::create(
      b.builder, b.loc, b.ptr(), /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, "g_frame_stack", mlir::Attribute(),
      /*alignment=*/8);
  mlir::Block *init = b.builder.createBlock(&stack.getInitializerRegion());
  b.builder.setInsertionPointToEnd(init);
  mlir::LLVM::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.nullPtr()});
}

void storeI64(SupportBuilder &b, mlir::Value value, mlir::Value pointer) {
  mlir::LLVM::StoreOp::create(b.builder, b.loc, value, pointer,
                              /*alignment=*/8);
}

// i64 LyFrame_Size(): the depth a function's frames start at.
void buildFrameSize(SupportBuilder &b) {
  auto fn = b.beginFunction("LyFrame_Size",
                            b.builder.getFunctionType({}, {b.i64()}));
  b.builder.setInsertionPointToEnd(fn.addEntryBlock());
  mlir::func::ReturnOp::create(
      b.builder, b.loc, mlir::ValueRange{b.loadI64(b.addrOf("g_frame_size"))});
}

// void LyFrame_Set(i64 index, ptr file, ptr name, ptr module, i64 line):
// writes the record at `index` and makes the depth `index + 1`.
// ⛔ Grown, not capped: a recursion deeper than a fixed array would have
// answered a frame that is not the caller's. Exhaustion aborts, as every
// runtime allocation does (the allocator must never raise).
void buildFrameSet(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyFrame_Set", b.builder.getFunctionType(
                         {b.i64(), b.ptr(), b.ptr(), b.ptr(), b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *grow = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  mlir::Block *write = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value index = entry->getArgument(0);
  mlir::Value capacity = b.loadI64(b.addrOf("g_frame_capacity"));
  mlir::Value fits =
      b.cmpi(mlir::arith::CmpIPredicate::slt, index, capacity);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, fits, write,
                                 mlir::ValueRange{}, grow, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(grow);
  mlir::Value doubled =
      mlir::arith::MulIOp::create(b.builder, b.loc, capacity, b.iconst(2));
  mlir::Value needed =
      mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(64));
  mlir::Value grown =
      mlir::arith::MaxSIOp::create(b.builder, b.loc, doubled, needed);
  mlir::Value bytes = mlir::arith::MulIOp::create(
      b.builder, b.loc, grown, b.iconst(kFrameWords * 8));
  mlir::Value stack =
      b.call("realloc", b.ptr(),
             mlir::ValueRange{b.loadPtrVal(b.addrOf("g_frame_stack")), bytes})
          .front();
  mlir::Value failed = b.ptrEq(stack, b.nullPtr());
  mlir::LLVM::StoreOp::create(b.builder, b.loc, stack,
                              b.addrOf("g_frame_stack"));
  storeI64(b, grown, b.addrOf("g_frame_capacity"));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                 mlir::ValueRange{}, write,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trap);
  b.call("abort", mlir::TypeRange{}, mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(write);
  mlir::Value base = b.gepI64(
      b.loadPtrVal(b.addrOf("g_frame_stack")),
      mlir::arith::MulIOp::create(b.builder, b.loc, index,
                                  b.iconst(kFrameWords)));
  for (unsigned word = 0; word < 3; ++word)
    storeI64(b, b.ptrToInt(entry->getArgument(word + 1)),
             b.gepI64(base, b.iconst(word)));
  storeI64(b, entry->getArgument(4), b.gepI64(base, b.iconst(3)));
  storeI64(b,
           mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1)),
           b.addrOf("g_frame_size"));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// Word `word` of the frame `level` calls above the reader's caller (level 0 is
// the function that called the reader), or `absent` when the stack is not
// that deep.
mlir::Value frameWord(SupportBuilder &b, mlir::func::FuncOp fn,
                      mlir::Value level, mlir::Value word,
                      mlir::Value absent) {
  mlir::Region &body = fn.getBody();
  mlir::Block *read = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body, body.end(), {b.i64()},
                                            {b.loc});
  b.builder.setInsertionPointToEnd(&body.front());
  mlir::Value size = b.loadI64(b.addrOf("g_frame_size"));
  mlir::Value index = mlir::arith::SubIOp::create(
      b.builder, b.loc,
      mlir::arith::SubIOp::create(b.builder, b.loc, size, b.iconst(1)), level);
  mlir::Value inRange = mlir::arith::AndIOp::create(
      b.builder, b.loc,
      b.cmpi(mlir::arith::CmpIPredicate::sge, level, b.iconst(0)),
      b.cmpi(mlir::arith::CmpIPredicate::sge, index, b.iconst(0)));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, inRange, read,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{absent});
  b.builder.setInsertionPointToEnd(read);
  mlir::Value base = b.gepI64(
      b.loadPtrVal(b.addrOf("g_frame_stack")),
      mlir::arith::MulIOp::create(b.builder, b.loc, index,
                                  b.iconst(kFrameWords)));
  mlir::cf::BranchOp::create(
      b.builder, b.loc, done,
      mlir::ValueRange{b.loadI64(b.gepI64(base, word))});
  b.builder.setInsertionPointToEnd(done);
  return done->getArgument(0);
}

// i64 LyFrame_Line(i64 level): the line, or -1 past the bottom of the stack.
void buildFrameLine(SupportBuilder &b) {
  auto fn = b.beginFunction("LyFrame_Line",
                            b.builder.getFunctionType({b.i64()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value line = frameWord(b, fn, entry->getArgument(0), b.iconst(3),
                               b.iconst(-1));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{line});
}

// The text at `field` (0 file, 1 function, 2 module) of the frame `level`
// calls up, as a pointer; null past the bottom of the stack.
mlir::Value frameText(SupportBuilder &b, mlir::func::FuncOp fn,
                      mlir::Value level, mlir::Value field) {
  b.builder.setInsertionPointToEnd(&fn.getBody().front());
  return b.intToPtr(frameWord(b, fn, level, field, b.iconst(0)));
}

// i64 LyFrame_TextLen(i64 level, i64 field): 0 past the bottom of the stack.
void buildFrameTextLen(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyFrame_TextLen",
      b.builder.getFunctionType({b.i64(), b.i64()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Value text =
      frameText(b, fn, entry->getArgument(0), entry->getArgument(1));
  mlir::Region &body = fn.getBody();
  mlir::Block *measure = b.builder.createBlock(&body);
  mlir::Block *none = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(text.getParentBlock());
  mlir::cf::CondBranchOp::create(b.builder, b.loc,
                                 b.ptrEq(text, b.nullPtr()), none,
                                 mlir::ValueRange{}, measure,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(measure);
  mlir::func::ReturnOp::create(
      b.builder, b.loc,
      mlir::ValueRange{b.call("strlen", b.i64(), mlir::ValueRange{text})
                           .front()});
  b.builder.setInsertionPointToEnd(none);
  mlir::func::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst(0)});
}

// void LyFrame_TextCopy(i64 level, i64 field, memref<?xi8> dest, i64 len) at
// the lowered ABI (level, field, alloc, aligned, offset, size, stride, len).
void buildFrameTextCopy(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyFrame_TextCopy",
      b.builder.getFunctionType({b.i64(), b.i64(), b.ptr(), b.ptr(), b.i64(),
                                 b.i64(), b.i64(), b.i64()},
                                {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Value text =
      frameText(b, fn, entry->getArgument(0), entry->getArgument(1));
  mlir::Region &body = fn.getBody();
  mlir::Block *copy = b.builder.createBlock(&body);
  mlir::Block *none = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(text.getParentBlock());
  mlir::cf::CondBranchOp::create(b.builder, b.loc,
                                 b.ptrEq(text, b.nullPtr()), none,
                                 mlir::ValueRange{}, copy, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(copy);
  mlir::Value dest = b.gepI8(entry->getArgument(3), entry->getArgument(4));
  mlir::LLVM::MemcpyOp::create(b.builder, b.loc, dest, text,
                               entry->getArgument(7), /*isVolatile=*/false);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(none);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

} // namespace

void buildFrameSupport(SupportBuilder &b) {
  declareFrameSupport(b);
  buildFrameSize(b);
  buildFrameSet(b);
  buildFrameLine(b);
  buildFrameTextLen(b);
  buildFrameTextCopy(b);
}

} // namespace py::runtime_library
