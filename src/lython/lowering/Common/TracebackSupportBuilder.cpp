#include "Common/SupportBuilder.h"
#include "ExceptionTaxonomy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace py::runtime_library {
namespace {

// ---------------------------------------------------------------------------
// Traceback cluster: the per-process frame stack, push/pop accounting, and the
// uncaught-exception printer (CPython-style traceback with source lines and
// `~~~^^` markers). Faithful translation of the former native module.
// ---------------------------------------------------------------------------

mlir::Type tracebackFrameType(SupportBuilder &b) {
  auto frame = mlir::LLVM::LLVMStructType::getIdentified(
      b.builder.getContext(), "TracebackFrame");
  if (frame.getBody().empty())
    (void)frame.setBody({b.ptr(), b.ptr(), b.i32(), b.i32(), b.i32(), b.i32(),
                         b.i32(), b.i32()},
                        /*isPacked=*/false);
  return frame;
}

mlir::Type tracebackStackType(SupportBuilder &b) {
  return mlir::LLVM::LLVMArrayType::get(tracebackFrameType(b), 1024);
}

void declareTracebackSupport(SupportBuilder &b) {
  b.declareExternal("malloc",
                    b.builder.getFunctionType({b.i64()}, {b.ptr()}));
  b.declareExternal("fopen", b.builder.getFunctionType({b.ptr(), b.ptr()},
                                                       {b.ptr()}));
  b.declareExternal("fgets", b.builder.getFunctionType(
                                 {b.ptr(), b.i32(), b.ptr()}, {b.ptr()}));
  b.declareExternal("fclose",
                    b.builder.getFunctionType({b.ptr()}, {b.i32()}));
  // Variadic: must be an llvm.func so the call carries the vararg callee type.
  if (!b.module.lookupSymbol("snprintf")) {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::LLVMFuncOp::create(
        b.builder, b.loc, "snprintf",
        mlir::LLVM::LLVMFunctionType::get(b.i32(),
                                          {b.ptr(), b.i64(), b.ptr()},
                                          /*isVarArg=*/true));
  }

  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::Internal,
                                 "g_traceback_size",
                                 b.builder.getIntegerAttr(b.i64(), 0),
                                 /*alignment=*/8);
    // Exception-chaining state: heap chain-node addresses (0 = none) for the
    // in-flight exception's __context__ / __cause__, and its
    // __suppress_context__ flag.
    for (llvm::StringRef name :
         {"g_exc_context_node", "g_exc_cause_node", "g_exc_suppress_context"})
      mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                   /*isConstant=*/false,
                                   mlir::LLVM::Linkage::Internal, name,
                                   b.builder.getIntegerAttr(b.i64(), 0),
                                   /*alignment=*/8);
    auto stack = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, tracebackStackType(b), /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, "g_traceback_stack", mlir::Attribute(),
        /*alignment=*/8);
    mlir::Block *init = b.builder.createBlock(&stack.getInitializerRegion());
    b.builder.setInsertionPointToEnd(init);
    mlir::Value zero =
        mlir::LLVM::ZeroOp::create(b.builder, b.loc, tracebackStackType(b));
    mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{zero});
  }

  b.stringGlobal(".tb_read_mode", "r");
  b.stringGlobal(".tb_indent", "    ");
  b.stringGlobal(".tb_newline", "\n");
  b.stringGlobal(".tb_header", "Traceback (most recent call last):\n");
  b.stringGlobal(".tb_sep_context",
                 "\nDuring handling of the above exception, another exception "
                 "occurred:\n\n");
  b.stringGlobal(".tb_sep_cause",
                 "\nThe above exception was the direct cause of the following "
                 "exception:\n\n");
  b.stringGlobal(".tb_fmt_frame", "  File \"%s\", line %d, in %s\n");
  b.stringGlobal(".tb_fmt_class", "%s\n");
  b.stringGlobal(".tb_fmt_invalid", "%s: <invalid>\n");
  b.stringGlobal(".tb_fmt_unknown", "%s: <unknown>\n");
  b.stringGlobal(".tb_fmt_message", "%s: %s\n");
  for (const py::exceptions::BuiltinExceptionInfo &info :
       py::exceptions::kBuiltinExceptions)
    b.stringGlobal((llvm::Twine(".tb_class.") + info.name).str(), info.name);
}

// ptr copy_cstr(ptr cstr): malloc'd NUL-terminated copy ("" for null input).
void buildCopyCStr(SupportBuilder &b) {
  auto fn = b.beginFunction("copy_cstr",
                            b.builder.getFunctionType({b.ptr()}, {b.ptr()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *emptyCopy = b.builder.createBlock(&body);
  mlir::Block *emptyStore = b.builder.createBlock(&body);
  mlir::Block *realCopy = b.builder.createBlock(&body);
  mlir::Block *copyBytes = b.builder.createBlock(&body);
  mlir::Block *terminate = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value source = entry->getArgument(0);
  mlir::Value isNull = b.ptrEq(source, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, emptyCopy,
                                 mlir::ValueRange{}, realCopy,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(emptyCopy);
  mlir::Value one = b.iconst(1);
  mlir::Value emptyBlock =
      b.call("malloc", b.ptr(), mlir::ValueRange{one}).front();
  mlir::Value emptyFailed = b.ptrEq(emptyBlock, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, emptyFailed, trap,
                                 mlir::ValueRange{}, emptyStore,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(emptyStore);
  b.storeI8(b.iconst8(0), emptyBlock);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{emptyBlock});

  b.builder.setInsertionPointToEnd(realCopy);
  mlir::Value length =
      b.call("strlen", b.i64(), mlir::ValueRange{source}).front();
  mlir::Value withNul =
      mlir::arith::AddIOp::create(b.builder, b.loc, length, b.iconst(1));
  mlir::Value block =
      b.call("malloc", b.ptr(), mlir::ValueRange{withNul}).front();
  mlir::Value failed = b.ptrEq(block, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                 mlir::ValueRange{}, copyBytes,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(copyBytes);
  mlir::Value hasBytes = b.cmpi(mlir::arith::CmpIPredicate::ne, length,
                                b.iconst(0));
  auto copyIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                        hasBytes, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&copyIf.getThenRegion().front());
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, block, source, length,
                                 /*isVolatile=*/false);
  }
  mlir::cf::BranchOp::create(b.builder, b.loc, terminate, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(terminate);
  b.storeI8(b.iconst8(0), b.gepI8(block, length));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{block});

  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.ptr());
}

// ptr copy_i8_memref(ptr data, i64 offset, i64 len, i64 stride): malloc'd
// NUL-terminated copy of a strided byte view; invalid descriptors abort.
void buildCopyI8MemRef(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "copy_i8_memref",
      b.builder.getFunctionType({b.ptr(), b.i64(), b.i64(), b.i64()},
                                {b.ptr()}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkNull = b.builder.createBlock(&body);
  mlir::Block *allocate = b.builder.createBlock(&body);
  mlir::Block *loopHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *loopBody = b.builder.createBlock(&body);
  mlir::Block *terminate = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  mlir::Value data = entry->getArgument(0);
  mlir::Value offset = entry->getArgument(1);
  mlir::Value len = entry->getArgument(2);
  mlir::Value stride = entry->getArgument(3);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value offsetNeg =
      b.cmpi(mlir::arith::CmpIPredicate::slt, offset, zero);
  mlir::Value lenNeg = b.cmpi(mlir::arith::CmpIPredicate::slt, len, zero);
  mlir::Value strideBad =
      b.cmpi(mlir::arith::CmpIPredicate::slt, stride, one);
  mlir::Value invalid = b.orBit(b.orBit(offsetNeg, lenNeg), strideBad);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, invalid, trap,
                                 mlir::ValueRange{}, checkNull,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkNull);
  mlir::Value lenZero = b.cmpi(mlir::arith::CmpIPredicate::eq, len, zero);
  mlir::Value dataNull = b.ptrEq(data, b.nullPtr());
  mlir::Value lenNonZero = mlir::arith::XOrIOp::create(
      b.builder, b.loc, lenZero,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult());
  mlir::Value nullWithBytes =
      mlir::arith::AndIOp::create(b.builder, b.loc, dataNull, lenNonZero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, nullWithBytes, trap,
                                 mlir::ValueRange{}, allocate,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(allocate);
  mlir::Value withNul =
      mlir::arith::AddIOp::create(b.builder, b.loc, len, one);
  mlir::Value block =
      b.call("malloc", b.ptr(), mlir::ValueRange{withNul}).front();
  mlir::Value failed = b.ptrEq(block, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                 mlir::ValueRange{}, loopHead,
                                 mlir::ValueRange{zero});

  b.builder.setInsertionPointToEnd(loopHead);
  mlir::Value index = loopHead->getArgument(0);
  mlir::Value doneCopying =
      b.cmpi(mlir::arith::CmpIPredicate::eq, index, len);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, doneCopying, terminate,
                                 mlir::ValueRange{}, loopBody,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(loopBody);
  mlir::Value scaled =
      mlir::arith::MulIOp::create(b.builder, b.loc, index, stride);
  mlir::Value sourceIndex =
      mlir::arith::AddIOp::create(b.builder, b.loc, offset, scaled);
  mlir::Value byte = b.loadI8(b.gepI8(data, sourceIndex));
  b.storeI8(byte, b.gepI8(block, index));
  mlir::Value next = mlir::arith::AddIOp::create(b.builder, b.loc, index, one);
  mlir::cf::BranchOp::create(b.builder, b.loc, loopHead,
                             mlir::ValueRange{next});

  b.builder.setInsertionPointToEnd(terminate);
  b.storeI8(b.iconst8(0), b.gepI8(block, len));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{block});

  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.ptr());
}

// ptr frame_at(i64 index): address of g_traceback_stack[index].
void buildFrameAt(SupportBuilder &b) {
  auto fn = b.beginFunction("frame_at",
                            b.builder.getFunctionType({b.i64()}, {b.ptr()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value stack = b.addrOf("g_traceback_stack");
  mlir::Value frame = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), tracebackStackType(b), stack,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(
                                             entry->getArgument(0))},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{frame});
}

// void free_frame(ptr frame): frees the two owned name copies.
void buildFreeFrame(SupportBuilder &b) {
  auto fn = b.beginFunction("free_frame",
                            b.builder.getFunctionType({b.ptr()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Type frameType = tracebackFrameType(b);
  mlir::Value frame = entry->getArgument(0);
  for (std::int32_t field : {0, 1}) {
    mlir::Value pointer = b.loadPtrVal(b.frameField(frameType, frame, field));
    mlir::Value present = b.ptrNe(pointer, b.nullPtr());
    auto freeIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          present, /*withElseRegion=*/false);
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&freeIf.getThenRegion().front());
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{pointer});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// Shared frame-store tail for the two push entry points: writes the copied
// names, the five i32 words, bumps the stack size.
void emitFramePush(SupportBuilder &b, mlir::Value size, mlir::Value fileCopy,
                   mlir::Value functionCopy,
                   llvm::ArrayRef<mlir::Value> words) {
  mlir::Type frameType = tracebackFrameType(b);
  mlir::Value frame =
      b.call("frame_at", b.ptr(), mlir::ValueRange{size}).front();
  mlir::LLVM::StoreOp::create(b.builder, b.loc, fileCopy,
                              b.frameField(frameType, frame, 0),
                              /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, functionCopy,
                              b.frameField(frameType, frame, 1),
                              /*alignment=*/8);
  for (auto [index, word] : llvm::enumerate(words))
    mlir::LLVM::StoreOp::create(
        b.builder, b.loc, word,
        b.frameField(frameType, frame, 2 + static_cast<std::int32_t>(index)),
        /*alignment=*/4);
  mlir::Value bumped =
      mlir::arith::AddIOp::create(b.builder, b.loc, size, b.iconst(1));
  mlir::LLVM::StoreOp::create(b.builder, b.loc, bumped,
                              b.addrOf("g_traceback_size"), /*alignment=*/8);
}

mlir::Value loadTracebackSize(SupportBuilder &b) {
  return mlir::LLVM::LoadOp::create(b.builder, b.loc, b.i64(),
                                    b.addrOf("g_traceback_size"),
                                    /*alignment=*/8);
}

// LyTraceback_Push(file view: ptr/offset/size/stride via two descriptor arg
// groups, i32 line, i32 col): pushes a frame with copied names.
void buildTracebackPush(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_Push",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64(),
                                 b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64(),
                                 b.i32(), b.i32()},
                                {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *push = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value full =
      b.cmpi(mlir::arith::CmpIPredicate::uge, size, b.iconst(1024));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, full, trap,
                                 mlir::ValueRange{}, push,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(push);
  mlir::Value fileCopy =
      b.call("copy_i8_memref", b.ptr(),
             mlir::ValueRange{entry->getArgument(1), entry->getArgument(2),
                              entry->getArgument(3), entry->getArgument(4)})
          .front();
  mlir::Value functionCopy =
      b.call("copy_i8_memref", b.ptr(),
             mlir::ValueRange{entry->getArgument(6), entry->getArgument(7),
                              entry->getArgument(8), entry->getArgument(9)})
          .front();
  mlir::Value zero32 = b.iconst32(0);
  emitFramePush(b, size, fileCopy, functionCopy,
                {entry->getArgument(10), entry->getArgument(11), zero32,
                 zero32, zero32});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trap);
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyTraceback_PushCStringRange(file, function, line, col, endCol, colValid):
// C-string push carrying the caret range; marker flag set.
void buildTracebackPushCStringRange(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PushCStringRange",
      b.builder.getFunctionType(
          {b.ptr(), b.ptr(), b.i32(), b.i32(), b.i32(), b.i32()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *push = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value full =
      b.cmpi(mlir::arith::CmpIPredicate::uge, size, b.iconst(1024));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, full, trap,
                                 mlir::ValueRange{}, push,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(push);
  mlir::Value fileCopy =
      b.call("copy_cstr", b.ptr(), mlir::ValueRange{entry->getArgument(0)})
          .front();
  mlir::Value functionCopy =
      b.call("copy_cstr", b.ptr(), mlir::ValueRange{entry->getArgument(1)})
          .front();
  emitFramePush(b, size, fileCopy, functionCopy,
                {entry->getArgument(2), entry->getArgument(3),
                 entry->getArgument(4), entry->getArgument(5),
                 b.iconst32(1)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trap);
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void buildTracebackPushCString(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PushCString",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.i32(), b.i32()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  b.call("LyTraceback_PushCStringRange", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
                          entry->getArgument(2), entry->getArgument(3),
                          entry->getArgument(2), b.iconst32(0)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void buildTracebackPop(SupportBuilder &b) {
  auto fn =
      b.beginFunction("LyTraceback_Pop", b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value hasFrames =
      b.cmpi(mlir::arith::CmpIPredicate::ne, size, b.iconst(0));
  auto popIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                       hasFrames, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&popIf.getThenRegion().front());
    mlir::Value top =
        mlir::arith::SubIOp::create(b.builder, b.loc, size, b.iconst(1));
    mlir::Value frame =
        b.call("frame_at", b.ptr(), mlir::ValueRange{top}).front();
    b.call("free_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
    mlir::LLVM::StoreOp::create(b.builder, b.loc, top,
                                b.addrOf("g_traceback_size"),
                                /*alignment=*/8);
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

void buildTracebackClear(SupportBuilder &b) {
  auto fn =
      b.beginFunction("LyTraceback_Clear", b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *head = b.builder.createBlock(&body);
  mlir::Block *popOne = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(head);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value empty =
      b.cmpi(mlir::arith::CmpIPredicate::eq, size, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, empty, done,
                                 mlir::ValueRange{}, popOne,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(popOne);
  b.call("LyTraceback_Pop", mlir::TypeRange{}, mlir::ValueRange{});
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ---------------------------------------------------------------------------
// Exception chain nodes. A raise that interrupts the handling of another
// exception moves that exception (payload descriptors + traceback snapshot +
// its own chain) into a heap node referenced by the new exception's
// __context__ slot; `raise ... from` records a __cause__ node the same way.
//
// Node layout (21 i64 words, 168 bytes):
//   0      refcount (a node may be shared as both __context__ and __cause__)
//   1..15  ExceptionParts payload (3 sections x alloc/aligned/offset/size/
//          stride; owning reference to the exception object + message)
//   16     TracebackFrame array (malloc'd snapshot; owns the name strings)
//   17     frame count
//   18     __cause__ node (0 = none)
//   19     __context__ node (0 = none)
//   20     __suppress_context__ flag
// ---------------------------------------------------------------------------

constexpr std::int64_t kChainNodeBytes = 168;
constexpr std::int64_t kFrameBytes = 40;

mlir::Value nodeSlot(SupportBuilder &b, mlir::Value nodePtr,
                     std::int64_t slot) {
  return b.gepI64(nodePtr, b.iconst(slot));
}

// void release_chain_node(i64 node): drop one reference; at zero, release the
// chained nodes, the traceback snapshot, and the exception payload.
void buildReleaseChainNode(SupportBuilder &b) {
  auto fn = b.beginFunction("release_chain_node",
                            b.builder.getFunctionType({b.i64()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *alive = b.builder.createBlock(&body);
  mlir::Block *decOnly = b.builder.createBlock(&body);
  mlir::Block *destroy = b.builder.createBlock(&body);
  mlir::Block *freeHead =
      b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *freeOne = b.builder.createBlock(&body);
  mlir::Block *freeDone = b.builder.createBlock(&body);
  mlir::Block *freePayload = b.builder.createBlock(&body);
  mlir::Block *freeNode = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value node64 = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value isNull =
      b.cmpi(mlir::arith::CmpIPredicate::eq, node64, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, alive,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(alive);
  mlir::Value node = b.intToPtr(node64);
  mlir::Value refcount = b.loadI64(nodeSlot(b, node, 0));
  mlir::Value decremented =
      mlir::arith::SubIOp::create(b.builder, b.loc, refcount, b.iconst(1));
  mlir::Value stillShared =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, decremented, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, stillShared, decOnly,
                                 mlir::ValueRange{}, destroy,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(decOnly);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, decremented,
                              nodeSlot(b, node, 0), /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(destroy);
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadI64(nodeSlot(b, node, 18))});
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadI64(nodeSlot(b, node, 19))});
  mlir::cf::BranchOp::create(b.builder, b.loc, freeHead,
                             mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(freeHead);
  mlir::Value index = freeHead->getArgument(0);
  mlir::Value count = b.loadI64(nodeSlot(b, node, 17));
  mlir::Value framesDone =
      b.cmpi(mlir::arith::CmpIPredicate::sge, index, count);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, framesDone, freeDone,
                                 mlir::ValueRange{}, freeOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freeOne);
  mlir::Value framesPtr = b.intToPtr(b.loadI64(nodeSlot(b, node, 16)));
  mlir::Value frame = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), tracebackFrameType(b), framesPtr,
      mlir::ValueRange{index});
  b.call("free_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::Value next =
      mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, freeHead,
                             mlir::ValueRange{next});

  b.builder.setInsertionPointToEnd(freeDone);
  b.call("free_raw_i64_ptr", mlir::TypeRange{},
         mlir::ValueRange{b.loadI64(nodeSlot(b, node, 16))});
  mlir::Value headerWord = b.loadI64(nodeSlot(b, node, 2));
  mlir::Value becameZero = b.call("release_storage_raw_to_zero", b.i1(),
                                  mlir::ValueRange{headerWord})
                               .front();
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero, freePayload,
                                 mlir::ValueRange{}, freeNode,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freePayload);
  b.call("release_unicode_raw", mlir::TypeRange{},
         mlir::ValueRange{b.loadI64(nodeSlot(b, node, 7)),
                          b.loadI64(nodeSlot(b, node, 12))});
  b.call("free_raw_i64_ptr", mlir::TypeRange{}, mlir::ValueRange{headerWord});
  mlir::cf::BranchOp::create(b.builder, b.loc, freeNode, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freeNode);
  b.call("free_raw_i64_ptr", mlir::TypeRange{}, mlir::ValueRange{node64});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void release_current_chain(): drop the in-flight exception's chain state.
void buildReleaseCurrentChain(SupportBuilder &b) {
  auto fn = b.beginFunction("release_current_chain",
                            b.builder.getFunctionType({}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero = b.iconst(0);
  for (llvm::StringRef name : {"g_exc_cause_node", "g_exc_context_node"}) {
    mlir::Value slot = b.addrOf(name);
    b.call("release_chain_node", mlir::TypeRange{},
           mlir::ValueRange{b.loadI64(slot)});
    mlir::LLVM::StoreOp::create(b.builder, b.loc, zero, slot,
                                /*alignment=*/8);
  }
  mlir::LLVM::StoreOp::create(b.builder, b.loc, zero,
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyEH_StashCurrentAsContext(): move a pending exception (payload, traceback
// snapshot, chain) into a fresh node and make it the in-flight __context__.
// No-op when nothing is pending, so raise paths may call it unconditionally.
void buildStashCurrentAsContext(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StashCurrentAsContext",
                            b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *stash = b.builder.createBlock(&body);
  mlir::Block *snap = b.builder.createBlock(&body);
  mlir::Block *chain =
      b.builder.createBlock(&body, body.end(), {b.ptr()}, {b.loc});
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value pending = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.i1(), b.addrOf("g_current_exception"),
      /*alignment=*/4);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, pending, stash,
                                 mlir::ValueRange{}, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(stash);
  b.call("end_native_catch_if_active", mlir::TypeRange{}, {});
  mlir::Value node =
      b.call("malloc", b.ptr(), mlir::ValueRange{b.iconst(kChainNodeBytes)})
          .front();
  mlir::Value allocFailed = b.ptrEq(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, allocFailed, trap,
                                 mlir::ValueRange{}, snap, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(snap);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(1),
                              nodeSlot(b, node, 0), /*alignment=*/8);
  mlir::LLVM::MemcpyOp::create(b.builder, b.loc, nodeSlot(b, node, 1),
                               b.addrOf("g_current_parts"), b.iconst(120),
                               /*isVolatile=*/false);
  mlir::Value size = loadTracebackSize(b);
  mlir::Value haveFrames =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, size, b.iconst(0));
  auto framesIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          haveFrames, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&framesIf.getThenRegion().front());
    mlir::Value bytes = mlir::arith::MulIOp::create(b.builder, b.loc, size,
                                                    b.iconst(kFrameBytes));
    mlir::Value buffer =
        b.call("malloc", b.ptr(), mlir::ValueRange{bytes}).front();
    // Frame-name ownership moves wholesale from the global stack; a failed
    // allocation would strand it, so give up loudly instead.
    mlir::Value bufferMissing = b.ptrEq(buffer, b.nullPtr());
    mlir::cf::AssertOp::create(
        b.builder, b.loc,
        mlir::arith::XOrIOp::create(
            b.builder, b.loc, bufferMissing,
            mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1)
                .getResult()),
        "traceback snapshot allocation failed");
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, buffer,
                                 b.addrOf("g_traceback_stack"), bytes,
                                 /*isVolatile=*/false);
    mlir::Value bufferWord =
        mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), buffer);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, bufferWord,
                                nodeSlot(b, node, 16), /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, size, nodeSlot(b, node, 17),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                b.addrOf("g_traceback_size"), /*alignment=*/8);
    // Both regions keep the builder-synthesized scf.yield terminators.
    b.builder.setInsertionPointToStart(&framesIf.getElseRegion().front());
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                nodeSlot(b, node, 16), /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                nodeSlot(b, node, 17), /*alignment=*/8);
  }
  mlir::cf::BranchOp::create(b.builder, b.loc, chain, mlir::ValueRange{node});

  b.builder.setInsertionPointToEnd(chain);
  mlir::Value chainNode = chain->getArgument(0);
  mlir::Value zero = b.iconst(0);
  struct Slot {
    llvm::StringRef global;
    std::int64_t slot;
  };
  for (Slot s : {Slot{"g_exc_cause_node", 18}, Slot{"g_exc_context_node", 19},
                 Slot{"g_exc_suppress_context", 20}}) {
    mlir::Value global = b.addrOf(s.global);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.loadI64(global),
                                nodeSlot(b, chainNode, s.slot),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, zero, global,
                                /*alignment=*/8);
  }
  mlir::Value nodeWord =
      mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), chainNode);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, nodeWord,
                              b.addrOf("g_exc_context_node"), /*alignment=*/8);
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 0, 1).getResult(),
      b.addrOf("g_current_exception"), /*alignment=*/4);
  mlir::LLVM::MemsetOp::create(b.builder, b.loc, b.addrOf("g_current_parts"),
                               b.iconst8(0), b.iconst(120),
                               /*isVolatile=*/false);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trap);
  mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                             mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyEH_SetCurrentSuppress(): record `raise ... from None`.
void buildSetCurrentSuppress(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_SetCurrentSuppress",
                            b.builder.getFunctionType({}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(1),
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyEH_SetCurrentCause(exception triple as expanded descriptors): record the
// raised exception's explicit __cause__. `raise X from e` where `e` is the
// exception just stashed as __context__ shares that node; otherwise a fresh
// node retains the cause object (the call borrows its operands).
void buildSetCurrentCause(SupportBuilder &b) {
  llvm::SmallVector<mlir::Type, 15> inputs;
  for (int section = 0; section < 3; ++section) {
    inputs.push_back(b.ptr());
    inputs.push_back(b.ptr());
    inputs.push_back(b.i64());
    inputs.push_back(b.i64());
    inputs.push_back(b.i64());
  }
  auto fn = b.beginFunction("LyEH_SetCurrentCause",
                            b.builder.getFunctionType(inputs, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *compare = b.builder.createBlock(&body);
  mlir::Block *share = b.builder.createBlock(&body);
  mlir::Block *fresh = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value causeSlot = b.addrOf("g_exc_cause_node");
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadI64(causeSlot)});
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0), causeSlot,
                              /*alignment=*/8);
  mlir::Value context64 = b.loadI64(b.addrOf("g_exc_context_node"));
  mlir::Value contextMissing =
      b.cmpi(mlir::arith::CmpIPredicate::eq, context64, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, contextMissing, fresh,
                                 mlir::ValueRange{}, compare,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(compare);
  mlir::Value contextNode = b.intToPtr(context64);
  mlir::Value contextHeader = b.loadI64(nodeSlot(b, contextNode, 2));
  mlir::Value causeHeader = mlir::LLVM::PtrToIntOp::create(
      b.builder, b.loc, b.i64(), entry->getArgument(1));
  mlir::Value sameObject =
      b.cmpi(mlir::arith::CmpIPredicate::eq, contextHeader, causeHeader);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, sameObject, share,
                                 mlir::ValueRange{}, fresh,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(share);
  mlir::Value refcount = b.loadI64(nodeSlot(b, contextNode, 0));
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::AddIOp::create(b.builder, b.loc, refcount, b.iconst(1))
          .getResult(),
      nodeSlot(b, contextNode, 0), /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, context64, causeSlot,
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(fresh);
  mlir::Value node =
      b.call("malloc", b.ptr(), mlir::ValueRange{b.iconst(kChainNodeBytes)})
          .front();
  mlir::Value allocFailed = b.ptrEq(node, b.nullPtr());
  auto trapIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                        allocFailed, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&trapIf.getThenRegion().front());
    mlir::func::CallOp::create(b.builder, b.loc, "abort", mlir::TypeRange{},
                               mlir::ValueRange{});
  }
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(1),
                              nodeSlot(b, node, 0), /*alignment=*/8);
  for (int argIndex = 0; argIndex < 15; ++argIndex) {
    mlir::Value value = entry->getArgument(argIndex);
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))
      value =
          mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), value);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, value,
                                nodeSlot(b, node, 1 + argIndex),
                                /*alignment=*/8);
  }
  for (std::int64_t slot : {16, 17, 18, 19, 20})
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                nodeSlot(b, node, slot), /*alignment=*/8);
  mlir::Value headerWord = mlir::LLVM::PtrToIntOp::create(
      b.builder, b.loc, b.i64(), entry->getArgument(1));
  mlir::Value messageWord = mlir::LLVM::PtrToIntOp::create(
      b.builder, b.loc, b.i64(), entry->getArgument(6));
  b.call("retain_storage_raw", mlir::TypeRange{}, mlir::ValueRange{headerWord});
  b.call("retain_storage_raw", mlir::TypeRange{},
         mlir::ValueRange{messageWord});
  mlir::Value nodeWord =
      mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), node);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, nodeWord, causeSlot,
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(1),
                              b.addrOf("g_exc_suppress_context"),
                              /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ptr read_source_line(ptr path, i32 line): malloc'd copy of the line-th
// source line ("" when unavailable), trailing newline characters stripped.
void buildReadSourceLine(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "read_source_line",
      b.builder.getFunctionType({b.ptr(), b.i32()}, {b.ptr()}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkArgs = b.builder.createBlock(&body);
  mlir::Block *open = b.builder.createBlock(&body);
  mlir::Block *readHead = b.builder.createBlock(&body, body.end(), {b.i32()}, {b.loc});
  mlir::Block *readCheck = b.builder.createBlock(&body);
  mlir::Block *readNext = b.builder.createBlock(&body);
  mlir::Block *eof = b.builder.createBlock(&body);
  mlir::Block *found = b.builder.createBlock(&body);
  mlir::Block *trimHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *trimCheck = b.builder.createBlock(&body);
  mlir::Block *trimOne = b.builder.createBlock(&body);
  mlir::Block *finish = b.builder.createBlock(&body);
  mlir::Block *trap = b.builder.createBlock(&body);
  mlir::Value path = entry->getArgument(0);
  mlir::Value line = entry->getArgument(1);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value buffer =
      b.call("malloc", b.ptr(), mlir::ValueRange{b.iconst(512)}).front();
  mlir::Value failed = b.ptrEq(buffer, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                 mlir::ValueRange{}, checkArgs,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkArgs);
  b.storeI8(b.iconst8(0), buffer);
  mlir::Value pathNull = b.ptrEq(path, b.nullPtr());
  mlir::Value lineBad =
      b.cmpi(mlir::arith::CmpIPredicate::slt, line, b.iconst32(1));
  mlir::Value unusable = b.orBit(pathNull, lineBad);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, unusable, finish,
                                 mlir::ValueRange{}, open, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(open);
  mlir::Value file = b.call("fopen", b.ptr(),
                            mlir::ValueRange{path, b.addrOf(".tb_read_mode")})
                         .front();
  mlir::Value openFailed = b.ptrEq(file, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, openFailed, finish,
                                 mlir::ValueRange{}, readHead,
                                 mlir::ValueRange{b.iconst32(1)});

  b.builder.setInsertionPointToEnd(readHead);
  mlir::Value current = readHead->getArgument(0);
  mlir::Value got = b.call("fgets", b.ptr(),
                           mlir::ValueRange{buffer, b.iconst32(512), file})
                        .front();
  mlir::Value readFailed = b.ptrEq(got, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, readFailed, eof,
                                 mlir::ValueRange{}, readCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(readCheck);
  mlir::Value atLine = b.cmpi(mlir::arith::CmpIPredicate::eq, current, line);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, atLine, found,
                                 mlir::ValueRange{}, readNext,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(readNext);
  mlir::Value nextLine =
      mlir::arith::AddIOp::create(b.builder, b.loc, current, b.iconst32(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, readHead,
                             mlir::ValueRange{nextLine});

  b.builder.setInsertionPointToEnd(eof);
  b.storeI8(b.iconst8(0), buffer);
  b.call("fclose", b.i32(), mlir::ValueRange{file});
  mlir::cf::BranchOp::create(b.builder, b.loc, finish, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(found);
  b.call("fclose", b.i32(), mlir::ValueRange{file});
  mlir::Value initialLength =
      b.call("strlen", b.i64(), mlir::ValueRange{buffer}).front();
  mlir::cf::BranchOp::create(b.builder, b.loc, trimHead,
                             mlir::ValueRange{initialLength});

  b.builder.setInsertionPointToEnd(trimHead);
  mlir::Value remaining = trimHead->getArgument(0);
  mlir::Value trimDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, remaining, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, trimDone, finish,
                                 mlir::ValueRange{}, trimCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trimCheck);
  mlir::Value lastIndex =
      mlir::arith::SubIOp::create(b.builder, b.loc, remaining, b.iconst(1));
  mlir::Value lastPtr = b.gepI8(buffer, lastIndex);
  mlir::Value last = b.loadI8(lastPtr);
  mlir::Value isNewline =
      b.cmpi(mlir::arith::CmpIPredicate::eq, last, b.iconst8(10));
  mlir::Value isReturn =
      b.cmpi(mlir::arith::CmpIPredicate::eq, last, b.iconst8(13));
  // Trailing blanks go too: CPython displays the source line `.strip()`ed.
  mlir::Value isSpace =
      b.cmpi(mlir::arith::CmpIPredicate::eq, last, b.iconst8(32));
  mlir::Value isTab =
      b.cmpi(mlir::arith::CmpIPredicate::eq, last, b.iconst8(9));
  mlir::Value trimIt =
      b.orBit(b.orBit(isNewline, isReturn), b.orBit(isSpace, isTab));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, trimIt, trimOne,
                                 mlir::ValueRange{}, finish,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(trimOne);
  b.storeI8(b.iconst8(0), lastPtr);
  mlir::cf::BranchOp::create(b.builder, b.loc, trimHead,
                             mlir::ValueRange{lastIndex});

  b.builder.setInsertionPointToEnd(finish);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{buffer});

  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.ptr());
}

// ptr exception_class_name(i64 class_id): builtin exception-class name table
// (value selection; unknown ids display as "Exception").
void buildExceptionClassName(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "exception_class_name",
      b.builder.getFunctionType({b.i64()}, {b.ptr()}), /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value classId = entry->getArgument(0);
  mlir::Value name = b.addrOf(".tb_class.Exception");
  for (const py::exceptions::BuiltinExceptionInfo &info :
       py::exceptions::kBuiltinExceptions) {
    mlir::Value matches =
        b.cmpi(mlir::arith::CmpIPredicate::eq, classId, b.iconst(info.classId));
    name = mlir::arith::SelectOp::create(
        b.builder, b.loc, matches,
        b.addrOf((llvm::Twine(".tb_class.") + info.name).str()), name);
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{name});
}

// i64 leading_whitespace(ptr line): count of leading spaces/tabs.
void buildLeadingWhitespace(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "leading_whitespace",
      b.builder.getFunctionType({b.ptr()}, {b.i64()}), /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *head = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *check = b.builder.createBlock(&body);
  mlir::Block *advance = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Value line = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value length =
      b.call("strlen", b.i64(), mlir::ValueRange{line}).front();
  mlir::cf::BranchOp::create(b.builder, b.loc, head,
                             mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(head);
  mlir::Value index = head->getArgument(0);
  mlir::Value atEnd = b.cmpi(mlir::arith::CmpIPredicate::eq, index, length);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, atEnd, done,
                                 mlir::ValueRange{index}, check,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(check);
  mlir::Value ch = b.loadI8(b.gepI8(line, index));
  mlir::Value isSpace =
      b.cmpi(mlir::arith::CmpIPredicate::eq, ch, b.iconst8(32));
  mlir::Value isTab = b.cmpi(mlir::arith::CmpIPredicate::eq, ch, b.iconst8(9));
  mlir::Value isBlank = b.orBit(isSpace, isTab);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isBlank, advance,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{index});

  b.builder.setInsertionPointToEnd(advance);
  mlir::Value next =
      mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{next});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{done->getArgument(0)});
}

// void print_marker(ptr line, i32 col, i32 endCol): the CPython-style anchor
// underline for the failing range on stderr. Mirrors CPython 3.14's display
// heuristics on the source text alone (the runtime has no instruction
// anchors): a call/subscript range splits at its first `(`/`[` into
// `~~~^^^`, an operator range puts `^` over the operator run, and a range
// with no anchor renders all carets — unless it covers the whole line, which
// CPython suppresses entirely.
void buildPrintMarker(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_marker",
      b.builder.getFunctionType({b.ptr(), b.i32(), b.i32()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *findStart = b.builder.createBlock(&body);
  mlir::Block *scanHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *scanCheck = b.builder.createBlock(&body);
  mlir::Block *scanNext = b.builder.createBlock(&body);
  mlir::Block *haveStart = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *computeEnd = b.builder.createBlock(&body);
  mlir::Block *splitHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *splitCheck = b.builder.createBlock(&body);
  mlir::Block *opHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *opCheck = b.builder.createBlock(&body);
  mlir::Block *opEndHead = b.builder.createBlock(&body, body.end(),
                                                 {b.i64(), b.i64()},
                                                 {b.loc, b.loc});
  mlir::Block *noAnchor = b.builder.createBlock(&body);
  mlir::Block *emit = b.builder.createBlock(&body, body.end(),
                                            {b.i64(), b.i64()},
                                            {b.loc, b.loc});
  mlir::Block *padHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *padOne = b.builder.createBlock(&body);
  mlir::Block *charHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *charOne = b.builder.createBlock(&body);
  mlir::Block *newline = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value line = entry->getArgument(0);
  mlir::Value col = entry->getArgument(1);
  mlir::Value endCol = entry->getArgument(2);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value length =
      b.call("strlen", b.i64(), mlir::ValueRange{line}).front();
  mlir::Value emptyLine =
      b.cmpi(mlir::arith::CmpIPredicate::eq, length, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, emptyLine, done,
                                 mlir::ValueRange{}, findStart,
                                 mlir::ValueRange{});

  // Marker start: the given column when it lands inside the line, otherwise
  // the first non-blank character.
  b.builder.setInsertionPointToEnd(findStart);
  mlir::Value colPositive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, col, b.iconst32(0));
  mlir::Value colWide =
      mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(), col);
  mlir::Value colInLine =
      b.cmpi(mlir::arith::CmpIPredicate::slt, colWide, length);
  mlir::Value useColumn =
      mlir::arith::AndIOp::create(b.builder, b.loc, colPositive, colInLine);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, useColumn, haveStart,
                                 mlir::ValueRange{colWide}, scanHead,
                                 mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(scanHead);
  mlir::Value scanIndex = scanHead->getArgument(0);
  mlir::Value scanEnd =
      b.cmpi(mlir::arith::CmpIPredicate::eq, scanIndex, length);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, scanEnd, haveStart,
                                 mlir::ValueRange{scanIndex}, scanCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(scanCheck);
  mlir::Value scanCh = b.loadI8(b.gepI8(line, scanIndex));
  mlir::Value scanSpace =
      b.cmpi(mlir::arith::CmpIPredicate::eq, scanCh, b.iconst8(32));
  mlir::Value scanTab =
      b.cmpi(mlir::arith::CmpIPredicate::eq, scanCh, b.iconst8(9));
  mlir::Value scanBlank = b.orBit(scanSpace, scanTab);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, scanBlank, scanNext,
                                 mlir::ValueRange{}, haveStart,
                                 mlir::ValueRange{scanIndex});

  b.builder.setInsertionPointToEnd(scanNext);
  mlir::Value scanAdvance =
      mlir::arith::AddIOp::create(b.builder, b.loc, scanIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, scanHead,
                             mlir::ValueRange{scanAdvance});

  b.builder.setInsertionPointToEnd(haveStart);
  mlir::Value start = haveStart->getArgument(0);
  mlir::Value startPastEnd =
      b.cmpi(mlir::arith::CmpIPredicate::uge, start, length);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, startPastEnd, done,
                                 mlir::ValueRange{}, computeEnd,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(computeEnd);
  // Marker end: endCol when it is a usable range end, clamped to the line;
  // degenerate ranges underline a single character.
  mlir::Value endAfterCol =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endCol, col);
  mlir::Value endPositive =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endCol, b.iconst32(0));
  mlir::Value endUsable =
      mlir::arith::AndIOp::create(b.builder, b.loc, endAfterCol, endPositive);
  mlir::Value endWide =
      mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(), endCol);
  mlir::Value endOverLength =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endWide, length);
  mlir::Value endClamped = mlir::arith::SelectOp::create(
      b.builder, b.loc, endOverLength, length, endWide);
  mlir::Value endOrLength = mlir::arith::SelectOp::create(
      b.builder, b.loc, endUsable, endClamped, length);
  mlir::Value endTooSmall =
      b.cmpi(mlir::arith::CmpIPredicate::ule, endOrLength, start);
  mlir::Value startPlusOne =
      mlir::arith::AddIOp::create(b.builder, b.loc, start, b.iconst(1));
  mlir::Value plusOneOver =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, startPlusOne, length);
  mlir::Value plusOneClamped = mlir::arith::SelectOp::create(
      b.builder, b.loc, plusOneOver, length, startPlusOne);
  mlir::Value markerEnd = mlir::arith::SelectOp::create(
      b.builder, b.loc, endTooSmall, plusOneClamped, endOrLength);
  mlir::cf::BranchOp::create(b.builder, b.loc, splitHead,
                             mlir::ValueRange{start});

  // Anchor pass 1: a call/subscript splits at its first `(` / `[`; the head
  // renders as tildes, the trailer as carets.
  b.builder.setInsertionPointToEnd(splitHead);
  mlir::Value splitIndex = splitHead->getArgument(0);
  mlir::Value splitDone =
      b.cmpi(mlir::arith::CmpIPredicate::uge, splitIndex, markerEnd);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, splitDone, opHead,
                                 mlir::ValueRange{start}, splitCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(splitCheck);
  mlir::Value splitCh = b.loadI8(b.gepI8(line, splitIndex));
  mlir::Value isParen =
      b.cmpi(mlir::arith::CmpIPredicate::eq, splitCh, b.iconst8('('));
  mlir::Value isBracket =
      b.cmpi(mlir::arith::CmpIPredicate::eq, splitCh, b.iconst8('['));
  mlir::Value isSplit = b.orBit(isParen, isBracket);
  mlir::Value splitNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, splitIndex, b.iconst(1));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isSplit, emit,
                                 mlir::ValueRange{splitIndex, markerEnd},
                                 splitHead, mlir::ValueRange{splitNext});

  // Anchor pass 2: a binary-operator run gets the carets (`a / b` -> `~~^~~`).
  b.builder.setInsertionPointToEnd(opHead);
  mlir::Value opIndex = opHead->getArgument(0);
  mlir::Value opScanDone =
      b.cmpi(mlir::arith::CmpIPredicate::uge, opIndex, markerEnd);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, opScanDone, noAnchor,
                                 mlir::ValueRange{}, opCheck,
                                 mlir::ValueRange{});

  auto isOperatorChar = [&](mlir::Value ch) {
    mlir::Value result;
    for (char c : {'+', '-', '*', '/', '%', '@', '&', '|', '^', '<', '>'}) {
      mlir::Value matches =
          b.cmpi(mlir::arith::CmpIPredicate::eq, ch, b.iconst8(c));
      result = result ? b.orBit(result, matches) : matches;
    }
    return result;
  };

  b.builder.setInsertionPointToEnd(opCheck);
  mlir::Value opCh = b.loadI8(b.gepI8(line, opIndex));
  mlir::Value opFound = isOperatorChar(opCh);
  mlir::Value opNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, opIndex, b.iconst(1));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, opFound, opEndHead,
                                 mlir::ValueRange{opIndex, opNext}, opHead,
                                 mlir::ValueRange{opNext});

  b.builder.setInsertionPointToEnd(opEndHead);
  mlir::Value opStart = opEndHead->getArgument(0);
  mlir::Value opEnd = opEndHead->getArgument(1);
  mlir::Value opRunDone =
      b.cmpi(mlir::arith::CmpIPredicate::uge, opEnd, markerEnd);
  mlir::Block *opEndCheck = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(opEndHead);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, opRunDone, emit,
                                 mlir::ValueRange{opStart, opEnd}, opEndCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(opEndCheck);
  mlir::Value opEndCh = b.loadI8(b.gepI8(line, opEnd));
  mlir::Value opRunContinues = isOperatorChar(opEndCh);
  mlir::Value opEndNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, opEnd, b.iconst(1));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, opRunContinues, opEndHead,
                                 mlir::ValueRange{opStart, opEndNext}, emit,
                                 mlir::ValueRange{opStart, opEnd});

  // No anchors: CPython suppresses the marker line when the range covers the
  // whole (stripped) line, otherwise renders all carets.
  b.builder.setInsertionPointToEnd(noAnchor);
  mlir::Value coversStart =
      b.cmpi(mlir::arith::CmpIPredicate::eq, start, b.iconst(0));
  mlir::Value coversEnd =
      b.cmpi(mlir::arith::CmpIPredicate::uge, markerEnd, length);
  mlir::Value wholeLine =
      mlir::arith::AndIOp::create(b.builder, b.loc, coversStart, coversEnd);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, wholeLine, done,
                                 mlir::ValueRange{}, emit,
                                 mlir::ValueRange{start, markerEnd});

  b.builder.setInsertionPointToEnd(emit);
  mlir::Value stderrFd = b.iconst32(2);
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, b.addrOf(".tb_indent")});
  mlir::cf::BranchOp::create(b.builder, b.loc, padHead,
                             mlir::ValueRange{b.iconst(0)});

  // Alignment padding: tabs stay tabs so the marker lines up under the code.
  b.builder.setInsertionPointToEnd(padHead);
  mlir::Value padIndex = padHead->getArgument(0);
  mlir::Value padDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, padIndex, start);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, padDone, charHead,
                                 mlir::ValueRange{start}, padOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(padOne);
  mlir::Value padCh = b.loadI8(b.gepI8(line, padIndex));
  mlir::Value padIsTab =
      b.cmpi(mlir::arith::CmpIPredicate::eq, padCh, b.iconst8(9));
  mlir::Value padOut = mlir::arith::SelectOp::create(
      b.builder, b.loc, padIsTab, b.iconst8(9), b.iconst8(32));
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), padOut});
  mlir::Value padNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, padIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, padHead,
                             mlir::ValueRange{padNext});

  b.builder.setInsertionPointToEnd(charHead);
  mlir::Value charIndex = charHead->getArgument(0);
  mlir::Value charsDone =
      b.cmpi(mlir::arith::CmpIPredicate::uge, charIndex, markerEnd);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, charsDone, newline,
                                 mlir::ValueRange{}, charOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(charOne);
  mlir::Value caretStart = emit->getArgument(0);
  mlir::Value caretEnd = emit->getArgument(1);
  mlir::Value afterCaretStart =
      b.cmpi(mlir::arith::CmpIPredicate::uge, charIndex, caretStart);
  mlir::Value beforeCaretEnd =
      b.cmpi(mlir::arith::CmpIPredicate::ult, charIndex, caretEnd);
  mlir::Value inCaret = mlir::arith::AndIOp::create(
      b.builder, b.loc, afterCaretStart, beforeCaretEnd);
  mlir::Value markerCh = mlir::arith::SelectOp::create(
      b.builder, b.loc, inCaret, b.iconst8('^'), b.iconst8('~'));
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), markerCh});
  mlir::Value charNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, charIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, charHead,
                             mlir::ValueRange{charNext});

  b.builder.setInsertionPointToEnd(newline);
  b.call("write_len", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_newline"),
                          b.iconst(1)});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void print_trace_frame(ptr frame): "  File ..., line N, in fn" + the source
// line + optional marker, on stderr.
void buildPrintTraceFrame(SupportBuilder &b) {
  auto fn = b.beginFunction("print_trace_frame",
                            b.builder.getFunctionType({b.ptr()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *sourceShown = b.builder.createBlock(&body);
  mlir::Block *marker = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value frame = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Type frameType = tracebackFrameType(b);
  auto bufferType = mlir::LLVM::LLVMArrayType::get(b.i8(), 1024);
  mlir::Value bufferSlot = mlir::LLVM::AllocaOp::create(
      b.builder, b.loc, b.ptr(), bufferType, b.iconst32(1), /*alignment=*/1);
  mlir::Value buffer = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), bufferType, bufferSlot,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(0)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
  mlir::Value file = b.loadPtrVal(b.frameField(frameType, frame, 0));
  mlir::Value function = b.loadPtrVal(b.frameField(frameType, frame, 1));
  mlir::Value lineNo = b.loadI32(b.frameField(frameType, frame, 2));
  mlir::Value col = b.loadI32(b.frameField(frameType, frame, 3));
  mlir::Value endCol = b.loadI32(b.frameField(frameType, frame, 5));
  mlir::Value hasMarker = b.loadI32(b.frameField(frameType, frame, 6));
  auto snprintfType = mlir::LLVM::LLVMFunctionType::get(
      b.i32(), {b.ptr(), b.i64(), b.ptr()}, /*isVarArg=*/true);
  auto formatted = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_frame"),
                       file, lineNo, function});
  b.call("write_buffered", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), buffer, formatted.getResult()});
  mlir::Value sourceLine =
      b.call("read_source_line", b.ptr(), mlir::ValueRange{file, lineNo})
          .front();
  mlir::Value sourceLength =
      b.call("strlen", b.i64(), mlir::ValueRange{sourceLine}).front();
  mlir::Value haveSource =
      b.cmpi(mlir::arith::CmpIPredicate::ne, sourceLength, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveSource, sourceShown,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(sourceShown);
  mlir::Value indentWidth =
      b.call("leading_whitespace", b.i64(), mlir::ValueRange{sourceLine})
          .front();
  mlir::Value trimmed = b.gepI8(sourceLine, indentWidth);
  mlir::Value trimmedLength =
      b.call("strlen", b.i64(), mlir::ValueRange{trimmed}).front();
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_indent")});
  b.call("write_len", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), trimmed, trimmedLength});
  b.call("write_len", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_newline"),
                          b.iconst(1)});
  mlir::Value wantMarker =
      b.cmpi(mlir::arith::CmpIPredicate::ne, hasMarker, b.iconst32(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, wantMarker, marker,
                                 mlir::ValueRange{}, done,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(marker);
  // Columns are absolute; the printed line lost its indentation.
  mlir::Value indent32 =
      mlir::arith::TruncIOp::create(b.builder, b.loc, b.i32(), indentWidth);
  mlir::Value colPast =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, col, indent32);
  mlir::Value colShift =
      mlir::arith::SubIOp::create(b.builder, b.loc, col, indent32);
  mlir::Value colAdjusted = mlir::arith::SelectOp::create(
      b.builder, b.loc, colPast, colShift, b.iconst32(0));
  mlir::Value endPast =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, endCol, indent32);
  mlir::Value endShift =
      mlir::arith::SubIOp::create(b.builder, b.loc, endCol, indent32);
  mlir::Value endAdjusted = mlir::arith::SelectOp::create(
      b.builder, b.loc, endPast, endShift, b.iconst32(0));
  b.call("print_marker", mlir::TypeRange{},
         mlir::ValueRange{trimmed, colAdjusted, endAdjusted});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{sourceLine});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ptr utf8_message_cstr(ptr msg_header, ptr data, i64 offset, i64 len,
// i64 stride): malloc'd NUL-terminated UTF-8 re-encoding of a PEP 393 str's
// code-unit buffer. The str's character width lives at header+16; a missing
// or unexpected width degrades to a raw byte copy (latin1-safe for ASCII).
void buildUtf8MessageCStr(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "utf8_message_cstr",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()},
                                {b.ptr()}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *emptyCopy = b.builder.createBlock(&body);
  mlir::Block *widthLoad = b.builder.createBlock(&body);
  mlir::Block *alloc =
      b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *trap = b.builder.createBlock(&body);
  mlir::Value header = entry->getArgument(0);
  mlir::Value data = entry->getArgument(1);
  mlir::Value offset = entry->getArgument(2);
  mlir::Value len = entry->getArgument(3);
  mlir::Value stride = entry->getArgument(4);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value zero = b.iconst(0);
  mlir::Value one = b.iconst(1);
  mlir::Value dataNull = b.ptrEq(data, b.nullPtr());
  mlir::Value lenEmpty = b.cmpi(mlir::arith::CmpIPredicate::sle, len, zero);
  mlir::Value strideOdd = b.cmpi(mlir::arith::CmpIPredicate::ne, stride, one);
  mlir::Value unusable = b.orBit(b.orBit(dataNull, lenEmpty), strideOdd);
  mlir::Value headerNull = b.ptrEq(header, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, unusable, emptyCopy,
                                 mlir::ValueRange{}, widthLoad,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(emptyCopy);
  // Strided views fall back to a raw copy; empty/missing views produce "".
  auto rawIf = mlir::scf::IfOp::create(b.builder, b.loc,
                                       mlir::TypeRange{b.ptr()},
                                       b.orBit(dataNull, lenEmpty),
                                       /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&rawIf.getThenRegion().front());
    mlir::Value empty =
        b.call("copy_cstr", b.ptr(), mlir::ValueRange{b.nullPtr()}).front();
    mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{empty});
    b.builder.setInsertionPointToStart(&rawIf.getElseRegion().front());
    mlir::Value raw = b.call("copy_i8_memref", b.ptr(),
                             mlir::ValueRange{data, offset, len, stride})
                          .front();
    mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{raw});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{rawIf.getResult(0)});

  b.builder.setInsertionPointToEnd(widthLoad);
  auto widthIf = mlir::scf::IfOp::create(b.builder, b.loc,
                                         mlir::TypeRange{b.i64()}, headerNull,
                                         /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&widthIf.getThenRegion().front());
    mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{one});
    b.builder.setInsertionPointToStart(&widthIf.getElseRegion().front());
    mlir::Value stored = b.loadI64(b.gepI8(header, b.iconst(16)));
    mlir::Value two = b.iconst(2);
    mlir::Value four = b.iconst(4);
    mlir::Value isTwo = b.cmpi(mlir::arith::CmpIPredicate::eq, stored, two);
    mlir::Value isFour = b.cmpi(mlir::arith::CmpIPredicate::eq, stored, four);
    mlir::Value wide =
        mlir::arith::SelectOp::create(b.builder, b.loc, isFour, four, one);
    mlir::Value width =
        mlir::arith::SelectOp::create(b.builder, b.loc, isTwo, two, wide);
    mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{width});
  }
  mlir::cf::BranchOp::create(b.builder, b.loc, alloc,
                             mlir::ValueRange{widthIf.getResult(0)});

  b.builder.setInsertionPointToEnd(alloc);
  mlir::Value width = alloc->getArgument(0);
  mlir::Value count =
      mlir::arith::DivSIOp::create(b.builder, b.loc, len, width);
  mlir::Value capacity = mlir::arith::AddIOp::create(
      b.builder, b.loc,
      mlir::arith::MulIOp::create(b.builder, b.loc, count, b.iconst(4))
          .getResult(),
      one);
  mlir::Value out =
      b.call("malloc", b.ptr(), mlir::ValueRange{capacity}).front();
  mlir::Value outFailed = b.ptrEq(out, b.nullPtr());
  mlir::Block *encode = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(alloc);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, outFailed, trap,
                                 mlir::ValueRange{}, encode,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(encode);
  mlir::Value zeroIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
  mlir::Value oneIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
  mlir::Value countIndex = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.builder.getIndexType(), count);
  auto loop = mlir::scf::ForOp::create(
      b.builder, b.loc, zeroIndex, countIndex, oneIndex,
      mlir::ValueRange{zero},
      [&](mlir::OpBuilder &nested, mlir::Location loc, mlir::Value iv,
          mlir::ValueRange iter) {
        mlir::Value i =
            mlir::arith::IndexCastOp::create(nested, loc, b.i64(), iv);
        mlir::Value base = mlir::arith::AddIOp::create(
            nested, loc, offset,
            mlir::arith::MulIOp::create(nested, loc, i, width).getResult());
        auto loadByte = [&](std::int64_t at) {
          mlir::Value position = mlir::arith::AddIOp::create(
              nested, loc, base,
              mlir::arith::ConstantIntOp::create(nested, loc, b.i64(), at)
                  .getResult());
          mlir::Value pointer = mlir::LLVM::GEPOp::create(
              nested, loc, b.ptr(), b.i8(), data, mlir::ValueRange{position});
          mlir::Value byte = mlir::LLVM::LoadOp::create(nested, loc, b.i8(),
                                                        pointer,
                                                        /*alignment=*/1);
          return mlir::arith::ExtUIOp::create(nested, loc, b.i64(), byte)
              .getResult();
        };
        auto shifted = [&](mlir::Value value, std::int64_t by) {
          return mlir::arith::ShLIOp::create(
                     nested, loc, value,
                     mlir::arith::ConstantIntOp::create(nested, loc, b.i64(),
                                                        by)
                         .getResult())
              .getResult();
        };
        mlir::Value isOne = mlir::arith::CmpIOp::create(
            nested, loc, mlir::arith::CmpIPredicate::eq, width,
            mlir::arith::ConstantIntOp::create(nested, loc, b.i64(), 1)
                .getResult());
        auto cpIf = mlir::scf::IfOp::create(nested, loc,
                                            mlir::TypeRange{b.i64()}, isOne,
                                            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(nested);
          nested.setInsertionPointToStart(&cpIf.getThenRegion().front());
          mlir::scf::YieldOp::create(nested, loc,
                                     mlir::ValueRange{loadByte(0)});
          nested.setInsertionPointToStart(&cpIf.getElseRegion().front());
          mlir::Value low = mlir::arith::OrIOp::create(
              nested, loc, loadByte(0), shifted(loadByte(1), 8));
          mlir::Value isTwo = mlir::arith::CmpIOp::create(
              nested, loc, mlir::arith::CmpIPredicate::eq, width,
              mlir::arith::ConstantIntOp::create(nested, loc, b.i64(), 2)
                  .getResult());
          auto wideIf = mlir::scf::IfOp::create(nested, loc,
                                                mlir::TypeRange{b.i64()},
                                                isTwo,
                                                /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard inner(nested);
            nested.setInsertionPointToStart(&wideIf.getThenRegion().front());
            mlir::scf::YieldOp::create(nested, loc, mlir::ValueRange{low});
            nested.setInsertionPointToStart(&wideIf.getElseRegion().front());
            mlir::Value high = mlir::arith::OrIOp::create(
                nested, loc, shifted(loadByte(2), 16),
                shifted(loadByte(3), 24));
            mlir::Value full =
                mlir::arith::OrIOp::create(nested, loc, low, high);
            mlir::scf::YieldOp::create(nested, loc, mlir::ValueRange{full});
          }
          mlir::scf::YieldOp::create(nested, loc,
                                     mlir::ValueRange{wideIf.getResult(0)});
        }
        mlir::Value cp = cpIf.getResult(0);
        mlir::Value cursor = iter.front();
        auto storeByte = [&](mlir::Value value, mlir::Value at) {
          mlir::Value pointer = mlir::LLVM::GEPOp::create(
              nested, loc, b.ptr(), b.i8(), out, mlir::ValueRange{at});
          mlir::Value narrow =
              mlir::arith::TruncIOp::create(nested, loc, b.i8(), value);
          mlir::LLVM::StoreOp::create(nested, loc, narrow, pointer,
                                      /*alignment=*/1);
        };
        auto konst = [&](std::int64_t value) {
          return mlir::arith::ConstantIntOp::create(nested, loc, b.i64(),
                                                    value)
              .getResult();
        };
        auto orI = [&](mlir::Value a, mlir::Value c) {
          return mlir::arith::OrIOp::create(nested, loc, a, c).getResult();
        };
        auto andI = [&](mlir::Value a, mlir::Value c) {
          return mlir::arith::AndIOp::create(nested, loc, a, c).getResult();
        };
        auto shr = [&](mlir::Value a, std::int64_t by) {
          return mlir::arith::ShRUIOp::create(nested, loc, a, konst(by))
              .getResult();
        };
        auto at = [&](mlir::Value basePos, std::int64_t plus) {
          return mlir::arith::AddIOp::create(nested, loc, basePos,
                                             konst(plus))
              .getResult();
        };
        mlir::Value ltAscii = mlir::arith::CmpIOp::create(
            nested, loc, mlir::arith::CmpIPredicate::ult, cp, konst(0x80));
        auto encIf = mlir::scf::IfOp::create(nested, loc,
                                             mlir::TypeRange{b.i64()},
                                             ltAscii,
                                             /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(nested);
          nested.setInsertionPointToStart(&encIf.getThenRegion().front());
          storeByte(cp, cursor);
          mlir::scf::YieldOp::create(nested, loc,
                                     mlir::ValueRange{at(cursor, 1)});
          nested.setInsertionPointToStart(&encIf.getElseRegion().front());
          mlir::Value ltTwo = mlir::arith::CmpIOp::create(
              nested, loc, mlir::arith::CmpIPredicate::ult, cp,
              konst(0x800));
          auto twoIf = mlir::scf::IfOp::create(nested, loc,
                                               mlir::TypeRange{b.i64()},
                                               ltTwo,
                                               /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard inner(nested);
            nested.setInsertionPointToStart(&twoIf.getThenRegion().front());
            storeByte(orI(konst(0xC0), shr(cp, 6)), cursor);
            storeByte(orI(konst(0x80), andI(cp, konst(0x3F))), at(cursor, 1));
            mlir::scf::YieldOp::create(nested, loc,
                                       mlir::ValueRange{at(cursor, 2)});
            nested.setInsertionPointToStart(&twoIf.getElseRegion().front());
            mlir::Value ltThree = mlir::arith::CmpIOp::create(
                nested, loc, mlir::arith::CmpIPredicate::ult, cp,
                konst(0x10000));
            auto threeIf = mlir::scf::IfOp::create(nested, loc,
                                                   mlir::TypeRange{b.i64()},
                                                   ltThree,
                                                   /*withElseRegion=*/true);
            {
              mlir::OpBuilder::InsertionGuard innermost(nested);
              nested.setInsertionPointToStart(
                  &threeIf.getThenRegion().front());
              storeByte(orI(konst(0xE0), shr(cp, 12)), cursor);
              storeByte(orI(konst(0x80), andI(shr(cp, 6), konst(0x3F))),
                        at(cursor, 1));
              storeByte(orI(konst(0x80), andI(cp, konst(0x3F))),
                        at(cursor, 2));
              mlir::scf::YieldOp::create(nested, loc,
                                         mlir::ValueRange{at(cursor, 3)});
              nested.setInsertionPointToStart(
                  &threeIf.getElseRegion().front());
              storeByte(orI(konst(0xF0), shr(cp, 18)), cursor);
              storeByte(orI(konst(0x80), andI(shr(cp, 12), konst(0x3F))),
                        at(cursor, 1));
              storeByte(orI(konst(0x80), andI(shr(cp, 6), konst(0x3F))),
                        at(cursor, 2));
              storeByte(orI(konst(0x80), andI(cp, konst(0x3F))),
                        at(cursor, 3));
              mlir::scf::YieldOp::create(nested, loc,
                                         mlir::ValueRange{at(cursor, 4)});
            }
            mlir::scf::YieldOp::create(
                nested, loc, mlir::ValueRange{threeIf.getResult(0)});
          }
          mlir::scf::YieldOp::create(nested, loc,
                                     mlir::ValueRange{twoIf.getResult(0)});
        }
        mlir::scf::YieldOp::create(nested, loc,
                                   mlir::ValueRange{encIf.getResult(0)});
      });
  b.storeI8(b.iconst8(0), b.gepI8(out, loop.getResult(0)));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{out});

  b.builder.setInsertionPointToEnd(trap);
  b.emitTrap(b.ptr());
}

// void print_exception_summary(i64 class_id, ptr msg_header, message view):
// the final "Class: message" line (or the class-only / invalid / unknown
// forms). The message is re-encoded from code units to UTF-8 for display.
void buildPrintExceptionSummary(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_exception_summary",
      b.builder.getFunctionType(
          {b.i64(), b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *checkEmpty = b.builder.createBlock(&body);
  mlir::Block *checkNull = b.builder.createBlock(&body);
  mlir::Block *withMessage = b.builder.createBlock(&body);
  mlir::Block *classOnly = b.builder.createBlock(&body);
  mlir::Block *invalid = b.builder.createBlock(&body);
  mlir::Block *unknown = b.builder.createBlock(&body);
  mlir::Value classId = entry->getArgument(0);
  mlir::Value msgHeader = entry->getArgument(1);
  mlir::Value data = entry->getArgument(2);
  mlir::Value offset = entry->getArgument(3);
  mlir::Value len = entry->getArgument(4);
  mlir::Value stride = entry->getArgument(5);

  b.builder.setInsertionPointToEnd(entry);
  auto bufferType = mlir::LLVM::LLVMArrayType::get(b.i8(), 1024);
  mlir::Value bufferSlot = mlir::LLVM::AllocaOp::create(
      b.builder, b.loc, b.ptr(), bufferType, b.iconst32(1), /*alignment=*/1);
  mlir::Value buffer = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), bufferType, bufferSlot,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(0)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
  mlir::Value className =
      b.call("exception_class_name", b.ptr(), mlir::ValueRange{classId})
          .front();
  auto snprintfType = mlir::LLVM::LLVMFunctionType::get(
      b.i32(), {b.ptr(), b.i64(), b.ptr()}, /*isVarArg=*/true);
  auto emitBuffered = [&](mlir::Value formattedLength) {
    b.call("write_buffered", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), buffer, formattedLength});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  };
  mlir::Value zero = b.iconst(0);
  mlir::Value offsetNeg =
      b.cmpi(mlir::arith::CmpIPredicate::slt, offset, zero);
  mlir::Value lenNeg = b.cmpi(mlir::arith::CmpIPredicate::slt, len, zero);
  mlir::Value strideBad =
      b.cmpi(mlir::arith::CmpIPredicate::slt, stride, b.iconst(1));
  mlir::Value badView = b.orBit(b.orBit(offsetNeg, lenNeg), strideBad);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, badView, invalid,
                                 mlir::ValueRange{}, checkEmpty,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkEmpty);
  mlir::Value emptyMessage =
      b.cmpi(mlir::arith::CmpIPredicate::eq, len, zero);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, emptyMessage, classOnly,
                                 mlir::ValueRange{}, checkNull,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(checkNull);
  mlir::Value dataNull = b.ptrEq(data, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, dataNull, unknown,
                                 mlir::ValueRange{}, withMessage,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(withMessage);
  // write_len instead of snprintf: the re-encoded message may exceed the
  // 1023-byte snprintf clamp, and this path must not truncate valid UTF-8.
  mlir::Value message =
      b.call("utf8_message_cstr", b.ptr(),
             mlir::ValueRange{msgHeader, data, offset, len, stride})
          .front();
  mlir::Value stderrFd = b.iconst32(2);
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, className});
  b.stringGlobal(".tb_colon_space", ": ");
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, b.addrOf(".tb_colon_space")});
  b.call("write_cstr", mlir::TypeRange{}, mlir::ValueRange{stderrFd, message});
  b.call("write_len", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, b.addrOf(".tb_newline"), b.iconst(1)});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{message});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(classOnly);
  auto formattedClass = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_class"),
                       className});
  emitBuffered(formattedClass.getResult());

  b.builder.setInsertionPointToEnd(invalid);
  auto formattedInvalid = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_invalid"),
                       className});
  emitBuffered(formattedInvalid.getResult());

  b.builder.setInsertionPointToEnd(unknown);
  auto formattedUnknown = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_unknown"),
                       className});
  emitBuffered(formattedUnknown.getResult());
}

// void print_chain_node(i64 node): one section of a chained-exception report:
// the node's own chain first (recursively), the matching separator, then its
// traceback (when captured) and summary line.
void buildPrintChainNode(SupportBuilder &b) {
  auto fn = b.beginFunction("print_chain_node",
                            b.builder.getFunctionType({b.i64()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *present = b.builder.createBlock(&body);
  mlir::Block *causeBlock = b.builder.createBlock(&body);
  mlir::Block *contextCheck = b.builder.createBlock(&body);
  mlir::Block *contextBlock = b.builder.createBlock(&body);
  mlir::Block *ownSection = b.builder.createBlock(&body);
  mlir::Block *withHeader = b.builder.createBlock(&body);
  mlir::Block *frameHead =
      b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *frameOne = b.builder.createBlock(&body);
  mlir::Block *summary = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value node64 = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value missing =
      b.cmpi(mlir::arith::CmpIPredicate::eq, node64, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, missing, done,
                                 mlir::ValueRange{}, present,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(present);
  mlir::Value node = b.intToPtr(node64);
  mlir::Value cause = b.loadI64(nodeSlot(b, node, 18));
  mlir::Value haveCause =
      b.cmpi(mlir::arith::CmpIPredicate::ne, cause, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveCause, causeBlock,
                                 mlir::ValueRange{}, contextCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(causeBlock);
  b.call("print_chain_node", mlir::TypeRange{}, mlir::ValueRange{cause});
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_sep_cause")});
  mlir::cf::BranchOp::create(b.builder, b.loc, ownSection,
                             mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(contextCheck);
  mlir::Value context = b.loadI64(nodeSlot(b, node, 19));
  mlir::Value suppress = b.loadI64(nodeSlot(b, node, 20));
  mlir::Value haveContext =
      b.cmpi(mlir::arith::CmpIPredicate::ne, context, b.iconst(0));
  mlir::Value showContext = mlir::arith::AndIOp::create(
      b.builder, b.loc, haveContext,
      b.cmpi(mlir::arith::CmpIPredicate::eq, suppress, b.iconst(0)));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, showContext, contextBlock,
                                 mlir::ValueRange{}, ownSection,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(contextBlock);
  b.call("print_chain_node", mlir::TypeRange{}, mlir::ValueRange{context});
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_sep_context")});
  mlir::cf::BranchOp::create(b.builder, b.loc, ownSection,
                             mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(ownSection);
  mlir::Value count = b.loadI64(nodeSlot(b, node, 17));
  mlir::Value haveFrames =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, count, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveFrames, withHeader,
                                 mlir::ValueRange{}, summary,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(withHeader);
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_header")});
  mlir::cf::BranchOp::create(b.builder, b.loc, frameHead,
                             mlir::ValueRange{count});

  b.builder.setInsertionPointToEnd(frameHead);
  mlir::Value remaining = frameHead->getArgument(0);
  mlir::Value framesDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, remaining, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, framesDone, summary,
                                 mlir::ValueRange{}, frameOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(frameOne);
  mlir::Value top =
      mlir::arith::SubIOp::create(b.builder, b.loc, remaining, b.iconst(1));
  mlir::Value frames = b.intToPtr(b.loadI64(nodeSlot(b, node, 16)));
  mlir::Value frame = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), tracebackFrameType(b), frames,
      mlir::ValueRange{top});
  b.call("print_trace_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::cf::BranchOp::create(b.builder, b.loc, frameHead,
                             mlir::ValueRange{top});

  b.builder.setInsertionPointToEnd(summary);
  mlir::Value aligned = b.intToPtr(b.loadI64(nodeSlot(b, node, 2)));
  mlir::Value offset0 = b.loadI64(nodeSlot(b, node, 3));
  mlir::Value stride0 = b.loadI64(nodeSlot(b, node, 5));
  mlir::Value classIndex = mlir::arith::AddIOp::create(
      b.builder, b.loc, offset0,
      mlir::arith::MulIOp::create(b.builder, b.loc, stride0, b.iconst(2))
          .getResult());
  mlir::Value classId = b.loadI64(b.gepI64(aligned, classIndex));
  mlir::Value msgHeader = b.intToPtr(b.loadI64(nodeSlot(b, node, 7)));
  mlir::Value msgData = b.intToPtr(b.loadI64(nodeSlot(b, node, 12)));
  mlir::Value msgOffset = b.loadI64(nodeSlot(b, node, 13));
  mlir::Value msgLen = b.loadI64(nodeSlot(b, node, 14));
  mlir::Value msgStride = b.loadI64(nodeSlot(b, node, 15));
  b.call("print_exception_summary", mlir::TypeRange{},
         mlir::ValueRange{classId, msgHeader, msgData, msgOffset, msgLen,
                          msgStride});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyTraceback_PrintMessage(i64 class_id, ptr msg_header, message view):
// chained sections (cause/context, innermost first) + header + frames (most
// recent last, printed from the top of the stack downwards) + summary line,
// on stderr.
void buildTracebackPrintMessage(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PrintMessage",
      b.builder.getFunctionType(
          {b.i64(), b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *causeBlock = b.builder.createBlock(&body);
  mlir::Block *contextCheck = b.builder.createBlock(&body);
  mlir::Block *contextBlock = b.builder.createBlock(&body);
  mlir::Block *header = b.builder.createBlock(&body);
  mlir::Block *head = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *printOne = b.builder.createBlock(&body);
  mlir::Block *summary = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value cause = b.loadI64(b.addrOf("g_exc_cause_node"));
  mlir::Value haveCause =
      b.cmpi(mlir::arith::CmpIPredicate::ne, cause, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveCause, causeBlock,
                                 mlir::ValueRange{}, contextCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(causeBlock);
  b.call("print_chain_node", mlir::TypeRange{}, mlir::ValueRange{cause});
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_sep_cause")});
  mlir::cf::BranchOp::create(b.builder, b.loc, header, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(contextCheck);
  mlir::Value context = b.loadI64(b.addrOf("g_exc_context_node"));
  mlir::Value suppress = b.loadI64(b.addrOf("g_exc_suppress_context"));
  mlir::Value haveContext =
      b.cmpi(mlir::arith::CmpIPredicate::ne, context, b.iconst(0));
  mlir::Value showContext = mlir::arith::AndIOp::create(
      b.builder, b.loc, haveContext,
      b.cmpi(mlir::arith::CmpIPredicate::eq, suppress, b.iconst(0)));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, showContext, contextBlock,
                                 mlir::ValueRange{}, header,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(contextBlock);
  b.call("print_chain_node", mlir::TypeRange{}, mlir::ValueRange{context});
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_sep_context")});
  mlir::cf::BranchOp::create(b.builder, b.loc, header, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(header);
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_header")});
  mlir::cf::BranchOp::create(b.builder, b.loc, head,
                             mlir::ValueRange{loadTracebackSize(b)});

  b.builder.setInsertionPointToEnd(head);
  mlir::Value remaining = head->getArgument(0);
  mlir::Value doneFrames =
      b.cmpi(mlir::arith::CmpIPredicate::eq, remaining, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, doneFrames, summary,
                                 mlir::ValueRange{}, printOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(printOne);
  mlir::Value top =
      mlir::arith::SubIOp::create(b.builder, b.loc, remaining, b.iconst(1));
  mlir::Value frame =
      b.call("frame_at", b.ptr(), mlir::ValueRange{top}).front();
  b.call("print_trace_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{top});

  b.builder.setInsertionPointToEnd(summary);
  b.call("print_exception_summary", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
                          entry->getArgument(2), entry->getArgument(3),
                          entry->getArgument(4), entry->getArgument(5)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ---------------------------------------------------------------------------
// Exception-handling core: the Itanium C++ ABI bridge (LyPythonException as a
// 1-byte C++ exception carrying its payload in process globals), the current
// exception slot, and the program entry (LyRunPythonMain). Irreducibly llvm
// dialect: personality, invoke/landingpad, __cxa_* and typeinfo globals.
// ---------------------------------------------------------------------------

} // namespace

void buildTracebackSupport(SupportBuilder &b) {
  declareTracebackSupport(b);
  buildCopyCStr(b);
  buildCopyI8MemRef(b);
  buildFrameAt(b);
  buildFreeFrame(b);
  buildTracebackPush(b);
  buildTracebackPushCStringRange(b);
  buildTracebackPushCString(b);
  buildTracebackPop(b);
  buildTracebackClear(b);
  buildReleaseChainNode(b);
  buildReleaseCurrentChain(b);
  buildStashCurrentAsContext(b);
  buildSetCurrentSuppress(b);
  buildSetCurrentCause(b);
  buildReadSourceLine(b);
  buildExceptionClassName(b);
  buildLeadingWhitespace(b);
  buildPrintMarker(b);
  buildPrintTraceFrame(b);
  buildUtf8MessageCStr(b);
  buildPrintExceptionSummary(b);
  buildPrintChainNode(b);
  buildTracebackPrintMessage(b);
}

} // namespace py::runtime_library
