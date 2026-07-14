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
  mlir::Value trimIt = b.orBit(isNewline, isReturn);
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

// void print_marker(ptr line, i32 col, i32 endCol): the CPython-style
// `    ~~~~~^^` underline for the failing range on stderr.
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
  mlir::Block *emit = b.builder.createBlock(&body);
  mlir::Block *padHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *padOne = b.builder.createBlock(&body);
  mlir::Block *markers = b.builder.createBlock(&body);
  mlir::Block *caretsHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *caretsOne = b.builder.createBlock(&body);
  mlir::Block *tildes = b.builder.createBlock(&body);
  mlir::Block *tildesHead = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *tildesOne = b.builder.createBlock(&body);
  mlir::Block *tildesEnd = b.builder.createBlock(&body);
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
                                 mlir::ValueRange{}, emit,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(emit);
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
  mlir::Value markerWidth =
      mlir::arith::SubIOp::create(b.builder, b.loc, markerEnd, start);
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
  mlir::cf::CondBranchOp::create(b.builder, b.loc, padDone, markers,
                                 mlir::ValueRange{}, padOne,
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

  // Width <= 2 renders carets only; wider ranges render tildes with a two
  // caret tail.
  b.builder.setInsertionPointToEnd(markers);
  mlir::Value narrow =
      b.cmpi(mlir::arith::CmpIPredicate::ule, markerWidth, b.iconst(2));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, narrow, caretsHead,
                                 mlir::ValueRange{b.iconst(0)}, tildes,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(caretsHead);
  mlir::Value caretIndex = caretsHead->getArgument(0);
  mlir::Value caretsDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, caretIndex, markerWidth);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, caretsDone, newline,
                                 mlir::ValueRange{}, caretsOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(caretsOne);
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(94)});
  mlir::Value caretNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, caretIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, caretsHead,
                             mlir::ValueRange{caretNext});

  b.builder.setInsertionPointToEnd(tildes);
  mlir::Value tildeCount =
      mlir::arith::SubIOp::create(b.builder, b.loc, markerWidth, b.iconst(2));
  mlir::cf::BranchOp::create(b.builder, b.loc, tildesHead,
                             mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(tildesHead);
  mlir::Value tildeIndex = tildesHead->getArgument(0);
  mlir::Value tildesDone =
      b.cmpi(mlir::arith::CmpIPredicate::eq, tildeIndex, tildeCount);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, tildesDone, tildesEnd,
                                 mlir::ValueRange{}, tildesOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(tildesOne);
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(126)});
  mlir::Value tildeNext =
      mlir::arith::AddIOp::create(b.builder, b.loc, tildeIndex, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, tildesHead,
                             mlir::ValueRange{tildeNext});

  b.builder.setInsertionPointToEnd(tildesEnd);
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(94)});
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.iconst8(94)});
  mlir::cf::BranchOp::create(b.builder, b.loc, newline, mlir::ValueRange{});

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

// void print_exception_summary(i64 class_id, message view): the final
// "Class: message" line (or the class-only / invalid / unknown forms).
void buildPrintExceptionSummary(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_exception_summary",
      b.builder.getFunctionType({b.i64(), b.ptr(), b.i64(), b.i64(), b.i64()},
                                {}),
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
  mlir::Value data = entry->getArgument(1);
  mlir::Value offset = entry->getArgument(2);
  mlir::Value len = entry->getArgument(3);
  mlir::Value stride = entry->getArgument(4);

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
  mlir::Value message =
      b.call("copy_i8_memref", b.ptr(),
             mlir::ValueRange{data, offset, len, stride})
          .front();
  auto formattedMessage = mlir::LLVM::CallOp::create(
      b.builder, b.loc, snprintfType, "snprintf",
      mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_message"),
                       className, message});
  b.call("write_buffered", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), buffer,
                          formattedMessage.getResult()});
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

// LyTraceback_PrintMessage(i64 class_id, ptr unused, message view): header +
// frames (most recent last, printed from the top of the stack downwards) +
// summary line, on stderr.
void buildTracebackPrintMessage(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PrintMessage",
      b.builder.getFunctionType(
          {b.i64(), b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *head = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *printOne = b.builder.createBlock(&body);
  mlir::Block *summary = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
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
         mlir::ValueRange{entry->getArgument(0), entry->getArgument(2),
                          entry->getArgument(3), entry->getArgument(4),
                          entry->getArgument(5)});
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
  buildReadSourceLine(b);
  buildExceptionClassName(b);
  buildLeadingWhitespace(b);
  buildPrintMarker(b);
  buildPrintTraceFrame(b);
  buildPrintExceptionSummary(b);
  buildTracebackPrintMessage(b);
}

} // namespace py::runtime_library
