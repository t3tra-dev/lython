#include "Common/SupportBuilder.h"
#include "ExceptionTaxonomy.h"
#include "Runtime/ABI/BoxLayout.h"

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
    // Exception-chaining state: the in-flight exception's __context__ and
    // __cause__ nodes (null = none), and its __suppress_context__ flag.
    //
    // The two node slots hold POINTERS. They are owner sites in the model's
    // sense -- a global holding an owning reference -- and what a site holds is
    // an identity, not a number that happens to be one.
    for (llvm::StringRef name : {"g_exc_context_node", "g_exc_cause_node"}) {
      auto global = mlir::LLVM::GlobalOp::create(
          b.builder, b.loc, b.ptr(), /*isConstant=*/false,
          mlir::LLVM::Linkage::Internal, name, mlir::Attribute(),
          /*alignment=*/8);
      mlir::OpBuilder::InsertionGuard initGuard(b.builder);
      mlir::Block *init = b.builder.createBlock(&global.getInitializerRegion());
      b.builder.setInsertionPointToEnd(init);
      mlir::LLVM::ReturnOp::create(
          b.builder, b.loc,
          mlir::ValueRange{
              mlir::LLVM::ZeroOp::create(b.builder, b.loc, b.ptr())});
    }
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::Internal,
                                 "g_exc_suppress_context",
                                 b.builder.getIntegerAttr(b.i64(), 0),
                                 /*alignment=*/8);
    // ExceptionGroup display margin: every stderr line the printers start
    // while this is >= 0 gets that many spaces plus "| " (CPython's group
    // traceback gutter). -1 = plain display.
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::Internal,
                                 "g_tb_prefix_spaces",
                                 b.builder.getIntegerAttr(b.i64(), -1),
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
  b.stringGlobal(".tb_group_header",
                 "  + Exception Group Traceback (most recent call last):\n");
  b.stringGlobal(".tb_bar", "| ");
  b.stringGlobal(".tb_fmt_sep_first",
                 "+-+---------------- %lld ----------------\n");
  b.stringGlobal(".tb_fmt_sep_next",
                 "+---------------- %lld ----------------\n");
  b.stringGlobal(".tb_sep_close", "+------------------------------------\n");
  b.stringGlobal(".tb_fmt_group_one", " (%lld sub-exception)\n");
  b.stringGlobal(".tb_fmt_group_many", " (%lld sub-exceptions)\n");
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
// groups, i32 line, i32 col, i32 endLine, i32 endCol, i32 hasMarker): pushes
// a frame with copied names and an optional caret range.
void buildTracebackPush(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_Push",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64(),
                                 b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64(),
                                 b.i32(), b.i32(), b.i32(), b.i32(), b.i32()},
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
  emitFramePush(b, size, fileCopy, functionCopy,
                {entry->getArgument(10), entry->getArgument(11),
                 entry->getArgument(12), entry->getArgument(13),
                 entry->getArgument(14)});
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
// The layout is `exceptionChainNodeType` (SupportBuilder.h). It used to be 21
// untyped i64 words, which meant the payload's three aligned pointers were
// STORED AS INTEGERS and every reader turned them back into pointers to use
// them -- the direction the memory model refuses.
// ---------------------------------------------------------------------------

constexpr std::int64_t kFrameBytes = 40;

// void release_chain_node(ptr node): drop one reference; at zero, release the
// chained nodes, the traceback snapshot, and the exception payload.
void buildReleaseChainNode(SupportBuilder &b) {
  auto fn = b.beginFunction("release_chain_node",
                            b.builder.getFunctionType({b.ptr()}, {}),
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
  mlir::Value node = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value isNull = b.ptrEq(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, alive,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(alive);
  mlir::Value refcount = b.loadI64(nodeMember(b, node, kNodeRefcount));
  mlir::Value decremented =
      mlir::arith::SubIOp::create(b.builder, b.loc, refcount, b.iconst(1));
  mlir::Value stillShared =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, decremented, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, stillShared, decOnly,
                                 mlir::ValueRange{}, destroy,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(decOnly);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, decremented,
                              nodeMember(b, node, kNodeRefcount),
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(destroy);
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(nodeMember(b, node, kNodeCause))});
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(nodeMember(b, node, kNodeContext))});
  mlir::cf::BranchOp::create(b.builder, b.loc, freeHead,
                             mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(freeHead);
  mlir::Value index = freeHead->getArgument(0);
  mlir::Value count = b.loadI64(nodeMember(b, node, kNodeFrameCount));
  mlir::Value framesDone =
      b.cmpi(mlir::arith::CmpIPredicate::sge, index, count);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, framesDone, freeDone,
                                 mlir::ValueRange{}, freeOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freeOne);
  mlir::Value framesPtr = b.loadPtrVal(nodeMember(b, node, kNodeFrames));
  mlir::Value frame = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), tracebackFrameType(b), framesPtr,
      mlir::ValueRange{index});
  b.call("free_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::Value next =
      mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, freeHead,
                             mlir::ValueRange{next});

  b.builder.setInsertionPointToEnd(freeDone);
  b.call("free", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(nodeMember(b, node, kNodeFrames))});
  mlir::Value header = b.loadPtrVal(nodePartsField(b, node, 0, 1));
  mlir::Value becameZero = b.call("release_storage_raw_to_zero", b.i1(),
                                  mlir::ValueRange{header})
                               .front();
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero, freePayload,
                                 mlir::ValueRange{}, freeNode,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freePayload);
  b.call("release_exception_extras", mlir::TypeRange{},
         mlir::ValueRange{header});
  b.call("release_unicode_raw", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(nodePartsField(b, node, 1, 1)),
                          b.loadPtrVal(nodePartsField(b, node, 2, 1))});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{header});
  mlir::cf::BranchOp::create(b.builder, b.loc, freeNode, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freeNode);
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{node});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void release_taken_exception(ptr header, ptr msgHeader, ptr msgData): drop
// the reference `LyEH_TakeCurrentDescriptor` handed to its caller.
//
// "Take" is a transfer: the runtime's in-flight slot gives up its reference and
// the caller owns one. `LyRunPythonMain` is the only caller, and it printed the
// traceback and returned without ever discharging it -- so every program that
// ends by unwinding out of `__main__`, and every `sys.exit()`, lost the
// exception object (56 B) and, when it held the only reference, its message.
//
// Bounded and terminal, but a lost reference at a transfer boundary all the
// same, and one nothing could observe: the leak gate requires a subject to exit
// 0 on its own, so every program in this class was outside what it measures.
//
// The body is `release_chain_node`'s freePayload path, which owns the same kind
// of reference and already spells the sequence. It is a DECREMENT first
// (`release_storage_raw_to_zero`) and frees only at zero, so a payload the chain
// still shares survives this correctly rather than by anybody's argument that
// sharing cannot happen here.
void buildReleaseTakenException(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "release_taken_exception",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.ptr()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *alive = b.builder.createBlock(&body);
  mlir::Block *freePayload = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value header = entry->getArgument(0);
  mlir::Value msgHeader = entry->getArgument(1);
  mlir::Value msgData = entry->getArgument(2);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value isNull = b.cmpi(mlir::arith::CmpIPredicate::eq,
                              b.ptrToInt(header), b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, alive,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(alive);
  mlir::Value becameZero = b.call("release_storage_raw_to_zero", b.i1(),
                                  mlir::ValueRange{header})
                               .front();
  mlir::cf::CondBranchOp::create(b.builder, b.loc, becameZero, freePayload,
                                 mlir::ValueRange{}, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freePayload);
  b.call("release_exception_extras", mlir::TypeRange{},
         mlir::ValueRange{header});
  b.call("release_unicode_raw", mlir::TypeRange{},
         mlir::ValueRange{msgHeader, msgData});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{header});
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
  for (llvm::StringRef name : {"g_exc_cause_node", "g_exc_context_node"}) {
    mlir::Value slot = b.addrOf(name);
    b.call("release_chain_node", mlir::TypeRange{},
           mlir::ValueRange{b.loadPtrVal(slot)});
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), slot,
                                /*alignment=*/8);
  }
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
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
      b.call("malloc", b.ptr(), mlir::ValueRange{typeSizeBytes(b, exceptionChainNodeType(b))})
          .front();
  mlir::Value allocFailed = b.ptrEq(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, allocFailed, trap,
                                 mlir::ValueRange{}, snap, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(snap);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(1),
                              nodeMember(b, node, kNodeRefcount),
                              /*alignment=*/8);
  storeExceptionParts(b, nodeMember(b, node, kNodePayload),
                      loadExceptionParts(b, b.addrOf("g_current_parts")));
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
    mlir::LLVM::StoreOp::create(b.builder, b.loc, buffer,
                                nodeMember(b, node, kNodeFrames),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, size,
                                nodeMember(b, node, kNodeFrameCount),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                b.addrOf("g_traceback_size"), /*alignment=*/8);
    // Both regions keep the builder-synthesized scf.yield terminators.
    b.builder.setInsertionPointToStart(&framesIf.getElseRegion().front());
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(),
                                nodeMember(b, node, kNodeFrames),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                nodeMember(b, node, kNodeFrameCount),
                                /*alignment=*/8);
  }
  mlir::cf::BranchOp::create(b.builder, b.loc, chain, mlir::ValueRange{node});

  b.builder.setInsertionPointToEnd(chain);
  mlir::Value chainNode = chain->getArgument(0);
  for (llvm::StringRef name : {"g_exc_cause_node", "g_exc_context_node"}) {
    mlir::Value global = b.addrOf(name);
    mlir::LLVM::StoreOp::create(
        b.builder, b.loc, b.loadPtrVal(global),
        nodeMember(b, chainNode,
                   name == "g_exc_cause_node" ? kNodeCause : kNodeContext),
        /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), global,
                                /*alignment=*/8);
  }
  {
    mlir::Value global = b.addrOf("g_exc_suppress_context");
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.loadI64(global),
                                nodeMember(b, chainNode, kNodeSuppress),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0), global,
                                /*alignment=*/8);
  }
  mlir::LLVM::StoreOp::create(b.builder, b.loc, chainNode,
                              b.addrOf("g_exc_context_node"), /*alignment=*/8);
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 0, 1).getResult(),
      b.addrOf("g_current_exception"), /*alignment=*/4);
  clearExceptionParts(b, b.addrOf("g_current_parts"));
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
         mlir::ValueRange{b.loadPtrVal(causeSlot)});
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(), causeSlot,
                              /*alignment=*/8);
  mlir::Value contextNode = b.loadPtrVal(b.addrOf("g_exc_context_node"));
  mlir::Value contextMissing = b.ptrEq(contextNode, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, contextMissing, fresh,
                                 mlir::ValueRange{}, compare,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(compare);
  mlir::Value contextHeader =
      b.loadPtrVal(nodePartsField(b, contextNode, 0, 1));
  mlir::Value sameObject =
      b.ptrEq(contextHeader, entry->getArgument(1));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, sameObject, share,
                                 mlir::ValueRange{}, fresh,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(share);
  mlir::Value refcount = b.loadI64(nodeMember(b, contextNode, kNodeRefcount));
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::AddIOp::create(b.builder, b.loc, refcount, b.iconst(1))
          .getResult(),
      nodeMember(b, contextNode, kNodeRefcount), /*alignment=*/8);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, contextNode, causeSlot,
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(fresh);
  mlir::Value node =
      b.call("malloc", b.ptr(), mlir::ValueRange{typeSizeBytes(b, exceptionChainNodeType(b))})
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
                              nodeMember(b, node, kNodeRefcount),
                              /*alignment=*/8);
  for (std::int32_t section = 0; section < 3; ++section)
    for (std::int32_t field = 0; field < 5; ++field)
      mlir::LLVM::StoreOp::create(
          b.builder, b.loc, entry->getArgument(section * 5 + field),
          nodePartsField(b, node, section, field), /*alignment=*/8);
  for (std::int32_t member : {kNodeFrames, kNodeCause, kNodeContext})
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.nullPtr(),
                                nodeMember(b, node, member), /*alignment=*/8);
  for (std::int32_t member : {kNodeFrameCount, kNodeSuppress})
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                nodeMember(b, node, member), /*alignment=*/8);
  b.call("retain_storage_raw", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(1)});
  b.call("retain_storage_raw", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(6)});
  mlir::LLVM::StoreOp::create(b.builder, b.loc, node, causeSlot,
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
  // Source exception classes: the per-program hook owns their names (null
  // for ids it does not know, which keeps the builtin selection).
  auto userName = mlir::func::CallOp::create(
      b.builder, b.loc, "__ly_user_exception_class_name", b.ptr(),
      mlir::ValueRange{classId});
  mlir::Value null = mlir::LLVM::ZeroOp::create(b.builder, b.loc, b.ptr());
  mlir::Value missing = mlir::LLVM::ICmpOp::create(
      b.builder, b.loc, mlir::LLVM::ICmpPredicate::eq, userName.getResult(0),
      null);
  name = mlir::arith::SelectOp::create(b.builder, b.loc, missing, name,
                                       userName.getResult(0));
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

// void print_marker(ptr line, i32 col, i32 endCol, i32 mode): the
// CPython-style anchor underline for the failing range on stderr. Mirrors
// CPython 3.14's display heuristics on the source text alone (the runtime
// has no instruction anchors): a call/subscript range splits at its first
// `(`/`[` into `~~~^^^`, an operator range puts `^` over the operator run,
// and a range with no anchor renders all carets — unless it covers the
// whole line, which CPython suppresses entirely. Mode 2 skips the segment
// heuristics (plain range carets only): CPython extracts anchors only from
// call/binop/subscript nodes, so a yield delivered by throw() must not
// split at a `(` its operand happens to contain.
void buildPrintMarker(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_marker",
      b.builder.getFunctionType({b.ptr(), b.i32(), b.i32(), b.i32()}, {}),
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
  mlir::Value plainRange = b.cmpi(mlir::arith::CmpIPredicate::eq,
                                  entry->getArgument(3), b.iconst32(2));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, plainRange, noAnchor,
                                 mlir::ValueRange{}, splitHead,
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
  b.call("write_prefix", mlir::TypeRange{}, mlir::ValueRange{stderrFd});
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

// void write_spaces(i32 fd, i64 count): margin padding for the group display.
void buildWriteSpaces(SupportBuilder &b) {
  auto fn = b.beginFunction("write_spaces",
                            b.builder.getFunctionType({b.i32(), b.i64()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *head =
      b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *one = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(b.builder, b.loc, head,
                             mlir::ValueRange{b.iconst(0)});
  b.builder.setInsertionPointToEnd(head);
  mlir::Value index = head->getArgument(0);
  mlir::Value finished =
      b.cmpi(mlir::arith::CmpIPredicate::sge, index, entry->getArgument(1));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, finished, done,
                                 mlir::ValueRange{}, one, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(one);
  b.call("write_char", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(0), b.iconst8(32)});
  mlir::Value next =
      mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{next});
  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void write_prefix(i32 fd): the group-display gutter (margin spaces + "| ")
// when g_tb_prefix_spaces >= 0; a no-op for the plain display, so the
// printers call it unconditionally at every line start.
void buildWritePrefix(SupportBuilder &b) {
  auto fn = b.beginFunction("write_prefix",
                            b.builder.getFunctionType({b.i32()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value spaces = b.loadI64(b.addrOf("g_tb_prefix_spaces"));
  mlir::Value active =
      b.cmpi(mlir::arith::CmpIPredicate::sge, spaces, b.iconst(0));
  auto activeIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          active, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&activeIf.getThenRegion().front());
    b.call("write_spaces", mlir::TypeRange{},
           mlir::ValueRange{entry->getArgument(0), spaces});
    b.call("write_cstr", mlir::TypeRange{},
           mlir::ValueRange{entry->getArgument(0), b.addrOf(".tb_bar")});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// i64 exception_group_member_count(ptr exc): the member-block count of an
// exception's extended word 3 (0 for a plain exception or a null pointer).
//
// The class check lives HERE and not in the callers: extended word 3 is the
// one payload block, shared by a group's members and by a plain exception's
// multi-value args, so "the block is non-empty" alone is not "this is a
// group". Two of the three callers already paired the count with
// LyEH_ClassIdMatches; the nested-member recursion did not, and a
// `KeyError('k')` inside a group -- which carries its key as one payload arg
// -- opened a "+-+--- 1 ---" section under itself.
void buildExceptionGroupMemberCount(SupportBuilder &b) {
  auto fn = b.beginFunction("exception_group_member_count",
                            b.builder.getFunctionType({b.ptr()}, {b.i64()}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *classCheck = b.builder.createBlock(&body);
  mlir::Block *load = b.builder.createBlock(&body);
  mlir::Block *blockLoad = b.builder.createBlock(&body);
  mlir::Block *zero = b.builder.createBlock(&body);
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value excNull = b.cmpi(mlir::arith::CmpIPredicate::eq,
                               b.ptrToInt(entry->getArgument(0)), b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, excNull, zero,
                                 mlir::ValueRange{}, classCheck,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(classCheck);
  mlir::Value classId =
      b.loadI64(b.gepI64(entry->getArgument(0), b.iconst(2)));
  mlir::Value isGroup = b.call("LyEH_ClassIdMatches", b.i1(),
                               mlir::ValueRange{classId, b.iconst(101)})
                            .front();
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isGroup, load,
                                 mlir::ValueRange{}, zero, mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(load);
  mlir::Value blockWord =
      b.loadI64(b.gepI64(entry->getArgument(0), b.iconst(3)));
  mlir::Value blockNull =
      b.cmpi(mlir::arith::CmpIPredicate::eq, blockWord, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, blockNull, zero,
                                 mlir::ValueRange{}, blockLoad,
                                 mlir::ValueRange{});
  b.builder.setInsertionPointToEnd(blockLoad);
  // Bit 62 of the count word is the tuple-repr flag, not part of the count.
  mlir::Value raw = b.loadI64(b.intToPtr(blockWord));
  mlir::Value count = mlir::arith::AndIOp::create(
      b.builder, b.loc, raw, b.iconst(0x3FFFFFFFFFFFFFFFLL));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{count});
  b.builder.setInsertionPointToEnd(zero);
  mlir::func::ReturnOp::create(b.builder, b.loc,
                               mlir::ValueRange{b.iconst(0)});
}

// void print_group_members(ptr exc, i64 margin): CPython's numbered member
// sections for an exception group, recursing into nested groups two spaces
// deeper. The member summaries reuse print_exception_summary through the
// prefix gutter.
void buildPrintGroupMembers(SupportBuilder &b) {
  auto fn = b.beginFunction("print_group_members",
                            b.builder.getFunctionType({b.ptr(), b.i64()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *walk = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value margin = entry->getArgument(1);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value count = b.call("exception_group_member_count", b.i64(),
                             mlir::ValueRange{entry->getArgument(0)})
                          .front();
  mlir::Value none = b.cmpi(mlir::arith::CmpIPredicate::sle, count,
                            b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, none, done,
                                 mlir::ValueRange{}, walk, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(walk);
  mlir::Value stderrFd = b.iconst32(2);
  mlir::Value childMargin =
      mlir::arith::AddIOp::create(b.builder, b.loc, margin, b.iconst(2));
  mlir::Value blockWord =
      b.loadI64(b.gepI64(entry->getArgument(0), b.iconst(3)));
  mlir::Value blockPtr = b.intToPtr(blockWord);
  auto bufferType = mlir::LLVM::LLVMArrayType::get(b.i8(), 128);
  mlir::Value bufferSlot = mlir::LLVM::AllocaOp::create(
      b.builder, b.loc, b.ptr(), bufferType, b.iconst32(1), /*alignment=*/1);
  mlir::Value buffer = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), bufferType, bufferSlot,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(0)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
  auto snprintfType = mlir::LLVM::LLVMFunctionType::get(
      b.i32(), {b.ptr(), b.i64(), b.ptr()}, /*isVarArg=*/true);
  mlir::Value zeroIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
  mlir::Value oneIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
  mlir::Value countIndex = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.builder.getIndexType(), count);
  auto loop = mlir::scf::ForOp::create(b.builder, b.loc, zeroIndex, countIndex,
                                       oneIndex);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(loop.getBody());
    mlir::Value position = mlir::arith::IndexCastOp::create(
        b.builder, b.loc, b.i64(), loop.getInductionVar());
    mlir::Value ordinal =
        mlir::arith::AddIOp::create(b.builder, b.loc, position, b.iconst(1));
    mlir::Value boxWords = mlir::arith::MulIOp::create(
        b.builder, b.loc, position,
        b.iconst(py::lowering::box_abi::kWordsPerBox));
    mlir::Value boxBase =
        mlir::arith::AddIOp::create(b.builder, b.loc, boxWords, b.iconst(1));
    mlir::Value boxPtr = b.gepI64(blockPtr, boxBase);
    mlir::Value ehWord = b.loadI64(
        b.gepI64(boxPtr, b.iconst(py::lowering::box_abi::kEntityWord)));
    // ⛔ Widened here and not carried in as a pointer: a member slot is a box
    // word, so this is the boundary the box layout imposes rather than a
    // pointer being thrown away and rebuilt (BoxLayout.cpp records why a box
    // cannot hold one).
    mlir::Value ehPtr = b.intToPtr(ehWord);
    // The message lanes come from the exception rather than from the box: the
    // box holds the entity and `__ly_exc_lane_words` reads the rest out of it,
    // which is the same answer with one copy instead of two.
    //
    // Declared rather than duplicated: this module is lowered on its own and
    // linked to the manifest afterwards, and the alternative is a second copy
    // of the exception's word layout in a file that already carries a copy of
    // the str shape word.
    b.declareExternal("__ly_exc_lane_words",
                      b.builder.getFunctionType(
                          {b.i64()}, {b.i64(), b.i64(), b.i64(), b.i64()}));
    mlir::ValueRange messageLanes =
        b.call("__ly_exc_lane_words",
               mlir::TypeRange{b.i64(), b.i64(), b.i64(), b.i64()},
               mlir::ValueRange{ehWord});
    mlir::Value mhWord = messageLanes[0];
    mlir::Value mbWord = messageLanes[2];
    mlir::Value mbLen = messageLanes[3];
    mlir::Value classId =
        b.loadI64(b.gepI64(ehPtr, b.iconst(2)));

    // Separator: the first section carries the parent connector ("+-+" at
    // the parent margin), later sections sit at the member margin.
    mlir::Value isFirst =
        b.cmpi(mlir::arith::CmpIPredicate::eq, position, b.iconst(0));
    auto sepIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                         isFirst, /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard sepGuard(b.builder);
      b.builder.setInsertionPointToStart(&sepIf.getThenRegion().front());
      b.call("write_spaces", mlir::TypeRange{},
             mlir::ValueRange{stderrFd, margin});
      auto formatted = mlir::LLVM::CallOp::create(
          b.builder, b.loc, snprintfType, "snprintf",
          mlir::ValueRange{buffer, b.iconst(128),
                           b.addrOf(".tb_fmt_sep_first"), ordinal});
      b.call("write_buffered", mlir::TypeRange{},
             mlir::ValueRange{stderrFd, buffer, formatted.getResult()});
      b.builder.setInsertionPointToStart(&sepIf.getElseRegion().front());
      b.call("write_spaces", mlir::TypeRange{},
             mlir::ValueRange{stderrFd, childMargin});
      auto formattedNext = mlir::LLVM::CallOp::create(
          b.builder, b.loc, snprintfType, "snprintf",
          mlir::ValueRange{buffer, b.iconst(128),
                           b.addrOf(".tb_fmt_sep_next"), ordinal});
      b.call("write_buffered", mlir::TypeRange{},
             mlir::ValueRange{stderrFd, buffer, formattedNext.getResult()});
    }

    // Member summary behind the gutter, then nested members.
    mlir::Value prefixSlot = b.addrOf("g_tb_prefix_spaces");
    mlir::LLVM::StoreOp::create(b.builder, b.loc, childMargin, prefixSlot,
                                /*alignment=*/8);
    b.call("print_exception_summary", mlir::TypeRange{},
           mlir::ValueRange{classId, ehPtr, b.intToPtr(mhWord),
                            b.intToPtr(mbWord), b.iconst(0), mbLen,
                            b.iconst(1)});
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(-1), prefixSlot,
                                /*alignment=*/8);
    mlir::Value memberCount = b.call("exception_group_member_count", b.i64(),
                                     mlir::ValueRange{ehPtr})
                                  .front();
    mlir::Value nested =
        b.cmpi(mlir::arith::CmpIPredicate::sgt, memberCount, b.iconst(0));
    auto nestedIf = mlir::scf::IfOp::create(b.builder, b.loc,
                                            mlir::TypeRange{}, nested,
                                            /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard nestedGuard(b.builder);
      b.builder.setInsertionPointToStart(&nestedIf.getThenRegion().front());
      b.call("print_group_members", mlir::TypeRange{},
             mlir::ValueRange{ehPtr, childMargin});
    }
  }
  b.call("write_spaces", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, childMargin});
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, b.addrOf(".tb_sep_close")});
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
  b.call("write_prefix", mlir::TypeRange{}, mlir::ValueRange{b.iconst32(2)});
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
  b.call("write_prefix", mlir::TypeRange{}, mlir::ValueRange{b.iconst32(2)});
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
         mlir::ValueRange{trimmed, colAdjusted, endAdjusted, hasMarker});
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
    // The shape word packs the byte count above the width's three bits
    // (`__ly_unicode_alloc`); reading it whole compared a length against 2 and
    // 4, fell through to latin-1, and printed a UCS-2 message as its first
    // code unit and a NUL.
    mlir::Value shape = b.loadI64(b.gepI8(header, b.iconst(16)));
    mlir::Value stored = mlir::arith::AndIOp::create(b.builder, b.loc, shape,
                                                     b.iconst(7))
                             .getResult();
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

// void print_exception_summary(i64 class_id, ptr exc, ptr msg_header,
// message view): the final "Class: message" line (or the class-only /
// invalid / unknown forms). The message is re-encoded from code units to
// UTF-8 for display. An exception group appends CPython's
// " (N sub-exception[s])" count read through `exc` (null = no payload known).
void buildPrintExceptionSummary(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "print_exception_summary",
      b.builder.getFunctionType(
          {b.i64(), b.ptr(), b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()},
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
  mlir::Value excPtr = entry->getArgument(1);
  mlir::Value msgHeader = entry->getArgument(2);
  mlir::Value data = entry->getArgument(3);
  mlir::Value offset = entry->getArgument(4);
  mlir::Value len = entry->getArgument(5);
  mlir::Value stride = entry->getArgument(6);

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
  mlir::Value memberCount = b.call("exception_group_member_count", b.i64(),
                                   mlir::ValueRange{excPtr})
                                .front();
  mlir::Value isGroupClass = b.call("LyEH_ClassIdMatches", b.i1(),
                                    mlir::ValueRange{classId, b.iconst(101)})
                                 .front();
  mlir::Value groupSuffix = mlir::arith::AndIOp::create(
      b.builder, b.loc, isGroupClass,
      b.cmpi(mlir::arith::CmpIPredicate::sgt, memberCount, b.iconst(0)));
  auto snprintfType = mlir::LLVM::LLVMFunctionType::get(
      b.i32(), {b.ptr(), b.i64(), b.ptr()}, /*isVarArg=*/true);
  auto emitBuffered = [&](mlir::Value formattedLength) {
    b.call("write_prefix", mlir::TypeRange{}, mlir::ValueRange{b.iconst32(2)});
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
  b.call("write_prefix", mlir::TypeRange{}, mlir::ValueRange{stderrFd});
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, className});
  b.stringGlobal(".tb_colon_space", ": ");
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{stderrFd, b.addrOf(".tb_colon_space")});
  // KeyError stores repr(key) as its message (LyKeyError_Init and the
  // runtime missing-key raises), so the display prints it verbatim.
  b.call("write_cstr", mlir::TypeRange{}, mlir::ValueRange{stderrFd, message});
  auto suffixIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          groupSuffix,
                                          /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&suffixIf.getThenRegion().front());
    mlir::Value plural = b.cmpi(mlir::arith::CmpIPredicate::sgt, memberCount,
                                b.iconst(1));
    mlir::Value format = mlir::LLVM::SelectOp::create(
        b.builder, b.loc, plural, b.addrOf(".tb_fmt_group_many"),
        b.addrOf(".tb_fmt_group_one"));
    auto formatted = mlir::LLVM::CallOp::create(
        b.builder, b.loc, snprintfType, "snprintf",
        mlir::ValueRange{buffer, b.iconst(1024), format, memberCount});
    b.call("write_buffered", mlir::TypeRange{},
           mlir::ValueRange{stderrFd, buffer, formatted.getResult()});
    b.builder.setInsertionPointToStart(&suffixIf.getElseRegion().front());
    b.call("write_len", mlir::TypeRange{},
           mlir::ValueRange{stderrFd, b.addrOf(".tb_newline"), b.iconst(1)});
  }
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{message});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(classOnly);
  // A message-less exception group still shows the count: CPython renders
  // "ExceptionGroup:  (N sub-exceptions)" (str() of the group is the empty
  // message plus the count suffix).
  auto classOnlySuffix = mlir::scf::IfOp::create(
      b.builder, b.loc, mlir::TypeRange{}, groupSuffix,
      /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&classOnlySuffix.getThenRegion().front());
    b.call("write_prefix", mlir::TypeRange{}, mlir::ValueRange{b.iconst32(2)});
    b.call("write_cstr", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), className});
    b.call("write_cstr", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_colon_space")});
    mlir::Value plural = b.cmpi(mlir::arith::CmpIPredicate::sgt, memberCount,
                                b.iconst(1));
    mlir::Value format = mlir::LLVM::SelectOp::create(
        b.builder, b.loc, plural, b.addrOf(".tb_fmt_group_many"),
        b.addrOf(".tb_fmt_group_one"));
    auto formatted = mlir::LLVM::CallOp::create(
        b.builder, b.loc, snprintfType, "snprintf",
        mlir::ValueRange{buffer, b.iconst(1024), format, memberCount});
    b.call("write_buffered", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), buffer, formatted.getResult()});
    b.builder.setInsertionPointToStart(&classOnlySuffix.getElseRegion().front());
    b.call("write_prefix", mlir::TypeRange{}, mlir::ValueRange{b.iconst32(2)});
    auto formattedClass = mlir::LLVM::CallOp::create(
        b.builder, b.loc, snprintfType, "snprintf",
        mlir::ValueRange{buffer, b.iconst(1024), b.addrOf(".tb_fmt_class"),
                         className});
    b.call("write_buffered", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), buffer,
                            formattedClass.getResult()});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});

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

// void print_chain_node(ptr node): one section of a chained-exception report:
// the node's own chain first (recursively), the matching separator, then its
// traceback (when captured) and summary line.
void buildPrintChainNode(SupportBuilder &b) {
  auto fn = b.beginFunction("print_chain_node",
                            b.builder.getFunctionType({b.ptr()}, {}),
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
  mlir::Value node = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value missing = b.ptrEq(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, missing, done,
                                 mlir::ValueRange{}, present,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(present);
  mlir::Value cause = b.loadPtrVal(nodeMember(b, node, kNodeCause));
  mlir::Value haveCause = b.ptrNe(cause, b.nullPtr());
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
  mlir::Value context = b.loadPtrVal(nodeMember(b, node, kNodeContext));
  mlir::Value suppress = b.loadI64(nodeMember(b, node, kNodeSuppress));
  mlir::Value haveContext = b.ptrNe(context, b.nullPtr());
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
  mlir::Value count = b.loadI64(nodeMember(b, node, kNodeFrameCount));
  mlir::Value aligned = b.loadPtrVal(nodePartsField(b, node, 0, 1));
  mlir::Value offset0 = b.loadI64(nodePartsField(b, node, 0, 2));
  mlir::Value stride0 = b.loadI64(nodePartsField(b, node, 0, 4));
  mlir::Value classIndex = mlir::arith::AddIOp::create(
      b.builder, b.loc, offset0,
      mlir::arith::MulIOp::create(b.builder, b.loc, stride0, b.iconst(2))
          .getResult());
  mlir::Value classId = b.loadI64(b.gepI64(aligned, classIndex));
  // Chained exception groups keep CPython's group rendering: the group
  // header (when a traceback exists), the "| " gutter, and the member tree.
  mlir::Value chainMembers = b.call("exception_group_member_count", b.i64(),
                                    mlir::ValueRange{aligned})
                                 .front();
  mlir::Value chainIsGroupClass =
      b.call("LyEH_ClassIdMatches", b.i1(),
             mlir::ValueRange{classId, b.iconst(101)})
          .front();
  mlir::Value chainGroup = mlir::arith::AndIOp::create(
      b.builder, b.loc, chainIsGroupClass,
      b.cmpi(mlir::arith::CmpIPredicate::sgt, chainMembers, b.iconst(0)));
  auto gutterIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          chainGroup,
                                          /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&gutterIf.getThenRegion().front());
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(2),
                                b.addrOf("g_tb_prefix_spaces"),
                                /*alignment=*/8);
  }
  mlir::Value haveFrames =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, count, b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveFrames, withHeader,
                                 mlir::ValueRange{}, summary,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(withHeader);
  auto headerIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                          chainGroup,
                                          /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&headerIf.getThenRegion().front());
    b.call("write_cstr", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_group_header")});
    b.builder.setInsertionPointToStart(&headerIf.getElseRegion().front());
    b.call("write_cstr", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_header")});
  }
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
  mlir::Value frames = b.loadPtrVal(nodeMember(b, node, kNodeFrames));
  mlir::Value frame = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), tracebackFrameType(b), frames,
      mlir::ValueRange{top});
  b.call("print_trace_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::cf::BranchOp::create(b.builder, b.loc, frameHead,
                             mlir::ValueRange{top});

  b.builder.setInsertionPointToEnd(summary);
  mlir::Value msgHeader = b.loadPtrVal(nodePartsField(b, node, 1, 1));
  mlir::Value msgData = b.loadPtrVal(nodePartsField(b, node, 2, 1));
  mlir::Value msgOffset = b.loadI64(nodePartsField(b, node, 2, 2));
  mlir::Value msgLen = b.loadI64(nodePartsField(b, node, 2, 3));
  mlir::Value msgStride = b.loadI64(nodePartsField(b, node, 2, 4));
  b.call("print_exception_summary", mlir::TypeRange{},
         mlir::ValueRange{classId, aligned, msgHeader, msgData, msgOffset,
                          msgLen, msgStride});
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(-1),
                              b.addrOf("g_tb_prefix_spaces"),
                              /*alignment=*/8);
  auto membersIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                           chainGroup,
                                           /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&membersIf.getThenRegion().front());
    b.call("print_group_members", mlir::TypeRange{},
           mlir::ValueRange{aligned, b.iconst(2)});
  }
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// ---------------------------------------------------------------------------
// except* star frames (PEP 654). A star frame parks the caught exception as
// a chain node (the stash machinery already bundles payload + traceback +
// chain into one node) and owns the unmatched residual between clauses plus
// every exception a clause body raised.
//
// The layout is `starFrameType` (SupportBuilder.h), which also says what the
// removed `parent` word was for.
// ---------------------------------------------------------------------------

// The frame is the first argument of every star function now. It used to be a
// mutable global holding a raw pointer, with slot 3 linking the frames into a
// stack -- neither of which the memory model can express, because a descriptor
// cannot be stored and an identity laundered through an integer is outside it
// by the model's own statement. As a parameter the nesting is lexical, and
// `except_star.begin` hands the value to the ops that need it.
mlir::Value starFrameArg(mlir::func::FuncOp fn) {
  return fn.getBody().front().getArgument(0);
}

mlir::Value starFrameSlot(SupportBuilder &b, mlir::Value frame,
                          std::int32_t member) {
  return starFrameMember(b, frame, member);
}

// void release_exception_storage_raw(ptr eh, ptr mh, ptr mb): one owned
// exception reference by raw pointers (the star frame paths hold exceptions
// outside memref descriptors).
void buildReleaseExceptionStorageRaw(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "release_exception_storage_raw",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.ptr()}, {}),
      /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value becameZero =
      b.call("release_storage_raw_to_zero", b.i1(),
             mlir::ValueRange{entry->getArgument(0)})
          .front();
  auto freeIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                        becameZero, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&freeIf.getThenRegion().front());
    b.call("release_exception_extras", mlir::TypeRange{},
           mlir::ValueRange{entry->getArgument(0)});
    b.call("release_unicode_raw", mlir::TypeRange{},
           mlir::ValueRange{entry->getArgument(1), entry->getArgument(2)});
    b.call("free", mlir::TypeRange{},
           mlir::ValueRange{entry->getArgument(0)});
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void release_star_node(ptr node, i64 hasPayload): drop a star frame's
// residual node. With a live payload this is a plain chain-node release;
// after the payload moved out only the shell (frames, chained nodes) is
// left, and release_chain_node's payload release would touch a stale word.
void buildReleaseStarNode(SupportBuilder &b) {
  auto fn = b.beginFunction("release_star_node",
                            b.builder.getFunctionType({b.ptr(), b.i64()}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *live = b.builder.createBlock(&body);
  mlir::Block *payload = b.builder.createBlock(&body);
  mlir::Block *shell = b.builder.createBlock(&body);
  mlir::Block *freeHead =
      b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *freeOne = b.builder.createBlock(&body);
  mlir::Block *freeDone = b.builder.createBlock(&body);
  mlir::Block *done = b.builder.createBlock(&body);
  mlir::Value node = entry->getArgument(0);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value isNull = b.ptrEq(node, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, isNull, done,
                                 mlir::ValueRange{}, live, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(live);
  mlir::Value hasPayload = b.cmpi(mlir::arith::CmpIPredicate::ne,
                                  entry->getArgument(1), b.iconst(0));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, hasPayload, payload,
                                 mlir::ValueRange{}, shell,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(payload);
  b.call("release_chain_node", mlir::TypeRange{}, mlir::ValueRange{node});
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(shell);
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(nodeMember(b, node, kNodeCause))});
  b.call("release_chain_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(nodeMember(b, node, kNodeContext))});
  mlir::cf::BranchOp::create(b.builder, b.loc, freeHead,
                             mlir::ValueRange{b.iconst(0)});

  b.builder.setInsertionPointToEnd(freeHead);
  mlir::Value index = freeHead->getArgument(0);
  mlir::Value count = b.loadI64(nodeMember(b, node, kNodeFrameCount));
  mlir::Value framesDone =
      b.cmpi(mlir::arith::CmpIPredicate::sge, index, count);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, framesDone, freeDone,
                                 mlir::ValueRange{}, freeOne,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(freeOne);
  mlir::Value framesPtr = b.loadPtrVal(nodeMember(b, node, kNodeFrames));
  mlir::Value frame = mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), tracebackFrameType(b), framesPtr,
      mlir::ValueRange{index});
  b.call("free_frame", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::Value next =
      mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1));
  mlir::cf::BranchOp::create(b.builder, b.loc, freeHead,
                             mlir::ValueRange{next});

  b.builder.setInsertionPointToEnd(freeDone);
  b.call("free", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(nodeMember(b, node, kNodeFrames))});
  freeSoleChainNode(b, node, "release_star_node");
  mlir::cf::BranchOp::create(b.builder, b.loc, done, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(done);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_StarBegin(): push a star frame parking the caught exception.
void buildStarBegin(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarBegin",
                            b.builder.getFunctionType({}, {b.ptr()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame =
      b.call("malloc", b.ptr(),
             mlir::ValueRange{typeSizeBytes(b, starFrameType(b))})
          .front();
  mlir::Value allocFailed = b.ptrEq(frame, b.nullPtr());
  mlir::cf::AssertOp::create(
      b.builder, b.loc,
      mlir::arith::XOrIOp::create(
          b.builder, b.loc, allocFailed,
          mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1)
              .getResult()),
      "except* frame allocation failed");
  mlir::LLVM::MemsetOp::create(b.builder, b.loc, frame, b.iconst8(0),
                               typeSizeBytes(b, starFrameType(b)),
                               /*isVolatile=*/false);
  b.call("LyEH_StashCurrentException", mlir::TypeRange{},
         mlir::ValueRange{starFrameMember(b, frame, kStarResidual)});
  mlir::Value node =
      b.loadPtrVal(starFrameMember(b, frame, kStarResidual));
  mlir::Value present = b.ptrNe(node, b.nullPtr());
  mlir::Value presentWord = mlir::arith::ExtUIOp::create(
      b.builder, b.loc, b.i64(), present);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, presentWord,
                              starFrameMember(b, frame, kStarPresent),
                              /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{frame});
}

// i1 LyEH_StarHasResidual(): whether the innermost frame still holds an
// unmatched slice.
void buildStarHasResidual(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarHasResidual",
                            b.builder.getFunctionType({b.ptr()}, {b.i1()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);
  mlir::Value present =
      b.loadI64(starFrameSlot(b, frame, kStarPresent));
  mlir::Value result =
      b.cmpi(mlir::arith::CmpIPredicate::ne, present, b.iconst(0));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{result});
}

// (memref<3xi64>, memref<2xi64>, memref<?xi8>) LyEH_StarResidualParts(i64
// frame): borrowed views of the residual exception (requires one).
void buildStarResidualParts(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyEH_StarResidualParts",
      b.builder.getFunctionType({b.ptr()}, exceptionTripleTypes(b.builder)));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);
  mlir::Value node = b.loadPtrVal(starFrameSlot(b, frame, kStarResidual));
  llvm::SmallVector<mlir::Type, 3> types = exceptionTripleTypes(b.builder);
  llvm::SmallVector<mlir::Value, 3> results;
  // Sizes come from the contract, not the stored descriptor: the header
  // sections are fixed-shape and the bytes section carries its length.
  std::int64_t staticSizes[3] = {3, 2, -1};
  for (std::int32_t section = 0; section < 3; ++section) {
    mlir::Value pointer = b.loadPtrVal(nodePartsField(b, node, section, 1));
    mlir::Value size = staticSizes[section] < 0
                           ? b.loadI64(nodePartsField(b, node, section, 3))
                           : b.iconst(staticSizes[section]);
    results.push_back(buildMemRef1D(b, types[section],
                                    MemRef1DParts{pointer, pointer, b.iconst(0),
                                                  size, b.iconst(1)}));
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, results);
}

// void LyEH_StarApplyMatch(i64 frame, i1 has_rest, matched triple, rest
// triple): a clause matched. The matched slice becomes the current exception
// (fresh copy of the residual node's traceback, no inherited chain), the rest
// replaces the node's payload, and the frame's old payload reference is
// dropped.
void buildStarApplyMatch(SupportBuilder &b) {
  llvm::SmallVector<mlir::Type, 3> triple = exceptionTripleTypes(b.builder);
  llvm::SmallVector<mlir::Type, 8> inputs{b.ptr(), b.i1()};
  inputs.append(triple.begin(), triple.end());
  inputs.append(triple.begin(), triple.end());
  auto fn = b.beginFunction("LyEH_StarApplyMatch",
                            b.builder.getFunctionType(inputs, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);
  mlir::Value node = b.loadPtrVal(starFrameSlot(b, frame, kStarResidual));
  mlir::Value payload = nodeMember(b, node, kNodePayload);

  // Drop the frame's reference to the old residual payload.
  b.call("release_exception_storage_raw", mlir::TypeRange{},
         mlir::ValueRange{
             b.loadPtrVal(partsField(b, payload, 0, 1)),
             b.loadPtrVal(partsField(b, payload, 1, 1)),
             b.loadPtrVal(partsField(b, payload, 2, 1))});

  // Rest into the node payload; absent rest zeroes it and clears the flag.
  mlir::Value hasRest = entry->getArgument(1);
  auto restIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                        hasRest, /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&restIf.getThenRegion().front());
    storeExceptionTriple(b, payload, entry->getArguments().slice(5, 3));
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(1),
                                starFrameSlot(b, frame, kStarPresent), /*alignment=*/8);
    b.builder.setInsertionPointToStart(&restIf.getElseRegion().front());
    clearExceptionParts(b, payload);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                starFrameSlot(b, frame, kStarPresent), /*alignment=*/8);
  }

  // Matched becomes the current exception.
  storeExceptionTriple(b, b.addrOf("g_current_parts"),
                       entry->getArguments().slice(2, 3));
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      b.addrOf("g_current_exception"), /*alignment=*/4);

  // Fresh copy of the residual node's traceback (name strings duplicated:
  // the handler's discard consumes this copy, later slices reuse the node's).
  mlir::Value count = b.loadI64(nodeMember(b, node, kNodeFrameCount));
  mlir::Value framesPtr = b.loadPtrVal(nodeMember(b, node, kNodeFrames));
  mlir::Value countIndex = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.builder.getIndexType(), count);
  mlir::Value zeroIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
  mlir::Value oneIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
  auto copyLoop = mlir::scf::ForOp::create(b.builder, b.loc, zeroIndex,
                                           countIndex, oneIndex);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(copyLoop.getBody());
    mlir::Value position = mlir::arith::IndexCastOp::create(
        b.builder, b.loc, b.i64(), copyLoop.getInductionVar());
    mlir::Value source = mlir::LLVM::GEPOp::create(
        b.builder, b.loc, b.ptr(), tracebackFrameType(b), framesPtr,
        mlir::ValueRange{position});
    mlir::Value target =
        b.call("frame_at", b.ptr(), mlir::ValueRange{position}).front();
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, target, source,
                                 b.iconst(kFrameBytes), /*isVolatile=*/false);
    mlir::Type frameType = tracebackFrameType(b);
    for (std::int64_t nameField = 0; nameField < 2; ++nameField) {
      mlir::Value slot = b.frameField(frameType, target, nameField);
      mlir::Value copied =
          b.call("copy_cstr", b.ptr(), mlir::ValueRange{b.loadPtrVal(slot)})
              .front();
      mlir::LLVM::StoreOp::create(b.builder, b.loc, copied, slot,
                                  /*alignment=*/8);
    }
  }
  mlir::LLVM::StoreOp::create(b.builder, b.loc, count,
                              b.addrOf("g_traceback_size"), /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_StarCollect(): park the pending exception (raised by a clause
// body) in the frame's collected list.
void buildStarCollect(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarCollect",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);
  mlir::Value count = b.loadI64(starFrameSlot(b, frame, kStarCollected));
  mlir::Value inRange = b.cmpi(mlir::arith::CmpIPredicate::slt, count,
                               b.iconst(kStarClauseLimit));
  mlir::cf::AssertOp::create(b.builder, b.loc, inRange,
                             "except* clause raise limit exceeded");
  b.call("LyEH_StashCurrentException", mlir::TypeRange{},
         mlir::ValueRange{starClauseCell(b, frame, count)});
  mlir::Value next =
      mlir::arith::AddIOp::create(b.builder, b.loc, count, b.iconst(1));
  mlir::LLVM::StoreOp::create(b.builder, b.loc, next,
                              starFrameSlot(b, frame, kStarCollected), /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// i64 LyEH_StarCollectedCount() / i64 LyEH_StarNodesPtr(): finish inputs.
void buildStarCollectedCount(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarCollectedCount",
                            b.builder.getFunctionType({b.ptr()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value count =
      b.loadI64(starFrameSlot(b, starFrameArg(fn), kStarCollected));
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{count});
}

void buildStarNodesPtr(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarNodesPtr",
                            b.builder.getFunctionType({b.ptr()}, {b.ptr()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::func::ReturnOp::create(
      b.builder, b.loc,
      mlir::ValueRange{starFrameSlot(b, starFrameArg(fn), kStarClauses)});
}

// void star_pop_frame(): unlink and free the frame shell (payload handling
// is each caller's business; here the residual node is fully released).
void buildStarPop(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarPop",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);
  b.call("release_star_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(starFrameSlot(b, frame, kStarResidual)),
                          b.loadI64(starFrameSlot(b, frame, kStarPresent))});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{frame});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// The shared "throw the pending token" tail for the star rethrow paths
// (a func-level function: its callers are func-level; __cxa_throw never
// returns, the trailing return only satisfies the verifier).
void buildStarThrowPending(SupportBuilder &b) {
  auto fn = b.beginFunction("star_throw_pending",
                            b.builder.getFunctionType({}, {}),
                            /*isPrivate=*/true);
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  b.call("end_native_catch_if_active", mlir::TypeRange{}, {});
  auto carrier = mlir::LLVM::CallOp::create(
      b.builder, b.loc, mlir::TypeRange{b.ptr()}, "__cxa_allocate_exception",
      mlir::ValueRange{b.iconst(1)});
  mlir::LLVM::CallOp::create(
      b.builder, b.loc, mlir::TypeRange{}, "__cxa_throw",
      mlir::ValueRange{carrier.getResult(),
                       b.addrOf("_ZTI17LyPythonException"), b.nullPtr()});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_StarRethrowResidual(): nothing was collected — the leftover
// slice rethrows with its original traceback and chain.
void buildStarRethrowResidual(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarRethrowResidual",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);
  b.call("LyEH_UnstashException", mlir::TypeRange{},
         mlir::ValueRange{starFrameSlot(b, frame, kStarResidual)});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{frame});
  b.call("star_throw_pending", mlir::TypeRange{}, {});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_StarRethrowSoleCollected(): exactly one clause raise and no
// residual — that exception continues as itself (context intact).
void buildStarRethrowSoleCollected(SupportBuilder &b) {
  auto fn = b.beginFunction("LyEH_StarRethrowSoleCollected",
                            b.builder.getFunctionType({b.ptr()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);
  b.call("LyEH_UnstashException", mlir::TypeRange{},
         mlir::ValueRange{
             starClauseCell(b, frame, b.iconst(0))});
  b.call("release_star_node", mlir::TypeRange{},
         mlir::ValueRange{b.loadPtrVal(starFrameSlot(b, frame, kStarResidual)),
                          b.loadI64(starFrameSlot(b, frame, kStarPresent))});
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{frame});
  b.call("star_throw_pending", mlir::TypeRange{}, {});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_StarThrowCombined(i64 frame, combined triple): several leftovers
// were wrapped into a fresh group (star_combine). The residual node donates its
// traceback and chain to the combined exception; collected nodes hand their
// member references to the group and die.
void buildStarThrowCombined(SupportBuilder &b) {
  llvm::SmallVector<mlir::Type, 3> triple = exceptionTripleTypes(b.builder);
  llvm::SmallVector<mlir::Type, 4> inputs{b.ptr()};
  inputs.append(triple.begin(), triple.end());
  auto fn = b.beginFunction("LyEH_StarThrowCombined",
                            b.builder.getFunctionType(inputs, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value frame = starFrameArg(fn);

  // Collected nodes: the combined group retained each member, so a plain
  // node release moves ownership to the group.
  mlir::Value count = b.loadI64(starFrameSlot(b, frame, kStarCollected));
  mlir::Value countIndex = mlir::arith::IndexCastOp::create(
      b.builder, b.loc, b.builder.getIndexType(), count);
  mlir::Value zeroIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
  mlir::Value oneIndex =
      mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
  auto releaseLoop = mlir::scf::ForOp::create(b.builder, b.loc, zeroIndex,
                                              countIndex, oneIndex);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(releaseLoop.getBody());
    mlir::Value position = mlir::arith::IndexCastOp::create(
        b.builder, b.loc, b.i64(), releaseLoop.getInductionVar());
    b.call("release_chain_node", mlir::TypeRange{},
           mlir::ValueRange{b.loadPtrVal(
               starClauseCell(b, frame, position))});
  }

  // Residual node: its member reference moved into the group; its traceback
  // and chain become the combined exception's, and the shell dies.
  mlir::Value node = b.loadPtrVal(starFrameSlot(b, frame, kStarResidual));
  mlir::Value hasNode = b.ptrNe(node, b.nullPtr());
  auto nodeIf = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{},
                                        hasNode, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&nodeIf.getThenRegion().front());
    mlir::Value hadPayload = b.loadI64(starFrameSlot(b, frame, kStarPresent));
    mlir::Value payloadLive = b.cmpi(mlir::arith::CmpIPredicate::ne,
                                     hadPayload, b.iconst(0));
    auto payloadIf = mlir::scf::IfOp::create(
        b.builder, b.loc, mlir::TypeRange{}, payloadLive,
        /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard payloadGuard(b.builder);
      b.builder.setInsertionPointToStart(&payloadIf.getThenRegion().front());
      b.call("release_exception_storage_raw", mlir::TypeRange{},
             mlir::ValueRange{
                 b.loadPtrVal(nodePartsField(b, node, 0, 1)),
                 b.loadPtrVal(nodePartsField(b, node, 1, 1)),
                 b.loadPtrVal(nodePartsField(b, node, 2, 1))});
    }
    // CPython gives the combined group no traceback of its own (its members
    // carry theirs); drop the residual node's snapshot instead of donating.
    mlir::Value frames = b.loadPtrVal(nodeMember(b, node, kNodeFrames));
    mlir::Value frameCount = b.loadI64(nodeMember(b, node, kNodeFrameCount));
    mlir::Value frameCountIndex = mlir::arith::IndexCastOp::create(
        b.builder, b.loc, b.builder.getIndexType(), frameCount);
    mlir::Value zeroIdx =
        mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 0);
    mlir::Value oneIdx =
        mlir::arith::ConstantIndexOp::create(b.builder, b.loc, 1);
    auto freeFramesLoop = mlir::scf::ForOp::create(b.builder, b.loc, zeroIdx,
                                                   frameCountIndex, oneIdx);
    {
      mlir::OpBuilder::InsertionGuard framesGuard(b.builder);
      b.builder.setInsertionPointToStart(freeFramesLoop.getBody());
      mlir::Value framePosition = mlir::arith::IndexCastOp::create(
          b.builder, b.loc, b.i64(), freeFramesLoop.getInductionVar());
      mlir::Value framePtr = mlir::LLVM::GEPOp::create(
          b.builder, b.loc, b.ptr(), tracebackFrameType(b), frames,
          mlir::ValueRange{framePosition});
      b.call("free_frame", mlir::TypeRange{}, mlir::ValueRange{framePtr});
    }
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{frames});
    mlir::LLVM::StoreOp::create(b.builder, b.loc,
                                b.loadPtrVal(nodeMember(b, node, kNodeCause)),
                                b.addrOf("g_exc_cause_node"),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc,
                                b.loadPtrVal(nodeMember(b, node, kNodeContext)),
                                b.addrOf("g_exc_context_node"),
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc,
                                b.loadI64(nodeMember(b, node, kNodeSuppress)),
                                b.addrOf("g_exc_suppress_context"),
                                /*alignment=*/8);
    freeSoleChainNode(b, node, "LyEH_StarThrowCombined");
  }

  // Install the combined exception and throw.
  storeExceptionTriple(b, b.addrOf("g_current_parts"),
                       entry->getArguments().slice(1, 3));
  mlir::LLVM::StoreOp::create(
      b.builder, b.loc,
      mlir::arith::ConstantIntOp::create(b.builder, b.loc, 1, 1).getResult(),
      b.addrOf("g_current_exception"), /*alignment=*/4);
  b.call("free", mlir::TypeRange{}, mlir::ValueRange{frame});
  // `star_throw_pending` does not return; the trailing return only satisfies
  // the verifier, which is why this is not `llvm.unreachable`.
  mlir::func::CallOp::create(b.builder, b.loc, "star_throw_pending",
                             mlir::TypeRange{}, mlir::ValueRange{});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyEH_StarDiscardSplit(triple): release a half that star_split handed
// back and the clause did not install.
//
// star_split owns BOTH halves it returns -- the matching leaf and the leftover
// are each retained before they are yielded, because the assembler that takes
// them (LyEH_StarApplyMatch for the matched half, the residual group builder
// for the rest) consumes a reference. When NOTHING matched there is no
// assembler: the clause skips LyEH_StarApplyMatch entirely and both halves
// reach nobody. The leftover half is then a reference the frame's own residual
// already accounts for, so it is dropped here rather than installed.
//
// Why NOT stop retaining in the no-match arm instead: the same arm of
// `__ly_exc_star_split_rec` runs for a non-matching MEMBER of a group, whose
// reference the enclosing residual group does take. One arm, two callers, and
// only the caller knows which it is.
void buildStarDiscardSplit(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyEH_StarDiscardSplit",
      b.builder.getFunctionType(exceptionTripleTypes(b.builder), {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  auto aligned = [&](int section) {
    return explodeMemRef1D(b, entry->getArgument(section)).aligned;
  };
  b.call("release_exception_storage_raw", mlir::TypeRange{},
         mlir::ValueRange{aligned(0), aligned(1), aligned(2)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// LyTraceback_PrintMessage(i64 class_id, ptr exc, ptr msg_header,
// message view): chained sections (cause/context, innermost first) + header
// + frames (most recent last, printed from the top of the stack downwards)
// + summary line, on stderr. An exception group renders CPython's group
// traceback instead: the "+ Exception Group Traceback" header, frames and
// summary behind the "| " gutter, then the numbered member tree.
void buildTracebackPrintMessage(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyTraceback_PrintMessage",
      b.builder.getFunctionType(
          {b.i64(), b.ptr(), b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()},
          {}));
  mlir::Block *entry = fn.addEntryBlock();
  mlir::Region &body = fn.getBody();
  mlir::Block *causeBlock = b.builder.createBlock(&body);
  mlir::Block *contextCheck = b.builder.createBlock(&body);
  mlir::Block *contextBlock = b.builder.createBlock(&body);
  mlir::Block *header = b.builder.createBlock(&body);
  mlir::Block *plainHeader = b.builder.createBlock(&body);
  mlir::Block *groupHeader = b.builder.createBlock(&body);
  mlir::Block *head = b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
  mlir::Block *printOne = b.builder.createBlock(&body);
  mlir::Block *summary = b.builder.createBlock(&body);
  mlir::Block *groupMembers = b.builder.createBlock(&body);
  mlir::Block *finish = b.builder.createBlock(&body);

  b.builder.setInsertionPointToEnd(entry);
  mlir::Value cause = b.loadPtrVal(b.addrOf("g_exc_cause_node"));
  mlir::Value haveCause = b.ptrNe(cause, b.nullPtr());
  mlir::cf::CondBranchOp::create(b.builder, b.loc, haveCause, causeBlock,
                                 mlir::ValueRange{}, contextCheck,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(causeBlock);
  b.call("print_chain_node", mlir::TypeRange{}, mlir::ValueRange{cause});
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_sep_cause")});
  mlir::cf::BranchOp::create(b.builder, b.loc, header, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(contextCheck);
  mlir::Value context = b.loadPtrVal(b.addrOf("g_exc_context_node"));
  mlir::Value suppress = b.loadI64(b.addrOf("g_exc_suppress_context"));
  mlir::Value haveContext = b.ptrNe(context, b.nullPtr());
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
  mlir::Value memberCount = b.call("exception_group_member_count", b.i64(),
                                   mlir::ValueRange{entry->getArgument(1)})
                                .front();
  mlir::Value isGroupClass =
      b.call("LyEH_ClassIdMatches", b.i1(),
             mlir::ValueRange{entry->getArgument(0), b.iconst(101)})
          .front();
  mlir::Value groupDisplay = mlir::arith::AndIOp::create(
      b.builder, b.loc, isGroupClass,
      b.cmpi(mlir::arith::CmpIPredicate::sgt, memberCount, b.iconst(0)));
  mlir::cf::CondBranchOp::create(b.builder, b.loc, groupDisplay, groupHeader,
                                 mlir::ValueRange{}, plainHeader,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(plainHeader);
  b.call("write_cstr", mlir::TypeRange{},
         mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_header")});
  mlir::cf::BranchOp::create(b.builder, b.loc, head,
                             mlir::ValueRange{loadTracebackSize(b)});

  // Group display: everything from here to the summary sits behind the
  // "  | " gutter (the member tree resets it before the numbered sections).
  b.builder.setInsertionPointToEnd(groupHeader);
  // A traceback-less group (the except* combined rethrow) skips the header
  // line but keeps the gutter for its summary, like CPython.
  mlir::Value groupFrames = loadTracebackSize(b);
  mlir::Value groupHasFrames =
      b.cmpi(mlir::arith::CmpIPredicate::sgt, groupFrames, b.iconst(0));
  auto groupHeaderIf = mlir::scf::IfOp::create(
      b.builder, b.loc, mlir::TypeRange{}, groupHasFrames,
      /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToStart(&groupHeaderIf.getThenRegion().front());
    b.call("write_cstr", mlir::TypeRange{},
           mlir::ValueRange{b.iconst32(2), b.addrOf(".tb_group_header")});
  }
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(2),
                              b.addrOf("g_tb_prefix_spaces"),
                              /*alignment=*/8);
  mlir::cf::BranchOp::create(b.builder, b.loc, head,
                             mlir::ValueRange{groupFrames});

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
                          entry->getArgument(4), entry->getArgument(5),
                          entry->getArgument(6)});
  mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(-1),
                              b.addrOf("g_tb_prefix_spaces"),
                              /*alignment=*/8);
  mlir::cf::CondBranchOp::create(b.builder, b.loc, groupDisplay, groupMembers,
                                 mlir::ValueRange{}, finish,
                                 mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(groupMembers);
  b.call("print_group_members", mlir::TypeRange{},
         mlir::ValueRange{entry->getArgument(1), b.iconst(2)});
  mlir::cf::BranchOp::create(b.builder, b.loc, finish, mlir::ValueRange{});

  b.builder.setInsertionPointToEnd(finish);
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
  buildReleaseTakenException(b);
  buildReleaseCurrentChain(b);
  buildStashCurrentAsContext(b);
  buildSetCurrentSuppress(b);
  buildSetCurrentCause(b);
  buildReadSourceLine(b);
  buildExceptionClassName(b);
  buildLeadingWhitespace(b);
  buildWriteSpaces(b);
  buildWritePrefix(b);
  buildExceptionGroupMemberCount(b);
  buildPrintMarker(b);
  buildPrintTraceFrame(b);
  buildUtf8MessageCStr(b);
  buildPrintExceptionSummary(b);
  buildPrintGroupMembers(b);
  buildPrintChainNode(b);
  buildTracebackPrintMessage(b);
  buildReleaseExceptionStorageRaw(b);
  buildReleaseStarNode(b);
  buildStarBegin(b);
  buildStarHasResidual(b);
  buildStarResidualParts(b);
  buildStarApplyMatch(b);
  buildStarCollect(b);
  buildStarCollectedCount(b);
  buildStarNodesPtr(b);
  buildStarPop(b);
  buildStarThrowPending(b);
  buildStarRethrowResidual(b);
  buildStarRethrowSoleCollected(b);
  buildStarThrowCombined(b);
  buildStarDiscardSplit(b);
}

} // namespace py::runtime_library
