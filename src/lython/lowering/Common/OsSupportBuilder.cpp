#include "Common/SupportBuilder.h"
#include "ExceptionTaxonomy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"

// Host boundary of the OS/time cluster: the calls behind modules/posix.mlir
// and modules/time.mlir. They live here rather than in those manifests
// because the manifests are embedded as target-INDEPENDENT bytecode while
// every routine below reads at least one target-dependent fact — the errno
// accessor's symbol, a `struct stat` / `struct dirent` / `struct tm` byte
// offset, or CLOCK_MONOTONIC's value (see HostTargetLayout).
//
// The boundary is drawn so that no Python-level semantics live here: each
// routine is one libc call plus the descriptor/struct marshalling, returns a
// raw i32/i64 status, and leaves errno reporting, str construction and
// exception raising to the manifest above it.

namespace py::runtime_library {
namespace {

// A memref<?xT> parameter at the lowered ABI: (allocated, aligned, offset,
// size, stride). Only three of the five words are ever needed here.
struct View {
  mlir::Value aligned;
  mlir::Value offset;
  mlir::Value stride;
};

void appendMemRef(SupportBuilder &b, llvm::SmallVectorImpl<mlir::Type> &types) {
  types.append({b.ptr(), b.ptr(), b.i64(), b.i64(), b.i64()});
}

View viewAt(mlir::Block *entry, unsigned first) {
  return View{entry->getArgument(first + 1), entry->getArgument(first + 2),
              entry->getArgument(first + 4)};
}

// malloc'd NUL-terminated copy of a byte view; the caller frees it.
mlir::Value cstr(SupportBuilder &b, const View &view, mlir::Value length) {
  return b
      .call("copy_i8_memref", b.ptr(),
            mlir::ValueRange{view.aligned, view.offset, length, view.stride})
      .front();
}

// The first byte the view addresses. Every buffer the manifests pass in comes
// from memref.alloc, so the stride is 1 and the offset is the only adjustment.
mlir::Value viewBase(SupportBuilder &b, const View &view) {
  return b.gepI8(view.aligned, view.offset);
}

// out[index] of a memref<?xi64> view.
mlir::Value wordSlot(SupportBuilder &b, const View &view, std::int64_t index) {
  mlir::Value at =
      mlir::arith::AddIOp::create(b.builder, b.loc, view.offset, b.iconst(index));
  return b.gepI64(view.aligned, at);
}

mlir::LLVM::LLVMFunctionType snprintfType(SupportBuilder &b) {
  return mlir::LLVM::LLVMFunctionType::get(b.i32(), {b.ptr(), b.i64(), b.ptr()},
                                           /*isVarArg=*/true);
}

void declareOsExternals(SupportBuilder &b) {
  b.declareExternal(b.host.errnoAccessor,
                    b.builder.getFunctionType({}, {b.ptr()}));
  b.declareExternal("strerror", b.builder.getFunctionType({b.i32()}, {b.ptr()}));
  b.declareExternal("getcwd", b.builder.getFunctionType({b.ptr(), b.i64()},
                                                        {b.ptr()}));
  b.declareExternal("chdir", b.builder.getFunctionType({b.ptr()}, {b.i32()}));
  b.declareExternal("mkdir",
                    b.builder.getFunctionType({b.ptr(), b.i32()}, {b.i32()}));
  b.declareExternal("rmdir", b.builder.getFunctionType({b.ptr()}, {b.i32()}));
  b.declareExternal("unlink", b.builder.getFunctionType({b.ptr()}, {b.i32()}));
  b.declareExternal("rename", b.builder.getFunctionType({b.ptr(), b.ptr()},
                                                        {b.i32()}));
  b.declareExternal("access",
                    b.builder.getFunctionType({b.ptr(), b.i32()}, {b.i32()}));
  b.declareExternal(b.host.statSymbol,
                    b.builder.getFunctionType({b.ptr(), b.ptr()}, {b.i32()}));
  b.declareExternal(b.host.lstatSymbol,
                    b.builder.getFunctionType({b.ptr(), b.ptr()}, {b.i32()}));
  b.declareExternal("opendir", b.builder.getFunctionType({b.ptr()}, {b.ptr()}));
  b.declareExternal(b.host.readdirSymbol,
                    b.builder.getFunctionType({b.ptr()}, {b.ptr()}));
  b.declareExternal("closedir",
                    b.builder.getFunctionType({b.ptr()}, {b.i32()}));
  b.declareExternal("getenv", b.builder.getFunctionType({b.ptr()}, {b.ptr()}));
  b.declareExternal(
      "setenv",
      b.builder.getFunctionType({b.ptr(), b.ptr(), b.i32()}, {b.i32()}));
  b.declareExternal("unsetenv",
                    b.builder.getFunctionType({b.ptr()}, {b.i32()}));
  b.declareExternal(
      "clock_gettime",
      b.builder.getFunctionType({b.i32(), b.ptr()}, {b.i32()}));
  b.declareExternal("nanosleep", b.builder.getFunctionType({b.ptr(), b.ptr()},
                                                           {b.i32()}));
  b.declareExternal("localtime_r", b.builder.getFunctionType({b.ptr(), b.ptr()},
                                                             {b.ptr()}));
  b.declareExternal("gmtime_r", b.builder.getFunctionType({b.ptr(), b.ptr()},
                                                          {b.ptr()}));
  b.declareExternal("mktime", b.builder.getFunctionType({b.ptr()}, {b.i64()}));
  b.declareExternal("strftime",
                    b.builder.getFunctionType(
                        {b.ptr(), b.i64(), b.ptr(), b.ptr()}, {b.i64()}));
  if (!b.module.lookupSymbol("snprintf")) {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::LLVMFuncOp::create(b.builder, b.loc, "snprintf",
                                   snprintfType(b));
  }
  // `char **environ` is exported by every libc this targets; the walk below
  // needs the vector itself, which getenv cannot hand over.
  if (!b.module.lookupSymbol("environ")) {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.ptr(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::External, "environ",
                                 mlir::Attribute(), /*alignment=*/8);
  }
}

// The process-identity calls. One wrapper each rather than a dispatcher: the
// manifest cannot declare a raw libc symbol (only support-module exports
// survive the native verifier), and the bodies are one call apiece.
void buildIdentityCalls(SupportBuilder &b) {
  const std::pair<llvm::StringRef, llvm::StringRef> entries[] = {
      {"LyHost_GetPid", "getpid"},   {"LyHost_GetPPid", "getppid"},
      {"LyHost_GetUid", "getuid"},   {"LyHost_GetEUid", "geteuid"},
      {"LyHost_GetGid", "getgid"},   {"LyHost_GetEGid", "getegid"},
  };
  for (const auto &entry : entries) {
    b.declareExternal(entry.second, b.builder.getFunctionType({}, {b.i32()}));
    auto fn = b.beginFunction(entry.first,
                              b.builder.getFunctionType({}, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value raw = b.call(entry.second, b.i32(), mlir::ValueRange{}).front();
    mlir::Value widened =
        mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(), raw);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{widened});
  }
}

// i64 LyHost_Strerror(i32 code, memref out, i64 cap): the message's byte
// length (possibly > cap so the manifest can size a retry).
void buildStrerror(SupportBuilder &b) {
  llvm::SmallVector<mlir::Type, 8> inputs{b.i32()};
  appendMemRef(b, inputs);
  inputs.push_back(b.i64());
  auto fn = b.beginFunction("LyHost_Strerror",
                            b.builder.getFunctionType(inputs, {b.i64()}));
  mlir::Block *block = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(block);
  mlir::Value text =
      b.call("strerror", b.ptr(), mlir::ValueRange{block->getArgument(0)})
          .front();
  mlir::Value length =
      b.call("strlen", b.i64(), mlir::ValueRange{text}).front();
  mlir::Value cap = block->getArgument(6);
  mlir::Value fits = b.cmpi(mlir::arith::CmpIPredicate::sle, length, cap);
  mlir::Value copy =
      mlir::arith::SelectOp::create(b.builder, b.loc, fits, length, cap);
  mlir::LLVM::MemcpyOp::create(b.builder, b.loc, viewBase(b, viewAt(block, 1)),
                               text, copy, /*isVolatile=*/false);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{length});
}

// i32 LyHost_Errno(): the current errno. Darwin spells the accessor
// `__error()`, glibc `__errno_location()`; both return a pointer to the
// thread's slot, so the read is the same once the name is chosen.
void buildErrno(SupportBuilder &b) {
  auto fn =
      b.beginFunction("LyHost_Errno", b.builder.getFunctionType({}, {b.i32()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value slot =
      b.call(b.host.errnoAccessor, b.ptr(), mlir::ValueRange{}).front();
  mlir::Value value = b.loadI32(slot);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{value});
}

// i64 LyHost_OSErrorClassId(i32 errno): the OSError subclass CPython's
// oserror_use_init dispatch would pick, straight off kOSErrorErrnoMap. A
// select chain rather than a switch: the table is short and branch-free code
// keeps the routine leaf-callable from anywhere.
void buildOSErrorClassId(SupportBuilder &b) {
  auto fn = b.beginFunction("LyHost_OSErrorClassId",
                            b.builder.getFunctionType({b.i32()}, {b.i64()}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::Value err = entry->getArgument(0);
  mlir::Value result = b.iconst(py::exceptions::kOSErrorClassId);
  for (const py::exceptions::OSErrorErrnoMapping &row :
       py::exceptions::kOSErrorErrnoMap) {
    int value = b.host.bsdErrnoValues ? row.darwinValue : row.linuxValue;
    mlir::Value matches =
        b.cmpi(mlir::arith::CmpIPredicate::eq, err, b.iconst32(value));
    result = mlir::arith::SelectOp::create(b.builder, b.loc, matches,
                                           b.iconst(row.classId), result);
  }
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{result});
}

// The OSError message formatters. CPython's OSError.__str__ renders
// "[Errno %d] %s" plus ": %r" for one filename and ": %r -> %r" for two;
// snprintf reproduces it exactly and returns the length written (clamped to
// the buffer, which the manifest sizes at 1 KiB).
void buildOSErrorMessages(SupportBuilder &b) {
  b.stringGlobal(".os_fmt_errno", "[Errno %d] %s");
  b.stringGlobal(".os_fmt_errno_path", "[Errno %d] %s: '%s'");
  b.stringGlobal(".os_fmt_errno_path2", "[Errno %d] %s: '%s' -> '%s'");

  // The three share the clamp and the strerror lookup.
  auto emit = [&](llvm::StringRef name, unsigned paths) {
    llvm::SmallVector<mlir::Type, 20> inputs{b.i32()};
    for (unsigned index = 0; index < paths; ++index) {
      appendMemRef(b, inputs);
      inputs.push_back(b.i64());
    }
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction(name,
                              b.builder.getFunctionType(inputs, {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value err = entry->getArgument(0);
    llvm::SmallVector<mlir::Value, 2> pathCStrs;
    unsigned cursor = 1;
    for (unsigned index = 0; index < paths; ++index) {
      pathCStrs.push_back(
          cstr(b, viewAt(entry, cursor), entry->getArgument(cursor + 5)));
      cursor += 6;
    }
    View out = viewAt(entry, cursor);
    mlir::Value cap = entry->getArgument(cursor + 5);
    mlir::Value text =
        b.call("strerror", b.ptr(), mlir::ValueRange{err}).front();
    llvm::SmallVector<mlir::Value, 6> args{
        viewBase(b, out), cap,
        b.addrOf(paths == 0   ? ".os_fmt_errno"
                 : paths == 1 ? ".os_fmt_errno_path"
                              : ".os_fmt_errno_path2"),
        err, text};
    args.append(pathCStrs.begin(), pathCStrs.end());
    mlir::Value written = mlir::LLVM::CallOp::create(b.builder, b.loc,
                                                     snprintfType(b),
                                                     "snprintf", args)
                              .getResult();
    for (mlir::Value pathCStr : pathCStrs)
      b.call("free", mlir::TypeRange{}, mlir::ValueRange{pathCStr});
    mlir::Value length =
        mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(), written);
    mlir::Value limit =
        mlir::arith::SubIOp::create(b.builder, b.loc, cap, b.iconst(1));
    mlir::Value overflowed =
        b.cmpi(mlir::arith::CmpIPredicate::sgt, length, limit);
    mlir::Value clamped = mlir::arith::SelectOp::create(b.builder, b.loc,
                                                        overflowed, limit,
                                                        length);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{clamped});
  };
  emit("LyHost_OSErrorMessage", 0);
  emit("LyHost_OSErrorMessagePath", 1);
  emit("LyHost_OSErrorMessagePath2", 2);
}

// The single-path filesystem calls: NUL-terminate, call, free, return the
// libc status verbatim so the manifest reads errno once on failure.
void buildPathCalls(SupportBuilder &b) {
  // (name, libc symbol, extra i64 argument truncated to i32 or none)
  struct Entry {
    llvm::StringRef name;
    llvm::StringRef callee;
    bool hasMode;
  };
  const Entry entries[] = {
      {"LyHost_Chdir", "chdir", false},
      {"LyHost_Rmdir", "rmdir", false},
      {"LyHost_Unlink", "unlink", false},
      {"LyHost_Mkdir", "mkdir", true},
      {"LyHost_Access", "access", true},
  };
  for (const Entry &entry : entries) {
    llvm::SmallVector<mlir::Type, 8> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    if (entry.hasMode)
      inputs.push_back(b.i64());
    auto fn = b.beginFunction(entry.name,
                              b.builder.getFunctionType(inputs, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value pathCStr = cstr(b, viewAt(block, 0), block->getArgument(5));
    llvm::SmallVector<mlir::Value, 2> args{pathCStr};
    if (entry.hasMode)
      args.push_back(mlir::arith::TruncIOp::create(b.builder, b.loc, b.i32(),
                                                   block->getArgument(6)));
    mlir::Value status = b.call(entry.callee, b.i32(), args).front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{pathCStr});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    // i32 LyHost_Rename(memref src, i64 slen, memref dst, i64 dlen).
    llvm::SmallVector<mlir::Type, 12> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_Rename",
                              b.builder.getFunctionType(inputs, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value src = cstr(b, viewAt(block, 0), block->getArgument(5));
    mlir::Value dst = cstr(b, viewAt(block, 6), block->getArgument(11));
    mlir::Value status =
        b.call("rename", b.i32(), mlir::ValueRange{src, dst}).front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{src});
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{dst});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    // i64 LyHost_GetCwd(memref out, i64 cap): the byte length, or -1.
    llvm::SmallVector<mlir::Type, 8> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_GetCwd",
                              b.builder.getFunctionType(inputs, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *ok = b.builder.createBlock(&body);
    mlir::Block *fail = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(block);
    mlir::Value out = viewBase(b, viewAt(block, 0));
    mlir::Value got =
        b.call("getcwd", b.ptr(), mlir::ValueRange{out, block->getArgument(5)})
            .front();
    mlir::cf::CondBranchOp::create(b.builder, b.loc, b.ptrEq(got, b.nullPtr()),
                                   fail, mlir::ValueRange{}, ok,
                                   mlir::ValueRange{});
    b.builder.setInsertionPointToEnd(ok);
    mlir::Value length =
        b.call("strlen", b.i64(), mlir::ValueRange{out}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{length});
    b.builder.setInsertionPointToEnd(fail);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{b.iconst(-1)});
  }
}

// One os.stat_result field: load `width` bytes at `offset` and widen to i64.
// A negative width marks a signed field (Darwin's dev_t is int32_t).
void storeStatField(SupportBuilder &b, mlir::Value statBuffer,
                    const View &out, std::int64_t slot, const int (&field)[2]) {
  unsigned bits = static_cast<unsigned>(std::abs(field[1])) * 8;
  mlir::Value at = b.gepI8(statBuffer, b.iconst(field[0]));
  mlir::Value raw = mlir::LLVM::LoadOp::create(
      b.builder, b.loc, b.builder.getIntegerType(bits), at,
      /*alignment=*/static_cast<unsigned>(std::abs(field[1])));
  mlir::Value widened =
      bits == 64 ? raw
                 : (field[1] < 0 ? mlir::arith::ExtSIOp::create(
                                       b.builder, b.loc, b.i64(), raw)
                                       .getResult()
                                 : mlir::arith::ExtUIOp::create(
                                       b.builder, b.loc, b.i64(), raw)
                                       .getResult());
  mlir::LLVM::StoreOp::create(b.builder, b.loc, widened,
                              wordSlot(b, out, slot), /*alignment=*/8);
}

// i32 LyHost_Stat / LyHost_LStat (memref path, i64 len, memref<?xi64> out):
// 0 with out[0..9] = (mode, ino, dev, nlink, uid, gid, size, atime, mtime,
// ctime), or the libc status. The buffer is 256 bytes: every supported
// target's `struct stat` is 144 or less.
void buildStatCalls(SupportBuilder &b) {
  for (bool follow : {true, false}) {
    llvm::SmallVector<mlir::Type, 12> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    appendMemRef(b, inputs);
    auto fn = b.beginFunction(follow ? "LyHost_Stat" : "LyHost_LStat",
                              b.builder.getFunctionType(inputs, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *ok = b.builder.createBlock(&body);
    mlir::Block *fail = b.builder.createBlock(&body, body.end(), {b.i32()},
                                              {b.loc});
    b.builder.setInsertionPointToEnd(block);
    auto bufferType = mlir::LLVM::LLVMArrayType::get(b.i8(), 256);
    mlir::Value slot = mlir::LLVM::AllocaOp::create(
        b.builder, b.loc, b.ptr(), bufferType, b.iconst32(1), /*alignment=*/8);
    mlir::Value buffer = mlir::LLVM::GEPOp::create(
        b.builder, b.loc, b.ptr(), bufferType, slot,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                           mlir::LLVM::GEPArg(0)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    mlir::Value pathCStr = cstr(b, viewAt(block, 0), block->getArgument(5));
    mlir::Value status =
        b.call(follow ? b.host.statSymbol : b.host.lstatSymbol, b.i32(),
               mlir::ValueRange{pathCStr, buffer})
            .front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{pathCStr});
    mlir::cf::CondBranchOp::create(
        b.builder, b.loc,
        b.cmpi(mlir::arith::CmpIPredicate::ne, status, b.iconst32(0)), fail,
        mlir::ValueRange{status}, ok, mlir::ValueRange{});

    b.builder.setInsertionPointToEnd(ok);
    View out = viewAt(block, 6);
    storeStatField(b, buffer, out, 0, b.host.statMode);
    storeStatField(b, buffer, out, 1, b.host.statIno);
    storeStatField(b, buffer, out, 2, b.host.statDev);
    storeStatField(b, buffer, out, 3, b.host.statNlink);
    storeStatField(b, buffer, out, 4, b.host.statUid);
    storeStatField(b, buffer, out, 5, b.host.statGid);
    storeStatField(b, buffer, out, 6, b.host.statSize);
    storeStatField(b, buffer, out, 7, b.host.statAtime);
    storeStatField(b, buffer, out, 8, b.host.statMtime);
    storeStatField(b, buffer, out, 9, b.host.statCtime);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{b.iconst32(0)});

    b.builder.setInsertionPointToEnd(fail);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{fail->getArgument(0)});
  }
}

// The directory walk. Handles cross the boundary as i64 (ptrtoint of the
// DIR*) exactly like _io's FILE*, and the entry name is copied out as bytes
// so no memref-world code ever dereferences a `struct dirent`.
void buildDirectoryCalls(SupportBuilder &b) {
  {
    llvm::SmallVector<mlir::Type, 8> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_OpenDir",
                              b.builder.getFunctionType(inputs, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value pathCStr = cstr(b, viewAt(block, 0), block->getArgument(5));
    mlir::Value dir =
        b.call("opendir", b.ptr(), mlir::ValueRange{pathCStr}).front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{pathCStr});
    mlir::Value handle =
        mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), dir);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{handle});
  }

  {
    // i64 LyHost_ReadDirName(i64 dir, memref out, i64 cap): the name's byte
    // length (which may exceed cap, so the manifest can retry), or -1 at end.
    llvm::SmallVector<mlir::Type, 8> inputs{b.i64()};
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_ReadDirName",
                              b.builder.getFunctionType(inputs, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *ok = b.builder.createBlock(&body);
    mlir::Block *done = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(block);
    mlir::Value dir = b.intToPtr(block->getArgument(0));
    mlir::Value record =
        b.call(b.host.readdirSymbol, b.ptr(), mlir::ValueRange{dir}).front();
    mlir::cf::CondBranchOp::create(b.builder, b.loc,
                                   b.ptrEq(record, b.nullPtr()), done,
                                   mlir::ValueRange{}, ok, mlir::ValueRange{});
    b.builder.setInsertionPointToEnd(ok);
    mlir::Value name = b.gepI8(record, b.iconst(b.host.direntNameOffset));
    mlir::Value length =
        b.call("strlen", b.i64(), mlir::ValueRange{name}).front();
    mlir::Value cap = block->getArgument(6);
    mlir::Value fits = b.cmpi(mlir::arith::CmpIPredicate::sle, length, cap);
    mlir::Value copy =
        mlir::arith::SelectOp::create(b.builder, b.loc, fits, length, cap);
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, viewBase(b, viewAt(block, 1)),
                                 name, copy, /*isVolatile=*/false);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{length});
    b.builder.setInsertionPointToEnd(done);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{b.iconst(-1)});
  }

  {
    auto fn = b.beginFunction("LyHost_CloseDir",
                              b.builder.getFunctionType({b.i64()}, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value status =
        b.call("closedir", b.i32(),
               mlir::ValueRange{b.intToPtr(block->getArgument(0))})
            .front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }
}

// The environment cluster. getenv/setenv/unsetenv are per-name; the three
// LyHost_Environ* accessors walk `char **environ` the way LyHost_Argv* walks
// argv, so the manifest builds the list[str] of "KEY=VALUE" entries with the
// same code shape.
void buildEnvironmentCalls(SupportBuilder &b) {
  {
    // i64 LyHost_GetEnv(memref name, i64 nlen, memref out, i64 cap): the
    // value's byte length (possibly > cap), or -1 when unset.
    llvm::SmallVector<mlir::Type, 12> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_GetEnv",
                              b.builder.getFunctionType(inputs, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *ok = b.builder.createBlock(&body);
    mlir::Block *unset = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(block);
    mlir::Value nameCStr = cstr(b, viewAt(block, 0), block->getArgument(5));
    mlir::Value value =
        b.call("getenv", b.ptr(), mlir::ValueRange{nameCStr}).front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{nameCStr});
    mlir::cf::CondBranchOp::create(b.builder, b.loc,
                                   b.ptrEq(value, b.nullPtr()), unset,
                                   mlir::ValueRange{}, ok, mlir::ValueRange{});
    b.builder.setInsertionPointToEnd(ok);
    mlir::Value length =
        b.call("strlen", b.i64(), mlir::ValueRange{value}).front();
    mlir::Value cap = block->getArgument(11);
    mlir::Value fits = b.cmpi(mlir::arith::CmpIPredicate::sle, length, cap);
    mlir::Value copy =
        mlir::arith::SelectOp::create(b.builder, b.loc, fits, length, cap);
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, viewBase(b, viewAt(block, 6)),
                                 value, copy, /*isVolatile=*/false);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{length});
    b.builder.setInsertionPointToEnd(unset);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{b.iconst(-1)});
  }

  {
    llvm::SmallVector<mlir::Type, 14> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_SetEnv",
                              b.builder.getFunctionType(inputs, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value name = cstr(b, viewAt(block, 0), block->getArgument(5));
    mlir::Value value = cstr(b, viewAt(block, 6), block->getArgument(11));
    mlir::Value status = b.call("setenv", b.i32(),
                                mlir::ValueRange{name, value, b.iconst32(1)})
                             .front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{name});
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{value});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    llvm::SmallVector<mlir::Type, 8> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_UnsetEnv",
                              b.builder.getFunctionType(inputs, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value name = cstr(b, viewAt(block, 0), block->getArgument(5));
    mlir::Value status =
        b.call("unsetenv", b.i32(), mlir::ValueRange{name}).front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{name});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    // i64 LyHost_EnvironCount(): entries before the NULL terminator.
    auto fn = b.beginFunction("LyHost_EnvironCount",
                              b.builder.getFunctionType({}, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *head =
        b.builder.createBlock(&body, body.end(), {b.i64()}, {b.loc});
    mlir::Block *step = b.builder.createBlock(&body);
    mlir::Block *done = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(block);
    mlir::Value vector = b.loadPtrVal(b.addrOf("environ"));
    mlir::cf::BranchOp::create(b.builder, b.loc, head,
                               mlir::ValueRange{b.iconst(0)});
    b.builder.setInsertionPointToEnd(head);
    mlir::Value index = head->getArgument(0);
    mlir::Value slot = b.loadPtrVal(b.gepI64(vector, index));
    mlir::cf::CondBranchOp::create(b.builder, b.loc, b.ptrEq(slot, b.nullPtr()),
                                   done, mlir::ValueRange{}, step,
                                   mlir::ValueRange{});
    b.builder.setInsertionPointToEnd(step);
    mlir::Value next =
        mlir::arith::AddIOp::create(b.builder, b.loc, index, b.iconst(1));
    mlir::cf::BranchOp::create(b.builder, b.loc, head, mlir::ValueRange{next});
    b.builder.setInsertionPointToEnd(done);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{index});
  }

  {
    auto fn = b.beginFunction("LyHost_EnvironLen",
                              b.builder.getFunctionType({b.i64()}, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value vector = b.loadPtrVal(b.addrOf("environ"));
    mlir::Value slot = b.loadPtrVal(b.gepI64(vector, block->getArgument(0)));
    mlir::Value length =
        b.call("strlen", b.i64(), mlir::ValueRange{slot}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{length});
  }

  {
    llvm::SmallVector<mlir::Type, 8> inputs{b.i64()};
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_EnvironCopy",
                              b.builder.getFunctionType(inputs, {}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value vector = b.loadPtrVal(b.addrOf("environ"));
    mlir::Value slot = b.loadPtrVal(b.gepI64(vector, block->getArgument(0)));
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc,
                                 viewBase(b, viewAt(block, 1)), slot,
                                 block->getArgument(6), /*isVolatile=*/false);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  }
}

// A 64-byte alloca: bigger than `struct timespec` (16) and `struct tm` (56)
// on every supported target.
mlir::Value scratch(SupportBuilder &b, unsigned bytes) {
  auto type = mlir::LLVM::LLVMArrayType::get(b.i8(), bytes);
  mlir::Value slot = mlir::LLVM::AllocaOp::create(
      b.builder, b.loc, b.ptr(), type, b.iconst32(1), /*alignment=*/8);
  return mlir::LLVM::GEPOp::create(
      b.builder, b.loc, b.ptr(), type, slot,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                         mlir::LLVM::GEPArg(0)},
      mlir::LLVM::GEPNoWrapFlags::inbounds);
}

// The clock and calendar cluster.
//
// `struct tm`'s first nine members are `int tm_sec` through `int tm_isdst` in
// that order on every POSIX libc, so the nine 4-byte words at offsets 0..32
// are the one struct layout here that needs no per-target table. tm_gmtoff
// follows the ints (offset 40 after padding) on both Darwin and glibc.
void buildTimeCalls(SupportBuilder &b) {
  {
    // i64 LyHost_ClockNs(i64 monotonic): nanoseconds since the epoch, or
    // since an unspecified start when `monotonic` is nonzero.
    auto fn = b.beginFunction("LyHost_ClockNs",
                              b.builder.getFunctionType({b.i64()}, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value spec = scratch(b, 16);
    mlir::Value wantMonotonic = b.cmpi(mlir::arith::CmpIPredicate::ne,
                                       block->getArgument(0), b.iconst(0));
    mlir::Value clockId = mlir::arith::SelectOp::create(
        b.builder, b.loc, wantMonotonic, b.iconst32(b.host.clockMonotonic),
        b.iconst32(0));
    b.call("clock_gettime", b.i32(), mlir::ValueRange{clockId, spec});
    mlir::Value seconds = b.loadI64(spec);
    mlir::Value nanos = b.loadI64(b.gepI64(spec, b.iconst(1)));
    mlir::Value scaled = mlir::arith::MulIOp::create(
        b.builder, b.loc, seconds, b.iconst(1000000000));
    mlir::Value total =
        mlir::arith::AddIOp::create(b.builder, b.loc, scaled, nanos);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{total});
  }

  {
    // i32 LyHost_SleepNs(i64 nanoseconds): one nanosleep, no EINTR retry (the
    // manifest reports the interruption so `except InterruptedError` sees it,
    // as CPython does before PEP 475's retry loop).
    auto fn = b.beginFunction("LyHost_SleepNs",
                              b.builder.getFunctionType({b.i64()}, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value spec = scratch(b, 16);
    mlir::Value total = block->getArgument(0);
    mlir::Value billion = b.iconst(1000000000);
    mlir::Value seconds =
        mlir::arith::DivSIOp::create(b.builder, b.loc, total, billion);
    mlir::Value nanos =
        mlir::arith::RemSIOp::create(b.builder, b.loc, total, billion);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, seconds, spec,
                                /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, nanos,
                                b.gepI64(spec, b.iconst(1)), /*alignment=*/8);
    mlir::Value status = b.call("nanosleep", b.i32(),
                                mlir::ValueRange{spec, b.nullPtr()})
                             .front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    // i32 LyHost_TimeFields(i64 seconds, i64 utc, memref<?xi64> out): the
    // nine `struct tm` ints plus tm_gmtoff, as out[0..10]. Nonzero on failure
    // (a seconds value the platform's calendar cannot represent).
    llvm::SmallVector<mlir::Type, 8> inputs{b.i64(), b.i64()};
    appendMemRef(b, inputs);
    auto fn = b.beginFunction("LyHost_TimeFields",
                              b.builder.getFunctionType(inputs, {b.i32()}));
    mlir::Block *block = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *ok = b.builder.createBlock(&body);
    mlir::Block *fail = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(block);
    mlir::Value clockSeconds = scratch(b, 8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, block->getArgument(0),
                                clockSeconds, /*alignment=*/8);
    mlir::Value tm = scratch(b, 64);
    mlir::Value utc =
        b.cmpi(mlir::arith::CmpIPredicate::ne, block->getArgument(1),
               b.iconst(0));
    auto pick = mlir::scf::IfOp::create(b.builder, b.loc, mlir::TypeRange{b.ptr()},
                                        utc, /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(b.builder);
      b.builder.setInsertionPointToStart(pick.thenBlock());
      mlir::Value result = b.call("gmtime_r", b.ptr(),
                                  mlir::ValueRange{clockSeconds, tm})
                               .front();
      mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{result});
      b.builder.setInsertionPointToStart(pick.elseBlock());
      mlir::Value local = b.call("localtime_r", b.ptr(),
                                 mlir::ValueRange{clockSeconds, tm})
                              .front();
      mlir::scf::YieldOp::create(b.builder, b.loc, mlir::ValueRange{local});
    }
    mlir::cf::CondBranchOp::create(b.builder, b.loc,
                                   b.ptrEq(pick.getResult(0), b.nullPtr()),
                                   fail, mlir::ValueRange{}, ok,
                                   mlir::ValueRange{});
    b.builder.setInsertionPointToEnd(ok);
    View out = viewAt(block, 2);
    for (std::int64_t index = 0; index < 9; ++index) {
      mlir::Value at = b.gepI8(tm, b.iconst(index * 4));
      mlir::Value raw = b.loadI32(at);
      mlir::Value widened =
          mlir::arith::ExtSIOp::create(b.builder, b.loc, b.i64(), raw);
      mlir::LLVM::StoreOp::create(b.builder, b.loc, widened,
                                  wordSlot(b, out, index), /*alignment=*/8);
    }
    mlir::Value gmtoff = b.loadI64(b.gepI8(tm, b.iconst(40)));
    mlir::LLVM::StoreOp::create(b.builder, b.loc, gmtoff, wordSlot(b, out, 9),
                                /*alignment=*/8);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{b.iconst32(0)});
    b.builder.setInsertionPointToEnd(fail);
    mlir::func::ReturnOp::create(b.builder, b.loc,
                                 mlir::ValueRange{b.iconst32(-1)});
  }

  // Both remaining routines start from the nine-word field vector, so they
  // share the `struct tm` materialization.
  auto materializeTm = [&](mlir::Block *block, const View &fields) {
    mlir::Value tm = scratch(b, 64);
    // memset via explicit zero stores: tm_gmtoff / tm_zone must not be junk.
    for (std::int64_t index = 0; index < 8; ++index)
      mlir::LLVM::StoreOp::create(b.builder, b.loc, b.iconst(0),
                                  b.gepI64(tm, b.iconst(index)),
                                  /*alignment=*/8);
    for (std::int64_t index = 0; index < 9; ++index) {
      mlir::Value word = b.loadI64(wordSlot(b, fields, index));
      mlir::Value narrowed =
          mlir::arith::TruncIOp::create(b.builder, b.loc, b.i32(), word);
      mlir::LLVM::StoreOp::create(b.builder, b.loc, narrowed,
                                  b.gepI8(tm, b.iconst(index * 4)),
                                  /*alignment=*/4);
    }
    (void)block;
    return tm;
  };

  {
    // i64 LyHost_Strftime(memref fmt, i64 fmtlen, memref<?xi64> fields,
    // memref out, i64 cap): bytes written, or -1 when the result does not fit.
    llvm::SmallVector<mlir::Type, 20> inputs;
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    appendMemRef(b, inputs);
    appendMemRef(b, inputs);
    inputs.push_back(b.i64());
    auto fn = b.beginFunction("LyHost_Strftime",
                              b.builder.getFunctionType(inputs, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value fmtCStr = cstr(b, viewAt(block, 0), block->getArgument(5));
    mlir::Value tm = materializeTm(block, viewAt(block, 6));
    mlir::Value written =
        b.call("strftime", b.i64(),
               mlir::ValueRange{viewBase(b, viewAt(block, 11)),
                                block->getArgument(16), fmtCStr, tm})
            .front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{fmtCStr});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{written});
  }

  {
    // i64 LyHost_Mktime(memref<?xi64> fields): the local-time epoch seconds,
    // or -1. tm_isdst comes from fields[8] so a caller can ask for the
    // platform's DST guess with -1.
    llvm::SmallVector<mlir::Type, 8> inputs;
    appendMemRef(b, inputs);
    auto fn = b.beginFunction("LyHost_Mktime",
                              b.builder.getFunctionType(inputs, {b.i64()}));
    mlir::Block *block = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(block);
    mlir::Value tm = materializeTm(block, viewAt(block, 0));
    mlir::Value seconds =
        b.call("mktime", b.i64(), mlir::ValueRange{tm}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{seconds});
  }
}

} // namespace

void buildOsSupport(SupportBuilder &b) {
  declareOsExternals(b);
  buildIdentityCalls(b);
  buildStrerror(b);
  buildErrno(b);
  buildOSErrorClassId(b);
  buildOSErrorMessages(b);
  buildPathCalls(b);
  buildStatCalls(b);
  buildDirectoryCalls(b);
  buildEnvironmentCalls(b);
  buildTimeCalls(b);
}

} // namespace py::runtime_library
