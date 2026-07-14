#include "Common/SupportBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallVector.h"

// Host boundary of the native support module: raw write, exit status, argv,
// FILE* wrappers and the in-memory buffer used by _io. These are NOT moved
// into the sys.mlir/_io.mlir manifests (where their Python-visible halves
// live) because they are not cleanly module-owned:
//  - LyHost_WriteBytes is also called by builtins.mlir's print path, which
//    must link into every program whether or not _io is imported;
//  - g_sys_exit_status is read back by LyRunPythonMain (the always-linked
//    program entry) when a SystemExit reaches the top, so the global cannot
//    live in a conditionally-imported module;
//  - the FILE*/buffer wrappers take the post-expansion memref descriptor ABI
//    and lean on support-private helpers (copy_i8_memref) that the manifest
//    lowering path cannot see.

namespace py::runtime_library {
namespace {


// LyHost_WriteBytes(i32 fd, memref<?xi8> descriptor, i64 len) at the lowered
// ABI (fd, alloc ptr, aligned ptr, offset, size, stride, len): the raw
// pointer extraction from the descriptor is the irreducibly-llvm part of the
// print path; everything above it (print semantics, the end="\n"
// terminator) lives in builtins.mlir.
void buildHostWriteBytes(SupportBuilder &b) {
  auto fn = b.beginFunction(
      "LyHost_WriteBytes",
      b.builder.getFunctionType({b.i32(), b.ptr(), b.ptr(), b.i64(), b.i64(),
                                 b.i64(), b.i64()},
                                {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::func::CallOp::create(
      b.builder, b.loc, "print_bytes", mlir::TypeRange{},
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(2),
                       entry->getArgument(3), entry->getArgument(4),
                       entry->getArgument(5), entry->getArgument(6)});
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// void LyHost_SetExitStatus(i64 status): records the sys.exit status in
// g_sys_exit_status. LyRunPythonMain reads it back when a SystemExit reaches
// the top level; a global (not an exception payload) because the exception
// object layout has no slot beyond the message string.
void buildHostSetExitStatus(SupportBuilder &b) {
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::Internal,
                                 "g_sys_exit_status",
                                 b.builder.getIntegerAttr(b.i64(), 0),
                                 /*alignment=*/8);
  }
  auto fn = b.beginFunction(
      "LyHost_SetExitStatus", b.builder.getFunctionType({b.i64()}, {}));
  mlir::Block *entry = fn.addEntryBlock();
  b.builder.setInsertionPointToEnd(entry);
  mlir::LLVM::StoreOp::create(b.builder, b.loc, entry->getArgument(0),
                              b.addrOf("g_sys_exit_status"), /*alignment=*/8);
  mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
}

// argv cluster: LyHost_InitArgs records the process argc/argv (called by the
// AOT main shim and the JIT driver before the program body runs); the
// LyHost_Argv* accessors hand the raw C strings to sys.mlir's LySys_GetArgv,
// which builds the Python list[str]. The raw char** walk is the
// irreducibly-llvm part; everything above it lives in the manifest.
void buildHostArgvSupport(SupportBuilder &b) {
  {
    mlir::OpBuilder::InsertionGuard guard(b.builder);
    b.builder.setInsertionPointToEnd(b.module.getBody());
    mlir::LLVM::GlobalOp::create(b.builder, b.loc, b.i64(),
                                 /*isConstant=*/false,
                                 mlir::LLVM::Linkage::Internal, "g_argv_count",
                                 b.builder.getIntegerAttr(b.i64(), 0),
                                 /*alignment=*/8);
    auto vector = mlir::LLVM::GlobalOp::create(
        b.builder, b.loc, b.ptr(), /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, "g_argv_vector", mlir::Attribute(),
        /*alignment=*/8);
    mlir::Block *init = b.builder.createBlock(&vector.getInitializerRegion());
    b.builder.setInsertionPointToEnd(init);
    mlir::Value null = mlir::LLVM::ZeroOp::create(b.builder, b.loc, b.ptr());
    mlir::LLVM::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{null});
  }

  {
    auto fn = b.beginFunction(
        "LyHost_InitArgs", b.builder.getFunctionType({b.i32(), b.ptr()}, {}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value count = mlir::arith::ExtSIOp::create(
        b.builder, b.loc, b.i64(), entry->getArgument(0));
    mlir::LLVM::StoreOp::create(b.builder, b.loc, count,
                                b.addrOf("g_argv_count"), /*alignment=*/8);
    mlir::LLVM::StoreOp::create(b.builder, b.loc, entry->getArgument(1),
                                b.addrOf("g_argv_vector"), /*alignment=*/8);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  }

  {
    auto fn = b.beginFunction("LyHost_ArgvCount",
                              b.builder.getFunctionType({}, {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value count = mlir::LLVM::LoadOp::create(
        b.builder, b.loc, b.i64(), b.addrOf("g_argv_count"), /*alignment=*/8);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{count});
  }

  {
    auto fn = b.beginFunction("LyHost_ArgvLen",
                              b.builder.getFunctionType({b.i64()}, {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value vector = b.loadPtrVal(b.addrOf("g_argv_vector"));
    // char** elements have i64 width; gepI64 walks them.
    mlir::Value argument =
        b.loadPtrVal(b.gepI64(vector, entry->getArgument(0)));
    mlir::Value length =
        b.call("strlen", b.i64(), mlir::ValueRange{argument}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{length});
  }

  {
    // LyHost_ArgvCopy(i64 index, memref<?xi8> dest, i64 len) at the lowered
    // ABI (index, alloc, aligned, offset, size, stride, len).
    auto fn = b.beginFunction(
        "LyHost_ArgvCopy",
        b.builder.getFunctionType({b.i64(), b.ptr(), b.ptr(), b.i64(),
                                   b.i64(), b.i64(), b.i64()},
                                  {}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value vector = b.loadPtrVal(b.addrOf("g_argv_vector"));
    mlir::Value source =
        b.loadPtrVal(b.gepI64(vector, entry->getArgument(0)));
    mlir::Value dest = b.gepI8(entry->getArgument(2), entry->getArgument(3));
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, dest, source,
                                 entry->getArgument(6), /*isVolatile=*/false);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  }
}

// FILE* cluster: the host boundary for _io.mlir's file-backed streams. libc
// FILE* (fopen family) rather than raw open(2) because the O_* flag values
// are target-dependent while the fopen mode strings are portable, and the
// manifest bytecode is embedded target-independently. Handles cross the
// boundary as i64 (ptrtoint of the FILE*).
void buildHostFileSupport(SupportBuilder &b) {
  {
    // i64 LyHost_FOpen(memref<?xi8> path, i64 path_len, memref<?xi8> mode,
    // i64 mode_len): NUL-terminate both views, fopen, 0 on failure.
    auto fn = b.beginFunction(
        "LyHost_FOpen",
        b.builder.getFunctionType({b.ptr(), b.ptr(), b.i64(), b.i64(),
                                   b.i64(), b.i64(), b.ptr(), b.ptr(),
                                   b.i64(), b.i64(), b.i64(), b.i64()},
                                  {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value pathCStr =
        b.call("copy_i8_memref", b.ptr(),
               mlir::ValueRange{entry->getArgument(1), entry->getArgument(2),
                                entry->getArgument(5), entry->getArgument(4)})
            .front();
    mlir::Value modeCStr =
        b.call("copy_i8_memref", b.ptr(),
               mlir::ValueRange{entry->getArgument(7), entry->getArgument(8),
                                entry->getArgument(11),
                                entry->getArgument(10)})
            .front();
    mlir::Value file =
        b.call("fopen", b.ptr(), mlir::ValueRange{pathCStr, modeCStr})
            .front();
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{pathCStr});
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{modeCStr});
    mlir::Value handle =
        mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), file);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{handle});
  }

  {
    // i64 LyHost_FRead(i64 file, memref<?xi8> dest, i64 offset, i64 max):
    // fread into dest[offset..offset+max); returns the byte count.
    auto fn = b.beginFunction(
        "LyHost_FRead",
        b.builder.getFunctionType({b.i64(), b.ptr(), b.ptr(), b.i64(),
                                   b.i64(), b.i64(), b.i64(), b.i64()},
                                  {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value base = b.gepI8(entry->getArgument(2), entry->getArgument(3));
    mlir::Value dest = b.gepI8(base, entry->getArgument(6));
    mlir::Value one = b.iconst(1);
    mlir::Value count =
        b.call("fread", b.i64(),
               mlir::ValueRange{dest, one, entry->getArgument(7), file})
            .front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{count});
  }

  {
    // i32 LyHost_FGetc(i64 file): fgetc (-1 on EOF).
    auto fn = b.beginFunction(
        "LyHost_FGetc", b.builder.getFunctionType({b.i64()}, {b.i32()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value ch = b.call("fgetc", b.i32(), mlir::ValueRange{file}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{ch});
  }

  {
    // i64 LyHost_FWrite(i64 file, memref<?xi8> bytes, i64 len).
    auto fn = b.beginFunction(
        "LyHost_FWrite",
        b.builder.getFunctionType({b.i64(), b.ptr(), b.ptr(), b.i64(),
                                   b.i64(), b.i64(), b.i64()},
                                  {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value src = b.gepI8(entry->getArgument(2), entry->getArgument(3));
    mlir::Value one = b.iconst(1);
    mlir::Value count =
        b.call("fwrite", b.i64(),
               mlir::ValueRange{src, one, entry->getArgument(6), file})
            .front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{count});
  }

  {
    // i32 LyHost_FClose(i64 file).
    auto fn = b.beginFunction(
        "LyHost_FClose", b.builder.getFunctionType({b.i64()}, {b.i32()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value status =
        b.call("fclose", b.i32(), mlir::ValueRange{file}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    // i32 LyHost_FFlush(i64 file).
    auto fn = b.beginFunction(
        "LyHost_FFlush", b.builder.getFunctionType({b.i64()}, {b.i32()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value status =
        b.call("fflush", b.i32(), mlir::ValueRange{file}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    // i32 LyHost_Fileno(i64 file).
    auto fn = b.beginFunction(
        "LyHost_Fileno", b.builder.getFunctionType({b.i64()}, {b.i32()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value fd = b.call("fileno", b.i32(), mlir::ValueRange{file}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{fd});
  }

  {
    // i32 LyHost_FSeek(i64 file, i64 offset, i32 whence): fseek (the C
    // SEEK_SET/CUR/END values equal Python's 0/1/2 on every libc).
    auto fn = b.beginFunction(
        "LyHost_FSeek",
        b.builder.getFunctionType({b.i64(), b.i64(), b.i32()}, {b.i32()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value status =
        b.call("fseek", b.i32(),
               mlir::ValueRange{file, entry->getArgument(1),
                                entry->getArgument(2)})
            .front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }

  {
    // i64 LyHost_FTell(i64 file).
    auto fn = b.beginFunction(
        "LyHost_FTell", b.builder.getFunctionType({b.i64()}, {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value where =
        b.call("ftell", b.i64(), mlir::ValueRange{file}).front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{where});
  }

  {
    // i32 LyHost_FUngetc(i64 file, i32 ch): push one byte back so the
    // character-limited text read can stop exactly on the next lead byte.
    auto fn = b.beginFunction(
        "LyHost_FUngetc",
        b.builder.getFunctionType({b.i64(), b.i32()}, {b.i32()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value pushed =
        b.call("ungetc", b.i32(),
               mlir::ValueRange{entry->getArgument(1), file})
            .front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{pushed});
  }

  {
    // i32 LyHost_FTruncate(i64 file, i64 size): flush the FILE* buffer so
    // ftruncate sees the written bytes, then truncate through the fd.
    auto fn = b.beginFunction(
        "LyHost_FTruncate",
        b.builder.getFunctionType({b.i64(), b.i64()}, {b.i32()}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value file = b.intToPtr(entry->getArgument(0));
    mlir::Value flushed =
        b.call("fflush", b.i32(), mlir::ValueRange{file}).front();
    (void)flushed;
    mlir::Value fd = b.call("fileno", b.i32(), mlir::ValueRange{file}).front();
    mlir::Value status =
        b.call("ftruncate", b.i32(),
               mlir::ValueRange{fd, entry->getArgument(1)})
            .front();
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{status});
  }
}

// raw-buffer cluster: the host boundary for _io.mlir's in-memory streams
// (StringIO/BytesIO). The stream object stores the malloc'd payload as an
// i64 pointer INSIDE its header words, so growth updates the header in
// place and the object's physical values never change (the same indirection
// trick as the FILE* handle); memref-world code cannot dereference a raw
// pointer, so the copies cross this boundary.
void buildHostBufferSupport(SupportBuilder &b) {
  {
    // i64 LyHost_BufferGrow(i64 buffer, i64 new_capacity): realloc (NULL in
    // -> malloc); aborts on exhaustion like the runtime's other allocators.
    auto fn = b.beginFunction(
        "LyHost_BufferGrow",
        b.builder.getFunctionType({b.i64(), b.i64()}, {b.i64()}));
    mlir::Block *entry = fn.addEntryBlock();
    mlir::Region &body = fn.getBody();
    mlir::Block *ok = b.builder.createBlock(&body);
    mlir::Block *trap = b.builder.createBlock(&body);
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value buffer = b.intToPtr(entry->getArgument(0));
    mlir::Value grown =
        b.call("realloc", b.ptr(),
               mlir::ValueRange{buffer, entry->getArgument(1)})
            .front();
    mlir::Value failed = b.ptrEq(grown, b.nullPtr());
    mlir::cf::CondBranchOp::create(b.builder, b.loc, failed, trap,
                                   mlir::ValueRange{}, ok, mlir::ValueRange{});
    b.builder.setInsertionPointToEnd(ok);
    mlir::Value handle =
        mlir::LLVM::PtrToIntOp::create(b.builder, b.loc, b.i64(), grown);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{handle});
    b.builder.setInsertionPointToEnd(trap);
    b.emitTrap(b.i64());
  }

  {
    // void LyHost_BufferCopyIn(i64 buffer, i64 at, memref<?xi8> src, i64 len).
    auto fn = b.beginFunction(
        "LyHost_BufferCopyIn",
        b.builder.getFunctionType({b.i64(), b.i64(), b.ptr(), b.ptr(),
                                   b.i64(), b.i64(), b.i64(), b.i64()},
                                  {}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value dest =
        b.gepI8(b.intToPtr(entry->getArgument(0)), entry->getArgument(1));
    mlir::Value source =
        b.gepI8(entry->getArgument(3), entry->getArgument(4));
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, dest, source,
                                 entry->getArgument(7), /*isVolatile=*/false);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  }

  {
    // void LyHost_BufferCopyOut(i64 buffer, i64 at, memref<?xi8> dest,
    // i64 len).
    auto fn = b.beginFunction(
        "LyHost_BufferCopyOut",
        b.builder.getFunctionType({b.i64(), b.i64(), b.ptr(), b.ptr(),
                                   b.i64(), b.i64(), b.i64(), b.i64()},
                                  {}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value source =
        b.gepI8(b.intToPtr(entry->getArgument(0)), entry->getArgument(1));
    mlir::Value dest = b.gepI8(entry->getArgument(3), entry->getArgument(4));
    mlir::LLVM::MemcpyOp::create(b.builder, b.loc, dest, source,
                                 entry->getArgument(7), /*isVolatile=*/false);
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  }

  {
    // void LyHost_BufferFree(i64 buffer): free(NULL) is a no-op.
    auto fn = b.beginFunction("LyHost_BufferFree",
                              b.builder.getFunctionType({b.i64()}, {}));
    mlir::Block *entry = fn.addEntryBlock();
    b.builder.setInsertionPointToEnd(entry);
    mlir::Value buffer = b.intToPtr(entry->getArgument(0));
    b.call("free", mlir::TypeRange{}, mlir::ValueRange{buffer});
    mlir::func::ReturnOp::create(b.builder, b.loc, mlir::ValueRange{});
  }
}

} // namespace

void buildHostSupport(SupportBuilder &b) {
  buildHostWriteBytes(b);
  buildHostSetExitStatus(b);
  buildHostArgvSupport(b);
  buildHostFileSupport(b);
  buildHostBufferSupport(b);
}

} // namespace py::runtime_library
