#include "AppleAMX.h"

#include "../../Primitive/TensorGemm.h"
#include "../../Primitive/TensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::lowering::arch::apple {
namespace {

namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// Instruction encoding
//===----------------------------------------------------------------------===//
//
// Apple's matrix coprocessor is driven by reserved A64 instruction words:
//
//   word = 0x00201000 | (opcode << 5) | Rn        (Rn = GPR number)
//   word = 0x00201000 | (opcode << 5) | imm5      (SET/CLR only)
//
// There is no assembler mnemonic, so each instruction is emitted as a `.word`
// with the register-allocated GPR number spliced into the low five bits. The
// `0$0` trick renders the operand register (`x0`..`x30`) as a hex literal
// (`0x0`..`0x30`) and then corrects the hex-vs-decimal skew above x15 by
// subtracting 6 per 16.

enum AmxOpcode : unsigned {
  kAmxLdx = 0,
  kAmxLdy = 1,
  kAmxLdz = 4,
  kAmxStz = 5,
  kAmxFma32 = 12,
  kAmxSetClr = 17,
};

std::string amxGprAsm(unsigned opcode) {
  return ".word (0x201000 + (" + std::to_string(opcode) +
         " << 5) + 0$0 - ((0$0 >> 4) * 6))";
}

// SET/CLR take an immediate, not a register. The three NOPs match the issue
// sequence Apple's own code uses around the state-toggling instructions.
std::string amxImmAsm(unsigned opcode, unsigned imm5) {
  return "nop\nnop\nnop\n.word (0x201000 + (" + std::to_string(opcode) +
         " << 5) + " + std::to_string(imm5) + ")";
}

void emitAmxImm(mlir::OpBuilder &builder, mlir::Location loc, unsigned opcode,
                unsigned imm5) {
  LLVM::InlineAsmOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      amxImmAsm(opcode, imm5), /*constraints=*/"",
      /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(builder.getContext(), LLVM::AsmDialect::AD_ATT),
      mlir::ArrayAttr());
}

void emitAmxGpr(mlir::OpBuilder &builder, mlir::Location loc, unsigned opcode,
                mlir::Value operand) {
  LLVM::InlineAsmOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{operand},
      amxGprAsm(opcode), /*constraints=*/"r",
      /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(builder.getContext(), LLVM::AsmDialect::AD_ATT),
      mlir::ArrayAttr());
}

//===----------------------------------------------------------------------===//
// Run-time engine probe
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kMatrixBackendName{"LyMatrixBackend"};
constexpr llvm::StringLiteral kBackendGlobalName{"__lython_matrix_backend"};
constexpr llvm::StringLiteral kCpuFamilyEnvName{"__lython_cpufamily_name"};

// hw.cpufamily values for the SoCs whose AMX operand semantics are known.
// These are opaque identifiers with no ordering: XNU documents that features
// must never be inferred from their relative value, so the check is an exact
// allowlist. M5 and later are deliberately absent -- an unknown generation
// silently reinterpreting operand bits is the failure mode this guards.
constexpr std::uint32_t kKnownCpuFamilies[] = {
    0x1b588bb3u, // M1 / Pro / Max / Ultra
    0xda33d83du, // M2 / Pro / Max / Ultra
    0xfa33415eu, // M3 (Ibiza)
    0x5f4dea93u, // M3 Pro (Lobos)
    0x72015832u, // M3 Max / Ultra (Palma)
    0x6f5129acu, // M4 (Donan)
    0x17d5b93au, // M4 Pro / Max (Brava)
};

LLVM::LLVMFuncOp ensureFuncDecl(mlir::ModuleOp module, mlir::OpBuilder &builder,
                                llvm::StringRef name,
                                LLVM::LLVMFunctionType type) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  return LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(), name, type);
}

mlir::Value constI64(mlir::OpBuilder &builder, mlir::Location loc,
                     std::int64_t value) {
  return LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                  builder.getI64IntegerAttr(value));
}

mlir::Value constI32(mlir::OpBuilder &builder, mlir::Location loc,
                     std::int32_t value) {
  return LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                  builder.getI32IntegerAttr(value));
}

// LyMatrixBackend() -> i64: 1 when this process may issue AMX instructions,
// 0 otherwise. The answer is cached in an unsynchronized global; the race is
// benign because every racing writer computes the same value.
//
// The probe forks: a child that runs SET/CLR cannot leave AMX state or a
// SIGILL behind in this process, whereas an in-process signal handler would
// have to unwind out of a half-open AMX region. It runs before the kernel
// dispatches to worker threads, so the fork is single-threaded.
void buildMatrixBackend(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto i64 = builder.getI64Type();
  auto i32 = builder.getI32Type();
  auto ptr = LLVM::LLVMPointerType::get(context);
  mlir::Location loc = builder.getUnknownLoc();

  builder.setInsertionPointToEnd(module.getBody());
  auto global = LLVM::GlobalOp::create(builder, loc, i64, /*isConstant=*/false,
                                       LLVM::Linkage::Internal,
                                       kBackendGlobalName,
                                       builder.getI64IntegerAttr(0));
  llvm::StringRef nameLiteral("hw.cpufamily\0", 13);
  auto nameType = LLVM::LLVMArrayType::get(builder.getI8Type(), 13);
  auto nameGlobal = LLVM::GlobalOp::create(
      builder, loc, nameType, /*isConstant=*/true, LLVM::Linkage::Internal,
      kCpuFamilyEnvName, builder.getStringAttr(nameLiteral));

  auto sysctlFn = ensureFuncDecl(
      module, builder, "sysctlbyname",
      LLVM::LLVMFunctionType::get(i32, {ptr, ptr, ptr, ptr, i64}));
  auto forkFn = ensureFuncDecl(module, builder, "fork",
                               LLVM::LLVMFunctionType::get(i32, {}));
  auto waitpidFn = ensureFuncDecl(
      module, builder, "waitpid", LLVM::LLVMFunctionType::get(i32, {i32, ptr, i32}));
  auto exitFn = ensureFuncDecl(module, builder, "_exit",
                               LLVM::LLVMFunctionType::get(
                                   LLVM::LLVMVoidType::get(context), {i32}));

  builder.setInsertionPointToEnd(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(builder, loc, kMatrixBackendName,
                                     LLVM::LLVMFunctionType::get(i64, {}),
                                     LLVM::Linkage::Internal);

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::Block *probe = fn.addBlock();
  mlir::Block *checkFamily = fn.addBlock();
  mlir::Block *doFork = fn.addBlock();
  mlir::Block *child = fn.addBlock();
  mlir::Block *parent = fn.addBlock();
  parent->addArgument(i32, loc);
  mlir::Block *store = fn.addBlock();
  store->addArgument(i64, loc);
  mlir::Block *done = fn.addBlock();
  done->addArgument(i64, loc);

  builder.setInsertionPointToStart(entry);
  mlir::Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
  mlir::Value cached = LLVM::LoadOp::create(builder, loc, i64, globalPtr);
  mlir::Value zero = constI64(builder, loc, 0);
  mlir::Value one = constI64(builder, loc, 1);
  mlir::Value isCached = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::ne, cached, zero);
  // Cache holds answer+1 so that zero means "not yet probed".
  mlir::Value cachedAnswer = LLVM::SubOp::create(builder, loc, cached, one);
  LLVM::CondBrOp::create(builder, loc, isCached, done,
                         mlir::ValueRange{cachedAnswer}, probe,
                         mlir::ValueRange{});

  builder.setInsertionPointToStart(probe);
  mlir::Value familySlot =
      LLVM::AllocaOp::create(builder, loc, ptr, i32, one, /*alignment=*/8);
  mlir::Value lenSlot =
      LLVM::AllocaOp::create(builder, loc, ptr, i64, one, /*alignment=*/8);
  LLVM::StoreOp::create(builder, loc, constI32(builder, loc, 0), familySlot);
  LLVM::StoreOp::create(builder, loc, constI64(builder, loc, 4), lenSlot);
  mlir::Value namePtr = LLVM::AddressOfOp::create(builder, loc, nameGlobal);
  mlir::Value null = LLVM::ZeroOp::create(builder, loc, ptr);
  mlir::Value rc =
      LLVM::CallOp::create(builder, loc, sysctlFn,
                           mlir::ValueRange{namePtr, familySlot, lenSlot, null,
                                            constI64(builder, loc, 0)})
          .getResult();
  mlir::Value sysctlOk = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::eq, rc, constI32(builder, loc, 0));
  LLVM::CondBrOp::create(builder, loc, sysctlOk, checkFamily,
                         mlir::ValueRange{}, store, mlir::ValueRange{one});

  builder.setInsertionPointToStart(checkFamily);
  mlir::Value family = LLVM::LoadOp::create(builder, loc, i32, familySlot);
  mlir::Value known = LLVM::ConstantOp::create(builder, loc, builder.getI1Type(),
                                               builder.getBoolAttr(false));
  for (std::uint32_t candidate : kKnownCpuFamilies) {
    mlir::Value match = LLVM::ICmpOp::create(
        builder, loc, LLVM::ICmpPredicate::eq, family,
        constI32(builder, loc, static_cast<std::int32_t>(candidate)));
    known = LLVM::OrOp::create(builder, loc, known, match);
  }
  LLVM::CondBrOp::create(builder, loc, known, doFork, mlir::ValueRange{}, store,
                         mlir::ValueRange{one});

  builder.setInsertionPointToStart(doFork);
  mlir::Value pid =
      LLVM::CallOp::create(builder, loc, forkFn, mlir::ValueRange{})
          .getResult();
  mlir::Value isChild = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::eq, pid, constI32(builder, loc, 0));
  LLVM::CondBrOp::create(builder, loc, isChild, child, mlir::ValueRange{},
                         parent, mlir::ValueRange{pid});

  builder.setInsertionPointToStart(child);
  emitAmxImm(builder, loc, kAmxSetClr, 0);
  emitAmxImm(builder, loc, kAmxSetClr, 1);
  LLVM::CallOp::create(builder, loc, exitFn,
                       mlir::ValueRange{constI32(builder, loc, 0)});
  LLVM::UnreachableOp::create(builder, loc);

  builder.setInsertionPointToStart(parent);
  mlir::Value childPid = parent->getArgument(0);
  mlir::Value forkFailed = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::slt, childPid,
      constI32(builder, loc, 0));
  mlir::Block *wait = fn.addBlock();
  LLVM::CondBrOp::create(builder, loc, forkFailed, store,
                         mlir::ValueRange{one}, wait, mlir::ValueRange{});

  builder.setInsertionPointToStart(wait);
  mlir::Value statusSlot =
      LLVM::AllocaOp::create(builder, loc, ptr, i32, one, /*alignment=*/8);
  LLVM::StoreOp::create(builder, loc, constI32(builder, loc, 0), statusSlot);
  LLVM::CallOp::create(builder, loc, waitpidFn,
                       mlir::ValueRange{childPid, statusSlot,
                                        constI32(builder, loc, 0)});
  mlir::Value status = LLVM::LoadOp::create(builder, loc, i32, statusSlot);
  // WIFEXITED(s) && WEXITSTATUS(s) == 0  =>  (s & 0xff) == 0
  mlir::Value low =
      LLVM::AndOp::create(builder, loc, status, constI32(builder, loc, 0xff));
  mlir::Value exitedClean = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::eq, low, constI32(builder, loc, 0));
  mlir::Value two = constI64(builder, loc, 2);
  mlir::Value answer =
      LLVM::SelectOp::create(builder, loc, exitedClean, two, one);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{answer}, store);

  builder.setInsertionPointToStart(store);
  LLVM::StoreOp::create(builder, loc, store->getArgument(0), globalPtr);
  mlir::Value result =
      LLVM::SubOp::create(builder, loc, store->getArgument(0), one);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{result}, done);

  builder.setInsertionPointToStart(done);
  LLVM::ReturnOp::create(builder, loc, done->getArgument(0));
}

//===----------------------------------------------------------------------===//
// f32 matmul kernel
//===----------------------------------------------------------------------===//

// Verified on M4 Max (hw.cpufamily 0x17d5b93a) against known inputs:
//
//   FMA32(operand):  Z[j*4 + (Zbase & 3)][i] += X[i] * Y[j]      i,j = 0..15
//
//   operand bits  9:0   X offset in bytes into the X registers
//                19:10  Y offset in bytes into the Y registers
//                21:20  Z row base (f32 keeps only the low two bits: 4 tiles)
//
// The kernel leaves every one of those operand fields at zero: it reloads X and
// Y from memory each reduction step rather than indexing within them, and picks
// the Z row through LDZ/STZ instead.
//
// The X/Y registers are 8 x 64 bytes each; a 64-byte slice is 16 f32.
constexpr unsigned kAmxTile = 16;
// f32 spreads one tile's rows over Z at a stride of four.
constexpr unsigned kAmxF32ZRowStride = 4;

struct AmxMatmulShape {
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value out;
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
};

bool isContiguousRowMajorF32(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      !type.getElementType().isF32())
    return false;
  llvm::SmallVector<std::int64_t, 2> strides;
  std::int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)))
    return false;
  // The row stride may be any static value (a chunked view keeps the parent's
  // pitch); only the innermost run has to be contiguous, since LDX/LDY/STZ
  // move 64 contiguous bytes.
  return strides.size() == 2 && strides[1] == 1 &&
         !mlir::ShapedType::isDynamic(strides[0]);
}

std::optional<AmxMatmulShape> matchAmxMatmul(mlir::linalg::MatmulOp matmul) {
  // The AMX kernel transposes A itself and so needs the default [M][K] layout.
  if (matmul->getNumResults() != 0 || matmul.getNumDpsInputs() != 2 ||
      matmul.getNumDpsInits() != 1 || !hasDefaultMatmulMaps(matmul))
    return std::nullopt;
  mlir::Value lhs = matmul.getDpsInputOperand(0)->get();
  mlir::Value rhs = matmul.getDpsInputOperand(1)->get();
  mlir::Value out = matmul.getDpsInitOperand(0)->get();
  if (!isContiguousRowMajorF32(lhs) || !isContiguousRowMajorF32(rhs) ||
      !isContiguousRowMajorF32(out))
    return std::nullopt;

  auto lhsType = mlir::cast<mlir::MemRefType>(lhs.getType());
  auto rhsType = mlir::cast<mlir::MemRefType>(rhs.getType());
  auto outType = mlir::cast<mlir::MemRefType>(out.getType());
  std::int64_t m = lhsType.getDimSize(0);
  std::int64_t k = lhsType.getDimSize(1);
  std::int64_t n = rhsType.getDimSize(1);
  if (rhsType.getDimSize(0) != k || outType.getDimSize(0) != m ||
      outType.getDimSize(1) != n)
    return std::nullopt;
  // One tile per step, no masking: keep to exact multiples until the kernel
  // grows a remainder path.
  if (m % kAmxTile || n % kAmxTile || m <= 0 || n <= 0 || k <= 0)
    return std::nullopt;
  return AmxMatmulShape{lhs, rhs, out, m, n, k};
}

// Byte address of `memref[row][col]`, as an i64 for the instruction operand.
mlir::Value elementAddress(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value memref, mlir::Value row, mlir::Value col,
                           std::int64_t rowStride) {
  auto i64 = builder.getI64Type();
  mlir::Value base = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
      builder, loc, memref);
  mlir::Value baseI64 =
      mlir::arith::IndexCastOp::create(builder, loc, i64, base);
  mlir::Value rowI64 = mlir::arith::IndexCastOp::create(builder, loc, i64, row);
  mlir::Value colI64 = mlir::arith::IndexCastOp::create(builder, loc, i64, col);
  mlir::Value stride =
      mlir::arith::ConstantIntOp::create(builder, loc, i64, rowStride);
  mlir::Value linear = mlir::arith::AddIOp::create(
      builder, loc, mlir::arith::MulIOp::create(builder, loc, rowI64, stride),
      colI64);
  mlir::Value elementBytes =
      mlir::arith::ConstantIntOp::create(builder, loc, i64, 4);
  mlir::Value byteOffset =
      mlir::arith::MulIOp::create(builder, loc, linear, elementBytes);
  return mlir::arith::AddIOp::create(builder, loc, baseI64, byteOffset);
}

std::int64_t rowStrideOf(mlir::Value value) {
  auto type = mlir::cast<mlir::MemRefType>(value.getType());
  llvm::SmallVector<std::int64_t, 2> strides;
  std::int64_t offset = 0;
  (void)type.getStridesAndOffset(strides, offset);
  return strides[0];
}

// Transposed copy of A into Ap[k][m]: the kernel needs A's column k as 16
// contiguous f32, and LDY only moves contiguous bytes.
mlir::Value packLhsTransposed(mlir::OpBuilder &builder, mlir::Location loc,
                              const AmxMatmulShape &shape) {
  auto packedType = mlir::MemRefType::get(
      {shape.k, shape.m}, builder.getF32Type());
  mlir::Value packed =
      mlir::memref::AllocOp::create(builder, loc, packedType,
                                    mlir::ValueRange{},
                                    builder.getI64IntegerAttr(128))
          .getResult();
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Value kUpper = mlir::arith::ConstantIndexOp::create(builder, loc, shape.k);
  mlir::Value mUpper = mlir::arith::ConstantIndexOp::create(builder, loc, shape.m);
  auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kUpper, one);
  {
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(kLoop.getBody());
    auto mLoop = mlir::scf::ForOp::create(builder, loc, zero, mUpper, one);
    mlir::OpBuilder::InsertionGuard g2(builder);
    builder.setInsertionPointToStart(mLoop.getBody());
    mlir::Value v = mlir::memref::LoadOp::create(
        builder, loc, shape.lhs,
        mlir::ValueRange{mLoop.getInductionVar(), kLoop.getInductionVar()});
    mlir::memref::StoreOp::create(
        builder, loc, v, packed,
        mlir::ValueRange{kLoop.getInductionVar(), mLoop.getInductionVar()});
  }
  return packed;
}

// C[m0..m0+16][n0..n0+16] += A * B for one 16x16 tile.
//
// X holds B's row slice (contiguous) and Y holds the packed A column, so
// Z[j*4][i] lands as C[m0+j][n0+i] -- a row of C, which STZ can write back
// directly. Feeding A through X instead would transpose the result and force a
// scatter.
void emitAmxTile(mlir::OpBuilder &builder, mlir::Location loc,
                 const AmxMatmulShape &shape, mlir::Value packedLhs,
                 mlir::Value m0, mlir::Value n0, mlir::Value zeroSeed) {
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  std::int64_t outStride = rowStrideOf(shape.out);
  std::int64_t rhsStride = rowStrideOf(shape.rhs);
  std::int64_t packedStride = shape.m;

  // Seed the accumulator. FMA32 always accumulates, so the Z tile has to start
  // at whatever the contraction's initial value is: C itself, or zero when the
  // zero-init elision dropped C's fill and made "ignore C" the contract.
  for (unsigned j = 0; j < kAmxTile; ++j) {
    mlir::Value addr;
    if (zeroSeed) {
      addr = zeroSeed;
    } else {
      mlir::Value row = mlir::arith::AddIOp::create(
          builder, loc, m0,
          mlir::arith::ConstantIndexOp::create(builder, loc, j));
      addr = elementAddress(builder, loc, shape.out, row, n0, outStride);
    }
    mlir::Value zrow = mlir::arith::ConstantIntOp::create(
        builder, loc, builder.getI64Type(),
        static_cast<std::int64_t>(j * kAmxF32ZRowStride) << 56);
    emitAmxGpr(builder, loc, kAmxLdz,
               mlir::arith::OrIOp::create(builder, loc, addr, zrow));
  }

  mlir::Value kUpper =
      mlir::arith::ConstantIndexOp::create(builder, loc, shape.k);
  auto kLoop = mlir::scf::ForOp::create(builder, loc, zero, kUpper, one);
  {
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(kLoop.getBody());
    mlir::Value k = kLoop.getInductionVar();
    mlir::Value xAddr =
        elementAddress(builder, loc, shape.rhs, k, n0, rhsStride);
    mlir::Value yAddr =
        elementAddress(builder, loc, packedLhs, k, m0, packedStride);
    emitAmxGpr(builder, loc, kAmxLdx, xAddr);
    emitAmxGpr(builder, loc, kAmxLdy, yAddr);
    emitAmxGpr(builder, loc, kAmxFma32,
               mlir::arith::ConstantIntOp::create(builder, loc,
                                                  builder.getI64Type(), 0));
  }

  for (unsigned j = 0; j < kAmxTile; ++j) {
    mlir::Value row = mlir::arith::AddIOp::create(
        builder, loc, m0,
        mlir::arith::ConstantIndexOp::create(builder, loc, j));
    mlir::Value addr =
        elementAddress(builder, loc, shape.out, row, n0, outStride);
    mlir::Value zrow = mlir::arith::ConstantIntOp::create(
        builder, loc, builder.getI64Type(),
        static_cast<std::int64_t>(j * kAmxF32ZRowStride) << 56);
    emitAmxGpr(builder, loc, kAmxStz,
               mlir::arith::OrIOp::create(builder, loc, addr, zrow));
  }
}

void emitAmxMatmul(mlir::OpBuilder &builder, mlir::Location loc,
                   const AmxMatmulShape &shape, bool zeroInit) {
  mlir::Value packedLhs = packLhsTransposed(builder, loc, shape);

  // One zeroed 64-byte line, reloaded into each Z row of a tile. Cheaper than
  // zero-filling C, which would cost a full O(M*N) pass the contraction is
  // about to overwrite anyway.
  mlir::Value zeroSeed;
  mlir::Value seed;
  if (zeroInit) {
    auto seedType = mlir::MemRefType::get({kAmxTile}, builder.getF32Type());
    seed = mlir::memref::AllocOp::create(builder, loc, seedType,
                                         mlir::ValueRange{},
                                         builder.getI64IntegerAttr(128))
               .getResult();
    mlir::Value fzero = mlir::arith::ConstantFloatOp::create(
        builder, loc, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::linalg::FillOp::create(builder, loc, mlir::ValueRange{fzero},
                                 mlir::ValueRange{seed});
    mlir::Value base = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
        builder, loc, seed);
    zeroSeed = mlir::arith::IndexCastOp::create(builder, loc,
                                                builder.getI64Type(), base);
  }

  // AMX state must not outlive the region: enter once, leave on the single
  // exit. Nothing between SET and CLR may call out, which holds here -- the
  // body is loads, stores and AMX words only.
  emitAmxImm(builder, loc, kAmxSetClr, 0);
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value tile =
      mlir::arith::ConstantIndexOp::create(builder, loc, kAmxTile);
  mlir::Value mUpper =
      mlir::arith::ConstantIndexOp::create(builder, loc, shape.m);
  mlir::Value nUpper =
      mlir::arith::ConstantIndexOp::create(builder, loc, shape.n);
  auto nLoop = mlir::scf::ForOp::create(builder, loc, zero, nUpper, tile);
  {
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(nLoop.getBody());
    auto mLoop = mlir::scf::ForOp::create(builder, loc, zero, mUpper, tile);
    mlir::OpBuilder::InsertionGuard g2(builder);
    builder.setInsertionPointToStart(mLoop.getBody());
    emitAmxTile(builder, loc, shape, packedLhs, mLoop.getInductionVar(),
                nLoop.getInductionVar(), zeroSeed);
  }
  emitAmxImm(builder, loc, kAmxSetClr, 1);
  mlir::memref::DeallocOp::create(builder, loc, packedLhs);
  if (seed)
    mlir::memref::DeallocOp::create(builder, loc, seed);
}

class MatmulAMXLoweringPass
    : public mlir::PassWrapper<MatmulAMXLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulAMXLoweringPass)

  explicit MatmulAMXLoweringPass(py::TensorLoweringTarget target = {})
      : target(target) {}

  llvm::StringRef getArgument() const final {
    return "lython-apple-amx-matmul-lowering";
  }
  llvm::StringRef getDescription() const final {
    return "lower primitive f32 linalg.matmul to a run-time-guarded Apple AMX "
           "kernel";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<std::pair<mlir::linalg::MatmulOp, AmxMatmulShape>, 8>
        candidates;
    module.walk([&](mlir::linalg::MatmulOp matmul) {
      if (std::optional<AmxMatmulShape> shape = matchAmxMatmul(matmul))
        candidates.push_back({matmul, *shape});
    });
    if (candidates.empty())
      return;

    mlir::OpBuilder builder(&getContext());
    auto i64 = builder.getI64Type();
    // Declared, not defined: the body needs libc and reserved instruction
    // words, which only exist once the module is LLVM dialect.
    if (!module.lookupSymbol(kMatrixBackendName)) {
      mlir::OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(module.getBody());
      auto decl = mlir::func::FuncOp::create(
          builder, builder.getUnknownLoc(), kMatrixBackendName,
          builder.getFunctionType({}, {i64}));
      decl.setPrivate();
    }

    for (auto &[matmul, shape] : candidates) {
      if (!matmul->getBlock())
        continue;
      mlir::Location loc = matmul.getLoc();
      builder.setInsertionPoint(matmul);
      mlir::Value backend =
          mlir::func::CallOp::create(builder, loc, kMatrixBackendName,
                                     mlir::TypeRange{i64}, mlir::ValueRange{})
              .getResult(0);
      mlir::Value one =
          mlir::arith::ConstantIntOp::create(builder, loc, i64, 1);
      mlir::Value useAmx = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, backend, one);
      auto ifOp = mlir::scf::IfOp::create(builder, loc, useAmx,
                                          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(ifOp.thenBlock());
        bool zeroInit = matmul->hasAttr(kMatmulZeroInitAttr) ||
                        matmul->hasAttr(kMatmulZeroInitFirstReductionAttr);
        emitAmxMatmul(builder, loc, shape, zeroInit);
      }
      // The machine may answer no: keep the portable contraction on the else
      // path for the later tiling/vector passes to specialize.
      {
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(ifOp.elseBlock());
        builder.clone(*matmul.getOperation());
      }
      matmul.erase();
    }
  }

private:
  py::TensorLoweringTarget target;
};

} // namespace

bool usesAMX(const py::TensorLoweringTarget &target) {
  return target.usesAppleAMX();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMatmulAMXLoweringPass(py::TensorLoweringTarget target) {
  return std::make_unique<MatmulAMXLoweringPass>(target);
}

mlir::LogicalResult materializeMatrixBackendProbe(mlir::ModuleOp module) {
  auto placeholder = module.lookupSymbol<LLVM::LLVMFuncOp>(kMatrixBackendName);
  if (!placeholder || !placeholder.isExternal())
    return mlir::success();

  // The declaration the kernel calls becomes the definition here, where libc
  // calls and reserved instruction words can be spelled.
  placeholder.erase();
  mlir::OpBuilder builder(module.getContext());
  buildMatrixBackend(module, builder);
  return mlir::success();
}

} // namespace py::lowering::arch::apple
