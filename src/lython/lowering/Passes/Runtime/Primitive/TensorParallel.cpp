#include "TensorParallel.h"

#include "TensorGemm.h"
#include "TensorSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::lowering {
namespace {

// Dispatch is only worth a fork-join for contractions whose work dwarfs the
// thread spawn cost (~tens of microseconds per worker): 512^3 and up. The
// chunk count is a compile-time bound; the runtime clamps further via
// LYTHON_NUM_THREADS and the trip count.
constexpr std::uint64_t kMatmulParallelMinWork = 33ull << 20;
constexpr int64_t kMatmulParallelMaxChunks = 8;
// Floor on a chunk's rows, whatever kernel consumes it: below this the fork-join
// and the B panel each worker re-reads stop being amortised.
constexpr int64_t kMatmulParallelMinChunkRows = 64;
// The packed schedule additionally needs a chunk its macro tile can divide --
// selectPackedMTile picks MC from the divisors of the chunk's rows, and its
// register tile is 2x32.
//
// Why the SME kernel is exempt: it sweeps M itself in 2*VL row blocks and masks
// the partial one, so any chunk height is correct for it. Requiring the divisor
// anyway is what left every M without one running single-threaded -- M=1000,
// 1500 and 2000 emitted no workers at all, and measured 0.20-0.27x of Accelerate
// on a 16-core machine against 0.78-0.93x for shapes that did get cut.
constexpr int64_t kMatmulParallelPackedRowQuantum = 64;

constexpr llvm::StringLiteral kParallelDispatchAttr =
    kParallelDispatchAttrName;
constexpr llvm::StringLiteral kParallelBodyAttr{"ly.parallel.body"};

bool staticIdentityLike(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  return type && type.hasStaticShape() &&
         isPrimitiveElementType(type.getElementType());
}

// Equal chunks are structural, not a preference: the cut is one loop over one
// matmul on a subview, so every chunk has to be the same static height.
// `quantum` is the extra divisor the consuming kernel needs, or 1 if it needs
// none.
std::optional<int64_t> selectChunkCount(int64_t rows, int64_t quantum) {
  int64_t bound = std::min<int64_t>(kMatmulParallelMaxChunks,
                                    rows / kMatmulParallelMinChunkRows);
  for (int64_t chunks = bound; chunks >= 2; --chunks)
    if (rows % chunks == 0 && (rows / chunks) % quantum == 0)
      return chunks;
  return std::nullopt;
}

// One operand of a chunked op: which operand, and which of its dimensions the
// chunk index cuts. The dimension is not always the leading one -- a
// transposed LHS stores A as [K][M], so its M lives in dimension 1.
struct ChunkedOperand {
  unsigned index;
  unsigned dim;
};

// Cut `op` into `chunks` equal M ranges inside a loop tagged for parallel
// dispatch. Operands not listed are read whole by every chunk.
//
// Why this is matmul-only today: the chunk views carry a dynamic offset, and
// only the matmul path keeps them away from the affine vectorizer, which
// rejects a non-identity layout map outright. An elementwise op lowers through
// linalg -> affine loops -> affine vectorizer and would hit exactly that.
void chunkDimension(mlir::Operation *op, int64_t chunks, int64_t rows,
                    llvm::ArrayRef<ChunkedOperand> chunkedOperands,
                    mlir::OpBuilder &builder) {
  mlir::Location loc = op->getLoc();
  builder.setInsertionPoint(op);
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value upper = mlir::arith::ConstantIndexOp::create(builder, loc, chunks);
  mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  auto loop = mlir::scf::ForOp::create(builder, loc, zero, upper, one);
  loop->setAttr(kParallelDispatchAttr, builder.getUnitAttr());

  builder.setInsertionPointToStart(loop.getBody());
  mlir::Value rowsValue =
      mlir::arith::ConstantIndexOp::create(builder, loc, rows);
  mlir::Value rowStart = mlir::arith::MulIOp::create(
      builder, loc, loop.getInductionVar(), rowsValue);

  llvm::SmallVector<mlir::Value, 3> views;
  for (const ChunkedOperand &operand : chunkedOperands) {
    mlir::Value source = op->getOperand(operand.index);
    auto type = mlir::cast<mlir::MemRefType>(source.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      bool cut = dim == static_cast<int64_t>(operand.dim);
      offsets.push_back(cut ? mlir::OpFoldResult(rowStart)
                            : mlir::OpFoldResult(builder.getIndexAttr(0)));
      sizes.push_back(builder.getIndexAttr(cut ? rows : type.getDimSize(dim)));
      strides.push_back(builder.getIndexAttr(1));
    }
    views.push_back(mlir::memref::SubViewOp::create(builder, loc, source,
                                                    offsets, sizes, strides)
                        .getResult());
  }

  mlir::Operation *cloned = builder.clone(*op);
  for (auto [view, operand] : llvm::zip(views, chunkedOperands))
    cloned->setOperand(operand.index, view);
  op->erase();
}

class MatmulParallelChunkPass
    : public mlir::PassWrapper<MatmulParallelChunkPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulParallelChunkPass)

  llvm::StringRef getArgument() const final {
    return "lython-matmul-parallel-chunk";
  }
  llvm::StringRef getDescription() const final {
    return "split large matmuls into row chunks tagged for parallel dispatch";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    struct ChunkPlan {
      mlir::Operation *op;
      int64_t chunks;
      int64_t rows;
      llvm::SmallVector<ChunkedOperand, 3> operands;
    };
    llvm::SmallVector<ChunkPlan, 8> plans;

    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      // Both LHS layouts chunk the same way; only where M sits differs.
      bool lhsTransposed = hasTransposedLhsMatmulMaps(matmul);
      if (matmul->getNumResults() != 0 ||
          (!hasDefaultMatmulMaps(matmul) && !lhsTransposed))
        return;
      mlir::Value lhs = matmul.getDpsInputOperand(0)->get();
      mlir::Value rhs = matmul.getDpsInputOperand(1)->get();
      mlir::Value out = matmul.getDpsInitOperand(0)->get();
      if (!staticIdentityLike(lhs) || !staticIdentityLike(rhs) ||
          !staticIdentityLike(out))
        return;
      auto lhsType = mlir::cast<mlir::MemRefType>(lhs.getType());
      auto rhsType = mlir::cast<mlir::MemRefType>(rhs.getType());
      if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
        return;
      // [M][K] keeps M in dimension 0; [K][M] keeps it in dimension 1.
      unsigned lhsMDim = lhsTransposed ? 1 : 0;
      int64_t m = lhsType.getDimSize(lhsMDim);
      int64_t k = lhsType.getDimSize(lhsTransposed ? 0 : 1);
      std::uint64_t work = 1;
      for (int64_t dim : {m, rhsType.getDimSize(1), k}) {
        if (dim <= 0)
          return;
        work *= static_cast<std::uint64_t>(dim);
      }
      if (work < kMatmulParallelMinWork)
        return;
      // A transposed LHS is the mark the SME kernel will take this one, and
      // that kernel masks its own partial row block -- so only the packed
      // schedule's macro tile imposes a divisor here.
      std::optional<int64_t> chunks = selectChunkCount(
          m, lhsTransposed ? 1 : kMatmulParallelPackedRowQuantum);
      if (!chunks)
        return;
      // The LHS's M and the OUT's rows are cut; the RHS panel is read whole by
      // every chunk.
      plans.push_back(ChunkPlan{matmul,
                                *chunks,
                                m / *chunks,
                                {ChunkedOperand{0, lhsMDim},
                                 ChunkedOperand{2, 0}}});
    });

    mlir::OpBuilder builder(&getContext());
    for (const ChunkPlan &plan : plans)
      chunkDimension(plan.op, plan.chunks, plan.rows, plan.operands, builder);
  }
};

class ParallelLoopOutliningPass
    : public mlir::PassWrapper<ParallelLoopOutliningPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelLoopOutliningPass)

  llvm::StringRef getArgument() const final {
    return "lython-parallel-loop-outlining";
  }
  llvm::StringRef getDescription() const final {
    return "outline parallel-dispatch loops into per-worker body functions";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    llvm::SmallVector<mlir::scf::ForOp, 8> loops;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (loop->hasAttr(kParallelDispatchAttr) && loop.getNumResults() == 0)
        loops.push_back(loop);
    });

    mlir::OpBuilder builder(&getContext());
    unsigned counter = 0;
    for (mlir::scf::ForOp loop : loops) {
      llvm::SetVector<mlir::Value> captures;
      mlir::getUsedValuesDefinedAbove(loop.getRegion(), captures);

      mlir::Location loc = loop.getLoc();
      mlir::Type indexType = builder.getIndexType();
      llvm::SmallVector<mlir::Type, 8> argTypes{indexType, indexType,
                                                indexType};
      for (mlir::Value capture : captures)
        argTypes.push_back(capture.getType());

      std::string name =
          ("__lython_parallel_body_" + llvm::Twine(counter++)).str();
      builder.setInsertionPointToEnd(getOperation().getBody());
      auto body = mlir::func::FuncOp::create(
          builder, loc, name, builder.getFunctionType(argTypes, {}));
      body.setPrivate();
      body->setAttr(kParallelBodyAttr, builder.getUnitAttr());

      mlir::Block *entry = body.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      auto inner = mlir::scf::ForOp::create(builder, loc, entry->getArgument(0),
                                            entry->getArgument(1),
                                            entry->getArgument(2));
      mlir::IRMapping mapping;
      mapping.map(loop.getInductionVar(), inner.getInductionVar());
      for (auto [index, capture] : llvm::enumerate(captures))
        mapping.map(capture, entry->getArgument(index + 3));
      builder.setInsertionPointToStart(inner.getBody());
      for (mlir::Operation &op : loop.getBody()->without_terminator())
        builder.clone(op, mapping);
      builder.setInsertionPointToEnd(entry);
      mlir::func::ReturnOp::create(builder, loc);

      builder.setInsertionPoint(loop);
      llvm::SmallVector<mlir::Value, 8> operands{
          loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
      operands.append(captures.begin(), captures.end());
      mlir::func::CallOp::create(builder, loc, body, operands);
      loop.erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// LLVM-level dispatch
//===----------------------------------------------------------------------===//

namespace LLVM = mlir::LLVM;

constexpr llvm::StringLiteral kParallelForName{"LyParallelFor"};
constexpr llvm::StringLiteral kParallelApplyName{"LyParallelApply"};
constexpr llvm::StringLiteral kParallelWorkerName{"LyParallelWorker"};
constexpr llvm::StringLiteral kParallelThreadsName{"LyParallelThreads"};
constexpr llvm::StringLiteral kThreadsGlobalName{"__lython_parallel_threads"};
constexpr llvm::StringLiteral kThreadsEnvName{"__lython_parallel_env"};
constexpr llvm::StringLiteral kThreadsEnvSetName{"__lython_parallel_env_set"};
constexpr llvm::StringLiteral kAttrStateName{"__lython_dispatch_attr_state"};
constexpr llvm::StringLiteral kAttrInitPtrName{"__lython_attr_init_fn"};
constexpr llvm::StringLiteral kAttrSetParPtrName{"__lython_attr_setpar_fn"};
constexpr llvm::StringLiteral kAttrDestroyPtrName{"__lython_attr_destroy_fn"};
constexpr llvm::StringLiteral kAttrApplyPtrName{"__lython_attr_apply_fn"};
constexpr llvm::StringLiteral kClusterWidthName{"__lython_cluster_width"};
constexpr llvm::StringLiteral kAttrWidthFnName{"LyDispatchAttrWidth"};
constexpr llvm::StringLiteral kParallelApplyAttrName{"LyParallelApplyAttr"};
// Default worker count: online CPUs, which the chunk count caps in practice.
// Why not a small fixed pool: a shared matrix unit (Apple's SME sits per
// P-cluster) only gets a second unit's worth of throughput once the scheduler
// spreads the workers across clusters, and it will not do that for a couple of
// threads -- 4 workers measured 35% slower than 8 on 1024^3 f32.
// LYTHON_NUM_THREADS overrides.
constexpr int64_t kFallbackParallelThreads = 8;
constexpr int64_t kMaxParallelThreads = 64;
// _SC_NPROCESSORS_ONLN. The value is libc-defined, not ABI-shared.
constexpr int64_t kDarwinScNprocessorsOnln = 58;
constexpr int64_t kLinuxScNprocessorsOnln = 84;

std::optional<int64_t> onlineProcessorsSysconfName(mlir::ModuleOp module) {
  auto triple = module->getAttrOfType<mlir::StringAttr>("ly.target.triple");
  if (!triple)
    return std::nullopt;
  if (triple.getValue().contains("darwin"))
    return kDarwinScNprocessorsOnln;
  if (triple.getValue().contains("linux"))
    return kLinuxScNprocessorsOnln;
  return std::nullopt;
}

LLVM::LLVMFuncOp ensureFuncDecl(mlir::ModuleOp module, mlir::OpBuilder &builder,
                                llvm::StringRef name,
                                LLVM::LLVMFunctionType type) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  return LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(), name,
                                  type);
}

mlir::Value constI64(mlir::OpBuilder &builder, mlir::Location loc,
                     int64_t value) {
  return LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                  builder.getI64IntegerAttr(value));
}

// struct WorkerArg { void (*fn)(i64, i64, ptr); ptr ctx; i64 begin; i64 end; }
LLVM::LLVMStructType workerArgType(mlir::MLIRContext *context) {
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = mlir::IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(context, {ptr, ptr, i64, i64});
}

LLVM::LLVMFunctionType bodyFnType(mlir::MLIRContext *context) {
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = mlir::IntegerType::get(context, 64);
  return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                     {i64, i64, ptr});
}

// LyParallelThreads: cached LYTHON_NUM_THREADS (default 4, clamped to
// [1, 64]). The unsynchronized global is benign: initialization is
// idempotent and every racing writer stores the same value.
void buildParallelThreads(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto i64 = builder.getI64Type();
  auto i32 = builder.getI32Type();
  auto ptr = LLVM::LLVMPointerType::get(context);
  mlir::Location loc = builder.getUnknownLoc();

  builder.setInsertionPointToEnd(module.getBody());
  auto global = LLVM::GlobalOp::create(
      builder, loc, i64, /*isConstant=*/false, LLVM::Linkage::Internal,
      kThreadsGlobalName, builder.getI64IntegerAttr(0));

  builder.setInsertionPointToEnd(module.getBody());
  // Whether LYTHON_NUM_THREADS was present at all: the dispatch fast path
  // steps aside whenever the user pinned a thread count, so measurements pin
  // exactly what they ask for.
  LLVM::GlobalOp::create(builder, loc, i64, /*isConstant=*/false,
                         LLVM::Linkage::Internal, kThreadsEnvSetName,
                         builder.getI64IntegerAttr(0));

  builder.setInsertionPointToEnd(module.getBody());
  auto envType = LLVM::LLVMArrayType::get(builder.getI8Type(), 19);
  auto envGlobal = LLVM::GlobalOp::create(
      builder, loc, envType, /*isConstant=*/true, LLVM::Linkage::Internal,
      kThreadsEnvName,
      builder.getStringAttr(llvm::StringRef("LYTHON_NUM_THREADS\0", 19)));

  auto getenvFn = ensureFuncDecl(
      module, builder, "getenv", LLVM::LLVMFunctionType::get(ptr, {ptr}));
  auto strtolFn = ensureFuncDecl(
      module, builder, "strtol",
      LLVM::LLVMFunctionType::get(i64, {ptr, ptr, i32}));
  std::optional<int64_t> sysconfName = onlineProcessorsSysconfName(module);
  LLVM::LLVMFuncOp sysconfFn;
  if (sysconfName)
    sysconfFn = ensureFuncDecl(module, builder, "sysconf",
                               LLVM::LLVMFunctionType::get(i64, {i32}));

  builder.setInsertionPointToEnd(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(builder, loc, kParallelThreadsName,
                                     LLVM::LLVMFunctionType::get(i64, {}),
                                     LLVM::Linkage::Internal);
  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::Block *slow = fn.addBlock();
  mlir::Block *haveEnv = fn.addBlock();
  haveEnv->addArgument(ptr, loc);
  mlir::Block *store = fn.addBlock();
  store->addArgument(i64, loc);
  mlir::Block *done = fn.addBlock();
  done->addArgument(i64, loc);

  builder.setInsertionPointToStart(entry);
  mlir::Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
  mlir::Value cached = LLVM::LoadOp::create(builder, loc, i64, globalPtr);
  mlir::Value zero = constI64(builder, loc, 0);
  mlir::Value isInit = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::ne, cached, zero);
  LLVM::CondBrOp::create(builder, loc, isInit, done, mlir::ValueRange{cached},
                         slow, mlir::ValueRange{});

  builder.setInsertionPointToStart(slow);
  mlir::Value envPtr = LLVM::AddressOfOp::create(builder, loc, envGlobal);
  mlir::Value env = LLVM::CallOp::create(builder, loc, getenvFn,
                                         mlir::ValueRange{envPtr})
                        .getResult();
  mlir::Value null = LLVM::ZeroOp::create(builder, loc, ptr);
  mlir::Value hasEnv = LLVM::ICmpOp::create(builder, loc,
                                            LLVM::ICmpPredicate::ne, env, null);
  mlir::Value fallback;
  if (sysconfName) {
    mlir::Value name = LLVM::ConstantOp::create(
        builder, loc, i32, builder.getI32IntegerAttr(*sysconfName));
    mlir::Value online =
        LLVM::CallOp::create(builder, loc, sysconfFn, mlir::ValueRange{name})
            .getResult();
    // sysconf returns -1 when the name is unsupported.
    mlir::Value onlineOk = LLVM::ICmpOp::create(
        builder, loc, LLVM::ICmpPredicate::sgt, online,
        constI64(builder, loc, 0));
    fallback = LLVM::SelectOp::create(
        builder, loc, onlineOk, online,
        constI64(builder, loc, kFallbackParallelThreads));
  } else {
    fallback = constI64(builder, loc, kFallbackParallelThreads);
  }
  LLVM::CondBrOp::create(builder, loc, hasEnv, haveEnv, mlir::ValueRange{env},
                         store, mlir::ValueRange{fallback});

  builder.setInsertionPointToStart(haveEnv);
  {
    auto flagGlobal = module.lookupSymbol<LLVM::GlobalOp>(kThreadsEnvSetName);
    mlir::Value flagPtr = LLVM::AddressOfOp::create(builder, loc, flagGlobal);
    LLVM::StoreOp::create(builder, loc, constI64(builder, loc, 1), flagPtr);
  }
  mlir::Value base = LLVM::ConstantOp::create(builder, loc, i32,
                                              builder.getI32IntegerAttr(10));
  mlir::Value parsed =
      LLVM::CallOp::create(builder, loc, strtolFn,
                           mlir::ValueRange{haveEnv->getArgument(0), null,
                                            base})
          .getResult();
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{parsed}, store);

  // Clamp on the way to the cache so both the parsed and the probed value land
  // in range: a machine with more online CPUs than the worker cap, and a
  // negative or zero LYTHON_NUM_THREADS, both reach here.
  builder.setInsertionPointToStart(store);
  mlir::Value raw = store->getArgument(0);
  mlir::Value one = constI64(builder, loc, 1);
  mlir::Value cap = constI64(builder, loc, kMaxParallelThreads);
  mlir::Value belowOne =
      LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::slt, raw, one);
  mlir::Value atLeastOne =
      LLVM::SelectOp::create(builder, loc, belowOne, one, raw);
  mlir::Value aboveCap = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::sgt, atLeastOne, cap);
  mlir::Value clamped =
      LLVM::SelectOp::create(builder, loc, aboveCap, cap, atLeastOne);
  LLVM::StoreOp::create(builder, loc, clamped, globalPtr);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{clamped}, done);

  builder.setInsertionPointToStart(done);
  LLVM::ReturnOp::create(builder, loc, done->getArgument(0));
}

// LyParallelWorker(ptr arg) -> ptr: unpack {fn, ctx, begin, end} and run.
void buildParallelWorker(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = builder.getI64Type();
  mlir::Location loc = builder.getUnknownLoc();
  auto argType = workerArgType(context);

  builder.setInsertionPointToEnd(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(builder, loc, kParallelWorkerName,
                                     LLVM::LLVMFunctionType::get(ptr, {ptr}),
                                     LLVM::Linkage::Internal);
  mlir::Block *entry = fn.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  mlir::Value arg = entry->getArgument(0);
  auto field = [&](int64_t index, mlir::Type type) -> mlir::Value {
    mlir::Value fieldPtr = LLVM::GEPOp::create(
        builder, loc, ptr, argType, arg,
        llvm::ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    return LLVM::LoadOp::create(builder, loc, type, fieldPtr);
  };
  mlir::Value body = field(0, ptr);
  mlir::Value ctx = field(1, ptr);
  mlir::Value begin = field(2, i64);
  mlir::Value end = field(3, i64);
  llvm::SmallVector<mlir::Value, 4> operands{body, begin, end, ctx};
  LLVM::CallOp::create(builder, loc, bodyFnType(context), operands);
  mlir::Value null = LLVM::ZeroOp::create(builder, loc, ptr);
  LLVM::ReturnOp::create(builder, loc, null);
}

// struct ApplyCtx { void (*fn)(i64, i64, ptr); ptr ctx; i64 n; i64 slices; }
LLVM::LLVMStructType applyCtxType(mlir::MLIRContext *context) {
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = mlir::IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(context, {ptr, ptr, i64, i64});
}

// LyParallelApply(ptr applyCtx, i64 index): the dispatch_apply_f callback --
// turn the iteration index into this slice's half-open range and run it.
void buildParallelApply(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = builder.getI64Type();
  mlir::Location loc = builder.getUnknownLoc();
  auto ctxType = applyCtxType(context);
  auto voidType = LLVM::LLVMVoidType::get(context);

  builder.setInsertionPointToEnd(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(
      builder, loc, kParallelApplyName,
      LLVM::LLVMFunctionType::get(voidType, {ptr, i64}),
      LLVM::Linkage::Internal);
  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::Block *run = fn.addBlock();
  mlir::Block *done = fn.addBlock();

  builder.setInsertionPointToStart(entry);
  mlir::Value arg = entry->getArgument(0);
  mlir::Value index = entry->getArgument(1);
  auto field = [&](int64_t i, mlir::Type type) -> mlir::Value {
    mlir::Value p = LLVM::GEPOp::create(
        builder, loc, ptr, ctxType, arg,
        llvm::ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(i)});
    return LLVM::LoadOp::create(builder, loc, type, p);
  };
  mlir::Value body = field(0, ptr);
  mlir::Value ctx = field(1, ptr);
  mlir::Value n = field(2, i64);
  mlir::Value slices = field(3, i64);
  // begin = index*n/slices, end = (index+1)*n/slices: an even split that needs
  // no divisibility, so a slice count the extents cannot divide is still legal.
  mlir::Value one = constI64(builder, loc, 1);
  mlir::Value begin = LLVM::SDivOp::create(
      builder, loc, LLVM::MulOp::create(builder, loc, index, n), slices);
  mlir::Value end = LLVM::SDivOp::create(
      builder, loc,
      LLVM::MulOp::create(builder, loc,
                          LLVM::AddOp::create(builder, loc, index, one), n),
      slices);
  mlir::Value nonEmpty =
      LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::slt, begin, end);
  LLVM::CondBrOp::create(builder, loc, nonEmpty, run, mlir::ValueRange{}, done,
                         mlir::ValueRange{});

  builder.setInsertionPointToStart(run);
  LLVM::CallOp::create(builder, loc, bodyFnType(context),
                       llvm::SmallVector<mlir::Value, 4>{body, begin, end, ctx});
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, done);

  builder.setInsertionPointToStart(done);
  LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
}

// C-string constant global (idempotent).
LLVM::GlobalOp ensureCString(mlir::ModuleOp module, mlir::OpBuilder &builder,
                             llvm::StringRef name, llvm::StringRef text) {
  if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  std::string bytes = (text + llvm::Twine('\0')).str();
  auto type = LLVM::LLVMArrayType::get(builder.getI8Type(), bytes.size());
  return LLVM::GlobalOp::create(builder, builder.getUnknownLoc(), type,
                                /*isConstant=*/true, LLVM::Linkage::Internal,
                                name, builder.getStringAttr(bytes));
}

// LyDispatchAttrWidth() -> i64: how many per-cluster workers the libdispatch
// parallelism SPI can place, or 0 when the fast path is unusable.
//
// Two facts are resolved once and cached: whether the SPI entry points exist
// (dlsym -- they are unexported-in-headers but present since macOS 15, and a
// binary must keep launching on systems where they are not), and how many
// P-clusters the machine has (hw.perflevel0 sysctls, public). The
// unsynchronized cache is benign for the same reason the thread-count cache
// is: initialization is idempotent and racing writers store the same values.
void buildDispatchAttrWidth(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = builder.getI64Type();
  auto i32 = builder.getI32Type();
  mlir::Location loc = builder.getUnknownLoc();

  builder.setInsertionPointToEnd(module.getBody());
  auto stateGlobal = LLVM::GlobalOp::create(
      builder, loc, i64, /*isConstant=*/false, LLVM::Linkage::Internal,
      kAttrStateName, builder.getI64IntegerAttr(0));
  llvm::SmallVector<LLVM::GlobalOp, 4> fnGlobals;
  for (llvm::StringRef name :
       {kAttrInitPtrName, kAttrSetParPtrName, kAttrDestroyPtrName,
        kAttrApplyPtrName}) {
    builder.setInsertionPointToEnd(module.getBody());
    fnGlobals.push_back(LLVM::GlobalOp::create(
        builder, loc, ptr, /*isConstant=*/false, LLVM::Linkage::Internal,
        name, mlir::Attribute{}));
  }
  builder.setInsertionPointToEnd(module.getBody());
  auto widthGlobal = LLVM::GlobalOp::create(
      builder, loc, i64, /*isConstant=*/false, LLVM::Linkage::Internal,
      kClusterWidthName, builder.getI64IntegerAttr(0));

  auto dlsymFn = ensureFuncDecl(module, builder, "dlsym",
                                LLVM::LLVMFunctionType::get(ptr, {ptr, ptr}));
  auto sysctlFn = ensureFuncDecl(
      module, builder, "sysctlbyname",
      LLVM::LLVMFunctionType::get(i32, {ptr, ptr, ptr, ptr, i64}));

  LLVM::GlobalOp symInit = ensureCString(module, builder, "__ly_str_attr_init",
                                         "dispatch_apply_attr_init");
  LLVM::GlobalOp symSet = ensureCString(
      module, builder, "__ly_str_attr_set",
      "dispatch_apply_attr_set_parallelism");
  LLVM::GlobalOp symDestroy = ensureCString(
      module, builder, "__ly_str_attr_destroy", "dispatch_apply_attr_destroy");
  LLVM::GlobalOp symApply = ensureCString(
      module, builder, "__ly_str_attr_apply", "dispatch_apply_with_attr_f");
  LLVM::GlobalOp sysPcpu = ensureCString(
      module, builder, "__ly_str_perf_pcpu", "hw.perflevel0.physicalcpu");
  LLVM::GlobalOp sysPerL2 = ensureCString(
      module, builder, "__ly_str_perf_l2", "hw.perflevel0.cpusperl2");

  builder.setInsertionPointToEnd(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(builder, loc, kAttrWidthFnName,
                                     LLVM::LLVMFunctionType::get(i64, {}),
                                     LLVM::Linkage::Internal);
  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::Block *resolve = fn.addBlock();
  mlir::Block *good = fn.addBlock();
  good->addArgument(i64, loc);
  mlir::Block *bad = fn.addBlock();
  mlir::Block *done = fn.addBlock();
  done->addArgument(i64, loc);

  builder.setInsertionPointToStart(entry);
  mlir::Value zero = constI64(builder, loc, 0);
  mlir::Value statePtr = LLVM::AddressOfOp::create(builder, loc, stateGlobal);
  mlir::Value widthPtr = LLVM::AddressOfOp::create(builder, loc, widthGlobal);
  mlir::Value state = LLVM::LoadOp::create(builder, loc, i64, statePtr);
  mlir::Value cachedWidth = LLVM::LoadOp::create(builder, loc, i64, widthPtr);
  // state 0 = unresolved, otherwise the cached width (0 when unusable) is the
  // answer.
  mlir::Value unresolved = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::eq, state, zero);
  LLVM::CondBrOp::create(builder, loc, unresolved, resolve, mlir::ValueRange{},
                         done, mlir::ValueRange{cachedWidth});

  builder.setInsertionPointToStart(resolve);
  // RTLD_DEFAULT on Darwin.
  mlir::Value rtldDefault = LLVM::IntToPtrOp::create(
      builder, loc, ptr, constI64(builder, loc, -2));
  llvm::SmallVector<mlir::Value, 4> resolved;
  mlir::Value allFound;
  llvm::SmallVector<LLVM::GlobalOp, 4> symNames{symInit, symSet, symDestroy,
                                                symApply};
  for (auto [index, sym] : llvm::enumerate(symNames)) {
    mlir::Value namePtr = LLVM::AddressOfOp::create(builder, loc, sym);
    mlir::Value fnPtr =
        LLVM::CallOp::create(builder, loc, dlsymFn,
                             mlir::ValueRange{rtldDefault, namePtr})
            .getResult();
    resolved.push_back(fnPtr);
    mlir::Value nonNull = LLVM::ICmpOp::create(
        builder, loc, LLVM::ICmpPredicate::ne, fnPtr,
        LLVM::ZeroOp::create(builder, loc, ptr));
    allFound = index == 0
                   ? nonNull
                   : LLVM::AndOp::create(builder, loc, allFound, nonNull)
                         .getResult();
  }

  // Clusters = physical P cores / cores per L2. Both are 4-byte sysctls.
  auto readSysctl = [&](LLVM::GlobalOp nameGlobal) -> mlir::Value {
    mlir::Value one = constI64(builder, loc, 1);
    mlir::Value out = LLVM::AllocaOp::create(builder, loc, ptr, i32, one);
    mlir::Value sizeSlot = LLVM::AllocaOp::create(builder, loc, ptr, i64, one);
    LLVM::StoreOp::create(builder, loc, constI64(builder, loc, 4), sizeSlot);
    LLVM::StoreOp::create(
        builder, loc,
        LLVM::ConstantOp::create(builder, loc, i32,
                                 builder.getI32IntegerAttr(0)),
        out);
    mlir::Value namePtr = LLVM::AddressOfOp::create(builder, loc, nameGlobal);
    mlir::Value nullPtr = LLVM::ZeroOp::create(builder, loc, ptr);
    mlir::Value rc =
        LLVM::CallOp::create(
            builder, loc, sysctlFn,
            mlir::ValueRange{namePtr, out, sizeSlot, nullPtr, zero})
            .getResult();
    mlir::Value value32 = LLVM::LoadOp::create(builder, loc, i32, out);
    mlir::Value value = LLVM::SExtOp::create(builder, loc, i64, value32);
    mlir::Value ok = LLVM::ICmpOp::create(
        builder, loc, LLVM::ICmpPredicate::eq, rc,
        LLVM::ConstantOp::create(builder, loc, i32,
                                 builder.getI32IntegerAttr(0)));
    return LLVM::SelectOp::create(builder, loc, ok, value, zero);
  };
  mlir::Value pcpu = readSysctl(sysPcpu);
  mlir::Value perL2 = readSysctl(sysPerL2);
  mlir::Value perL2Ok = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::sgt, perL2, zero);
  mlir::Value safePerL2 = LLVM::SelectOp::create(
      builder, loc, perL2Ok, perL2, constI64(builder, loc, 1));
  mlir::Value clusters = LLVM::SDivOp::create(builder, loc, pcpu, safePerL2);
  mlir::Value clustersOk = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::sge, clusters,
      constI64(builder, loc, 2));
  mlir::Value usable = LLVM::AndOp::create(
      builder, loc, LLVM::AndOp::create(builder, loc, allFound, perL2Ok),
      clustersOk);
  LLVM::CondBrOp::create(builder, loc, usable, good,
                         mlir::ValueRange{clusters}, bad, mlir::ValueRange{});

  builder.setInsertionPointToStart(good);
  for (auto [index, global] : llvm::enumerate(fnGlobals)) {
    mlir::Value p = LLVM::AddressOfOp::create(builder, loc, global);
    LLVM::StoreOp::create(builder, loc, resolved[index], p);
  }
  LLVM::StoreOp::create(builder, loc, good->getArgument(0), widthPtr);
  LLVM::StoreOp::create(builder, loc, constI64(builder, loc, 1), statePtr);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{good->getArgument(0)},
                     done);

  builder.setInsertionPointToStart(bad);
  LLVM::StoreOp::create(builder, loc, zero, widthPtr);
  LLVM::StoreOp::create(builder, loc, constI64(builder, loc, 2), statePtr);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{zero}, done);

  builder.setInsertionPointToStart(done);
  LLVM::ReturnOp::create(builder, loc, done->getArgument(0));
}

// LyParallelApplyAttr(ptr ctx, i64 index, i64 worker): the with_attr_f
// callback carries one more argument than dispatch_apply_f's; the range logic
// is identical, so forward.
void buildParallelApplyAttr(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = builder.getI64Type();
  auto voidType = LLVM::LLVMVoidType::get(context);
  mlir::Location loc = builder.getUnknownLoc();

  builder.setInsertionPointToEnd(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(
      builder, loc, kParallelApplyAttrName,
      LLVM::LLVMFunctionType::get(voidType, {ptr, i64, i64}),
      LLVM::Linkage::Internal);
  mlir::Block *entry = fn.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  auto applyFn = module.lookupSymbol<LLVM::LLVMFuncOp>(kParallelApplyName);
  LLVM::CallOp::create(
      builder, loc, applyFn,
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1)});
  LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
}

// LyParallelFor(ptr fn, ptr ctx, i64 n) on Darwin: hand the slices to
// dispatch_apply_f and let libdispatch place them.
//
// Why not the pthread spawn below, which is otherwise equivalent: on Apple
// Silicon the SME unit and the L2 are per-cluster, so two workers are worth 2x
// only if they land on different clusters -- and raw pthread_create does not
// choose. Measured on M4 Max with the same SME kernel, two workers over eight
// runs: pthread 1947-3633 GFLOP/s (median 1973, i.e. usually no speedup at all,
// 85% spread), dispatch_apply_f 3748-3758 every time. libdispatch knows the
// topology; pthreads only get lucky.
//
// Why not dispatch_apply_attr_set_parallelism, which is what Accelerate itself
// calls: it is SPI, and measured the same -- 3751 median against 3646 for the
// public entry point.
void buildParallelForDispatch(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = builder.getI64Type();
  mlir::Location loc = builder.getUnknownLoc();
  auto ctxType = applyCtxType(context);
  auto voidType = LLVM::LLVMVoidType::get(context);

  auto applyFn = ensureFuncDecl(
      module, builder, "dispatch_apply_f",
      LLVM::LLVMFunctionType::get(voidType, {i64, ptr, ptr, ptr}));
  auto threadsFn = module.lookupSymbol<LLVM::LLVMFuncOp>(kParallelThreadsName);

  builder.setInsertionPointToEnd(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(
      builder, loc, kParallelForName,
      LLVM::LLVMFunctionType::get(voidType, {ptr, ptr, i64}),
      LLVM::Linkage::Internal);
  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::Block *serial = fn.addBlock();
  mlir::Block *parallel = fn.addBlock();
  mlir::Block *attrPath = fn.addBlock();
  attrPath->addArgument(builder.getI64Type(), loc);
  mlir::Block *plainPath = fn.addBlock();
  mlir::Block *done = fn.addBlock();

  mlir::Value fnArg = entry->getArgument(0);
  mlir::Value ctxArg = entry->getArgument(1);
  mlir::Value nArg = entry->getArgument(2);

  builder.setInsertionPointToStart(entry);
  mlir::Value zero = constI64(builder, loc, 0);
  mlir::Value one = constI64(builder, loc, 1);
  mlir::Value slot = LLVM::AllocaOp::create(builder, loc, ptr, ctxType, one);
  // The parallelism-attr descriptor: 64 opaque bytes, per libdispatch's own
  // layout contract for dispatch_apply_attr_s.
  auto attrBufType = LLVM::LLVMArrayType::get(builder.getI8Type(), 64);
  mlir::Value attrBuf =
      LLVM::AllocaOp::create(builder, loc, ptr, attrBufType, one);
  mlir::Value threads =
      LLVM::CallOp::create(builder, loc, threadsFn, mlir::ValueRange{})
          .getResult();
  mlir::Value nBelow = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::slt, nArg, threads);
  mlir::Value width =
      LLVM::SelectOp::create(builder, loc, nBelow, nArg, threads);
  mlir::Value widthOk =
      LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::sgt, width, one);
  mlir::Value nPositive =
      LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::sgt, nArg, zero);
  LLVM::CondBrOp::create(builder, loc,
                         LLVM::AndOp::create(builder, loc, widthOk, nPositive),
                         parallel, mlir::ValueRange{}, serial,
                         mlir::ValueRange{});

  // One slice: no reason to involve libdispatch at all.
  builder.setInsertionPointToStart(serial);
  LLVM::CallOp::create(
      builder, loc, bodyFnType(context),
      llvm::SmallVector<mlir::Value, 4>{fnArg, zero, nArg, ctxArg});
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, done);

  builder.setInsertionPointToStart(parallel);
  auto store = [&](int64_t i, mlir::Value v) {
    mlir::Value p = LLVM::GEPOp::create(
        builder, loc, ptr, ctxType, slot,
        llvm::ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(i)});
    LLVM::StoreOp::create(builder, loc, v, p);
  };
  store(0, fnArg);
  store(1, ctxArg);
  store(2, nArg);
  // Two ways to place the slices. The parallelism attr asks libdispatch for
  // one worker per cluster, which is the only thing that reliably lands two
  // workers on two SME units when each fork lives ~200us: measured on M4 Max
  // chained 512^3 dispatches, two plain slices ran 1860 GFLOP/s (both workers
  // on one cluster more often than not) while two attr-placed slices ran 2886
  // at a third of the CPU the eight-slice oversubscription burns for its
  // 2947. Oversubscription stays as the fallback -- it reaches the second
  // cluster by flooding, which is exactly what it costs.
  //
  // The attr path steps aside when LYTHON_NUM_THREADS is set: a pinned thread
  // count means a measurement, and it should measure the plain path it pins.
  auto envSetGlobal = module.lookupSymbol<LLVM::GlobalOp>(kThreadsEnvSetName);
  mlir::Value envSetPtr =
      LLVM::AddressOfOp::create(builder, loc, envSetGlobal);
  mlir::Value envSet =
      LLVM::LoadOp::create(builder, loc, builder.getI64Type(), envSetPtr);
  auto widthFn = module.lookupSymbol<LLVM::LLVMFuncOp>(kAttrWidthFnName);
  mlir::Value attrWidthRaw =
      LLVM::CallOp::create(builder, loc, widthFn, mlir::ValueRange{})
          .getResult();
  mlir::Value envUnset = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::eq, envSet, zero);
  mlir::Value attrWidth0 = LLVM::SelectOp::create(builder, loc, envUnset,
                                                  attrWidthRaw, zero);
  mlir::Value nBelowAttr = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::slt, nArg, attrWidth0);
  mlir::Value attrWidth =
      LLVM::SelectOp::create(builder, loc, nBelowAttr, nArg, attrWidth0);
  mlir::Value useAttr = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::sge, attrWidth,
      constI64(builder, loc, 2));
  LLVM::CondBrOp::create(builder, loc, useAttr, attrPath,
                         mlir::ValueRange{attrWidth}, plainPath,
                         mlir::ValueRange{});

  builder.setInsertionPointToStart(attrPath);
  {
    mlir::Value slices = attrPath->getArgument(0);
    mlir::Value p3 = LLVM::GEPOp::create(
        builder, loc, ptr, ctxType, slot,
        llvm::ArrayRef<LLVM::GEPArg>{0, 3});
    LLVM::StoreOp::create(builder, loc, slices, p3);
    // Zero the descriptor before init, as the header contract demands.
    for (int64_t i = 0; i < 8; ++i) {
      mlir::Value p = LLVM::GEPOp::create(
          builder, loc, ptr, builder.getI64Type(), attrBuf,
          llvm::ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i)});
      LLVM::StoreOp::create(builder, loc, zero, p);
    }
    auto loadFnPtr = [&](llvm::StringRef global) {
      mlir::Value p = LLVM::AddressOfOp::create(
          builder, loc, module.lookupSymbol<LLVM::GlobalOp>(global));
      return LLVM::LoadOp::create(builder, loc, ptr, p).getResult();
    };
    auto voidType = LLVM::LLVMVoidType::get(context);
    auto initType = LLVM::LLVMFunctionType::get(voidType, {ptr});
    mlir::Value initPtr = loadFnPtr(kAttrInitPtrName);
    LLVM::CallOp::create(builder, loc, initType,
                         mlir::ValueRange{initPtr, attrBuf});
    // entity = -1 (every cluster), one worker on each.
    auto setType = LLVM::LLVMFunctionType::get(voidType, {ptr, i64, i64});
    mlir::Value setPtr = loadFnPtr(kAttrSetParPtrName);
    LLVM::CallOp::create(builder, loc, setType,
                         mlir::ValueRange{setPtr, attrBuf,
                                          constI64(builder, loc, -1), one});
    auto applyType =
        LLVM::LLVMFunctionType::get(voidType, {i64, ptr, ptr, ptr});
    mlir::Value applyPtr = loadFnPtr(kAttrApplyPtrName);
    mlir::Value thunk =
        LLVM::AddressOfOp::create(builder, loc, ptr, kParallelApplyAttrName);
    LLVM::CallOp::create(builder, loc, applyType,
                         mlir::ValueRange{applyPtr, slices, attrBuf, slot,
                                          thunk});
    mlir::Value destroyPtr = loadFnPtr(kAttrDestroyPtrName);
    LLVM::CallOp::create(builder, loc, initType,
                         mlir::ValueRange{destroyPtr, attrBuf});
  }
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, done);

  builder.setInsertionPointToStart(plainPath);
  store(3, width);
  mlir::Value apply =
      LLVM::AddressOfOp::create(builder, loc, ptr, kParallelApplyName);
  // DISPATCH_APPLY_AUTO is a null queue: it asks for the global queue matching
  // the caller's QoS, which is what keeps the workers off the E-cores.
  mlir::Value autoQueue = LLVM::ZeroOp::create(builder, loc, ptr);
  LLVM::CallOp::create(builder, loc, applyFn,
                       mlir::ValueRange{width, autoQueue, slot, apply});
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, done);

  builder.setInsertionPointToStart(done);
  LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
}

// LyParallelFor(ptr fn, ptr ctx, i64 n): split [0, n) across workers. The
// caller's thread runs the first range; the rest run on freshly spawned
// pthreads joined before returning, so no runtime state outlives the call
// and no worker ever touches the object runtime. A failed pthread_create
// falls back to running that range inline -- degraded parallelism, never a
// dropped range.
//
// Kept for the targets libdispatch is not part of the platform on; Darwin uses
// buildParallelForDispatch instead.
void buildParallelFor(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = module.getContext();
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = builder.getI64Type();
  auto i32 = builder.getI32Type();
  mlir::Location loc = builder.getUnknownLoc();
  auto argType = workerArgType(context);

  auto pthreadCreate = ensureFuncDecl(
      module, builder, "pthread_create",
      LLVM::LLVMFunctionType::get(i32, {ptr, ptr, ptr, ptr}));
  auto pthreadJoin =
      ensureFuncDecl(module, builder, "pthread_join",
                     LLVM::LLVMFunctionType::get(i32, {i64, ptr}));
  auto threadsFn = module.lookupSymbol<LLVM::LLVMFuncOp>(kParallelThreadsName);
  auto workerFn = module.lookupSymbol<LLVM::LLVMFuncOp>(kParallelWorkerName);

  builder.setInsertionPointToEnd(module.getBody());
  auto voidType = LLVM::LLVMVoidType::get(context);
  auto fn = LLVM::LLVMFuncOp::create(
      builder, loc, kParallelForName,
      LLVM::LLVMFunctionType::get(voidType, {ptr, ptr, i64}),
      LLVM::Linkage::Internal);

  mlir::Block *entry = fn.addEntryBlock(builder);
  mlir::Block *serial = fn.addBlock();
  mlir::Block *parallel = fn.addBlock();
  mlir::Block *spawnHead = fn.addBlock();
  spawnHead->addArgument(i64, loc);
  mlir::Block *spawnBody = fn.addBlock();
  mlir::Block *spawnFailed = fn.addBlock();
  mlir::Block *spawnNext = fn.addBlock();
  spawnNext->addArgument(i64, loc);
  mlir::Block *runMain = fn.addBlock();
  mlir::Block *joinHead = fn.addBlock();
  joinHead->addArgument(i64, loc);
  mlir::Block *joinBody = fn.addBlock();
  mlir::Block *joinCheck = fn.addBlock();
  mlir::Block *joinNext = fn.addBlock();
  mlir::Block *done = fn.addBlock();

  mlir::Value fnArg = entry->getArgument(0);
  mlir::Value ctxArg = entry->getArgument(1);
  mlir::Value nArg = entry->getArgument(2);

  builder.setInsertionPointToStart(entry);
  mlir::Value zero = constI64(builder, loc, 0);
  mlir::Value one = constI64(builder, loc, 1);
  mlir::Value threads =
      LLVM::CallOp::create(builder, loc, threadsFn, mlir::ValueRange{})
          .getResult();
  mlir::Value nBelow = LLVM::ICmpOp::create(builder, loc,
                                            LLVM::ICmpPredicate::slt, nArg,
                                            threads);
  mlir::Value width =
      LLVM::SelectOp::create(builder, loc, nBelow, nArg, threads);
  // chunk = ceildiv(n, width); workers = ceildiv(n, chunk) drops empty tails.
  mlir::Value widthOk = LLVM::ICmpOp::create(builder, loc,
                                             LLVM::ICmpPredicate::sgt, width,
                                             one);
  mlir::Value nPositive = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::sgt, nArg, zero);
  mlir::Value goParallel =
      LLVM::AndOp::create(builder, loc, widthOk, nPositive);
  LLVM::CondBrOp::create(builder, loc, goParallel, parallel,
                         mlir::ValueRange{}, serial, mlir::ValueRange{});

  builder.setInsertionPointToStart(serial);
  {
    mlir::Value skip = LLVM::ICmpOp::create(
        builder, loc, LLVM::ICmpPredicate::sle, nArg, zero);
    mlir::Block *serialRun = fn.addBlock();
    LLVM::CondBrOp::create(builder, loc, skip, done, mlir::ValueRange{},
                           serialRun, mlir::ValueRange{});
    builder.setInsertionPointToStart(serialRun);
    llvm::SmallVector<mlir::Value, 4> operands{fnArg, zero, nArg, ctxArg};
    LLVM::CallOp::create(builder, loc, bodyFnType(context), operands);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, done);
  }

  builder.setInsertionPointToStart(parallel);
  mlir::Value nMinus = LLVM::SubOp::create(builder, loc, nArg, one);
  mlir::Value nRounded = LLVM::AddOp::create(builder, loc, nMinus, width);
  mlir::Value chunk = LLVM::SDivOp::create(builder, loc, nRounded, width);
  mlir::Value chunkMinus = LLVM::SubOp::create(builder, loc, nArg, one);
  mlir::Value workersRounded =
      LLVM::AddOp::create(builder, loc, chunkMinus, chunk);
  mlir::Value workers =
      LLVM::SDivOp::create(builder, loc, workersRounded, chunk);
  mlir::Value maxWorkers = constI64(builder, loc, kMaxParallelThreads);
  mlir::Value threadSlots = LLVM::AllocaOp::create(
      builder, loc, ptr, i64, maxWorkers, /*alignment=*/16);
  mlir::Value argSlots = LLVM::AllocaOp::create(
      builder, loc, ptr, argType, maxWorkers, /*alignment=*/16);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{one}, spawnHead);

  // for i in [1, workers): spawn [i*chunk, min((i+1)*chunk, n))
  builder.setInsertionPointToStart(spawnHead);
  mlir::Value spawnIndex = spawnHead->getArgument(0);
  mlir::Value moreSpawns = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::slt, spawnIndex, workers);
  LLVM::CondBrOp::create(builder, loc, moreSpawns, spawnBody,
                         mlir::ValueRange{}, runMain, mlir::ValueRange{});

  builder.setInsertionPointToStart(spawnBody);
  mlir::Value begin = LLVM::MulOp::create(builder, loc, spawnIndex, chunk);
  mlir::Value rawEnd = LLVM::AddOp::create(builder, loc, begin, chunk);
  mlir::Value endOver = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::sgt, rawEnd, nArg);
  mlir::Value end = LLVM::SelectOp::create(builder, loc, endOver, nArg,
                                           rawEnd);
  mlir::Value argSlot =
      LLVM::GEPOp::create(builder, loc, ptr, argType, argSlots,
                          llvm::ArrayRef<LLVM::GEPArg>{spawnIndex});
  auto storeField = [&](int64_t index, mlir::Value value) {
    mlir::Value fieldPtr = LLVM::GEPOp::create(
        builder, loc, ptr, argType, argSlot,
        llvm::ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    LLVM::StoreOp::create(builder, loc, value, fieldPtr);
  };
  storeField(0, fnArg);
  storeField(1, ctxArg);
  storeField(2, begin);
  storeField(3, end);
  mlir::Value threadSlot =
      LLVM::GEPOp::create(builder, loc, ptr, i64, threadSlots,
                          llvm::ArrayRef<LLVM::GEPArg>{spawnIndex});
  mlir::Value workerAddr = LLVM::AddressOfOp::create(builder, loc, workerFn);
  mlir::Value nullAttr = LLVM::ZeroOp::create(builder, loc, ptr);
  mlir::Value rc =
      LLVM::CallOp::create(builder, loc, pthreadCreate,
                           mlir::ValueRange{threadSlot, nullAttr, workerAddr,
                                            argSlot})
          .getResult();
  mlir::Value rcZero = LLVM::ConstantOp::create(builder, loc, i32,
                                                builder.getI32IntegerAttr(0));
  mlir::Value spawned = LLVM::ICmpOp::create(builder, loc,
                                             LLVM::ICmpPredicate::eq, rc,
                                             rcZero);
  LLVM::CondBrOp::create(builder, loc, spawned, spawnNext,
                         mlir::ValueRange{spawnIndex}, spawnFailed,
                         mlir::ValueRange{});

  builder.setInsertionPointToStart(spawnFailed);
  {
    llvm::SmallVector<mlir::Value, 4> operands{fnArg, begin, end, ctxArg};
    LLVM::CallOp::create(builder, loc, bodyFnType(context), operands);
    LLVM::StoreOp::create(builder, loc, zero, threadSlot);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{spawnIndex}, spawnNext);
  }

  builder.setInsertionPointToStart(spawnNext);
  mlir::Value nextSpawn =
      LLVM::AddOp::create(builder, loc, spawnNext->getArgument(0), one);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{nextSpawn}, spawnHead);

  builder.setInsertionPointToStart(runMain);
  {
    llvm::SmallVector<mlir::Value, 4> operands{fnArg, zero, chunk, ctxArg};
    LLVM::CallOp::create(builder, loc, bodyFnType(context), operands);
    LLVM::BrOp::create(builder, loc, mlir::ValueRange{one}, joinHead);
  }

  builder.setInsertionPointToStart(joinHead);
  mlir::Value joinIndex = joinHead->getArgument(0);
  mlir::Value moreJoins = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::slt, joinIndex, workers);
  LLVM::CondBrOp::create(builder, loc, moreJoins, joinBody, mlir::ValueRange{},
                         done, mlir::ValueRange{});

  builder.setInsertionPointToStart(joinBody);
  mlir::Value joinSlot =
      LLVM::GEPOp::create(builder, loc, ptr, i64, threadSlots,
                          llvm::ArrayRef<LLVM::GEPArg>{joinIndex});
  mlir::Value thread = LLVM::LoadOp::create(builder, loc, i64, joinSlot);
  mlir::Value threadValid = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::ne, thread, zero);
  LLVM::CondBrOp::create(builder, loc, threadValid, joinCheck,
                         mlir::ValueRange{}, joinNext, mlir::ValueRange{});

  builder.setInsertionPointToStart(joinCheck);
  mlir::Value nullResult = LLVM::ZeroOp::create(builder, loc, ptr);
  LLVM::CallOp::create(builder, loc, pthreadJoin,
                       mlir::ValueRange{thread, nullResult});
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, joinNext);

  builder.setInsertionPointToStart(joinNext);
  mlir::Value nextJoin = LLVM::AddOp::create(builder, loc, joinIndex, one);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{nextJoin}, joinHead);

  builder.setInsertionPointToStart(done);
  LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
}

} // namespace

std::unique_ptr<mlir::Pass> createMatmulParallelChunkPass() {
  return std::make_unique<MatmulParallelChunkPass>();
}

std::unique_ptr<mlir::Pass> createParallelLoopOutliningPass() {
  return std::make_unique<ParallelLoopOutliningPass>();
}

namespace {

// A parallel body runs on worker threads that own no object-runtime state:
// it may allocate raw buffers (malloc is thread-safe) but must never reach
// the refcounted object runtime, whose ownership proofs assume the tokens
// stay on the spawning thread. Verified over the final LLVM IR, same style
// as the ctypes callback signal-safety check.
mlir::LogicalResult verifyParallelBodyIsKernelOnly(mlir::ModuleOp module,
                                                   LLVM::LLVMFuncOp entry) {
  llvm::SmallVector<LLVM::LLVMFuncOp, 8> worklist{entry};
  llvm::DenseSet<mlir::Operation *> visited{entry};
  while (!worklist.empty()) {
    LLVM::LLVMFuncOp function = worklist.pop_back_val();
    mlir::LogicalResult result = mlir::success();
    function.walk([&](LLVM::CallOp call) {
      std::optional<llvm::StringRef> callee = call.getCallee();
      if (!callee)
        return mlir::WalkResult::advance();
      if (callee->starts_with("Ly") && !callee->starts_with("LyParallel")) {
        call.emitError()
            << "parallel kernel body '" << entry.getSymName()
            << "' reaches the object runtime ('" << *callee
            << "'); worker threads may only run allocation and raw kernel "
               "code";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      if (auto next = module.lookupSymbol<LLVM::LLVMFuncOp>(*callee))
        if (!next.isExternal() && visited.insert(next).second)
          worklist.push_back(next);
      return mlir::WalkResult::advance();
    });
    if (mlir::failed(result))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult materializeParallelDispatch(mlir::ModuleOp module) {
  llvm::SmallVector<LLVM::LLVMFuncOp, 4> bodies;
  module.walk([&](LLVM::LLVMFuncOp function) {
    if (function->hasAttr(kParallelBodyAttr))
      bodies.push_back(function);
  });
  if (bodies.empty())
    return mlir::success();

  for (LLVM::LLVMFuncOp body : bodies)
    if (mlir::failed(verifyParallelBodyIsKernelOnly(module, body)))
      return mlir::failure();

  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  auto ptr = LLVM::LLVMPointerType::get(context);
  auto i64 = builder.getI64Type();

  buildParallelThreads(module, builder);
  // libdispatch is part of the platform on Darwin and places workers by the
  // machine's topology; elsewhere the pthread spawn is what there is.
  bool darwin = false;
  if (auto triple = module->getAttrOfType<mlir::StringAttr>("ly.target.triple"))
    darwin = triple.getValue().contains("darwin");
  if (darwin) {
    buildParallelApply(module, builder);
    buildDispatchAttrWidth(module, builder);
    buildParallelApplyAttr(module, builder);
    buildParallelForDispatch(module, builder);
  } else {
    buildParallelWorker(module, builder);
    buildParallelFor(module, builder);
  }
  auto parallelFor = module.lookupSymbol<LLVM::LLVMFuncOp>(kParallelForName);

  for (LLVM::LLVMFuncOp body : bodies) {
    body->removeAttr(kParallelBodyAttr);
    LLVM::LLVMFunctionType bodyType = body.getFunctionType();
    if (bodyType.getNumParams() < 3)
      return body.emitError()
             << "parallel body lost its range parameters during lowering";

    llvm::SmallVector<LLVM::CallOp, 4> calls;
    module.walk([&](LLVM::CallOp call) {
      if (call.getCallee() && *call.getCallee() == body.getSymName())
        calls.push_back(call);
    });

    // struct Ctx { i64 lb; i64 ub; i64 step; captures... }
    llvm::SmallVector<mlir::Type, 12> fields(bodyType.getParams().begin(),
                                             bodyType.getParams().end());
    auto ctxType = LLVM::LLVMStructType::getLiteral(context, fields);

    // Thunk: (begin, end, ctx) -> run body over trips [begin, end).
    mlir::Location loc = body.getLoc();
    builder.setInsertionPointToEnd(module.getBody());
    auto thunk = LLVM::LLVMFuncOp::create(
        builder, loc, (body.getSymName() + "_thunk").str(),
        bodyFnType(context), LLVM::Linkage::Internal);
    mlir::Block *entry = thunk.addEntryBlock(builder);
    builder.setInsertionPointToStart(entry);
    mlir::Value beginTrip = entry->getArgument(0);
    mlir::Value endTrip = entry->getArgument(1);
    mlir::Value ctx = entry->getArgument(2);
    auto loadField = [&](int64_t index) -> mlir::Value {
      mlir::Value fieldPtr = LLVM::GEPOp::create(
          builder, loc, ptr, ctxType, ctx,
          llvm::ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
      return LLVM::LoadOp::create(builder, loc, fields[index], fieldPtr);
    };
    mlir::Value lb = loadField(0);
    mlir::Value ub = loadField(1);
    mlir::Value step = loadField(2);
    mlir::Value lowerOffset =
        LLVM::MulOp::create(builder, loc, beginTrip, step);
    mlir::Value lower = LLVM::AddOp::create(builder, loc, lb, lowerOffset);
    mlir::Value upperOffset = LLVM::MulOp::create(builder, loc, endTrip, step);
    mlir::Value rawUpper = LLVM::AddOp::create(builder, loc, lb, upperOffset);
    mlir::Value upperOver = LLVM::ICmpOp::create(
        builder, loc, LLVM::ICmpPredicate::sgt, rawUpper, ub);
    mlir::Value upper =
        LLVM::SelectOp::create(builder, loc, upperOver, ub, rawUpper);
    llvm::SmallVector<mlir::Value, 12> bodyOperands{lower, upper, step};
    for (std::size_t index = 3; index < fields.size(); ++index)
      bodyOperands.push_back(loadField(static_cast<int64_t>(index)));
    LLVM::CallOp::create(builder, loc, body, bodyOperands);
    LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});

    for (LLVM::CallOp call : calls) {
      auto enclosing = call->getParentOfType<LLVM::LLVMFuncOp>();
      if (!enclosing)
        return call.emitError()
               << "parallel body call outside an LLVM function";
      // The context lives in the caller's entry block: an alloca at the call
      // site would grow the stack on every loop iteration around the call.
      builder.setInsertionPointToStart(&enclosing.getBody().front());
      mlir::Value one = constI64(builder, loc, 1);
      mlir::Value ctxSlot =
          LLVM::AllocaOp::create(builder, loc, ptr, ctxType, one,
                                 /*alignment=*/16);

      builder.setInsertionPoint(call);
      for (auto [index, operand] : llvm::enumerate(call.getArgOperands())) {
        mlir::Value fieldPtr = LLVM::GEPOp::create(
            builder, loc, ptr, ctxType, ctxSlot,
            llvm::ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
        LLVM::StoreOp::create(builder, loc, operand, fieldPtr);
      }
      mlir::Value lbValue = call.getArgOperands()[0];
      mlir::Value ubValue = call.getArgOperands()[1];
      mlir::Value stepValue = call.getArgOperands()[2];
      mlir::Value span = LLVM::SubOp::create(builder, loc, ubValue, lbValue);
      mlir::Value stepMinus =
          LLVM::SubOp::create(builder, loc, stepValue, one);
      mlir::Value spanRounded =
          LLVM::AddOp::create(builder, loc, span, stepMinus);
      mlir::Value trips =
          LLVM::SDivOp::create(builder, loc, spanRounded, stepValue);
      mlir::Value thunkAddr = LLVM::AddressOfOp::create(builder, loc, thunk);
      LLVM::CallOp::create(builder, loc, parallelFor,
                           mlir::ValueRange{thunkAddr, ctxSlot, trips});
      call.erase();
    }
  }
  (void)i64;
  return mlir::success();
}

} // namespace py::lowering
