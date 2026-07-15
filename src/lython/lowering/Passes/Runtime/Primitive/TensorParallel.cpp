#include "TensorParallel.h"

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
// Rows per chunk must stay a multiple of the largest macro M tile (the
// packed schedule's MC = 128): a chunk narrower than the tile hands the
// packed pipeline a peeled-remainder-only nest it does not support.
constexpr int64_t kMatmulParallelRowQuantum = 128;

constexpr llvm::StringLiteral kParallelDispatchAttr{"ly.parallel.dispatch"};
constexpr llvm::StringLiteral kParallelBodyAttr{"ly.parallel.body"};

bool staticIdentityLike(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  return type && type.hasStaticShape() &&
         isPrimitiveElementType(type.getElementType());
}

std::optional<int64_t> selectChunkCount(int64_t rows) {
  int64_t bound = std::min<int64_t>(kMatmulParallelMaxChunks,
                                    rows / kMatmulParallelRowQuantum);
  for (int64_t chunks = bound; chunks >= 2; --chunks)
    if (rows % chunks == 0 &&
        (rows / chunks) % kMatmulParallelRowQuantum == 0)
      return chunks;
  return std::nullopt;
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
    llvm::SmallVector<mlir::linalg::MatmulOp, 8> candidates;
    getOperation().walk([&](mlir::linalg::MatmulOp matmul) {
      if (matmul->getNumResults() != 0)
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
      std::uint64_t work = 1;
      for (int64_t dim : {lhsType.getDimSize(0), rhsType.getDimSize(1),
                          lhsType.getDimSize(1)}) {
        if (dim <= 0)
          return;
        work *= static_cast<std::uint64_t>(dim);
      }
      if (work < kMatmulParallelMinWork)
        return;
      if (!selectChunkCount(lhsType.getDimSize(0)))
        return;
      candidates.push_back(matmul);
    });

    mlir::OpBuilder builder(&getContext());
    for (mlir::linalg::MatmulOp matmul : candidates) {
      mlir::Value lhs = matmul.getDpsInputOperand(0)->get();
      mlir::Value out = matmul.getDpsInitOperand(0)->get();
      auto lhsType = mlir::cast<mlir::MemRefType>(lhs.getType());
      auto outType = mlir::cast<mlir::MemRefType>(out.getType());
      int64_t m = lhsType.getDimSize(0);
      int64_t k = lhsType.getDimSize(1);
      int64_t n = outType.getDimSize(1);
      int64_t chunks = *selectChunkCount(m);
      int64_t rows = m / chunks;

      mlir::Location loc = matmul.getLoc();
      builder.setInsertionPoint(matmul);
      mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
      mlir::Value upper =
          mlir::arith::ConstantIndexOp::create(builder, loc, chunks);
      mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
      auto loop = mlir::scf::ForOp::create(builder, loc, zero, upper, one);
      loop->setAttr(kParallelDispatchAttr, builder.getUnitAttr());

      builder.setInsertionPointToStart(loop.getBody());
      mlir::Value rowsValue =
          mlir::arith::ConstantIndexOp::create(builder, loc, rows);
      mlir::Value rowStart = mlir::arith::MulIOp::create(
          builder, loc, loop.getInductionVar(), rowsValue);
      auto chunkView = [&](mlir::Value source,
                           int64_t columns) -> mlir::Value {
        return mlir::memref::SubViewOp::create(
                   builder, loc, source,
                   mlir::ArrayRef<mlir::OpFoldResult>{rowStart,
                                                      builder.getIndexAttr(0)},
                   mlir::ArrayRef<mlir::OpFoldResult>{
                       builder.getIndexAttr(rows),
                       builder.getIndexAttr(columns)},
                   mlir::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(1),
                                                      builder.getIndexAttr(1)})
            .getResult();
      };
      mlir::Value lhsChunk = chunkView(lhs, k);
      mlir::Value outChunk = chunkView(out, n);
      mlir::Operation *cloned = builder.clone(*matmul.getOperation());
      cloned->setOperand(0, lhsChunk);
      cloned->setOperand(2, outChunk);
      matmul.erase();
    }
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
constexpr llvm::StringLiteral kParallelWorkerName{"LyParallelWorker"};
constexpr llvm::StringLiteral kParallelThreadsName{"LyParallelThreads"};
constexpr llvm::StringLiteral kThreadsGlobalName{"__lython_parallel_threads"};
constexpr llvm::StringLiteral kThreadsEnvName{"__lython_parallel_env"};
// Default worker cap: matrix units are per-cluster on current Apple silicon
// and per-core SIMD saturates memory bandwidth quickly elsewhere, so a small
// pool captures most of the win; LYTHON_NUM_THREADS overrides.
constexpr int64_t kDefaultParallelThreads = 4;
constexpr int64_t kMaxParallelThreads = 64;

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
  mlir::Value fallback = constI64(builder, loc, kDefaultParallelThreads);
  LLVM::CondBrOp::create(builder, loc, hasEnv, haveEnv, mlir::ValueRange{env},
                         store, mlir::ValueRange{fallback});

  builder.setInsertionPointToStart(haveEnv);
  mlir::Value base = LLVM::ConstantOp::create(builder, loc, i32,
                                              builder.getI32IntegerAttr(10));
  mlir::Value parsed =
      LLVM::CallOp::create(builder, loc, strtolFn,
                           mlir::ValueRange{haveEnv->getArgument(0), null,
                                            base})
          .getResult();
  mlir::Value one = constI64(builder, loc, 1);
  mlir::Value cap = constI64(builder, loc, kMaxParallelThreads);
  mlir::Value belowOne = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::slt, parsed, one);
  mlir::Value atLeastOne =
      LLVM::SelectOp::create(builder, loc, belowOne, one, parsed);
  mlir::Value aboveCap = LLVM::ICmpOp::create(
      builder, loc, LLVM::ICmpPredicate::sgt, atLeastOne, cap);
  mlir::Value clamped =
      LLVM::SelectOp::create(builder, loc, aboveCap, cap, atLeastOne);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{clamped}, store);

  builder.setInsertionPointToStart(store);
  LLVM::StoreOp::create(builder, loc, store->getArgument(0), globalPtr);
  LLVM::BrOp::create(builder, loc, mlir::ValueRange{store->getArgument(0)},
                     done);

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

// LyParallelFor(ptr fn, ptr ctx, i64 n): split [0, n) across workers. The
// caller's thread runs the first range; the rest run on freshly spawned
// pthreads joined before returning, so no runtime state outlives the call
// and no worker ever touches the object runtime. A failed pthread_create
// falls back to running that range inline -- degraded parallelism, never a
// dropped range.
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
  buildParallelWorker(module, builder);
  buildParallelFor(module, builder);
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
