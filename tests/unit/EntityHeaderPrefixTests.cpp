// What this pins: which producers of a rank-1 i64 handle have already stored the
// entity's refcount/class prefix by the time the handle is available, which is
// the question `borrowEdgeRetainIsSpellable` asks before writing a borrow-edge
// retain through a prefix view.
//
// This is a unit test rather than a golden because the property is arithmetic on
// offsets and provenance, not a runtime value: every case below is decided by
// looking at one op. The behaviours it separates DO have runtime consequences --
// a wrong "yes" on raw storage aborts with `Ly_IncRef observed non-positive
// refcount`, a wrong "no" on a call result ships an over-release surviving
// `--release` -- and `tests/golden/cases/dict_key_mutation*` and
// `cross_container_box_fronted_fields` pin the first from the outside. Neither
// of those can pin the second, because the population that needs the retain
// (a one-lane container handle merged across two groups) only aborts when a loop
// COMPLETES, and no golden covering it existed when this was written.
//
// The width columns are deliberate: 9 is `builtins.list` and 16 is both
// `builtins.object` AND the transient payload box (ABI/HandleWidthRegistry.h),
// so the tests state that the answer does NOT come from the width.

#include "Runtime/ABI/EntityHeaderPrefix.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <gtest/gtest.h>

namespace {

using namespace py::lowering;

class HandleFixture {
public:
  HandleFixture() {
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>();
    loc = mlir::UnknownLoc::get(&context);
    module = mlir::ModuleOp::create(loc);
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(module->getBody());
    function = mlir::func::FuncOp::create(
        builder, loc, "probe",
        builder.getFunctionType({memrefOf(9)}, {memrefOf(9)}));
    function.addEntryBlock();
  }

  mlir::MemRefType memrefOf(std::int64_t words) {
    return mlir::MemRefType::get({words}, mlir::IntegerType::get(&context, 64));
  }

  mlir::OpBuilder bodyBuilder() {
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(&function.getBody().front());
    return builder;
  }

  // The entry block argument, which is the producer the shipped predicate
  // already accepted.
  mlir::Value entryArgument() { return function.getBody().front().getArgument(0); }

  mlir::Value alloc(std::int64_t words) {
    mlir::OpBuilder builder = bodyBuilder();
    return mlir::memref::AllocOp::create(builder, loc, memrefOf(words))
        .getResult();
  }

  mlir::Value alloca_(std::int64_t words) {
    mlir::OpBuilder builder = bodyBuilder();
    return mlir::memref::AllocaOp::create(builder, loc, memrefOf(words))
        .getResult();
  }

  // A runtime allocator/constructor result: an entity its callee finished.
  mlir::Value callResult(std::int64_t words) {
    mlir::OpBuilder declare(&context);
    declare.setInsertionPointToStart(module->getBody());
    std::string name = "make_" + std::to_string(words);
    if (!module->lookupSymbol(name)) {
      auto declared = mlir::func::FuncOp::create(
          declare, loc, name, declare.getFunctionType({}, {memrefOf(words)}));
      declared.setPrivate();
    }
    mlir::OpBuilder builder = bodyBuilder();
    return mlir::func::CallOp::create(builder, loc, name,
                                      mlir::TypeRange{memrefOf(words)},
                                      mlir::ValueRange{})
        .getResult(0);
  }

  mlir::Value prefixSubview(mlir::Value source, std::int64_t offset,
                            std::int64_t size) {
    mlir::OpBuilder builder = bodyBuilder();
    llvm::SmallVector<mlir::OpFoldResult, 1> offsets{builder.getIndexAttr(offset)};
    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{builder.getIndexAttr(size)};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
    llvm::SmallVector<std::int64_t, 1> resultShape{size};
    auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());
    auto resultType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            resultShape, sourceType, offsets, sizes, strides));
    return mlir::memref::SubViewOp::create(builder, loc, resultType, source,
                                           offsets, sizes, strides)
        .getResult();
  }

  mlir::Value dynamicCast(mlir::Value source) {
    mlir::OpBuilder builder = bodyBuilder();
    auto dynamicType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                             builder.getI64Type());
    return mlir::memref::CastOp::create(builder, loc, dynamicType, source)
        .getResult();
  }

  mlir::MLIRContext context;
  mlir::Location loc = mlir::UnknownLoc::get(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
};

// --- the layout half -------------------------------------------------------

TEST(EntityHeaderPrefixTest, LayoutAcceptsEveryConvertedOneLaneWidth) {
  HandleFixture fixture;
  // Every width the registry assigns to a one-lane contract, plus the two-word
  // header interface itself. A width that stopped answering yes here would take
  // that contract's retains and releases with it.
  for (std::int64_t words : {2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 64})
    EXPECT_TRUE(entity_header::typeCarriesHeaderPrefix(fixture.memrefOf(words)))
        << "width " << words;
}

TEST(EntityHeaderPrefixTest, LayoutRejectsWhatCannotHoldTwoWords) {
  HandleFixture fixture;
  EXPECT_FALSE(entity_header::typeCarriesHeaderPrefix(fixture.memrefOf(1)));
  EXPECT_FALSE(entity_header::typeCarriesHeaderPrefix(fixture.memrefOf(0)));
  // A literal's constant block is i8 and is NonObject -- no refcount word, no
  // class id, no deallocator (ABI/ConstantData.h). It is not in the population.
  EXPECT_FALSE(entity_header::typeCarriesHeaderPrefix(mlir::MemRefType::get(
      {32}, mlir::IntegerType::get(&fixture.context, 8))));
  // Rank is part of the layout claim: words 0 and 1 of a 2-D buffer are not a
  // prefix of anything.
  EXPECT_FALSE(entity_header::typeCarriesHeaderPrefix(mlir::MemRefType::get(
      {4, 4}, mlir::IntegerType::get(&fixture.context, 64))));
  EXPECT_FALSE(entity_header::typeCarriesHeaderPrefix(
      mlir::IntegerType::get(&fixture.context, 64)));
}

TEST(EntityHeaderPrefixTest, LayoutAcceptsADynamicExtent) {
  HandleFixture fixture;
  // The shared `memref<?xi64>` helper spelling set and frozenset both cast to.
  EXPECT_TRUE(entity_header::typeCarriesHeaderPrefix(mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic},
      mlir::IntegerType::get(&fixture.context, 64))));
}

// --- the provenance half, which is what actually separates the cases --------

TEST(EntityHeaderPrefixTest, ABlockArgumentIsInitialized) {
  HandleFixture fixture;
  EXPECT_TRUE(
      entity_header::prefixIsInitializedAtDefinition(fixture.entryArgument()));
}

TEST(EntityHeaderPrefixTest, ACallResultIsInitialized) {
  HandleFixture fixture;
  // memref<9xi64> is `builtins.list`: the header the shipped over-release needed
  // and did not get, because it arrives as a call result rather than a block
  // argument (rfc/stdlib-semantics.md, family D).
  EXPECT_TRUE(
      entity_header::prefixIsInitializedAtDefinition(fixture.callResult(9)));
}

TEST(EntityHeaderPrefixTest, RawStorageIsNotInitialized) {
  HandleFixture fixture;
  // memref<16xi64> by memref.alloc is the transient payload box, whose prefix
  // `boxRuntimeObject` stores in the ops AFTER the alloc. The three goldens that
  // the naive widening breaks are exactly this shape.
  EXPECT_FALSE(entity_header::prefixIsInitializedAtDefinition(fixture.alloc(16)));
  EXPECT_FALSE(entity_header::prefixIsInitializedAtDefinition(fixture.alloca_(16)));
}

TEST(EntityHeaderPrefixTest, WidthDoesNotDecide) {
  HandleFixture fixture;
  // The same width answers both ways, and the same producer answers the same way
  // at both widths. That is the statement the recorded justification got wrong:
  // it named layout as the separator when both populations carry the prefix.
  EXPECT_TRUE(entity_header::prefixIsInitializedAtDefinition(
      fixture.callResult(16)));
  EXPECT_FALSE(entity_header::prefixIsInitializedAtDefinition(fixture.alloc(16)));
  EXPECT_TRUE(entity_header::prefixIsInitializedAtDefinition(
      fixture.callResult(9)));
  EXPECT_FALSE(entity_header::prefixIsInitializedAtDefinition(fixture.alloc(9)));
}

TEST(EntityHeaderPrefixTest, ACastIsFollowedToItsSource) {
  HandleFixture fixture;
  EXPECT_TRUE(entity_header::prefixIsInitializedAtDefinition(
      fixture.dynamicCast(fixture.callResult(11))));
  EXPECT_FALSE(entity_header::prefixIsInitializedAtDefinition(
      fixture.dynamicCast(fixture.alloc(11))));
}

TEST(EntityHeaderPrefixTest, AZeroOffsetPrefixSubviewIsFollowed) {
  HandleFixture fixture;
  // This is the view `spellHeaderPrefix` actually emits: the leading two words
  // of a wider handle.
  EXPECT_TRUE(entity_header::prefixIsInitializedAtDefinition(
      fixture.prefixSubview(fixture.callResult(9), /*offset=*/0, /*size=*/2)));
}

TEST(EntityHeaderPrefixTest, ANonZeroOffsetSubviewIsNotAPrefix) {
  HandleFixture fixture;
  // Word 0 of a view taken at offset 2 is the container's LENGTH word
  // (ABI/ContainerLayout.h), not a refcount. Following to the source here would
  // authorise a retain against the wrong word of an initialised entity, which is
  // worse than declining: it would be reading a plausible small integer.
  EXPECT_FALSE(entity_header::prefixIsInitializedAtDefinition(
      fixture.prefixSubview(fixture.callResult(9), /*offset=*/2, /*size=*/2)));
}

} // namespace
