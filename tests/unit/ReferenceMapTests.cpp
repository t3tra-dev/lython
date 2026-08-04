// What this pins: that a name reaching an entity and a name denoting a
// REFERENCE to it are different questions, and which answer `own::ReferenceMap`
// gives to the second.
//
// A unit test rather than a golden because the property is about IR shape, not
// about a runtime value: every case below is decided by looking at one or two
// ops. The behaviours it separates do have runtime consequences -- confusing a
// retain-minted token with the reference it was minted on is the defect run in
// rfc/test-suite-debt.md, seven fixes over one sentence -- but a golden can only
// see the SUM of the releases, which is exactly what stayed correct while the
// attribution did not. That is why this is checked from the inside.
//
// The map is the only place the question is answered. These cases are its
// contract: the three that produce a reference, the one that renames without
// producing, and the two "no claim" answers that keep a caller conservative.

#include "Reference.h"

#include "Contracts.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace {

namespace own = py::ownership;

struct Fixture {
  mlir::MLIRContext context;
  mlir::OpBuilder builder{&context};
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp host;

  Fixture() {
    context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    host = mlir::func::FuncOp::create(
        builder, builder.getUnknownLoc(), "host",
        builder.getFunctionType({}, {}));
    host.addEntryBlock();
    builder.setInsertionPointToStart(&host.getBody().front());
  }

  mlir::Type handle() { return mlir::MemRefType::get({2}, builder.getI64Type()); }

  // A callee whose result 0 the caller receives a reference to.
  mlir::func::FuncOp owningCallee(llvm::StringRef name) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module->getBody());
    auto fn = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(), name,
                                         builder.getFunctionType({}, {handle()}));
    fn.setPrivate();
    fn->setAttr(own::kOwnedResultsAttr,
                builder.getI64ArrayAttr({0}));
    return fn;
  }

  // The manifest's retain primitive.
  mlir::func::FuncOp retainCallee() {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module->getBody());
    auto fn = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(),
                                         "Ly_IncRef",
                                         builder.getFunctionType({handle()}, {}));
    fn.setPrivate();
    fn->setAttr(py::contracts::kManifestPrimitiveAttr,
                builder.getStringAttr("retain"));
    return fn;
  }

  mlir::Value call(mlir::func::FuncOp callee, mlir::ValueRange args = {}) {
    auto op = mlir::func::CallOp::create(builder, builder.getUnknownLoc(),
                                         callee, args);
    return op.getNumResults() ? op.getResult(0) : mlir::Value{};
  }

  // The identity cast the lowering spells an owned-local token as.
  mlir::Value marker(mlir::Value input) {
    auto cast = mlir::UnrealizedConversionCastOp::create(
        builder, builder.getUnknownLoc(), mlir::TypeRange{input.getType()},
        mlir::ValueRange{input});
    cast->setAttr(own::kOwnedLocalObjectAttr, builder.getUnitAttr());
    cast->setAttr(own::kOwnedLocalObjectContractAttr,
                  builder.getStringAttr("builtins.int"));
    return cast.getResult(0);
  }
};

// Everything the map needs, built the way the passes build it.
struct Analysis {
  explicit Analysis(mlir::ModuleOp module)
      : contracts(module), aliases(), map(contracts, aliases) {
    aliases.build(module);
  }
  py::ownership::FuncContractCache contracts;
  own::AliasAnalysis aliases;
  own::ReferenceMap map;
};

TEST(ReferenceMapTest, AnOwnedCallResultNamesItsOwnReference) {
  Fixture f;
  mlir::Value produced = f.call(f.owningCallee("make"));
  mlir::func::ReturnOp::create(f.builder, f.builder.getUnknownLoc());

  Analysis a(*f.module);
  own::Reference reference = a.map.of(produced);
  ASSERT_TRUE(static_cast<bool>(reference));
  EXPECT_EQ(reference.creator, produced.getDefiningOp());
  // Received from a call, not taken on top of one: a liveness walk may not drop
  // pins under this name, because it may be the reference that just died.
  EXPECT_FALSE(a.map.isMinted(reference));
}

TEST(ReferenceMapTest, ARetainMintedTokenIsNotTheReferenceItWasMintedOn) {
  Fixture f;
  mlir::Value produced = f.call(f.owningCallee("make"));
  mlir::func::CallOp::create(f.builder, f.builder.getUnknownLoc(),
                             f.retainCallee(), mlir::ValueRange{produced});
  mlir::Value token = f.marker(produced);
  mlir::func::ReturnOp::create(f.builder, f.builder.getUnknownLoc());

  Analysis a(*f.module);
  own::Reference source = a.map.of(produced);
  own::Reference minted = a.map.of(token);
  ASSERT_TRUE(static_cast<bool>(source));
  ASSERT_TRUE(static_cast<bool>(minted));

  // THE WHOLE POINT. `underlyingObjectValue` walks through the marker's cast, so
  // the two share an entity root and every alias; they are still two increments
  // with two releases.
  EXPECT_NE(source, minted);
  EXPECT_EQ(own::underlyingObjectValue(token), produced);
  EXPECT_TRUE(a.aliases.same(token, produced));

  // And the minted one is the one a liveness walk may drop pins under.
  EXPECT_TRUE(a.map.isMinted(minted));
  EXPECT_FALSE(a.map.isMinted(source));
}

TEST(ReferenceMapTest, AMarkerWithNoRetainRepublishesRatherThanMints) {
  Fixture f;
  mlir::Value produced = f.call(f.owningCallee("make"));
  mlir::Value republished = f.marker(produced); // no retain beside it
  mlir::func::ReturnOp::create(f.builder, f.builder.getUnknownLoc());

  Analysis a(*f.module);
  // Same attribute, opposite answer: it adds no increment, so it denotes what it
  // was given. A walk that treated it as its own reference would disown the
  // release of the one it shares.
  EXPECT_EQ(a.map.of(republished), a.map.of(produced));
  EXPECT_FALSE(a.map.isMinted(a.map.of(republished)));
}

TEST(ReferenceMapTest, ACastOfATokenStillDenotesTheToken) {
  Fixture f;
  mlir::Value produced = f.call(f.owningCallee("make"));
  mlir::func::CallOp::create(f.builder, f.builder.getUnknownLoc(),
                             f.retainCallee(), mlir::ValueRange{produced});
  mlir::Value token = f.marker(produced);
  auto view = mlir::memref::CastOp::create(
      f.builder, f.builder.getUnknownLoc(),
      mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                            f.builder.getI64Type()),
      token);
  mlir::func::ReturnOp::create(f.builder, f.builder.getUnknownLoc());

  Analysis a(*f.module);
  // A rename carries the reference. The set-of-marker-ops this replaced could
  // only recognise the marker itself, so a cast of it read as "no claim" and the
  // walk kept a pin it did not own.
  EXPECT_EQ(a.map.of(view.getResult()), a.map.of(token));
  EXPECT_NE(a.map.of(view.getResult()), a.map.of(produced));
}

TEST(ReferenceMapTest, AnUnownedResultAndABlockArgumentMakeNoClaim) {
  Fixture f;
  mlir::OpBuilder::InsertionGuard guard(f.builder);
  f.builder.setInsertionPointToEnd(f.module->getBody());
  auto borrowing = mlir::func::FuncOp::create(
      f.builder, f.builder.getUnknownLoc(), "borrow",
      f.builder.getFunctionType({}, {f.handle()}));
  borrowing.setPrivate(); // no owned_results: the caller receives nothing
  f.builder.setInsertionPointToStart(&f.host.getBody().front());
  mlir::Value borrowed = f.call(borrowing);
  mlir::func::ReturnOp::create(f.builder, f.builder.getUnknownLoc());

  Analysis a(*f.module);
  // No claim is not "no reference": it is "this analysis cannot name one", and
  // every caller must read it as "could be mine". That is what keeps a walk's
  // pre-existing conservative behaviour where the map has nothing to add.
  EXPECT_FALSE(static_cast<bool>(a.map.of(borrowed)));
  EXPECT_FALSE(static_cast<bool>(
      a.map.of(f.host.getBody().front().addArgument(f.handle(),
                                                    f.builder.getUnknownLoc()))));
}

} // namespace
