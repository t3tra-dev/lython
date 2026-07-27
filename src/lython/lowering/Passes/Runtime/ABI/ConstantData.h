#pragma once

// Storage for a literal's bytes/words: a shared read-only `memref.global`, not a
// frame slot.
//
// Why this exists as its own altitude. A `str`, `bytes`, or beyond-i64 `int`
// literal is lowered by handing its content to the contract's `__new__` as a
// rank-1 memref, and every one of those initializers allocates its own payload
// and copies out of what it was given (`LyUnicode_FromBytes` /
// `LyBytes_FromBytes` / `LyLong_FromDigits`). The block is therefore a
// compile-time-constant ARGUMENT, never the object's payload, and its storage is
// a physical-ABI question rather than an ownership one: it is `NonObject` -- no
// refcount word, no class id, no deallocator, no release obligation -- both
// before and after this change, so nothing in the ownership model moves.
//
// Why NOT `memref.alloca` plus a store per element, which is what each site did:
// an alloca outside the entry block is not `AllocaInst::isStaticAlloca()`, so
// LLVM lowers it to a runtime stack adjustment that nothing reclaims until the
// function returns. A literal in a loop body grew the frame by its own size on
// every iteration -- measured at 275,000 iterations of a 20-byte `str` before the
// stack guard raised RecursionError, and 300,000 of a 60-digit `int`.
//
// Why NOT the per-site entry-block hoist that `box_abi::allocaBoxWords` uses: a
// hoisted slot still has to be written on each execution, and there is nothing to
// write. The content is known when the global is created.
//
// Why `constant`, beyond being true: it places the block in read-only data, so a
// future consumer that tried to write through it would fault rather than silently
// corrupt every other occurrence of the same literal.

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/xxhash.h"

#include <string>

namespace py::lowering::constant_data {

// Interns `elements` as a private read-only global of `module` and returns a
// dynamic-extent rank-1 view of it, built at the builder's current insertion
// point. `kind` names the family in the symbol; `contentKey` identifies the
// content for the symbol name.
//
// Why `contentKey` is passed in rather than hashed off the attribute: attribute
// hashing is by uniqued-storage pointer, which is not stable across processes, so
// deriving the symbol from it would make the emitted IR differ run to run. The
// caller has the literal's own text, which is stable by construction. And why the
// name is content-derived rather than a counter: a counter makes the symbol
// depend on lowering order. The stored value is compared before any name is
// reused, so a key collision costs a suffix rather than the wrong content.
inline mlir::Value internReadOnlyBlock(mlir::ModuleOp module,
                                       mlir::OpBuilder &builder,
                                       mlir::Location loc, llvm::StringRef kind,
                                       llvm::StringRef contentKey,
                                       mlir::DenseElementsAttr elements) {
  auto shaped = llvm::cast<mlir::ShapedType>(elements.getType());
  mlir::Type elementType = shaped.getElementType();
  auto globalType = mlir::MemRefType::get(shaped.getShape(), elementType);

  std::string name;
  for (unsigned suffix = 0;; ++suffix) {
    name = ("__ly_const_" + kind + "_").str();
    name += llvm::utohexstr(llvm::xxh3_64bits(contentKey), /*LowerCase=*/true);
    name += "_" + std::to_string(shaped.getNumElements());
    if (suffix != 0)
      name += "_" + std::to_string(suffix);

    // Why any symbol and not just a memref.global: a name taken by some other op
    // is still taken, and defining it twice fails the module's symbol-table
    // verifier rather than reusing anything.
    mlir::Operation *existing = module.lookupSymbol(name);
    if (!existing) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      mlir::memref::GlobalOp::create(builder, loc, name,
                                     builder.getStringAttr("private"),
                                     globalType, elements,
                                     /*constant=*/true, /*alignment=*/nullptr);
      break;
    }
    auto reusable = mlir::dyn_cast<mlir::memref::GlobalOp>(existing);
    if (reusable && reusable.getConstant() &&
        reusable.getType() == globalType &&
        reusable.getInitialValueAttr() == elements)
      break;
    // Same name, different content: take the next suffix, never the block.
  }

  mlir::Value global =
      mlir::memref::GetGlobalOp::create(builder, loc, globalType, name)
          .getResult();
  // The consumers all take a dynamic extent plus an explicit element count, so
  // the static extent is cast away here rather than at each call site.
  auto dynamicType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
  return mlir::memref::CastOp::create(builder, loc, dynamicType, global)
      .getResult();
}

} // namespace py::lowering::constant_data
