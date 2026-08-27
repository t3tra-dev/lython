#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace py {

class ClassOp;

namespace type_object {

inline constexpr llvm::StringLiteral kBaseException = "BaseException";
inline constexpr llvm::StringLiteral kException = "Exception";

ClassOp lookup(mlir::Operation *from, llvm::StringRef name);
mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>>
mroNames(mlir::Operation *from, llvm::StringRef name);
mlir::LogicalResult verifyBases(ClassOp op);
mlir::FailureOr<bool> isSubclassOf(mlir::Operation *from,
                                   llvm::StringRef derived,
                                   llvm::StringRef base);
// Quiet subtype QUERY: false (no diagnostic) when either name is not a
// py.class symbol. Use from subtype lattice checks; isSubclassOf is for
// verification sites where an unknown name is an error.
// The value the class body bound to a class-level name, or nothing when the
// class has no such static. Here rather than in the lowering: the EMITTER
// writes `class_static_attr_names`/`_values` and two lowering files read them
// back, each with its own copy of the pairing rule -- one attribute layout,
// one accessor.
// Value paired with `name` in an op's parallel `<...>_names` / `<...>_values`
// array attributes, the layout the emitter uses for every static binding it
// hands to the lowering (class statics, module statics).
std::optional<mlir::Attribute> pairedAttributeValue(mlir::Operation *op,
                                                    llvm::StringRef namesAttr,
                                                    llvm::StringRef valuesAttr,
                                                    llvm::StringRef name);
std::optional<mlir::Attribute> staticAttributeValue(ClassOp classOp,
                                                    llvm::StringRef name);
// ⛔ THE BASE NEED NOT BE A `py.class`, which is what separates this from
// `isKnownSubclassOf`. A source class records its base by the bare NAME it was
// written with, and a manifest base -- `class MyErr(Exception)` -- has no
// symbol to look up here, so that walk answered False and `MyErr` was not an
// `Exception` to anything that asked the lattice.
bool reachesDeclaredBase(mlir::Operation *from, llvm::StringRef derived,
                         llvm::StringRef base);

bool isKnownSubclassOf(mlir::Operation *from, llvm::StringRef derived,
                       llvm::StringRef base);
} // namespace type_object
} // namespace py
