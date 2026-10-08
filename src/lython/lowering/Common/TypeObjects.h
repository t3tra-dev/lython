#pragma once

// A class is its type object: an immortal static named after the class's
// qualified name (`__ly_type.builtins.int`, `__ly_type.__main__.Point`), and
// an object's header word 1 holds its address, as CPython's `ob_type` does.
// Two classes are the same class exactly when they are the same symbol, so
// nothing allocates numbers and nothing can collide.
//
// The words of a type object, each an i64 (an address word holds the address
// zero-extended; every supported target is little-endian):
//
//   0  refcount, INT64_MAX: immortal, as every static object here
//   1  ob_type: 0 -- a type object is not yet a value a program can hold
//   2  tp_base: the base's type object, 0 at the root (object)
//   3  tp_name: a NUL-terminated name, the one a default repr prints
//   4  the name's length, without the NUL
//   5  tp_mro: the class and its ancestors in method resolution order, an
//      array of type-object addresses ending in 0
//   6.. the slots, CPython's tp_repr, tp_hash, ...: each the function a boxed
//      value of the class is dispatched to for one boxed hook
//      (`__ly_repr_boxed_by_contract`, ...), or 0 where the class has none
//      and the hook misses
//
// Every module that names a class declares its type object and, before it is
// translated, defines what it declared (`defineDeclared`) `linkonce_odr`: the
// program and each runtime module it links carry the same definition of a
// runtime class, and the link keeps one.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace py::type_objects {

inline constexpr llvm::StringLiteral kSymbolPrefix{"__ly_type."};

inline constexpr std::int64_t kRefcountWord = 0;
inline constexpr std::int64_t kTypeWord = 1;
inline constexpr std::int64_t kBaseWord = 2;
inline constexpr std::int64_t kNameWord = 3;
inline constexpr std::int64_t kNameLengthWord = 4;
inline constexpr std::int64_t kMroWord = 5;

// The boxed hooks, each with its slot word. A unary hook is `(ptr slot, i64
// class) -> (results..., i1 hit)` and its slot `(ptr slot) -> (results..., i1)`;
// a binary one is `(ptr lhs, ptr rhs, i64 class, i64 rhs class)` and its slot
// `(ptr lhs, ptr rhs, i64 rhs class)`.
struct SlotHook {
  llvm::StringLiteral name;
  std::int64_t word;
  bool binary;
  // The native runtime calls it (release_payload_slot_ptr), so the program
  // defines it whether or not anything of its own does.
  bool calledByRuntime = false;
};
inline constexpr SlotHook kSlotHooks[] = {
    {"__ly_repr_boxed_by_contract", 6, false},
    {"__ly_str_boxed_by_contract", 7, false},
    {"__ly_hash_boxed_by_contract", 8, false},
    {"__ly_eq_boxed_by_contract", 9, true},
    {"__ly_lt_boxed_by_contract", 10, true},
    {"__ly_le_boxed_by_contract", 11, true},
    {"__ly_gt_boxed_by_contract", 12, true},
    {"__ly_ge_boxed_by_contract", 13, true},
    {"__ly_release_boxed_by_contract", 14, false, true},
};
inline constexpr std::int64_t kWordCount = 15;
const SlotHook *slotHookNamed(llvm::StringRef hookName);

// A program's source classes, recorded while their `py.class` ops still exist
// (the runtime lowering erases them): [qualified name, display name, base,
// MRO after the class...].
inline constexpr llvm::StringLiteral kSourceTypesAttr{"ly.type_objects"};

std::string symbolFor(llvm::StringRef qualifiedName);
std::optional<llvm::StringRef> qualifiedNameOf(llvm::StringRef symbol);

// The class's type-object address, as the i64 word a header keeps. Declares
// the type object in the module the builder is inserting into.
mlir::Value classWord(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef qualifiedName);
mlir::Value classWord(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::ModuleOp module, llvm::StringRef qualifiedName);
void declare(mlir::ModuleOp module, llvm::StringRef qualifiedName);

struct Definition {
  std::string qualifiedName;
  // What tp_name holds: the name a default repr prints.
  std::string name;
  // The base's qualified name; empty only for object.
  std::string base;
  // The ancestors in method resolution order, the class itself left out.
  std::vector<std::string> mro;
};

// A runtime class's definition (RuntimeClasses.h): its base is CPython's --
// the exception taxonomy's for an exception, int for bool, object otherwise.
std::optional<Definition> runtimeDefinition(llvm::StringRef contract);

// Records a source class for `defineDeclared`.
void recordSource(mlir::ModuleOp module, const Definition &definition);

// Records that `qualifiedName`'s slot for `hook` is the function `slot`.
// `defineDeclared` defines every class with a slot, referenced or not: a
// runtime module may make an object of it (an OSError subclass, from errno)
// that this program's hooks then dispatch, and the program's definition is
// the one the link keeps.
void recordSlot(mlir::ModuleOp module, llvm::StringRef qualifiedName,
                const SlotHook &hook, llvm::StringRef slot);

// After the conversion to the LLVM dialect: gives each declared boxed hook
// its body -- the class's slot, called, or a miss where it has none.
mlir::LogicalResult defineSlotHooks(mlir::ModuleOp module);

// Defines every type object the module declares, and the bases they name.
// A declared name that is neither a runtime class nor a recorded source class
// is an error: something named a class nothing defines.
mlir::LogicalResult defineDeclared(mlir::ModuleOp module, unsigned pointerBits);

// The manifests spell a class word `arith.constant {ly.class_of = "X"}`;
// each becomes `classWord(X)`. Run once, right after the manifests are
// imported, before anything folds a constant.
mlir::LogicalResult resolveManifestClassWords(mlir::ModuleOp module);

// A manifest static object (`memref.global` with `ly.class_of`, and
// `ly.class_stride` for a byte image of several records) holds its class
// word at byte 8 of every record. A dense initializer cannot spell an
// address, so the class is recorded before the conversion to the LLVM
// dialect and written into the converted global after it.
struct StaticClassWords {
  struct Entry {
    std::string qualifiedName;
    std::int64_t stride = 0; // 0: the whole image is one record
    // `ly.static.shared`: linkonce_odr, so every module's copy is one object.
    bool shared = false;
  };
  llvm::StringMap<Entry> globals;
};
mlir::FailureOr<StaticClassWords> collectStaticClassWords(mlir::ModuleOp module);
mlir::LogicalResult patchStaticClassWords(mlir::ModuleOp module,
                                          const StaticClassWords &words,
                                          unsigned pointerBits);

// A static image laid out word by word, for an LLVM global's initializer: a
// field is bytes as written, or a symbol's address plus a byte offset into it
// in a word. ⛔ Not a ptrtoint to i64 for an address on a 32-bit target:
// widening an address is not a relocation wasm-ld or an ELF linker can express
// ("unsupported expression in static initializer"); the word is the pointer
// and a zero half, low half first (every supported target is little-endian).
struct ImageField {
  llvm::SmallVector<std::int8_t, 16> bytes;
  std::string symbol;
  std::int64_t offset = 0;
};
void appendImageWord(llvm::SmallVectorImpl<ImageField> &fields,
                     std::int64_t word);
void appendImageBytes(llvm::SmallVectorImpl<ImageField> &fields,
                      llvm::ArrayRef<std::int8_t> bytes);
void appendImageAddress(llvm::SmallVectorImpl<ImageField> &fields,
                        llvm::StringRef symbol, std::int64_t offset = 0);
// Gives `global` a packed-struct type and an initializer that is `fields`.
void setImage(mlir::OpBuilder &builder, mlir::Location loc,
              mlir::LLVM::GlobalOp global, llvm::ArrayRef<ImageField> fields,
              unsigned pointerBits);

// The pointer width the module is lowered for.
unsigned pointerBitsOf(mlir::ModuleOp module);

} // namespace py::type_objects
