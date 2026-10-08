// A static object's read-only image: an object laid out at compile time with
// the immortal refcount, read back through its contract's `from_static`
// primitive (a constant tuple, a bytes literal).
//
// ⛔ Not a memref.global: an image whose words point into itself (a tuple's
// items address, a bytes payload address) needs a relocation, which a dense
// initializer cannot spell; an LLVM global's initializer region can.

#include "Runtime/Core/Lowerer.h"
#include "Common/TypeObjects.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"

namespace py::lowering {

mlir::Value RuntimeBundleLowerer::materializeStaticObjectAddress(
    mlir::Location loc, llvm::StringRef kind, llvm::StringRef runtimeClass,
    llvm::StringRef contentKey, llvm::ArrayRef<std::int64_t> words,
    llvm::ArrayRef<std::pair<unsigned, std::int64_t>> selfAddressWords,
    llvm::ArrayRef<std::int8_t> tail) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  // Content-derived and stable across processes, like constant_data's names;
  // the relocated words are part of the content only by position.
  std::string key = contentKey.str();
  for (std::int64_t word : words)
    key += "," + std::to_string(word);
  std::string name = ("__ly_static_" + kind + "_").str() +
                     llvm::utohexstr(llvm::xxh3_64bits(key), true) + "_" +
                     std::to_string(words.size()) + "_" +
                     std::to_string(tail.size());
  if (!module.lookupSymbol(name)) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto global = mlir::LLVM::GlobalOp::create(
        builder, loc, builder.getI64Type(), /*isConstant=*/true,
        mlir::LLVM::Linkage::Private, name, mlir::Attribute(),
        /*alignment=*/16);
    type_objects::declare(module, runtimeClass);
    llvm::SmallVector<type_objects::ImageField, 8> fields;
    for (auto [index, word] : llvm::enumerate(words)) {
      unsigned wordIndex = static_cast<unsigned>(index);
      // Word 1 is the class: its type object's address.
      if (wordIndex == 1) {
        type_objects::appendImageAddress(
            fields, type_objects::symbolFor(runtimeClass));
        continue;
      }
      auto relocated = llvm::find_if(selfAddressWords, [&](const auto &entry) {
        return entry.first == wordIndex;
      });
      if (relocated != selfAddressWords.end()) {
        type_objects::appendImageAddress(fields, name, relocated->second);
        continue;
      }
      type_objects::appendImageWord(fields, word);
    }
    type_objects::appendImageBytes(fields, tail);
    type_objects::setImage(builder, loc, global, fields,
                           type_objects::pointerBitsOf(module));
  }
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, loc, ptrType, name).getResult();
  return mlir::LLVM::PtrToIntOp::create(builder, loc, builder.getI64Type(),
                                        address);
}

} // namespace py::lowering
