// A static object's read-only image: an object laid out at compile time with
// the immortal refcount, read back through its contract's `from_static`
// primitive (a constant tuple, a bytes literal).
//
// ⛔ Not a memref.global: an image whose words point into itself (a tuple's
// items address, a bytes payload address) needs a relocation, which a dense
// initializer cannot spell; an LLVM global's initializer region can.

#include "Runtime/Core/Lowerer.h"
#include "Runtime/Ctypes/Internal.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"

namespace py::lowering {

mlir::Value RuntimeBundleLowerer::materializeStaticObjectAddress(
    mlir::Location loc, llvm::StringRef kind, llvm::StringRef contentKey,
    llvm::ArrayRef<std::int64_t> words,
    llvm::ArrayRef<std::pair<unsigned, std::int64_t>> selfAddressWords,
    llvm::ArrayRef<std::int8_t> tail) {
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i8 = builder.getI8Type();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  // ⭐ A WORD IS TWO i32 HALVES ON A 32-BIT TARGET, low half first (every
  // supported target is little-endian), so a relocated word is the pointer
  // and a zero. ⛔ Not a ptrtoint to i64: widening an address is not a
  // relocation wasm-ld or an ELF linker can express ("unsupported expression
  // in static initializer").
  auto facts = ctypes::targetPlatformFacts(module);
  bool narrow = facts && facts->pointerWidth == 32;
  mlir::Type half = builder.getI32Type();
  auto wordsType = narrow ? mlir::LLVM::LLVMArrayType::get(half, words.size() * 2)
                          : mlir::LLVM::LLVMArrayType::get(i64, words.size());
  auto tailType = mlir::LLVM::LLVMArrayType::get(i8, tail.size());
  auto imageType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {wordsType, tailType});
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
        builder, loc, imageType, /*isConstant=*/true,
        mlir::LLVM::Linkage::Private, name, mlir::Attribute(),
        /*alignment=*/16);
    mlir::Block *init = builder.createBlock(&global.getInitializerRegion());
    builder.setInsertionPointToStart(init);
    mlir::Value image =
        mlir::LLVM::UndefOp::create(builder, loc, imageType).getResult();
    mlir::Value self =
        mlir::LLVM::AddressOfOp::create(builder, loc, ptrType, name).getResult();
    auto insert = [&](mlir::Value value, std::int64_t position) {
      image = mlir::LLVM::InsertValueOp::create(
          builder, loc, image, value, llvm::ArrayRef<std::int64_t>{0, position});
    };
    for (auto [index, word] : llvm::enumerate(words)) {
      unsigned wordIndex = static_cast<unsigned>(index);
      auto relocated = llvm::find_if(selfAddressWords, [&](const auto &entry) {
        return entry.first == wordIndex;
      });
      std::int64_t position = static_cast<std::int64_t>(index);
      if (relocated != selfAddressWords.end()) {
        mlir::Value at = mlir::LLVM::GEPOp::create(
            builder, loc, ptrType, i8, self,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{
                static_cast<std::int32_t>(relocated->second)});
        if (narrow) {
          insert(mlir::LLVM::PtrToIntOp::create(builder, loc, half, at),
                 2 * position);
          insert(mlir::LLVM::ConstantOp::create(builder, loc, half,
                                                builder.getI32IntegerAttr(0)),
                 2 * position + 1);
        } else {
          insert(mlir::LLVM::PtrToIntOp::create(builder, loc, i64, at),
                 position);
        }
        continue;
      }
      if (narrow) {
        auto bits = static_cast<std::uint64_t>(word);
        insert(mlir::LLVM::ConstantOp::create(
                   builder, loc, half,
                   builder.getI32IntegerAttr(
                       static_cast<std::int32_t>(bits & 0xFFFFFFFFu))),
               2 * position);
        insert(mlir::LLVM::ConstantOp::create(
                   builder, loc, half,
                   builder.getI32IntegerAttr(
                       static_cast<std::int32_t>(bits >> 32))),
               2 * position + 1);
      } else {
        insert(mlir::LLVM::ConstantOp::create(builder, loc, i64,
                                              builder.getI64IntegerAttr(word)),
               position);
      }
    }
    if (!tail.empty()) {
      auto bytes = mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({static_cast<std::int64_t>(tail.size())},
                                      i8),
          tail);
      mlir::Value tailValue =
          mlir::LLVM::ConstantOp::create(builder, loc, tailType, bytes);
      image = mlir::LLVM::InsertValueOp::create(
          builder, loc, image, tailValue, llvm::ArrayRef<std::int64_t>{1});
    }
    mlir::LLVM::ReturnOp::create(builder, loc, image);
  }
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, loc, ptrType, name).getResult();
  return mlir::LLVM::PtrToIntOp::create(builder, loc, i64, address);
}

} // namespace py::lowering
