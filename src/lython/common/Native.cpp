#include "Native.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Triple.h"

#include <optional>

namespace py::native {

std::uint64_t expectedPointerWidth(llvm::StringRef tripleText) {
  llvm::Triple triple(tripleText);
  if (triple.isArch64Bit())
    return 64;
  if (triple.isArch32Bit())
    return 32;
  return 0;
}

std::uint64_t expectedCLongWidth(llvm::StringRef tripleText,
                                 std::uint64_t pointerWidth) {
  llvm::Triple triple(tripleText);
  if (triple.isOSWindows())
    return 32;
  return pointerWidth == 64 ? 64 : 32;
}

bool isSupportedNativeTarget(llvm::StringRef tripleText) {
  llvm::Triple triple(tripleText);
  return triple.isOSDarwin() || triple.isOSLinux() || triple.isOSWindows();
}

std::optional<TargetPlatformFacts>
readTargetPlatformFacts(mlir::ModuleOp module) {
  auto triple = module->getAttrOfType<mlir::StringAttr>(kTargetTripleAttr);
  auto pointerWidth =
      module->getAttrOfType<mlir::IntegerAttr>(kTargetPointerWidthAttr);
  auto cLongWidth =
      module->getAttrOfType<mlir::IntegerAttr>(kTargetCLongWidthAttr);
  if (!triple || !pointerWidth || !cLongWidth)
    return std::nullopt;
  if (pointerWidth.getInt() <= 0 || cLongWidth.getInt() <= 0)
    return std::nullopt;

  TargetPlatformFacts facts;
  facts.triple = triple.getValue().str();
  facts.pointerWidth = static_cast<std::uint64_t>(pointerWidth.getInt());
  facts.cLongWidth = static_cast<std::uint64_t>(cLongWidth.getInt());
  if (facts.pointerWidth == 0 || facts.pointerWidth % 8 != 0 ||
      facts.cLongWidth == 0 || facts.cLongWidth % 8 != 0)
    return std::nullopt;
  return facts;
}

mlir::LogicalResult verifyTargetPlatformFacts(mlir::ModuleOp module) {
  auto tripleAttr = module->getAttrOfType<mlir::StringAttr>(kTargetTripleAttr);
  if (!tripleAttr)
    return module.emitError() << "missing " << kTargetTripleAttr;
  llvm::Triple triple(tripleAttr.getValue());
  if (triple.getArch() == llvm::Triple::UnknownArch)
    return module.emitError()
           << kTargetTripleAttr << " '" << tripleAttr.getValue()
           << "' has unknown architecture";
  if (!isSupportedNativeTarget(tripleAttr.getValue()))
    return module.emitError()
           << kTargetTripleAttr << " '" << tripleAttr.getValue()
           << "' is not supported by embedded native runtime selection";

  auto pointerWidthAttr =
      module->getAttrOfType<mlir::IntegerAttr>(kTargetPointerWidthAttr);
  if (!pointerWidthAttr)
    return module.emitError() << "missing " << kTargetPointerWidthAttr;
  if (pointerWidthAttr.getInt() <= 0 || pointerWidthAttr.getInt() % 8 != 0)
    return module.emitError() << kTargetPointerWidthAttr
                              << " must be a positive bit width divisible by 8";
  std::uint64_t pointerWidth =
      static_cast<std::uint64_t>(pointerWidthAttr.getInt());
  std::uint64_t expectedPointer = expectedPointerWidth(tripleAttr.getValue());
  if (expectedPointer == 0)
    return module.emitError()
           << "cannot derive pointer width from target triple '"
           << tripleAttr.getValue() << "'";
  if (pointerWidth != expectedPointer)
    return module.emitError()
           << kTargetPointerWidthAttr << " is " << pointerWidth
           << " but target triple '" << tripleAttr.getValue() << "' requires "
           << expectedPointer;

  auto cLongWidthAttr =
      module->getAttrOfType<mlir::IntegerAttr>(kTargetCLongWidthAttr);
  if (!cLongWidthAttr)
    return module.emitError() << "missing " << kTargetCLongWidthAttr;
  if (cLongWidthAttr.getInt() <= 0 || cLongWidthAttr.getInt() % 8 != 0)
    return module.emitError() << kTargetCLongWidthAttr
                              << " must be a positive bit width divisible by 8";
  std::uint64_t cLongWidth =
      static_cast<std::uint64_t>(cLongWidthAttr.getInt());
  std::uint64_t expectedCLong =
      expectedCLongWidth(tripleAttr.getValue(), pointerWidth);
  if (cLongWidth != expectedCLong)
    return module.emitError() << kTargetCLongWidthAttr << " is " << cLongWidth
                              << " but target triple '" << tripleAttr.getValue()
                              << "' requires " << expectedCLong;

  return mlir::success();
}

llvm::StringRef stripCtypesModule(llvm::StringRef contract) {
  if (contract.consume_front("ctypes."))
    return contract;
  if (contract.consume_front("_ctypes."))
    return contract;
  return contract;
}

std::optional<NativeABIType>
ctypesScalarLayout(llvm::StringRef contract,
                   const std::optional<TargetPlatformFacts> &facts) {
  llvm::StringRef name = stripCtypesModule(contract);
  if (std::optional<NativeABIType> fixed =
          llvm::StringSwitch<std::optional<NativeABIType>>(name)
              .Cases({"c_bool", "c_ubyte", "c_uint8"},
                     NativeABIType{1, 1, NativeABIKind::UnsignedInteger})
              .Cases({"c_byte", "c_int8", "c_char"},
                     NativeABIType{1, 1, NativeABIKind::SignedInteger})
              .Cases({"c_ushort", "c_uint16"},
                     NativeABIType{2, 2, NativeABIKind::UnsignedInteger})
              .Cases({"c_short", "c_int16"},
                     NativeABIType{2, 2, NativeABIKind::SignedInteger})
              .Cases({"c_uint", "c_uint32"},
                     NativeABIType{4, 4, NativeABIKind::UnsignedInteger})
              .Cases({"c_int", "c_int32"},
                     NativeABIType{4, 4, NativeABIKind::SignedInteger})
              .Case("c_float", NativeABIType{4, 4, NativeABIKind::Floating})
              .Cases({"c_ulonglong", "c_uint64"},
                     NativeABIType{8, 8, NativeABIKind::UnsignedInteger})
              .Cases({"c_longlong", "c_int64"},
                     NativeABIType{8, 8, NativeABIKind::SignedInteger})
              .Case("c_double", NativeABIType{8, 8, NativeABIKind::Floating})
              .Default(std::nullopt))
    return fixed;

  if (!facts)
    return std::nullopt;
  return llvm::StringSwitch<std::optional<NativeABIType>>(name)
      .Case("c_ulong", NativeABIType{facts->cLongBytes(), facts->cLongBytes(),
                                     NativeABIKind::UnsignedInteger})
      .Cases({"c_long", "HRESULT"},
             NativeABIType{facts->cLongBytes(), facts->cLongBytes(),
                           NativeABIKind::SignedInteger})
      .Case("c_size_t",
            NativeABIType{facts->pointerBytes(), facts->pointerBytes(),
                          NativeABIKind::UnsignedInteger})
      .Case("c_ssize_t",
            NativeABIType{facts->pointerBytes(), facts->pointerBytes(),
                          NativeABIKind::SignedInteger})
      .Cases({"c_void_p", "_Pointer"},
             NativeABIType{facts->pointerBytes(), facts->pointerBytes(),
                           NativeABIKind::Pointer})
      .Default(std::nullopt);
}

bool isIntegerABI(const NativeABIType &type) {
  return type.kind == NativeABIKind::SignedInteger ||
         type.kind == NativeABIKind::UnsignedInteger;
}

bool isFloatingABI(const NativeABIType &type) {
  return type.kind == NativeABIKind::Floating;
}

bool isPointerABI(const NativeABIType &type) {
  return type.kind == NativeABIKind::Pointer;
}

} // namespace py::native
