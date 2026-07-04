#include "Runtime/Ctypes/Internal.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/ErrorHandling.h"

namespace py::runtime_lowering::ctypes {

std::string targetFactsLabel(const std::optional<TargetPlatformFacts> &facts) {
  if (!facts)
    return "missing target facts";
  return "target '" + facts->triple + "'";
}

bool isFixedOrTargetDependentCtypesScalar(llvm::StringRef contract) {
  llvm::StringRef name = stripCtypesModule(contract);
  return llvm::StringSwitch<bool>(name)
      .Cases("c_bool", "c_byte", "c_ubyte", "c_short", "c_ushort", "c_int",
             "c_uint", true)
      .Cases("c_long", "c_ulong", "c_longlong", "c_ulonglong", true)
      .Cases("c_int8", "c_uint8", "c_int16", "c_uint16", "c_int32", "c_uint32",
             "c_int64", "c_uint64", true)
      .Cases("c_ssize_t", "c_size_t", "c_void_p", true)
      .Default(false);
}

bool isCtypesIntegralLike(llvm::StringRef contract) {
  llvm::StringRef name = stripCtypesModule(contract);
  return llvm::StringSwitch<bool>(name)
      .Cases("c_bool", "c_byte", "c_ubyte", "c_short", "c_ushort", "c_int",
             "c_uint", true)
      .Cases("c_long", "c_ulong", "c_longlong", "c_ulonglong", true)
      .Cases("c_int8", "c_uint8", "c_int16", "c_uint16", "c_int32", "c_uint32",
             "c_int64", "c_uint64", true)
      .Cases("c_ssize_t", "c_size_t", "c_void_p", true)
      .Default(false);
}

bool isCtypesVoidPointer(llvm::StringRef contract) {
  return stripCtypesModule(contract) == "c_void_p";
}

bool isCtypesPointerContract(llvm::StringRef contract) {
  return stripCtypesModule(contract) == "_Pointer";
}

bool isNoneContractName(llvm::StringRef contract) {
  return contract == "types.NoneType";
}

bool isNoneBundle(const RuntimeBundle &bundle) {
  return isNoneContractName(bundle.contractName());
}

bool isStaticSequenceBundle(const RuntimeBundle &bundle) {
  return bundle.contractName() == "builtins.list" ||
         bundle.contractName() == "builtins.tuple" ||
         !bundle.sequenceElementBundles.empty();
}

std::string ctypesContractFromBundle(const RuntimeBundle &bundle) {
  if (bundle.ctypes)
    return bundle.ctypes->ctypeName;
  if (bundle.kind == RuntimeBundle::Kind::TypeObject)
    return bundle.instanceContractName();
  return bundle.contractName();
}

mlir::Type ctypesTypeFromBundle(const RuntimeBundle &bundle) {
  if (bundle.ctypes && bundle.ctypes->ctype)
    return bundle.ctypes->ctype;
  if (bundle.kind == RuntimeBundle::Kind::TypeObject)
    return bundle.instanceContract;
  return bundle.contract;
}

std::optional<std::string> ctypesTypeObjectName(const RuntimeBundle &bundle) {
  if (bundle.kind != RuntimeBundle::Kind::TypeObject)
    return std::nullopt;
  std::string contract = bundle.instanceContractName();
  llvm::StringRef name(contract);
  if (name.starts_with("ctypes.") || name.starts_with("_ctypes."))
    return contract;
  return std::nullopt;
}

std::optional<llvm::StringRef>
ctypesFromAddressTarget(llvm::StringRef binding) {
  if (!binding.consume_front("ctypes.from_address:"))
    return std::nullopt;
  if (binding.empty())
    return std::nullopt;
  return binding;
}

std::optional<llvm::StringRef> ctypesFromBufferTarget(llvm::StringRef binding) {
  if (!binding.consume_front("ctypes.from_buffer:"))
    return std::nullopt;
  if (binding.empty())
    return std::nullopt;
  return binding;
}

std::optional<llvm::StringRef>
ctypesFromBufferCopyTarget(llvm::StringRef binding) {
  if (!binding.consume_front("ctypes.from_buffer_copy:"))
    return std::nullopt;
  if (binding.empty())
    return std::nullopt;
  return binding;
}

void keepAliveSource(RuntimeCtypesEvidence &evidence,
                     const RuntimeBundle &source) {
  if (source.objectValue.contract)
    evidence.keepAlive.push_back(source.objectValue);
  if (source.ctypes)
    evidence.keepAlive.append(source.ctypes->keepAlive.begin(),
                              source.ctypes->keepAlive.end());
}

void keepAliveBufferSource(RuntimeBufferEvidence &evidence,
                           const RuntimeBundle &source) {
  if (source.objectValue.contract)
    evidence.keepAlive.push_back(source.objectValue);
  if (source.buffer)
    evidence.keepAlive.append(source.buffer->keepAlive.begin(),
                              source.buffer->keepAlive.end());
  if (source.ctypes)
    evidence.keepAlive.append(source.ctypes->keepAlive.begin(),
                              source.ctypes->keepAlive.end());
}

mlir::Value cdataStorageAddress(const RuntimeCtypesEvidence &evidence) {
  return evidence.storageAddressValue;
}

mlir::Value cdataStorageAddressValid(const RuntimeCtypesEvidence &evidence) {
  return evidence.storageAddressValid;
}

mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                       bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1)
      .getResult();
}

mlir::Value constantI64(mlir::OpBuilder &builder, mlir::Location loc,
                        std::int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64).getResult();
}

mlir::Value constantIndex(mlir::OpBuilder &builder, mlir::Location loc,
                          std::int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value).getResult();
}

std::string ctypesLibraryABI(llvm::StringRef contract) {
  return llvm::StringSwitch<std::string>(contract)
      .Case("ctypes.CDLL", "cdecl")
      .Case("ctypes.WinDLL", "stdcall")
      .Default("");
}

bool isKnownTrue(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>();
  return constant && constant.value() != 0;
}

bool isIntegerScalarLayout(const CtypesLayout &layout) {
  return layout.kind == CtypesLayout::ABIKind::SignedInteger ||
         layout.kind == CtypesLayout::ABIKind::UnsignedInteger;
}

bool isFloatingScalarLayout(const CtypesLayout &layout) {
  return layout.kind == CtypesLayout::ABIKind::Floating;
}

bool isPointerScalarLayout(const CtypesLayout &layout) {
  return layout.kind == CtypesLayout::ABIKind::Pointer;
}

llvm::StringRef
ctypesProvenanceName(RuntimeCtypesEvidence::Provenance provenance) {
  switch (provenance) {
  case RuntimeCtypesEvidence::Provenance::None:
    return "none";
  case RuntimeCtypesEvidence::Provenance::NativeCell:
    return "native_cell";
  case RuntimeCtypesEvidence::Provenance::CallRegionBorrow:
    return "call_region_borrow";
  case RuntimeCtypesEvidence::Provenance::ExternalAddress:
    return "external_address";
  case RuntimeCtypesEvidence::Provenance::Cast:
    return "cast";
  case RuntimeCtypesEvidence::Provenance::BufferView:
    return "buffer_view";
  case RuntimeCtypesEvidence::Provenance::CallbackThunk:
    return "callback_thunk";
  }
  llvm_unreachable("unhandled ctypes provenance");
}

llvm::StringRef ctypesLifetimeName(RuntimeCtypesEvidence::Lifetime lifetime) {
  switch (lifetime) {
  case RuntimeCtypesEvidence::Lifetime::Unknown:
    return "unknown";
  case RuntimeCtypesEvidence::Lifetime::CallRegion:
    return "call_region";
  case RuntimeCtypesEvidence::Lifetime::Owner:
    return "owner";
  case RuntimeCtypesEvidence::Lifetime::External:
    return "external";
  case RuntimeCtypesEvidence::Lifetime::Static:
    return "static";
  }
  llvm_unreachable("unhandled ctypes lifetime");
}

std::optional<std::int64_t> knownI64Constant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constant)
    return std::nullopt;
  return constant.value();
}

bool fitsStaticIntegerLayout(std::int64_t value, const CtypesLayout &layout) {
  if (!isIntegerScalarLayout(layout))
    return false;
  unsigned bits = static_cast<unsigned>(layout.size * 8);
  if (layout.kind == CtypesLayout::ABIKind::SignedInteger) {
    if (bits >= 64)
      return true;
    std::int64_t min = -(std::int64_t{1} << (bits - 1));
    std::int64_t max = (std::int64_t{1} << (bits - 1)) - 1;
    return min <= value && value <= max;
  }
  if (value < 0)
    return false;
  if (bits >= 64)
    return true;
  std::uint64_t max = (std::uint64_t{1} << bits) - 1;
  return static_cast<std::uint64_t>(value) <= max;
}

mlir::IntegerType nativeIntegerType(mlir::Builder &builder,
                                    const CtypesLayout &layout) {
  return builder.getIntegerType(static_cast<unsigned>(layout.size * 8));
}

mlir::IntegerType
nativePointerIntegerType(mlir::Builder &builder,
                         const std::optional<TargetPlatformFacts> &facts) {
  unsigned width = facts ? static_cast<unsigned>(facts->pointerWidth) : 64;
  return builder.getIntegerType(width);
}

mlir::LLVM::LLVMPointerType nativePointerType(mlir::MLIRContext *context) {
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  return mlir::LLVM::LLVMPointerType::get(context);
}

mlir::Value coerceNativeInteger(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value,
                                mlir::IntegerType targetType) {
  auto sourceType = mlir::cast<mlir::IntegerType>(value.getType());
  if (sourceType == targetType)
    return value;
  if (sourceType.getWidth() > targetType.getWidth())
    return builder.create<mlir::arith::TruncIOp>(loc, targetType, value)
        .getResult();
  return builder.create<mlir::arith::ExtSIOp>(loc, targetType, value)
      .getResult();
}

mlir::Value
loadNativeIntegerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value address, mlir::IntegerType nativeType,
                             const std::optional<TargetPlatformFacts> &facts);

std::optional<mlir::Value> extractNativeIntegerArgument(
    mlir::Operation *op, mlir::OpBuilder &builder, const RuntimeBundle &source,
    llvm::StringRef expectedContract, const CtypesLayout &layout,
    const std::optional<TargetPlatformFacts> &facts) {
  if (!isIntegerScalarLayout(layout) && !isPointerScalarLayout(layout))
    return std::nullopt;
  mlir::IntegerType nativeType = nativeIntegerType(builder, layout);
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      stripCtypesModule(source.ctypes->ctypeName) ==
          stripCtypesModule(expectedContract) &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    return loadNativeIntegerFromAddress(builder, op->getLoc(),
                                        source.ctypes->storageAddressValue,
                                        nativeType, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      stripCtypesModule(source.ctypes->ctypeName) ==
          stripCtypesModule(expectedContract) &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid)) {
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->scalarValue, nativeType);
  }
  if (layout.kind == CtypesLayout::ABIKind::SignedInteger && layout.size == 8 &&
      source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid)) {
    return source.primitiveI64->value;
  }
  if (isPointerScalarLayout(layout) && source.primitiveI64 &&
      source.primitiveI64->value && source.primitiveI64->valid &&
      isKnownTrue(source.primitiveI64->valid)) {
    return coerceNativeInteger(builder, op->getLoc(),
                               source.primitiveI64->value, nativeType);
  }
  if (source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid)) {
    std::optional<std::int64_t> constant =
        knownI64Constant(source.primitiveI64->value);
    if (constant && fitsStaticIntegerLayout(*constant, layout))
      return coerceNativeInteger(builder, op->getLoc(),
                                 source.primitiveI64->value, nativeType);
  }
  return std::nullopt;
}

mlir::Value
integerToNativePointer(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value value,
                       const std::optional<TargetPlatformFacts> &facts) {
  mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
  mlir::Value raw = coerceNativeInteger(builder, loc, value, pointerInteger);
  return builder.create<mlir::LLVM::IntToPtrOp>(
      loc, nativePointerType(builder.getContext()), raw);
}

mlir::Value nativePointerToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value pointer) {
  return builder.create<mlir::LLVM::PtrToIntOp>(loc, builder.getI64Type(),
                                                pointer);
}

mlir::Value addressWithOffset(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value address, std::int64_t offset,
                              const std::optional<TargetPlatformFacts> &facts) {
  mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
  mlir::Value base = coerceNativeInteger(builder, loc, address, pointerInteger);
  if (offset == 0)
    return base;
  mlir::Value delta = builder.create<mlir::arith::ConstantIntOp>(
      loc, offset, pointerInteger.getWidth());
  return builder.create<mlir::arith::AddIOp>(loc, base, delta).getResult();
}

mlir::Value
nativePointerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value address,
                         const std::optional<TargetPlatformFacts> &facts) {
  return integerToNativePointer(builder, loc, address, facts);
}

mlir::Value
addressOfNativeCellAlloca(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value nativeValue,
                          const std::optional<TargetPlatformFacts> &facts) {
  auto nativeType = mlir::cast<mlir::IntegerType>(nativeValue.getType());
  auto bufferType = mlir::MemRefType::get({1}, nativeType);
  mlir::Value buffer = builder.create<mlir::memref::AllocaOp>(loc, bufferType);
  mlir::Value zero = constantIndex(builder, loc, 0);
  builder.create<mlir::memref::StoreOp>(loc, nativeValue, buffer,
                                        mlir::ValueRange{zero});
  mlir::Value pointerIndex =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
  return builder
      .create<mlir::arith::IndexCastOp>(
          loc, nativePointerIntegerType(builder, facts), pointerIndex)
      .getResult();
}

mlir::Value addressOfZeroedNativeBytesAlloca(
    mlir::OpBuilder &builder, mlir::Location loc, std::uint64_t size,
    const std::optional<TargetPlatformFacts> &facts) {
  std::uint64_t allocationSize = std::max<std::uint64_t>(size, 1);
  auto byteType = builder.getIntegerType(8);
  auto bufferType = mlir::MemRefType::get(
      {static_cast<std::int64_t>(allocationSize)}, byteType);
  mlir::Value buffer = builder.create<mlir::memref::AllocaOp>(loc, bufferType);
  mlir::Value lower = constantIndex(builder, loc, 0);
  mlir::Value upper =
      constantIndex(builder, loc, static_cast<std::int64_t>(allocationSize));
  mlir::Value step = constantIndex(builder, loc, 1);
  mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
  auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    builder.create<mlir::memref::StoreOp>(
        loc, zero, buffer, mlir::ValueRange{loop.getInductionVar()});
  }
  mlir::Value pointerIndex =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
  return builder
      .create<mlir::arith::IndexCastOp>(
          loc, nativePointerIntegerType(builder, facts), pointerIndex)
      .getResult();
}

mlir::Value
loadNativeIntegerFromAddress(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value address, mlir::IntegerType nativeType,
                             const std::optional<TargetPlatformFacts> &facts) {
  mlir::Value pointer = nativePointerFromAddress(builder, loc, address, facts);
  return builder.create<mlir::LLVM::LoadOp>(loc, nativeType, pointer)
      .getResult();
}

void storeNativeIntegerToAddress(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value address,
    mlir::Value value, mlir::IntegerType nativeType,
    const std::optional<TargetPlatformFacts> &facts) {
  mlir::Value pointer = nativePointerFromAddress(builder, loc, address, facts);
  mlir::Value nativeValue =
      coerceNativeInteger(builder, loc, value, nativeType);
  builder.create<mlir::LLVM::StoreOp>(loc, nativeValue, pointer);
}

mlir::Value widenNativeInteger(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value value, const CtypesLayout &layout) {
  auto sourceType = mlir::cast<mlir::IntegerType>(value.getType());
  mlir::IntegerType i64 = builder.getI64Type();
  if (sourceType == i64)
    return value;
  if (sourceType.getWidth() > i64.getWidth())
    return builder.create<mlir::arith::TruncIOp>(loc, i64, value).getResult();
  if (layout.kind == CtypesLayout::ABIKind::UnsignedInteger ||
      layout.kind == CtypesLayout::ABIKind::Pointer)
    return builder.create<mlir::arith::ExtUIOp>(loc, i64, value).getResult();
  return builder.create<mlir::arith::ExtSIOp>(loc, i64, value).getResult();
}

std::string describeNativeArgumentSource(const RuntimeBundle &source);

void copyNativeBytes(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value destinationAddress, mlir::Value sourceAddress,
                     std::uint64_t byteCount,
                     const std::optional<TargetPlatformFacts> &facts) {
  mlir::Type byteType = builder.getIntegerType(8);
  for (std::uint64_t offset = 0; offset < byteCount; ++offset) {
    mlir::Value sourceByteAddress = addressWithOffset(
        builder, loc, sourceAddress, static_cast<std::int64_t>(offset), facts);
    mlir::Value destinationByteAddress =
        addressWithOffset(builder, loc, destinationAddress,
                          static_cast<std::int64_t>(offset), facts);
    mlir::Value sourcePointer =
        nativePointerFromAddress(builder, loc, sourceByteAddress, facts);
    mlir::Value destinationPointer =
        nativePointerFromAddress(builder, loc, destinationByteAddress, facts);
    mlir::Value byte =
        builder.create<mlir::LLVM::LoadOp>(loc, byteType, sourcePointer)
            .getResult();
    builder.create<mlir::LLVM::StoreOp>(loc, byte, destinationPointer);
  }
}

mlir::LogicalResult storeCtypesValueToAddress(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Value destinationAddress, mlir::Type expectedType,
    llvm::StringRef expectedContract, const CtypesLayout &layout,
    const RuntimeBundle &source,
    const std::optional<TargetPlatformFacts> &facts) {
  if (isIntegerScalarLayout(layout) || isPointerScalarLayout(layout)) {
    std::optional<mlir::Value> value = extractNativeIntegerArgument(
        op, builder, source, expectedContract, layout, facts);
    if (!value)
      return op->emitError() << "ctypes value for " << expectedContract
                             << " has no compatible scalar evidence ("
                             << describeNativeArgumentSource(source) << ")";
    storeNativeIntegerToAddress(builder, op->getLoc(), destinationAddress,
                                *value, nativeIntegerType(builder, layout),
                                facts);
    return mlir::success();
  }

  if (layout.kind != CtypesLayout::ABIKind::Aggregate)
    return op->emitError() << "ctypes value for " << expectedContract
                           << " has unsupported ABI layout";
  if (!source.ctypes ||
      source.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return op->emitError() << "ctypes aggregate value for " << expectedContract
                           << " requires _CData cell evidence";
  if (expectedType && source.ctypes->ctype &&
      expectedType != source.ctypes->ctype)
    return op->emitError() << "ctypes aggregate assignment expects "
                           << expectedType << " but got "
                           << source.ctypes->ctype;

  mlir::Value sourceAddress = cdataStorageAddress(*source.ctypes);
  mlir::Value sourceValid = cdataStorageAddressValid(*source.ctypes);
  if (!sourceAddress || !sourceValid || !isKnownTrue(sourceValid))
    return op->emitError() << "ctypes aggregate value for " << expectedContract
                           << " has no materialized storage address";

  std::optional<CtypesLayout> sourceLayout = ctypesStaticLayoutForType(
      module,
      source.ctypes->ctype
          ? source.ctypes->ctype
          : ctypesContractType(op->getContext(), source.ctypes->ctypeName),
      facts);
  if (!sourceLayout)
    sourceLayout = ctypesStaticLayout(module, source.ctypes->ctypeName, facts);
  if (!sourceLayout || sourceLayout->size != layout.size)
    return op->emitError() << "ctypes aggregate value for " << expectedContract
                           << " has incompatible storage size";

  copyNativeBytes(builder, op->getLoc(), destinationAddress, sourceAddress,
                  layout.size, facts);
  return mlir::success();
}

RuntimeBufferEvidence
makeCtypesBufferEvidence(mlir::OpBuilder &builder, mlir::Location loc,
                         const CtypesLayout &layout, mlir::Value address,
                         mlir::Value valid, const RuntimeBundle &owner,
                         bool writable) {
  RuntimeBufferEvidence buffer;
  buffer.addressValue = address;
  buffer.addressValid = valid;
  buffer.byteLength =
      constantI64(builder, loc, static_cast<std::int64_t>(layout.size));
  buffer.byteLengthValid = constantI1(builder, loc, true);
  buffer.readable = true;
  buffer.writable = writable;
  buffer.cContiguous = true;
  keepAliveBufferSource(buffer, owner);
  return buffer;
}

void attachCtypesBufferEvidence(mlir::OpBuilder &builder, mlir::Location loc,
                                RuntimeBundle &bundle,
                                RuntimeCtypesEvidence &ctypes,
                                const CtypesLayout &layout, bool writable) {
  mlir::Value storageAddress = cdataStorageAddress(ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return;
  bundle.buffer = makeCtypesBufferEvidence(builder, loc, layout, storageAddress,
                                           storageValid, bundle, writable);
}

mlir::FailureOr<RuntimeBundle> materializeCtypesAddressView(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Type ctype, llvm::StringRef ctypeName, const CtypesLayout &layout,
    mlir::Value storageAddress, mlir::Value storageValid,
    const RuntimeBundle &owner) {
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op->emitError() << "ctypes view for " << ctypeName
                           << " requires a statically valid storage address";

  RuntimeBundle result = RuntimeBundle::object(ctype, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  evidence.provenance = RuntimeCtypesEvidence::Provenance::BufferView;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
  evidence.ctypeName = ctypeName.str();
  evidence.ctype = ctype;
  evidence.storageAddressValue = storageAddress;
  evidence.storageAddressValid = storageValid;
  evidence.addressValue = storageAddress;
  evidence.addressValid = storageValid;
  evidence.materializedObject = true;
  keepAliveSource(evidence, owner);

  if (isIntegerScalarLayout(layout) || isPointerScalarLayout(layout)) {
    mlir::Value nativeValue = loadNativeIntegerFromAddress(
        builder, op->getLoc(), storageAddress,
        nativeIntegerType(builder, layout), targetPlatformFacts(module));
    evidence.scalarValue =
        widenNativeInteger(builder, op->getLoc(), nativeValue, layout);
    evidence.scalarValid = constantI1(builder, op->getLoc(), true);
  }

  attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence, layout,
                             /*writable=*/true);
  result.ctypes = std::move(evidence);
  return result;
}

mlir::FailureOr<RuntimeBundle> materializeCtypesPythonReadResult(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Type ctype, llvm::StringRef ctypeName, const CtypesLayout &layout,
    mlir::Value storageAddress, mlir::Value storageValid,
    const RuntimeBundle &owner) {
  if (isIntegerScalarLayout(layout) || isPointerScalarLayout(layout)) {
    if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
      return op->emitError() << "ctypes scalar read for " << ctypeName
                             << " requires a statically valid storage address";
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    mlir::Value nativeValue =
        loadNativeIntegerFromAddress(builder, op->getLoc(), storageAddress,
                                     nativeIntegerType(builder, layout), facts);
    RuntimeBundle result = RuntimeBundle::object(
        runtimeContractType(op->getContext(), "builtins.int"), {});
    result.primitiveI64 = RuntimePrimitiveI64Evidence{
        widenNativeInteger(builder, op->getLoc(), nativeValue, layout),
        constantI1(builder, op->getLoc(), true)};
    return result;
  }

  return materializeCtypesAddressView(op, builder, module, ctype, ctypeName,
                                      layout, storageAddress, storageValid,
                                      owner);
}

std::string describeNativeArgumentSource(const RuntimeBundle &source);

} // namespace py::runtime_lowering::ctypes
