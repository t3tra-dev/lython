#include "Runtime/Core/Lowerer.h"
#include "Runtime/Ctypes/Internal.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::lowering {

// The NATIVE cell: one process-lifetime i64 holding a module global's raw
// machine word. A read is a plain llvm.load and a write a plain llvm.store,
// so touching one never allocates -- the async-signal-safe channel a signal
// handler exchanges state through. Two populations live here: a ctypes
// address (lowerAddressGlobal*) and a runtime-internal module's `int`
// (lowerNativeIntGlobal*, which boxes on read and unboxes on write).
mlir::LLVM::GlobalOp
RuntimeBundleLowerer::nativeGlobalCell(mlir::Operation *op,
                                       llvm::StringRef name) {
  std::string symbol = ("__ly_module_global_" + name).str();
  if (auto existing = module.lookupSymbol<mlir::LLVM::GlobalOp>(symbol))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  mlir::Type i64 = builder.getI64Type();
  auto global = mlir::LLVM::GlobalOp::create(
      builder, op->getLoc(), i64, /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, symbol,
      builder.getI64IntegerAttr(0), /*alignment=*/8);
  return global;
}

mlir::Value RuntimeBundleLowerer::loadNativeGlobalWord(mlir::Operation *op,
                                                       llvm::StringRef name) {
  mlir::LLVM::GlobalOp cell = RuntimeBundleLowerer::nativeGlobalCell(op, name);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  return mlir::LLVM::LoadOp::create(builder, op->getLoc(),
                                    builder.getI64Type(), address);
}

void RuntimeBundleLowerer::storeNativeGlobalWord(mlir::Operation *op,
                                                 llvm::StringRef name,
                                                 mlir::Value word) {
  mlir::LLVM::GlobalOp cell = RuntimeBundleLowerer::nativeGlobalCell(op, name);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  mlir::LLVM::StoreOp::create(builder, op->getLoc(), word, address);
}

namespace {

// Element type of a rank-1 memref, or null when the value is not one.
mlir::Type rankOneElementType(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return {};
  return memref.getElementType();
}

// MLIR's rank-1 memref descriptor, which only the store side's cast names.
mlir::Type memrefDescriptorType(mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  auto ptr = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type i64 = builder.getI64Type();
  auto arrayOne = mlir::LLVM::LLVMArrayType::get(i64, 1);
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptr, ptr, i64, arrayOne, arrayOne});
}

} // namespace

// Cells backing one module-global OBJECT value: one i64 llvm.global per
// stored word (`_init` bound flag, then `_p<i>`/`_s<i>` pointer and size
// words per rank-1 memref physical value, or `_v<i>` for scalar physical
// values such as a union tag). Objects park here with one retained
// reference; rebinding releases the previous holder.
mlir::LLVM::GlobalOp
RuntimeBundleLowerer::moduleObjectGlobalCell(mlir::Operation *op,
                                             llvm::StringRef name,
                                             llvm::StringRef suffix,
                                             mlir::Type cellType) {
  std::string symbol =
      ("__ly_module_global_obj_" + name + "_" + suffix).str();
  if (auto existing = module.lookupSymbol<mlir::LLVM::GlobalOp>(symbol))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(cellType))
    return mlir::LLVM::GlobalOp::create(
        builder, module.getLoc(), cellType, /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, symbol, builder.getI64IntegerAttr(0),
        /*alignment=*/8);
  // A null pointer is not an integer attribute, so the initial value has to
  // come from a body rather than from `getI64IntegerAttr(0)`.
  auto global = mlir::LLVM::GlobalOp::create(
      builder, module.getLoc(), cellType, /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, symbol, mlir::Attribute(),
      /*alignment=*/8);
  mlir::OpBuilder::InsertionGuard initGuard(builder);
  builder.setInsertionPointToEnd(
      builder.createBlock(&global.getInitializerRegion()));
  mlir::LLVM::ReturnOp::create(
      builder, module.getLoc(),
      mlir::ValueRange{
          mlir::LLVM::ZeroOp::create(builder, module.getLoc(), cellType)});
  return global;
}

mlir::Value RuntimeBundleLowerer::loadObjectGlobalWord(
    mlir::Operation *op, llvm::StringRef name, llvm::StringRef suffix) {
  mlir::LLVM::GlobalOp cell = RuntimeBundleLowerer::moduleObjectGlobalCell(
      op, name, suffix, builder.getI64Type());
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  return mlir::LLVM::LoadOp::create(builder, op->getLoc(),
                                    builder.getI64Type(), address);
}

void RuntimeBundleLowerer::storeObjectGlobalWord(mlir::Operation *op,
                                                 llvm::StringRef name,
                                                 llvm::StringRef suffix,
                                                 mlir::Value word) {
  mlir::LLVM::GlobalOp cell = RuntimeBundleLowerer::moduleObjectGlobalCell(
      op, name, suffix, builder.getI64Type());
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  mlir::LLVM::StoreOp::create(builder, op->getLoc(), word, address);
}

mlir::Value RuntimeBundleLowerer::loadObjectGlobalPointer(
    mlir::Operation *op, llvm::StringRef name, llvm::StringRef suffix) {
  mlir::Type ptr = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::LLVM::GlobalOp cell =
      RuntimeBundleLowerer::moduleObjectGlobalCell(op, name, suffix, ptr);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  return mlir::LLVM::LoadOp::create(builder, op->getLoc(), ptr, address);
}

void RuntimeBundleLowerer::storeObjectGlobalPointer(mlir::Operation *op,
                                                    llvm::StringRef name,
                                                    llvm::StringRef suffix,
                                                    mlir::Value pointer) {
  mlir::Type ptr = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::LLVM::GlobalOp cell =
      RuntimeBundleLowerer::moduleObjectGlobalCell(op, name, suffix, ptr);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op->getLoc(), cell);
  mlir::LLVM::StoreOp::create(builder, op->getLoc(), pointer, address);
}

// Reassemble the physical value group of a module-global object from its
// cells (the inverse of the store side in lowerGlobalSet).
mlir::LogicalResult RuntimeBundleLowerer::loadObjectGlobalValues(
    mlir::Operation *op, llvm::StringRef name,
    llvm::ArrayRef<mlir::Type> valueTypes,
    llvm::SmallVectorImpl<mlir::Value> &values) {
  mlir::Location loc = op->getLoc();
  for (auto [index, valueType] : llvm::enumerate(valueTypes)) {
    std::string slot = std::to_string(index);
    if (rankOneElementType(valueType)) {
      // The cell holds the payload's aligned pointer, so the view is built
      // from a pointer and never from an integer. This used to call
      // `__ly_global_view_*`, which exists so a MANIFEST body can get a
      // descriptor through a call instead of a cast (see
      // Passes/Lowering.cpp); this side is the compiler's own and has no such
      // constraint, so it assembles the descriptor where it stands.
      mlir::Value pointer =
          RuntimeBundleLowerer::loadObjectGlobalPointer(op, name, "p" + slot);
      mlir::Value size =
          RuntimeBundleLowerer::loadObjectGlobalWord(op, name, "s" + slot);
      values.push_back(RuntimeBundleLowerer::memrefFromBoxPointer(
          builder, loc, pointer, size,
          mlir::cast<mlir::MemRefType>(valueType)));
      continue;
    }
    mlir::Value word =
        RuntimeBundleLowerer::loadObjectGlobalWord(op, name, "v" + slot);
    if (valueType.isInteger(64)) {
      values.push_back(word);
    } else if (valueType.isInteger(1)) {
      values.push_back(
          mlir::arith::TruncIOp::create(builder, loc, builder.getI1Type(),
                                        word)
              .getResult());
    } else if (valueType.isF64()) {
      values.push_back(mlir::arith::BitcastOp::create(
                           builder, loc, builder.getF64Type(), word)
                           .getResult());
    } else {
      return op->emitError() << "module global '" << name << "' value "
                             << index << " has unsupported scalar type "
                             << valueType;
    }
  }
  return mlir::success();
}

// An address global's cell is the machine word itself: no header, no
// refcount, so a read is one `llvm.load` and a signal handler may take it.
// The bundle it hands back is the same ctypes CELL evidence
// `ctypes.cast(x, c_void_p)` produces, which is what makes a global
// interchangeable with a cast result everywhere downstream.
bool RuntimeBundleLowerer::isAddressGlobalType(mlir::Type type) {
  return ctypes::isCtypesVoidPointer(runtimeContractName(type));
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerAddressGlobalGet(py::GlobalGetOp op) {
  mlir::Type type = op.getResult().getType();
  builder.setInsertionPoint(op);
  mlir::Value word =
      RuntimeBundleLowerer::loadNativeGlobalWord(op, op.getName());
  mlir::Value valid =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);

  RuntimeBundle result = RuntimeBundle::object(type, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  // ⛔ Why NOT Provenance::Cast, which is what the c_void_p cast records: a
  // cast keeps its SOURCE alive, and this word outlives every source. Static
  // says the address is not owned by anything in this frame, which is the
  // only claim a process-lifetime cell can make.
  evidence.provenance = RuntimeCtypesEvidence::Provenance::ExternalAddress;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = runtimeContractName(type);
  evidence.ctype = type;
  evidence.addressValue = word;
  evidence.addressValid = valid;
  evidence.scalarValue = word;
  evidence.scalarValid = valid;
  result.ctypes = std::move(evidence);
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerAddressGlobalSet(py::GlobalSetOp op,
                                            const RuntimeBundle &value) {
  builder.setInsertionPoint(op);
  std::optional<ctypes::TargetPlatformFacts> facts =
      ctypes::targetPlatformFacts(module);
  std::optional<mlir::Value> word =
      ctypes::extractPointerAddressInteger(op, builder, value, facts);
  if (!word)
    return op.emitError()
           << "module global '" << op.getName()
           << "' is an address cell, so its value must carry a concrete "
              "pointer: None, a c_void_p, or a cast of one";
  mlir::Value stored = *word;
  if (!stored.getType().isInteger(64))
    stored = mlir::arith::ExtUIOp::create(builder, op.getLoc(),
                                          builder.getI64Type(), stored)
                 .getResult();
  RuntimeBundleLowerer::storeNativeGlobalWord(op, op.getName(), stored);
  erase.push_back(op);
  return mlir::success();
}

// ⭐ An int module global is a Python integer and is BOXED; a machine address
// is spelled with its ctypes type and keeps the word. Nothing in the surface
// distinguished the two before, and the single unboxed i64 cell that served
// both is why `x: int = 1` grown past 2**63 raised "int too large to convert
// to a native 64-bit integer" where CPython printed the value -- while the
// same loop over a LOCAL printed it correctly here too.
//
// The boxing is keyed on a `ly.global.boxed` mark the emitter puts on the two
// ops that reach the MODULE-GLOBAL population.
//
// ⛔ Why NOT box every py.global.get/set instead, which is one line here:
// default cells and class-attribute slots ride the SAME two ops, and boxing
// them made `golden/cases/default_once` print 0 for a default that must
// evaluate once. Marking per population is what tells the three apart.
//
// ⛔ Why NOT box the runtime's own modules too: `stackguard_support.py` keeps
// libc function pointers in module globals and reads them from a signal
// handler, and an object global's read retains. `EmitOptions::runtimeInternal`
// leaves those on the word, so the exemption is one flag set by one tool
// rather than a name test -- the runtime's modules are emitted as `__main__`
// as well, so the emitter cannot tell them apart by module name.
mlir::LogicalResult RuntimeBundleLowerer::lowerGlobalGet(py::GlobalGetOp op) {
  if (RuntimeBundleLowerer::isAddressGlobalType(op.getResult().getType()))
    return RuntimeBundleLowerer::lowerAddressGlobalGet(op);
  if (runtimeContractName(op.getResult().getType()) != "builtins.int" ||
      op->hasAttr("ly.global.boxed"))
    return RuntimeBundleLowerer::lowerObjectGlobalGet(op);
  return RuntimeBundleLowerer::lowerNativeIntGlobalGet(op);
}

// A runtime-internal module's `int` global: the word IS the value, boxed on
// read into the primitive lane so ordinary int ops still work on it.
mlir::LogicalResult
RuntimeBundleLowerer::lowerNativeIntGlobalGet(py::GlobalGetOp op) {
  builder.setInsertionPoint(op);
  mlir::Value raw =
      RuntimeBundleLowerer::loadNativeGlobalWord(op, op.getName());
  mlir::Value valid =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, runtimeContractType(context, "builtins.int"), raw, valid,
          result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

// The object path: cells hold the value group's raw words plus one
// retained reference. Reads reassemble the group, retain it for the reader
// (the refcount insertion releases it after use), and an unbound read
// raises RuntimeError through the shared manifest raise helper.
mlir::LogicalResult
RuntimeBundleLowerer::lowerObjectGlobalGet(py::GlobalGetOp op) {
  mlir::Type type = op.getResult().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, type,
                                                 "module global object ABI");
  if (mlir::failed(valueTypes))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value bound = RuntimeBundleLowerer::loadObjectGlobalWord(
      op, op.getName(), "init");
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value unbound = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, bound, zero);
  mlir::func::FuncOp raise =
      module.lookupSymbol<mlir::func::FuncOp>("__ly_raise_static_message");
  if (!raise)
    return op.emitError() << "runtime manifest raise helper is missing";
  auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                       unbound, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&guard.getThenRegion().front());
    std::string message = ("module global '" + op.getName() +
                           "' referenced before assignment")
                              .str();
    mlir::Value messageBuffer =
        RuntimeBundleLowerer::materializeByteBuffer(loc, message);
    mlir::Value classId =
        mlir::arith::ConstantIntOp::create(builder, loc, 51, 64).getResult();
    mlir::Value length =
        mlir::arith::ConstantIntOp::create(
            builder, loc, static_cast<std::int64_t>(message.size()), 64)
            .getResult();
    mlir::func::CallOp::create(
        builder, loc, raise,
        mlir::ValueRange{classId, messageBuffer, length});
  }

  llvm::SmallVector<mlir::Value, 8> values;
  if (mlir::failed(RuntimeBundleLowerer::loadObjectGlobalValues(
          op, op.getName(), *valueTypes, values)))
    return mlir::failure();
  // The reader takes its own reference so a later rebinding of the global
  // cannot release the object out from under it -- AND records that the frame
  // is the holder, which is what makes `refcount-insertion` give it back.
  //
  // Why NOT `retainAggregateSlot` here, which is what this did: that helper
  // emits the retain alone. It is the SLOT idiom -- the reference it takes
  // belongs to the aggregate being stored into, so the store is the record and
  // no frame token is wanted. Nothing is stored here; the reader is the holder,
  // and a retain with no holder recorded is a leak by construction. Measured:
  // one reference per object-global read, so `ORIGIN.x + ORIGIN.y` left the
  // instance at 3 and the rebinding release could not reach 0.
  if (ownership::isObjectHeaderLikeType(values.front().getType())) {
    std::optional<RuntimeValue> owned =
        RuntimeBundleLowerer::retainEvidenceElement(
            op.getOperation(), RuntimeValue::object(type, values),
            /*atOperation=*/true);
    // Not a fall-back to the borrowed binding: for a header-fronted global the
    // reference IS required, so a missing retain primitive has to be reported
    // rather than quietly dropped back to the shape that leaks.
    if (!owned)
      return op.emitError()
             << "module global '" << op.getName() << "' of " << type
             << " has no runtime retain primitive, so a read cannot take the "
                "reference that protects it from a later rebinding";
    values.assign(owned->values.begin(), owned->values.end());
  }
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
          op.getOperation(), type, values, result,
          ownership::OwnershipKind::Own)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerObjectGlobalSet(py::GlobalSetOp op) {
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (!value)
    return op.emitError() << "module global assignment value has no bundle";
  mlir::Type type = op.getValue().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, type,
                                                 "module global object ABI");
  if (mlir::failed(valueTypes))
    return mlir::failure();

  // An int that never needed a box carries only its i64 lane, and the cell
  // stores objects. Boxing here is what lets a module global hold a value
  // wider than the machine word.
  //
  // ⛔ Why NOT let the fresh object BE the slot's reference and skip the
  // retain below: the materialized `__new__` result is an owned local like
  // any other, and `insertOwnedResultReleases` releases it after the store.
  // Skipping the retain left the slot holding a freed object, and the next
  // read of that global aborted with "Ly_IncRef observed non-positive
  // refcount" -- `golden.cases.module_globals` and
  // `golden.cases.float_floordiv_mod_round`, both of which store a boxed
  // value and read it back.
  RuntimeBundle boxed;
  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(*value)) {
    builder.setInsertionPoint(op);
    mlir::FailureOr<RuntimeValue> object =
        RuntimeBundleLowerer::materializePrimitiveI64Object(op, *value);
    if (mlir::failed(object))
      return mlir::failure();
    boxed = RuntimeBundle::objectWithOwnership(object->contract, object->values,
                                               ownership::OwnershipKind::Own);
    value = &boxed;
  }
  llvm::ArrayRef<mlir::Value> newValues = value->physicalValues();
  if (newValues.size() != valueTypes->size())
    return op.emitError() << "module global '" << op.getName()
                          << "' assignment value group has "
                          << newValues.size() << " values, expected "
                          << valueTypes->size();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  // Retain the new value before releasing the old one so a self-assignment
  // (X = X) never drops the object to zero in between.
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
          op.getOperation(), type, newValues, "module.global")))
    return mlir::failure();

  mlir::Value bound = RuntimeBundleLowerer::loadObjectGlobalWord(
      op, op.getName(), "init");
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value wasBound = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, bound, zero);
  auto releaseOld = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                            wasBound,
                                            /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&releaseOld.getThenRegion().front());
    llvm::SmallVector<mlir::Value, 8> oldValues;
    if (mlir::failed(RuntimeBundleLowerer::loadObjectGlobalValues(
            op, op.getName(), *valueTypes, oldValues)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op.getOperation(), type, oldValues, "module.global")))
      return mlir::failure();
  }

  for (auto [index, pair] : llvm::enumerate(llvm::zip(newValues,
                                                      *valueTypes))) {
    auto [newValue, valueType] = pair;
    std::string slot = std::to_string(index);
    if (rankOneElementType(valueType)) {
      // ⛔ The descriptor's aligned member, NOT
      // `memref.extract_aligned_pointer_as_index`. Both reach the pointer and
      // only this one leaves it a pointer: the index op is documented in the
      // memory model as where provenance is lost, and the cell it fed held an
      // integer that the read side had to widen back. The cast is erased
      // against the func-to-LLVM conversion's own inverse.
      mlir::Value descriptor =
          mlir::UnrealizedConversionCastOp::create(
              builder, loc, mlir::TypeRange{memrefDescriptorType(builder)},
              newValue)
              .getResult(0);
      mlir::Value pointer = mlir::LLVM::ExtractValueOp::create(
          builder, loc, descriptor, llvm::ArrayRef<std::int64_t>{1});
      RuntimeBundleLowerer::storeObjectGlobalPointer(op, op.getName(),
                                                     "p" + slot, pointer);
      auto memref = mlir::cast<mlir::MemRefType>(valueType);
      mlir::Value size;
      if (memref.hasStaticShape()) {
        size = mlir::arith::ConstantIntOp::create(
                   builder, loc, memref.getDimSize(0), 64)
                   .getResult();
      } else {
        mlir::Value dim =
            mlir::memref::DimOp::create(builder, loc, newValue, 0);
        size = mlir::arith::IndexCastOp::create(builder, loc,
                                                builder.getI64Type(), dim);
      }
      RuntimeBundleLowerer::storeObjectGlobalWord(op, op.getName(),
                                                  "s" + slot, size);
      continue;
    }
    mlir::Value word;
    if (valueType.isInteger(64)) {
      word = newValue;
    } else if (valueType.isInteger(1)) {
      word = mlir::arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                          newValue)
                 .getResult();
    } else if (valueType.isF64()) {
      word = mlir::arith::BitcastOp::create(builder, loc,
                                            builder.getI64Type(), newValue)
                 .getResult();
    } else {
      return op.emitError() << "module global '" << op.getName()
                            << "' value " << index
                            << " has unsupported scalar type " << valueType;
    }
    RuntimeBundleLowerer::storeObjectGlobalWord(op, op.getName(), "v" + slot,
                                                word);
  }
  mlir::Value one =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 64).getResult();
  RuntimeBundleLowerer::storeObjectGlobalWord(op, op.getName(), "init", one);
  erase.push_back(op);
  return mlir::success();
}

// The store side of the same split (see lowerGlobalGet): an address cell takes
// the word, a marked int takes the object path, and only the runtime's own
// modules still unbox an int into the native cell below.
//
// ⛔ Why NOT drop `int` from the storage-backed set in `EmitterClasses.cpp`
// altogether, so these globals take `lowerObjectGlobalSet` like every other
// contract: `stackguard_support.py` keeps function POINTERS and a file
// descriptor in int globals and reads them from a signal handler that may not
// allocate. That is what `EmitOptions::runtimeInternal` exempts instead.
//
// The measurement that located the boundary, kept because it is the reason
// the exemption is per module and not per name:
//
//   - the breakage is EXACTLY 31 errors in that ONE file. Nothing else in the
//     tree needs the native cell: `io.py`'s SEEK_* and `posixpath.py`'s
//     _S_IF* are small constants an arbitrary-precision cell serves fine, and
//     every other `: int =` under `runtime/lib` is a local, not a global.
//   - so the exemption wants to be per-module, and `moduleName` cannot carry
//     it: the runtime's own modules are emitted as `__main__` too, so the
//     collector cannot tell them apart by name.
mlir::LogicalResult RuntimeBundleLowerer::lowerGlobalSet(py::GlobalSetOp op) {
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (!value)
    return op.emitError() << "module global assignment value has no bundle";
  // The operand already carries the DECLARED type: the emitter coerces every
  // module-global assignment to it before building this op.
  if (RuntimeBundleLowerer::isAddressGlobalType(op.getValue().getType()))
    return RuntimeBundleLowerer::lowerAddressGlobalSet(op, *value);
  if (runtimeContractName(op.getValue().getType()) != "builtins.int" ||
      op->hasAttr("ly.global.boxed"))
    return RuntimeBundleLowerer::lowerObjectGlobalSet(op);
  return RuntimeBundleLowerer::lowerNativeIntGlobalSet(op, *value);
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerNativeIntGlobalSet(py::GlobalSetOp op,
                                              const RuntimeBundle &value) {
  builder.setInsertionPoint(op);
  mlir::Value raw;
  // The WORD is only the value when its VALID flag is a compile-time true.
  // A runtime flag means the callee's speculative i64 lane is a placeholder --
  // `range`'s element carries `valid = arith.constant false` -- and storing it
  // printed the dummy 0 for `COUNT = add(COUNT, i)` in a loop.
  //
  // Why NOT keep the word and prove validity later: there is nothing to prove.
  // The lane is invalid BY CONSTRUCTION on the boxed path, and the boxed
  // payload is the only authority. Asking the flag is the same question
  // `AttributeOps`, `GetItemOps` and `SpecialMethodOps` ask of this lane.
  //
  // Why this could not be asked before: the unbox path it falls to needs a
  // release for the boxed value it reads, and the placement could not find one
  // when the box is produced inside the prim/boxed dispatch's `scf.if` and the
  // store sits in a later block (`g_limit = limit` in stackguard_support,
  // behind `if limit == 0: return`). That was a hole in
  // `insertOwnedResultReleases`, not in this file -- it refused ordinary
  // `t = a + b; if c: print(t)` too -- and is repaired by
  // `liftGroupToEnclosingRegionOp` (Passes/Ownership.cpp).
  std::optional<RuntimeSymbol> unbox =
      manifest.primitive(value.contractName(), "unbox.i64");
  bool boxedIsReachable =
      unbox &&
      unbox->function.getNumArguments() == value.physicalValues().size();
  if (primitiveI64LaneKnownValid(value.primitiveI64)) {
    raw = value.primitiveI64->value;
  } else if (boxedIsReachable) {
    mlir::func::CallOp unboxCall = RuntimeBundleLowerer::createRuntimeCall(
        op.getLoc(), *unbox, value.physicalValues());
    raw = unboxCall.getResult(0);
  } else if (value.primitiveI64 && value.primitiveI64->value) {
    // No boxed payload to read instead: a primitive-i64 clone lane carries
    // only the (value, valid) pair, so the lane IS the sole carrier and a
    // runtime flag is no reason to refuse it. Demanding the unbox here took
    // out three goldens whose global is assigned from exactly that shape
    // ("module global assignment value builtins.int has no unbox.i64
    // primitive"); the flag only chooses between two carriers when there are
    // two.
    raw = value.primitiveI64->value;
  } else {
    return op.emitError() << "module global assignment value "
                          << value.contractName()
                          << " has no unbox.i64 primitive";
  }
  RuntimeBundleLowerer::storeNativeGlobalWord(op, op.getName(), raw);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
