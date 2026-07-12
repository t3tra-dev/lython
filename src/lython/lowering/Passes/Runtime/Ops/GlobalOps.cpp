#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

// Module-level int globals are backed by a single process-lifetime i64 cell.
// Reads/writes are a plain llvm.load/llvm.store, so accessing a module global
// never allocates -- an async-signal-safe channel for signal handlers to
// exchange primitive state (see docs/lowering-architecture.md). The stored
// representation is the UNBOXED i64 value; the boxed int object is
// reconstructed on demand at each read (box-on-read), and the value is
// unboxed at each write (unbox-on-write).
mlir::LLVM::GlobalOp
RuntimeBundleLowerer::moduleGlobalStorage(mlir::Operation *op,
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

mlir::LogicalResult RuntimeBundleLowerer::lowerGlobalGet(py::GlobalGetOp op) {
  if (runtimeContractName(op.getResult().getType()) != "builtins.int")
    return op.emitError()
           << "module global '" << op.getName()
           << "' lowering supports only builtins.int storage";
  mlir::LLVM::GlobalOp storage =
      RuntimeBundleLowerer::moduleGlobalStorage(op, op.getName());

  builder.setInsertionPoint(op);
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op.getLoc(), storage);
  mlir::Value raw = mlir::LLVM::LoadOp::create(builder, op.getLoc(),
                                               builder.getI64Type(), address);
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

mlir::LogicalResult RuntimeBundleLowerer::lowerGlobalSet(py::GlobalSetOp op) {
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (!value)
    return op.emitError() << "module global assignment value has no bundle";
  if (runtimeContractName(op.getValue().getType()) != "builtins.int")
    return op.emitError()
           << "module global '" << op.getName()
           << "' lowering supports only builtins.int storage";
  mlir::LLVM::GlobalOp storage =
      RuntimeBundleLowerer::moduleGlobalStorage(op, op.getName());

  builder.setInsertionPoint(op);
  mlir::Value raw;
  if (value->primitiveI64 && value->primitiveI64->value) {
    raw = value->primitiveI64->value;
  } else {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(value->contractName(), "unbox.i64");
    if (!unbox ||
        unbox->function.getNumArguments() != value->physicalValues().size())
      return op.emitError() << "module global assignment value "
                            << value->contractName()
                            << " has no unbox.i64 primitive";
    mlir::func::CallOp unboxCall = RuntimeBundleLowerer::createRuntimeCall(
        op.getLoc(), *unbox, value->physicalValues());
    raw = unboxCall.getResult(0);
  }
  mlir::Value address =
      mlir::LLVM::AddressOfOp::create(builder, op.getLoc(), storage);
  mlir::LLVM::StoreOp::create(builder, op.getLoc(), raw, address);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
