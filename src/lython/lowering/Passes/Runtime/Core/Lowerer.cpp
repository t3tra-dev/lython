#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

RuntimeBundleLowerer::RuntimeBundleLowerer(mlir::ModuleOp module)
    : module(module), context(module.getContext()), builder(context),
      manifest(module) {}

mlir::LogicalResult RuntimeBundleLowerer::lowerModule() {
  if (mlir::failed(manifest.verify()))
    return mlir::failure();
  if (mlir::failed(buildReturnedValueSummaries()))
    return mlir::failure();
  if (mlir::failed(buildReturnedCallableSummaries()))
    return mlir::failure();
  if (mlir::failed(buildCallableProtocolArgumentABIs()))
    return mlir::failure();
  if (mlir::failed(buildCallableArgumentEvidenceABIs()))
    return mlir::failure();
  if (mlir::failed(buildCallableAggregateEvidenceABIs()))
    return mlir::failure();
  if (mlir::failed(buildPrimitiveI64CallableClones()))
    return mlir::failure();
  if (mlir::failed(buildReturnedCoroutineSummaries()))
    return mlir::failure();
  if (mlir::failed(buildReturnedObjectEvidenceSummaries()))
    return mlir::failure();
  if (mlir::failed(buildReturnedStaticObjectSummaries()))
    return mlir::failure();
  if (mlir::failed(buildGeneratorResumeCloneSignatures()))
    return mlir::failure();
  if (mlir::failed(prepareCallableFunctionABIs()))
    return mlir::failure();
  if (mlir::failed(buildGeneratorResumeBodies()))
    return mlir::failure();
  if (mlir::failed(synthesizeSourceClassDeallocators()))
    return mlir::failure();
  if (mlir::failed(lowerStructuredTryOps()))
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 64> pyOps;
  module.walk([&](mlir::Operation *op) {
    if (!op->getDialect() || op->getDialect()->getNamespace() != "py")
      return;
    if (auto function = op->getParentOfType<mlir::func::FuncOp>())
      if (function->hasAttr("ly.generator.body_result"))
        return;
    if (RuntimeBundleLowerer::isCallableProtocolTemplate(
            op->getParentOfType<mlir::func::FuncOp>()))
      return;
    pyOps.push_back(op);
  });
  for (mlir::Operation *op : pyOps) {
    if (llvm::is_contained(erase, op))
      continue;
    if (mlir::failed(ensureOperationOperandBundles(op)))
      return mlir::failure();
    if (llvm::is_contained(erase, op))
      continue;
    if (mlir::failed(lowerPyOp(op)))
      return mlir::failure();
  }
  if (mlir::failed(eraseSourceGeneratorBodyFunctions()))
    return mlir::failure();
  if (mlir::failed(lowerFunctionReturns()))
    return mlir::failure();
  if (mlir::failed(eraseCallableProtocolTemplateFunctions()))
    return mlir::failure();
  if (mlir::failed(dropControlFlowLogicalBranchOperands()))
    return mlir::failure();
  if (mlir::failed(dropUnusedLogicalBlockArguments()))
    return mlir::failure();
  if (mlir::failed(eraseLoweredPyOps()))
    return mlir::failure();
  if (mlir::failed(eraseControlFlowLogicalBlockArguments()))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::eraseCallableLogicalEntryArgs()))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::generateBoxedReprHook()))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::generateBoxedReleaseHook()))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::generateGeneratorDropHook()))
    return mlir::failure();
  // Class ops survive eraseLoweredPyOps so the hooks above can resolve
  // source-class ids and method symbols; drop them now that dispatch is built.
  llvm::SmallVector<py::ClassOp, 8> classOps;
  module.walk([&](py::ClassOp classOp) { classOps.push_back(classOp); });
  for (py::ClassOp classOp : classOps)
    classOp->erase();
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::eraseSourceGeneratorBodyFunctions() {
  llvm::SmallVector<mlir::func::FuncOp, 8> generators;
  module.walk([&](mlir::func::FuncOp function) {
    if (function->hasAttr("ly.generator.body_result"))
      generators.push_back(function);
  });
  for (mlir::func::FuncOp function : generators) {
    llvm::erase_if(callableLogicalEntryArgCounts,
                   [&](CallableLogicalEntryArgs entryArgs) {
                     return entryArgs.function == function;
                   });
    function.erase();
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::eraseCallableProtocolTemplateFunctions() {
  llvm::SmallVector<mlir::func::FuncOp, 8> templates;
  module.walk([&](mlir::func::FuncOp function) {
    if (RuntimeBundleLowerer::isCallableProtocolTemplate(function))
      templates.push_back(function);
  });
  for (mlir::func::FuncOp function : templates)
    function.erase();
  return mlir::success();
}

} // namespace py::lowering
