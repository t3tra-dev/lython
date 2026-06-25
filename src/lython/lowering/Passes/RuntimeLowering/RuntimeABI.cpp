#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

RuntimeValue RuntimeValue::object(mlir::Type contract,
                                  mlir::ValueRange values) {
  RuntimeValue value;
  value.contract = contract;
  value.values.append(values.begin(), values.end());
  return value;
}

std::string RuntimeValue::contractName() const {
  return runtimeContractName(contract);
}

RuntimeBundle RuntimeBundle::object(mlir::Type contract,
                                    mlir::ValueRange values) {
  RuntimeBundle bundle;
  bundle.kind = Kind::Object;
  bundle.contract = contract;
  bundle.objectValue = RuntimeValue::object(contract, values);
  return bundle;
}

RuntimeBundle RuntimeBundle::aggregate(mlir::Type contract,
                                       mlir::ValueRange operands) {
  RuntimeBundle bundle;
  bundle.kind = Kind::Aggregate;
  bundle.contract = contract;
  bundle.aggregateOperands.append(operands.begin(), operands.end());
  return bundle;
}

RuntimeBundle RuntimeBundle::builtinCallable(mlir::Type contract,
                                             llvm::StringRef binding) {
  RuntimeBundle bundle;
  bundle.kind = Kind::BuiltinCallable;
  bundle.contract = contract;
  bundle.binding = binding.str();
  return bundle;
}

RuntimeBundle RuntimeBundle::typeObject(mlir::Type typeContract,
                                        mlir::Type instanceContract) {
  RuntimeBundle bundle;
  bundle.kind = Kind::TypeObject;
  bundle.contract = typeContract;
  bundle.instanceContract = instanceContract;
  return bundle;
}

llvm::ArrayRef<mlir::Value> RuntimeBundle::physicalValues() const {
  return objectValue.values;
}

std::string RuntimeBundle::contractName() const {
  if (kind == Kind::Object)
    return objectValue.contractName();
  return runtimeContractName(contract);
}

std::string RuntimeBundle::instanceContractName() const {
  return runtimeContractName(instanceContract);
}

} // namespace py::runtime_lowering
