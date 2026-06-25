#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

std::string runtimeKey(llvm::StringRef contract, llvm::StringRef role,
                       llvm::StringRef name) {
  std::string key;
  key.reserve(contract.size() + role.size() + name.size() + 2);
  key.append(contract);
  key.push_back('\x1f');
  key.append(role);
  key.push_back('\x1f');
  key.append(name);
  return key;
}

bool isIntegerLiteralSpelling(llvm::StringRef spelling) {
  if (spelling.empty())
    return false;
  if (spelling.front() == '-')
    spelling = spelling.drop_front();
  return !spelling.empty() &&
         llvm::all_of(spelling, [](char ch) { return ch >= '0' && ch <= '9'; });
}

std::string runtimeContractName(mlir::Type type) {
  if (auto contract = mlir::dyn_cast<py::ContractType>(type))
    return contract.getContractName().str();
  if (mlir::isa<py::CallableType>(type))
    return "builtins.function";
  if (auto protocol = mlir::dyn_cast<py::ProtocolType>(type))
    if (protocol.getProtocolName() == "Callable")
      return "builtins.function";
  if (auto literal = mlir::dyn_cast<py::LiteralType>(type)) {
    llvm::StringRef spelling = literal.getSpelling();
    if (spelling == "True" || spelling == "False")
      return "builtins.bool";
    if (spelling == "None")
      return "types.NoneType";
    if (spelling.starts_with("\"") && spelling.ends_with("\""))
      return "builtins.str";
    if (isIntegerLiteralSpelling(spelling))
      return "builtins.int";
  }
  return "";
}

mlir::Type runtimeContractType(mlir::MLIRContext *context,
                               llvm::StringRef contract) {
  return py::ContractType::get(context, contract);
}

bool RuntimeSymbol::hasClassIdArgument(unsigned inputIndex) const {
  return llvm::is_contained(classIdArgumentIndices, inputIndex);
}

const RuntimeDefaultArgument *
RuntimeSymbol::defaultArgument(unsigned inputIndex) const {
  for (const RuntimeDefaultArgument &argument : defaultArguments)
    if (argument.inputIndex == inputIndex)
      return &argument;
  return nullptr;
}

bool sameTypeSequence(llvm::ArrayRef<mlir::Type> lhs,
                      llvm::ArrayRef<mlir::Type> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left != right)
      return false;
  return true;
}

std::string describeTypeSequence(llvm::ArrayRef<mlir::Type> types) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "(";
  llvm::interleaveComma(types, os, [&](mlir::Type type) { type.print(os); });
  os << ")";
  return os.str();
}

std::string describeValueTypes(mlir::ValueRange values) {
  llvm::SmallVector<mlir::Type, 4> types;
  for (mlir::Value value : values)
    types.push_back(value.getType());
  return describeTypeSequence(types);
}

llvm::SmallVector<mlir::Type, 4> takePrefix(llvm::ArrayRef<mlir::Type> types,
                                            unsigned count) {
  llvm::SmallVector<mlir::Type, 4> prefix;
  prefix.append(types.begin(), types.begin() + count);
  return prefix;
}

llvm::SmallVector<mlir::Type, 4> takeSlice(llvm::ArrayRef<mlir::Type> types,
                                           unsigned begin, unsigned end) {
  llvm::SmallVector<mlir::Type, 4> slice;
  slice.append(types.begin() + begin, types.begin() + end);
  return slice;
}

} // namespace py::runtime_lowering
