#include "Runtime/Ctypes/Internal.h"

namespace py::runtime_lowering::ctypes {

std::string describeNativeArgumentSource(const RuntimeBundle &source) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "source contract=" << source.contractName();
  os << ", primitive_i64=" << (source.primitiveI64 ? "yes" : "no");
  if (source.primitiveI64 && source.primitiveI64->valid)
    os << ", primitive_valid_known_true="
       << (isKnownTrue(source.primitiveI64->valid) ? "yes" : "no");
  if (source.ctypes) {
    os << ", ctypes=" << source.ctypes->ctypeName;
    os << ", ctypes_kind=";
    switch (source.ctypes->kind) {
    case RuntimeCtypesEvidence::Kind::Module:
      os << "module";
      break;
    case RuntimeCtypesEvidence::Kind::Cell:
      os << "cell";
      break;
    case RuntimeCtypesEvidence::Kind::Pointer:
      os << "pointer";
      break;
    case RuntimeCtypesEvidence::Kind::Library:
      os << "library";
      break;
    case RuntimeCtypesEvidence::Kind::Symbol:
      os << "symbol";
      break;
    case RuntimeCtypesEvidence::Kind::FieldDescriptor:
      os << "field_descriptor";
      break;
    }
    os << ", provenance=" << ctypesProvenanceName(source.ctypes->provenance);
    os << ", lifetime=" << ctypesLifetimeName(source.ctypes->lifetime);
    os << ", scalar=" << (source.ctypes->scalarValue ? "yes" : "no");
    if (source.ctypes->scalarValid)
      os << ", scalar_valid_known_true="
         << (isKnownTrue(source.ctypes->scalarValid) ? "yes" : "no");
    os << ", storage_address="
       << (source.ctypes->storageAddressValue ? "yes" : "no");
    if (source.ctypes->storageAddressValid)
      os << ", storage_address_valid_known_true="
         << (isKnownTrue(source.ctypes->storageAddressValid) ? "yes" : "no");
    os << ", address=" << (source.ctypes->addressValue ? "yes" : "no");
    if (source.ctypes->addressValid)
      os << ", address_valid_known_true="
         << (isKnownTrue(source.ctypes->addressValid) ? "yes" : "no");
    os << ", keepalive_edges=" << source.ctypes->keepAlive.size();
  }
  return os.str();
}

} // namespace py::runtime_lowering::ctypes
