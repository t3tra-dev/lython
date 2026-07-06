#include "Runtime/Ctypes/Internal.h"
#include "llvm/ADT/StringSwitch.h"

namespace py::lowering::ctypes {

struct CtypesNameMap {
  llvm::StringLiteral name;
  llvm::StringLiteral contract;
};

constexpr CtypesNameMap kCtypesPublicTypes[] = {
    {"CDLL", "ctypes.CDLL"},
    {"WinDLL", "ctypes.WinDLL"},
    {"OleDLL", "ctypes.OleDLL"},
    {"PyDLL", "ctypes.PyDLL"},
    {"LibraryLoader", "ctypes.LibraryLoader"},
    {"ArgumentError", "ctypes.ArgumentError"},
    {"Structure", "_ctypes.Structure"},
    {"Union", "_ctypes.Union"},
    {"Array", "_ctypes.Array"},
    {"_CData", "_ctypes._CData"},
    {"_SimpleCData", "_ctypes._SimpleCData"},
    {"_Pointer", "_ctypes._Pointer"},
    {"_CArgObject", "_ctypes._CArgObject"},
    {"CFuncPtr", "_ctypes.CFuncPtr"},
    {"CField", "_ctypes.CField"},
    {"py_object", "ctypes.py_object"},
    {"c_bool", "ctypes.c_bool"},
    {"c_byte", "ctypes.c_byte"},
    {"c_ubyte", "ctypes.c_ubyte"},
    {"c_short", "ctypes.c_short"},
    {"c_ushort", "ctypes.c_ushort"},
    {"c_int", "ctypes.c_int"},
    {"c_uint", "ctypes.c_uint"},
    {"c_long", "ctypes.c_long"},
    {"c_ulong", "ctypes.c_ulong"},
    {"c_longlong", "ctypes.c_longlong"},
    {"c_ulonglong", "ctypes.c_ulonglong"},
    {"c_int8", "ctypes.c_int8"},
    {"c_uint8", "ctypes.c_uint8"},
    {"c_int16", "ctypes.c_int16"},
    {"c_uint16", "ctypes.c_uint16"},
    {"c_int32", "ctypes.c_int32"},
    {"c_uint32", "ctypes.c_uint32"},
    {"c_int64", "ctypes.c_int64"},
    {"c_uint64", "ctypes.c_uint64"},
    {"c_ssize_t", "ctypes.c_ssize_t"},
    {"c_size_t", "ctypes.c_size_t"},
    {"c_float", "ctypes.c_float"},
    {"c_double", "ctypes.c_double"},
    {"c_longdouble", "ctypes.c_longdouble"},
    {"c_char", "ctypes.c_char"},
    {"c_wchar", "ctypes.c_wchar"},
    {"c_void_p", "ctypes.c_void_p"},
    {"c_voidp", "ctypes.c_void_p"},
    {"c_char_p", "ctypes.c_char_p"},
    {"c_wchar_p", "ctypes.c_wchar_p"},
    {"HRESULT", "ctypes.HRESULT"},
    {"c_time_t", "ctypes.c_time_t"},
};

constexpr CtypesNameMap kCtypesInternalTypes[] = {
    {"_CData", "_ctypes._CData"},
    {"_CanCastTo", "_ctypes._CanCastTo"},
    {"_PointerLike", "_ctypes._PointerLike"},
    {"_CArgObject", "_ctypes._CArgObject"},
    {"_SimpleCData", "_ctypes._SimpleCData"},
    {"_Pointer", "_ctypes._Pointer"},
    {"Array", "_ctypes.Array"},
    {"CFuncPtr", "_ctypes.CFuncPtr"},
    {"CField", "_ctypes.CField"},
    {"Structure", "_ctypes.Structure"},
    {"Union", "_ctypes.Union"},
};

constexpr CtypesNameMap kCtypesWinTypes[] = {
    {"DWORD", "ctypes.c_ulong"},    {"WORD", "ctypes.c_ushort"},
    {"BYTE", "ctypes.c_ubyte"},     {"BOOL", "ctypes.c_long"},
    {"HANDLE", "ctypes.c_void_p"},  {"LPVOID", "ctypes.c_void_p"},
    {"LPCVOID", "ctypes.c_void_p"},
};

llvm::StringRef stripCtypesModule(llvm::StringRef contract) {
  if (contract.consume_front("ctypes."))
    return contract;
  if (contract.consume_front("_ctypes."))
    return contract;
  return contract;
}

std::optional<std::string> lookupCtypesName(llvm::ArrayRef<CtypesNameMap> table,
                                            llvm::StringRef name) {
  for (const CtypesNameMap &entry : table)
    if (entry.name == name)
      return entry.contract.str();
  return std::nullopt;
}

std::optional<std::string> ctypesModuleAttrContract(llvm::StringRef moduleName,
                                                    llvm::StringRef attr) {
  if (moduleName == "ctypes")
    return lookupCtypesName(kCtypesPublicTypes, attr);
  if (moduleName == "_ctypes")
    return lookupCtypesName(kCtypesInternalTypes, attr);
  if (moduleName == "ctypes.wintypes")
    return lookupCtypesName(kCtypesWinTypes, attr);
  return std::nullopt;
}

std::optional<std::string> ctypesBareNameContract(llvm::StringRef name) {
  if (std::optional<std::string> publicName =
          lookupCtypesName(kCtypesPublicTypes, name))
    return publicName;
  if (std::optional<std::string> internalName =
          lookupCtypesName(kCtypesInternalTypes, name))
    return internalName;
  return lookupCtypesName(kCtypesWinTypes, name);
}

std::optional<std::string> ctypesQualifiedNameContract(llvm::StringRef name) {
  if (name.starts_with("ctypes.") || name.starts_with("_ctypes.")) {
    auto split = name.rsplit('.');
    if (std::optional<std::string> contract =
            ctypesModuleAttrContract(split.first, split.second))
      return contract;
    if (name.consume_front("ctypes.wintypes."))
      return ctypesModuleAttrContract("ctypes.wintypes", name);
  }
  return ctypesBareNameContract(name);
}

bool isStaticCtypesFunctionName(llvm::StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Cases({"sizeof", "alignment", "byref", "pointer", "POINTER", "cast",
              "addressof"},
             true)
      .Default(false);
}

mlir::Type ctypesContractType(mlir::MLIRContext *context,
                              llvm::StringRef contract) {
  return py::ContractType::get(context, contract);
}

RuntimeBundle makeCtypesModuleBundle(mlir::Type resultType,
                                     llvm::StringRef moduleName) {
  RuntimeBundle bundle = RuntimeBundle::object(resultType, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Module;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = moduleName.str();
  evidence.ctype = resultType;
  bundle.ctypes = std::move(evidence);
  return bundle;
}

} // namespace py::lowering::ctypes
