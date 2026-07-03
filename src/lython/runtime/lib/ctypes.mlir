// Contract manifest for the statically supported ctypes surface.
//
// Source contracts:
//   - typeshed stdlib/ctypes/__init__.pyi on main. typeshed does not publish a
//     refs/heads/3.14 branch; the attempted raw 3.14 URL returned 404.
//   - typeshed stdlib/_ctypes.pyi on main.
//   - CPython 3.14 Lib/ctypes/__init__.py and Modules/_ctypes.
//
// This module is intentionally a contract manifest, not a dynamic ctypes
// runtime. CPython materializes _ctypes._CData objects with b_ptr,
// b_needsfree, b_base, and b_objects. Lython keeps the same semantic boundary
// as static proof/evidence terms (`ly.ctypes.*`) so erased ctypes values can
// lower to native cells, pointers, and calls without allocating Python objects.

module attributes {ly.typing.module = "ctypes"} {
  py.class @ReadableBuffer attributes {
    base_names = ["object"],
    ly.typing.abstract,
    ly.typeshed.contract = "_typeshed.ReadableBuffer"
  } {}

  py.class @WriteableBuffer attributes {
    base_names = ["ReadableBuffer"],
    ly.typing.abstract,
    ly.typeshed.contract = "_typeshed.WriteableBuffer"
  } {}

  py.class @_CData attributes {
    base_names = ["object", "WriteableBuffer"],
    ly.typing.abstract,
    ly.ctypes.proof_root,
    field_names = ["_b_base_", "_b_needsfree_", "_objects"],
    field_contract_types = [
      !py.contract<"builtins.int">,
      !py.contract<"builtins.bool">,
      !py.union<!py.protocol<"Mapping", [!py.contract<"typing.Any">, !py.contract<"builtins.int">]>, !py.literal<None>>
    ],
    method_names = ["from_address", "from_buffer", "from_buffer_copy",
                    "__ctypes_from_outparam__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"_ctypes._CData">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.overload<[
        !py.protocol<"Callable", [!py.type<!py.contract<"_ctypes._CData">>, !py.contract<"_typeshed.WriteableBuffer">] -> [!py.self]>,
        !py.protocol<"Callable", [!py.type<!py.contract<"_ctypes._CData">>, !py.contract<"_typeshed.WriteableBuffer">, !py.contract<"builtins.int">] -> [!py.self]>
      ]>,
      !py.overload<[
        !py.protocol<"Callable", [!py.type<!py.contract<"_ctypes._CData">>, !py.contract<"_typeshed.ReadableBuffer">] -> [!py.self]>,
        !py.protocol<"Callable", [!py.type<!py.contract<"_ctypes._CData">>, !py.contract<"_typeshed.ReadableBuffer">, !py.contract<"builtins.int">] -> [!py.self]>
      ]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes._CData">] -> [!py.self]>
    ],
    method_kinds = ["classmethod", "classmethod", "classmethod", "instance"]
  } {}

  py.class @_CanCastTo attributes {
    base_names = ["_CData"], ly.typing.abstract
  } {}

  py.class @_PointerLike attributes {
    base_names = ["_CanCastTo"], ly.typing.abstract
  } {}

  py.class @_CArgObject attributes {
    base_names = ["object"], ly.typing.final,
    ly.ctypes.kind = "call_argument"
  } {}

  py.class @_SimpleCData attributes {
    base_names = ["_CData"], ly.typing.params = ["T"],
    ly.typing.base_args = [[]],
    ly.ctypes.kind = "native_cell",
    field_names = ["value"],
    field_contract_types = [!py.contract<"$T">],
    method_names = ["__init__", "__ctypes_from_outparam__"],
    method_contracts = [
      !py.overload<[
        !py.protocol<"Callable", [!py.contract<"_ctypes._SimpleCData">] -> [!py.literal<None>]>,
        !py.protocol<"Callable", [!py.contract<"_ctypes._SimpleCData">, !py.contract<"$T">] -> [!py.literal<None>]>
      ]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes._SimpleCData">] -> [!py.contract<"$T">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @_Pointer attributes {
    base_names = ["_PointerLike", "_CData"],
    ly.typing.params = ["CT"],
    ly.typing.base_args = [[], []],
    ly.ctypes.kind = "pointer",
    field_names = ["contents"],
    field_contract_types = [!py.contract<"$CT">],
    method_names = ["__init__", "__getitem__", "__setitem__", "__bool__"],
    method_contracts = [
      !py.overload<[
        !py.protocol<"Callable", [!py.contract<"_ctypes._Pointer">] -> [!py.literal<None>]>,
        !py.protocol<"Callable", [!py.contract<"_ctypes._Pointer">, !py.contract<"$CT">] -> [!py.literal<None>]>
      ]>,
      !py.overload<[
        !py.protocol<"Callable", [!py.contract<"_ctypes._Pointer">, !py.contract<"builtins.int">] -> [!py.contract<"typing.Any">]>,
        !py.protocol<"Callable", [!py.contract<"_ctypes._Pointer">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.list", [!py.contract<"typing.Any">]>]>
      ]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes._Pointer">, !py.contract<"builtins.int">, !py.contract<"typing.Any">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes._Pointer">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance"]
  } {}

  py.class @Array attributes {
    base_names = ["_CData"], ly.typing.params = ["CT"],
    ly.typing.base_args = [[]],
    ly.ctypes.kind = "array",
    field_names = ["raw", "value"],
    field_contract_types = [
      !py.contract<"builtins.bytes">,
      !py.contract<"typing.Any">
    ],
    method_names = ["__init__", "__getitem__", "__setitem__", "__iter__", "__len__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_ctypes.Array">, !py.paramspec<"P">] -> [!py.literal<None>]>,
      !py.overload<[
        !py.protocol<"Callable", [!py.contract<"_ctypes.Array">, !py.contract<"builtins.int">] -> [!py.contract<"typing.Any">]>,
        !py.protocol<"Callable", [!py.contract<"_ctypes.Array">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.list", [!py.contract<"typing.Any">]>]>
      ]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.Array">, !py.contract<"builtins.int">, !py.contract<"typing.Any">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.Array">] -> [!py.protocol<"Iterator", [!py.contract<"typing.Any">]>]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.Array">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance"]
  } {}

  py.class @CFuncPtr attributes {
    base_names = ["_PointerLike", "_CData"],
    ly.typing.base_args = [[], []],
    ly.ctypes.kind = "function_pointer",
    field_names = ["restype", "argtypes", "errcheck"],
    field_contract_types = [
      !py.union<!py.type<!py.contract<"_ctypes._CData">>, !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"typing.Any">]>, !py.literal<None>>,
      !py.protocol<"Sequence", [!py.type<!py.contract<"_ctypes._CData">>]>,
      !py.contract<"typing.Any">
    ],
    method_names = ["__call__", "__bool__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_ctypes.CFuncPtr">, !py.paramspec<"P">] -> [!py.contract<"typing.Any">]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.CFuncPtr">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @CField attributes {
    base_names = ["object"], ly.typing.params = ["CT", "GetT", "SetT"],
    ly.ctypes.kind = "field_descriptor",
    field_names = ["offset", "size", "name", "type", "byte_offset",
                   "byte_size", "is_bitfield", "bit_offset", "bit_size",
                   "is_anonymous"],
    field_contract_types = [
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.str">,
      !py.type<!py.contract<"$CT">>,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.bool">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.bool">
    ],
    method_names = ["__get__", "__set__"],
    method_contracts = [
      !py.overload<[
        !py.protocol<"Callable", [!py.contract<"_ctypes.CField">, !py.literal<None>, !py.union<!py.type<!py.contract<"builtins.object">>, !py.literal<None>>] -> [!py.self]>,
        !py.protocol<"Callable", [!py.contract<"_ctypes.CField">, !py.contract<"typing.Any">, !py.union<!py.type<!py.contract<"builtins.object">>, !py.literal<None>>] -> [!py.contract<"$GetT">]>
      ]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.CField">, !py.contract<"typing.Any">, !py.contract<"$SetT">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @Structure attributes {
    base_names = ["_CData"], ly.ctypes.kind = "struct",
    field_names = ["_fields_", "_pack_", "_anonymous_", "_align_", "_layout_"],
    field_contract_types = [
      !py.protocol<"Sequence", [!py.contract<"builtins.tuple">]>,
      !py.contract<"builtins.int">,
      !py.protocol<"Sequence", [!py.contract<"builtins.str">]>,
      !py.contract<"builtins.int">,
      !py.union<!py.literal<"\"ms\"">, !py.literal<"\"gcc-sysv\"">>
    ],
    method_names = ["__init__", "__getattr__", "__setattr__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_ctypes.Structure">, !py.paramspec<"P">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.Structure">, !py.contract<"builtins.str">] -> [!py.contract<"typing.Any">]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.Structure">, !py.contract<"builtins.str">, !py.contract<"typing.Any">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance"]
  } {}

  py.class @Union attributes {
    base_names = ["_CData"], ly.ctypes.kind = "union",
    field_names = ["_fields_", "_pack_", "_anonymous_", "_align_"],
    field_contract_types = [
      !py.protocol<"Sequence", [!py.contract<"builtins.tuple">]>,
      !py.contract<"builtins.int">,
      !py.protocol<"Sequence", [!py.contract<"builtins.str">]>,
      !py.contract<"builtins.int">
    ],
    method_names = ["__init__", "__getattr__", "__setattr__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_ctypes.Union">, !py.paramspec<"P">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.Union">, !py.contract<"builtins.str">] -> [!py.contract<"typing.Any">]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.Union">, !py.contract<"builtins.str">, !py.contract<"typing.Any">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance"]
  } {}

  py.class @CDLL attributes {
    base_names = ["object"], ly.ctypes.kind = "library",
    field_names = ["_name", "_handle", "_FuncPtr"],
    field_contract_types = [
      !py.contract<"builtins.str">,
      !py.contract<"builtins.int">,
      !py.type<!py.contract<"_ctypes.CFuncPtr">>
    ],
    method_names = ["__init__", "__getattr__", "__getitem__"],
    method_contracts = [
      !py.overload<[
        !py.protocol<"Callable", [!py.contract<"ctypes.CDLL">, !py.union<!py.contract<"builtins.str">, !py.literal<None>>] -> [!py.literal<None>]>,
        !py.protocol<"Callable", [!py.contract<"ctypes.CDLL">, !py.union<!py.contract<"builtins.str">, !py.literal<None>>, !py.contract<"builtins.int">, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.contract<"builtins.bool">, !py.contract<"builtins.bool">, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.literal<None>]>
      ]>,
      !py.protocol<"Callable", [!py.contract<"ctypes.CDLL">, !py.contract<"builtins.str">] -> [!py.contract<"_ctypes.CFuncPtr">]>,
      !py.protocol<"Callable", [!py.contract<"ctypes.CDLL">, !py.contract<"builtins.str">] -> [!py.contract<"_ctypes.CFuncPtr">]>
    ],
    method_kinds = ["instance", "instance", "instance"]
  } {}

  py.class @WinDLL attributes {
    base_names = ["CDLL"], ly.ctypes.kind = "library",
    ly.ctypes.abi = "stdcall"
  } {}

  py.class @OleDLL attributes {
    base_names = ["CDLL"], ly.ctypes.kind = "library",
    ly.ctypes.abi = "ole"
  } {}

  py.class @PyDLL attributes {
    base_names = ["CDLL"], ly.ctypes.kind = "library",
    ly.ctypes.abi = "pythonapi"
  } {}

  py.class @LibraryLoader attributes {
    base_names = ["object"], ly.typing.params = ["DLLT"],
    ly.ctypes.kind = "library_loader",
    method_names = ["__init__", "__getattr__", "__getitem__", "LoadLibrary"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"ctypes.LibraryLoader">, !py.type<!py.contract<"$DLLT">>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"ctypes.LibraryLoader">, !py.contract<"builtins.str">] -> [!py.contract<"$DLLT">]>,
      !py.protocol<"Callable", [!py.contract<"ctypes.LibraryLoader">, !py.contract<"builtins.str">] -> [!py.contract<"$DLLT">]>,
      !py.protocol<"Callable", [!py.contract<"ctypes.LibraryLoader">, !py.contract<"builtins.str">] -> [!py.contract<"$DLLT">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance"]
  } {}

  py.class @ArgumentError attributes {base_names = ["Exception"]} {}

  // Scalar ctype classes. `ly.ctypes.code` follows CPython's _type_ code;
  // width/signedness are boundary ABI facts consumed by static ctypes evidence.
  py.class @py_object attributes {base_names = ["_CanCastTo", "_SimpleCData"],
                                 ly.typing.base_args = [[], [!py.contract<"typing.Any">]],
                                 ly.ctypes.kind = "py_object",
                                 ly.ctypes.code = "O"} {}
  py.class @c_bool attributes {base_names = ["_SimpleCData"],
                              ly.typing.base_args = [[!py.contract<"builtins.bool">]],
                              ly.ctypes.kind = "scalar", ly.ctypes.code = "?",
                              ly.ctypes.width = 8 : i64} {}
  py.class @c_byte attributes {base_names = ["_SimpleCData"],
                              ly.typing.base_args = [[!py.contract<"builtins.int">]],
                              ly.ctypes.kind = "scalar", ly.ctypes.code = "b",
                              ly.ctypes.width = 8 : i64,
                              ly.ctypes.signed} {}
  py.class @c_ubyte attributes {base_names = ["_SimpleCData"],
                               ly.typing.base_args = [[!py.contract<"builtins.int">]],
                               ly.ctypes.kind = "scalar", ly.ctypes.code = "B",
                               ly.ctypes.width = 8 : i64,
                               ly.ctypes.unsigned} {}
  py.class @c_short attributes {base_names = ["_SimpleCData"],
                               ly.typing.base_args = [[!py.contract<"builtins.int">]],
                               ly.ctypes.kind = "scalar", ly.ctypes.code = "h",
                               ly.ctypes.width = 16 : i64,
                               ly.ctypes.signed} {}
  py.class @c_ushort attributes {base_names = ["_SimpleCData"],
                                ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                ly.ctypes.kind = "scalar", ly.ctypes.code = "H",
                                ly.ctypes.width = 16 : i64,
                                ly.ctypes.unsigned} {}
  py.class @c_int attributes {base_names = ["_SimpleCData"],
                             ly.typing.base_args = [[!py.contract<"builtins.int">]],
                             ly.ctypes.kind = "scalar", ly.ctypes.code = "i",
                             ly.ctypes.width = 32 : i64,
                             ly.ctypes.signed} {}
  py.class @c_uint attributes {base_names = ["_SimpleCData"],
                              ly.typing.base_args = [[!py.contract<"builtins.int">]],
                              ly.ctypes.kind = "scalar", ly.ctypes.code = "I",
                              ly.ctypes.width = 32 : i64,
                              ly.ctypes.unsigned} {}
  py.class @c_long attributes {base_names = ["_SimpleCData"],
                              ly.typing.base_args = [[!py.contract<"builtins.int">]],
                              ly.ctypes.kind = "scalar", ly.ctypes.code = "l",
                              ly.ctypes.width = -1 : i64,
                              ly.ctypes.c_integer = "long",
                              ly.ctypes.signed} {}
  py.class @c_ulong attributes {base_names = ["_SimpleCData"],
                               ly.typing.base_args = [[!py.contract<"builtins.int">]],
                               ly.ctypes.kind = "scalar", ly.ctypes.code = "L",
                               ly.ctypes.width = -1 : i64,
                               ly.ctypes.c_integer = "unsigned long",
                               ly.ctypes.unsigned} {}
  py.class @c_longlong attributes {base_names = ["_SimpleCData"],
                                  ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                  ly.ctypes.kind = "scalar", ly.ctypes.code = "q",
                                  ly.ctypes.width = 64 : i64,
                                  ly.ctypes.signed} {}
  py.class @c_ulonglong attributes {base_names = ["_SimpleCData"],
                                   ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                   ly.ctypes.kind = "scalar", ly.ctypes.code = "Q",
                                   ly.ctypes.width = 64 : i64,
                                   ly.ctypes.unsigned} {}
  py.class @c_int8 attributes {base_names = ["c_byte"], ly.ctypes.alias_of = "ctypes.c_byte"} {}
  py.class @c_uint8 attributes {base_names = ["c_ubyte"], ly.ctypes.alias_of = "ctypes.c_ubyte"} {}
  py.class @c_int16 attributes {base_names = ["c_short"], ly.ctypes.alias_of = "ctypes.c_short"} {}
  py.class @c_uint16 attributes {base_names = ["c_ushort"], ly.ctypes.alias_of = "ctypes.c_ushort"} {}
  py.class @c_int32 attributes {base_names = ["c_int"], ly.ctypes.alias_of = "ctypes.c_int"} {}
  py.class @c_uint32 attributes {base_names = ["c_uint"], ly.ctypes.alias_of = "ctypes.c_uint"} {}
  py.class @c_int64 attributes {base_names = ["c_longlong"], ly.ctypes.alias_of = "ctypes.c_longlong"} {}
  py.class @c_uint64 attributes {base_names = ["c_ulonglong"], ly.ctypes.alias_of = "ctypes.c_ulonglong"} {}
  py.class @c_ssize_t attributes {base_names = ["_SimpleCData"],
                                 ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                 ly.ctypes.kind = "scalar",
                                 ly.ctypes.c_integer = "ssize_t",
                                 ly.ctypes.width = -1 : i64,
                                 ly.ctypes.signed} {}
  py.class @c_size_t attributes {base_names = ["_SimpleCData"],
                                ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                ly.ctypes.kind = "scalar",
                                ly.ctypes.c_integer = "size_t",
                                ly.ctypes.width = -1 : i64,
                                ly.ctypes.unsigned} {}
  py.class @c_float attributes {base_names = ["_SimpleCData"],
                               ly.typing.base_args = [[!py.contract<"builtins.float">]],
                               ly.ctypes.kind = "scalar", ly.ctypes.code = "f",
                               ly.ctypes.width = 32 : i64} {}
  py.class @c_double attributes {base_names = ["_SimpleCData"],
                                ly.typing.base_args = [[!py.contract<"builtins.float">]],
                                ly.ctypes.kind = "scalar", ly.ctypes.code = "d",
                                ly.ctypes.width = 64 : i64} {}
  py.class @c_longdouble attributes {base_names = ["_SimpleCData"],
                                    ly.typing.base_args = [[!py.contract<"builtins.float">]],
                                    ly.ctypes.kind = "unsupported_scalar",
                                    ly.ctypes.code = "g"} {}
  py.class @c_char attributes {base_names = ["_SimpleCData"],
                              ly.typing.base_args = [[!py.contract<"builtins.bytes">]],
                              ly.ctypes.kind = "scalar", ly.ctypes.code = "c",
                              ly.ctypes.width = 8 : i64} {}
  py.class @c_wchar attributes {base_names = ["_SimpleCData"],
                               ly.typing.base_args = [[!py.contract<"builtins.str">]],
                               ly.ctypes.kind = "unsupported_scalar",
                               ly.ctypes.code = "u"} {}
  py.class @c_void_p attributes {base_names = ["_PointerLike", "_SimpleCData"],
                                ly.typing.base_args = [[], [!py.union<!py.contract<"builtins.int">, !py.literal<None>>]],
                                ly.ctypes.kind = "void_pointer",
                                ly.ctypes.code = "P"} {}
  py.class @c_char_p attributes {base_names = ["_PointerLike", "_SimpleCData"],
                                ly.typing.base_args = [[], [!py.union<!py.contract<"builtins.bytes">, !py.literal<None>>]],
                                ly.ctypes.kind = "unsupported_pointer",
                                ly.ctypes.code = "z"} {}
  py.class @c_wchar_p attributes {base_names = ["_PointerLike", "_SimpleCData"],
                                 ly.typing.base_args = [[], [!py.union<!py.contract<"builtins.str">, !py.literal<None>>]],
                                 ly.ctypes.kind = "unsupported_pointer",
                                 ly.ctypes.code = "Z"} {}

  py.class @HRESULT attributes {base_names = ["c_long"],
                               ly.ctypes.alias_of = "ctypes.c_long"} {}
  py.class @c_time_t attributes {base_names = ["_SimpleCData"],
                                ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                ly.ctypes.kind = "scalar",
                                ly.ctypes.c_integer = "time_t",
                                ly.ctypes.width = -1 : i64,
                                ly.ctypes.signed} {}
}
