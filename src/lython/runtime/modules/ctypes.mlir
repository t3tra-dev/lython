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

module attributes {
  ly.typing.module = "ctypes",
  ly.typing.class_exports = [
    "ctypes.CDLL=ctypes.CDLL",
    "ctypes.WinDLL=ctypes.WinDLL",
    "ctypes.OleDLL=ctypes.OleDLL",
    "ctypes.PyDLL=ctypes.PyDLL",
    "ctypes.LibraryLoader=ctypes.LibraryLoader",
    "ctypes.ArgumentError=ctypes.ArgumentError",
    "ctypes.Structure=_ctypes.Structure",
    "ctypes.Union=_ctypes.Union",
    "ctypes.Array=_ctypes.Array",
    "ctypes._CData=_ctypes._CData",
    "ctypes._SimpleCData=_ctypes._SimpleCData",
    "ctypes._Pointer=_ctypes._Pointer",
    "ctypes._CArgObject=_ctypes._CArgObject",
    "ctypes.CFuncPtr=_ctypes.CFuncPtr",
    "ctypes.CField=_ctypes.CField",
    "ctypes.py_object=ctypes.py_object",
    "ctypes.c_bool=ctypes.c_bool",
    "ctypes.c_byte=ctypes.c_byte",
    "ctypes.c_ubyte=ctypes.c_ubyte",
    "ctypes.c_short=ctypes.c_short",
    "ctypes.c_ushort=ctypes.c_ushort",
    "ctypes.c_int=ctypes.c_int",
    "ctypes.c_uint=ctypes.c_uint",
    "ctypes.c_long=ctypes.c_long",
    "ctypes.c_ulong=ctypes.c_ulong",
    "ctypes.c_longlong=ctypes.c_longlong",
    "ctypes.c_ulonglong=ctypes.c_ulonglong",
    "ctypes.c_int8=ctypes.c_int8",
    "ctypes.c_uint8=ctypes.c_uint8",
    "ctypes.c_int16=ctypes.c_int16",
    "ctypes.c_uint16=ctypes.c_uint16",
    "ctypes.c_int32=ctypes.c_int32",
    "ctypes.c_uint32=ctypes.c_uint32",
    "ctypes.c_int64=ctypes.c_int64",
    "ctypes.c_uint64=ctypes.c_uint64",
    "ctypes.c_ssize_t=ctypes.c_ssize_t",
    "ctypes.c_size_t=ctypes.c_size_t",
    "ctypes.c_float=ctypes.c_float",
    "ctypes.c_double=ctypes.c_double",
    "ctypes.c_longdouble=ctypes.c_longdouble",
    "ctypes.c_char=ctypes.c_char",
    "ctypes.c_wchar=ctypes.c_wchar",
    "ctypes.c_void_p=ctypes.c_void_p",
    "ctypes.c_voidp=ctypes.c_void_p",
    "ctypes.c_char_p=ctypes.c_char_p",
    "ctypes.c_wchar_p=ctypes.c_wchar_p",
    "ctypes.HRESULT=ctypes.HRESULT",
    "ctypes.c_time_t=ctypes.c_time_t",
    "_ctypes._CData=_ctypes._CData",
    "_ctypes._CanCastTo=_ctypes._CanCastTo",
    "_ctypes._PointerLike=_ctypes._PointerLike",
    "_ctypes._CArgObject=_ctypes._CArgObject",
    "_ctypes._SimpleCData=_ctypes._SimpleCData",
    "_ctypes._Pointer=_ctypes._Pointer",
    "_ctypes.Array=_ctypes.Array",
    "_ctypes.CFuncPtr=_ctypes.CFuncPtr",
    "_ctypes.CField=_ctypes.CField",
    "_ctypes.Structure=_ctypes.Structure",
    "_ctypes.Union=_ctypes.Union",
    "ctypes.wintypes.DWORD=ctypes.c_ulong",
    "ctypes.wintypes.WORD=ctypes.c_ushort",
    "ctypes.wintypes.BYTE=ctypes.c_ubyte",
    "ctypes.wintypes.BOOL=ctypes.c_long",
    "ctypes.wintypes.HANDLE=ctypes.c_void_p",
    "ctypes.wintypes.LPVOID=ctypes.c_void_p",
    "ctypes.wintypes.LPCVOID=ctypes.c_void_p"
  ],
  ly.typing.callable_exports = [
    "ctypes.sizeof",
    "ctypes.alignment",
    "ctypes.byref",
    "ctypes.pointer",
    "ctypes.POINTER",
    "ctypes.cast",
    "ctypes.addressof",
    "ctypes.CFUNCTYPE"
  ],
  // Manifest Callable contracts for ctypes free functions. These replace the
  // C++ ctypesStaticCallableFactory signatures so imported ctypes callables are
  // typed from the manifest. Names not listed here still fall back to the C++
  // factory (byref/pointer/POINTER/cast remain generic function objects).
  ly.typing.function_names = [
    "ctypes.sizeof",
    "ctypes.alignment",
    "ctypes.addressof",
    "ctypes.byref",
    "ctypes.CFUNCTYPE",
    "ctypes.cast"
  ],
  ly.typing.function_contracts = [
    !py.callable<[!py.union<!py.contract<"_ctypes._CData">, !py.type<!py.contract<"_ctypes._CData">>>], arg_names = ["obj"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.union<!py.contract<"_ctypes._CData">, !py.type<!py.contract<"_ctypes._CData">>>], arg_names = ["obj"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"_ctypes._CData">], arg_names = ["obj"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"typing.Any">], arg_names = ["obj"], arg_defaults = [false], returns = [!py.contract<"_ctypes._CArgObject">]>,
    !py.overload<[
      !py.callable<[!py.type<!py.typevar<"R">>], vararg = !py.contract<"builtins.tuple", [!py.contract<"typing.Any">]>, arg_names = ["restype"], arg_defaults = [false], returns = [!py.type<!py.contract<"_ctypes.CFuncPtr", [!py.typevar<"R">]>>]>,
      !py.callable<[!py.literal<None>], vararg = !py.contract<"builtins.tuple", [!py.contract<"typing.Any">]>, arg_names = ["restype"], arg_defaults = [false], returns = [!py.type<!py.contract<"_ctypes.CFuncPtr", [!py.literal<None>]>>]>
    ]>,
    !py.callable<[!py.contract<"typing.Any">, !py.type<!py.typevar<"T">>], arg_names = ["obj", "typ"], arg_defaults = [false, false], returns = [!py.typevar<"T">]>
  ]
} {
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
    base_names = ["WriteableBuffer"],
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
    // T = the Python type calls of this pointer return. `fn.restype = c_X`
    // refines it via field_param_bindings ("field:param:via_base"): the
    // assigned scalar's `_SimpleCData[V]` base argument becomes T, so
    // `__call__` (returning $T below) types concretely instead of Any.
    ly.typing.params = ["T"],
    ly.typing.param_defaults = [!py.contract<"typing.Any">],
    ly.typing.field_param_bindings = ["restype:T:_SimpleCData"],
    field_names = ["restype", "argtypes", "errcheck"],
    field_contract_types = [
      !py.union<!py.type<!py.contract<"_ctypes._CData">>, !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"typing.Any">]>, !py.literal<None>>,
      !py.protocol<"Sequence", [!py.type<!py.contract<"_ctypes._CData">>]>,
      !py.contract<"typing.Any">
    ],
    method_names = ["__init__", "__call__", "__bool__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_ctypes.CFuncPtr">, !py.contract<"typing.Any">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.CFuncPtr">, !py.paramspec<"P">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"_ctypes.CFuncPtr">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance"]
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
    // Subclasses declare their layout via a `_fields_` class assignment;
    // each field READS/WRITES as the ctype's Python value type (the
    // `_SimpleCData[V]` base argument). "attr:via_base" -- the emitter's
    // generic aggregate-fields rule consumes this.
    ly.typing.fields_spec = "_fields_:_SimpleCData",
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
    ly.typing.fields_spec = "_fields_:_SimpleCData",
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
