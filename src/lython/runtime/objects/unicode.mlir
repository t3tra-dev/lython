// `str` and its iterator -- CPython's Objects/unicodeobject.c, with what it
// includes from Objects/stringlib/ (find, split, partition) and
// unicode_format.h (str.format templates).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// Deviations from CPython:
//   - str.split takes the whitespace cap (`split(maxsplit=1)` and its
//     `split(None, 1)` spelling); str.rsplit does not, because its cap
//     withholds splits from the right, which the left-to-right walk cannot
//     produce. The cap rides the whitespace overload as a bare int, so
//     `split(1)` -- a TypeError in CPython, which reads the int as the
//     separator -- is accepted here.
//   - A decode failure says "invalid utf-8 sequence" rather than CPython's
//     codec message naming the byte and its position.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.str", "builtins.str_iterator"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyBytes_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"}
  func.func private @LyHost_WriteBytes(i32, memref<?xi8>, i64)
  func.func private @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 10 : i64, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.list"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyTuple_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 11 : i64, ly.runtime.contract = "builtins.tuple", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.tuple"}
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"}
  func.func private @__ly_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) -> index
  func.func private @__ly_box_slot_base_index(%slot: index) -> index
  func.func private @__ly_box_store_entity(%items: memref<?xi64>, %slot: i64, %class_id: i64, %entity: i64)
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_entity_word_get(%ptr: i64, %slot: i64) -> i64
  func.func private @__ly_entity_word_set(%ptr: i64, %slot: i64, %value: i64)
  memref.global "private" constant @__ly_fmt_msg_name_str : memref<3xi8>
  memref.global "private" constant @__ly_fmt_msg_z_str : memref<65xi8>
  func.func private @__ly_fmt_parse_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %out: memref<?xi64>) -> i1
  func.func private @__ly_fmt_raise_alt_str()
  func.func private @__ly_fmt_raise_bytes(%message: memref<?xi8>, %length: i64)
  func.func private @__ly_fmt_raise_cannot_group(%gcp: i64, %wcp: i64)
  func.func private @__ly_fmt_raise_eq_align_str()
  func.func private @__ly_fmt_raise_invalid_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %name: memref<?xi8>, %name_len: i64)
  func.func private @__ly_fmt_raise_sign_str()
  func.func private @__ly_fmt_raise_unknown_code(%code: i64, %name: memref<?xi8>, %name_len: i64)
  func.func private @__ly_fmt_str_from_cps(%cps: memref<?xi32>, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_global_view_i8(%pointer: i64, %size: i64) -> memref<?xi8>
  func.func private @__ly_hash_bytes(%ptr: i64, %len: i64) -> i64
  func.func private @__ly_hash_fixup(%h: i64) -> i64
  func.func private @__ly_list_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.list", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_repeat_overflows(%len: i64, %n: i64) -> i1
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64)
  func.func private @__ly_slice_raise_zero_step()
  func.func private @__ly_tuple_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}

  py.class @str attributes {
    base_names = ["Sequence", "Hashable"],
    ly.typing.base_args = [[!py.contract<"builtins.str">], []],
    method_names = ["__new__", "__len__", "__iter__", "__getitem__", "__getslice__",
                    "__add__", "__contains__", "__eq__", "__lt__", "__le__",
                    "__gt__", "__ge__", "join", "startswith", "startswith",
                    "startswith", "endswith", "endswith", "endswith", "__repr__",
                    "__str__", "__ne__", "encode", "upper", "lower",
                    "casefold", "title", "capitalize", "swapcase", "isalpha",
                    "isspace", "isdecimal", "isdigit", "isnumeric", "isupper",
                    "islower", "isprintable", "istitle", "isalnum", "isidentifier",
                    "isascii", "find", "find", "find", "rfind",
                    "rfind", "rfind", "index", "index", "index",
                    "rindex", "rindex", "rindex", "count", "count",
                    "count", "replace", "replace", "strip", "strip",
                    "lstrip", "lstrip", "rstrip", "rstrip", "removeprefix",
                    "removesuffix", "center", "center", "ljust", "ljust",
                    "rjust", "rjust", "zfill", "expandtabs", "expandtabs",
                    "__mul__", "split", "split", "split", "split",
                    "rsplit", "rsplit", "rsplit", "splitlines", "splitlines",
                    "partition", "rpartition", "__hash__", "__format__", "__ascii__",
                    "__fmt_next__", "__fmt_prefix__", "__fmt_tail__", "__fmt_conv__", "__fmt_spec__",
                    "__fmt_end__", "__fmt_pick__", "__ly_iadd__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.str">>, !py.contract<"builtins.object">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str_iterator">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.protocol<"Iterable", [!py.contract<"builtins.str">]>] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.tuple", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.tuple", [!py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  py.class @str_iterator attributes {
    base_names = ["Iterator"],
    ly.typing.base_args = [[!py.contract<"builtins.str">]],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.str_iterator">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.str_iterator">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  func.func private @__ly_str_boxed_by_contract(%box: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>, i1)

  // str(x) of a boxed payload value, with CPython's fallback chain: the class's
  // own __str__, then its __repr__, then the default <C object at 0x...>.
  func.func private @__ly_str_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "str_boxed_or_default", ly.runtime.result_contract = "builtins.str"} {
    %h, %b, %ok = func.call @__ly_str_boxed_by_contract(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
    cf.cond_br %ok, ^hooked, ^fallback

  ^hooked:
    func.return %h, %b : memref<2xi64>, memref<?xi8>

  ^fallback:
    %rh, %rb = func.call @__ly_repr_boxed_or_default(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %rh, %rb : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    // str is one allocation; the header view carries it.
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // str.encode() with the default encoding: re-encode the adaptive-width
  // code units to UTF-8.
  func.func @LyUnicode_Encode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "encode", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %length = func.call @__ly_unicode_utf8_length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %length_index = arith.index_cast %length : i64 to index
    %buffer = memref.alloc(%length_index) : memref<?xi8>
    func.call @__ly_unicode_utf8_fill(%header, %bytes, %buffer) : (memref<2xi64>, memref<?xi8>, memref<?xi8>) -> ()
    %result_header = func.call @LyBytes_FromBytes(%buffer, %c0, %length) : (memref<?xi8>, index, i64) -> memref<4xi64>
    memref.dealloc %buffer : memref<?xi8>
    func.return %result_header : memref<4xi64>
  }

  // ASCII-insensitive equality of a str operand against an ASCII literal:
  // encoding/error-handler names are latin-1 width by construction.
  func.func private @__ly_unicode_equals_ascii(%header: memref<2xi64>, %bytes: memref<?xi8>, %expected: memref<?xi8>, %expected_len: i64) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %true_bit = arith.constant true
    %false_bit = arith.constant false
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %narrow = arith.cmpi eq, %width, %one : i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %same_len = arith.cmpi eq, %len, %expected_len : i64
    %comparable = arith.andi %narrow, %same_len : i1
    %result = scf.if %comparable -> (i1) {
      %expected_index = arith.index_cast %expected_len : i64 to index
      %all = scf.for %i = %c0 to %expected_index step %c1 iter_args(%acc = %true_bit) -> (i1) {
        %a = memref.load %bytes[%i] : memref<?xi8>
        %b = memref.load %expected[%i] : memref<?xi8>
        %eq = arith.cmpi eq, %a, %b : i8
        %next = arith.andi %acc, %eq : i1
        scf.yield %next : i1
      }
      scf.yield %all : i1
    } else {
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  memref.global "private" constant @__ly_unicode_msg_int_too_large_c_int : memref<40xi8> = dense<[80, 121, 116, 104, 111, 110, 32, 105, 110, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 99, 111, 110, 118, 101, 114, 116, 32, 116, 111, 32, 67, 32, 105, 110, 116]>
  memref.global "private" constant @__ly_unicode_msg_repeat_too_long : memref<27xi8> = dense<[114, 101, 112, 101, 97, 116, 101, 100, 32, 115, 116, 114, 105, 110, 103, 32, 105, 115, 32, 116, 111, 111, 32, 108, 111, 110, 103]>
  // ===== impls: unicode =====

  // Retain a borrowed str and hand the same object back as an owned result.
  // Evidence-selected container elements are retained through this primitive so
  // they survive their container's release (checked retain premise).
  func.func private @LyUnicode_Shape() -> (memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.str", ly.runtime.shape}

  memref.global "private" constant @__ly_unicode_msg_string_index_out_of_range : memref<25xi8> = dense<[115, 116, 114, 105, 110, 103, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  func.func private @__ly_unicode_raise_index_error() {
    %class_id = arith.constant 55 : i64
    %length = arith.constant 25 : i64
    %message_static = memref.get_global @__ly_unicode_msg_string_index_out_of_range : memref<25xi8>
    %message = memref.cast %message_static : memref<25xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // "invalid utf-8 sequence"
  memref.global "private" constant @__ly_unicode_msg_invalid_utf8 : memref<22xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 117, 116, 102, 45, 56, 32, 115, 101, 113, 117, 101, 110, 99, 101]>

  func.func private @__ly_unicode_raise_decode_error() {
    %class_id = arith.constant 122 : i64
    %length = arith.constant 22 : i64
    %message_static = memref.get_global @__ly_unicode_msg_invalid_utf8 : memref<22xi8>
    %message = memref.cast %message_static : memref<22xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // PEP 393-style adaptive-width representation. One entity, one allocation:
  //   [0,16)               header [refcount, class id = 4]
  //   [16,24)              shape word: (data byte count << 3) | width, where
  //                        width is 1 (latin-1) / 2 (UCS-2) / 4 (UCS-4)
  //   [24,32)              capacity: data bytes the block has room for
  //   [32, 32+capacity)    little-endian code units
  // The shape word lives OUTSIDE the two-word header (rather than in spare
  // bits of header[1]) because header[1] is the class id that boxed payload
  // handles, repr dispatch and the release hook compare by equality.
  //
  // ⭐ THE BYTE COUNT IS RECORDED, AND IT USED TO LIVE ONLY IN THE DESCRIPTOR.
  // `__ly_unicode_count` read it with `memref.dim` off the bytes lane, so the
  // length existed only in the SECOND of the contract's two physical values --
  // which means a box had to cache that lane to be able to read the string
  // back. Three bits are enough for the width, and nothing else was using the
  // other sixty-one.
  //
  // ⭐ AND THE CAPACITY IS RECORDED, WHICH IS WHAT `s += x` NEEDS. CPython's
  // `unicode_concatenate` resizes the left operand in place when it holds the
  // only reference; `resize_compact` gets to call realloc, and this cannot --
  // the memory-safety argument (BoxLayout.h) rests on no allocation moving
  // under a held word. Room the block already has is the way to append without
  // moving, so the block carries how much it has.
  //
  // Why NOT a fourth word: it is 8 bytes on every string in the program. What
  // it buys is the difference between O(n) and O(n^2) for an accumulating
  // append, measured at 160,000 appends of ten bytes: 2.20 s -> the linear
  // path, against CPython's 0.01 s.
  //
  // ⛔ THE CAPACITY IS NOT SLACK BY DEFAULT. Every constructor here asks for
  // exactly what it stores, so a program that never appends pays the word and
  // nothing else; `LyUnicode_IAdd` is the only caller that asks for room, and
  // only when it has already had to move once.
  //
  // Canonical-form invariant: every constructor picks the smallest width that
  // fits the widest code point, so equal strings always have identical width
  // and identical code-unit bytes (equality can stay bytewise). Capacity is not
  // part of it -- two equal strings may have different room after them.
  func.func private @__ly_unicode_data_offset() -> i64 {
    %offset = arith.constant 32 : i64
    func.return %offset : i64
  }

  func.func private @__ly_unicode_alloc(%count: i64, %width: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "alloc"} {
    %data_bytes = arith.muli %count, %width : i64
    %header, %bytes = func.call @__ly_unicode_alloc_capacity(%count, %width, %data_bytes) : (i64, i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // The empty str every empty result is, as CPython's is a singleton: the
  // static block LyUnicode_FromStatic reads (width 1, nothing stored).
  memref.global "private" constant @__ly_str_empty : memref<32xi8> = dense<[-1, -1, -1, -1, -1, -1, -1, 127, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> {alignment = 16 : i64}
  func.func private @__ly_unicode_alloc_capacity(%count: i64, %width: i64, %capacity: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %zero_room = arith.constant 0 : i64
    %empty_room = arith.maxsi %capacity, %count : i64
    %is_empty = arith.cmpi eq, %empty_room, %zero_room : i64
    %r:2 = scf.if %is_empty -> (memref<2xi64>, memref<?xi8>) {
      %static = memref.get_global @__ly_str_empty : memref<32xi8>
      %block = memref.cast %static : memref<32xi8> to memref<?xi8>
      %h, %b = func.call @LyUnicode_FromStatic(%block) : (memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    } else {
      %prefix_bytes = arith.constant 32 : i64
      %one_byte = arith.constant 1 : i64
      %count_checked = func.call @__ly_alloc_count(%count, %width, %prefix_bytes) : (i64, i64, i64) -> index
      %data_bytes = arith.muli %count, %width : i64
      %room = arith.maxsi %capacity, %data_bytes : i64
      %byte_count = arith.index_cast %data_bytes : i64 to index
      %room_index = func.call @__ly_alloc_count(%room, %one_byte, %prefix_bytes) : (i64, i64, i64) -> index
      %block_prefix = arith.constant 32 : index
      %block_bytes = arith.addi %room_index, %block_prefix : index
      %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
      %header_offset = arith.constant 0 : index
      %width_offset = arith.constant 16 : index
      %capacity_offset = arith.constant 24 : index
      %bytes_offset = arith.constant 32 : index
      %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
      %width_view = memref.view %block[%width_offset][] : memref<?xi8> to memref<1xi64>
      %capacity_view = memref.view %block[%capacity_offset][] : memref<?xi8> to memref<1xi64>
      %bytes = memref.view %block[%bytes_offset][%byte_count] : memref<?xi8> to memref<?xi8>
      %one = arith.constant 1 : i64
      %layout_str = arith.constant 4 : i64
      %refcount_slot = arith.constant 0 : index
      %layout_slot = arith.constant 1 : index
      %width_slot = arith.constant 0 : index
      %shape_shift = arith.constant 3 : i64
      %shape_bytes = arith.shli %data_bytes, %shape_shift : i64
      %shape = arith.ori %shape_bytes, %width : i64
      memref.store %one, %header[%refcount_slot] : memref<2xi64>
      memref.store %layout_str, %header[%layout_slot] : memref<2xi64>
      memref.store %shape, %width_view[%width_slot] : memref<1xi64>
      memref.store %room, %capacity_view[%width_slot] : memref<1xi64>
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %r#0, %r#1 : memref<2xi64>, memref<?xi8>
  }

  // Character width of an existing str. Read through the header pointer: the
  // width word sits between the two public views, reachable from neither, and
  // every str header view is anchored at the block base by construction.
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64 {
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %width = func.call @__ly_unicode_raw_width(%ptr) : (i64) -> i64
    func.return %width : i64
  }

  // The shape word of a str block, by the block's address.
  func.func private @__ly_unicode_shape_word(%hdr_ptr: i64) -> i64 {
    %sixteen = arith.constant 16 : i64
    %addr = arith.addi %hdr_ptr, %sixteen : i64
    %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %shape = llvm.load %llptr : !llvm.ptr -> i64
    func.return %shape : i64
  }

  // ⭐ THE NON-FIRST LANES OF A CONTRACT, FROM THE FIRST ONE'S ADDRESS. A
  // payload box names one entity; every contract that is more than one physical
  // value answers here with (pointer, size) per remaining lane, so the box does
  // not have to carry them. One of these per multi-lane contract is what the
  // box's own width rests on.
  func.func private @__ly_unicode_lane_words(%hdr_ptr: i64) -> (i64, i64) attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "lane_words"} {
    %prefix = func.call @__ly_unicode_data_offset() : () -> i64
    %bytes_ptr = arith.addi %hdr_ptr, %prefix : i64
    %byte_len = func.call @__ly_unicode_raw_bytes(%hdr_ptr) : (i64) -> i64
    func.return %bytes_ptr, %byte_len : i64, i64
  }

  // Data bytes the block has room for (`__ly_unicode_alloc_capacity`).
  func.func private @__ly_unicode_raw_capacity(%hdr_ptr: i64) -> i64 {
    %offset = arith.constant 24 : i64
    %addr = arith.addi %hdr_ptr, %offset : i64
    %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %capacity = llvm.load %llptr : !llvm.ptr -> i64
    func.return %capacity : i64
  }

  // Republishes the length after an in-place append. The width does not move --
  // an append that would widen the string does not take the in-place path.
  func.func private @__ly_unicode_set_bytes(%hdr_ptr: i64, %byte_len: i64) {
    %sixteen = arith.constant 16 : i64
    %shift = arith.constant 3 : i64
    %mask = arith.constant 7 : i64
    %shape = func.call @__ly_unicode_shape_word(%hdr_ptr) : (i64) -> i64
    %width = arith.andi %shape, %mask : i64
    %shifted = arith.shli %byte_len, %shift : i64
    %next = arith.ori %shifted, %width : i64
    %addr = arith.addi %hdr_ptr, %sixteen : i64
    %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    llvm.store %next, %llptr : i64, !llvm.ptr
    func.return
  }

  // Byte length of a str's code-unit buffer, recovered from the block rather
  // than from a descriptor -- which is what lets a box hold only the block.
  func.func private @__ly_unicode_raw_bytes(%hdr_ptr: i64) -> i64 {
    %shape = func.call @__ly_unicode_shape_word(%hdr_ptr) : (i64) -> i64
    %shift = arith.constant 3 : i64
    %bytes = arith.shrui %shape, %shift : i64
    func.return %bytes : i64
  }

  // ⛔ THE BLOCK AND NOT THE DESCRIPTOR. The bytes lane is still taken as an
  // argument because every caller has it, but the length comes from the shape
  // word: a str read back out of a box has no descriptor to ask, and having the
  // two answers come from different places is how they would drift.
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64 {
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %shape = func.call @__ly_unicode_shape_word(%ptr) : (i64) -> i64
    %mask = arith.constant 7 : i64
    %width = arith.andi %shape, %mask : i64
    %shift = arith.constant 3 : i64
    %data_bytes = arith.shrui %shape, %shift : i64
    %count = arith.divsi %data_bytes, %width : i64
    func.return %count : i64
  }

  func.func private @__ly_unicode_width_for(%cp: i64) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %latin1_limit = arith.constant 256 : i64
    %ucs2_limit = arith.constant 65536 : i64
    %fits1 = arith.cmpi ult, %cp, %latin1_limit : i64
    %fits2 = arith.cmpi ult, %cp, %ucs2_limit : i64
    %wide = arith.select %fits2, %two, %four : i64
    %width = arith.select %fits1, %one, %wide : i64
    func.return %width : i64
  }

  // Code point at code-point index %i of a %width-wide code-unit buffer.
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %eight = arith.constant 8 : i64
    %sixteen = arith.constant 16 : i64
    %twenty_four = arith.constant 24 : i64
    %width_index = arith.index_cast %width : i64 to index
    %off = arith.muli %i, %width_index : index
    %is1 = arith.cmpi eq, %width, %one : i64
    %cp = scf.if %is1 -> (i64) {
      %b0 = memref.load %bytes[%off] : memref<?xi8>
      %v = arith.extui %b0 : i8 to i64
      scf.yield %v : i64
    } else {
      %off1 = arith.addi %off, %c1 : index
      %b0 = memref.load %bytes[%off] : memref<?xi8>
      %b1 = memref.load %bytes[%off1] : memref<?xi8>
      %v0 = arith.extui %b0 : i8 to i64
      %v1 = arith.extui %b1 : i8 to i64
      %v1s = arith.shli %v1, %eight : i64
      %lo = arith.ori %v0, %v1s : i64
      %is2 = arith.cmpi eq, %width, %two : i64
      %inner = scf.if %is2 -> (i64) {
        scf.yield %lo : i64
      } else {
        %off2 = arith.addi %off, %c2 : index
        %off3 = arith.addi %off, %c3 : index
        %b2 = memref.load %bytes[%off2] : memref<?xi8>
        %b3 = memref.load %bytes[%off3] : memref<?xi8>
        %v2 = arith.extui %b2 : i8 to i64
        %v3 = arith.extui %b3 : i8 to i64
        %v2s = arith.shli %v2, %sixteen : i64
        %v3s = arith.shli %v3, %twenty_four : i64
        %hi = arith.ori %v2s, %v3s : i64
        %full = arith.ori %lo, %hi : i64
        scf.yield %full : i64
      }
      scf.yield %inner : i64
    }
    func.return %cp : i64
  }

  func.func private @__ly_unicode_put(%bytes: memref<?xi8>, %width: i64, %i: index, %cp: i64) {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %eight = arith.constant 8 : i64
    %sixteen = arith.constant 16 : i64
    %twenty_four = arith.constant 24 : i64
    %width_index = arith.index_cast %width : i64 to index
    %off = arith.muli %i, %width_index : index
    %b0 = arith.trunci %cp : i64 to i8
    memref.store %b0, %bytes[%off] : memref<?xi8>
    %is_wide = arith.cmpi ne, %width, %one : i64
    scf.if %is_wide {
      %off1 = arith.addi %off, %c1 : index
      %s1 = arith.shrui %cp, %eight : i64
      %b1 = arith.trunci %s1 : i64 to i8
      memref.store %b1, %bytes[%off1] : memref<?xi8>
      %is_widest = arith.cmpi ne, %width, %two : i64
      scf.if %is_widest {
        %off2 = arith.addi %off, %c2 : index
        %off3 = arith.addi %off, %c3 : index
        %s2 = arith.shrui %cp, %sixteen : i64
        %s3 = arith.shrui %cp, %twenty_four : i64
        %b2 = arith.trunci %s2 : i64 to i8
        %b3 = arith.trunci %s3 : i64 to i8
        memref.store %b2, %bytes[%off2] : memref<?xi8>
        memref.store %b3, %bytes[%off3] : memref<?xi8>
      }
    }
    func.return
  }

  // Decode one UTF-8 sequence at byte offset %i (relative to %start) of a
  // %len-byte input. Returns (code point, next offset, ok). Strict: truncated
  // sequences, stray continuation bytes, overlong forms, surrogates and
  // values above U+10FFFF are rejected (ok = false).
  func.func private @__ly_utf8_step(%bytes: memref<?xi8>, %start: index, %len: i64, %i: i64) -> (i64, i64, i1) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %false_bit = arith.constant false
    %i_index = arith.index_cast %i : i64 to index
    %pos0 = arith.addi %start, %i_index : index
    %b0_i8 = memref.load %bytes[%pos0] : memref<?xi8>
    %b0 = arith.extui %b0_i8 : i8 to i64
    %ascii_limit = arith.constant 128 : i64
    %is_ascii = arith.cmpi ult, %b0, %ascii_limit : i64
    %result:3 = scf.if %is_ascii -> (i64, i64, i1) {
      %next = arith.addi %i, %one : i64
      scf.yield %b0, %next, %true_bit : i64, i64, i1
    } else {
      %lead2_lo = arith.constant 194 : i64
      %lead2_hi = arith.constant 223 : i64
      %lead3_lo = arith.constant 224 : i64
      %lead3_hi = arith.constant 239 : i64
      %lead4_lo = arith.constant 240 : i64
      %lead4_hi = arith.constant 244 : i64
      %ge2 = arith.cmpi uge, %b0, %lead2_lo : i64
      %le2 = arith.cmpi ule, %b0, %lead2_hi : i64
      %is2 = arith.andi %ge2, %le2 : i1
      %ge3 = arith.cmpi uge, %b0, %lead3_lo : i64
      %le3 = arith.cmpi ule, %b0, %lead3_hi : i64
      %is3 = arith.andi %ge3, %le3 : i1
      %ge4 = arith.cmpi uge, %b0, %lead4_lo : i64
      %le4 = arith.cmpi ule, %b0, %lead4_hi : i64
      %is4 = arith.andi %ge4, %le4 : i1
      %n34 = arith.select %is4, %four, %zero : i64
      %n3 = arith.select %is3, %three, %n34 : i64
      %n = arith.select %is2, %two, %n3 : i64
      %lead_ok = arith.cmpi ne, %n, %zero : i64
      %end = arith.addi %i, %n : i64
      %enough = arith.cmpi sle, %end, %len : i64
      %head_ok = arith.andi %lead_ok, %enough : i1
      %decoded:3 = scf.if %head_ok -> (i64, i64, i1) {
        %mask2 = arith.constant 31 : i64
        %mask3 = arith.constant 15 : i64
        %mask4 = arith.constant 7 : i64
        %init4 = arith.andi %b0, %mask4 : i64
        %init3_raw = arith.andi %b0, %mask3 : i64
        %init2_raw = arith.andi %b0, %mask2 : i64
        %init34 = arith.select %is3, %init3_raw, %init4 : i64
        %init = arith.select %is2, %init2_raw, %init34 : i64
        %n_index = arith.index_cast %n : i64 to index
        %tail:2 = scf.for %j = %c1 to %n_index step %c1 iter_args(%acc = %init, %ok = %true_bit) -> (i64, i1) {
          %pos = arith.addi %pos0, %j : index
          %cj_i8 = memref.load %bytes[%pos] : memref<?xi8>
          %cj = arith.extui %cj_i8 : i8 to i64
          %cont_mask = arith.constant 192 : i64
          %cont_tag = arith.constant 128 : i64
          %tag = arith.andi %cj, %cont_mask : i64
          %is_cont = arith.cmpi eq, %tag, %cont_tag : i64
          %payload_mask = arith.constant 63 : i64
          %payload = arith.andi %cj, %payload_mask : i64
          %shifted = arith.shli %acc, %six : i64
          %next_acc = arith.ori %shifted, %payload : i64
          %next_ok = arith.andi %ok, %is_cont : i1
          scf.yield %next_acc, %next_ok : i64, i1
        }
        // Range checks per sequence length. 2-byte overlongs are already
        // impossible (lead >= 0xC2).
        %min3 = arith.constant 2048 : i64
        %surrogate_lo = arith.constant 55296 : i64
        %surrogate_hi = arith.constant 57343 : i64
        %min4 = arith.constant 65536 : i64
        %max_cp = arith.constant 1114111 : i64
        %ge_min3 = arith.cmpi uge, %tail#0, %min3 : i64
        %ge_slo = arith.cmpi uge, %tail#0, %surrogate_lo : i64
        %le_shi = arith.cmpi ule, %tail#0, %surrogate_hi : i64
        %is_surrogate = arith.andi %ge_slo, %le_shi : i1
        %not_surrogate = arith.xori %is_surrogate, %true_bit : i1
        %ok3 = arith.andi %ge_min3, %not_surrogate : i1
        %ge_min4 = arith.cmpi uge, %tail#0, %min4 : i64
        %le_max = arith.cmpi ule, %tail#0, %max_cp : i64
        %ok4 = arith.andi %ge_min4, %le_max : i1
        %range34 = arith.select %is4, %ok4, %true_bit : i1
        %range_ok = arith.select %is3, %ok3, %range34 : i1
        %all_ok = arith.andi %tail#1, %range_ok : i1
        scf.yield %tail#0, %end, %all_ok : i64, i64, i1
      } else {
        %stop = arith.addi %i, %one : i64
        scf.yield %zero, %stop, %false_bit : i64, i64, i1
      }
      scf.yield %decoded#0, %decoded#1, %decoded#2 : i64, i64, i1
    }
    func.return %result#0, %result#1, %result#2 : i64, i64, i1
  }

  // str construction from UTF-8 bytes (literals, bytes.decode, host text).
  // Pass 1 validates and finds the widest code point; pass 2 decodes into the
  // smallest fitting width. Invalid UTF-8 raises (CPython raises
  // UnicodeDecodeError; the byte previously landed in the payload unchanged).
  // A str whose whole block -- header, shape, capacity, code units -- was laid
  // out at compile time in read-only data, with the immortal refcount: a
  // literal. Nothing allocates and nothing is ever written; retain and release
  // take their immortal early exit.
  func.func @LyUnicode_FromStatic(%block: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "from_static"} {
    %header_at = arith.constant 0 : index
    %shape_at = arith.constant 16 : index
    %bytes_at = arith.constant 32 : index
    %slot = arith.constant 0 : index
    %header = memref.view %block[%header_at][] {ly.ownership.object_header} : memref<?xi8> to memref<2xi64>
    %shape_view = memref.view %block[%shape_at][] : memref<?xi8> to memref<1xi64>
    %shape = memref.load %shape_view[%slot] : memref<1xi64>
    %three = arith.constant 3 : i64
    %data_bytes = arith.shrui %shape, %three : i64
    %count = arith.index_cast %data_bytes : i64 to index
    %bytes = memref.view %block[%bytes_at][%count] : memref<?xi8> to memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %scan:3 = scf.while (%i = %zero, %count = %zero, %maxcp = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
      %more = arith.cmpi slt, %i, %len : i64
      scf.condition(%more) %i, %count, %maxcp : i64, i64, i64
    } do {
    ^bb0(%i: i64, %count: i64, %maxcp: i64):
      %cp, %next, %ok = func.call @__ly_utf8_step(%bytes, %start, %len, %i) : (memref<?xi8>, index, i64, i64) -> (i64, i64, i1)
      %bad = arith.xori %ok, %true_bit : i1
      scf.if %bad {
        func.call @__ly_unicode_raise_decode_error() : () -> ()
      }
      %bigger = arith.cmpi ugt, %cp, %maxcp : i64
      %new_max = arith.select %bigger, %cp, %maxcp : i64
      %new_count = arith.addi %count, %one : i64
      scf.yield %next, %new_count, %new_max : i64, i64, i64
    }
    %width = func.call @__ly_unicode_width_for(%scan#2) : (i64) -> i64
    %header, %out = func.call @__ly_unicode_alloc(%scan#1, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %count_index = arith.index_cast %scan#1 : i64 to index
    %fill:2 = scf.while (%i = %zero, %k = %c0) : (i64, index) -> (i64, index) {
      %more = arith.cmpi slt, %k, %count_index : index
      scf.condition(%more) %i, %k : i64, index
    } do {
    ^bb0(%i: i64, %k: index):
      %cp, %next, %ok = func.call @__ly_utf8_step(%bytes, %start, %len, %i) : (memref<?xi8>, index, i64, i64) -> (i64, i64, i1)
      func.call @__ly_unicode_put(%out, %width, %k, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      %next_k = arith.addi %k, %c1 : index
      scf.yield %next, %next_k : i64, index
    }
    func.return %header, %out : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_CodepointLength(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__len__"} {
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.return %count : i64
  }

  func.func @LyUnicode_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__bool__"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %zero = arith.constant 0 : index
    %result = arith.cmpi ne, %dim, %zero : index
    func.return %result : i1
  }

  // Canonical form makes equality bytewise: equal strings share width, and
  // the width check also rejects the cross-width byte collisions (for
  // example latin-1 "\x00\x01" vs UCS-2 U+0100).
  func.func @LyUnicode_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__eq__"} {
    %c0 = arith.constant 0 : index
    %lhs_dim = memref.dim %lhs_bytes, %c0 : memref<?xi8>
    %rhs_dim = memref.dim %rhs_bytes, %c0 : memref<?xi8>
    %lhs_width = func.call @__ly_unicode_width(%lhs_header) : (memref<2xi64>) -> i64
    %rhs_width = func.call @__ly_unicode_width(%rhs_header) : (memref<2xi64>) -> i64
    %same_len = arith.cmpi eq, %lhs_dim, %rhs_dim : index
    %same_width = arith.cmpi eq, %lhs_width, %rhs_width : i64
    %comparable = arith.andi %same_len, %same_width : i1
    %result = scf.if %comparable -> (i1) {
      %step = arith.constant 1 : index
      %true = arith.constant true
      %all_equal = scf.for %index = %c0 to %lhs_dim step %step iter_args(%current = %true) -> (i1) {
        %lhs_byte = memref.load %lhs_bytes[%index] : memref<?xi8>
        %rhs_byte = memref.load %rhs_bytes[%index] : memref<?xi8>
        %byte_equal = arith.cmpi eq, %lhs_byte, %rhs_byte : i8
        %next = arith.andi %current, %byte_equal : i1
        scf.yield %next : i1
      }
      scf.yield %all_equal : i1
    } else {
      %false = arith.constant false
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  func.func @LyUnicode_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyUnicode_EqBool(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i1
    %true = arith.constant true
    %false = arith.constant false
    %result = arith.select %eq, %false, %true : i1
    func.return %result : i1
  }

  // Lexicographic code-point comparison (-1/0/1): the first differing code
  // point decides (matching CPython's code-point ordering), else the shorter
  // string orders first.
  func.func private @LyUnicode_Compare(%lhs_header: memref<2xi64>, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64>, %rhs_bytes: memref<?xi8>) -> i64 {
    %lhs_width = func.call @__ly_unicode_width(%lhs_header) : (memref<2xi64>) -> i64
    %rhs_width = func.call @__ly_unicode_width(%rhs_header) : (memref<2xi64>) -> i64
    %lhs_len = func.call @__ly_unicode_count(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @__ly_unicode_count(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %min_len = arith.minsi %lhs_len, %rhs_len : i64
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %plus_one = arith.constant 1 : i64
    %lower = arith.constant 0 : index
    %upper = arith.index_cast %min_len : i64 to index
    %step = arith.constant 1 : index
    %cp_cmp = scf.for %index = %lower to %upper step %step iter_args(%acc = %zero) -> (i64) {
      %lhs_cp = func.call @__ly_unicode_get(%lhs_bytes, %lhs_width, %index) : (memref<?xi8>, i64, index) -> i64
      %rhs_cp = func.call @__ly_unicode_get(%rhs_bytes, %rhs_width, %index) : (memref<?xi8>, i64, index) -> i64
      %lt = arith.cmpi ult, %lhs_cp, %rhs_cp : i64
      %gt = arith.cmpi ugt, %lhs_cp, %rhs_cp : i64
      %gt_val = arith.select %gt, %plus_one, %zero : i64
      %this = arith.select %lt, %minus_one, %gt_val : i64
      %decided = arith.cmpi ne, %acc, %zero : i64
      %next = arith.select %decided, %acc, %this : i64
      scf.yield %next : i64
    }
    %len_lt = arith.cmpi slt, %lhs_len, %rhs_len : i64
    %len_gt = arith.cmpi sgt, %lhs_len, %rhs_len : i64
    %len_gt_val = arith.select %len_gt, %plus_one, %zero : i64
    %len_cmp = arith.select %len_lt, %minus_one, %len_gt_val : i64
    %prefix_equal = arith.cmpi eq, %cp_cmp, %zero : i64
    %result = arith.select %prefix_equal, %len_cmp, %cp_cmp : i64
    func.return %result : i64
  }

  func.func @LyUnicode_LtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__lt__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi slt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_LeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__le__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_GtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__gt__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sgt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_GeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__ge__"} {
    %cmp = func.call @LyUnicode_Compare(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sge, %cmp, %zero : i64
    func.return %result : i1
  }

  // CPython's latin-1 cache (`unicode_latin1`): every one-code-point str
  // below 256 is one of these 256 immortal static objects, 48 bytes apart,
  // laid out as LyUnicode_FromStatic reads one.
  memref.global "private" constant @__ly_str_latin1 : memref<12288xi8> = dense<"0xFFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000000000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000001000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000002000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000003000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000004000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000005000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000006000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000007000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000008000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000009000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000000A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000000B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000000C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000000D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000000E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000000F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000010000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000011000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000012000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000013000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000014000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000015000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000016000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000017000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000018000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000019000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000001A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000001B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000001C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000001D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000001E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000001F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000020000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000021000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000022000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000023000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000024000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000025000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000026000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000027000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000028000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000029000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000002A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000002B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000002C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000002D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000002E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000002F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000030000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000031000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000032000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000033000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000034000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000035000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000036000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000037000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000038000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000039000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000003A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000003B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000003C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000003D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000003E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000003F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000040000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000041000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000042000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000043000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000044000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000045000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000046000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000047000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000048000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000049000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000004A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000004B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000004C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000004D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000004E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000004F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000050000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000051000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000052000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000053000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000054000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000055000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000056000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000057000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000058000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000059000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000005A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000005B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000005C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000005D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000005E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000005F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000060000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000061000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000062000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000063000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000064000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000065000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000066000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000067000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000068000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000069000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000006A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000006B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000006C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000006D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000006E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000006F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000070000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000071000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000072000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000073000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000074000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000075000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000076000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000077000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000078000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000079000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000007A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000007B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000007C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000007D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000007E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000007F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000080000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000081000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000082000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000083000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000084000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000085000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000086000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000087000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000088000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000089000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000008A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000008B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000008C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000008D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000008E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000008F000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000090000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000091000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000092000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000093000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000094000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000095000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000096000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000097000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000098000000000000000000000000000000FFFFFFFFFFFFFF7F04000000000000000900000000000000010000000000000099000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000009A000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000009B000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000009C000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000009D000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000009E000000000000000000000000000000FFFFFFFFFFFFFF7F0400000000000000090000000000000001000000000000009F000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A0000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A1000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A2000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A3000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A4000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A5000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A6000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A7000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A8000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000A9000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000AA000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000AB000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000AC000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000AD000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000AE000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000AF000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B0000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B1000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B2000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B3000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B4000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B5000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B6000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B7000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B8000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000B9000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000BA000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000BB000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000BC000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000BD000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000BE000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000BF000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C0000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C1000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C2000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C3000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C4000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C5000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C6000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C7000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C8000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000C9000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000CA000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000CB000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000CC000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000CD000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000CE000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000CF000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D0000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D1000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D2000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D3000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D4000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D5000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D6000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D7000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D8000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000D9000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000DA000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000DB000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000DC000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000DD000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000DE000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000DF000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E0000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E1000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E2000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E3000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E4000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E5000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E6000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E7000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E8000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000E9000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000EA000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000EB000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000EC000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000ED000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000EE000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000EF000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F0000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F1000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F2000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F3000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F4000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F5000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F6000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F7000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F8000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000F9000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000FA000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000FB000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000FC000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000FD000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000FE000000000000000000000000000000FFFFFFFFFFFFFF7F040000000000000009000000000000000100000000000000FF000000000000000000000000000000"> {alignment = 16 : i64}

  // The str of one code point: the shared one below 256, a fresh one above.
  func.func private @__ly_unicode_single(%cp: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %latin1_limit = arith.constant 256 : i64
    %cached = arith.cmpi ult, %cp, %latin1_limit : i64
    %r:2 = scf.if %cached -> (memref<2xi64>, memref<?xi8>) {
      %table_static = memref.get_global @__ly_str_latin1 : memref<12288xi8>
      %table = memref.cast %table_static : memref<12288xi8> to memref<?xi8>
      %stride = arith.constant 48 : i64
      %offset = arith.muli %cp, %stride : i64
      %offset_index = arith.index_cast %offset : i64 to index
      %entry = memref.view %table[%offset_index][] : memref<?xi8> to memref<48xi8>
      %block = memref.cast %entry : memref<48xi8> to memref<?xi8>
      %h, %b = func.call @LyUnicode_FromStatic(%block) : (memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    } else {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : i64
      %width = func.call @__ly_unicode_width_for(%cp) : (i64) -> i64
      %h, %b = func.call @__ly_unicode_alloc(%one, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      func.call @__ly_unicode_put(%b, %width, %c0, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    }
    func.return %r#0, %r#1 : memref<2xi64>, memref<?xi8>
  }

  // O(1) code-point indexing: fixed-width code units make the scan-free load
  // possible; the result re-canonicalizes to the smallest width for that one
  // code point.
  func.func @LyUnicode_GetItem(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %raw_index: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__getitem__"} {
    %codepoints = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %is_negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %codepoints : i64
    %index = arith.select %is_negative, %from_end, %raw_index : i64
    %lower_ok = arith.cmpi sge, %index, %zero : i64
    %upper_ok = arith.cmpi slt, %index, %codepoints : i64
    %valid = arith.andi %lower_ok, %upper_ok : i1
    %result:2 = scf.if %valid -> (memref<2xi64>, memref<?xi8>) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %index_index = arith.index_cast %index : i64 to index
      %cp = func.call @__ly_unicode_get(%bytes, %width, %index_index) : (memref<?xi8>, i64, index) -> i64
      %result_header, %result_bytes = func.call @__ly_unicode_single(%cp) : (i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
    } else {
      func.call @__ly_unicode_raise_index_error() : () -> ()
      %empty_width = arith.constant 1 : i64
      %result_header, %result_bytes = func.call @__ly_unicode_alloc(%zero, %empty_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // s[i:j:k] -- slice over code point ordinals. The adaptive-width storage
  // gives O(1) indexing; the copy re-canonicalizes the width (pass 1 finds
  // the widest selected code point) so equal strings keep identical
  // representations, and a negative step reverses in iteration order.
  func.func @LyUnicode_GetSlice(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %start_raw: i64, %stop_raw: i64, %step_raw: i64, %mask: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__getslice__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    // The throw does not return; the substitute keeps the IR division-safe.
    %step = arith.select %step_zero, %one, %step_raw : i1, i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %adj:2 = func.call @__ly_slice_adjust(%count, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %take = arith.index_cast %adj#1 : i64 to index

    // Pass 1: canonical width of the selected code points. A width-1 source
    // has none above 255, so the pass has nothing to find and is skipped --
    // and it was the more expensive of the two, calling
    // `__ly_unicode_width_for` per character on top of the read.
    %narrow = arith.cmpi eq, %width, %one : i64
    %max_width = scf.if %narrow -> (i64) {
      scf.yield %one : i64
    } else {
      %scan = scf.for %k = %c0 to %take step %c1 iter_args(%acc = %one) -> (i64) {
        %k64 = arith.index_cast %k : index to i64
        %offset = arith.muli %k64, %step : i64
        %ordinal = arith.addi %adj#0, %offset : i64
        %ordinal_index = arith.index_cast %ordinal : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %ordinal_index) : (memref<?xi8>, i64, index) -> i64
        %cp_width = func.call @__ly_unicode_width_for(%cp) : (i64) -> i64
        %wider = arith.cmpi sgt, %cp_width, %acc : i64
        %next = arith.select %wider, %cp_width, %acc : i1, i64
        scf.yield %next : i64
      }
      scf.yield %scan : i64
    }

    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%adj#1, %max_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    // A step of one is a contiguous run, which is the copy the byte path is
    // for; anything else has to walk the ordinals.
    %step_one = arith.cmpi eq, %step, %one : i64
    scf.if %step_one {
      %from = arith.index_cast %adj#0 : i64 to index
      func.call @__ly_unicode_copy_run(%out_bytes, %max_width, %c0, %bytes, %width, %from, %take) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
    } else {
      scf.for %k = %c0 to %take step %c1 {
        %k64 = arith.index_cast %k : index to i64
        %offset = arith.muli %k64, %step : i64
        %ordinal = arith.addi %adj#0, %offset : i64
        %ordinal_index = arith.index_cast %ordinal : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %ordinal_index) : (memref<?xi8>, i64, index) -> i64
        func.call @__ly_unicode_put(%out_bytes, %max_width, %k, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Copy(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "copy"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_header, %new_bytes = func.call @__ly_unicode_alloc(%count, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    scf.for %i = %c0 to %dim step %c1 {
      %byte = memref.load %bytes[%i] : memref<?xi8>
      memref.store %byte, %new_bytes[%i] : memref<?xi8>
    }
    func.return %new_header, %new_bytes : memref<2xi64>, memref<?xi8>
  }

  // str(s) returns s itself (retained) -- CPython identity semantics.
  func.func @LyUnicode_Str(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Printability query generated beside the UCD tables
  // (runtime/modules/_ucd.mlir).
  func.func private @__ly_ucd_is_printable(%cp: i64) -> i1

  // Repr expansion width in characters for a code point that is neither the
  // chosen quote nor a backslash: 1 = pass through (UCD-printable, matching
  // CPython's unicode_repr), 2 = \t \n \r, 4/6/10 = \xNN / \uNNNN /
  // \U00NNNNNN for non-printable code points by magnitude.
  func.func private @__ly_unicode_repr_class(%cp: i64) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %ten = arith.constant 10 : i64
    %tab = arith.constant 9 : i64
    %nl = arith.constant 10 : i64
    %cr = arith.constant 13 : i64
    %latin1_limit = arith.constant 256 : i64
    %ucs2_limit = arith.constant 65536 : i64
    %is_tab = arith.cmpi eq, %cp, %tab : i64
    %is_nl = arith.cmpi eq, %cp, %nl : i64
    %is_cr = arith.cmpi eq, %cp, %cr : i64
    %tn = arith.ori %is_tab, %is_nl : i1
    %is_tnr = arith.ori %tn, %is_cr : i1
    %printable = func.call @__ly_ucd_is_printable(%cp) : (i64) -> i1
    %fits1 = arith.cmpi ult, %cp, %latin1_limit : i64
    %fits2 = arith.cmpi ult, %cp, %ucs2_limit : i64
    %six_or_ten = arith.select %fits2, %six, %ten : i64
    %escape = arith.select %fits1, %four, %six_or_ten : i64
    %escape_or_pass = arith.select %printable, %one, %escape : i64
    %class = arith.select %is_tnr, %two, %escape_or_pass : i64
    func.return %class : i64
  }

  // %digits lowercase hex digits of %value, most significant first, written
  // at code-point positions [%pos, %pos+%digits).
  func.func private @__ly_unicode_put_hex(%bytes: memref<?xi8>, %width: i64, %pos: index, %value: i64, %digits: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %four = arith.constant 4 : i64
    %fifteen = arith.constant 15 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_a_minus_ten = arith.constant 87 : i64
    scf.for %d = %c0 to %digits step %c1 {
      %last = arith.subi %digits, %c1 : index
      %rev = arith.subi %last, %d : index
      %rev_i64 = arith.index_cast %rev : index to i64
      %shift = arith.muli %rev_i64, %four : i64
      %shifted = arith.shrui %value, %shift : i64
      %nibble = arith.andi %shifted, %fifteen : i64
      %is_decimal = arith.cmpi ult, %nibble, %ten : i64
      %base = arith.select %is_decimal, %ascii_zero, %ascii_a_minus_ten : i64
      %ch = arith.addi %nibble, %base : i64
      %dst = arith.addi %pos, %d : index
      func.call @__ly_unicode_put(%bytes, %width, %dst, %ch) : (memref<?xi8>, i64, index, i64) -> ()
    }
    func.return
  }

  // CPython `str.__repr__`: wrap in quotes and escape. The quote is `'` unless
  // the string contains `'` and no `"` (then `"`), matching unicode_repr.
  // Escapes: \\ , the chosen quote, \t \n \r, and \xNN per
  // __ly_unicode_repr_class. Code-point oriented; the output re-canonicalizes
  // to the smallest width that fits what is actually emitted.
  func.func @LyUnicode_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %zero64 = arith.constant 0 : i64
    %one64 = arith.constant 1 : i64
    %two64 = arith.constant 2 : i64
    %four64 = arith.constant 4 : i64
    %six64 = arith.constant 6 : i64
    %ten64 = arith.constant 10 : i64
    %sq = arith.constant 39 : i64
    %dq = arith.constant 34 : i64
    %bs = arith.constant 92 : i64
    %tab = arith.constant 9 : i64
    %nl = arith.constant 10 : i64
    %t_char = arith.constant 116 : i64
    %n_char = arith.constant 110 : i64
    %r_char = arith.constant 114 : i64
    %x_char = arith.constant 120 : i64
    %u_char = arith.constant 117 : i64
    %cap_u_char = arith.constant 85 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index

    // Pass 1: count `'`/`"`, the escaped body length (quotes counted as 1)
    // and the widest pass-through code point.
    %scan:4 = scf.for %i = %c0 to %count_index step %c1
        iter_args(%csq = %zero64, %cdq = %zero64, %common = %zero64, %maxcp = %zero64)
        -> (i64, i64, i64, i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_sq = arith.cmpi eq, %cp, %sq : i64
      %is_dq = arith.cmpi eq, %cp, %dq : i64
      %is_bs = arith.cmpi eq, %cp, %bs : i64
      %is_quote = arith.ori %is_sq, %is_dq : i1
      %class = func.call @__ly_unicode_repr_class(%cp) : (i64) -> i64
      %quote_or_class = arith.select %is_quote, %one64, %class : i64
      %contrib = arith.select %is_bs, %two64, %quote_or_class : i64
      %passes = arith.cmpi eq, %contrib, %one64 : i64
      %candidate = arith.select %passes, %cp, %zero64 : i64
      %bigger = arith.cmpi ugt, %candidate, %maxcp : i64
      %new_max = arith.select %bigger, %candidate, %maxcp : i64
      %add_sq = arith.select %is_sq, %one64, %zero64 : i64
      %add_dq = arith.select %is_dq, %one64, %zero64 : i64
      %csq2 = arith.addi %csq, %add_sq : i64
      %cdq2 = arith.addi %cdq, %add_dq : i64
      %common2 = arith.addi %common, %contrib : i64
      scf.yield %csq2, %cdq2, %common2, %new_max : i64, i64, i64, i64
    }

    %sq_present = arith.cmpi ne, %scan#0, %zero64 : i64
    %dq_absent = arith.cmpi eq, %scan#1, %zero64 : i64
    %use_double = arith.andi %sq_present, %dq_absent : i1
    %quote = arith.select %use_double, %dq, %sq : i64
    %count_q = arith.select %use_double, %scan#1, %scan#0 : i64
    %body = arith.addi %scan#2, %count_q : i64
    %total = arith.addi %body, %two64 : i64
    %out_width = func.call @__ly_unicode_width_for(%scan#3) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_unicode_put(%out_bytes, %out_width, %c0, %quote) : (memref<?xi8>, i64, index, i64) -> ()

    // Pass 2: fill the escaped body starting at position 1.
    scf.for %i = %c0 to %count_index step %c1 iter_args(%pos = %c1) -> (index) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_bs = arith.cmpi eq, %cp, %bs : i64
      %is_quote = arith.cmpi eq, %cp, %quote : i64
      %two_char = arith.ori %is_bs, %is_quote : i1
      %next_pos = scf.if %two_char -> (index) {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %bs) : (memref<?xi8>, i64, index, i64) -> ()
        %pos1 = arith.addi %pos, %c1 : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %pos1, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        %advanced = arith.addi %pos, %c2 : index
        scf.yield %advanced : index
      } else {
        %class = func.call @__ly_unicode_repr_class(%cp) : (i64) -> i64
        %is_pass = arith.cmpi eq, %class, %one64 : i64
        %inner = scf.if %is_pass -> (index) {
          func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %cp) : (memref<?xi8>, i64, index, i64) -> ()
          %advanced = arith.addi %pos, %c1 : index
          scf.yield %advanced : index
        } else {
          func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %bs) : (memref<?xi8>, i64, index, i64) -> ()
          %pos1 = arith.addi %pos, %c1 : index
          %is_two = arith.cmpi eq, %class, %two64 : i64
          %escaped = scf.if %is_two -> (index) {
            %is_tab = arith.cmpi eq, %cp, %tab : i64
            %is_nl = arith.cmpi eq, %cp, %nl : i64
            %nr_char = arith.select %is_nl, %n_char, %r_char : i64
            %second = arith.select %is_tab, %t_char, %nr_char : i64
            func.call @__ly_unicode_put(%out_bytes, %out_width, %pos1, %second) : (memref<?xi8>, i64, index, i64) -> ()
            %advanced = arith.addi %pos, %c2 : index
            scf.yield %advanced : index
          } else {
            %is_four = arith.cmpi eq, %class, %four64 : i64
            %is_six = arith.cmpi eq, %class, %six64 : i64
            %ubig_marker = arith.select %is_six, %u_char, %cap_u_char : i64
            %marker = arith.select %is_four, %x_char, %ubig_marker : i64
            func.call @__ly_unicode_put(%out_bytes, %out_width, %pos1, %marker) : (memref<?xi8>, i64, index, i64) -> ()
            %digits8 = arith.select %is_six, %c4, %c8 : index
            %digits = arith.select %is_four, %c2, %digits8 : index
            %pos2 = arith.addi %pos, %c2 : index
            func.call @__ly_unicode_put_hex(%out_bytes, %out_width, %pos2, %cp, %digits) : (memref<?xi8>, i64, index, i64, index) -> ()
            %class_index = arith.index_cast %class : i64 to index
            %advanced = arith.addi %pos, %class_index : index
            scf.yield %advanced : index
          }
          scf.yield %escaped : index
        }
        scf.yield %inner : index
      }
      scf.yield %next_pos : index
    }

    %total_idx = arith.index_cast %total : i64 to index
    %last = arith.subi %total_idx, %c1 : index
    func.call @__ly_unicode_put(%out_bytes, %out_width, %last, %quote) : (memref<?xi8>, i64, index, i64) -> ()
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // ⭐ A SAME-WIDTH COPY IS BYTES, NOT CODEPOINTS. `__ly_unicode_get` and
  // `__ly_unicode_put` each dispatch on the width and reassemble the value a
  // byte at a time, so copying one string into another cost two calls and a
  // branch per character where CPython, when the two kinds agree, calls
  // memcpy. The widths agree in every case that matters: an ASCII string
  // concatenated with an ASCII string, sliced, joined or repeated.
  //
  // ⛔ IT DOES NOT BECOME A `memcpy`, WHICH THIS SAID IT WOULD. The claim was
  // that the loop is the shape loop-idiom recognition rewrites; disassembling
  // the O2 output says otherwise -- the two memrefs may alias for all LLVM
  // knows, so what comes out is a VECTORIZED copy behind a runtime overlap
  // check, not a call. That is fast enough to be beside the point: 400,000
  // concatenations of a thousand bytes take 0.02 s against CPython's 0.03 s.
  // The note is here because the wrong reason for a right number is how the
  // next change to this loop gets argued.
  //
  // Why NOT call memcpy directly: the manifest would have to declare it and
  // hand it raw addresses, and the pipeline rejects a descriptor built inline
  // (Passes/Runtime/Passes/Lowering.cpp). A byte loop over the two memrefs
  // says the same thing in the vocabulary this layer has.
  func.func private @__ly_unicode_copy_bytes(%dst: memref<?xi8>, %dst_off: index, %src: memref<?xi8>, %src_off: index, %count: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %b = %c0 to %count step %c1 {
      %s = arith.addi %src_off, %b : index
      %d = arith.addi %dst_off, %b : index
      %v = memref.load %src[%s] : memref<?xi8>
      memref.store %v, %dst[%d] : memref<?xi8>
    }
    func.return
  }

  // Copy %len codepoints from %src[%src_at] to %dst[%dst_at], taking the byte
  // path when the two widths agree and the widening one when they do not.
  func.func private @__ly_unicode_copy_run(%dst: memref<?xi8>, %dst_width: i64, %dst_at: index, %src: memref<?xi8>, %src_width: i64, %src_at: index, %len: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %same = arith.cmpi eq, %dst_width, %src_width : i64
    scf.if %same {
      %w = arith.index_cast %src_width : i64 to index
      %bytes = arith.muli %len, %w : index
      %d = arith.muli %dst_at, %w : index
      %s = arith.muli %src_at, %w : index
      func.call @__ly_unicode_copy_bytes(%dst, %d, %src, %s, %bytes) : (memref<?xi8>, index, memref<?xi8>, index, index) -> ()
    } else {
      scf.for %index = %c0 to %len step %c1 {
        %si = arith.addi %src_at, %index : index
        %di = arith.addi %dst_at, %index : index
        %cp = func.call @__ly_unicode_get(%src, %src_width, %si) : (memref<?xi8>, i64, index) -> i64
        func.call @__ly_unicode_put(%dst, %dst_width, %di, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
    }
    func.return
  }

  // ⭐ CPython's `unicode_concatenate` (ceval.c): when the left operand holds
  // the ONLY reference, `s += x` appends into the block it already has instead
  // of building a new string, which is what makes an accumulating append linear
  // rather than quadratic. Measured at 160,000 appends of ten bytes, the
  // allocating path is 2.20 s.
  //
  // ⛔ THE UNIQUENESS TEST IS THE REFCOUNT AND THE CALL SITE TOGETHER, and
  // neither alone is enough. A refcount of one does NOT mean unique here: a
  // borrowed parameter is passed without a bump, so `def f(p): p += x` would
  // see one and rewrite the CALLER's string. The lowering therefore only routes
  // to this function when the receiver bundle is OWNED by the frame -- the
  // frame's own reference is the one being counted -- and hands a borrowed one
  // to `__add__` instead. The refcount then answers the other half: `t = s;
  // s += x` leaves t's reference behind, the count is two, and the append
  // allocates so t keeps what it had.
  //
  // ⛔ AND THE RECEIVER IS BORROWED, not transferred. Transferring is what this
  // means, but saying it that way makes the frame's rebinding a release of a
  // consumed token -- the affine verifier refuses `s += x` in a loop and on a
  // parameter for that reason, in the two shapes that matter most. Borrowing
  // and retaining the result instead leaves the call the same shape as
  // `__add__`, which every one of those places already accepts.
  //
  // ⛔ AND NOT `memref.realloc`. Growing in place is the whole point, and a
  // realloc moves the block -- the memory-safety argument (BoxLayout.h) rests
  // on no allocation moving under a held word, so the room has to be there
  // already. That is what the capacity word is for.
  func.func @LyUnicode_IAdd(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__ly_iadd__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i64
    %refcount_slot = arith.constant 0 : index
    %refcount = memref.load %lhs_header[%refcount_slot] : memref<2xi64>
    %unique = arith.cmpi eq, %refcount, %one : i64
    %lhs_width = func.call @__ly_unicode_width(%lhs_header) : (memref<2xi64>) -> i64
    %rhs_width = func.call @__ly_unicode_width(%rhs_header) : (memref<2xi64>) -> i64
    %same_kind = arith.cmpi sle, %rhs_width, %lhs_width : i64
    %lhs_len = func.call @__ly_unicode_count(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @__ly_unicode_count(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %total_len = arith.addi %lhs_len, %rhs_len : i64
    %lhs_ptr_index = memref.extract_aligned_pointer_as_index %lhs_header : memref<2xi64> -> index
    %lhs_ptr = arith.index_cast %lhs_ptr_index : index to i64
    %capacity = func.call @__ly_unicode_raw_capacity(%lhs_ptr) : (i64) -> i64
    %needed = arith.muli %total_len, %lhs_width : i64
    %has_room = arith.cmpi sle, %needed, %capacity : i64
    %rhs_ptr_index = memref.extract_aligned_pointer_as_index %rhs_header : memref<2xi64> -> index
    %rhs_ptr = arith.index_cast %rhs_ptr_index : index to i64
    // `s += s` reads the block it is writing into. The two runs do not overlap
    // -- the source ends where the destination begins -- but the source's own
    // length word moves under the copy, so the self case takes the other path.
    %distinct = arith.cmpi ne, %lhs_ptr, %rhs_ptr : i64
    %kind_ok = arith.andi %unique, %same_kind : i1
    %usable = arith.andi %kind_ok, %distinct : i1
    %in_place = arith.andi %usable, %has_room : i1
    %result:2 = scf.if %in_place -> (memref<2xi64>, memref<?xi8>) {
      %data_offset = func.call @__ly_unicode_data_offset() : () -> i64
      %data_ptr = arith.addi %lhs_ptr, %data_offset : i64
      %grown = func.call @__ly_global_view_i8(%data_ptr, %needed) : (i64, i64) -> memref<?xi8>
      %lhs_at = arith.index_cast %lhs_len : i64 to index
      %rhs_upper = arith.index_cast %rhs_len : i64 to index
      func.call @__ly_unicode_copy_run(%grown, %lhs_width, %lhs_at, %rhs_bytes, %rhs_width, %c0, %rhs_upper) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
      func.call @__ly_unicode_set_bytes(%lhs_ptr, %needed) : (i64, i64) -> ()
      // The result is a reference of its own: the caller releases the binding
      // this grew out of, and that binding and the result are the same object.
      %retained = memref.cast %lhs_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
      func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
      scf.yield %lhs_header, %grown : memref<2xi64>, memref<?xi8>
    } else {
      // ⭐ THE ONLY PLACE THAT ASKS FOR SLACK, and it asks the way CPython's
      // list_resize does: an eighth over, plus a floor. A string that is never
      // appended to is allocated exactly, so the capacity word costs it 8 bytes
      // and no more; one that IS appended to stops paying per append after the
      // first move.
      %width = arith.maxsi %lhs_width, %rhs_width : i64
      %exact = arith.muli %total_len, %width : i64
      %eighth_shift = arith.constant 3 : i64
      %eighth = arith.shrui %exact, %eighth_shift : i64
      %floor = arith.constant 32 : i64
      %slack = arith.addi %exact, %eighth : i64
      %room = arith.addi %slack, %floor : i64
      %header, %bytes = func.call @__ly_unicode_alloc_capacity(%total_len, %width, %room) : (i64, i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %lhs_upper = arith.index_cast %lhs_len : i64 to index
      %rhs_upper = arith.index_cast %rhs_len : i64 to index
      func.call @__ly_unicode_copy_run(%bytes, %width, %c0, %lhs_bytes, %lhs_width, %c0, %lhs_upper) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
      func.call @__ly_unicode_copy_run(%bytes, %width, %lhs_upper, %rhs_bytes, %rhs_width, %c0, %rhs_upper) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_width = func.call @__ly_unicode_width(%lhs_header) : (memref<2xi64>) -> i64
    %rhs_width = func.call @__ly_unicode_width(%rhs_header) : (memref<2xi64>) -> i64
    %lhs_len = func.call @__ly_unicode_count(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @__ly_unicode_count(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    // Operands are canonical (smallest fitting width), so the wider operand
    // width IS the smallest width that fits the concatenation.
    %width = arith.maxsi %lhs_width, %rhs_width : i64
    %total_len = arith.addi %lhs_len, %rhs_len : i64
    %header, %bytes = func.call @__ly_unicode_alloc(%total_len, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %lhs_upper = arith.index_cast %lhs_len : i64 to index
    %rhs_upper = arith.index_cast %rhs_len : i64 to index
    func.call @__ly_unicode_copy_run(%bytes, %width, %c0, %lhs_bytes, %lhs_width, %c0, %lhs_upper) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
    func.call @__ly_unicode_copy_run(%bytes, %width, %lhs_upper, %rhs_bytes, %rhs_width, %c0, %rhs_upper) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // ===== str search / comparison methods =====

  // 7-field UCD ctype accessor (defined in the generated _ucd.mlir).
  func.func private @__ly_ucd_ctype(%cp: i64) -> (i64, i64, i64, i64, i64, i64, i64)

  func.func private @__ly_unicode_cp_is_space(%cp: i64) -> i1 {
    %zero = arith.constant 0 : i64
    %space_bit = arith.constant 16 : i64
    %u, %l, %f, %t, %dec, %dig, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64, i64)
    %masked = arith.andi %flags, %space_bit : i64
    %is_space = arith.cmpi ne, %masked, %zero : i64
    func.return %is_space : i1
  }

  // "substring not found"
  memref.global "private" constant @__ly_unicode_msg_substring_not_found : memref<19xi8> = dense<[115, 117, 98, 115, 116, 114, 105, 110, 103, 32, 110, 111, 116, 32, 102, 111, 117, 110, 100]>
  // "The fill character must be exactly one character long"
  memref.global "private" constant @__ly_unicode_msg_bad_fill : memref<53xi8> = dense<[84, 104, 101, 32, 102, 105, 108, 108, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 32, 109, 117, 115, 116, 32, 98, 101, 32, 101, 120, 97, 99, 116, 108, 121, 32, 111, 110, 101, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 32, 108, 111, 110, 103]>

  // CPython ADJUST_INDICES: negative indices count from the end (clamped to
  // 0), the end clamps to the length, and the start deliberately does NOT
  // clamp downward to the length -- find("", past-the-end) must miss.
  func.func private @__ly_unicode_adjust_range(%len: i64, %start_raw: i64, %end_raw: i64) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %start_neg = arith.cmpi slt, %start_raw, %zero : i64
    %start_shift = arith.addi %start_raw, %len : i64
    %start_from_end = arith.maxsi %start_shift, %zero : i64
    %start = arith.select %start_neg, %start_from_end, %start_raw : i64
    %end_over = arith.cmpi sgt, %end_raw, %len : i64
    %end_neg = arith.cmpi slt, %end_raw, %zero : i64
    %end_shift = arith.addi %end_raw, %len : i64
    %end_from_end = arith.maxsi %end_shift, %zero : i64
    %end_in = arith.select %end_neg, %end_from_end, %end_raw : i64
    %end = arith.select %end_over, %len, %end_in : i64
    func.return %start, %end : i64, i64
  }

  // s[si .. si+n) == t[ti .. ti+n), by code point.
  func.func private @__ly_unicode_match_at(%s_bytes: memref<?xi8>, %s_width: i64, %si: index, %t_bytes: memref<?xi8>, %t_width: i64, %ti: index, %n: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %all = scf.for %j = %c0 to %n step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %sj = arith.addi %si, %j : index
      %tj = arith.addi %ti, %j : index
      %scp = func.call @__ly_unicode_get(%s_bytes, %s_width, %sj) : (memref<?xi8>, i64, index) -> i64
      %tcp = func.call @__ly_unicode_get(%t_bytes, %t_width, %tj) : (memref<?xi8>, i64, index) -> i64
      %eq = arith.cmpi eq, %scp, %tcp : i64
      %next = arith.andi %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %all : i1
  }

  // First (or, %reverse, last) index in [start, end-n] where the needle
  // matches; -1 when the window is empty or nothing matches. Indices are
  // pre-adjusted code-point positions.
  // ⭐ CPython's STRINGLIB_BLOOM (Objects/stringlib/fastsearch.h): a 64-bit set
  // keyed by the low six bits of the code point. It only ever answers "this
  // character is CERTAINLY NOT in the needle", and that one-sided answer is
  // what licenses skipping the whole needle length instead of one position.
  func.func private @__ly_unicode_bloom_add(%mask: i64, %cp: i64) -> i64 {
    %sixty_three = arith.constant 63 : i64
    %one = arith.constant 1 : i64
    %bit = arith.andi %cp, %sixty_three : i64
    %set = arith.shli %one, %bit : i64
    %next = arith.ori %mask, %set : i64
    func.return %next : i64
  }

  func.func private @__ly_unicode_bloom_has(%mask: i64, %cp: i64) -> i1 {
    %sixty_three = arith.constant 63 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %bit = arith.andi %cp, %sixty_three : i64
    %set = arith.shli %one, %bit : i64
    %hit = arith.andi %mask, %set : i64
    %present = arith.cmpi ne, %hit, %zero : i64
    func.return %present : i1
  }

  // `default_find`: check the needle's LAST character first, and on a miss ask
  // the bloom set about the character just past the window. Absent means no
  // alignment overlapping it can match, so the position advances by the whole
  // needle; present means it advances by the gap to the previous occurrence of
  // the last character.
  //
  // ⛔ Where CPython reads one past the window unconditionally -- its buffers
  // carry a NUL sentinel -- this checks the bound first. Ours do not, and the
  // conservative answer when there is no next character (advance by one) is
  // the one CPython would give for a needle that contains NUL anyway.
  //
  // ⛔ AND THE TWO-WAY ALGORITHM IS DELIBERATELY NOT HERE. CPython keeps this
  // same filtered scan for everything below `n < 2500 || (m < 100 && n < 30000)
  // || m < 6` and only then switches to Crochemore-Perrin, whose critical
  // factorization and memory table are what bound the periodic worst case.
  // Measured against CPython 3.14.5 inside the regime where IT switches -- a
  // 50,000-character haystack with a 21-character needle that never matches,
  // 300 times: 13.8 ms here against 10.4 there; and on the periodic shape
  // two-way exists to bound (`"aaaaaaaaab" * 5000` searched for
  // `"a" * 19 + "b"`) 11.4 ms here against 12.9 there. A filtered scan is
  // within a third of it on the first and ahead on the second, which is not
  // what four hundred lines of factorization buys back.
  // ⭐ THE SAME WALK WITH BYTE LOADS, for the case where both strings are
  // latin-1. `__ly_unicode_get` is a call and a width branch, the walk does two
  // of them per position and `__ly_unicode_match_at` another two per candidate,
  // and the widths do not change inside any of those loops. Profiling
  // `s.replace("the", "THE!")` put 82% of the time in this function.
  //
  // Why NOT a width parameter and one body: the branch is what costs, so the
  // body has to be picked before the loop rather than inside it.
  func.func private @__ly_unicode_find_fwd_bytes(%s_bytes: memref<?xi8>, %t_bytes: memref<?xi8>, %start: i64, %end: i64, %n: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %mlast = arith.subi %n, %one : i64
    %mlast_index = arith.index_cast %mlast : i64 to index
    %last = memref.load %t_bytes[%mlast_index] : memref<?xi8>
    %last_cp = arith.extui %last : i8 to i64
    %pre:2 = scf.for %k = %c0 to %mlast_index step %c1 iter_args(%mask = %zero, %gap = %mlast) -> (i64, i64) {
      %b = memref.load %t_bytes[%k] : memref<?xi8>
      %cp = arith.extui %b : i8 to i64
      %next_mask = func.call @__ly_unicode_bloom_add(%mask, %cp) : (i64, i64) -> i64
      %same = arith.cmpi eq, %b, %last : i8
      %k64 = arith.index_cast %k : index to i64
      %behind = arith.subi %mlast, %k64 : i64
      %candidate = arith.subi %behind, %one : i64
      %next_gap = arith.select %same, %candidate, %gap : i64
      scf.yield %next_mask, %next_gap : i64, i64
    }
    %mask = func.call @__ly_unicode_bloom_add(%pre#0, %last_cp) : (i64, i64) -> i64
    %gap_step = arith.addi %pre#1, %one : i64
    %long_step = arith.addi %n, %one : i64
    %n_index = arith.index_cast %n : i64 to index
    %w = arith.subi %end, %n : i64
    // ⛔ THE BOUNDS TEST ON THE LOOKAHEAD IS HOISTED OUT, because it can only
    // fail at one position. The character the bloom filter asks about is
    // `s[i + n]`, which is inside the haystack for every `i < w` and past its
    // end at exactly `i == w` -- so the walk runs to `w` without the test and
    // the last alignment is checked after it. CPython does not have this branch
    // at all: its buffers carry a NUL sentinel and it reads one past the window
    // unconditionally.
    %walk:2 = scf.while (%i = %start, %ans = %minus_one) : (i64, i64) -> (i64, i64) {
      %in_range = arith.cmpi slt, %i, %w : i64
      %not_yet = arith.cmpi eq, %ans, %minus_one : i64
      %go = arith.andi %in_range, %not_yet : i1
      scf.condition(%go) %i, %ans : i64, i64
    } do {
    ^bb0(%i: i64, %ans: i64):
      %at = arith.addi %i, %mlast : i64
      %at_index = arith.index_cast %at : i64 to index
      %b = memref.load %s_bytes[%at_index] : memref<?xi8>
      %is_last = arith.cmpi eq, %b, %last : i8
      %nextpos = arith.addi %at, %one : i64
      %next_index = arith.index_cast %nextpos : i64 to index
      %nb = memref.load %s_bytes[%next_index] : memref<?xi8>
      %ncp = arith.extui %nb : i8 to i64
      %present = func.call @__ly_unicode_bloom_has(%mask, %ncp) : (i64, i64) -> i1
      %skippable = arith.xori %present, %true : i1
      %step:2 = scf.if %is_last -> (i64, i64) {
        %i_index = arith.index_cast %i : i64 to index
        %hit = scf.for %j = %c0 to %n_index step %c1 iter_args(%acc = %true) -> (i1) {
          %sj = arith.addi %i_index, %j : index
          %sb = memref.load %s_bytes[%sj] : memref<?xi8>
          %tb = memref.load %t_bytes[%j] : memref<?xi8>
          %eq = arith.cmpi eq, %sb, %tb : i8
          %next = arith.andi %acc, %eq : i1
          scf.yield %next : i1
        }
        %after:2 = scf.if %hit -> (i64, i64) {
          scf.yield %i, %i : i64, i64
        } else {
          %chosen = arith.select %skippable, %long_step, %gap_step : i64
          %ni = arith.addi %i, %chosen : i64
          scf.yield %ni, %ans : i64, i64
        }
        scf.yield %after#0, %after#1 : i64, i64
      } else {
        %chosen = arith.select %skippable, %long_step, %one : i64
        %ni = arith.addi %i, %chosen : i64
        scf.yield %ni, %ans : i64, i64
      }
      scf.yield %step#0, %step#1 : i64, i64
    }
    // The last alignment, whose lookahead would be past the end. Nothing skips
    // from here, so only the compare is left.
    %missed = arith.cmpi eq, %walk#1, %minus_one : i64
    %reached = arith.cmpi sle, %walk#0, %w : i64
    %check_last = arith.andi %missed, %reached : i1
    %answer = scf.if %check_last -> (i64) {
      %tail_index = arith.index_cast %w : i64 to index
      %hit = scf.for %j = %c0 to %n_index step %c1 iter_args(%acc = %true) -> (i1) {
        %sj = arith.addi %tail_index, %j : index
        %sb = memref.load %s_bytes[%sj] : memref<?xi8>
        %tb = memref.load %t_bytes[%j] : memref<?xi8>
        %eq = arith.cmpi eq, %sb, %tb : i8
        %next = arith.andi %acc, %eq : i1
        scf.yield %next : i1
      }
      %found = arith.select %hit, %w, %minus_one : i64
      scf.yield %found : i64
    } else {
      scf.yield %walk#1 : i64
    }
    func.return %answer : i64
  }

  func.func private @__ly_unicode_find_fwd(%s_bytes: memref<?xi8>, %s_width: i64, %t_bytes: memref<?xi8>, %t_width: i64, %start: i64, %end: i64, %n: i64) -> i64 {
    %one_w = arith.constant 1 : i64
    %s_narrow = arith.cmpi eq, %s_width, %one_w : i64
    %t_narrow = arith.cmpi eq, %t_width, %one_w : i64
    %both_narrow = arith.andi %s_narrow, %t_narrow : i1
    %picked = scf.if %both_narrow -> (i64) {
      %r = func.call @__ly_unicode_find_fwd_bytes(%s_bytes, %t_bytes, %start, %end, %n) : (memref<?xi8>, memref<?xi8>, i64, i64, i64) -> i64
      scf.yield %r : i64
    } else {
      %r = func.call @__ly_unicode_find_fwd_wide(%s_bytes, %s_width, %t_bytes, %t_width, %start, %end, %n) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64) -> i64
      scf.yield %r : i64
    }
    func.return %picked : i64
  }

  func.func private @__ly_unicode_find_fwd_wide(%s_bytes: memref<?xi8>, %s_width: i64, %t_bytes: memref<?xi8>, %t_width: i64, %start: i64, %end: i64, %n: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %mlast = arith.subi %n, %one : i64
    %mlast_index = arith.index_cast %mlast : i64 to index
    %last = func.call @__ly_unicode_get(%t_bytes, %t_width, %mlast_index) : (memref<?xi8>, i64, index) -> i64
    %pre:2 = scf.for %k = %c0 to %mlast_index step %c1 iter_args(%mask = %zero, %gap = %mlast) -> (i64, i64) {
      %cp = func.call @__ly_unicode_get(%t_bytes, %t_width, %k) : (memref<?xi8>, i64, index) -> i64
      %next_mask = func.call @__ly_unicode_bloom_add(%mask, %cp) : (i64, i64) -> i64
      %same = arith.cmpi eq, %cp, %last : i64
      %k64 = arith.index_cast %k : index to i64
      %behind = arith.subi %mlast, %k64 : i64
      %candidate = arith.subi %behind, %one : i64
      %next_gap = arith.select %same, %candidate, %gap : i64
      scf.yield %next_mask, %next_gap : i64, i64
    }
    %mask = func.call @__ly_unicode_bloom_add(%pre#0, %last) : (i64, i64) -> i64
    %gap_step = arith.addi %pre#1, %one : i64
    %long_step = arith.addi %n, %one : i64
    %n_index = arith.index_cast %n : i64 to index
    %w = arith.subi %end, %n : i64
    %walk:2 = scf.while (%i = %start, %ans = %minus_one) : (i64, i64) -> (i64, i64) {
      %in_range = arith.cmpi sle, %i, %w : i64
      %not_yet = arith.cmpi eq, %ans, %minus_one : i64
      %go = arith.andi %in_range, %not_yet : i1
      scf.condition(%go) %i, %ans : i64, i64
    } do {
    ^bb0(%i: i64, %ans: i64):
      %at = arith.addi %i, %mlast : i64
      %at_index = arith.index_cast %at : i64 to index
      %cp = func.call @__ly_unicode_get(%s_bytes, %s_width, %at_index) : (memref<?xi8>, i64, index) -> i64
      %is_last = arith.cmpi eq, %cp, %last : i64
      %nextpos = arith.addi %at, %one : i64
      %has_next = arith.cmpi slt, %nextpos, %end : i64
      %skippable = scf.if %has_next -> (i1) {
        %next_index = arith.index_cast %nextpos : i64 to index
        %ncp = func.call @__ly_unicode_get(%s_bytes, %s_width, %next_index) : (memref<?xi8>, i64, index) -> i64
        %present = func.call @__ly_unicode_bloom_has(%mask, %ncp) : (i64, i64) -> i1
        %absent = arith.xori %present, %true : i1
        scf.yield %absent : i1
      } else {
        scf.yield %false : i1
      }
      %step:2 = scf.if %is_last -> (i64, i64) {
        %i_index = arith.index_cast %i : i64 to index
        %hit = func.call @__ly_unicode_match_at(%s_bytes, %s_width, %i_index, %t_bytes, %t_width, %c0, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        %after:2 = scf.if %hit -> (i64, i64) {
          scf.yield %i, %i : i64, i64
        } else {
          %chosen = arith.select %skippable, %long_step, %gap_step : i64
          %ni = arith.addi %i, %chosen : i64
          scf.yield %ni, %ans : i64, i64
        }
        scf.yield %after#0, %after#1 : i64, i64
      } else {
        %chosen = arith.select %skippable, %long_step, %one : i64
        %ni = arith.addi %i, %chosen : i64
        scf.yield %ni, %ans : i64, i64
      }
      scf.yield %step#0, %step#1 : i64, i64
    }
    func.return %walk#1 : i64
  }

  // `default_rfind`: the same thing from the other end, keyed on the needle's
  // FIRST character and asking the bloom set about the character just before
  // the window.
  func.func private @__ly_unicode_find_rev(%s_bytes: memref<?xi8>, %s_width: i64, %t_bytes: memref<?xi8>, %t_width: i64, %start: i64, %end: i64, %n: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %mlast = arith.subi %n, %one : i64
    %n_index = arith.index_cast %n : i64 to index
    %first = func.call @__ly_unicode_get(%t_bytes, %t_width, %c0) : (memref<?xi8>, i64, index) -> i64
    %mask0 = func.call @__ly_unicode_bloom_add(%zero, %first) : (i64, i64) -> i64
    // CPython walks the needle from the back and keeps assigning, so the skip
    // it ends with is the one for the SMALLEST index whose character equals the
    // first. Walking forward and keeping the first assignment is the same
    // answer; `skip` still holding `mlast` is what says none was made, because
    // every assignment is at most mlast - 1.
    %pre:2 = scf.for %k = %c1 to %n_index step %c1 iter_args(%mask = %mask0, %skip = %mlast) -> (i64, i64) {
      %cp = func.call @__ly_unicode_get(%t_bytes, %t_width, %k) : (memref<?xi8>, i64, index) -> i64
      %next_mask = func.call @__ly_unicode_bloom_add(%mask, %cp) : (i64, i64) -> i64
      %same = arith.cmpi eq, %cp, %first : i64
      %unset = arith.cmpi eq, %skip, %mlast : i64
      %take = arith.andi %same, %unset : i1
      %k64 = arith.index_cast %k : index to i64
      %candidate = arith.subi %k64, %one : i64
      %next_skip = arith.select %take, %candidate, %skip : i64
      scf.yield %next_mask, %next_skip : i64, i64
    }
    %skip_step = arith.addi %pre#1, %one : i64
    %long_step = arith.addi %n, %one : i64
    %w = arith.subi %end, %n : i64
    %walk:2 = scf.while (%i = %w, %ans = %minus_one) : (i64, i64) -> (i64, i64) {
      %in_range = arith.cmpi sge, %i, %start : i64
      %not_yet = arith.cmpi eq, %ans, %minus_one : i64
      %go = arith.andi %in_range, %not_yet : i1
      scf.condition(%go) %i, %ans : i64, i64
    } do {
    ^bb0(%i: i64, %ans: i64):
      %i_index = arith.index_cast %i : i64 to index
      %cp = func.call @__ly_unicode_get(%s_bytes, %s_width, %i_index) : (memref<?xi8>, i64, index) -> i64
      %is_first = arith.cmpi eq, %cp, %first : i64
      %prevpos = arith.subi %i, %one : i64
      %has_prev = arith.cmpi sge, %prevpos, %start : i64
      %skippable = scf.if %has_prev -> (i1) {
        %prev_index = arith.index_cast %prevpos : i64 to index
        %pcp = func.call @__ly_unicode_get(%s_bytes, %s_width, %prev_index) : (memref<?xi8>, i64, index) -> i64
        %present = func.call @__ly_unicode_bloom_has(%pre#0, %pcp) : (i64, i64) -> i1
        %absent = arith.xori %present, %true : i1
        scf.yield %absent : i1
      } else {
        scf.yield %false : i1
      }
      %step:2 = scf.if %is_first -> (i64, i64) {
        %hit = func.call @__ly_unicode_match_at(%s_bytes, %s_width, %i_index, %t_bytes, %t_width, %c0, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        %after:2 = scf.if %hit -> (i64, i64) {
          scf.yield %i, %i : i64, i64
        } else {
          %chosen = arith.select %skippable, %long_step, %skip_step : i64
          %ni = arith.subi %i, %chosen : i64
          scf.yield %ni, %ans : i64, i64
        }
        scf.yield %after#0, %after#1 : i64, i64
      } else {
        %chosen = arith.select %skippable, %long_step, %one : i64
        %ni = arith.subi %i, %chosen : i64
        scf.yield %ni, %ans : i64, i64
      }
      scf.yield %step#0, %step#1 : i64, i64
    }
    func.return %walk#1 : i64
  }

  func.func private @__ly_unicode_find_core(%s_bytes: memref<?xi8>, %s_width: i64, %t_bytes: memref<?xi8>, %t_width: i64, %start: i64, %end: i64, %n: i64, %reverse: i1) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    // ⭐ A ONE-CHARACTER NEEDLE IS TESTED FIRST AND SCANS PLAINLY, which is
    // what CPython's FASTSEARCH does before anything else. The bloom filter
    // costs a second read per position -- the character past the window -- and
    // for a single character the candidate read IS the whole comparison, so
    // the filter doubles the work rather than saving it.
    //
    // ⛔ Written out here rather than called, and tested before the window
    // check rather than inside it, because what pays for this is `split`: it
    // restarts the search at every separator, so splitting a 100-piece line is
    // 200 finds that each scan one or two positions and the PER-CALL cost is
    // the whole cost. Measured on `"a,b,...".split(",")` x 100,000: the plain
    // scan with no filter at all is 118 ms, this shape is 131, and folding the
    // case into the bloom loop as a branch instead of dispatching is 140. So
    // the filter costs `split` 10% for the 2.5x it gives `find`.
    //
    // ⛔ AND MAKING `split` SINGLE-PASS DOES NOT PAY THAT BACK -- measured, not
    // reasoned. It walks the string twice, once to count the pieces so the
    // list can be allocated at its final length and once to cut them, so the
    // obvious move is to append to a growing list and search once. Built that
    // way it is 180.3 ms against the two-pass 128.1, and 152.7 even with the
    // list pre-grown so no reallocation happens at all. Removing an ENTIRE
    // search pass loses to the per-piece list bookkeeping it costs -- a
    // capacity load, a length store and an items view per piece -- because
    // each of those searches scans one or two positions while each piece
    // allocates a string. `split`'s remaining 2x against CPython is in
    // `__ly_unicode_slice` and the allocation under it, not in the searching.
    %single = arith.cmpi eq, %n, %one : i64
    %answer = scf.if %single -> (i64) {
      %target = func.call @__ly_unicode_get(%t_bytes, %t_width, %c0) : (memref<?xi8>, i64, index) -> i64
      %last_pos = arith.subi %end, %one : i64
      %from = arith.select %reverse, %last_pos, %start : i64
      // ⭐ A LATIN-1 HAYSTACK IS SCANNED AS BYTES, which is the shape CPython
      // reaches memchr for. `__ly_unicode_get` is a call and a width branch per
      // position, and the width does not change inside the loop; taking it out
      // leaves a byte compare. Every ASCII string is here, and so is every
      // caller that searches one -- find, split, partition, replace, count,
      // index.
      //
      // ⛔ AND CALLING `memchr` ITSELF IS SLOWER, measured rather than reasoned.
      // The loop cannot vectorize -- it exits when it finds the byte -- so a
      // libc call that does looks like the answer. It is not: the callers that
      // matter search SHORT spans. `s.replace("o", "0")` over 880 characters is
      // sixty searches of about fourteen bytes each, and a call per search cost
      // more than the scan it replaced: 0.13 s -> 0.19 s, and `find` plus
      // `split` over the same string 0.59 s -> 0.80 s.
      %byte_wide = arith.cmpi eq, %s_width, %one : i64
      %found = scf.if %byte_wide -> (i64) {
        %byte_max = arith.constant 255 : i64
        %unrepresentable = arith.cmpi ugt, %target, %byte_max : i64
        %scanned = scf.if %unrepresentable -> (i64) {
          // A code point past latin-1 cannot occur in a latin-1 string.
          scf.yield %minus_one : i64
        } else {
          %target_byte = arith.trunci %target : i64 to i8
          %bytes_walk:2 = scf.while (%i = %from, %ans = %minus_one) : (i64, i64) -> (i64, i64) {
            %above = arith.cmpi sge, %i, %start : i64
            %below = arith.cmpi slt, %i, %end : i64
            %in_range = arith.andi %above, %below : i1
            %not_yet = arith.cmpi eq, %ans, %minus_one : i64
            %go = arith.andi %in_range, %not_yet : i1
            scf.condition(%go) %i, %ans : i64, i64
          } do {
          ^bb0(%i: i64, %ans: i64):
            %i_index = arith.index_cast %i : i64 to index
            %b = memref.load %s_bytes[%i_index] : memref<?xi8>
            %hit = arith.cmpi eq, %b, %target_byte : i8
            %next_ans = arith.select %hit, %i, %ans : i64
            %back = arith.subi %i, %one : i64
            %fwd = arith.addi %i, %one : i64
            %ni = arith.select %reverse, %back, %fwd : i64
            scf.yield %ni, %next_ans : i64, i64
          }
          scf.yield %bytes_walk#1 : i64
        }
        scf.yield %scanned : i64
      } else {
        %walk:2 = scf.while (%i = %from, %ans = %minus_one) : (i64, i64) -> (i64, i64) {
          %above = arith.cmpi sge, %i, %start : i64
          %below = arith.cmpi slt, %i, %end : i64
          %in_range = arith.andi %above, %below : i1
          %not_yet = arith.cmpi eq, %ans, %minus_one : i64
          %go = arith.andi %in_range, %not_yet : i1
          scf.condition(%go) %i, %ans : i64, i64
        } do {
        ^bb0(%i: i64, %ans: i64):
          %i_index = arith.index_cast %i : i64 to index
          %cp = func.call @__ly_unicode_get(%s_bytes, %s_width, %i_index) : (memref<?xi8>, i64, index) -> i64
          %hit = arith.cmpi eq, %cp, %target : i64
          %next_ans = arith.select %hit, %i, %ans : i64
          %back = arith.subi %i, %one : i64
          %fwd = arith.addi %i, %one : i64
          %ni = arith.select %reverse, %back, %fwd : i64
          scf.yield %ni, %next_ans : i64, i64
        }
        scf.yield %walk#1 : i64
      }
      scf.yield %found : i64
    } else {
      %limit = arith.subi %end, %n : i64
      %viable = arith.cmpi sle, %start, %limit : i64
      %found = scf.if %viable -> (i64) {
        // An empty needle matches at the near end of the window, which for the
        // reverse direction is the far one.
        %empty = arith.cmpi eq, %n, %zero : i64
        %either = scf.if %empty -> (i64) {
          %pos = arith.select %reverse, %limit, %start : i64
          scf.yield %pos : i64
        } else {
          %directed = scf.if %reverse -> (i64) {
            %r = func.call @__ly_unicode_find_rev(%s_bytes, %s_width, %t_bytes, %t_width, %start, %end, %n) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64) -> i64
            scf.yield %r : i64
          } else {
            %f = func.call @__ly_unicode_find_fwd(%s_bytes, %s_width, %t_bytes, %t_width, %start, %end, %n) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64) -> i64
            scf.yield %f : i64
          }
          scf.yield %directed : i64
        }
        scf.yield %either : i64
      } else {
        scf.yield %minus_one : i64
      }
      scf.yield %found : i64
    }
    func.return %answer : i64
  }

  func.func @LyUnicode_StartsWith(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %prefix_header: memref<2xi64> {ly.ownership.object_header}, %prefix_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "startswith"} {
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%prefix_header, %prefix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %tail = arith.addi %start, %n : i64
    %fits = arith.cmpi sle, %tail, %end : i64
    %result = scf.if %fits -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %prefix_width = func.call @__ly_unicode_width(%prefix_header) : (memref<2xi64>) -> i64
      %si = arith.index_cast %start : i64 to index
      %ti = arith.constant 0 : index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %si, %prefix_bytes, %prefix_width, %ti, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  func.func @LyUnicode_EndsWith(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %suffix_header: memref<2xi64> {ly.ownership.object_header}, %suffix_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "endswith"} {
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%suffix_header, %suffix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %pos = arith.subi %end, %n : i64
    %fits = arith.cmpi sge, %pos, %start : i64
    %result = scf.if %fits -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %suffix_width = func.call @__ly_unicode_width(%suffix_header) : (memref<2xi64>) -> i64
      %si = arith.index_cast %pos : i64 to index
      %ti = arith.constant 0 : index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %si, %suffix_bytes, %suffix_width, %ti, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  // Shared find/rfind entry: adjusted range, then the linear core scan.
  func.func private @__ly_unicode_find_method(%header: memref<2xi64>, %bytes: memref<?xi8>, %sub_header: memref<2xi64>, %sub_bytes: memref<?xi8>, %start_raw: i64, %end_raw: i64, %reverse: i1) -> i64 {
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%sub_header, %sub_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sub_width = func.call @__ly_unicode_width(%sub_header) : (memref<2xi64>) -> i64
    %found = func.call @__ly_unicode_find_core(%bytes, %width, %sub_bytes, %sub_width, %start, %end, %n, %reverse) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
    func.return %found : i64
  }

  func.func @LyUnicode_Find(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "find", ly.runtime.result_contract = "builtins.int"} {
    %false_bit = arith.constant false
    %found = func.call @__ly_unicode_find_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %false_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %result = func.call @LyLong_FromI64(%found) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // `sub in s` is find() != -1 over the whole string (CPython's
  // unicode_contains), reduced to the i1 the __contains__ dispatch expects
  // instead of boxing an index nobody reads.
  func.func @LyUnicode_Contains(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__contains__"} {
    %false_bit = arith.constant false
    %start = arith.constant 0 : i64
    %end = arith.constant 9223372036854775807 : i64
    %zero = arith.constant 0 : i64
    %found = func.call @__ly_unicode_find_method(%header, %bytes, %sub_header, %sub_bytes, %start, %end, %false_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %present = arith.cmpi sge, %found, %zero : i64
    func.return %present : i1
  }

  func.func @LyUnicode_RFind(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rfind", ly.runtime.result_contract = "builtins.int"} {
    %true_bit = arith.constant true
    %found = func.call @__ly_unicode_find_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %true_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %result = func.call @LyLong_FromI64(%found) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func private @__ly_unicode_index_method(%header: memref<2xi64>, %bytes: memref<?xi8>, %sub_header: memref<2xi64>, %sub_bytes: memref<?xi8>, %start_raw: i64, %end_raw: i64, %reverse: i1) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %found = func.call @__ly_unicode_find_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %reverse) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %missing = arith.cmpi slt, %found, %zero : i64
    scf.if %missing {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 19 : i64
      %static = memref.get_global @__ly_unicode_msg_substring_not_found : memref<19xi8>
      %message = memref.cast %static : memref<19xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %result = func.call @LyLong_FromI64(%found) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func @LyUnicode_Index(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "index", ly.runtime.result_contract = "builtins.int"} {
    %false_bit = arith.constant false
    %result = func.call @__ly_unicode_index_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %false_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func @LyUnicode_RIndex(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rindex", ly.runtime.result_contract = "builtins.int"} {
    %true_bit = arith.constant true
    %result = func.call @__ly_unicode_index_method(%header, %bytes, %sub_header, %sub_bytes, %start_raw, %end_raw, %true_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64, i64, i1) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // Non-overlapping occurrence count in [start, end).
  func.func @LyUnicode_CountSub(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sub_header: memref<2xi64> {ly.ownership.object_header}, %sub_bytes: memref<?xi8>, %start_raw: i64 {ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "count", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_unicode_count(%sub_header, %sub_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %is_empty = arith.cmpi eq, %n, %zero : i64
    %total = scf.if %is_empty -> (i64) {
      %span = arith.subi %end, %start : i64
      %viable = arith.cmpi sge, %span, %zero : i64
      %hits = arith.addi %span, %one : i64
      %count = arith.select %viable, %hits, %zero : i64
      scf.yield %count : i64
    } else {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %sub_width = func.call @__ly_unicode_width(%sub_header) : (memref<2xi64>) -> i64
      %n_index = arith.index_cast %n : i64 to index
      %scan:2 = scf.while (%pos = %start, %count = %zero) : (i64, i64) -> (i64, i64) {
        %tail = arith.addi %pos, %n : i64
        %more = arith.cmpi sle, %tail, %end : i64
        scf.condition(%more) %pos, %count : i64, i64
      } do {
      ^bb0(%pos: i64, %count: i64):
        %pos_index = arith.index_cast %pos : i64 to index
        %ti = arith.constant 0 : index
        %eq = func.call @__ly_unicode_match_at(%bytes, %width, %pos_index, %sub_bytes, %sub_width, %ti, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        %skip = arith.select %eq, %n, %one : i64
        %bump = arith.select %eq, %one, %zero : i64
        %next_pos = arith.addi %pos, %skip : i64
        %next_count = arith.addi %count, %bump : i64
        scf.yield %next_pos, %next_count : i64, i64
      }
      scf.yield %scan#1 : i64
    }
    %result = func.call @LyLong_FromI64(%total) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // ===== str slicing / transform methods =====

  // Copy of the code points [start, end). Every producer must re-scan for
  // the widest code point: a canonical substring of a wide string can be
  // narrower than its source (equality stays bytewise only if slices
  // re-canonicalize).
  // Fresh, unaliased copy of a str. The conditional raise paths need this:
  // an exception __init__ consumes its message, and consuming the (aliased)
  // source value inside one branch would unbalance its ownership token on
  // the other; a fresh copy is created and consumed inside the same branch.
  func.func @LyUnicode_Clone(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "clone", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %copy:2 = func.call @__ly_unicode_slice(%header, %bytes, %c0, %count_index) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %copy#0, %copy#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_unicode_slice(%header: memref<2xi64>, %bytes: memref<?xi8>, %start: index, %end: index) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one_i64 = arith.constant 1 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    // A width-1 source cannot yield a wider slice, so the scan that picks the
    // narrowest output width has nothing to find. CPython skips
    // _PyUnicode_FindMaxChar on a latin-1 source for the same reason, and
    // `__ly_unicode_width_for(0)` is 1, so the scan's answer is already right
    // when it is not run.
    %narrow = arith.cmpi eq, %width, %one_i64 : i64
    %maxcp = scf.if %narrow -> (i64) {
      scf.yield %zero : i64
    } else {
      %scan = scf.for %i = %start to %end step %c1 iter_args(%acc = %zero) -> (i64) {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %bigger = arith.cmpi ugt, %cp, %acc : i64
        %next = arith.select %bigger, %cp, %acc : i64
        scf.yield %next : i64
      }
      scf.yield %scan : i64
    }
    %span = arith.subi %end, %start : index
    %count = arith.index_cast %span : index to i64
    %out_width = func.call @__ly_unicode_width_for(%maxcp) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%count, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %c0, %bytes, %width, %start, %span) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // Retained self, the CPython identity-return fast path (strip() with
  // nothing to strip etc. still copies; only methods documented to return
  // the receiver unchanged use this).
  func.func private @__ly_unicode_retain_self(%header: memref<2xi64>, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Membership of %cp in the chars string (linear: strip char sets are tiny).
  func.func private @__ly_unicode_cp_in_str(%cp: i64, %t_bytes: memref<?xi8>, %t_width: i64, %t_count: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false_bit = arith.constant false
    %found = scf.for %j = %c0 to %t_count step %c1 iter_args(%acc = %false_bit) -> (i1) {
      %tcp = func.call @__ly_unicode_get(%t_bytes, %t_width, %j) : (memref<?xi8>, i64, index) -> i64
      %eq = arith.cmpi eq, %cp, %tcp : i64
      %next = arith.ori %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %found : i1
  }

  // Shared strip walk. %mode bit 1 = strip left, bit 2 = strip right.
  // %use_chars false = Unicode whitespace (str.strip()); true = membership
  // in the chars operand.
  func.func private @__ly_unicode_strip_core(%header: memref<2xi64>, %bytes: memref<?xi8>, %mode: i64, %use_chars: i1, %ch_bytes: memref<?xi8>, %ch_width: i64, %ch_count: index) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %true_bit = arith.constant true
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index

    %left_mask = arith.andi %mode, %one : i64
    %strip_left = arith.cmpi ne, %left_mask, %zero : i64
    %begin = scf.if %strip_left -> (index) {
      %scan:2 = scf.while (%i = %c0, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ult, %i, %count_index : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_unicode_cp_in_str(%cp, %ch_bytes, %ch_width, %ch_count) : (i64, memref<?xi8>, i64, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
          scf.yield %sp : i1
        }
        %next = arith.addi %i, %c1 : index
        %keep = arith.select %stripped, %next, %i : index
        scf.yield %keep, %stripped : index, i1
      }
      scf.yield %scan#0 : index
    } else {
      scf.yield %c0 : index
    }

    %right_mask = arith.andi %mode, %two : i64
    %strip_right = arith.cmpi ne, %right_mask, %zero : i64
    %finish = scf.if %strip_right -> (index) {
      %scan:2 = scf.while (%i = %count_index, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ugt, %i, %begin : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %prev = arith.subi %i, %c1 : index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %prev) : (memref<?xi8>, i64, index) -> i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_unicode_cp_in_str(%cp, %ch_bytes, %ch_width, %ch_count) : (i64, memref<?xi8>, i64, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
          scf.yield %sp : i1
        }
        %keep = arith.select %stripped, %prev, %i : index
        scf.yield %keep, %stripped : index, i1
      }
      scf.yield %scan#0 : index
    } else {
      scf.yield %count_index : index
    }

    %out:2 = func.call @__ly_unicode_slice(%header, %bytes, %begin, %finish) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %out#0, %out#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_unicode_strip_ws(%header: memref<2xi64>, %bytes: memref<?xi8>, %mode: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %false_bit = arith.constant false
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %result:2 = func.call @__ly_unicode_strip_core(%header, %bytes, %mode, %false_bit, %bytes, %one, %c0) : (memref<2xi64>, memref<?xi8>, i64, i1, memref<?xi8>, i64, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_unicode_strip_chars(%header: memref<2xi64>, %bytes: memref<?xi8>, %chars_header: memref<2xi64>, %chars_bytes: memref<?xi8>, %mode: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %true_bit = arith.constant true
    %ch_width = func.call @__ly_unicode_width(%chars_header) : (memref<2xi64>) -> i64
    %ch_count = func.call @__ly_unicode_count(%chars_header, %chars_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %ch_count_index = arith.index_cast %ch_count : i64 to index
    %result:2 = func.call @__ly_unicode_strip_core(%header, %bytes, %mode, %true_bit, %chars_bytes, %ch_width, %ch_count_index) : (memref<2xi64>, memref<?xi8>, i64, i1, memref<?xi8>, i64, index) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Strip(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 3 : i64
    %result:2 = func.call @__ly_unicode_strip_ws(%header, %bytes, %mode) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_LStrip(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 1 : i64
    %result:2 = func.call @__ly_unicode_strip_ws(%header, %bytes, %mode) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RStrip(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 2 : i64
    %result:2 = func.call @__ly_unicode_strip_ws(%header, %bytes, %mode) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_StripChars(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %chars_header: memref<2xi64> {ly.ownership.object_header}, %chars_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 3 : i64
    %result:2 = func.call @__ly_unicode_strip_chars(%header, %bytes, %chars_header, %chars_bytes, %mode) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_LStripChars(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %chars_header: memref<2xi64> {ly.ownership.object_header}, %chars_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 1 : i64
    %result:2 = func.call @__ly_unicode_strip_chars(%header, %bytes, %chars_header, %chars_bytes, %mode) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RStripChars(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %chars_header: memref<2xi64> {ly.ownership.object_header}, %chars_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.str"} {
    %mode = arith.constant 2 : i64
    %result:2 = func.call @__ly_unicode_strip_chars(%header, %bytes, %chars_header, %chars_bytes, %mode) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RemovePrefix(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %prefix_header: memref<2xi64> {ly.ownership.object_header}, %prefix_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "removeprefix", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %n = func.call @__ly_unicode_count(%prefix_header, %prefix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %fits = arith.cmpi sle, %n, %len : i64
    %matched = scf.if %fits -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %pwidth = func.call @__ly_unicode_width(%prefix_header) : (memref<2xi64>) -> i64
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %c0, %prefix_bytes, %pwidth, %c0, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    %result:2 = scf.if %matched -> (memref<2xi64>, memref<?xi8>) {
      %n_index = arith.index_cast %n : i64 to index
      %len_index = arith.index_cast %len : i64 to index
      %sliced:2 = func.call @__ly_unicode_slice(%header, %bytes, %n_index, %len_index) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %sliced#0, %sliced#1 : memref<2xi64>, memref<?xi8>
    } else {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RemoveSuffix(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %suffix_header: memref<2xi64> {ly.ownership.object_header}, %suffix_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "removesuffix", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %n = func.call @__ly_unicode_count(%suffix_header, %suffix_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %pos = arith.subi %len, %n : i64
    %nonempty = arith.cmpi sgt, %n, %zero : i64
    %fits = arith.cmpi sge, %pos, %zero : i64
    %check = arith.andi %nonempty, %fits : i1
    %matched = scf.if %check -> (i1) {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %swidth = func.call @__ly_unicode_width(%suffix_header) : (memref<2xi64>) -> i64
      %n_index = arith.index_cast %n : i64 to index
      %pos_index = arith.index_cast %pos : i64 to index
      %eq = func.call @__ly_unicode_match_at(%bytes, %width, %pos_index, %suffix_bytes, %swidth, %c0, %n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    %result:2 = scf.if %matched -> (memref<2xi64>, memref<?xi8>) {
      %pos_index = arith.index_cast %pos : i64 to index
      %sliced:2 = func.call @__ly_unicode_slice(%header, %bytes, %c0, %pos_index) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %sliced#0, %sliced#1 : memref<2xi64>, memref<?xi8>
    } else {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // str.replace via the unified walk (i <= len when old is empty, so the
  // trailing insertion happens; a match consumes old and suppresses the
  // char emit). Pass 1 measures (count, widest); pass 2 writes.
  // ⭐ THE COMMON SHAPE IS FIND-AND-COPY-RUNS, which is what CPython's
  // `unicode_replace` does: FASTSEARCH to the next occurrence, memcpy the span
  // in between. The general path below walks EVERY position, tries a full
  // substring compare at each, and moves the answer one code point at a time
  // through `__ly_unicode_get`/`__ly_unicode_put` -- 880 characters replaced
  // 200,000 times took 0.65 s against CPython's 0.09 s.
  //
  // ⛔ LATIN-1 INPUT ONLY, and the reason is the canonical form rather than the
  // copying. The output's width has to be the smallest that fits it, or two
  // equal strings can have different bytes and equality stops being bytewise
  // -- and for a wider input the answer depends on whether the widest character
  // survived the replacement, which is a scan of the retained runs. A width-1
  // input cannot narrow, so the answer is `max(1, the replacement's width)`
  // with nothing to scan. Width 1 is every ASCII and every latin-1 string.
  //
  // ⛔ AND AN EMPTY NEEDLE IS NOT THIS SHAPE. `"ab".replace("", "-")` inserts
  // between characters and at both ends; there is nothing to find and no run to
  // copy, so it stays on the general path.
  func.func @LyUnicode_Replace(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %old_header: memref<2xi64> {ly.ownership.object_header}, %old_bytes: memref<?xi8>, %new_header: memref<2xi64> {ly.ownership.object_header}, %new_bytes: memref<?xi8>, %limit: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "replace", ly.runtime.result_contract = "builtins.str"} {
    %one_width = arith.constant 1 : i64
    %zero_len = arith.constant 0 : i64
    %in_width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %needle_n = func.call @__ly_unicode_count(%old_header, %old_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %is_latin1 = arith.cmpi eq, %in_width, %one_width : i64
    %has_needle = arith.cmpi sgt, %needle_n, %zero_len : i64
    %fast = arith.andi %is_latin1, %has_needle : i1
    %picked:2 = scf.if %fast -> (memref<2xi64>, memref<?xi8>) {
      %r:2 = func.call @__ly_unicode_replace_runs(%header, %bytes, %old_header, %old_bytes, %new_header, %new_bytes, %limit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %r#0, %r#1 : memref<2xi64>, memref<?xi8>
    } else {
      %r:2 = func.call @__ly_unicode_replace_scan(%header, %bytes, %old_header, %old_bytes, %new_header, %new_bytes, %limit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %r#0, %r#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %picked#0, %picked#1 : memref<2xi64>, memref<?xi8>
  }

  // Occurrence-driven replace: count with `__ly_unicode_find_core`, allocate
  // once, then copy the spans between matches with `__ly_unicode_copy_run`.
  func.func private @__ly_unicode_replace_runs(%header: memref<2xi64>, %bytes: memref<?xi8>, %old_header: memref<2xi64>, %old_bytes: memref<?xi8>, %new_header: memref<2xi64>, %new_bytes: memref<?xi8>, %limit: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %false_bit = arith.constant false
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %old_width = func.call @__ly_unicode_width(%old_header) : (memref<2xi64>) -> i64
    %new_width = func.call @__ly_unicode_width(%new_header) : (memref<2xi64>) -> i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %old_n = func.call @__ly_unicode_count(%old_header, %old_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_n = func.call @__ly_unicode_count(%new_header, %new_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_n_index = arith.index_cast %new_n : i64 to index

    // ⭐ A SAME-LENGTH REPLACEMENT NEEDS NO COUNTING PASS, which is CPython's
    // `replace_1char_inplace` generalised: the answer is the receiver with some
    // spans overwritten, so the length is known, the whole string is copied
    // once, and the walk poke the replacements in. `"o" -> "0"` over 880
    // characters is the shape this is for, and it halves the searching.
    %same_len = arith.cmpi eq, %old_n, %new_n : i64
    %first = func.call @__ly_unicode_find_core(%bytes, %width, %old_bytes, %old_width, %zero, %len, %old_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
    %matched_any = arith.cmpi sge, %first, %zero : i64
    %budgeted = arith.cmpi ne, %limit, %zero : i64
    %will_replace = arith.andi %matched_any, %budgeted : i1
    %overwrite = arith.andi %will_replace, %same_len : i1
    %stitched:2 = scf.if %overwrite -> (memref<2xi64>, memref<?xi8>) {
      %out_width = arith.maxsi %width, %new_width : i64
      %out_header, %out_bytes = func.call @__ly_unicode_alloc(%len, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %len_index = arith.index_cast %len : i64 to index
      func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %c0, %bytes, %width, %c0, %len_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
      // The search reads the INPUT, never the half-rewritten output: a
      // replacement can spell the needle again (`"ab".replace("ab", "ba")`)
      // and searching what has been written would find it.
      %poke:2 = scf.while (%i = %first, %rem = %limit) : (i64, i64) -> (i64, i64) {
        %found = arith.cmpi sge, %i, %zero : i64
        %budget = arith.cmpi ne, %rem, %zero : i64
        %go = arith.andi %found, %budget : i1
        scf.condition(%go) %i, %rem : i64, i64
      } do {
      ^bb0(%i: i64, %rem: i64):
        %at = arith.index_cast %i : i64 to index
        func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %at, %new_bytes, %new_width, %c0, %new_n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
        %after = arith.addi %i, %old_n : i64
        %next = func.call @__ly_unicode_find_core(%bytes, %width, %old_bytes, %old_width, %after, %len, %old_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        %spent = arith.subi %rem, %one : i64
        scf.yield %next, %spent : i64, i64
      }
      scf.yield %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
    } else {
      %r:2 = func.call @__ly_unicode_replace_spans(%header, %bytes, %old_header, %old_bytes, %new_header, %new_bytes, %limit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %r#0, %r#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %stitched#0, %stitched#1 : memref<2xi64>, memref<?xi8>
  }

  // Different-length replacement: count the occurrences, allocate once, then
  // copy the spans between them.
  func.func private @__ly_unicode_replace_spans(%header: memref<2xi64>, %bytes: memref<?xi8>, %old_header: memref<2xi64>, %old_bytes: memref<?xi8>, %new_header: memref<2xi64>, %new_bytes: memref<?xi8>, %limit: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %false_bit = arith.constant false
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %old_width = func.call @__ly_unicode_width(%old_header) : (memref<2xi64>) -> i64
    %new_width = func.call @__ly_unicode_width(%new_header) : (memref<2xi64>) -> i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %old_n = func.call @__ly_unicode_count(%old_header, %old_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_n = func.call @__ly_unicode_count(%new_header, %new_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_n_index = arith.index_cast %new_n : i64 to index

    %tally:3 = scf.while (%i = %zero, %rem = %limit, %n = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
      %budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %room = arith.cmpi sle, %tail, %len : i64
      %go = arith.andi %budget, %room : i1
      scf.condition(%go) %i, %rem, %n : i64, i64, i64
    } do {
    ^bb0(%i: i64, %rem: i64, %n: i64):
      %hit = func.call @__ly_unicode_find_core(%bytes, %width, %old_bytes, %old_width, %i, %len, %old_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
      %found = arith.cmpi sge, %hit, %zero : i64
      %after = arith.addi %hit, %old_n : i64
      // A miss ends the walk by stepping past what the guard admits.
      %stop = arith.addi %len, %one : i64
      %next_i = arith.select %found, %after, %stop : i64
      %bumped = arith.addi %n, %one : i64
      %next_n = arith.select %found, %bumped, %n : i64
      %spent = arith.subi %rem, %one : i64
      %next_rem = arith.select %found, %spent, %rem : i64
      scf.yield %next_i, %next_rem, %next_n : i64, i64, i64
    }
    %none = arith.cmpi eq, %tally#2, %zero : i64
    %result:2 = scf.if %none -> (memref<2xi64>, memref<?xi8>) {
      // Nothing matched, so the answer IS the receiver -- the same object with
      // one more reference, which is `unicode_result_unchanged`.
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    } else {
      %delta = arith.subi %new_n, %old_n : i64
      %grown = arith.muli %tally#2, %delta : i64
      %total = arith.addi %len, %grown : i64
      %out_width = arith.maxsi %width, %new_width : i64
      %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %count_index = arith.index_cast %tally#2 : i64 to index
      %walk:2 = scf.for %k = %c0 to %count_index step %c1 iter_args(%i = %zero, %pos = %c0) -> (i64, index) {
        %hit = func.call @__ly_unicode_find_core(%bytes, %width, %old_bytes, %old_width, %i, %len, %old_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        %run = arith.subi %hit, %i : i64
        %run_index = arith.index_cast %run : i64 to index
        %from_index = arith.index_cast %i : i64 to index
        func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %pos, %bytes, %width, %from_index, %run_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
        %after_run = arith.addi %pos, %run_index : index
        func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %after_run, %new_bytes, %new_width, %c0, %new_n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
        %next_pos = arith.addi %after_run, %new_n_index : index
        %next_i = arith.addi %hit, %old_n : i64
        scf.yield %next_i, %next_pos : i64, index
      }
      %tail_len = arith.subi %len, %walk#0 : i64
      %tail_index = arith.index_cast %tail_len : i64 to index
      %tail_from = arith.index_cast %walk#0 : i64 to index
      func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %walk#1, %bytes, %width, %tail_from, %tail_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
      scf.yield %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_unicode_replace_scan(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %old_header: memref<2xi64> {ly.ownership.object_header}, %old_bytes: memref<?xi8>, %new_header: memref<2xi64> {ly.ownership.object_header}, %new_bytes: memref<?xi8>, %limit: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %old_width = func.call @__ly_unicode_width(%old_header) : (memref<2xi64>) -> i64
    %new_width = func.call @__ly_unicode_width(%new_header) : (memref<2xi64>) -> i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %old_n = func.call @__ly_unicode_count(%old_header, %old_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_n = func.call @__ly_unicode_count(%new_header, %new_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %old_n_index = arith.index_cast %old_n : i64 to index
    %new_n_index = arith.index_cast %new_n : i64 to index
    %new_max = scf.for %j = %c0 to %new_n_index step %c1 iter_args(%acc = %zero) -> (i64) {
      %cp = func.call @__ly_unicode_get(%new_bytes, %new_width, %j) : (memref<?xi8>, i64, index) -> i64
      %bigger = arith.cmpi ugt, %cp, %acc : i64
      %next = arith.select %bigger, %cp, %acc : i64
      scf.yield %next : i64
    }
    %old_empty = arith.cmpi eq, %old_n, %zero : i64
    %bound = scf.if %old_empty -> (i64) {
      %plus = arith.addi %len, %one : i64
      scf.yield %plus : i64
    } else {
      scf.yield %len : i64
    }

    %measure:4 = scf.while (%i = %zero, %rem = %limit, %total = %zero, %maxcp = %zero) : (i64, i64, i64, i64) -> (i64, i64, i64, i64) {
      %more = arith.cmpi slt, %i, %bound : i64
      scf.condition(%more) %i, %rem, %total, %maxcp : i64, i64, i64, i64
    } do {
    ^bb0(%i: i64, %rem: i64, %total: i64, %maxcp: i64):
      %has_budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %in_range = arith.cmpi sle, %tail, %len : i64
      %viable = arith.andi %has_budget, %in_range : i1
      %matched = scf.if %viable -> (i1) {
        %i_index = arith.index_cast %i : i64 to index
        %eq = func.call @__ly_unicode_match_at(%bytes, %width, %i_index, %old_bytes, %old_width, %c0, %old_n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %true_a = arith.constant true
      %not_skip = arith.xori %skip_char, %true_a : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %new_contrib = arith.select %matched, %new_n, %zero : i64
      %char_contrib = arith.select %emit_char, %one, %zero : i64
      %next_total_a = arith.addi %total, %new_contrib : i64
      %next_total = arith.addi %next_total_a, %char_contrib : i64
      %match_max = arith.select %matched, %new_max, %zero : i64
      %cp = scf.if %emit_char -> (i64) {
        %i_index = arith.index_cast %i : i64 to index
        %value = func.call @__ly_unicode_get(%bytes, %width, %i_index) : (memref<?xi8>, i64, index) -> i64
        scf.yield %value : i64
      } else {
        scf.yield %zero : i64
      }
      %m1 = arith.maxui %maxcp, %match_max : i64
      %next_max = arith.maxui %m1, %cp : i64
      %stride = arith.select %skip_char, %old_n, %one : i64
      %next_i = arith.addi %i, %stride : i64
      %dec = arith.select %matched, %one, %zero : i64
      %next_rem = arith.subi %rem, %dec : i64
      scf.yield %next_i, %next_rem, %next_total, %next_max : i64, i64, i64, i64
    }

    %out_width = func.call @__ly_unicode_width_for(%measure#3) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%measure#2, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)

    %write:3 = scf.while (%i = %zero, %rem = %limit, %pos = %c0) : (i64, i64, index) -> (i64, i64, index) {
      %more = arith.cmpi slt, %i, %bound : i64
      scf.condition(%more) %i, %rem, %pos : i64, i64, index
    } do {
    ^bb0(%i: i64, %rem: i64, %pos: index):
      %has_budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %in_range = arith.cmpi sle, %tail, %len : i64
      %viable = arith.andi %has_budget, %in_range : i1
      %matched = scf.if %viable -> (i1) {
        %i_index = arith.index_cast %i : i64 to index
        %eq = func.call @__ly_unicode_match_at(%bytes, %width, %i_index, %old_bytes, %old_width, %c0, %old_n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %after_new = scf.if %matched -> (index) {
        scf.for %j = %c0 to %new_n_index step %c1 {
          %cp = func.call @__ly_unicode_get(%new_bytes, %new_width, %j) : (memref<?xi8>, i64, index) -> i64
          %dst = arith.addi %pos, %j : index
          func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        }
        %advanced = arith.addi %pos, %new_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %true_b = arith.constant true
      %not_skip = arith.xori %skip_char, %true_b : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %after_char = scf.if %emit_char -> (index) {
        %i_index = arith.index_cast %i : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i_index) : (memref<?xi8>, i64, index) -> i64
        func.call @__ly_unicode_put(%out_bytes, %out_width, %after_new, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        %advanced = arith.addi %after_new, %c1 : index
        scf.yield %advanced : index
      } else {
        scf.yield %after_new : index
      }
      %stride = arith.select %skip_char, %old_n, %one : i64
      %next_i = arith.addi %i, %stride : i64
      %dec = arith.select %matched, %one, %zero : i64
      %next_rem = arith.subi %rem, %dec : i64
      scf.yield %next_i, %next_rem, %after_char : i64, i64, index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // center/ljust/rjust core: %place 0 = pad left (rjust), 1 = pad right
  // (ljust), 2 = center with CPython's left = marg/2 + (marg & width & 1).
  func.func private @__ly_unicode_pad(%header: memref<2xi64>, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64>, %fill_bytes: memref<?xi8>, %place: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %fill_n = func.call @__ly_unicode_count(%fill_header, %fill_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %single = arith.cmpi eq, %fill_n, %one : i64
    %true_bit = arith.constant true
    %bad = arith.xori %single, %true_bit : i1
    scf.if %bad {
      %class_id = arith.constant 52 : i64
      %length = arith.constant 53 : i64
      %static = memref.get_global @__ly_unicode_msg_bad_fill : memref<53xi8>
      %message = memref.cast %static : memref<53xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %len_index_bound = arith.index_cast %len : i64 to index
    %no_pad = arith.cmpi sle, %target, %len : i64
    %result:2 = scf.if %no_pad -> (memref<2xi64>, memref<?xi8>) {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    } else {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %fill_width = func.call @__ly_unicode_width(%fill_header) : (memref<2xi64>) -> i64
      %fill_cp = func.call @__ly_unicode_get(%fill_bytes, %fill_width, %c0) : (memref<?xi8>, i64, index) -> i64
      %self_max = scf.for %i = %c0 to %len_index_bound step %c1 iter_args(%acc = %zero) -> (i64) {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %bigger = arith.cmpi ugt, %cp, %acc : i64
        %next = arith.select %bigger, %cp, %acc : i64
        scf.yield %next : i64
      }
      %maxcp = arith.maxui %self_max, %fill_cp : i64
      %out_width = func.call @__ly_unicode_width_for(%maxcp) : (i64) -> i64
      %str_prefix = arith.constant 32 : i64
      func.call @__ly_check_alloc_count(%target, %out_width, %str_prefix) : (i64, i64, i64) -> ()
      %out_header, %out_bytes = func.call @__ly_unicode_alloc(%target, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %margin = arith.subi %target, %len : i64
      %half = arith.divsi %margin, %two : i64
      %mw = arith.andi %margin, %target : i64
      %mw1 = arith.andi %mw, %one : i64
      %center_left = arith.addi %half, %mw1 : i64
      %is_left = arith.cmpi eq, %place, %zero : i64
      %is_right = arith.cmpi eq, %place, %one : i64
      %center_or = arith.select %is_right, %zero, %center_left : i64
      %left = arith.select %is_left, %margin, %center_or : i64
      %left_index = arith.index_cast %left : i64 to index
      %target_index = arith.index_cast %target : i64 to index
      scf.for %i = %c0 to %target_index step %c1 {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %i, %fill_cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.for %i = %c0 to %len_index_bound step %c1 {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %dst = arith.addi %left_index, %i : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.yield %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_RJust(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = " "}, %fill_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rjust", ly.runtime.result_contract = "builtins.str"} {
    %place = arith.constant 0 : i64
    %result:2 = func.call @__ly_unicode_pad(%header, %bytes, %target, %fill_header, %fill_bytes, %place) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_LJust(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = " "}, %fill_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "ljust", ly.runtime.result_contract = "builtins.str"} {
    %place = arith.constant 1 : i64
    %result:2 = func.call @__ly_unicode_pad(%header, %bytes, %target, %fill_header, %fill_bytes, %place) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Center(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64, %fill_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = " "}, %fill_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "center", ly.runtime.result_contract = "builtins.str"} {
    %place = arith.constant 2 : i64
    %result:2 = func.call @__ly_unicode_pad(%header, %bytes, %target, %fill_header, %fill_bytes, %place) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // str.zfill: zero-pad on the left, keeping a leading sign in front.
  func.func @LyUnicode_ZFill(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %target: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "zfill", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %ascii_zero = arith.constant 48 : i64
    %plus = arith.constant 43 : i64
    %minus = arith.constant 45 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %no_pad = arith.cmpi sle, %target, %len : i64
    %result:2 = scf.if %no_pad -> (memref<2xi64>, memref<?xi8>) {
      %kept:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
    } else {
      %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
      %len_index = arith.index_cast %len : i64 to index
      %self_max = scf.for %i = %c0 to %len_index step %c1 iter_args(%acc = %zero) -> (i64) {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %bigger = arith.cmpi ugt, %cp, %acc : i64
        %next = arith.select %bigger, %cp, %acc : i64
        scf.yield %next : i64
      }
      %maxcp = arith.maxui %self_max, %ascii_zero : i64
      %out_width = func.call @__ly_unicode_width_for(%maxcp) : (i64) -> i64
      %str_prefix = arith.constant 32 : i64
      func.call @__ly_check_alloc_count(%target, %out_width, %str_prefix) : (i64, i64, i64) -> ()
      %out_header, %out_bytes = func.call @__ly_unicode_alloc(%target, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %has_any = arith.cmpi sgt, %len, %zero : i64
      %first = scf.if %has_any -> (i64) {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %c0) : (memref<?xi8>, i64, index) -> i64
        scf.yield %cp : i64
      } else {
        scf.yield %zero : i64
      }
      %is_plus = arith.cmpi eq, %first, %plus : i64
      %is_minus = arith.cmpi eq, %first, %minus : i64
      %signed = arith.ori %is_plus, %is_minus : i1
      %sign_len = arith.select %signed, %c1, %c0 : index
      scf.if %signed {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %c0, %first) : (memref<?xi8>, i64, index, i64) -> ()
      }
      %margin = arith.subi %target, %len : i64
      %margin_index = arith.index_cast %margin : i64 to index
      %zeros_end = arith.addi %sign_len, %margin_index : index
      scf.for %i = %sign_len to %zeros_end step %c1 {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %i, %ascii_zero) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.for %i = %sign_len to %len_index step %c1 {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %dst = arith.addi %margin_index, %i : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
      scf.yield %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // str.expandtabs: a tab advances to the next multiple of tabsize within
  // the current line (\n and \r reset the column); tabsize <= 0 deletes
  // tabs, matching CPython.
  func.func @LyUnicode_ExpandTabs(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %tabsize: i64 {ly.runtime.default_i64 = 8 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "expandtabs", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %tab = arith.constant 9 : i64
    %nl = arith.constant 10 : i64
    %cr = arith.constant 13 : i64
    %space = arith.constant 32 : i64
    // CPython's tabsize is a C int.
    %int_max = arith.constant 2147483647 : i64
    %int_min = arith.constant -2147483648 : i64
    %above_int = arith.cmpi sgt, %tabsize, %int_max : i64
    %below_int = arith.cmpi slt, %tabsize, %int_min : i64
    %past_int = arith.ori %above_int, %below_int : i1
    scf.if %past_int {
      %class_id = arith.constant 104 : i64
      %length = arith.constant 40 : i64
      %message_static = memref.get_global @__ly_unicode_msg_int_too_large_c_int : memref<40xi8>
      %message = memref.cast %message_static : memref<40xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %positive = arith.cmpi sgt, %tabsize, %zero : i64

    %measure:3 = scf.for %i = %c0 to %len_index step %c1 iter_args(%total = %zero, %col = %zero, %maxcp = %zero) -> (i64, i64, i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_tab = arith.cmpi eq, %cp, %tab : i64
      %step:3 = scf.if %is_tab -> (i64, i64, i64) {
        %rem = scf.if %positive -> (i64) {
          %m = arith.remsi %col, %tabsize : i64
          %incr = arith.subi %tabsize, %m : i64
          scf.yield %incr : i64
        } else {
          scf.yield %zero : i64
        }
        %fill_max = arith.cmpi sgt, %rem, %zero : i64
        %contrib_max = arith.select %fill_max, %space, %zero : i64
        %next_col = arith.addi %col, %rem : i64
        scf.yield %rem, %next_col, %contrib_max : i64, i64, i64
      } else {
        %is_nl = arith.cmpi eq, %cp, %nl : i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %resets = arith.ori %is_nl, %is_cr : i1
        %bumped = arith.addi %col, %one : i64
        %next_col = arith.select %resets, %zero, %bumped : i64
        scf.yield %one, %next_col, %cp : i64, i64, i64
      }
      %next_total = arith.addi %total, %step#0 : i64
      %next_max = arith.maxui %maxcp, %step#2 : i64
      scf.yield %next_total, %step#1, %next_max : i64, i64, i64
    }

    %out_width = func.call @__ly_unicode_width_for(%measure#2) : (i64) -> i64
    %str_prefix = arith.constant 32 : i64
    func.call @__ly_check_alloc_count(%measure#0, %out_width, %str_prefix) : (i64, i64, i64) -> ()
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%measure#0, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)

    scf.for %i = %c0 to %len_index step %c1 iter_args(%pos = %c0, %col = %zero) -> (index, i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_tab = arith.cmpi eq, %cp, %tab : i64
      %step:2 = scf.if %is_tab -> (index, i64) {
        %rem = scf.if %positive -> (i64) {
          %m = arith.remsi %col, %tabsize : i64
          %incr = arith.subi %tabsize, %m : i64
          scf.yield %incr : i64
        } else {
          scf.yield %zero : i64
        }
        %rem_index = arith.index_cast %rem : i64 to index
        scf.for %j = %c0 to %rem_index step %c1 {
          %dst = arith.addi %pos, %j : index
          func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %space) : (memref<?xi8>, i64, index, i64) -> ()
        }
        %next_pos = arith.addi %pos, %rem_index : index
        %next_col = arith.addi %col, %rem : i64
        scf.yield %next_pos, %next_col : index, i64
      } else {
        func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %cp) : (memref<?xi8>, i64, index, i64) -> ()
        %next_pos = arith.addi %pos, %c1 : index
        %is_nl = arith.cmpi eq, %cp, %nl : i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %resets = arith.ori %is_nl, %is_cr : i1
        %bumped = arith.addi %col, %one : i64
        %next_col = arith.select %resets, %zero, %bumped : i64
        scf.yield %next_pos, %next_col : index, i64
      }
      scf.yield %step#0, %step#1 : index, i64
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // str * int (left str only; negative repeats give the empty string).
  func.func @LyUnicode_Mul(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %repeat: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__mul__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %n = arith.maxsi %repeat, %zero : i64
    %overflows = func.call @__ly_repeat_overflows(%len, %n) : (i64, i64) -> i1
    scf.if %overflows {
      %class_id = arith.constant 104 : i64
      %length = arith.constant 27 : i64
      %message_static = memref.get_global @__ly_unicode_msg_repeat_too_long : memref<27xi8>
      %message = memref.cast %message_static : memref<27xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %total = arith.muli %len, %n : i64
    %is_empty = arith.cmpi eq, %total, %zero : i64
    %out_width = arith.select %is_empty, %one, %width : i64
    %str_prefix = arith.constant 32 : i64
    func.call @__ly_check_alloc_count(%total, %out_width, %str_prefix) : (i64, i64, i64) -> ()
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %len_index = arith.index_cast %len : i64 to index
    // No trips over nothing (see `__ly_seq_fill_repeat`).
    %no_chars = arith.cmpi eq, %len, %zero : i64
    %trips = arith.select %no_chars, %zero, %n : i64
    %n_index = arith.index_cast %trips : i64 to index
    scf.for %k = %c0 to %n_index step %c1 {
      %base = arith.muli %k, %len_index : index
      scf.for %i = %c0 to %len_index step %c1 {
        %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
        %dst = arith.addi %base, %i : index
        func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %cp) : (memref<?xi8>, i64, index, i64) -> ()
      }
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // ===== str splitting / joining =====

  // "empty separator"
  memref.global "private" constant @__ly_unicode_msg_empty_separator : memref<15xi8> = dense<[101, 109, 112, 116, 121, 32, 115, 101, 112, 97, 114, 97, 116, 111, 114]>

  // Pack an owned str into payload box slot %slot: the header's address and the
  // class id, which is the whole box. The reference transfers to the container;
  // the deallocator releases it through the boxed-release hook, and the bytes
  // come back from the block the address names.
  //
  // ⛔ NOTHING IS CLEARED ANY MORE. A slot is reused -- `LyList_SetItemBox`
  // overwrites one that may have held a different entity -- and the clearing
  // was there because the box had lanes a narrower contract would leave behind.
  // Every word of a box is written now.
  func.func private @__ly_unicode_store_item(%items: memref<?xi64>, %slot: i64, %eh: memref<2xi64> {ly.ownership.object_header}, %eb: memref<?xi8>) attributes {ly.ownership.transfer_args = [2]} {
    %one = arith.constant 1 : i64
    %str_class = arith.constant 4 : i64
    %hdr_ptr_index = memref.extract_aligned_pointer_as_index %eh : memref<2xi64> -> index
    %hdr_ptr = arith.index_cast %hdr_ptr_index : index to i64
    func.call @__ly_box_store_entity(%items, %slot, %str_class, %hdr_ptr) : (memref<?xi64>, i64, i64, i64) -> ()
    func.return
  }

  // Store a freshly sliced [start, end) segment into box slot %slot.
  func.func private @__ly_unicode_store_slice(%items: memref<?xi64>, %slot: i64, %header: memref<2xi64>, %bytes: memref<?xi8>, %start: index, %end: index) {
    %piece:2 = func.call @__ly_unicode_slice(%header, %bytes, %start, %end) : (memref<2xi64>, memref<?xi8>, index, index) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_unicode_store_item(%items, %slot, %piece#0, %piece#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // str.split(sep[, maxsplit]) -- explicit separator form. The empty
  // separator raises like CPython.
  func.func @LyUnicode_Split(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %false_bit = arith.constant false
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }

    // Pass 1: number of separators actually used.
    %true_split1 = arith.constant true
    %count:3 = scf.while (%pos = %zero, %used = %zero, %go = %true_split1) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %used, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %used: i64, %go: i1):
      %budget_left = arith.cmpi ne, %used, %maxsplit : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      %next_pos_hit = arith.addi %hit, %sep_n : i64
      %next_pos = arith.select %matched, %next_pos_hit, %pos : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_used = arith.addi %used, %bump : i64
      scf.yield %next_pos, %next_used, %matched : i64, i64, i1
    }

    %segments = arith.addi %count#1, %one : i64
    %list = func.call @LyList_FromLength(%segments) : (i64) -> memref<5xi64>
    %list_items = func.call @__ly_list_items(%list) : (memref<5xi64>) -> memref<?xi64>

    // Pass 2: emit the segments.
    %true_split2 = arith.constant true
    %emit:3 = scf.while (%pos = %zero, %slot = %zero, %go = %true_split2) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %slot, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %slot: i64, %go: i1):
      %remaining = arith.subi %segments, %one : i64
      %budget_left = arith.cmpi slt, %slot, %remaining : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      scf.if %matched {
        %pos_index = arith.index_cast %pos : i64 to index
        %hit_index = arith.index_cast %hit : i64 to index
        func.call @__ly_unicode_store_slice(%list_items, %slot, %header, %bytes, %pos_index, %hit_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      %next_pos_hit = arith.addi %hit, %sep_n : i64
      %next_pos = arith.select %matched, %next_pos_hit, %pos : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      scf.yield %next_pos, %next_slot, %matched : i64, i64, i1
    }
    %tail_start = arith.index_cast %emit#0 : i64 to index
    %len_index = arith.index_cast %len : i64 to index
    func.call @__ly_unicode_store_slice(%list_items, %emit#1, %header, %bytes, %tail_start, %len_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
    func.return %list : memref<5xi64>
  }

  // str.rsplit(sep[, maxsplit]): scans right-to-left (a left scan with a
  // skip count is wrong when candidate matches overlap: "aaa".rsplit("aa")
  // is ['a', ''], not ['', 'a']), filling slots from the back.
  func.func @LyUnicode_RSplit(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rsplit", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %true_bit = arith.constant true
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }

    %true_rsplit1 = arith.constant true
    %count:3 = scf.while (%end = %len, %used = %zero, %go = %true_rsplit1) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %end, %used, %go : i64, i64, i1
    } do {
    ^bb0(%end: i64, %used: i64, %go: i1):
      %budget_left = arith.cmpi ne, %used, %maxsplit : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %zero, %end, %sep_n, %true_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      %next_end = arith.select %matched, %hit, %end : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_used = arith.addi %used, %bump : i64
      scf.yield %next_end, %next_used, %matched : i64, i64, i1
    }

    %segments = arith.addi %count#1, %one : i64
    %list = func.call @LyList_FromLength(%segments) : (i64) -> memref<5xi64>
    %list_items = func.call @__ly_list_items(%list) : (memref<5xi64>) -> memref<?xi64>

    %true_rsplit2 = arith.constant true
    %emit:3 = scf.while (%end = %len, %slot = %count#1, %go = %true_rsplit2) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %end, %slot, %go : i64, i64, i1
    } do {
    ^bb0(%end: i64, %slot: i64, %go: i1):
      %budget_left = arith.cmpi sgt, %slot, %zero : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %zero, %end, %sep_n, %true_bit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      scf.if %matched {
        %seg_start = arith.addi %hit, %sep_n : i64
        %seg_start_index = arith.index_cast %seg_start : i64 to index
        %end_index = arith.index_cast %end : i64 to index
        func.call @__ly_unicode_store_slice(%list_items, %slot, %header, %bytes, %seg_start_index, %end_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      %next_end = arith.select %matched, %hit, %end : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_slot = arith.subi %slot, %bump : i64
      scf.yield %next_end, %next_slot, %matched : i64, i64, i1
    }
    %c0 = arith.constant 0 : index
    %head_end = arith.index_cast %emit#0 : i64 to index
    func.call @__ly_unicode_store_slice(%list_items, %zero, %header, %bytes, %c0, %head_end) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
    func.return %list : memref<5xi64>
  }

  // Whitespace split (str.split() / str.rsplit() with no separator): runs
  // of Unicode whitespace delimit; leading/trailing whitespace produces no
  // empty segments. Unlimited maxsplit makes the two directions agree, so
  // one forward implementation serves both names.
  // maxsplit < 0 means unlimited. The cap only engages when the string has
  // MORE runs than the cap allows (`"  a  ".split(maxsplit=1)` is ['a'], not
  // ['a  ']): with no split actually withheld there is no remainder, and the
  // remainder is the only part that keeps its interior and trailing spaces.
  func.func private @__ly_unicode_split_ws_core(%header: memref<2xi64>, %bytes: memref<?xi8>, %maxsplit: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %len_index = arith.index_cast %len : i64 to index

    // Pass 1: segment count = number of non-space runs.
    %false_ws1 = arith.constant false
    %count:2 = scf.for %i = %c0 to %len_index step %c1 iter_args(%segs = %zero, %in_run = %false_ws1) -> (i64, i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_space = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
      %true_w = arith.constant true
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts = arith.andi %non_space, %not_in_run : i1
      %bump = arith.select %starts, %one, %zero : i64
      %next_segs = arith.addi %segs, %bump : i64
      scf.yield %next_segs, %non_space : i64, i1
    }

    // The cap engages only when it actually withholds a split.
    %capped = arith.cmpi sge, %maxsplit, %zero : i64
    %over = arith.cmpi sgt, %count#0, %maxsplit : i64
    %cap_active = arith.andi %capped, %over : i1
    %cap_count = arith.addi %maxsplit, %one : i64
    %emitted = arith.select %cap_active, %cap_count, %count#0 : i64
    %tail_slot = arith.subi %emitted, %one : i64

    %list = func.call @LyList_FromLength(%emitted) : (i64) -> memref<5xi64>
    %list_items = func.call @__ly_list_items(%list) : (memref<5xi64>) -> memref<?xi64>

    // Pass 2: emit each run. Once the tail is emitted `done` stays set and
    // every producer below is gated on it: the remainder is one slice from
    // its run's start to the end of the string, spaces and all.
    %false_ws2 = arith.constant false
    %false_done = arith.constant false
    scf.for %i = %c0 to %len_index step %c1 iter_args(%slot = %zero, %run_start = %len_index, %in_run = %false_ws2, %done = %false_done) -> (i64, index, i1, i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_space = func.call @__ly_unicode_cp_is_space(%cp) : (i64) -> i1
      %true_w = arith.constant true
      %not_done = arith.xori %done, %true_w : i1
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts_raw = arith.andi %non_space, %not_in_run : i1
      %starts = arith.andi %starts_raw, %not_done : i1
      %ends_raw = arith.andi %is_space, %in_run : i1
      %ends = arith.andi %ends_raw, %not_done : i1
      %at_tail = arith.cmpi eq, %slot, %tail_slot : i64
      %tail_here = arith.andi %cap_active, %at_tail : i1
      %is_tail = arith.andi %starts, %tail_here : i1
      %new_start = arith.select %starts, %i, %run_start : index
      scf.if %is_tail {
        func.call @__ly_unicode_store_slice(%list_items, %slot, %header, %bytes, %i, %len_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      scf.if %ends {
        func.call @__ly_unicode_store_slice(%list_items, %slot, %header, %bytes, %run_start, %i) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      %bump = arith.select %ends, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      %next_done = arith.ori %done, %is_tail : i1
      %still_running = arith.xori %next_done, %true_w : i1
      %next_in_run = arith.andi %non_space, %still_running : i1
      %last = arith.subi %len_index, %c1 : index
      %is_last = arith.cmpi eq, %i, %last : index
      %closes_raw = arith.andi %is_last, %non_space : i1
      %closes = arith.andi %closes_raw, %still_running : i1
      scf.if %closes {
        func.call @__ly_unicode_store_slice(%list_items, %next_slot, %header, %bytes, %new_start, %len_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      }
      scf.yield %next_slot, %new_start, %next_in_run, %next_done : i64, index, i1, i1
    }
    func.return %list : memref<5xi64>
  }

  func.func @LyUnicode_SplitWS(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %result = func.call @__ly_unicode_split_ws_core(%header, %bytes, %maxsplit) : (memref<2xi64>, memref<?xi8>, i64) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  func.func @LyUnicode_RSplitWS(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rsplit", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    // ⛔ No maxsplit overload for rsplit: its cap withholds splits from the
    // RIGHT ("a b c".rsplit(maxsplit=1) is ['a b', 'c']), which this
    // left-to-right walk cannot produce. Refusing the argument is the answer
    // until the mirrored walk exists.
    %unlimited = arith.constant -1 : i64
    %result = func.call @__ly_unicode_split_ws_core(%header, %bytes, %unlimited) : (memref<2xi64>, memref<?xi8>, i64) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // Unicode line boundaries per CPython str.splitlines: \n \v \f \r \x1c
  // \x1d \x1e \x85 \u2028 \u2029, with \r\n as one boundary.
  func.func private @__ly_unicode_cp_is_linebreak(%cp: i64) -> i1 {
    %nl = arith.constant 10 : i64
    %vt = arith.constant 11 : i64
    %ff = arith.constant 12 : i64
    %cr = arith.constant 13 : i64
    %fs = arith.constant 28 : i64
    %gs = arith.constant 29 : i64
    %rs = arith.constant 30 : i64
    %nel = arith.constant 133 : i64
    %ls = arith.constant 8232 : i64
    %ps = arith.constant 8233 : i64
    %in_c0 = arith.cmpi sge, %cp, %nl : i64
    %le_cr = arith.cmpi sle, %cp, %cr : i64
    %ctl = arith.andi %in_c0, %le_cr : i1
    %ge_fs = arith.cmpi sge, %cp, %fs : i64
    %le_rs = arith.cmpi sle, %cp, %rs : i64
    %seps = arith.andi %ge_fs, %le_rs : i1
    %is_nel = arith.cmpi eq, %cp, %nel : i64
    %is_ls = arith.cmpi eq, %cp, %ls : i64
    %is_ps = arith.cmpi eq, %cp, %ps : i64
    %a = arith.ori %ctl, %seps : i1
    %b = arith.ori %is_nel, %is_ls : i1
    %c = arith.ori %b, %is_ps : i1
    %result = arith.ori %a, %c : i1
    func.return %result : i1
  }

  func.func private @__ly_unicode_splitlines_core(%header: memref<2xi64>, %bytes: memref<?xi8>, %keepends: i1) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]} {
    %c1i = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %cr = arith.constant 13 : i64
    %nl = arith.constant 10 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64

    // Pass 1: line count.
    %count:2 = scf.while (%i = %zero, %lines = %zero) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi slt, %i, %len : i64
      scf.condition(%more) %i, %lines : i64, i64
    } do {
    ^bb0(%i: i64, %lines: i64):
      // Scan to the end of this line's content.
      %true_lines1 = arith.constant true
      %scan:2 = scf.while (%j = %i, %go = %true_lines1) : (i64, i1) -> (i64, i1) {
        %in_bounds = arith.cmpi slt, %j, %len : i64
        %continue = arith.andi %in_bounds, %go : i1
        scf.condition(%continue) %j, %go : i64, i1
      } do {
      ^bb1(%j: i64, %go: i1):
        %j_index = arith.index_cast %j : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %j_index) : (memref<?xi8>, i64, index) -> i64
        %brk = func.call @__ly_unicode_cp_is_linebreak(%cp) : (i64) -> i1
        %true_x = arith.constant true
        %not_brk = arith.xori %brk, %true_x : i1
        %next = arith.addi %j, %one : i64
        %sel = arith.select %brk, %j, %next : i64
        scf.yield %sel, %not_brk : i64, i1
      }
      %ended = arith.cmpi slt, %scan#0, %len : i64
      %skip = scf.if %ended -> (i64) {
        %eol_index = arith.index_cast %scan#0 : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %eol_index) : (memref<?xi8>, i64, index) -> i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %next_pos = arith.addi %scan#0, %one : i64
        %has_next = arith.cmpi slt, %next_pos, %len : i64
        %pair = arith.andi %is_cr, %has_next : i1
        %crlf = scf.if %pair -> (i1) {
          %next_index = arith.index_cast %next_pos : i64 to index
          %cp2 = func.call @__ly_unicode_get(%bytes, %width, %next_index) : (memref<?xi8>, i64, index) -> i64
          %is_nl = arith.cmpi eq, %cp2, %nl : i64
          scf.yield %is_nl : i1
        } else {
          %false_x = arith.constant false
          scf.yield %false_x : i1
        }
        %stride = arith.select %crlf, %two, %one : i64
        scf.yield %stride : i64
      } else {
        scf.yield %zero : i64
      }
      %next_i = arith.addi %scan#0, %skip : i64
      %next_lines = arith.addi %lines, %one : i64
      scf.yield %next_i, %next_lines : i64, i64
    }

    %list = func.call @LyList_FromLength(%count#1) : (i64) -> memref<5xi64>
    %list_items = func.call @__ly_list_items(%list) : (memref<5xi64>) -> memref<?xi64>

    // Pass 2: emit each line ([start, eol) or [start, eol+break)).
    %emitted:2 = scf.while (%i = %zero, %slot = %zero) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi slt, %i, %len : i64
      scf.condition(%more) %i, %slot : i64, i64
    } do {
    ^bb0(%i: i64, %slot: i64):
      %true_lines2 = arith.constant true
      %scan:2 = scf.while (%j = %i, %go = %true_lines2) : (i64, i1) -> (i64, i1) {
        %in_bounds = arith.cmpi slt, %j, %len : i64
        %continue = arith.andi %in_bounds, %go : i1
        scf.condition(%continue) %j, %go : i64, i1
      } do {
      ^bb1(%j: i64, %go: i1):
        %j_index = arith.index_cast %j : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %j_index) : (memref<?xi8>, i64, index) -> i64
        %brk = func.call @__ly_unicode_cp_is_linebreak(%cp) : (i64) -> i1
        %true_x = arith.constant true
        %not_brk = arith.xori %brk, %true_x : i1
        %next = arith.addi %j, %one : i64
        %sel = arith.select %brk, %j, %next : i64
        scf.yield %sel, %not_brk : i64, i1
      }
      %ended = arith.cmpi slt, %scan#0, %len : i64
      %skip = scf.if %ended -> (i64) {
        %eol_index = arith.index_cast %scan#0 : i64 to index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %eol_index) : (memref<?xi8>, i64, index) -> i64
        %is_cr = arith.cmpi eq, %cp, %cr : i64
        %next_pos = arith.addi %scan#0, %one : i64
        %has_next = arith.cmpi slt, %next_pos, %len : i64
        %pair = arith.andi %is_cr, %has_next : i1
        %crlf = scf.if %pair -> (i1) {
          %next_index = arith.index_cast %next_pos : i64 to index
          %cp2 = func.call @__ly_unicode_get(%bytes, %width, %next_index) : (memref<?xi8>, i64, index) -> i64
          %is_nl = arith.cmpi eq, %cp2, %nl : i64
          scf.yield %is_nl : i1
        } else {
          %false_x = arith.constant false
          scf.yield %false_x : i1
        }
        %stride = arith.select %crlf, %two, %one : i64
        scf.yield %stride : i64
      } else {
        scf.yield %zero : i64
      }
      %with_break = arith.addi %scan#0, %skip : i64
      %seg_end = arith.select %keepends, %with_break, %scan#0 : i64
      %i_index = arith.index_cast %i : i64 to index
      %seg_end_index = arith.index_cast %seg_end : i64 to index
      func.call @__ly_unicode_store_slice(%list_items, %slot, %header, %bytes, %i_index, %seg_end_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      %next_i = arith.addi %scan#0, %skip : i64
      %next_slot = arith.addi %slot, %one : i64
      scf.yield %next_i, %next_slot : i64, i64
    }
    func.return %list : memref<5xi64>
  }

  func.func @LyUnicode_SplitLines(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "splitlines", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %false_bit = arith.constant false
    %result = func.call @__ly_unicode_splitlines_core(%header, %bytes, %false_bit) : (memref<2xi64>, memref<?xi8>, i1) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  func.func @LyUnicode_SplitLinesKeep(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %keepends: i1) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "splitlines", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.str"} {
    %result = func.call @__ly_unicode_splitlines_core(%header, %bytes, %keepends) : (memref<2xi64>, memref<?xi8>, i1) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // str.partition / str.rpartition: a 3-tuple of fresh strs (the separator
  // element is a retained reference to the operand, transferred to the
  // tuple).
  func.func private @__ly_unicode_partition_core(%header: memref<2xi64>, %bytes: memref<?xi8>, %sep_header: memref<2xi64>, %sep_bytes: memref<?xi8>, %reverse: i1) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %tuple = func.call @LyTuple_FromLength(%three) : (i64) -> memref<5xi64>
    %tuple_items = func.call @__ly_tuple_items(%tuple) : (memref<5xi64>) -> memref<?xi64>
    %hit = func.call @__ly_unicode_find_core(%bytes, %width, %sep_bytes, %sep_width, %zero, %len, %sep_n, %reverse) : (memref<?xi8>, i64, memref<?xi8>, i64, i64, i64, i64, i1) -> i64
    %matched = arith.cmpi sge, %hit, %zero : i64
    scf.if %matched {
      %hit_index = arith.index_cast %hit : i64 to index
      %after = arith.addi %hit, %sep_n : i64
      %after_index = arith.index_cast %after : i64 to index
      %len_index = arith.index_cast %len : i64 to index
      func.call @__ly_unicode_store_slice(%tuple_items, %zero, %header, %bytes, %c0, %hit_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
      %sep_copy:2 = func.call @__ly_unicode_retain_self(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @__ly_unicode_store_item(%tuple_items, %one, %sep_copy#0, %sep_copy#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      func.call @__ly_unicode_store_slice(%tuple_items, %two, %header, %bytes, %after_index, %len_index) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>, index, index) -> ()
    } else {
      %len_index = arith.index_cast %len : i64 to index
      %whole:2 = func.call @__ly_unicode_retain_self(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      %empty1:2 = func.call @__ly_unicode_alloc(%zero, %one) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %empty2:2 = func.call @__ly_unicode_alloc(%zero, %one) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      %whole_slot = arith.select %reverse, %two, %zero : i64
      %e1_slot = arith.select %reverse, %zero, %one : i64
      %e2_slot = arith.select %reverse, %one, %two : i64
      func.call @__ly_unicode_store_item(%tuple_items, %whole_slot, %whole#0, %whole#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      func.call @__ly_unicode_store_item(%tuple_items, %e1_slot, %empty1#0, %empty1#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      func.call @__ly_unicode_store_item(%tuple_items, %e2_slot, %empty2#0, %empty2#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
    }
    func.return %tuple : memref<5xi64>
  }

  func.func @LyUnicode_Partition(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "partition", ly.runtime.result_contract = "builtins.tuple", ly.runtime.element_contract = "builtins.str"} {
    %false_bit = arith.constant false
    %result = func.call @__ly_unicode_partition_core(%header, %bytes, %sep_header, %sep_bytes, %false_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i1) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  func.func @LyUnicode_RPartition(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "rpartition", ly.runtime.result_contract = "builtins.tuple", ly.runtime.element_contract = "builtins.str"} {
    %true_bit = arith.constant true
    %result = func.call @__ly_unicode_partition_core(%header, %bytes, %sep_header, %sep_bytes, %true_bit) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>, i1) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // Code point %i of a boxed element's code-unit buffer, addressed by the
  // raw pointer words a payload box carries (native container elements have
  // no memref views to reconstruct in manifest code).
  func.func private @__ly_unicode_get_raw(%ptr: i64, %width: i64, %i: index) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %i_i64 = arith.index_cast %i : index to i64
    %off = arith.muli %i_i64, %width : i64
    %addr = arith.addi %ptr, %off : i64
    %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %is1 = arith.cmpi eq, %width, %one : i64
    %cp = scf.if %is1 -> (i64) {
      %b = llvm.load %llptr : !llvm.ptr -> i8
      %v = arith.extui %b : i8 to i64
      scf.yield %v : i64
    } else {
      %is2 = arith.cmpi eq, %width, %two : i64
      %inner = scf.if %is2 -> (i64) {
        %h = llvm.load %llptr : !llvm.ptr -> i16
        %v = arith.extui %h : i16 to i64
        scf.yield %v : i64
      } else {
        %w = llvm.load %llptr : !llvm.ptr -> i32
        %v = arith.extui %w : i32 to i64
        scf.yield %v : i64
      }
      scf.yield %inner : i64
    }
    func.return %cp : i64
  }

  // (header ptr, code-unit ptr, byte length) words of element %slot.
  // The bytes come from the block the entity word names, the way
  // `__ly_bytes_item_words` reads a bytes element: the box holds one address.
  func.func private @__ly_unicode_item_words(%items: memref<?xi64>, %slot: index) -> (i64, i64, i64) {
    %c2 = arith.constant 0 : index
    %base = func.call @__ly_box_slot_base_index(%slot) : (index) -> index
    %hdr_slot = arith.addi %base, %c2 : index
    %hdr = memref.load %items[%hdr_slot] : memref<?xi64>
    %ptr, %blen = func.call @__ly_unicode_lane_words(%hdr) : (i64) -> (i64, i64)
    func.return %hdr, %ptr, %blen : i64, i64, i64
  }

  // Character width of a boxed str element (the width word at header+16).
  func.func private @__ly_unicode_raw_width(%hdr_ptr: i64) -> i64 {
    %shape = func.call @__ly_unicode_shape_word(%hdr_ptr) : (i64) -> i64
    %mask = arith.constant 7 : i64
    %width = arith.andi %shape, %mask : i64
    func.return %width : i64
  }

  // str.join over a runtime list/tuple of strs (identical physical layout).
  func.func @LyUnicode_Join(%sep_header: memref<2xi64> {ly.ownership.object_header}, %sep_bytes: memref<?xi8>, %n: i64, %seq_items: memref<?xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "join", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sep_n = func.call @__ly_unicode_count(%sep_header, %sep_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %sep_width = func.call @__ly_unicode_width(%sep_header) : (memref<2xi64>) -> i64
    %n_index = arith.index_cast %n : i64 to index

    // Pass 1: total code points and widest element (operands are canonical,
    // so widths bound the widest code point exactly).
    %measure:2 = scf.for %k = %c0 to %n_index step %c1 iter_args(%total = %zero, %wmax = %one) -> (i64, i64) {
      %hdr, %ptr, %blen = func.call @__ly_unicode_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %w = func.call @__ly_unicode_raw_width(%hdr) : (i64) -> i64
      %count = arith.divsi %blen, %w : i64
      %next_total = arith.addi %total, %count : i64
      %wider = arith.cmpi sgt, %w, %wmax : i64
      %next_wmax = arith.select %wider, %w, %wmax : i64
      scf.yield %next_total, %next_wmax : i64, i64
    }
    %has_seps = arith.cmpi sgt, %n, %one : i64
    %sep_uses_i64 = arith.subi %n, %one : i64
    %sep_uses = arith.select %has_seps, %sep_uses_i64, %zero : i64
    %sep_total = arith.muli %sep_uses, %sep_n : i64
    %total = arith.addi %measure#0, %sep_total : i64
    %true_j = arith.constant true
    %sep_matters = arith.andi %has_seps, %true_j : i1
    %wmax_with_sep = arith.maxsi %measure#1, %sep_width : i64
    %out_width = arith.select %sep_matters, %wmax_with_sep, %measure#1 : i64

    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%total, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)

    // Pass 2: elements with separators between them.
    scf.for %k = %c0 to %n_index step %c1 iter_args(%pos = %c0) -> (index) {
      %is_first = arith.cmpi eq, %k, %c0 : index
      %true_k = arith.constant true
      %needs_sep = arith.xori %is_first, %true_k : i1
      %after_sep = scf.if %needs_sep -> (index) {
        %sep_n_index = arith.index_cast %sep_n : i64 to index
        func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %pos, %sep_bytes, %sep_width, %c0, %sep_n_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
        %advanced = arith.addi %pos, %sep_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %hdr, %ptr, %blen = func.call @__ly_unicode_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %w = func.call @__ly_unicode_raw_width(%hdr) : (i64) -> i64
      %count = arith.divsi %blen, %w : i64
      %count_index = arith.index_cast %count : i64 to index
      // ⭐ THE ELEMENT IS COPIED AS A RUN, and reaching it takes a descriptor
      // first. A boxed element is addressed by the raw pointer its box holds,
      // and `__ly_unicode_get_raw` was the only reader for that -- one call and
      // a width branch per character. Building the view costs a few
      // instructions per ELEMENT and hands the copy to the same byte path every
      // other str copy takes.
      %elem_bytes = func.call @__ly_global_view_i8(%ptr, %blen) : (i64, i64) -> memref<?xi8>
      func.call @__ly_unicode_copy_run(%out_bytes, %out_width, %after_sep, %elem_bytes, %w, %c0, %count_index) : (memref<?xi8>, i64, index, memref<?xi8>, i64, index, index) -> ()
      %next_pos = arith.addi %after_sep, %count_index : index
      scf.yield %next_pos : index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // UTF-8 byte length of the encoded form (encode / print paths).
  func.func private @__ly_unicode_utf8_length(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %lim1 = arith.constant 128 : i64
    %lim2 = arith.constant 2048 : i64
    %lim3 = arith.constant 65536 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %total = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %zero) -> (i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %fits1 = arith.cmpi ult, %cp, %lim1 : i64
      %fits2 = arith.cmpi ult, %cp, %lim2 : i64
      %fits3 = arith.cmpi ult, %cp, %lim3 : i64
      %three_or_four = arith.select %fits3, %three, %four : i64
      %two_plus = arith.select %fits2, %two, %three_or_four : i64
      %contrib = arith.select %fits1, %one, %two_plus : i64
      %next = arith.addi %acc, %contrib : i64
      scf.yield %next : i64
    }
    func.return %total : i64
  }

  // Encode into a caller-provided UTF-8 buffer of exactly
  // __ly_unicode_utf8_length bytes.
  func.func private @__ly_unicode_utf8_fill(%header: memref<2xi64>, %bytes: memref<?xi8>, %out: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %six = arith.constant 6 : i64
    %twelve = arith.constant 12 : i64
    %eighteen = arith.constant 18 : i64
    %lim1 = arith.constant 128 : i64
    %lim2 = arith.constant 2048 : i64
    %lim3 = arith.constant 65536 : i64
    %cont_tag = arith.constant 128 : i64
    %payload_mask = arith.constant 63 : i64
    %lead2_tag = arith.constant 192 : i64
    %lead3_tag = arith.constant 224 : i64
    %lead4_tag = arith.constant 240 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %count_index step %c1 iter_args(%pos = %c0) -> (index) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %fits1 = arith.cmpi ult, %cp, %lim1 : i64
      %next = scf.if %fits1 -> (index) {
        %b0 = arith.trunci %cp : i64 to i8
        memref.store %b0, %out[%pos] : memref<?xi8>
        %advanced = arith.addi %pos, %c1 : index
        scf.yield %advanced : index
      } else {
        %fits2 = arith.cmpi ult, %cp, %lim2 : i64
        %inner2 = scf.if %fits2 -> (index) {
          %hi = arith.shrui %cp, %six : i64
          %b0v = arith.ori %hi, %lead2_tag : i64
          %lo = arith.andi %cp, %payload_mask : i64
          %b1v = arith.ori %lo, %cont_tag : i64
          %b0 = arith.trunci %b0v : i64 to i8
          %b1 = arith.trunci %b1v : i64 to i8
          %pos1 = arith.addi %pos, %c1 : index
          memref.store %b0, %out[%pos] : memref<?xi8>
          memref.store %b1, %out[%pos1] : memref<?xi8>
          %advanced = arith.addi %pos, %c2 : index
          scf.yield %advanced : index
        } else {
          %fits3 = arith.cmpi ult, %cp, %lim3 : i64
          %inner3 = scf.if %fits3 -> (index) {
            %hi = arith.shrui %cp, %twelve : i64
            %b0v = arith.ori %hi, %lead3_tag : i64
            %mid_raw = arith.shrui %cp, %six : i64
            %mid = arith.andi %mid_raw, %payload_mask : i64
            %b1v = arith.ori %mid, %cont_tag : i64
            %lo = arith.andi %cp, %payload_mask : i64
            %b2v = arith.ori %lo, %cont_tag : i64
            %b0 = arith.trunci %b0v : i64 to i8
            %b1 = arith.trunci %b1v : i64 to i8
            %b2 = arith.trunci %b2v : i64 to i8
            %pos1 = arith.addi %pos, %c1 : index
            %pos2 = arith.addi %pos, %c2 : index
            memref.store %b0, %out[%pos] : memref<?xi8>
            memref.store %b1, %out[%pos1] : memref<?xi8>
            memref.store %b2, %out[%pos2] : memref<?xi8>
            %advanced = arith.addi %pos, %c3 : index
            scf.yield %advanced : index
          } else {
            %hi = arith.shrui %cp, %eighteen : i64
            %b0v = arith.ori %hi, %lead4_tag : i64
            %mid1_raw = arith.shrui %cp, %twelve : i64
            %mid1 = arith.andi %mid1_raw, %payload_mask : i64
            %b1v = arith.ori %mid1, %cont_tag : i64
            %mid2_raw = arith.shrui %cp, %six : i64
            %mid2 = arith.andi %mid2_raw, %payload_mask : i64
            %b2v = arith.ori %mid2, %cont_tag : i64
            %lo = arith.andi %cp, %payload_mask : i64
            %b3v = arith.ori %lo, %cont_tag : i64
            %b0 = arith.trunci %b0v : i64 to i8
            %b1 = arith.trunci %b1v : i64 to i8
            %b2 = arith.trunci %b2v : i64 to i8
            %b3 = arith.trunci %b3v : i64 to i8
            %pos1 = arith.addi %pos, %c1 : index
            %pos2 = arith.addi %pos, %c2 : index
            %pos3 = arith.addi %pos, %c3 : index
            memref.store %b0, %out[%pos] : memref<?xi8>
            memref.store %b1, %out[%pos1] : memref<?xi8>
            memref.store %b2, %out[%pos2] : memref<?xi8>
            memref.store %b3, %out[%pos3] : memref<?xi8>
            %advanced = arith.addi %pos, %c4 : index
            scf.yield %advanced : index
          }
          scf.yield %inner3 : index
        }
        scf.yield %inner2 : index
      }
      scf.yield %next : index
    }
    func.return
  }

  func.func @LyUnicode_Print(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print"} {
    %stdout = arith.constant 1 : i32
    %length = func.call @__ly_unicode_utf8_length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %length_index = arith.index_cast %length : i64 to index
    %buffer = memref.alloc(%length_index) : memref<?xi8>
    func.call @__ly_unicode_utf8_fill(%header, %bytes, %buffer) : (memref<2xi64>, memref<?xi8>, memref<?xi8>) -> ()
    func.call @LyHost_WriteBytes(%stdout, %buffer, %length) : (i32, memref<?xi8>, i64) -> ()
    memref.dealloc %buffer : memref<?xi8>
    func.return
  }

  func.func @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %buffer = memref.alloca() : memref<21xi8>
    %zero = arith.constant 0 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_minus = arith.constant 45 : i8
    %end = arith.constant 20 : index
    %last = arith.constant 19 : index

    %is_negative = arith.cmpi slt, %value, %zero : i64
    %negated = arith.subi %zero, %value : i64
    %abs_value = arith.select %is_negative, %negated, %value : i64
    %is_zero = arith.cmpi eq, %abs_value, %zero : i64
    cf.cond_br %is_zero, ^format_zero, ^format_digits

  ^format_zero:
    %zero_ch_i64 = arith.constant 48 : i64
    %zero_ch = arith.trunci %zero_ch_i64 : i64 to i8
    memref.store %zero_ch, %buffer[%last] : memref<21xi8>
    cf.br ^finish(%last : index)

  ^format_digits:
    %lower = arith.constant 0 : index
    %upper = arith.constant 20 : index
    %step = arith.constant 1 : index
    %result:2 = scf.for %i = %lower to %upper step %step iter_args(%n = %abs_value, %pos = %last) -> (i64, index) {
      %active = arith.cmpi ne, %n, %zero : i64
      %next:2 = scf.if %active -> (i64, index) {
        %digit = arith.remui %n, %ten : i64
        %digit_ch_i64 = arith.addi %digit, %ascii_zero : i64
        %digit_ch = arith.trunci %digit_ch_i64 : i64 to i8
        memref.store %digit_ch, %buffer[%pos] : memref<21xi8>
        %quotient = arith.divui %n, %ten : i64
        %one_index = arith.constant 1 : index
        %next_pos = arith.subi %pos, %one_index : index
        scf.yield %quotient, %next_pos : i64, index
      } else {
        scf.yield %n, %pos : i64, index
      }
      scf.yield %next#0, %next#1 : i64, index
    }
    %one_finish = arith.constant 1 : index
    %first_digit = arith.addi %result#1, %one_finish : index
    %start = scf.if %is_negative -> (index) {
      %minus_pos = arith.subi %first_digit, %one_finish : index
      memref.store %ascii_minus, %buffer[%minus_pos] : memref<21xi8>
      scf.yield %minus_pos : index
    } else {
      scf.yield %first_digit : index
    }
    cf.br ^finish(%start : index)

  ^finish(%start_index: index):
    %length_index = arith.subi %end, %start_index : index
    %length = arith.index_cast %length_index : index to i64
    %buffer_view = memref.cast %buffer : memref<21xi8> to memref<?xi8>
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer_view, %start_index, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Format(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %spec_header: memref<2xi64> {ly.ownership.object_header}, %spec_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__format__", ly.runtime.result_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %minus_one = arith.constant -1 : i64
    %true_uf = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %s0 = arith.constant 0 : index
    %s1 = arith.constant 1 : index
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %s5 = arith.constant 5 : index
    %s6 = arith.constant 6 : index
    %s7 = arith.constant 7 : index
    %s8 = arith.constant 8 : index
    %spec_store = memref.alloca() : memref<10xi64>
    %spec = memref.cast %spec_store : memref<10xi64> to memref<?xi64>
    %ok = func.call @__ly_fmt_parse_spec(%spec_header, %spec_bytes, %spec) : (memref<2xi64>, memref<?xi8>, memref<?xi64>) -> i1
    %names = memref.get_global @__ly_fmt_msg_name_str : memref<3xi8>
    %name = memref.cast %names : memref<3xi8> to memref<?xi8>
    %nlen = arith.constant 3 : i64
    %bad = arith.xori %ok, %true_uf : i1
    scf.if %bad {
      func.call @__ly_fmt_raise_invalid_spec(%spec_header, %spec_bytes, %name, %nlen) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> ()
    }
    %fill_rec = memref.load %spec[%s0] : memref<?xi64>
    %align_rec = memref.load %spec[%s1] : memref<?xi64>
    %sign_rec = memref.load %spec[%s2] : memref<?xi64>
    %alt_rec = memref.load %spec[%s3] : memref<?xi64>
    %zero_rec = memref.load %spec[%s4] : memref<?xi64>
    %width_rec = memref.load %spec[%s5] : memref<?xi64>
    %group_rec = memref.load %spec[%s6] : memref<?xi64>
    %prec_rec = memref.load %spec[%s7] : memref<?xi64>
    %type_rec = memref.load %spec[%s8] : memref<?xi64>
    // grouping first (CPython reports the type char with it), then the
    // string-specific shape restrictions
    %grouped = arith.cmpi ne, %group_rec, %zero : i64
    scf.if %grouped {
      %cs = arith.constant 115 : i64
      %tz = arith.cmpi eq, %type_rec, %zero : i64
      %wcp = arith.select %tz, %cs, %type_rec : i64
      func.call @__ly_fmt_raise_cannot_group(%group_rec, %wcp) : (i64, i64) -> ()
    }
    %signed = arith.cmpi ne, %sign_rec, %zero : i64
    scf.if %signed {
      func.call @__ly_fmt_raise_sign_str() : () -> ()
    }
    %eq_cp = arith.constant 61 : i64
    %eq_align = arith.cmpi eq, %align_rec, %eq_cp : i64
    scf.if %eq_align {
      func.call @__ly_fmt_raise_eq_align_str() : () -> ()
    }
    %alt_on = arith.cmpi ne, %alt_rec, %zero : i64
    scf.if %alt_on {
      func.call @__ly_fmt_raise_alt_str() : () -> ()
    }
    %cs2 = arith.constant 115 : i64
    %t_ok0 = arith.cmpi eq, %type_rec, %zero : i64
    %t_ok1 = arith.cmpi eq, %type_rec, %cs2 : i64
    %t_ok = arith.ori %t_ok0, %t_ok1 : i1
    %t_bad = arith.xori %t_ok, %true_uf : i1
    scf.if %t_bad {
      func.call @__ly_fmt_raise_unknown_code(%type_rec, %name, %nlen) : (i64, memref<?xi8>, i64) -> ()
    }
    // 'z' after the type check, like CPython ('zd' reports the unknown code)
    %s9z = arith.constant 9 : index
    %z_rec = memref.load %spec[%s9z] : memref<?xi64>
    %z_on = arith.cmpi ne, %z_rec, %zero : i64
    scf.if %z_on {
      %zms = memref.get_global @__ly_fmt_msg_z_str : memref<65xi8>
      %zm = memref.cast %zms : memref<65xi8> to memref<?xi8>
      %zl = arith.constant 65 : i64
      func.call @__ly_fmt_raise_bytes(%zm, %zl) : (memref<?xi8>, i64) -> ()
    }

    %wid = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %n = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %has_prec = arith.cmpi sge, %prec_rec, %zero : i64
    %clamped = arith.minsi %n, %prec_rec : i64
    %shown = arith.select %has_prec, %clamped, %n : i64
    %width_c = arith.maxsi %width_rec, %zero : i64
    %pad_raw = arith.subi %width_c, %shown : i64
    %pad = arith.maxsi %pad_raw, %zero : i64
    %total = arith.addi %shown, %pad : i64
    %code_unit = arith.constant 4 : i64
    %nothing_before = arith.constant 0 : i64
    func.call @__ly_check_alloc_count(%total, %code_unit, %nothing_before) : (i64, i64, i64) -> ()
    %zero_flag = arith.cmpi ne, %zero_rec, %zero : i64
    %fill_unset = arith.cmpi eq, %fill_rec, %minus_one : i64
    %fill_zero = arith.constant 48 : i64
    %fill_space = arith.constant 32 : i64
    %fill_def = arith.select %zero_flag, %fill_zero, %fill_space : i64
    %fill_cp = arith.select %fill_unset, %fill_def, %fill_rec : i64
    %lt_cp = arith.constant 60 : i64
    %gt_cp = arith.constant 62 : i64
    %caret_cp = arith.constant 94 : i64
    %align_unset = arith.cmpi eq, %align_rec, %zero : i64
    %align_cp = arith.select %align_unset, %lt_cp, %align_rec : i64
    %is_gt = arith.cmpi eq, %align_cp, %gt_cp : i64
    %is_caret = arith.cmpi eq, %align_cp, %caret_cp : i64
    %half = arith.divui %pad, %two : i64
    %left0 = arith.select %is_gt, %pad, %zero : i64
    %left = arith.select %is_caret, %half, %left0 : i64
    %right = arith.subi %pad, %left : i64

    %total_idx = arith.index_cast %total : i64 to index
    %buf = memref.alloc(%total_idx) : memref<?xi32>
    %fill32 = arith.trunci %fill_cp : i64 to i32
    %left_idx = arith.index_cast %left : i64 to index
    scf.for %i = %c0 to %left_idx step %c1 {
      memref.store %fill32, %buf[%i] : memref<?xi32>
    }
    %shown_idx = arith.index_cast %shown : i64 to index
    scf.for %i = %c0 to %shown_idx step %c1 {
      %cp = func.call @__ly_unicode_get(%bytes, %wid, %i) : (memref<?xi8>, i64, index) -> i64
      %cp32 = arith.trunci %cp : i64 to i32
      %o = arith.addi %left_idx, %i : index
      memref.store %cp32, %buf[%o] : memref<?xi32>
    }
    %base2 = arith.addi %left_idx, %shown_idx : index
    %right_idx = arith.index_cast %right : i64 to index
    scf.for %i = %c0 to %right_idx step %c1 {
      %o = arith.addi %base2, %i : index
      memref.store %fill32, %buf[%o] : memref<?xi32>
    }
    %rh, %rb = func.call @__ly_fmt_str_from_cps(%buf, %total) : (memref<?xi32>, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buf : memref<?xi32>
    func.return %rh, %rb : memref<2xi64>, memref<?xi8>
  }

  // ===== impls: runtime str.format templates =====
  // A runtime (non-literal) template cannot be matched against the argument
  // list statically, so the emitter chains these scanners per argument:
  // prefix/next/conv/spec/end walk the template, pick selects the variant
  // the runtime conversion character asks for. Only auto-numbered fields
  // are accepted -- manual/named fields would need runtime argument
  // selection, which the static ABI deliberately avoids (R4).

  // "Single '}' encountered in format string"
  memref.global "private" constant @__ly_fmtrt_msg_single_rbrace : memref<39xi8> = dense<[83, 105, 110, 103, 108, 101, 32, 39, 125, 39, 32, 101, 110, 99, 111, 117, 110, 116, 101, 114, 101, 100, 32, 105, 110, 32, 102, 111, 114, 109, 97, 116, 32, 115, 116, 114, 105, 110, 103]>
  // "Single '{' encountered in format string"
  memref.global "private" constant @__ly_fmtrt_msg_single_lbrace : memref<39xi8> = dense<[83, 105, 110, 103, 108, 101, 32, 39, 123, 39, 32, 101, 110, 99, 111, 117, 110, 116, 101, 114, 101, 100, 32, 105, 110, 32, 102, 111, 114, 109, 97, 116, 32, 115, 116, 114, 105, 110, 103]>
  // "too many positional arguments for the runtime format template"
  memref.global "private" constant @__ly_fmtrt_msg_too_many_args : memref<61xi8> = dense<[116, 111, 111, 32, 109, 97, 110, 121, 32, 112, 111, 115, 105, 116, 105, 111, 110, 97, 108, 32, 97, 114, 103, 117, 109, 101, 110, 116, 115, 32, 102, 111, 114, 32, 116, 104, 101, 32, 114, 117, 110, 116, 105, 109, 101, 32, 102, 111, 114, 109, 97, 116, 32, 116, 101, 109, 112, 108, 97, 116, 101]>
  // "Replacement index out of range for positional args tuple"
  memref.global "private" constant @__ly_fmtrt_msg_index_range : memref<56xi8> = dense<[82, 101, 112, 108, 97, 99, 101, 109, 101, 110, 116, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101, 32, 102, 111, 114, 32, 112, 111, 115, 105, 116, 105, 111, 110, 97, 108, 32, 97, 114, 103, 115, 32, 116, 117, 112, 108, 101]>
  // "runtime str.format templates support only auto-numbered fields ('{}')"
  memref.global "private" constant @__ly_fmtrt_msg_auto_only : memref<69xi8> = dense<[114, 117, 110, 116, 105, 109, 101, 32, 115, 116, 114, 46, 102, 111, 114, 109, 97, 116, 32, 116, 101, 109, 112, 108, 97, 116, 101, 115, 32, 115, 117, 112, 112, 111, 114, 116, 32, 111, 110, 108, 121, 32, 97, 117, 116, 111, 45, 110, 117, 109, 98, 101, 114, 101, 100, 32, 102, 105, 101, 108, 100, 115, 32, 40, 39, 123, 125, 39, 41]>
  // "runtime str.format templates do not support nested replacement fields"
  memref.global "private" constant @__ly_fmtrt_msg_nested_spec : memref<69xi8> = dense<[114, 117, 110, 116, 105, 109, 101, 32, 115, 116, 114, 46, 102, 111, 114, 109, 97, 116, 32, 116, 101, 109, 112, 108, 97, 116, 101, 115, 32, 100, 111, 32, 110, 111, 116, 32, 115, 117, 112, 112, 111, 114, 116, 32, 110, 101, 115, 116, 101, 100, 32, 114, 101, 112, 108, 97, 99, 101, 109, 101, 110, 116, 32, 102, 105, 101, 108, 100, 115]>
  // "invalid conversion character; expected 's', 'r', or 'a'"
  memref.global "private" constant @__ly_fmtrt_msg_bad_conv : memref<55xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 99, 111, 110, 118, 101, 114, 115, 105, 111, 110, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 59, 32, 101, 120, 112, 101, 99, 116, 101, 100, 32, 39, 115, 39, 44, 32, 39, 114, 39, 44, 32, 111, 114, 32, 39, 97, 39]>

  func.func private @__ly_fmtrt_raise(%message: memref<?xi8>, %length: i64) {
    %value_error = arith.constant 53 : i64
    func.call @__ly_raise_static_message(%value_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmtrt_raise_single_rbrace() {
    %ms = memref.get_global @__ly_fmtrt_msg_single_rbrace : memref<39xi8>
    %m = memref.cast %ms : memref<39xi8> to memref<?xi8>
    %l = arith.constant 39 : i64
    func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmtrt_raise_single_lbrace() {
    %ms = memref.get_global @__ly_fmtrt_msg_single_lbrace : memref<39xi8>
    %m = memref.cast %ms : memref<39xi8> to memref<?xi8>
    %l = arith.constant 39 : i64
    func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  // Shared walk from a cursor to the next replacement field: returns the
  // field's '{' position or -1 at end. Raises on stray '}'.
  func.func private @__ly_fmtrt_scan_to_field(%header: memref<2xi64>, %bytes: memref<?xi8>, %cursor: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %minus_one = arith.constant -1 : i64
    %lb = arith.constant 123 : i64
    %rb = arith.constant 125 : i64
    %wid = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %n = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %res:2 = scf.while (%i = %cursor, %found = %minus_one) : (i64, i64) -> (i64, i64) {
      %in = arith.cmpi slt, %i, %n : i64
      %not_found = arith.cmpi eq, %found, %minus_one : i64
      %continue = arith.andi %in, %not_found : i1
      scf.condition(%continue) %i, %found : i64, i64
    } do {
    ^bb0(%i: i64, %found: i64):
      %ix = arith.index_cast %i : i64 to index
      %c = func.call @__ly_unicode_get(%bytes, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      %ip1 = arith.addi %i, %one : i64
      %has_next = arith.cmpi slt, %ip1, %n : i64
      %cn = scf.if %has_next -> (i64) {
        %ix1 = arith.index_cast %ip1 : i64 to index
        %v = func.call @__ly_unicode_get(%bytes, %wid, %ix1) : (memref<?xi8>, i64, index) -> i64
        scf.yield %v : i64
      } else {
        scf.yield %zero : i64
      }
      %is_lb = arith.cmpi eq, %c, %lb : i64
      %is_rb = arith.cmpi eq, %c, %rb : i64
      %next_lb = arith.cmpi eq, %cn, %lb : i64
      %next_rb = arith.cmpi eq, %cn, %rb : i64
      %esc_lb = arith.andi %is_lb, %next_lb : i1
      %esc_rb = arith.andi %is_rb, %next_rb : i1
      %esc = arith.ori %esc_lb, %esc_rb : i1
      %ni:2 = scf.if %esc -> (i64, i64) {
        %skip = arith.addi %i, %two : i64
        scf.yield %skip, %minus_one : i64, i64
      } else {
        %r:2 = scf.if %is_lb -> (i64, i64) {
          scf.yield %i, %i : i64, i64
        } else {
          scf.if %is_rb {
            func.call @__ly_fmtrt_raise_single_rbrace() : () -> ()
          }
          scf.yield %ip1, %minus_one : i64, i64
        }
        scf.yield %r#0, %r#1 : i64, i64
      }
      scf.yield %ni#0, %ni#1 : i64, i64
    }
    func.return %res#1 : i64
  }

  // Decoded literal text from the cursor to the next field (or the end).
  func.func private @__ly_fmtrt_literal_until_field(%header: memref<2xi64>, %bytes: memref<?xi8>, %cursor: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %minus_one = arith.constant -1 : i64
    %lb = arith.constant 123 : i64
    %rb = arith.constant 125 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %wid = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %n = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %cap0 = arith.subi %n, %cursor : i64
    %cap = arith.maxsi %cap0, %one : i64
    %cap_idx = arith.index_cast %cap : i64 to index
    %buf = memref.alloc(%cap_idx) : memref<?xi32>
    %res:3 = scf.while (%i = %cursor, %o = %zero, %stop = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
      %in = arith.cmpi slt, %i, %n : i64
      %going = arith.cmpi eq, %stop, %zero : i64
      %continue = arith.andi %in, %going : i1
      scf.condition(%continue) %i, %o, %stop : i64, i64, i64
    } do {
    ^bb0(%i: i64, %o: i64, %stop: i64):
      %ix = arith.index_cast %i : i64 to index
      %c = func.call @__ly_unicode_get(%bytes, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      %ip1 = arith.addi %i, %one : i64
      %has_next = arith.cmpi slt, %ip1, %n : i64
      %cn = scf.if %has_next -> (i64) {
        %ix1 = arith.index_cast %ip1 : i64 to index
        %v = func.call @__ly_unicode_get(%bytes, %wid, %ix1) : (memref<?xi8>, i64, index) -> i64
        scf.yield %v : i64
      } else {
        scf.yield %zero : i64
      }
      %is_lb = arith.cmpi eq, %c, %lb : i64
      %is_rb = arith.cmpi eq, %c, %rb : i64
      %next_lb = arith.cmpi eq, %cn, %lb : i64
      %next_rb = arith.cmpi eq, %cn, %rb : i64
      %esc_lb = arith.andi %is_lb, %next_lb : i1
      %esc_rb = arith.andi %is_rb, %next_rb : i1
      %esc = arith.ori %esc_lb, %esc_rb : i1
      %r:3 = scf.if %esc -> (i64, i64, i64) {
        %o_idx = arith.index_cast %o : i64 to index
        %c32 = arith.trunci %c : i64 to i32
        memref.store %c32, %buf[%o_idx] : memref<?xi32>
        %no = arith.addi %o, %one : i64
        %skip = arith.addi %i, %two : i64
        scf.yield %skip, %no, %zero : i64, i64, i64
      } else {
        %r2:3 = scf.if %is_lb -> (i64, i64, i64) {
          scf.yield %i, %o, %one : i64, i64, i64
        } else {
          scf.if %is_rb {
            func.call @__ly_fmtrt_raise_single_rbrace() : () -> ()
          }
          %o_idx = arith.index_cast %o : i64 to index
          %c32 = arith.trunci %c : i64 to i32
          memref.store %c32, %buf[%o_idx] : memref<?xi32>
          %no = arith.addi %o, %one : i64
          scf.yield %ip1, %no, %zero : i64, i64, i64
        }
        scf.yield %r2#0, %r2#1, %r2#2 : i64, i64, i64
      }
      scf.yield %r#0, %r#1, %r#2 : i64, i64, i64
    }
    %h, %b = func.call @__ly_fmt_str_from_cps(%buf, %res#1) : (memref<?xi32>, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buf : memref<?xi32>
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // Field structure at '{' position: (name_len, conv 0/1/2/3, spec_start,
  // spec_end, end-after-'}'). Raises on malformed fields.
  func.func private @__ly_fmtrt_field_parts(%header: memref<2xi64>, %bytes: memref<?xi8>, %pos: i64) -> (i64, i64, i64, i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %true_frt = arith.constant true
    %minus_one = arith.constant -1 : i64
    %lb = arith.constant 123 : i64
    %rb = arith.constant 125 : i64
    %colon_ch = arith.constant 58 : i64
    %bang_ch = arith.constant 33 : i64
    %wid = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %n = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %start = arith.addi %pos, %one : i64
    %scan:4 = scf.while (%j = %start, %depth = %one, %colon = %minus_one, %bang = %minus_one) : (i64, i64, i64, i64) -> (i64, i64, i64, i64) {
      %in = arith.cmpi slt, %j, %n : i64
      %open = arith.cmpi sgt, %depth, %zero : i64
      %continue = arith.andi %in, %open : i1
      scf.condition(%continue) %j, %depth, %colon, %bang : i64, i64, i64, i64
    } do {
    ^bb0(%j: i64, %depth: i64, %colon: i64, %bang: i64):
      %jx = arith.index_cast %j : i64 to index
      %c = func.call @__ly_unicode_get(%bytes, %wid, %jx) : (memref<?xi8>, i64, index) -> i64
      %is_lb = arith.cmpi eq, %c, %lb : i64
      %is_rb = arith.cmpi eq, %c, %rb : i64
      %inc = arith.select %is_lb, %one, %zero : i64
      %dec = arith.select %is_rb, %one, %zero : i64
      %d1 = arith.addi %depth, %inc : i64
      %nd = arith.subi %d1, %dec : i64
      %at1 = arith.cmpi eq, %depth, %one : i64
      %no_colon = arith.cmpi eq, %colon, %minus_one : i64
      %is_colon = arith.cmpi eq, %c, %colon_ch : i64
      %set_colon0 = arith.andi %at1, %no_colon : i1
      %set_colon = arith.andi %set_colon0, %is_colon : i1
      %ncolon = arith.select %set_colon, %j, %colon : i64
      %no_bang = arith.cmpi eq, %bang, %minus_one : i64
      %is_bang = arith.cmpi eq, %c, %bang_ch : i64
      %set_bang0 = arith.andi %at1, %no_bang : i1
      %set_bang1 = arith.andi %set_bang0, %no_colon : i1
      %set_bang = arith.andi %set_bang1, %is_bang : i1
      %nbang = arith.select %set_bang, %j, %bang : i64
      %closed = arith.cmpi eq, %nd, %zero : i64
      %jp1 = arith.addi %j, %one : i64
      %nj = arith.select %closed, %j, %jp1 : i64
      scf.yield %nj, %nd, %ncolon, %nbang : i64, i64, i64, i64
    }
    %unterminated = arith.cmpi ne, %scan#1, %zero : i64
    scf.if %unterminated {
      func.call @__ly_fmtrt_raise_single_lbrace() : () -> ()
    }
    %end = arith.maxsi %scan#0, %start : i64
    %has_bang = arith.cmpi ne, %scan#3, %minus_one : i64
    %has_colon = arith.cmpi ne, %scan#2, %minus_one : i64
    %colon_or_end = arith.select %has_colon, %scan#2, %end : i64
    %name_end = arith.select %has_bang, %scan#3, %colon_or_end : i64
    %name_len = arith.subi %name_end, %start : i64
    %conv = scf.if %has_bang -> (i64) {
      %conv_pos = arith.addi %scan#3, %one : i64
      %expected_end = arith.addi %scan#3, %one : i64
      %expected_end2 = arith.addi %expected_end, %one : i64
      %conv_end = arith.select %has_colon, %scan#2, %end : i64
      %well_formed = arith.cmpi eq, %conv_end, %expected_end2 : i64
      %bad_shape = arith.xori %well_formed, %true_frt : i1
      scf.if %bad_shape {
        %ms = memref.get_global @__ly_fmtrt_msg_bad_conv : memref<55xi8>
        %m = memref.cast %ms : memref<55xi8> to memref<?xi8>
        %l = arith.constant 55 : i64
        func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
      }
      %cx = arith.index_cast %conv_pos : i64 to index
      %cc = func.call @__ly_unicode_get(%bytes, %wid, %cx) : (memref<?xi8>, i64, index) -> i64
      %cr = arith.constant 114 : i64
      %cs = arith.constant 115 : i64
      %ca = arith.constant 97 : i64
      %is_r = arith.cmpi eq, %cc, %cr : i64
      %is_s = arith.cmpi eq, %cc, %cs : i64
      %is_a = arith.cmpi eq, %cc, %ca : i64
      %some0 = arith.ori %is_r, %is_s : i1
      %some = arith.ori %some0, %is_a : i1
      %bad = arith.xori %some, %true_frt : i1
      scf.if %bad {
        %ms = memref.get_global @__ly_fmtrt_msg_bad_conv : memref<55xi8>
        %m = memref.cast %ms : memref<55xi8> to memref<?xi8>
        %l = arith.constant 55 : i64
        func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
      }
      %three = arith.constant 3 : i64
      %two_c = arith.constant 2 : i64
      %v_a = arith.select %is_a, %three, %zero : i64
      %v_s = arith.select %is_s, %two_c, %v_a : i64
      %v = arith.select %is_r, %one, %v_s : i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %spec_start = scf.if %has_colon -> (i64) {
      %ss = arith.addi %scan#2, %one : i64
      scf.yield %ss : i64
    } else {
      scf.yield %end : i64
    }
    %after = arith.addi %end, %one : i64
    func.return %name_len, %conv, %spec_start, %end, %after : i64, i64, i64, i64, i64
  }

  func.func @LyUnicode_FmtNext(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %cursor: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__fmt_next__", ly.runtime.result_contract = "builtins.int"} {
    %pos = func.call @__ly_fmtrt_scan_to_field(%header, %bytes, %cursor) : (memref<2xi64>, memref<?xi8>, i64) -> i64
    %minus_one = arith.constant -1 : i64
    %missing = arith.cmpi eq, %pos, %minus_one : i64
    scf.if %missing {
      %ms = memref.get_global @__ly_fmtrt_msg_too_many_args : memref<61xi8>
      %m = memref.cast %ms : memref<61xi8> to memref<?xi8>
      %l = arith.constant 61 : i64
      func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
    }
    %h = func.call @LyLong_FromI64(%pos) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyUnicode_FmtPrefix(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %cursor: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__fmt_prefix__", ly.runtime.result_contract = "builtins.str"} {
    %h, %b = func.call @__ly_fmtrt_literal_until_field(%header, %bytes, %cursor) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_FmtTail(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %cursor: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__fmt_tail__", ly.runtime.result_contract = "builtins.str"} {
    %pos = func.call @__ly_fmtrt_scan_to_field(%header, %bytes, %cursor) : (memref<2xi64>, memref<?xi8>, i64) -> i64
    %minus_one = arith.constant -1 : i64
    %extra = arith.cmpi ne, %pos, %minus_one : i64
    scf.if %extra {
      %ms = memref.get_global @__ly_fmtrt_msg_index_range : memref<56xi8>
      %m = memref.cast %ms : memref<56xi8> to memref<?xi8>
      %l = arith.constant 56 : i64
      func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
    }
    %h, %b = func.call @__ly_fmtrt_literal_until_field(%header, %bytes, %cursor) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_FmtConv(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %pos: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__fmt_conv__", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %parts:5 = func.call @__ly_fmtrt_field_parts(%header, %bytes, %pos) : (memref<2xi64>, memref<?xi8>, i64) -> (i64, i64, i64, i64, i64)
    %named = arith.cmpi ne, %parts#0, %zero : i64
    scf.if %named {
      %ms = memref.get_global @__ly_fmtrt_msg_auto_only : memref<69xi8>
      %m = memref.cast %ms : memref<69xi8> to memref<?xi8>
      %l = arith.constant 69 : i64
      func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
    }
    %h = func.call @LyLong_FromI64(%parts#1) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyUnicode_FmtSpec(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %pos: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__fmt_spec__", ly.runtime.result_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lb = arith.constant 123 : i64
    %parts:5 = func.call @__ly_fmtrt_field_parts(%header, %bytes, %pos) : (memref<2xi64>, memref<?xi8>, i64) -> (i64, i64, i64, i64, i64)
    %wid = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %len0 = arith.subi %parts#3, %parts#2 : i64
    %len = arith.maxsi %len0, %zero : i64
    %cap = arith.maxsi %len, %one : i64
    %cap_idx = arith.index_cast %cap : i64 to index
    %buf = memref.alloc(%cap_idx) : memref<?xi32>
    %len_idx = arith.index_cast %len : i64 to index
    scf.for %i = %c0 to %len_idx step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %src = arith.addi %parts#2, %i_i64 : i64
      %sx = arith.index_cast %src : i64 to index
      %c = func.call @__ly_unicode_get(%bytes, %wid, %sx) : (memref<?xi8>, i64, index) -> i64
      %nested = arith.cmpi eq, %c, %lb : i64
      scf.if %nested {
        %ms = memref.get_global @__ly_fmtrt_msg_nested_spec : memref<69xi8>
        %m = memref.cast %ms : memref<69xi8> to memref<?xi8>
        %l = arith.constant 69 : i64
        func.call @__ly_fmtrt_raise(%m, %l) : (memref<?xi8>, i64) -> ()
      }
      %c32 = arith.trunci %c : i64 to i32
      memref.store %c32, %buf[%i] : memref<?xi32>
    }
    %h, %b = func.call @__ly_fmt_str_from_cps(%buf, %len) : (memref<?xi32>, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buf : memref<?xi32>
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_FmtEnd(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %pos: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__fmt_end__", ly.runtime.result_contract = "builtins.int"} {
    %parts:5 = func.call @__ly_fmtrt_field_parts(%header, %bytes, %pos) : (memref<2xi64>, memref<?xi8>, i64) -> (i64, i64, i64, i64, i64)
    %h = func.call @LyLong_FromI64(%parts#4) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  // Selects among the pre-formatted conversion variants (0 plain, 1 !r,
  // 2 !s, 3 !a); the emitter evaluates all four because the conversion
  // character is runtime data while dispatch must stay static.
  func.func @LyUnicode_FmtPick(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %conv: i64, %r_header: memref<2xi64> {ly.ownership.object_header}, %r_bytes: memref<?xi8>, %s_header: memref<2xi64> {ly.ownership.object_header}, %s_bytes: memref<?xi8>, %a_header: memref<2xi64> {ly.ownership.object_header}, %a_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__fmt_pick__", ly.runtime.result_contract = "builtins.str"} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %is_r = arith.cmpi eq, %conv, %one : i64
    %is_s = arith.cmpi eq, %conv, %two : i64
    %is_a = arith.cmpi eq, %conv, %three : i64
    %res:2 = scf.if %is_r -> (memref<2xi64>, memref<?xi8>) {
      %h, %b = func.call @LyUnicode_Copy(%r_header, %r_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    } else {
      %r2:2 = scf.if %is_s -> (memref<2xi64>, memref<?xi8>) {
        %h, %b = func.call @LyUnicode_Copy(%s_header, %s_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %h, %b : memref<2xi64>, memref<?xi8>
      } else {
        %r3:2 = scf.if %is_a -> (memref<2xi64>, memref<?xi8>) {
          %h, %b = func.call @LyUnicode_Copy(%a_header, %a_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          scf.yield %h, %b : memref<2xi64>, memref<?xi8>
        } else {
          %h, %b = func.call @LyUnicode_Copy(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          scf.yield %h, %b : memref<2xi64>, memref<?xi8>
        }
        scf.yield %r3#0, %r3#1 : memref<2xi64>, memref<?xi8>
      }
      scf.yield %r2#0, %r2#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %res#0, %res#1 : memref<2xi64>, memref<?xi8>
  }

  // ascii()'s escape step: ASCII code points pass through, the rest become
  // \xhh / \uhhhh / \Uhhhhhhhh (lowercase hex, like CPython). The receiver
  // is an already-repr'd string, so quoting is not re-applied here.
  func.func @LyUnicode_AsciiEscape(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__ascii__", ly.runtime.result_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %six = arith.constant 6 : i64
    %ten2 = arith.constant 10 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ascii_lim = arith.constant 128 : i64
    %xff = arith.constant 256 : i64
    %xffff = arith.constant 65536 : i64
    %wid = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %n = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %n_idx = arith.index_cast %n : i64 to index
    %out_len = scf.for %i = %c0 to %n_idx step %c1 iter_args(%acc = %zero) -> (i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %wid, %i) : (memref<?xi8>, i64, index) -> i64
      %is_ascii = arith.cmpi ult, %cp, %ascii_lim : i64
      %is_xx = arith.cmpi ult, %cp, %xff : i64
      %is_u4 = arith.cmpi ult, %cp, %xffff : i64
      %w_u4 = arith.select %is_u4, %six, %ten2 : i64
      %w_xx = arith.select %is_xx, %four, %w_u4 : i64
      %w = arith.select %is_ascii, %one, %w_xx : i64
      %nacc = arith.addi %acc, %w : i64
      scf.yield %nacc : i64
    }
    %out_idx = arith.index_cast %out_len : i64 to index
    %buf = memref.alloc(%out_idx) : memref<?xi8>
    %bs = arith.constant 92 : i8
    %x_ch = arith.constant 120 : i8
    %u_ch = arith.constant 117 : i8
    %U_ch = arith.constant 85 : i8
    %eight_ae = arith.constant 8 : i64
    %final = scf.for %i = %c0 to %n_idx step %c1 iter_args(%p = %c0) -> (index) {
      %cp = func.call @__ly_unicode_get(%bytes, %wid, %i) : (memref<?xi8>, i64, index) -> i64
      %is_ascii = arith.cmpi ult, %cp, %ascii_lim : i64
      %np = scf.if %is_ascii -> (index) {
        %b = arith.trunci %cp : i64 to i8
        memref.store %b, %buf[%p] : memref<?xi8>
        %q = arith.addi %p, %c1 : index
        scf.yield %q : index
      } else {
        %is_xx = arith.cmpi ult, %cp, %xff : i64
        %is_u4 = arith.cmpi ult, %cp, %xffff : i64
        %digits_u4 = arith.select %is_u4, %four, %eight_ae : i64
        %two_ae = arith.constant 2 : i64
        %ndigits = arith.select %is_xx, %two_ae, %digits_u4 : i64
        %esc_u4 = arith.select %is_u4, %u_ch, %U_ch : i8
        %esc = arith.select %is_xx, %x_ch, %esc_u4 : i8
        memref.store %bs, %buf[%p] : memref<?xi8>
        %q1 = arith.addi %p, %c1 : index
        memref.store %esc, %buf[%q1] : memref<?xi8>
        %q2 = arith.addi %q1, %c1 : index
        %nd_idx = arith.index_cast %ndigits : i64 to index
        %q3 = scf.for %d = %c0 to %nd_idx step %c1 iter_args(%q = %q2) -> (index) {
          %d_i64 = arith.index_cast %d : index to i64
          %rev = arith.subi %ndigits, %d_i64 : i64
          %rev_m1 = arith.subi %rev, %one : i64
          %shift_amt = arith.muli %rev_m1, %four : i64
          %shifted = arith.shrui %cp, %shift_amt : i64
          %fifteen = arith.constant 15 : i64
          %nib = arith.andi %shifted, %fifteen : i64
          %lt10 = arith.cmpi ult, %nib, %ten2 : i64
          %c48a = arith.constant 48 : i64
          %c87a = arith.constant 87 : i64
          %numc = arith.addi %nib, %c48a : i64
          %alpc = arith.addi %nib, %c87a : i64
          %ch = arith.select %lt10, %numc, %alpc : i64
          %ch8 = arith.trunci %ch : i64 to i8
          memref.store %ch8, %buf[%q] : memref<?xi8>
          %nq = arith.addi %q, %c1 : index
          scf.yield %nq : index
        }
        scf.yield %q3 : index
      }
      scf.yield %np : index
    }
    %rh, %rb = func.call @LyUnicode_FromBytes(%buf, %c0, %out_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buf : memref<?xi8>
    func.return %rh, %rb : memref<2xi64>, memref<?xi8>
  }

  // str.__hash__: SipHash-1-3 of the canonical code-unit bytes. Canonical
  // adaptive-width storage means equal strings share width and bytes, so the
  // byte hash is well-defined; the empty string hashes to 0 (CPython).
  func.func @LyUnicode_Hash(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__hash__"} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %is_empty = arith.cmpi eq, %len, %zero : i64
    %result = scf.if %is_empty -> (i64) {
      scf.yield %zero : i64
    } else {
      %ptr_index = memref.extract_aligned_pointer_as_index %bytes : memref<?xi8> -> index
      %ptr = arith.index_cast %ptr_index : index to i64
      %h = func.call @__ly_hash_bytes(%ptr, %len) : (i64, i64) -> i64
      %fixed = func.call @__ly_hash_fixup(%h) : (i64) -> i64
      scf.yield %fixed : i64
    }
    func.return %result : i64
  }

  // ===== impls: str_iterator =====
  func.func private @__ly_str_iterator_alloc(%position: i64, %length: i64) -> (memref<2xi64>, memref<2xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.str_iterator"], ly.ownership.owned_results = [0], ly.runtime.class_id = 7 : i64, ly.runtime.contract = "builtins.str_iterator", ly.runtime.primitive = "alloc"} {
    // One entity, one allocation: [0,16) header, [16,32) state view, [32,40)
    // the source str this walks -- the third lane, recorded so a box holding
    // the iterator can hand it back.
    %block_bytes = arith.constant 40 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %part_offset = arith.constant 16 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %state = memref.view %block[%part_offset][] : memref<?xi8> to memref<2xi64>
    %one = arith.constant 1 : i64
    %layout_str_iterator = arith.constant 7 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_str_iterator, %header[%layout_slot] : memref<2xi64>
    memref.store %position, %state[%position_slot] : memref<2xi64>
    memref.store %length, %state[%length_slot] : memref<2xi64>
    %zero = arith.constant 0 : i64
    %source_slot = arith.constant 4 : i64
    %header_ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %header_ptr = arith.index_cast %header_ptr_index : index to i64
    func.call @__ly_entity_word_set(%header_ptr, %source_slot, %zero) : (i64, i64, i64) -> ()
    func.return %header, %state : memref<2xi64>, memref<2xi64>
  }

  // (state, source header, source bytes) of a str iterator, from its address.
  func.func private @__ly_str_iterator_lane_words(%iter_ptr: i64) -> (i64, i64, i64, i64, i64, i64) attributes {ly.runtime.contract = "builtins.str_iterator", ly.runtime.primitive = "lane_words"} {
    %state_offset = arith.constant 16 : i64
    %two = arith.constant 2 : i64
    %source_slot = arith.constant 4 : i64
    %state_ptr = arith.addi %iter_ptr, %state_offset : i64
    %source_ptr = func.call @__ly_entity_word_get(%iter_ptr, %source_slot) : (i64, i64) -> i64
    %bytes_ptr, %byte_len = func.call @__ly_unicode_lane_words(%source_ptr) : (i64) -> (i64, i64)
    func.return %state_ptr, %two, %source_ptr, %two, %bytes_ptr, %byte_len : i64, i64, i64, i64, i64, i64
  }

  func.func @LyUnicode_Iter(%source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.str_iterator"} {
    %zero = arith.constant 0 : i64
    %length = func.call @LyUnicode_CodepointLength(%source_header, %source_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %iter_header, %state = func.call @__ly_str_iterator_alloc(%zero, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>)
    %source_slot = arith.constant 4 : i64
    %iter_ptr_index = memref.extract_aligned_pointer_as_index %iter_header : memref<2xi64> -> index
    %iter_ptr = arith.index_cast %iter_ptr_index : index to i64
    %source_ptr_index = memref.extract_aligned_pointer_as_index %source_header : memref<2xi64> -> index
    %source_ptr = arith.index_cast %source_ptr_index : index to i64
    func.call @__ly_entity_word_set(%iter_ptr, %source_slot, %source_ptr) : (i64, i64, i64) -> ()
    %source_header_view = memref.cast %source_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%source_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_Iter(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.str_iterator"} {
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_Next(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>, i1, memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str", "builtins.str_iterator"], ly.ownership.owned_results = [0, 3], ly.runtime.contract = "builtins.str_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.str", ly.runtime.next_contract = "builtins.str_iterator", ly.runtime.valid_result_index = 2 : i64} {
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    %position = memref.load %state[%position_slot] : memref<2xi64>
    %length = memref.load %state[%length_slot] : memref<2xi64>
    %valid = arith.cmpi slt, %position, %length : i64
    %one = arith.constant 1 : i64
    %next_position_candidate = arith.addi %position, %one : i64
    %next_position = arith.select %valid, %next_position_candidate, %position : i1, i64
    memref.store %next_position, %state[%position_slot] : memref<2xi64>
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()

    %element:2 = scf.if %valid -> (memref<2xi64>, memref<?xi8>) {
      %item_header, %item_bytes = func.call @LyUnicode_GetItem(%source_header, %source_bytes, %position) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %item_header, %item_bytes : memref<2xi64>, memref<?xi8>
    } else {
      %start = arith.constant 0 : index
      %zero = arith.constant 0 : i64
      %empty_header, %empty_bytes = func.call @LyUnicode_FromBytes(%source_bytes, %start, %zero) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %empty_header, %empty_bytes : memref<2xi64>, memref<?xi8>
    }

    func.return %element#0, %element#1, %valid, %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<?xi8>, i1, memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_DecRef(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %iter_header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyUnicode_DecRef(%source_header) : (memref<2xi64>) -> ()
    memref.dealloc %iter_header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
