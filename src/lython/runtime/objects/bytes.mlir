// `bytes` and its iterator -- CPython's Objects/bytesobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// Deviations from CPython:
//   - bytes.decode accepts only utf-8/strict and validates the arguments
//     eagerly (CPython's codec lookup is lazy); unknown encodings raise
//     LookupError up front.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.bytes", "builtins.bytes_iterator"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 10 : i64, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.list"}
  func.func private @LyList_Len(%self: memref<5xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__len__"}
  func.func private @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"}
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_SlotWordAsI64(%word: i64) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_as_i64"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_Encode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "encode", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"}
  func.func private @__ly_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) -> index
  func.func private @__ly_box_store_entity(%items: memref<?xi64>, %slot: i64, %class_id: i64, %entity: i64)
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_entity_word_get(%ptr: i64, %slot: i64) -> i64
  func.func private @__ly_entity_word_set(%ptr: i64, %slot: i64, %value: i64)
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_global_view_i8(%pointer: i64, %size: i64) -> memref<?xi8>
  func.func private @__ly_hash_bytes(%ptr: i64, %len: i64) -> i64
  func.func private @__ly_hash_fixup(%h: i64) -> i64
  func.func private @__ly_list_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.list", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_long_from_ascii(%bytes: memref<?xi8>) -> (memref<2xi64>, i1) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]}
  memref.global "private" constant @__ly_long_msg_invalid_int_literal_prefix : memref<40xi8>
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_repeat_overflows(%len: i64, %n: i64) -> i1
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64)
  func.func private @__ly_slice_raise_zero_step()
  func.func private @__ly_unicode_adjust_range(%len: i64, %start_raw: i64, %end_raw: i64) -> (i64, i64)
  func.func private @__ly_unicode_alloc(%count: i64, %width: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "alloc"}
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_equals_ascii(%header: memref<2xi64>, %bytes: memref<?xi8>, %expected: memref<?xi8>, %expected_len: i64) -> i1
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64
  memref.global "private" constant @__ly_unicode_msg_empty_separator : memref<15xi8>
  func.func private @__ly_unicode_put_hex(%bytes: memref<?xi8>, %width: i64, %pos: index, %value: i64, %digits: index)
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64

  py.class @bytes_iterator attributes {
    base_names = ["Iterator"],
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.bytes_iterator">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes_iterator">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @bytes attributes {
    base_names = ["Sequence", "Hashable"],
    ly.typing.base_args = [[!py.contract<"builtins.int">], []],
    ly.typing.final,
    method_names = ["__len__", "__getitem__", "__getslice__", "__add__", "__eq__", "__ne__",
                    "__bool__", "__repr__", "__str__", "decode", "decode",
                    "decode", "split", "split", "split", "find", "find", "find",
                    "count", "count", "count", "startswith", "startswith",
                    "startswith", "endswith", "endswith", "endswith", "strip",
                    "strip", "replace", "replace", "hex", "fromhex", "join",
                    "__mul__", "__hash__", "upper", "lower", "swapcase",
                    "capitalize", "title", "lstrip", "lstrip", "rstrip",
                    "rstrip", "removeprefix", "removesuffix", "isalpha",
                    "isdigit", "isalnum", "isspace", "isascii", "islower",
                    "isupper", "__contains__", "__contains__",
                    "__lt__", "__le__", "__gt__", "__ge__", "__iter__",
                    "__init__", "__init__", "__init__", "__init__",
                    "__init__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytes">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytes">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytes">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytes">>, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.protocol<"Iterable", [!py.contract<"builtins.bytes">]>] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytes_iterator">]>,
      // The four constructor spellings CPython's `bytes(...)` has, plus the
      // iterable one restricted to `list[int]` -- this port takes a list where
      // CPython takes any iterable of ints.
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytes">, !py.contract<"builtins.list", [!py.contract<"builtins.int">]>] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "classmethod", "instance", "instance",
                    "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance",
                    "instance", "instance", "instance", "instance", "instance"]
  } {}

  // ===== impls: bytes =====
  // Physical twin of str ([0,16) header + byte payload in one entity block)
  // with byte-oriented semantics: __len__/__getitem__ count BYTES (str
  // counts codepoints), __getitem__ returns int, repr spells b'...'.
  func.func private @LyBytes_Shape() -> memref<4xi64> attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.shape}

  memref.global "private" constant @__ly_bytes_msg_index_out_of_range : memref<18xi8> = dense<[105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  func.func private @__ly_bytes_raise_index_error() {
    %class_id = arith.constant 55 : i64
    %length = arith.constant 18 : i64
    %message_static = memref.get_global @__ly_bytes_msg_index_out_of_range : memref<18xi8>
    %message = memref.cast %message_static : memref<18xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // One-lane bytes entity (rfc/object-ownership-kernel.md stage 4b): the handle
  // IS the object and the payload is reached by loading its base out of word 2,
  // so a holder cannot keep a lane a reallocation left behind. Word layout is
  // bytes_abi in Passes/Runtime/ABI/StrBytesLayout.h.
  //
  // Why one block and not a separate payload allocation: the release interface
  // has a single operand, so a second allocation would need a second free the
  // deallocator has no way to name.
  // A bytes laid out at compile time in read-only data with the immortal
  // refcount, its payload address pointing into itself: a literal.
  func.func @LyBytes_FromStatic(%address: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.primitive = "from_static"} {
    %handle_bytes = arith.constant 32 : i64
    %raw = func.call @__ly_global_view_i8(%address, %handle_bytes) : (i64, i64) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %self = memref.view %raw[%c0][] {ly.ownership.object_header} : memref<?xi8> to memref<4xi64>
    func.return %self : memref<4xi64>
  }

  func.func private @__ly_bytes_alloc(%len: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.primitive = "alloc"} {
    %one_byte = arith.constant 1 : i64
    %prefix_bytes = arith.constant 32 : i64
    %byte_count = func.call @__ly_alloc_count(%len, %one_byte, %prefix_bytes) : (i64, i64, i64) -> index
    %block_prefix = arith.constant 32 : index
    %block_bytes = arith.addi %byte_count, %block_prefix : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %bytes_offset = arith.constant 32 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<4xi64>
    %bytes = memref.view %block[%bytes_offset][%byte_count] : memref<?xi8> to memref<?xi8>
    %one = arith.constant 1 : i64
    %layout_bytes = arith.constant 70 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %payload_slot = arith.constant 2 : index
    %length_slot = arith.constant 3 : index
    %payload_index = memref.extract_aligned_pointer_as_index %bytes : memref<?xi8> -> index
    %payload_word = arith.index_cast %payload_index : index to i64
    memref.store %one, %header[%refcount_slot] : memref<4xi64>
    memref.store %layout_bytes, %header[%layout_slot] : memref<4xi64>
    memref.store %payload_word, %header[%payload_slot] : memref<4xi64>
    memref.store %len, %header[%length_slot] : memref<4xi64>
    func.return %header : memref<4xi64>
  }

  // Borrowed view of the payload a bytes handle addresses. `interior_word` is
  // what makes release placement follow the handle through this call:
  // collectBoxWordDerivedViews (common/Ownership.cpp) seeds from interior-word
  // calls, and a helper without the attribute would leave the view pinning
  // nothing (rfc/lane-conversion-playbook.md step 3).
  //
  // Why not memref.view on the handle: the payload is addressed by a stored
  // word, not by an offset into the handle's own memref, so a view would
  // describe the handle's tail instead of the payload.
  func.func private @__ly_bytes_payload(%self: memref<4xi64>) -> memref<?xi8> attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.interior_word, ly.runtime.primitive = "payload_view"} {
    %payload_slot = arith.constant 2 : index
    %length_slot = arith.constant 3 : index
    %payload_word = memref.load %self[%payload_slot] : memref<4xi64>
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %view = func.call @__ly_global_view_i8(%payload_word, %length) : (i64, i64) -> memref<?xi8>
    func.return %view : memref<?xi8>
  }

  func.func @LyBytes_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"} {
    %header = func.call @__ly_bytes_alloc(%len) : (i64) -> memref<4xi64>
    %result_bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %byte_count = arith.index_cast %len : i64 to index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %index = %lower to %byte_count step %step {
      %source_index = arith.addi %start, %index : index
      %byte = memref.load %bytes[%source_index] : memref<?xi8>
      memref.store %byte, %result_bytes[%index] : memref<?xi8>
    }
    func.return %header : memref<4xi64>
  }

  // ⭐ `bytes(...)` HAD NO CALLABLE CONSTRUCTOR. Every bytes METHOD was here --
  // hex, fromhex, split, decode, the operators -- and the class itself could
  // not be called: `bytes([65, 66])` reported "builtins.bytes does not provide
  // manifest method '__init__'". `LyBytes_FromBytes` is declared `__new__` but
  // its signature is the runtime's (a raw buffer, a start and a length), which
  // no Python call can spell.
  //
  // ⛔ AN EMPTY `__init__` PER SHAPE, for the reason complex's records: the
  // constructor path calls both, and with no `__init__` of its own the MRO
  // walk reaches builtins.object's, whose input is a boxed object.
  func.func @LyBytes_NewEmpty() -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %header = func.call @__ly_bytes_alloc(%zero) : (i64) -> memref<4xi64>
    func.return %header : memref<4xi64>
  }

  func.func @LyBytes_InitEmpty(%self: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__init__"} {
    func.return
  }

  // `bytes(n)` is n zero bytes, which is CPython's buffer spelling.
  func.func @LyBytes_NewZeros(%count: i64) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %negative = arith.cmpi slt, %count, %zero : i64
    %length = arith.select %negative, %zero, %count : i64
    %one_byte = arith.constant 1 : i64
    %bytes_prefix = arith.constant 32 : i64
    func.call @__ly_check_alloc_count(%length, %one_byte, %bytes_prefix) : (i64, i64, i64) -> ()
    %header = func.call @__ly_bytes_alloc(%length) : (i64) -> memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %count_index = arith.index_cast %length : i64 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_byte = arith.constant 0 : i8
    scf.for %index = %c0 to %count_index step %c1 {
      memref.store %zero_byte, %payload[%index] : memref<?xi8>
    }
    func.return %header : memref<4xi64>
  }

  func.func @LyBytes_InitZeros(%self: memref<4xi64> {ly.ownership.object_header}, %count: i64) attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__init__"} {
    func.return
  }

  // `bytes(b)` copies, which is what CPython's constructor does for a
  // bytes-like argument.
  func.func @LyBytes_NewCopy(%source: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"} {
    %c0 = arith.constant 0 : index
    %length = func.call @LyBytes_Len(%source) : (memref<4xi64>) -> i64
    %payload = func.call @__ly_bytes_payload(%source) : (memref<4xi64>) -> memref<?xi8>
    %header = func.call @LyBytes_FromBytes(%payload, %c0, %length) : (memref<?xi8>, index, i64) -> memref<4xi64>
    func.return %header : memref<4xi64>
  }

  func.func @LyBytes_InitCopy(%self: memref<4xi64> {ly.ownership.object_header}, %source: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__init__"} {
    func.return
  }

  // `bytes(s, encoding)`. The encoding is accepted and checked the way
  // `str.encode` checks it; the payload is UTF-8 either way, which is the only
  // codec this runtime has.
  func.func @LyBytes_NewEncoded(%text_header: memref<2xi64> {ly.ownership.object_header}, %text_bytes: memref<?xi8>, %encoding_header: memref<2xi64> {ly.ownership.object_header}, %encoding_bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"} {
    %encoded = func.call @LyUnicode_Encode(%text_header, %text_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<4xi64>
    func.return %encoded : memref<4xi64>
  }

  func.func @LyBytes_InitEncoded(%self: memref<4xi64> {ly.ownership.object_header}, %text_header: memref<2xi64> {ly.ownership.object_header}, %text_bytes: memref<?xi8>, %encoding_header: memref<2xi64> {ly.ownership.object_header}, %encoding_bytes: memref<?xi8>) attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__init__"} {
    func.return
  }

  // `bytes(xs)` over a list of ints. Each slot is a box whose entity word
  // addresses the int's own header; the byte is that int's low eight bits,
  // which is what CPython stores after its 0..255 range check.
  //
  // ⛔ A LIST, where CPython takes any iterable of ints: this port has no
  // runtime iteration protocol to consume here, and a list is what the
  // spelling is written with.
  // bytes(list[int]): CPython's _PyBytes_FromList. Every element is checked
  // for range(0, 256) BEFORE the block is allocated, so the ValueError strands
  // nothing; the copy then truncates values already known to fit.
  func.func @LyBytes_NewFromList(%items: memref<5xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"} {
    %length = func.call @LyList_Len(%items) : (memref<5xi64>) -> i64
    %slots = func.call @__ly_list_items(%items) : (memref<5xi64>) -> memref<?xi64>
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %handle_words_index = arith.index_cast %handle_words : i64 to index
    %count = arith.index_cast %length : i64 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %byte_end = arith.constant 256 : i64
    %true = arith.constant true
    %all_fit = scf.for %index = %c0 to %count step %c1 iter_args(%ok = %true) -> (i1) {
      %base = arith.muli %index, %handle_words_index : index
      %entity_slot = arith.addi %base, %c0 : index
      %entity = memref.load %slots[%entity_slot] : memref<?xi64>
      %value, %fits = func.call @LyLong_SlotWordAsI64(%entity) : (i64) -> (i64, i1)
      %low = arith.cmpi sge, %value, %zero : i64
      %high = arith.cmpi slt, %value, %byte_end : i64
      %in_range = arith.andi %low, %high : i1
      %this_ok = arith.andi %fits, %in_range : i1
      %next = arith.andi %ok, %this_ok : i1
      scf.yield %next : i1
    }
    scf.if %all_fit {
    } else {
      func.call @__ly_bytes_raise_bytes_range() : () -> ()
    }
    %header = func.call @__ly_bytes_alloc(%length) : (i64) -> memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    scf.for %index = %c0 to %count step %c1 {
      %base = arith.muli %index, %handle_words_index : index
      %entity_slot = arith.addi %base, %c0 : index
      %entity = memref.load %slots[%entity_slot] : memref<?xi64>
      %value, %fits = func.call @LyLong_SlotWordAsI64(%entity) : (i64) -> (i64, i1)
      %byte = arith.trunci %value : i64 to i8
      memref.store %byte, %payload[%index] : memref<?xi8>
    }
    func.return %header : memref<4xi64>
  }

  func.func private @__ly_bytes_raise_bytes_range() {
    %message = memref.get_global @__ly_bytes_bytes_range_message : memref<30xi8>
    %message_dyn = memref.cast %message : memref<30xi8> to memref<?xi8>
    %len = arith.constant 30 : i64
    %class_id = arith.constant 53 : i64
    func.call @__ly_raise_static_message(%class_id, %message_dyn, %len) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // "bytes must be in range(0, 256)"
  memref.global "private" constant @__ly_bytes_bytes_range_message : memref<30xi8> = dense<[98, 121, 116, 101, 115, 32, 109, 117, 115, 116, 32, 98, 101, 32, 105, 110, 32, 114, 97, 110, 103, 101, 40, 48, 44, 32, 50, 53, 54, 41]>

  func.func @LyBytes_InitFromList(%self: memref<4xi64> {ly.ownership.object_header}, %items: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyBytes_Len(%header: memref<4xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__len__"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    func.return %length : i64
  }

  func.func @LyBytes_Bool(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__bool__"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %non_empty = arith.cmpi ne, %dim, %zero : index
    func.return %non_empty : i1
  }

  // bytes[i] is the byte VALUE (an int), unlike str's one-element slice.
  func.func @LyBytes_GetItem(%header: memref<4xi64> {ly.ownership.object_header}, %raw_index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__getitem__", ly.runtime.result_contract = "builtins.int"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    %is_negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %length : i64
    %index = arith.select %is_negative, %from_end, %raw_index : i1, i64
    %lower_ok = arith.cmpi sge, %index, %zero : i64
    %upper_ok = arith.cmpi slt, %index, %length : i64
    %valid = arith.andi %lower_ok, %upper_ok : i1
    %value = scf.if %valid -> (i64) {
      %at = arith.index_cast %index : i64 to index
      %byte = memref.load %bytes[%at] : memref<?xi8>
      %wide = arith.extui %byte : i8 to i64
      scf.yield %wide : i64
    } else {
      func.call @__ly_bytes_raise_index_error() : () -> ()
      scf.yield %zero : i64
    }
    %result = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // bytes[i:j:k] -- byte-indexed strided copy into a fresh bytes object.
  func.func @LyBytes_GetSlice(%header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__getslice__"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    // The throw does not return; the substitute keeps the IR division-safe.
    %step = arith.select %step_zero, %one, %step_raw : i1, i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %adj:2 = func.call @__ly_slice_adjust(%len, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %out_header = func.call @__ly_bytes_alloc(%adj#1) : (i64) -> memref<4xi64>
    %out_bytes = func.call @__ly_bytes_payload(%out_header) : (memref<4xi64>) -> memref<?xi8>
    %count_index = arith.index_cast %adj#1 : i64 to index
    scf.for %k = %c0 to %count_index step %c1 {
      %k64 = arith.index_cast %k : index to i64
      %offset = arith.muli %k64, %step : i64
      %src64 = arith.addi %adj#0, %offset : i64
      %src = arith.index_cast %src64 : i64 to index
      %byte = memref.load %bytes[%src] : memref<?xi8>
      memref.store %byte, %out_bytes[%k] : memref<?xi8>
    }
    func.return %out_header : memref<4xi64>
  }

  func.func @LyBytes_EqBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__eq__"} {
    %lhs_bytes = func.call @__ly_bytes_payload(%lhs_header) : (memref<4xi64>) -> memref<?xi8>
    %rhs_bytes = func.call @__ly_bytes_payload(%rhs_header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %lhs_ptr_index = memref.extract_aligned_pointer_as_index %lhs_bytes : memref<?xi8> -> index
    %lhs_ptr = arith.index_cast %lhs_ptr_index : index to i64
    %lhs_dim = memref.dim %lhs_bytes, %c0 : memref<?xi8>
    %lhs_len = arith.index_cast %lhs_dim : index to i64
    %rhs_ptr_index = memref.extract_aligned_pointer_as_index %rhs_bytes : memref<?xi8> -> index
    %rhs_ptr = arith.index_cast %rhs_ptr_index : index to i64
    %rhs_dim = memref.dim %rhs_bytes, %c0 : memref<?xi8>
    %rhs_len = arith.index_cast %rhs_dim : index to i64
    %equal = func.call @raw_bytes_equal(%lhs_ptr, %lhs_len, %rhs_ptr, %rhs_len) : (i64, i64, i64, i64) -> i1
    func.return %equal : i1
  }

  func.func @LyBytes_NeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__ne__"} {
    %equal = func.call @LyBytes_EqBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    %true_bit = arith.constant true
    %not_equal = arith.xori %equal, %true_bit : i1
    func.return %not_equal : i1
  }

  // bytes ordering is a lexicographic compare over unsigned bytes with the
  // shorter operand ordering first on a shared prefix -- CPython's
  // bytes_richcompare, whose memcmp is unsigned even though `char` is not.
  func.func private @__ly_bytes_compare(%lhs_header: memref<4xi64>, %rhs_header: memref<4xi64>) -> i64 {
    %lhs_bytes = func.call @__ly_bytes_payload(%lhs_header) : (memref<4xi64>) -> memref<?xi8>
    %rhs_bytes = func.call @__ly_bytes_payload(%rhs_header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %plus_one = arith.constant 1 : i64
    %lhs_dim = memref.dim %lhs_bytes, %c0 : memref<?xi8>
    %rhs_dim = memref.dim %rhs_bytes, %c0 : memref<?xi8>
    %lhs_len = arith.index_cast %lhs_dim : index to i64
    %rhs_len = arith.index_cast %rhs_dim : index to i64
    %min_dim = arith.minsi %lhs_dim, %rhs_dim : index
    %byte_cmp = scf.for %index = %c0 to %min_dim step %step iter_args(%acc = %zero) -> (i64) {
      %lhs_byte = memref.load %lhs_bytes[%index] : memref<?xi8>
      %rhs_byte = memref.load %rhs_bytes[%index] : memref<?xi8>
      %lhs_val = arith.extui %lhs_byte : i8 to i64
      %rhs_val = arith.extui %rhs_byte : i8 to i64
      %lt = arith.cmpi ult, %lhs_val, %rhs_val : i64
      %gt = arith.cmpi ugt, %lhs_val, %rhs_val : i64
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
    %prefix_equal = arith.cmpi eq, %byte_cmp, %zero : i64
    %result = arith.select %prefix_equal, %len_cmp, %byte_cmp : i64
    func.return %result : i64
  }

  func.func @LyBytes_LtBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__lt__"} {
    %cmp = func.call @__ly_bytes_compare(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi slt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyBytes_LeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__le__"} {
    %cmp = func.call @__ly_bytes_compare(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyBytes_GtBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__gt__"} {
    %cmp = func.call @__ly_bytes_compare(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sgt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyBytes_GeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__ge__"} {
    %cmp = func.call @__ly_bytes_compare(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sge, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyBytes_Concat(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__add__"} {
    %lhs_bytes = func.call @__ly_bytes_payload(%lhs_header) : (memref<4xi64>) -> memref<?xi8>
    %rhs_bytes = func.call @__ly_bytes_payload(%rhs_header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_dim = memref.dim %lhs_bytes, %c0 : memref<?xi8>
    %rhs_dim = memref.dim %rhs_bytes, %c0 : memref<?xi8>
    %total_index = arith.addi %lhs_dim, %rhs_dim : index
    %total = arith.index_cast %total_index : index to i64
    %header = func.call @__ly_bytes_alloc(%total) : (i64) -> memref<4xi64>
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    scf.for %i = %c0 to %lhs_dim step %c1 {
      %byte = memref.load %lhs_bytes[%i] : memref<?xi8>
      memref.store %byte, %bytes[%i] : memref<?xi8>
    }
    scf.for %i = %c0 to %rhs_dim step %c1 {
      %byte = memref.load %rhs_bytes[%i] : memref<?xi8>
      %dest = arith.addi %lhs_dim, %i : index
      memref.store %byte, %bytes[%dest] : memref<?xi8>
    }
    func.return %header : memref<4xi64>
  }

  // bytes_repr (Objects/bytesobject.c): b'...' with \t \n \r \' \\ kept as
  // two-character escapes, other non-printable bytes as \xHH. CPython picks
  // double quotes when the payload contains ' but not "; this port always
  // uses single quotes and escapes ' instead.
  func.func @LyBytes_Repr(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>

    %tab = arith.constant 9 : i8
    %newline = arith.constant 10 : i8
    %carriage = arith.constant 13 : i8
    %quote = arith.constant 39 : i8
    %backslash = arith.constant 92 : i8
    %space = arith.constant 32 : i8
    %tilde_plus_one = arith.constant 127 : i8

    // Pass 1: output length (b + quote + payload + quote).
    %payload_len = scf.for %i = %c0 to %dim step %c1 iter_args(%acc = %c0) -> (index) {
      %byte = memref.load %bytes[%i] : memref<?xi8>
      %is_tab = arith.cmpi eq, %byte, %tab : i8
      %is_nl = arith.cmpi eq, %byte, %newline : i8
      %is_cr = arith.cmpi eq, %byte, %carriage : i8
      %is_quote = arith.cmpi eq, %byte, %quote : i8
      %is_bs = arith.cmpi eq, %byte, %backslash : i8
      %pair_a = arith.ori %is_tab, %is_nl : i1
      %pair_b = arith.ori %pair_a, %is_cr : i1
      %pair = arith.ori %pair_b, %is_quote : i1
      %pair_full = arith.ori %pair, %is_bs : i1
      %ge_space = arith.cmpi uge, %byte, %space : i8
      %lt_del = arith.cmpi ult, %byte, %tilde_plus_one : i8
      %printable = arith.andi %ge_space, %lt_del : i1
      %escaped_two = arith.select %pair_full, %two, %four : index
      %width = arith.select %printable, %one, %escaped_two : index
      %width_final = arith.select %pair_full, %two, %width : index
      %next = arith.addi %acc, %width_final : index
      scf.yield %next : index
    }
    %prefix = arith.constant 3 : index
    %out_len_index = arith.addi %payload_len, %prefix : index
    %out = memref.alloc(%out_len_index) : memref<?xi8>

    // Pass 2: fill.
    %b_char = arith.constant 98 : i8
    memref.store %b_char, %out[%c0] : memref<?xi8>
    memref.store %quote, %out[%c1] : memref<?xi8>
    %payload_start = arith.constant 2 : index
    %end_pos = scf.for %i = %c0 to %dim step %c1 iter_args(%pos = %payload_start) -> (index) {
      %byte = memref.load %bytes[%i] : memref<?xi8>
      %is_tab = arith.cmpi eq, %byte, %tab : i8
      %is_nl = arith.cmpi eq, %byte, %newline : i8
      %is_cr = arith.cmpi eq, %byte, %carriage : i8
      %is_quote = arith.cmpi eq, %byte, %quote : i8
      %is_bs = arith.cmpi eq, %byte, %backslash : i8
      %pair_a = arith.ori %is_tab, %is_nl : i1
      %pair_b = arith.ori %pair_a, %is_cr : i1
      %pair_c = arith.ori %pair_b, %is_quote : i1
      %pair_full = arith.ori %pair_c, %is_bs : i1
      %ge_space = arith.cmpi uge, %byte, %space : i8
      %lt_del = arith.cmpi ult, %byte, %tilde_plus_one : i8
      %printable_raw = arith.andi %ge_space, %lt_del : i1
      %true_bit = arith.constant true
      %not_pair = arith.xori %pair_full, %true_bit : i1
      %printable = arith.andi %printable_raw, %not_pair : i1
      %next = scf.if %printable -> (index) {
        memref.store %byte, %out[%pos] : memref<?xi8>
        %advanced = arith.addi %pos, %one : index
        scf.yield %advanced : index
      } else {
        %escaped = scf.if %pair_full -> (index) {
          memref.store %backslash, %out[%pos] : memref<?xi8>
          %second_pos = arith.addi %pos, %one : index
          %t_char = arith.constant 116 : i8
          %n_char = arith.constant 110 : i8
          %r_char = arith.constant 114 : i8
          %escape_a = arith.select %is_tab, %t_char, %byte : i8
          %escape_b = arith.select %is_nl, %n_char, %escape_a : i8
          %escape_c = arith.select %is_cr, %r_char, %escape_b : i8
          memref.store %escape_c, %out[%second_pos] : memref<?xi8>
          %advanced = arith.addi %pos, %two : index
          scf.yield %advanced : index
        } else {
          // \xHH
          memref.store %backslash, %out[%pos] : memref<?xi8>
          %x_pos = arith.addi %pos, %one : index
          %x_char = arith.constant 120 : i8
          memref.store %x_char, %out[%x_pos] : memref<?xi8>
          %wide = arith.extui %byte : i8 to i64
          %sixteen = arith.constant 16 : i64
          %high = arith.divui %wide, %sixteen : i64
          %low = arith.remui %wide, %sixteen : i64
          %ten = arith.constant 10 : i64
          %digit_base = arith.constant 48 : i64
          %alpha_base = arith.constant 87 : i64
          %high_is_alpha = arith.cmpi uge, %high, %ten : i64
          %high_base = arith.select %high_is_alpha, %alpha_base, %digit_base : i64
          %high_char_wide = arith.addi %high, %high_base : i64
          %high_char = arith.trunci %high_char_wide : i64 to i8
          %low_is_alpha = arith.cmpi uge, %low, %ten : i64
          %low_base = arith.select %low_is_alpha, %alpha_base, %digit_base : i64
          %low_char_wide = arith.addi %low, %low_base : i64
          %low_char = arith.trunci %low_char_wide : i64 to i8
          %high_pos = arith.addi %pos, %two : index
          memref.store %high_char, %out[%high_pos] : memref<?xi8>
          %three = arith.constant 3 : index
          %low_pos = arith.addi %pos, %three : index
          memref.store %low_char, %out[%low_pos] : memref<?xi8>
          %advanced = arith.addi %pos, %four : index
          scf.yield %advanced : index
        }
        scf.yield %escaped : index
      }
      scf.yield %next : index
    }
    memref.store %quote, %out[%end_pos] : memref<?xi8>

    %out_len = arith.index_cast %out_len_index : index to i64
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%out, %c0, %out_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %out : memref<?xi8>
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytes_Str(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @LyBytes_Repr(%header) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // decode() with the default encoding: LyUnicode_FromBytes decodes the
  // UTF-8 into the adaptive-width payload and raises on invalid input.
  func.func @LyBytes_Decode(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%bytes, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // Same release interface as str (header only): the two contracts are
  // physical twins, and a wider input list here would win the
  // longest-inputTypes disambiguation and hijack str groups; the ownership
  // collector tells the twins apart by ly.runtime.result_contract instead.

  // ===== bytes methods =====

  // "unknown encoding: "
  memref.global "private" constant @__ly_bytes_msg_unknown_encoding : memref<18xi8> = dense<[117, 110, 107, 110, 111, 119, 110, 32, 101, 110, 99, 111, 100, 105, 110, 103, 58, 32]>
  // "only 'strict' error handling is supported for bytes.decode"
  memref.global "private" constant @__ly_bytes_msg_bad_errors : memref<58xi8> = dense<[111, 110, 108, 121, 32, 39, 115, 116, 114, 105, 99, 116, 39, 32, 101, 114, 114, 111, 114, 32, 104, 97, 110, 100, 108, 105, 110, 103, 32, 105, 115, 32, 115, 117, 112, 112, 111, 114, 116, 101, 100, 32, 102, 111, 114, 32, 98, 121, 116, 101, 115, 46, 100, 101, 99, 111, 100, 101]>
  // "empty separator"
  // (reuses @__ly_unicode_msg_empty_separator)
  // "subsection not found"
  memref.global "private" constant @__ly_bytes_msg_subsection_not_found : memref<20xi8> = dense<[115, 117, 98, 115, 101, 99, 116, 105, 111, 110, 32, 110, 111, 116, 32, 102, 111, 117, 110, 100]>
  // "non-hexadecimal number found in fromhex() arg at position "
  memref.global "private" constant @__ly_bytes_msg_fromhex : memref<58xi8> = dense<[110, 111, 110, 45, 104, 101, 120, 97, 100, 101, 99, 105, 109, 97, 108, 32, 110, 117, 109, 98, 101, 114, 32, 102, 111, 117, 110, 100, 32, 105, 110, 32, 102, 114, 111, 109, 104, 101, 120, 40, 41, 32, 97, 114, 103, 32, 97, 116, 32, 112, 111, 115, 105, 116, 105, 111, 110, 32]>
  // "fromhex() arg must contain an even number of hexadecimal digits" --
  // CPython's OTHER fromhex message: an odd digit count is not a bad
  // character at any position, and reporting it as one pointed past the end
  // of the string for `bytes.fromhex("abc")`.
  memref.global "private" constant @__ly_bytes_msg_fromhex_odd : memref<63xi8> = dense<[102, 114, 111, 109, 104, 101, 120, 40, 41, 32, 97, 114, 103, 32, 109, 117, 115, 116, 32, 99, 111, 110, 116, 97, 105, 110, 32, 97, 110, 32, 101, 118, 101, 110, 32, 110, 117, 109, 98, 101, 114, 32, 111, 102, 32, 104, 101, 120, 97, 100, 101, 99, 105, 109, 97, 108, 32, 100, 105, 103, 105, 116, 115]>

  // "utf-8" / "utf8" / "UTF-8" / "UTF8" (the spellings CPython sees most;
  // full codec-name normalization is out of scope until a codec registry
  // exists).
  func.func private @__ly_bytes_encoding_is_utf8(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i1 {
    %five = arith.constant 5 : i64
    %four = arith.constant 4 : i64
    %e1_ref = memref.get_global @__ly_bytes_enc_utf8_dash : memref<5xi8>
    %e1 = memref.cast %e1_ref : memref<5xi8> to memref<?xi8>
    %m1 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e1, %five) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %e2_ref = memref.get_global @__ly_bytes_enc_utf8 : memref<4xi8>
    %e2 = memref.cast %e2_ref : memref<4xi8> to memref<?xi8>
    %m2 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e2, %four) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %e3_ref = memref.get_global @__ly_bytes_enc_utf8_upper_dash : memref<5xi8>
    %e3 = memref.cast %e3_ref : memref<5xi8> to memref<?xi8>
    %m3 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e3, %five) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %e4_ref = memref.get_global @__ly_bytes_enc_utf8_upper : memref<4xi8>
    %e4 = memref.cast %e4_ref : memref<4xi8> to memref<?xi8>
    %m4 = func.call @__ly_unicode_equals_ascii(%header, %bytes, %e4, %four) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %a = arith.ori %m1, %m2 : i1
    %b = arith.ori %m3, %m4 : i1
    %result = arith.ori %a, %b : i1
    func.return %result : i1
  }
  memref.global "private" constant @__ly_bytes_enc_utf8_dash : memref<5xi8> = dense<[117, 116, 102, 45, 56]>
  memref.global "private" constant @__ly_bytes_enc_utf8 : memref<4xi8> = dense<[117, 116, 102, 56]>
  memref.global "private" constant @__ly_bytes_enc_utf8_upper_dash : memref<5xi8> = dense<[85, 84, 70, 45, 56]>
  memref.global "private" constant @__ly_bytes_enc_utf8_upper : memref<4xi8> = dense<[85, 84, 70, 56]>
  memref.global "private" constant @__ly_bytes_err_strict : memref<6xi8> = dense<[115, 116, 114, 105, 99, 116]>

  // bytes.decode(encoding): utf-8 only until a codec registry exists;
  // anything else raises LookupError like CPython's codec lookup.
  func.func @LyBytes_DecodeEncoding(%header: memref<4xi64> {ly.ownership.object_header}, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %is_utf8 = func.call @__ly_bytes_encoding_is_utf8(%enc_header, %enc_bytes) : (memref<2xi64>, memref<?xi8>) -> i1
    %true_bit = arith.constant true
    %bad = arith.xori %is_utf8, %true_bit : i1
    scf.if %bad {
      %class_id = arith.constant 60 : i64
      %length = arith.constant 18 : i64
      %start = arith.constant 0 : index
      %static = memref.get_global @__ly_bytes_msg_unknown_encoding : memref<18xi8>
      %message = memref.cast %static : memref<18xi8> to memref<?xi8>
      %prefix_h, %prefix_b = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %msg_h, %msg_b = func.call @LyUnicode_Concat(%prefix_h, %prefix_b, %enc_header, %enc_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%prefix_h) : (memref<2xi64>) -> ()
      %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %msg_h, %msg_b) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    }
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %result:2 = func.call @LyUnicode_FromBytes(%bytes, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // bytes.decode(encoding, errors): only 'strict' exists here, and it is
  // checked eagerly (CPython defers to first malformed input; deferring
  // would silently accept typos on clean input).
  func.func @LyBytes_DecodeEncodingErrors(%header: memref<4xi64> {ly.ownership.object_header}, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>, %err_header: memref<2xi64> {ly.ownership.object_header}, %err_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %six = arith.constant 6 : i64
    %strict_ref = memref.get_global @__ly_bytes_err_strict : memref<6xi8>
    %strict = memref.cast %strict_ref : memref<6xi8> to memref<?xi8>
    %is_strict = func.call @__ly_unicode_equals_ascii(%err_header, %err_bytes, %strict, %six) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> i1
    %true_bit = arith.constant true
    %bad = arith.xori %is_strict, %true_bit : i1
    scf.if %bad {
      %class_id = arith.constant 60 : i64
      %length = arith.constant 58 : i64
      %static = memref.get_global @__ly_bytes_msg_bad_errors : memref<58xi8>
      %message = memref.cast %static : memref<58xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %result:2 = func.call @LyBytes_DecodeEncoding(%header, %enc_header, %enc_bytes) : (memref<4xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_bytes_match_at(%s: memref<?xi8>, %si: index, %t: memref<?xi8>, %n: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %all = scf.for %j = %c0 to %n step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %sj = arith.addi %si, %j : index
      %a = memref.load %s[%sj] : memref<?xi8>
      %b = memref.load %t[%j] : memref<?xi8>
      %eq = arith.cmpi eq, %a, %b : i8
      %next = arith.andi %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %all : i1
  }

  func.func private @__ly_bytes_find_core(%s: memref<?xi8>, %t: memref<?xi8>, %start: i64, %end: i64, %n: i64, %reverse: i1) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %limit = arith.subi %end, %n : i64
    %viable = arith.cmpi sle, %start, %limit : i64
    %found = scf.if %viable -> (i64) {
      %n_index = arith.index_cast %n : i64 to index
      %span = arith.subi %limit, %start : i64
      %positions_i64 = arith.addi %span, %one : i64
      %positions = arith.index_cast %positions_i64 : i64 to index
      %scan = scf.for %k = %c0 to %positions step %c1 iter_args(%acc = %minus_one) -> (i64) {
        %k_i64 = arith.index_cast %k : index to i64
        %fwd = arith.addi %start, %k_i64 : i64
        %rev = arith.subi %limit, %k_i64 : i64
        %pos = arith.select %reverse, %rev, %fwd : i64
        %pos_index = arith.index_cast %pos : i64 to index
        %eq = func.call @__ly_bytes_match_at(%s, %pos_index, %t, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
        %not_yet = arith.cmpi eq, %acc, %minus_one : i64
        %take = arith.andi %eq, %not_yet : i1
        %next = arith.select %take, %pos, %acc : i64
        scf.yield %next : i64
      }
      scf.yield %scan : i64
    } else {
      scf.yield %minus_one : i64
    }
    func.return %found : i64
  }

  func.func private @__ly_bytes_len_of(%bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    func.return %len : i64
  }

  // `sub in data` and `value in data` -- both spellings CPython's
  // bytes.__contains__ takes. Declared through the Sequence base and never
  // implemented, so `b"a" in b"abc"` said so at the ABI ("declared by the
  // standard-library contract but has no runtime implementation").
  //
  // ⛔ The bytes form is `find(...) >= 0` rather than a second scan: the empty
  // sub is True for both, and the two answers cannot drift.
  func.func @LyBytes_ContainsBytes(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__contains__"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %sub_bytes = func.call @__ly_bytes_payload(%sub_header) : (memref<4xi64>) -> memref<?xi8>
    %zero = arith.constant 0 : i64
    %max = arith.constant 9223372036854775807 : i64
    %false_bit = arith.constant false
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %zero, %max) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%sub_bytes) : (memref<?xi8>) -> i64
    %found = func.call @__ly_bytes_find_core(%bytes, %sub_bytes, %start, %end, %n, %false_bit) : (memref<?xi8>, memref<?xi8>, i64, i64, i64, i1) -> i64
    %hit = arith.cmpi sge, %found, %zero : i64
    func.return %hit : i1
  }

  // ⛔ CPython RAISES for an int outside 0..255 rather than answering False --
  // `256 in b"abc"` is a ValueError, not a miss -- so the range check is part
  // of the operation and not an optimisation of it.
  func.func @LyBytes_ContainsInt(%header: memref<4xi64> {ly.ownership.object_header}, %value_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__contains__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %limit = arith.constant 256 : i64
    %false_bit = arith.constant false
    %value = func.call @LyLong_AsI64(%value_header) : (memref<2xi64>) -> i64
    %too_small = arith.cmpi slt, %value, %zero : i64
    %too_big = arith.cmpi sge, %value, %limit : i64
    %out_of_range = arith.ori %too_small, %too_big : i1
    scf.if %out_of_range {
      func.call @__ly_bytes_raise_byte_range() : () -> ()
    }
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %hit = scf.for %i = %c0 to %dim step %c1 iter_args(%seen = %false_bit) -> (i1) {
      %raw = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %raw : i8 to i64
      %same = arith.cmpi eq, %b, %value : i64
      %next = arith.ori %seen, %same : i1
      scf.yield %next : i1
    }
    func.return %hit : i1
  }

  func.func private @__ly_bytes_raise_byte_range() attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.primitive = "raise_byte_range"} {
    %c0 = arith.constant 0 : index
    %message = memref.get_global @__ly_bytes_byte_range_message : memref<29xi8>
    %message_dyn = memref.cast %message : memref<29xi8> to memref<?xi8>
    %len = arith.constant 29 : i64
    %class_id = arith.constant 53 : i64
    func.call @__ly_raise_static_message(%class_id, %message_dyn, %len) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // "byte must be in range(0, 256)"
  memref.global "private" constant @__ly_bytes_byte_range_message : memref<29xi8> = dense<[98, 121, 116, 101, 32, 109, 117, 115, 116, 32, 98, 101, 32, 105, 110, 32, 114, 97, 110, 103, 101, 40, 48, 44, 32, 50, 53, 54, 41]>

  func.func @LyBytes_Find(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "find", ly.runtime.result_contract = "builtins.int"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %sub_bytes = func.call @__ly_bytes_payload(%sub_header) : (memref<4xi64>) -> memref<?xi8>
    %false_bit = arith.constant false
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%sub_bytes) : (memref<?xi8>) -> i64
    %found = func.call @__ly_bytes_find_core(%bytes, %sub_bytes, %start, %end, %n, %false_bit) : (memref<?xi8>, memref<?xi8>, i64, i64, i64, i1) -> i64
    %result = func.call @LyLong_FromI64(%found) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func @LyBytes_CountSub(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "count", ly.runtime.result_contract = "builtins.int"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %sub_bytes = func.call @__ly_bytes_payload(%sub_header) : (memref<4xi64>) -> memref<?xi8>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%sub_bytes) : (memref<?xi8>) -> i64
    %is_empty = arith.cmpi eq, %n, %zero : i64
    %total = scf.if %is_empty -> (i64) {
      %span = arith.subi %end, %start : i64
      %viable = arith.cmpi sge, %span, %zero : i64
      %hits = arith.addi %span, %one : i64
      %count = arith.select %viable, %hits, %zero : i64
      scf.yield %count : i64
    } else {
      %n_index = arith.index_cast %n : i64 to index
      %scan:2 = scf.while (%pos = %start, %count = %zero) : (i64, i64) -> (i64, i64) {
        %tail = arith.addi %pos, %n : i64
        %more = arith.cmpi sle, %tail, %end : i64
        scf.condition(%more) %pos, %count : i64, i64
      } do {
      ^bb0(%pos: i64, %count: i64):
        %pos_index = arith.index_cast %pos : i64 to index
        %eq = func.call @__ly_bytes_match_at(%bytes, %pos_index, %sub_bytes, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
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

  func.func @LyBytes_StartsWith(%header: memref<4xi64> {ly.ownership.object_header}, %prefix_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "startswith"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %prefix_bytes = func.call @__ly_bytes_payload(%prefix_header) : (memref<4xi64>) -> memref<?xi8>
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%prefix_bytes) : (memref<?xi8>) -> i64
    %tail = arith.addi %start, %n : i64
    %fits = arith.cmpi sle, %tail, %end : i64
    %result = scf.if %fits -> (i1) {
      %si = arith.index_cast %start : i64 to index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_bytes_match_at(%bytes, %si, %prefix_bytes, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  func.func @LyBytes_EndsWith(%header: memref<4xi64> {ly.ownership.object_header}, %suffix_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "endswith"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %suffix_bytes = func.call @__ly_bytes_payload(%suffix_header) : (memref<4xi64>) -> memref<?xi8>
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %start, %end = func.call @__ly_unicode_adjust_range(%len, %start_raw, %end_raw) : (i64, i64, i64) -> (i64, i64)
    %n = func.call @__ly_bytes_len_of(%suffix_bytes) : (memref<?xi8>) -> i64
    %pos = arith.subi %end, %n : i64
    %fits = arith.cmpi sge, %pos, %start : i64
    %result = scf.if %fits -> (i1) {
      %si = arith.index_cast %pos : i64 to index
      %n_index = arith.index_cast %n : i64 to index
      %eq = func.call @__ly_bytes_match_at(%bytes, %si, %suffix_bytes, %n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    func.return %result : i1
  }

  func.func private @__ly_bytes_slice(%bytes: memref<?xi8>, %start: index, %end: index) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0]} {
    %span = arith.subi %end, %start : index
    %len = arith.index_cast %span : index to i64
    %result = func.call @LyBytes_FromBytes(%bytes, %start, %len) : (memref<?xi8>, index, i64) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  func.func private @__ly_bytes_byte_is_ascii_space(%b: i64) -> i1 {
    %space = arith.constant 32 : i64
    %tab = arith.constant 9 : i64
    %cr = arith.constant 13 : i64
    %is_space = arith.cmpi eq, %b, %space : i64
    %ge_tab = arith.cmpi sge, %b, %tab : i64
    %le_cr = arith.cmpi sle, %b, %cr : i64
    %ctl = arith.andi %ge_tab, %le_cr : i1
    %result = arith.ori %is_space, %ctl : i1
    func.return %result : i1
  }

  func.func private @__ly_bytes_byte_in(%b: i64, %chars: memref<?xi8>, %n: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false_bit = arith.constant false
    %found = scf.for %j = %c0 to %n step %c1 iter_args(%acc = %false_bit) -> (i1) {
      %cj_i8 = memref.load %chars[%j] : memref<?xi8>
      %cj = arith.extui %cj_i8 : i8 to i64
      %eq = arith.cmpi eq, %b, %cj : i64
      %next = arith.ori %acc, %eq : i1
      scf.yield %next : i1
    }
    func.return %found : i1
  }

  func.func private @__ly_bytes_strip_core(%bytes: memref<?xi8>, %mode: i64, %use_chars: i1, %chars: memref<?xi8>, %chars_n: index) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %true_bit = arith.constant true
    %dim = memref.dim %bytes, %c0 : memref<?xi8>

    %left_mask = arith.andi %mode, %one : i64
    %strip_left = arith.cmpi ne, %left_mask, %zero : i64
    %begin = scf.if %strip_left -> (index) {
      %scan:2 = scf.while (%i = %c0, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ult, %i, %dim : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %b_i8 = memref.load %bytes[%i] : memref<?xi8>
        %b = arith.extui %b_i8 : i8 to i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_bytes_byte_in(%b, %chars, %chars_n) : (i64, memref<?xi8>, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
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
      %scan:2 = scf.while (%i = %dim, %go = %true_bit) : (index, i1) -> (index, i1) {
        %more = arith.cmpi ugt, %i, %begin : index
        %continue = arith.andi %more, %go : i1
        scf.condition(%continue) %i, %go : index, i1
      } do {
      ^bb0(%i: index, %go: i1):
        %prev = arith.subi %i, %c1 : index
        %b_i8 = memref.load %bytes[%prev] : memref<?xi8>
        %b = arith.extui %b_i8 : i8 to i64
        %stripped = scf.if %use_chars -> (i1) {
          %in = func.call @__ly_bytes_byte_in(%b, %chars, %chars_n) : (i64, memref<?xi8>, index) -> i1
          scf.yield %in : i1
        } else {
          %sp = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
          scf.yield %sp : i1
        }
        %keep = arith.select %stripped, %prev, %i : index
        scf.yield %keep, %stripped : index, i1
      }
      scf.yield %scan#0 : index
    } else {
      scf.yield %dim : index
    }

    %result = func.call @__ly_bytes_slice(%bytes, %begin, %finish) : (memref<?xi8>, index, index) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  // ASCII byte classes, shared by the case maps and the is* predicates below.
  // Bytes are ASCII-only for these methods in CPython too: `b"\xc3\xa9".upper()`
  // leaves the high bytes alone, which is what a byte-wise map does.
  func.func private @__ly_bytes_byte_is_upper(%b: i64) -> i1 {
    %lo = arith.constant 65 : i64
    %hi = arith.constant 90 : i64
    %ge = arith.cmpi sge, %b, %lo : i64
    %le = arith.cmpi sle, %b, %hi : i64
    %in = arith.andi %ge, %le : i1
    func.return %in : i1
  }

  func.func private @__ly_bytes_byte_is_lower(%b: i64) -> i1 {
    %lo = arith.constant 97 : i64
    %hi = arith.constant 122 : i64
    %ge = arith.cmpi sge, %b, %lo : i64
    %le = arith.cmpi sle, %b, %hi : i64
    %in = arith.andi %ge, %le : i1
    func.return %in : i1
  }

  func.func private @__ly_bytes_byte_is_digit(%b: i64) -> i1 {
    %lo = arith.constant 48 : i64
    %hi = arith.constant 57 : i64
    %ge = arith.cmpi sge, %b, %lo : i64
    %le = arith.cmpi sle, %b, %hi : i64
    %in = arith.andi %ge, %le : i1
    func.return %in : i1
  }

  // Case map over the payload. %mode: 0 upper, 1 lower, 2 swapcase,
  // 3 capitalize, 4 title. capitalize and title need the position, which is
  // why one loop carries "the previous byte was a letter" rather than five
  // separate walks.
  func.func private @__ly_bytes_case_map(%bytes: memref<?xi8>, %mode: i64) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.primitive = "case_map", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %shift = arith.constant 32 : i64
    %false_bit = arith.constant false
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %out = func.call @__ly_bytes_alloc(%len) : (i64) -> memref<4xi64>
    %out_bytes = func.call @__ly_bytes_payload(%out) : (memref<4xi64>) -> memref<?xi8>
    %final = scf.for %i = %c0 to %dim step %c1 iter_args(%prev_alpha = %false_bit) -> (i1) {
      %raw = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %raw : i8 to i64
      %is_up = func.call @__ly_bytes_byte_is_upper(%b) : (i64) -> i1
      %is_low = func.call @__ly_bytes_byte_is_lower(%b) : (i64) -> i1
      %is_alpha = arith.ori %is_up, %is_low : i1
      %arith_true = arith.constant true
      %first = arith.cmpi eq, %i, %c0 : index
      // Which direction this position wants: upper, lower, or leave alone.
      %want_upper_mode0 = arith.cmpi eq, %mode, %zero : i64
      %want_lower_mode1 = arith.cmpi eq, %mode, %one : i64
      %is_swap = arith.cmpi eq, %mode, %two : i64
      %is_cap = arith.cmpi eq, %mode, %three : i64
      %is_title = arith.cmpi eq, %mode, %four : i64
      %cap_upper = arith.andi %is_cap, %first : i1
      %not_first = arith.xori %first, %arith_true : i1
      %cap_lower = arith.andi %is_cap, %not_first : i1
      %title_lower = arith.andi %is_title, %prev_alpha : i1
      %prev_not_alpha = arith.xori %prev_alpha, %arith_true : i1
      %title_upper = arith.andi %is_title, %prev_not_alpha : i1
      %swap_upper = arith.andi %is_swap, %is_low : i1
      %swap_lower = arith.andi %is_swap, %is_up : i1
      %up_a = arith.ori %want_upper_mode0, %cap_upper : i1
      %up_b = arith.ori %up_a, %title_upper : i1
      %to_upper = arith.ori %up_b, %swap_upper : i1
      %low_a = arith.ori %want_lower_mode1, %cap_lower : i1
      %low_b = arith.ori %low_a, %title_lower : i1
      %to_lower = arith.ori %low_b, %swap_lower : i1
      %do_upper = arith.andi %to_upper, %is_low : i1
      %do_lower = arith.andi %to_lower, %is_up : i1
      %lowered = arith.addi %b, %shift : i64
      %uppered = arith.subi %b, %shift : i64
      %after_up = arith.select %do_upper, %uppered, %b : i64
      %mapped = arith.select %do_lower, %lowered, %after_up : i64
      %out_byte = arith.trunci %mapped : i64 to i8
      memref.store %out_byte, %out_bytes[%i] : memref<?xi8>
      scf.yield %is_alpha : i1
    }
    func.return %out : memref<4xi64>
  }

  // All-bytes predicate. %kind: 0 alpha, 1 digit, 2 alnum, 3 space, 4 ascii,
  // 5 lower, 6 upper. CPython answers False on an empty bytes for every one of
  // these except isascii, and islower/isupper additionally need at least one
  // cased byte -- both are the "any cased seen" half of the fold.
  func.func private @__ly_bytes_class_all(%bytes: memref<?xi8>, %kind: i64) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.primitive = "class_all"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %ascii_limit = arith.constant 128 : i64
    %true_bit = arith.constant true
    %false_bit = arith.constant false
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %walk:2 = scf.for %i = %c0 to %dim step %c1 iter_args(%ok = %true_bit, %cased = %false_bit) -> (i1, i1) {
      %raw = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %raw : i8 to i64
      %is_up = func.call @__ly_bytes_byte_is_upper(%b) : (i64) -> i1
      %is_low = func.call @__ly_bytes_byte_is_lower(%b) : (i64) -> i1
      %is_digit = func.call @__ly_bytes_byte_is_digit(%b) : (i64) -> i1
      %is_space = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
      %is_alpha = arith.ori %is_up, %is_low : i1
      %is_alnum = arith.ori %is_alpha, %is_digit : i1
      %is_ascii = arith.cmpi slt, %b, %ascii_limit : i64
      %not_up = arith.xori %is_up, %true_bit : i1
      %not_low = arith.xori %is_low, %true_bit : i1
      %k0 = arith.cmpi eq, %kind, %zero : i64
      %k1 = arith.cmpi eq, %kind, %one : i64
      %k2 = arith.cmpi eq, %kind, %two : i64
      %k3 = arith.cmpi eq, %kind, %three : i64
      %k4 = arith.cmpi eq, %kind, %four : i64
      %k5 = arith.cmpi eq, %kind, %five : i64
      %sel0 = arith.select %k0, %is_alpha, %true_bit : i1
      %sel1 = arith.select %k1, %is_digit, %sel0 : i1
      %sel2 = arith.select %k2, %is_alnum, %sel1 : i1
      %sel3 = arith.select %k3, %is_space, %sel2 : i1
      %sel4 = arith.select %k4, %is_ascii, %sel3 : i1
      %sel5 = arith.select %k5, %not_up, %sel4 : i1
      %k6 = arith.cmpi eq, %kind, %six : i64
      %this_ok = arith.select %k6, %not_low, %sel5 : i1
      %next_ok = arith.andi %ok, %this_ok : i1
      %next_cased = arith.ori %cased, %is_alpha : i1
      scf.yield %next_ok, %next_cased : i1, i1
    }
    %empty = arith.cmpi eq, %dim, %c0 : index
    %kind_ascii = arith.cmpi eq, %kind, %four : i64
    %kind_lower = arith.cmpi eq, %kind, %five : i64
    %kind_upper = arith.cmpi eq, %kind, %six : i64
    %kind_cased = arith.ori %kind_lower, %kind_upper : i1
    // isascii is the one that is True for an empty bytes.
    %not_empty = arith.xori %empty, %true_bit : i1
    %nonempty_or_ascii = arith.ori %not_empty, %kind_ascii : i1
    %base = arith.andi %walk#0, %nonempty_or_ascii : i1
    %cased_ok = arith.ori %walk#1, %kind_ascii : i1
    %needs_cased = arith.select %kind_cased, %walk#1, %true_bit : i1
    %answer = arith.andi %base, %needs_cased : i1
    func.return %answer : i1
  }

  func.func @LyBytes_Upper(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "upper", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 0 : i64
    %out = func.call @__ly_bytes_case_map(%bytes, %mode) : (memref<?xi8>, i64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyBytes_Lower(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "lower", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 1 : i64
    %out = func.call @__ly_bytes_case_map(%bytes, %mode) : (memref<?xi8>, i64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyBytes_SwapCase(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "swapcase", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 2 : i64
    %out = func.call @__ly_bytes_case_map(%bytes, %mode) : (memref<?xi8>, i64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyBytes_Capitalize(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "capitalize", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 3 : i64
    %out = func.call @__ly_bytes_case_map(%bytes, %mode) : (memref<?xi8>, i64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyBytes_Title(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "title", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 4 : i64
    %out = func.call @__ly_bytes_case_map(%bytes, %mode) : (memref<?xi8>, i64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyBytes_LStrip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 1 : i64
    %false_bit = arith.constant false
    %c0 = arith.constant 0 : index
    %result = func.call @__ly_bytes_strip_core(%bytes, %mode, %false_bit, %bytes, %c0) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  func.func @LyBytes_LStripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %chars_bytes = func.call @__ly_bytes_payload(%chars_header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 1 : i64
    %true_bit = arith.constant true
    %c0 = arith.constant 0 : index
    %dim = memref.dim %chars_bytes, %c0 : memref<?xi8>
    %result = func.call @__ly_bytes_strip_core(%bytes, %mode, %true_bit, %chars_bytes, %dim) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  func.func @LyBytes_RStrip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 2 : i64
    %false_bit = arith.constant false
    %c0 = arith.constant 0 : index
    %result = func.call @__ly_bytes_strip_core(%bytes, %mode, %false_bit, %bytes, %c0) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  func.func @LyBytes_RStripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %chars_bytes = func.call @__ly_bytes_payload(%chars_header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 2 : i64
    %true_bit = arith.constant true
    %c0 = arith.constant 0 : index
    %dim = memref.dim %chars_bytes, %c0 : memref<?xi8>
    %result = func.call @__ly_bytes_strip_core(%bytes, %mode, %true_bit, %chars_bytes, %dim) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  // removeprefix/removesuffix: the affix test is the existing startswith /
  // endswith scan, and the answer is either a copy of the tail or a copy of
  // the whole -- never the receiver itself, because bytes handles are values
  // here and returning the argument would alias it.
  func.func @LyBytes_RemovePrefix(%header: memref<4xi64> {ly.ownership.object_header}, %affix_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "removeprefix", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %affix = func.call @__ly_bytes_payload(%affix_header) : (memref<4xi64>) -> memref<?xi8>
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %affix_len = func.call @__ly_bytes_len_of(%affix) : (memref<?xi8>) -> i64
    %affix_index = arith.index_cast %affix_len : i64 to index
    %fits = arith.cmpi sle, %affix_len, %len : i64
    %scan = scf.if %fits -> (i1) {
      %eq = func.call @__ly_bytes_match_at(%bytes, %c0, %affix, %affix_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    %kept = arith.subi %len, %affix_len : i64
    %start = arith.select %scan, %affix_index, %c0 : index
    %out_len = arith.select %scan, %kept, %len : i64
    %out = func.call @LyBytes_FromBytes(%bytes, %start, %out_len) : (memref<?xi8>, index, i64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyBytes_RemoveSuffix(%header: memref<4xi64> {ly.ownership.object_header}, %affix_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "removesuffix", ly.runtime.result_contract = "builtins.bytes"} {
    %c0 = arith.constant 0 : index
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %affix = func.call @__ly_bytes_payload(%affix_header) : (memref<4xi64>) -> memref<?xi8>
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %affix_len = func.call @__ly_bytes_len_of(%affix) : (memref<?xi8>) -> i64
    %zero_i64 = arith.constant 0 : i64
    %tail_start = arith.subi %len, %affix_len : i64
    %fits = arith.cmpi sge, %tail_start, %zero_i64 : i64
    %affix_index = arith.index_cast %affix_len : i64 to index
    %hit = scf.if %fits -> (i1) {
      %tail_index = arith.index_cast %tail_start : i64 to index
      %eq = func.call @__ly_bytes_match_at(%bytes, %tail_index, %affix, %affix_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
      scf.yield %eq : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    %out_len = arith.select %hit, %tail_start, %len : i64
    %out = func.call @LyBytes_FromBytes(%bytes, %c0, %out_len) : (memref<?xi8>, index, i64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyBytes_IsAlpha(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isalpha"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %kind = arith.constant 0 : i64
    %answer = func.call @__ly_bytes_class_all(%bytes, %kind) : (memref<?xi8>, i64) -> i1
    func.return %answer : i1
  }

  func.func @LyBytes_IsDigit(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isdigit"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %kind = arith.constant 1 : i64
    %answer = func.call @__ly_bytes_class_all(%bytes, %kind) : (memref<?xi8>, i64) -> i1
    func.return %answer : i1
  }

  func.func @LyBytes_IsAlnum(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isalnum"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %kind = arith.constant 2 : i64
    %answer = func.call @__ly_bytes_class_all(%bytes, %kind) : (memref<?xi8>, i64) -> i1
    func.return %answer : i1
  }

  func.func @LyBytes_IsSpace(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isspace"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %kind = arith.constant 3 : i64
    %answer = func.call @__ly_bytes_class_all(%bytes, %kind) : (memref<?xi8>, i64) -> i1
    func.return %answer : i1
  }

  func.func @LyBytes_IsAscii(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isascii"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %kind = arith.constant 4 : i64
    %answer = func.call @__ly_bytes_class_all(%bytes, %kind) : (memref<?xi8>, i64) -> i1
    func.return %answer : i1
  }

  func.func @LyBytes_IsLower(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "islower"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %kind = arith.constant 5 : i64
    %answer = func.call @__ly_bytes_class_all(%bytes, %kind) : (memref<?xi8>, i64) -> i1
    func.return %answer : i1
  }

  func.func @LyBytes_IsUpper(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isupper"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %kind = arith.constant 6 : i64
    %answer = func.call @__ly_bytes_class_all(%bytes, %kind) : (memref<?xi8>, i64) -> i1
    func.return %answer : i1
  }

  func.func @LyBytes_Strip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 3 : i64
    %false_bit = arith.constant false
    %c0 = arith.constant 0 : index
    %result = func.call @__ly_bytes_strip_core(%bytes, %mode, %false_bit, %bytes, %c0) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  func.func @LyBytes_StripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %chars_bytes = func.call @__ly_bytes_payload(%chars_header) : (memref<4xi64>) -> memref<?xi8>
    %mode = arith.constant 3 : i64
    %true_bit = arith.constant true
    %c0 = arith.constant 0 : index
    %dim = memref.dim %chars_bytes, %c0 : memref<?xi8>
    %result = func.call @__ly_bytes_strip_core(%bytes, %mode, %true_bit, %chars_bytes, %dim) : (memref<?xi8>, i64, i1, memref<?xi8>, index) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  func.func @LyBytes_Replace(%header: memref<4xi64> {ly.ownership.object_header}, %old_header: memref<4xi64> {ly.ownership.object_header}, %new_header: memref<4xi64> {ly.ownership.object_header}, %limit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "replace", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %old_bytes = func.call @__ly_bytes_payload(%old_header) : (memref<4xi64>) -> memref<?xi8>
    %new_bytes = func.call @__ly_bytes_payload(%new_header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %true_bit = arith.constant true
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %old_n = func.call @__ly_bytes_len_of(%old_bytes) : (memref<?xi8>) -> i64
    %new_n = func.call @__ly_bytes_len_of(%new_bytes) : (memref<?xi8>) -> i64
    %old_n_index = arith.index_cast %old_n : i64 to index
    %new_n_index = arith.index_cast %new_n : i64 to index
    %old_empty = arith.cmpi eq, %old_n, %zero : i64
    %bound = scf.if %old_empty -> (i64) {
      %plus = arith.addi %len, %one : i64
      scf.yield %plus : i64
    } else {
      scf.yield %len : i64
    }

    %measure:3 = scf.while (%i = %zero, %rem = %limit, %total = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
      %more = arith.cmpi slt, %i, %bound : i64
      scf.condition(%more) %i, %rem, %total : i64, i64, i64
    } do {
    ^bb0(%i: i64, %rem: i64, %total: i64):
      %has_budget = arith.cmpi ne, %rem, %zero : i64
      %tail = arith.addi %i, %old_n : i64
      %in_range = arith.cmpi sle, %tail, %len : i64
      %viable = arith.andi %has_budget, %in_range : i1
      %matched = scf.if %viable -> (i1) {
        %i_index = arith.index_cast %i : i64 to index
        %eq = func.call @__ly_bytes_match_at(%bytes, %i_index, %old_bytes, %old_n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %not_skip = arith.xori %skip_char, %true_bit : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %new_contrib = arith.select %matched, %new_n, %zero : i64
      %char_contrib = arith.select %emit_char, %one, %zero : i64
      %next_total_a = arith.addi %total, %new_contrib : i64
      %next_total = arith.addi %next_total_a, %char_contrib : i64
      %stride = arith.select %skip_char, %old_n, %one : i64
      %next_i = arith.addi %i, %stride : i64
      %dec = arith.select %matched, %one, %zero : i64
      %next_rem = arith.subi %rem, %dec : i64
      scf.yield %next_i, %next_rem, %next_total : i64, i64, i64
    }

    %out_header = func.call @__ly_bytes_alloc(%measure#2) : (i64) -> memref<4xi64>
    %out_bytes = func.call @__ly_bytes_payload(%out_header) : (memref<4xi64>) -> memref<?xi8>

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
        %eq = func.call @__ly_bytes_match_at(%bytes, %i_index, %old_bytes, %old_n_index) : (memref<?xi8>, index, memref<?xi8>, index) -> i1
        scf.yield %eq : i1
      } else {
        %false_bit = arith.constant false
        scf.yield %false_bit : i1
      }
      %after_new = scf.if %matched -> (index) {
        scf.for %j = %c0 to %new_n_index step %c1 {
          %b = memref.load %new_bytes[%j] : memref<?xi8>
          %dst = arith.addi %pos, %j : index
          memref.store %b, %out_bytes[%dst] : memref<?xi8>
        }
        %advanced = arith.addi %pos, %new_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %old_nonempty = arith.cmpi sgt, %old_n, %zero : i64
      %skip_char = arith.andi %matched, %old_nonempty : i1
      %in_str = arith.cmpi slt, %i, %len : i64
      %not_skip = arith.xori %skip_char, %true_bit : i1
      %emit_char = arith.andi %in_str, %not_skip : i1
      %after_char = scf.if %emit_char -> (index) {
        %i_index = arith.index_cast %i : i64 to index
        %b = memref.load %bytes[%i_index] : memref<?xi8>
        memref.store %b, %out_bytes[%after_new] : memref<?xi8>
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
    func.return %out_header : memref<4xi64>
  }

  // Box packing for bytes container elements (class id 70), the bytes
  // sibling of __ly_unicode_store_item.
  func.func private @__ly_bytes_store_item(%items: memref<?xi64>, %slot: i64, %eh: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.transfer_args = [2]} {
    %one = arith.constant 1 : i64
    %bytes_class = arith.constant 70 : i64
    %hdr_ptr_index = memref.extract_aligned_pointer_as_index %eh : memref<4xi64> -> index
    %hdr_ptr = arith.index_cast %hdr_ptr_index : index to i64
    func.call @__ly_box_store_entity(%items, %slot, %bytes_class, %hdr_ptr) : (memref<?xi64>, i64, i64, i64) -> ()
    func.return
  }

  // bytes.split(sep[, maxsplit]).
  func.func @LyBytes_Split(%header: memref<4xi64> {ly.ownership.object_header}, %sep_header: memref<4xi64> {ly.ownership.object_header}, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %sep_bytes = func.call @__ly_bytes_payload(%sep_header) : (memref<4xi64>) -> memref<?xi8>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %false_bit = arith.constant false
    %true_a = arith.constant true
    %len = func.call @__ly_bytes_len_of(%bytes) : (memref<?xi8>) -> i64
    %sep_n = func.call @__ly_bytes_len_of(%sep_bytes) : (memref<?xi8>) -> i64
    %sep_empty = arith.cmpi eq, %sep_n, %zero : i64
    scf.if %sep_empty {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 15 : i64
      %static = memref.get_global @__ly_unicode_msg_empty_separator : memref<15xi8>
      %message = memref.cast %static : memref<15xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }

    %count:3 = scf.while (%pos = %zero, %used = %zero, %go = %true_a) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %used, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %used: i64, %go: i1):
      %budget_left = arith.cmpi ne, %used, %maxsplit : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_bytes_find_core(%bytes, %sep_bytes, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, memref<?xi8>, i64, i64, i64, i1) -> i64
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

    %true_b = arith.constant true
    %emit:3 = scf.while (%pos = %zero, %slot = %zero, %go = %true_b) : (i64, i64, i1) -> (i64, i64, i1) {
      scf.condition(%go) %pos, %slot, %go : i64, i64, i1
    } do {
    ^bb0(%pos: i64, %slot: i64, %go: i1):
      %remaining = arith.subi %segments, %one : i64
      %budget_left = arith.cmpi slt, %slot, %remaining : i64
      %hit = scf.if %budget_left -> (i64) {
        %found = func.call @__ly_bytes_find_core(%bytes, %sep_bytes, %pos, %len, %sep_n, %false_bit) : (memref<?xi8>, memref<?xi8>, i64, i64, i64, i1) -> i64
        scf.yield %found : i64
      } else {
        scf.yield %minus_one : i64
      }
      %matched = arith.cmpi sge, %hit, %zero : i64
      scf.if %matched {
        %pos_index = arith.index_cast %pos : i64 to index
        %hit_index = arith.index_cast %hit : i64 to index
        %piece = func.call @__ly_bytes_slice(%bytes, %pos_index, %hit_index) : (memref<?xi8>, index, index) -> memref<4xi64>
        func.call @__ly_bytes_store_item(%list_items, %slot, %piece) : (memref<?xi64>, i64, memref<4xi64>) -> ()
      }
      %next_pos_hit = arith.addi %hit, %sep_n : i64
      %next_pos = arith.select %matched, %next_pos_hit, %pos : i64
      %bump = arith.select %matched, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      scf.yield %next_pos, %next_slot, %matched : i64, i64, i1
    }
    %tail_start = arith.index_cast %emit#0 : i64 to index
    %len_index = arith.index_cast %len : i64 to index
    %tail = func.call @__ly_bytes_slice(%bytes, %tail_start, %len_index) : (memref<?xi8>, index, index) -> memref<4xi64>
    func.call @__ly_bytes_store_item(%list_items, %emit#1, %tail) : (memref<?xi64>, i64, memref<4xi64>) -> ()
    func.return %list : memref<5xi64>
  }

  // bytes.split() -- ASCII whitespace runs.
  func.func @LyBytes_SplitWS(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %false_a = arith.constant false
    %dim = memref.dim %bytes, %c0 : memref<?xi8>

    %count:2 = scf.for %i = %c0 to %dim step %c1 iter_args(%segs = %zero, %in_run = %false_a) -> (i64, i1) {
      %b_i8 = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %b_i8 : i8 to i64
      %is_space = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
      %true_w = arith.constant true
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts = arith.andi %non_space, %not_in_run : i1
      %bump = arith.select %starts, %one, %zero : i64
      %next_segs = arith.addi %segs, %bump : i64
      scf.yield %next_segs, %non_space : i64, i1
    }

    %list = func.call @LyList_FromLength(%count#0) : (i64) -> memref<5xi64>
    %list_items = func.call @__ly_list_items(%list) : (memref<5xi64>) -> memref<?xi64>

    %false_b = arith.constant false
    scf.for %i = %c0 to %dim step %c1 iter_args(%slot = %zero, %run_start = %dim, %in_run = %false_b) -> (i64, index, i1) {
      %b_i8 = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %b_i8 : i8 to i64
      %is_space = func.call @__ly_bytes_byte_is_ascii_space(%b) : (i64) -> i1
      %true_w = arith.constant true
      %non_space = arith.xori %is_space, %true_w : i1
      %not_in_run = arith.xori %in_run, %true_w : i1
      %starts = arith.andi %non_space, %not_in_run : i1
      %ends = arith.andi %is_space, %in_run : i1
      %new_start = arith.select %starts, %i, %run_start : index
      scf.if %ends {
        %piece = func.call @__ly_bytes_slice(%bytes, %run_start, %i) : (memref<?xi8>, index, index) -> memref<4xi64>
        func.call @__ly_bytes_store_item(%list_items, %slot, %piece) : (memref<?xi64>, i64, memref<4xi64>) -> ()
      }
      %bump = arith.select %ends, %one, %zero : i64
      %next_slot = arith.addi %slot, %bump : i64
      %last = arith.subi %dim, %c1 : index
      %is_last = arith.cmpi eq, %i, %last : index
      %closes = arith.andi %is_last, %non_space : i1
      scf.if %closes {
        %piece = func.call @__ly_bytes_slice(%bytes, %new_start, %dim) : (memref<?xi8>, index, index) -> memref<4xi64>
        func.call @__ly_bytes_store_item(%list_items, %next_slot, %piece) : (memref<?xi64>, i64, memref<4xi64>) -> ()
      }
      scf.yield %next_slot, %new_start, %non_space : i64, index, i1
    }
    func.return %list : memref<5xi64>
  }

  // bytes.join over a runtime list/tuple of bytes.
  func.func @LyBytes_Join(%sep_header: memref<4xi64> {ly.ownership.object_header}, %n: i64, %seq_items: memref<?xi64>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "join", ly.runtime.result_contract = "builtins.bytes"} {
    %sep_bytes = func.call @__ly_bytes_payload(%sep_header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sep_n = func.call @__ly_bytes_len_of(%sep_bytes) : (memref<?xi8>) -> i64
    %n_index = arith.index_cast %n : i64 to index

    %measure = scf.for %k = %c0 to %n_index step %c1 iter_args(%total = %zero) -> (i64) {
      %hdr, %ptr, %blen = func.call @__ly_bytes_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %next = arith.addi %total, %blen : i64
      scf.yield %next : i64
    }
    %has_seps = arith.cmpi sgt, %n, %one : i64
    %sep_uses_i64 = arith.subi %n, %one : i64
    %sep_uses = arith.select %has_seps, %sep_uses_i64, %zero : i64
    %sep_total = arith.muli %sep_uses, %sep_n : i64
    %total = arith.addi %measure, %sep_total : i64

    %out_header = func.call @__ly_bytes_alloc(%total) : (i64) -> memref<4xi64>
    %out_bytes = func.call @__ly_bytes_payload(%out_header) : (memref<4xi64>) -> memref<?xi8>

    scf.for %k = %c0 to %n_index step %c1 iter_args(%pos = %c0) -> (index) {
      %is_first = arith.cmpi eq, %k, %c0 : index
      %true_k = arith.constant true
      %needs_sep = arith.xori %is_first, %true_k : i1
      %after_sep = scf.if %needs_sep -> (index) {
        %sep_n_index = arith.index_cast %sep_n : i64 to index
        scf.for %j = %c0 to %sep_n_index step %c1 {
          %b = memref.load %sep_bytes[%j] : memref<?xi8>
          %dst = arith.addi %pos, %j : index
          memref.store %b, %out_bytes[%dst] : memref<?xi8>
        }
        %advanced = arith.addi %pos, %sep_n_index : index
        scf.yield %advanced : index
      } else {
        scf.yield %pos : index
      }
      %hdr, %ptr, %blen = func.call @__ly_bytes_item_words(%seq_items, %k) : (memref<?xi64>, index) -> (i64, i64, i64)
      %blen_index = arith.index_cast %blen : i64 to index
      scf.for %j = %c0 to %blen_index step %c1 {
        %j_i64 = arith.index_cast %j : index to i64
        %addr = arith.addi %ptr, %j_i64 : i64
        %llptr = llvm.inttoptr %addr : i64 to !llvm.ptr
        %b = llvm.load %llptr : !llvm.ptr -> i8
        %dst = arith.addi %after_sep, %j : index
        memref.store %b, %out_bytes[%dst] : memref<?xi8>
      }
      %next_pos = arith.addi %after_sep, %blen_index : index
      scf.yield %next_pos : index
    }
    func.return %out_header : memref<4xi64>
  }

  // bytes.hex(): lowercase pairs, no separator.
  func.func @LyBytes_Hex(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "hex", ly.runtime.result_contract = "builtins.str"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %out_count = arith.muli %len, %two : i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%out_count, %one) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    scf.for %i = %c0 to %dim step %c1 {
      %b_i8 = memref.load %bytes[%i] : memref<?xi8>
      %b = arith.extui %b_i8 : i8 to i64
      %pos = arith.muli %i, %c2 : index
      func.call @__ly_unicode_put_hex(%out_bytes, %one, %pos, %b, %c2) : (memref<?xi8>, i64, index, i64, index) -> ()
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // bytes.fromhex(str): ASCII whitespace between pairs is ignored; anything
  // else (or an odd trailing digit) raises ValueError at its position.
  func.func @LyBytes_FromHex(%str_header: memref<2xi64> {ly.ownership.object_header}, %str_bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "fromhex"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %sixteen = arith.constant 16 : i64
    %minus_one = arith.constant -1 : i64
    %width = func.call @__ly_unicode_width(%str_header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%str_header, %str_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index

    // Pass 1: count digit pairs; report the first bad position.
    %scan:3 = scf.for %i = %c0 to %count_index step %c1 iter_args(%digits = %zero, %bad = %minus_one, %pending = %minus_one) -> (i64, i64, i64) {
      %cp = func.call @__ly_unicode_get(%str_bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %value = func.call @__ly_bytes_hex_digit(%cp) : (i64) -> i64
      %is_digit = arith.cmpi sge, %value, %zero : i64
      %is_ws = func.call @__ly_bytes_byte_is_ascii_space(%cp) : (i64) -> i1
      %i_i64 = arith.index_cast %i : index to i64
      %no_bad_yet = arith.cmpi eq, %bad, %minus_one : i64
      %has_pending = arith.cmpi sge, %pending, %zero : i64
      // whitespace inside a pair (pending digit) is an error at this index;
      // a non-digit non-ws is an error at this index.
      %true_h = arith.constant true
      %not_digit = arith.xori %is_digit, %true_h : i1
      %not_ws = arith.xori %is_ws, %true_h : i1
      %bad_char = arith.andi %not_digit, %not_ws : i1
      %ws_inside = arith.andi %is_ws, %has_pending : i1
      %new_bad_here = arith.ori %bad_char, %ws_inside : i1
      %record = arith.andi %new_bad_here, %no_bad_yet : i1
      %next_bad = arith.select %record, %i_i64, %bad : i64
      %bump = arith.select %is_digit, %one, %zero : i64
      %next_digits = arith.addi %digits, %bump : i64
      %next_pending_digit = arith.select %has_pending, %minus_one, %i_i64 : i64
      %next_pending = arith.select %is_digit, %next_pending_digit, %pending : i64
      scf.yield %next_digits, %next_bad, %next_pending : i64, i64, i64
    }
    // A trailing unpaired digit is an error at the end of the string.
    %odd = arith.andi %scan#0, %one : i64
    %is_odd = arith.cmpi ne, %odd, %zero : i64
    %no_bad = arith.cmpi eq, %scan#1, %minus_one : i64
    %odd_bad = arith.andi %is_odd, %no_bad : i1
    %bad_final = arith.select %odd_bad, %count, %scan#1 : i64
    %has_bad = arith.cmpi sge, %bad_final, %zero : i64
    scf.if %has_bad {
      %class_id = arith.constant 53 : i64
      %msg_h, %msg_b = scf.if %odd_bad -> (memref<2xi64>, memref<?xi8>) {
        %odd_length = arith.constant 63 : i64
        %odd_static = memref.get_global @__ly_bytes_msg_fromhex_odd : memref<63xi8>
        %odd_message = memref.cast %odd_static : memref<63xi8> to memref<?xi8>
        %odd_h, %odd_b = func.call @LyUnicode_FromBytes(%odd_message, %c0, %odd_length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %odd_h, %odd_b : memref<2xi64>, memref<?xi8>
      } else {
        %length = arith.constant 58 : i64
        %static = memref.get_global @__ly_bytes_msg_fromhex : memref<58xi8>
        %message = memref.cast %static : memref<58xi8> to memref<?xi8>
        %prefix_h, %prefix_b = func.call @LyUnicode_FromBytes(%message, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %pos_h, %pos_b = func.call @LyUnicode_FromI64(%bad_final) : (i64) -> (memref<2xi64>, memref<?xi8>)
        %joined_h, %joined_b = func.call @LyUnicode_Concat(%prefix_h, %prefix_b, %pos_h, %pos_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%prefix_h) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%pos_h) : (memref<2xi64>) -> ()
        scf.yield %joined_h, %joined_b : memref<2xi64>, memref<?xi8>
      }
      %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %msg_h, %msg_b) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
      func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    }
    %pairs = arith.divsi %scan#0, %two : i64
    %out_header = func.call @__ly_bytes_alloc(%pairs) : (i64) -> memref<4xi64>
    %out_bytes = func.call @__ly_bytes_payload(%out_header) : (memref<4xi64>) -> memref<?xi8>
    // Pass 2: emit the pairs.
    scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %minus_one, %slot = %c0) -> (i64, index) {
      %cp = func.call @__ly_unicode_get(%str_bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %value = func.call @__ly_bytes_hex_digit(%cp) : (i64) -> i64
      %is_digit = arith.cmpi sge, %value, %zero : i64
      %has_acc = arith.cmpi sge, %acc, %zero : i64
      %completes = arith.andi %is_digit, %has_acc : i1
      %next_slot = scf.if %completes -> (index) {
        %hi = arith.muli %acc, %sixteen : i64
        %byte = arith.addi %hi, %value : i64
        %byte_i8 = arith.trunci %byte : i64 to i8
        memref.store %byte_i8, %out_bytes[%slot] : memref<?xi8>
        %advanced = arith.addi %slot, %c1 : index
        scf.yield %advanced : index
      } else {
        scf.yield %slot : index
      }
      %start_acc = arith.select %has_acc, %minus_one, %value : i64
      %next_acc = arith.select %is_digit, %start_acc, %acc : i64
      scf.yield %next_acc, %next_slot : i64, index
    }
    func.return %out_header : memref<4xi64>
  }

  func.func private @__ly_bytes_hex_digit(%cp: i64) -> i64 {
    %zero_ch = arith.constant 48 : i64
    %nine_ch = arith.constant 57 : i64
    %a_ch = arith.constant 97 : i64
    %f_ch = arith.constant 102 : i64
    %A_ch = arith.constant 65 : i64
    %F_ch = arith.constant 70 : i64
    %ten = arith.constant 10 : i64
    %minus_one = arith.constant -1 : i64
    %ge0 = arith.cmpi sge, %cp, %zero_ch : i64
    %le9 = arith.cmpi sle, %cp, %nine_ch : i64
    %dec = arith.andi %ge0, %le9 : i1
    %dec_val = arith.subi %cp, %zero_ch : i64
    %gea = arith.cmpi sge, %cp, %a_ch : i64
    %lef = arith.cmpi sle, %cp, %f_ch : i64
    %lower = arith.andi %gea, %lef : i1
    %lower_delta = arith.subi %cp, %a_ch : i64
    %lower_val = arith.addi %lower_delta, %ten : i64
    %geA = arith.cmpi sge, %cp, %A_ch : i64
    %leF = arith.cmpi sle, %cp, %F_ch : i64
    %upper = arith.andi %geA, %leF : i1
    %upper_delta = arith.subi %cp, %A_ch : i64
    %upper_val = arith.addi %upper_delta, %ten : i64
    %v1 = arith.select %upper, %upper_val, %minus_one : i64
    %v2 = arith.select %lower, %lower_val, %v1 : i64
    %value = arith.select %dec, %dec_val, %v2 : i64
    func.return %value : i64
  }

  // bytes * int.
  func.func @LyBytes_Mul(%header: memref<4xi64> {ly.ownership.object_header}, %repeat: i64) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__mul__", ly.runtime.result_contract = "builtins.bytes"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %n = arith.maxsi %repeat, %zero : i64
    %overflows = func.call @__ly_repeat_overflows(%len, %n) : (i64, i64) -> i1
    scf.if %overflows {
      %class_id = arith.constant 104 : i64
      %length = arith.constant 27 : i64
      %message_static = memref.get_global @__ly_bytes_msg_repeat_too_long : memref<27xi8>
      %message = memref.cast %message_static : memref<27xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %total = arith.muli %len, %n : i64
    %one_byte = arith.constant 1 : i64
    %bytes_prefix = arith.constant 32 : i64
    func.call @__ly_check_alloc_count(%total, %one_byte, %bytes_prefix) : (i64, i64, i64) -> ()
    %out_header = func.call @__ly_bytes_alloc(%total) : (i64) -> memref<4xi64>
    %out_bytes = func.call @__ly_bytes_payload(%out_header) : (memref<4xi64>) -> memref<?xi8>
    // No trips over nothing (see `__ly_seq_fill_repeat`).
    %no_bytes = arith.cmpi eq, %len, %zero : i64
    %trips = arith.select %no_bytes, %zero, %n : i64
    %n_index = arith.index_cast %trips : i64 to index
    scf.for %k = %c0 to %n_index step %c1 {
      %base = arith.muli %k, %dim : index
      scf.for %i = %c0 to %dim step %c1 {
        %b = memref.load %bytes[%i] : memref<?xi8>
        %dst = arith.addi %base, %i : index
        memref.store %b, %out_bytes[%dst] : memref<?xi8>
      }
    }
    func.return %out_header : memref<4xi64>
  }

  func.func @LyBytes_DecRef(%header: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }
  memref.global "private" constant @__ly_bytes_msg_repeat_too_long : memref<27xi8> = dense<[114, 101, 112, 101, 97, 116, 101, 100, 32, 98, 121, 116, 101, 115, 32, 97, 114, 101, 32, 116, 111, 111, 32, 108, 111, 110, 103]>

  // The same message over a bytes subject: `int(b"x")` reports b'x', so the
  // repr comes from LyBytes_Repr and the prefix is shared with the str form.
  func.func private @__ly_bytes_raise_invalid_int_literal(%subject: memref<4xi64> {ly.ownership.object_header}) {
    %class_id = arith.constant 53 : i64
    %start = arith.constant 0 : index
    %prefix_length = arith.constant 40 : i64
    %prefix_static = memref.get_global @__ly_long_msg_invalid_int_literal_prefix : memref<40xi8>
    %prefix_bytes = memref.cast %prefix_static : memref<40xi8> to memref<?xi8>
    %prefix_h, %prefix_b = func.call @LyUnicode_FromBytes(%prefix_bytes, %start, %prefix_length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %quoted_h, %quoted_b = func.call @LyBytes_Repr(%subject) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    %full_h, %full_b = func.call @LyUnicode_Concat(%prefix_h, %prefix_b, %quoted_h, %quoted_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%prefix_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%quoted_h) : (memref<2xi64>) -> ()
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %full_h, %full_b) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // int(b"12"). CPython takes bytes anywhere int() takes str, over the same
  // ASCII scan; what differs is the repr the ValueError carries -- 'b' for
  // bytes -- which is why the digits are a shared helper and the raise is not.
  func.func @LyBytes_Int(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__int__", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %payload = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %parsed:2 = func.call @__ly_long_from_ascii(%payload) : (memref<?xi8>) -> (memref<2xi64>, i1)
    cf.cond_br %parsed#1, ^ok, ^invalid

  ^ok:
    func.return %parsed#0 : memref<2xi64>

  ^invalid:
    func.call @LyLong_DecRef(%parsed#0) : (memref<2xi64>) -> ()
    func.call @__ly_bytes_raise_invalid_int_literal(%header) : (memref<4xi64>) -> ()
    %uh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %uh : memref<2xi64>
  }

  // (handle ptr, payload ptr, byte length) of a boxed bytes element.
  //
  // Why this is not __ly_unicode_item_words: for a two-lane contract the box
  // caches BOTH lanes (pointer words 4/5, size words 9/10), so the payload
  // address is readable straight out of the box. A one-lane entity puts only
  // the handle there, and the payload address lives in the handle's word 2 --
  // which is the point of the conversion: a reallocation updates one place and
  // every reader sees it, instead of leaving a stale copy in each box.
  func.func private @__ly_bytes_item_words(%items: memref<?xi64>, %slot: index) -> (i64, i64, i64) {
    %c2 = arith.constant 0 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %base = arith.muli %slot, %c16 : index
    %hdr_slot = arith.addi %base, %c2 : index
    %hdr = memref.load %items[%hdr_slot] : memref<?xi64>
    %hdr_ptr = llvm.inttoptr %hdr : i64 to !llvm.ptr
    %payload_gep = llvm.getelementptr %hdr_ptr[%c2_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %ptr = llvm.load %payload_gep : !llvm.ptr -> i64
    %length_gep = llvm.getelementptr %hdr_ptr[%c3_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %blen = llvm.load %length_gep : !llvm.ptr -> i64
    func.return %hdr, %ptr, %blen : i64, i64, i64
  }

  func.func @LyBytes_Hash(%header: memref<4xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__hash__"} {
    %bytes = func.call @__ly_bytes_payload(%header) : (memref<4xi64>) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %ptr_index = memref.extract_aligned_pointer_as_index %bytes : memref<?xi8> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %dim = memref.dim %bytes, %c0 : memref<?xi8>
    %len = arith.index_cast %dim : index to i64
    %empty = arith.cmpi eq, %len, %zero : i64
    %digest = scf.if %empty -> (i64) {
      scf.yield %zero : i64
    } else {
      %h = func.call @__ly_hash_bytes(%ptr, %len) : (i64, i64) -> i64
      %fixed = func.call @__ly_hash_fixup(%h) : (i64) -> i64
      scf.yield %fixed : i64
    }
    func.return %digest : i64
  }

  func.func private @raw_bytes_equal(%p1: i64, %n1: i64, %p2: i64, %n2: i64) -> i1

  // ===== impls: bytes_iterator =====
  // Two lanes, like str_iterator: the handle carries position/length, the
  // source bytes object travels beside it. The payload is not a third lane --
  // __ly_bytes_payload derives it from the header, so a `bytes` that a
  // deallocator moved would not leave a stale view behind.
  func.func private @LyBytesIterator_Shape() -> (memref<2xi64>, memref<2xi64>, memref<4xi64>) attributes {ly.runtime.contract = "builtins.bytes_iterator", ly.runtime.shape}

  // One block: [0,2) the handle, [2,4) the (position, length) state lane,
  // [4] the source bytes object. The state used to be its own allocation and
  // the source only a lane, so neither was reachable from the handle -- which
  // is what a box holds.
  func.func private @__ly_bytes_iterator_alloc(%position: i64, %length: i64) -> (memref<2xi64>, memref<2xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.bytes_iterator"], ly.ownership.owned_results = [0], ly.runtime.class_id = 24 : i64, ly.runtime.contract = "builtins.bytes_iterator", ly.runtime.primitive = "alloc"} {
    %zero_index = arith.constant 0 : index
    %state_offset = arith.constant 16 : index
    %block_bytes = arith.constant 40 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header = memref.view %block[%zero_index][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %state = memref.view %block[%state_offset][] : memref<?xi8> to memref<2xi64>
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_bytes_iterator = arith.constant 24 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    %source_slot = arith.constant 4 : i64
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_bytes_iterator, %header[%layout_slot] : memref<2xi64>
    memref.store %position, %state[%position_slot] : memref<2xi64>
    memref.store %length, %state[%length_slot] : memref<2xi64>
    %header_ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %header_ptr = arith.index_cast %header_ptr_index : index to i64
    func.call @__ly_entity_word_set(%header_ptr, %source_slot, %zero) : (i64, i64, i64) -> ()
    func.return %header, %state : memref<2xi64>, memref<2xi64>
  }

  func.func @LyBytes_Iter(%source_header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<4xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.bytes_iterator"} {
    %zero = arith.constant 0 : i64
    %length = func.call @LyBytes_Len(%source_header) : (memref<4xi64>) -> i64
    %iter_header, %state = func.call @__ly_bytes_iterator_alloc(%zero, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>)
    %source_slot = arith.constant 4 : i64
    %iter_ptr_index = memref.extract_aligned_pointer_as_index %iter_header : memref<2xi64> -> index
    %iter_ptr = arith.index_cast %iter_ptr_index : index to i64
    %source_ptr_index = memref.extract_aligned_pointer_as_index %source_header : memref<4xi64> -> index
    %source_ptr = arith.index_cast %source_ptr_index : index to i64
    func.call @__ly_entity_word_set(%iter_ptr, %source_slot, %source_ptr) : (i64, i64, i64) -> ()
    %source_header_sub = memref.subview %source_header[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %source_header_view = memref.cast %source_header_sub : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%source_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iter_header, %state, %source_header : memref<2xi64>, memref<2xi64>, memref<4xi64>
  }

  func.func private @__ly_bytes_iterator_lane_words(%iter_ptr: i64) -> (i64, i64, i64, i64) attributes {ly.runtime.contract = "builtins.bytes_iterator", ly.runtime.primitive = "lane_words"} {
    %state_offset = arith.constant 16 : i64
    %two = arith.constant 2 : i64
    %six = arith.constant 6 : i64
    %source_slot = arith.constant 4 : i64
    %state_ptr = arith.addi %iter_ptr, %state_offset : i64
    %source_ptr = func.call @__ly_entity_word_get(%iter_ptr, %source_slot) : (i64, i64) -> i64
    func.return %state_ptr, %two, %source_ptr, %six : i64, i64, i64, i64
  }

  // ⛔ RETURNS THE RECOVERED LANES AND NOT THE ONES IT WAS HANDED. They are the
  // same two by construction, and reading the recovered pair here is what keeps
  // the record honest -- every `for b in some_bytes` compares them.
  func.func @LyBytesIterator_Iter(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<4xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.bytes_iterator"} {
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    %iter_ptr_index = memref.extract_aligned_pointer_as_index %iter_header : memref<2xi64> -> index
    %iter_ptr = arith.index_cast %iter_ptr_index : index to i64
    %state_ptr, %state_size, %source_ptr, %source_size = func.call @__ly_bytes_iterator_lane_words(%iter_ptr) : (i64) -> (i64, i64, i64, i64)
    %state_words = func.call @__ly_global_view_i64(%state_ptr, %state_size) : (i64, i64) -> memref<?xi64>
    %state_view = memref.cast %state_words : memref<?xi64> to memref<2xi64>
    %source_words = func.call @__ly_global_view_i64(%source_ptr, %source_size) : (i64, i64) -> memref<?xi64>
    %source_view = memref.cast %source_words : memref<?xi64> to memref<4xi64>
    func.return %iter_header, %state_view, %source_view : memref<2xi64>, memref<2xi64>, memref<4xi64>
  }

  func.func @LyBytesIterator_Next(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, i1, memref<2xi64>, memref<2xi64>, memref<4xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "builtins.bytes_iterator"], ly.ownership.owned_results = [0, 2], ly.runtime.contract = "builtins.bytes_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.int", ly.runtime.next_contract = "builtins.bytes_iterator", ly.runtime.valid_result_index = 1 : i64} {
    // ⛔ THE STATE COMES FROM THE RECORD AND NOT FROM THE LANE. They are the
    // same two words by construction, and stepping the recovered one is what
    // makes the record load-bearing: every `for b in some_bytes` walks it, so a
    // block that stopped describing its own lanes stops iterating.
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    %iter_ptr_index = memref.extract_aligned_pointer_as_index %iter_header : memref<2xi64> -> index
    %iter_ptr = arith.index_cast %iter_ptr_index : index to i64
    %state_ptr, %state_size, %recorded_source_ptr, %recorded_source_size = func.call @__ly_bytes_iterator_lane_words(%iter_ptr) : (i64) -> (i64, i64, i64, i64)
    %state_words = func.call @__ly_global_view_i64(%state_ptr, %state_size) : (i64, i64) -> memref<?xi64>
    %live_state = memref.cast %state_words : memref<?xi64> to memref<2xi64>
    %position = memref.load %live_state[%position_slot] : memref<2xi64>
    %length = memref.load %live_state[%length_slot] : memref<2xi64>
    %valid = arith.cmpi slt, %position, %length : i64
    %one = arith.constant 1 : i64
    %next_position_candidate = arith.addi %position, %one : i64
    %next_position = arith.select %valid, %next_position_candidate, %position : i1, i64
    memref.store %next_position, %live_state[%position_slot] : memref<2xi64>
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()

    %zero = arith.constant 0 : i64
    %bytes = func.call @__ly_bytes_payload(%source_header) : (memref<4xi64>) -> memref<?xi8>
    %element_value = scf.if %valid -> (i64) {
      %at = arith.index_cast %position : i64 to index
      %byte = memref.load %bytes[%at] : memref<?xi8>
      %wide = arith.extui %byte : i8 to i64
      scf.yield %wide : i64
    } else {
      scf.yield %zero : i64
    }
    %element = func.call @LyLong_FromI64(%element_value) : (i64) -> memref<2xi64>
    func.return %element, %valid, %iter_header, %state, %source_header : memref<2xi64>, i1, memref<2xi64>, memref<2xi64>, memref<4xi64>
  }

  func.func @LyBytesIterator_DecRef(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytes_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %iter_header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyBytes_DecRef(%source_header) : (memref<4xi64>) -> ()
    // The state is interior to the handle's block now; the handle view carries
    // the whole allocation, the way a str's header view does.
    memref.dealloc %iter_header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
