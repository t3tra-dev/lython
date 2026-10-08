// `bytearray` and its iterator -- CPython's Objects/bytearrayobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// The handle is bytes' four words (objects/bytes.mlir), so the bytes functions
// that only read their receiver -- searching, testing, decoding, comparing --
// read a bytearray as it is: most methods here are those functions under
// bytearray's contract, a bytes result copied into a bytearray where CPython's
// method returns one. What a bytearray adds is a payload in a block of its own,
// which PyByteArray_Resize's policy grows and shrinks.
//
// Deviations from CPython:
// - Where CPython takes any buffer, an argument here is bytes -- or a bytearray
//   where the argument is the whole value: construction, +, +=, extend, the
//   comparisons, `in` and slice assignment. An iterable of ints is a list of
//   ints.
// - No buffer exports, so nothing refuses a resize -- but `b += b`, whose
//   BufferError CPython raises from its own export of the argument, raises it.
// - The payload keeps no trailing NUL: nothing here reads it as a C string.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.bytearray", "builtins.bytearray_iterator"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyMemoryView_ToBytes(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "tobytes"}
  func.func private @LyBytes_Bool(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__bool__"}
  func.func private @LyBytes_Capitalize(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "capitalize", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_ContainsBytes(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__contains__"}
  func.func private @LyBytes_CountSub(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "count", ly.runtime.result_contract = "builtins.int"}
  func.func private @LyBytes_DecRef(%header: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.deallocator}
  func.func private @LyBytes_Decode(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyBytes_DecodeEncoding(%header: memref<4xi64> {ly.ownership.object_header}, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyBytes_DecodeEncodingErrors(%header: memref<4xi64> {ly.ownership.object_header}, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>, %err_header: memref<2xi64> {ly.ownership.object_header}, %err_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyBytes_EndsWith(%header: memref<4xi64> {ly.ownership.object_header}, %suffix_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "endswith"}
  func.func private @LyBytes_EqBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__eq__"}
  func.func private @LyBytes_Find(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "find", ly.runtime.result_contract = "builtins.int"}
  func.func private @LyBytes_FromHex(%str_header: memref<2xi64> {ly.ownership.object_header}, %str_bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "fromhex"}
  func.func private @LyBytes_GeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__ge__"}
  func.func private @LyBytes_GtBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__gt__"}
  func.func private @LyBytes_Hex(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "hex", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyBytes_IsAlnum(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isalnum"}
  func.func private @LyBytes_IsAlpha(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isalpha"}
  func.func private @LyBytes_IsAscii(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isascii"}
  func.func private @LyBytes_IsDigit(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isdigit"}
  func.func private @LyBytes_IsLower(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "islower"}
  func.func private @LyBytes_IsSpace(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isspace"}
  func.func private @LyBytes_IsUpper(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "isupper"}
  func.func private @LyBytes_Join(%sep_header: memref<4xi64> {ly.ownership.object_header}, %n: i64, %seq_items: memref<?xi64>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "join", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_LStrip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_LStripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_LeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__le__"}
  func.func private @LyBytes_Len(%header: memref<4xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__len__"}
  func.func private @LyBytes_Lower(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "lower", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_LtBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__lt__"}
  func.func private @LyBytes_NeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__ne__"}
  func.func private @LyBytes_NewEncoded(%text_header: memref<2xi64> {ly.ownership.object_header}, %text_bytes: memref<?xi8>, %encoding_header: memref<2xi64> {ly.ownership.object_header}, %encoding_bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"}
  func.func private @LyBytes_RStrip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_RStripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_RemovePrefix(%header: memref<4xi64> {ly.ownership.object_header}, %affix_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "removeprefix", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_RemoveSuffix(%header: memref<4xi64> {ly.ownership.object_header}, %affix_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "removesuffix", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_Replace(%header: memref<4xi64> {ly.ownership.object_header}, %old_header: memref<4xi64> {ly.ownership.object_header}, %new_header: memref<4xi64> {ly.ownership.object_header}, %limit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "replace", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_Repr(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyBytes_Split(%header: memref<4xi64> {ly.ownership.object_header}, %sep_header: memref<4xi64> {ly.ownership.object_header}, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytes"}
  func.func private @LyBytes_SplitWS(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytes"}
  func.func private @LyBytes_StartsWith(%header: memref<4xi64> {ly.ownership.object_header}, %prefix_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "startswith"}
  func.func private @LyBytes_Strip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_StripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_SwapCase(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "swapcase", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_Title(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "title", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_Upper(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "upper", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyList_Len(%self: memref<5xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__len__"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_SlotWordAsI64(%word: i64) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_as_i64"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"}
  func.func private @__ly_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) -> index
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_bytes_payload(%self: memref<4xi64>) -> memref<?xi8> attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.interior_word, ly.runtime.primitive = "payload_view"}
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_list_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.list", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8>
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64)
  func.func private @__ly_slice_raise_extended_mismatch(%prefix: memref<?xi8>, %prefix_len: i64, %src_len: i64, %slice_len: i64)
  func.func private @__ly_slice_raise_zero_step()
  func.func private @__ly_slice_unpack(%self: memref<5xi64>) -> (i64, i64, i64, i64)
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @free_raw_i64_ptr(%address: i64)
  func.func private @realloc_raw_i64_ptr(%address: i64, %bytes: i64) -> i64
  py.class @bytearray attributes {
    base_names = ["MutableSequence"],
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    ly.typing.final,
    ly.runtime.contract = "builtins.bytearray", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    method_names = ["__new__", "__new__", "__new__", "__new__", "__new__", "__new__", "__init__",
                    "__init__", "__init__", "__init__", "__init__", "__init__", "__len__",
                    "__bool__", "__getitem__", "__getslice__", "__getslice__", "__setitem__",
                    "__setslice__", "__setslice__", "__setslice__", "__setslice__",
                    "__setslice__", "__setslice__", "__delitem__", "__delslice__",
                    "__delslice__", "__contains__", "__contains__", "__contains__", "__iter__",
                    "__eq__", "__ne__", "__lt__", "__lt__", "__le__", "__le__", "__gt__",
                    "__gt__", "__ge__", "__ge__", "__add__", "__add__", "__iadd__", "__iadd__",
                    "__mul__", "__rmul__", "__imul__", "__repr__", "__str__", "append", "extend",
                    "extend", "extend", "insert", "pop", "pop", "remove", "clear", "reverse",
                    "copy", "decode", "decode", "decode", "hex", "fromhex", "find", "find",
                    "find", "count", "count", "count", "startswith", "startswith", "startswith",
                    "endswith", "endswith", "endswith", "isalpha", "isdigit", "isalnum",
                    "isspace", "isascii", "islower", "isupper", "upper", "lower", "swapcase",
                    "capitalize", "title", "strip", "strip", "lstrip", "lstrip", "rstrip",
                    "rstrip", "removeprefix", "removesuffix", "replace", "replace", "split",
                    "split", "split", "join", "__new__", "__init__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>, !py.contract<"builtins.bytes">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>, !py.contract<"builtins.bytearray">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>, !py.contract<"builtins.list", [!py.contract<"builtins.int">]>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.list", [!py.contract<"builtins.int">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.list", [!py.contract<"builtins.int">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.slice">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.slice">, !py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.slice">, !py.contract<"builtins.list", [!py.contract<"builtins.int">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.slice">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray_iterator">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.list", [!py.contract<"builtins.int">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.str">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>, !py.contract<"builtins.str">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytearray">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytearray">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.bytearray">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.protocol<"Iterable", [!py.contract<"builtins.bytes">]>] -> [!py.contract<"builtins.bytearray">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bytearray">>, !py.contract<"builtins.memoryview">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray">, !py.contract<"builtins.memoryview">] -> [!py.literal<None>]>
    ],
    method_kinds = ["classmethod", "classmethod", "classmethod", "classmethod", "classmethod",
                    "classmethod", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "classmethod", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "classmethod", "instance"]
  } {}

  py.class @bytearray_iterator attributes {
    base_names = ["Iterator"],
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray_iterator">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bytearray_iterator">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  // ===== the object (PyByteArrayObject) =====
  //
  // The handle is bytes' four words -- refcount, class 26, the payload's
  // address, the length -- and the payload lives in a block of its own:
  // [ob_alloc][ob_exports][ob_alloc bytes]. The block moves when
  // PyByteArray_Resize's policy says so, and the handle's address word moves
  // with it, so every holder reads the payload where it is now -- except a
  // memoryview, which holds the address itself and counts in ob_exports, and
  // while it does nothing may resize (CPython's _canresize).
  func.func private @LyByteArray_Shape() -> memref<4xi64> attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.shape}
  func.func private @LyByteArrayIterator_Shape() -> memref<4xi64> attributes {ly.runtime.contract = "builtins.bytearray_iterator", ly.runtime.shape}

  // The block's address: two words before the payload.
  func.func private @__ly_bytearray_block(%self: memref<4xi64>) -> i64 attributes {ly.runtime.contract = "builtins.bytearray"} {
    %payload_slot = arith.constant 2 : index
    %prefix = arith.constant 16 : i64
    %payload = memref.load %self[%payload_slot] : memref<4xi64>
    %block = arith.subi %payload, %prefix : i64
    func.return %block : i64
  }

  // ob_exports, the block's second word: the memoryviews holding the payload.
  func.func private @__ly_bytearray_exports(%self: memref<4xi64>) -> i64 attributes {ly.runtime.contract = "builtins.bytearray"} {
    %c1 = arith.constant 1 : index
    %two = arith.constant 2 : i64
    %block = func.call @__ly_bytearray_block(%self) : (memref<4xi64>) -> i64
    %view = func.call @__ly_global_view_i64(%block, %two) : (i64, i64) -> memref<?xi64>
    %exports = memref.load %view[%c1] : memref<?xi64>
    func.return %exports : i64
  }

  // One export taken (+1) or given back (-1) -- bytearray_getbuffer and
  // bytearray_releasebuffer.
  func.func private @__ly_bytearray_add_export(%self: memref<4xi64>, %delta: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %c1 = arith.constant 1 : index
    %two = arith.constant 2 : i64
    %block = func.call @__ly_bytearray_block(%self) : (memref<4xi64>) -> i64
    %view = func.call @__ly_global_view_i64(%block, %two) : (i64, i64) -> memref<?xi64>
    %exports = memref.load %view[%c1] : memref<?xi64>
    %next = arith.addi %exports, %delta : i64
    memref.store %next, %view[%c1] : memref<?xi64>
    func.return
  }

  // _canresize: BufferError while a memoryview holds the payload.
  func.func private @__ly_bytearray_check_resizable(%self: memref<4xi64>) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %exports = func.call @__ly_bytearray_exports(%self) : (memref<4xi64>) -> i64
    %exported = arith.cmpi sgt, %exports, %zero : i64
    scf.if %exported {
      %buffer_error = arith.constant {ly.class_id_of = "builtins.BufferError"} 105 : i64
      %message_length = arith.constant 51 : i64
      %static = memref.get_global @__ly_bytearray_msg_exported : memref<51xi8>
      %message = memref.cast %static : memref<51xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%buffer_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return
  }

  // ob_alloc, the block's first word.
  func.func private @__ly_bytearray_alloc_of(%self: memref<4xi64>) -> i64 attributes {ly.runtime.contract = "builtins.bytearray"} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i64
    %block = func.call @__ly_bytearray_block(%self) : (memref<4xi64>) -> i64
    %view = func.call @__ly_global_view_i64(%block, %one) : (i64, i64) -> memref<?xi64>
    %alloc = memref.load %view[%c0] : memref<?xi64>
    func.return %alloc : i64
  }

  // A block for `alloc` bytes -- `old_block` moved by realloc, which keeps
  // its export count, or a new one with none when it is 0 -- published as the
  // payload of `self`.
  func.func private @__ly_bytearray_reblock(%self: memref<4xi64>, %old_block: i64, %alloc: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %prefix = arith.constant 16 : i64
    %payload_slot = arith.constant 2 : index
    %bytes = func.call @__ly_alloc_count(%alloc, %one, %prefix) : (i64, i64, i64) -> index
    %bytes_word = arith.index_cast %bytes : index to i64
    // ⛔ Never no payload at all, even for an ob_alloc of 0: the address word
    // would point one past the block's end, which is legal but is not a
    // pointer INTO the block, so LeakSanitizer took an empty bytearray held to
    // the end of the program for a leaked one.
    %empty = arith.cmpi eq, %bytes_word, %zero : i64
    %room = arith.select %empty, %one, %bytes_word : i64
    %total = arith.addi %room, %prefix : i64
    %block = func.call @realloc_raw_i64_ptr(%old_block, %total) : (i64, i64) -> i64
    %view = func.call @__ly_global_view_i64(%block, %two) : (i64, i64) -> memref<?xi64>
    memref.store %alloc, %view[%c0] : memref<?xi64>
    %fresh = arith.cmpi eq, %old_block, %zero : i64
    scf.if %fresh {
      memref.store %zero, %view[%c1] : memref<?xi64>
    }
    %payload = arith.addi %block, %prefix : i64
    memref.store %payload, %self[%payload_slot] : memref<4xi64>
    func.return
  }

  // A new bytearray of `length` bytes, contents unset, its block sized as
  // PyByteArray_FromStringAndSize sizes one: length + 1, or nothing for an
  // empty one.
  func.func private @__ly_bytearray_alloc(%length: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %class_bytearray = arith.constant {ly.class_id_of = "builtins.bytearray"} 26 : i64
    %header_bytes = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %refcount_slot = arith.constant 0 : index
    %class_slot = arith.constant 1 : index
    %length_slot = arith.constant 3 : index
    %raw = memref.alloc(%header_bytes) {alignment = 16 : i64} : memref<?xi8>
    %self = memref.view %raw[%c0][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<4xi64>
    memref.store %one, %self[%refcount_slot] : memref<4xi64>
    memref.store %class_bytearray, %self[%class_slot] : memref<4xi64>
    memref.store %length, %self[%length_slot] : memref<4xi64>
    %empty = arith.cmpi eq, %length, %zero : i64
    %grown = arith.addi %length, %one : i64
    %alloc = arith.select %empty, %zero, %grown : i64
    func.call @__ly_bytearray_reblock(%self, %zero, %alloc) : (memref<4xi64>, i64, i64) -> ()
    func.return %self : memref<4xi64>
  }

  // PyByteArray_Resize: the length becomes `size`. The block moves only where
  // CPython's would: grown past ob_alloc -- to 1.125x plus a little for a
  // moderate step, to exactly size + 1 for a large one -- or shrunk below
  // half of it, to size + 1.
  func.func private @__ly_bytearray_resize(%self: memref<4xi64>, %size: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %six = arith.constant 6 : i64
    %nine = arith.constant 9 : i64
    %true = arith.constant true
    %length_slot = arith.constant 3 : index
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %unchanged = arith.cmpi eq, %size, %length : i64
    %alloc = func.call @__ly_bytearray_alloc_of(%self) : (memref<4xi64>) -> i64
    %needed = arith.addi %size, %one : i64
    %fits = arith.cmpi sle, %needed, %alloc : i64
    %half = arith.shrsi %alloc, %one : i64
    %major_down = arith.cmpi slt, %size, %half : i64
    %keeps_block_down = arith.xori %major_down, %true : i1
    %minor_down = arith.andi %fits, %keeps_block_down : i1
    %stays = arith.ori %unchanged, %minor_down : i1
    scf.if %unchanged {
    } else {
      func.call @__ly_bytearray_check_resizable(%self) : (memref<4xi64>) -> ()
    }
    scf.if %stays {
    } else {
      // `size <= alloc * 1.125`, exact in integers: the step past alloc is at
      // most an eighth of it.
      %step = arith.subi %size, %alloc : i64
      %eighth = arith.shrsi %alloc, %three : i64
      %moderate = arith.cmpi sle, %step, %eighth : i64
      %small = arith.cmpi slt, %size, %nine : i64
      %pad = arith.select %small, %three, %six : i64
      %size_eighth = arith.shrsi %size, %three : i64
      %over = arith.addi %size, %size_eighth : i64
      %moderate_alloc = arith.addi %over, %pad : i64
      %grow_alloc = arith.select %moderate, %moderate_alloc, %needed : i64
      %new_alloc = arith.select %fits, %needed, %grow_alloc : i64
      %block = func.call @__ly_bytearray_block(%self) : (memref<4xi64>) -> i64
      func.call @__ly_bytearray_reblock(%self, %block, %new_alloc) : (memref<4xi64>, i64, i64) -> ()
    }
    memref.store %size, %self[%length_slot] : memref<4xi64>
    func.return
  }

  // memmove inside one payload: `count` bytes from offset `from` to offset
  // `to`, copied from the end when the destination lies past the source.
  func.func private @__ly_bytearray_move(%bytes: memref<?xi8>, %to: i64, %from: i64, %count: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %n = arith.index_cast %count : i64 to index
    %backward = arith.cmpi sgt, %to, %from : i64
    scf.if %backward {
      %last = arith.subi %count, %one : i64
      scf.for %i = %c0 to %n step %c1 {
        %i64 = arith.index_cast %i : index to i64
        %k = arith.subi %last, %i64 : i64
        %src64 = arith.addi %from, %k : i64
        %dst64 = arith.addi %to, %k : i64
        %src = arith.index_cast %src64 : i64 to index
        %dst = arith.index_cast %dst64 : i64 to index
        %byte = memref.load %bytes[%src] : memref<?xi8>
        memref.store %byte, %bytes[%dst] : memref<?xi8>
      }
    } else {
      scf.for %i = %c0 to %n step %c1 {
        %i64 = arith.index_cast %i : index to i64
        %src64 = arith.addi %from, %i64 : i64
        %dst64 = arith.addi %to, %i64 : i64
        %src = arith.index_cast %src64 : i64 to index
        %dst = arith.index_cast %dst64 : i64 to index
        %byte = memref.load %bytes[%src] : memref<?xi8>
        memref.store %byte, %bytes[%dst] : memref<?xi8>
      }
    }
    func.return
  }

  // `count` bytes of `from` (from its offset `at`) into `to` at offset `into`.
  func.func private @__ly_bytearray_copy(%to: memref<?xi8>, %into: i64, %from: memref<?xi8>, %at: i64, %count: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %src64 = arith.addi %at, %i64 : i64
      %dst64 = arith.addi %into, %i64 : i64
      %src = arith.index_cast %src64 : i64 to index
      %dst = arith.index_cast %dst64 : i64 to index
      %byte = memref.load %from[%src] : memref<?xi8>
      memref.store %byte, %to[%dst] : memref<?xi8>
    }
    func.return
  }

  // A bytearray holding a copy of a bytes-shaped handle's payload -- a bytes
  // or a bytearray, whose words are the same.
  func.func private @__ly_bytearray_from_bytes(%source: memref<4xi64>) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 3 : index
    %length = memref.load %source[%length_slot] : memref<4xi64>
    %self = func.call @__ly_bytearray_alloc(%length) : (i64) -> memref<4xi64>
    %from = func.call @__ly_bytes_payload(%source) : (memref<4xi64>) -> memref<?xi8>
    %to = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    func.call @__ly_bytearray_copy(%to, %zero, %from, %zero, %length) : (memref<?xi8>, i64, memref<?xi8>, i64, i64) -> ()
    func.return %self : memref<4xi64>
  }

  // "byte must be in range(0, 256)"
  memref.global "private" constant @__ly_bytearray_msg_byte_range : memref<29xi8> = dense<[98, 121, 116, 101, 32, 109, 117, 115, 116, 32, 98, 101, 32, 105, 110, 32, 114, 97, 110, 103, 101, 40, 48, 44, 32, 50, 53, 54, 41]>
  // "bytearray index out of range"
  memref.global "private" constant @__ly_bytearray_msg_index : memref<28xi8> = dense<[98, 121, 116, 101, 97, 114, 114, 97, 121, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>
  // "negative count"
  memref.global "private" constant @__ly_bytearray_msg_negative_count : memref<14xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 99, 111, 117, 110, 116]>
  // "pop from empty bytearray"
  memref.global "private" constant @__ly_bytearray_msg_pop_empty : memref<24xi8> = dense<[112, 111, 112, 32, 102, 114, 111, 109, 32, 101, 109, 112, 116, 121, 32, 98, 121, 116, 101, 97, 114, 114, 97, 121]>
  // "pop index out of range"
  memref.global "private" constant @__ly_bytearray_msg_pop_index : memref<22xi8> = dense<[112, 111, 112, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>
  // "value not found in bytearray"
  memref.global "private" constant @__ly_bytearray_msg_not_found : memref<28xi8> = dense<[118, 97, 108, 117, 101, 32, 110, 111, 116, 32, 102, 111, 117, 110, 100, 32, 105, 110, 32, 98, 121, 116, 101, 97, 114, 114, 97, 121]>
  // "Existing exports of data: object cannot be re-sized"
  memref.global "private" constant @__ly_bytearray_msg_exported : memref<51xi8> = dense<[69, 120, 105, 115, 116, 105, 110, 103, 32, 101, 120, 112, 111, 114, 116, 115, 32, 111, 102, 32, 100, 97, 116, 97, 58, 32, 111, 98, 106, 101, 99, 116, 32, 99, 97, 110, 110, 111, 116, 32, 98, 101, 32, 114, 101, 45, 115, 105, 122, 101, 100]>
  // "attempt to assign bytes of size "
  memref.global "private" constant @__ly_bytearray_msg_assign_prefix : memref<32xi8> = dense<[97, 116, 116, 101, 109, 112, 116, 32, 116, 111, 32, 97, 115, 115, 105, 103, 110, 32, 98, 121, 116, 101, 115, 32, 111, 102, 32, 115, 105, 122, 101, 32]>

  // One of the messages above as the exception `class_id` names.
  func.func private @__ly_bytearray_raise(%class_id: i64, %message: memref<?xi8>, %length: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // _getbytevalue: a value is a byte or a ValueError.
  func.func private @__ly_bytearray_check_byte(%value: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %end = arith.constant 256 : i64
    %low = arith.cmpi sge, %value, %zero : i64
    %high = arith.cmpi slt, %value, %end : i64
    %fits = arith.andi %low, %high : i1
    scf.if %fits {
    } else {
      %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
      %length = arith.constant 29 : i64
      %static = memref.get_global @__ly_bytearray_msg_byte_range : memref<29xi8>
      %message = memref.cast %static : memref<29xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%value_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return
  }

  // The position an index names, from the end when negative; IndexError
  // "bytearray index out of range" past either end.
  func.func private @__ly_bytearray_index(%self: memref<4xi64>, %raw_index: i64) -> i64 attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 3 : index
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %length : i64
    %index = arith.select %negative, %from_end, %raw_index : i64
    %low = arith.cmpi sge, %index, %zero : i64
    %high = arith.cmpi slt, %index, %length : i64
    %valid = arith.andi %low, %high : i1
    scf.if %valid {
    } else {
      %index_error = arith.constant {ly.class_id_of = "builtins.IndexError"} 55 : i64
      %message_length = arith.constant 28 : i64
      %static = memref.get_global @__ly_bytearray_msg_index : memref<28xi8>
      %message = memref.cast %static : memref<28xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%index_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return %index : i64
  }

  // A bytearray of the ints a list holds, each checked to be a byte.
  func.func private @__ly_bytearray_of_list(%items: memref<5xi64>) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %length = func.call @LyList_Len(%items) : (memref<5xi64>) -> i64
    %slots = func.call @__ly_list_items(%items) : (memref<5xi64>) -> memref<?xi64>
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %handle_words_index = arith.index_cast %handle_words : i64 to index
    %count = arith.index_cast %length : i64 to index
    %big = arith.constant 256 : i64
    scf.for %i = %c0 to %count step %c1 {
      %base = arith.muli %i, %handle_words_index : index
      %entity = memref.load %slots[%base] : memref<?xi64>
      %value, %fits = func.call @LyLong_SlotWordAsI64(%entity) : (i64) -> (i64, i1)
      %checked = arith.select %fits, %value, %big : i64
      func.call @__ly_bytearray_check_byte(%checked) : (i64) -> ()
    }
    %self = func.call @__ly_bytearray_alloc(%length) : (i64) -> memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    scf.for %i = %c0 to %count step %c1 {
      %base = arith.muli %i, %handle_words_index : index
      %entity = memref.load %slots[%base] : memref<?xi64>
      %value, %fits = func.call @LyLong_SlotWordAsI64(%entity) : (i64) -> (i64, i1)
      %byte = arith.trunci %value : i64 to i8
      memref.store %byte, %payload[%i] : memref<?xi8>
    }
    func.return %self : memref<4xi64>
  }

  // ===== construction (bytearray___init___impl) =====
  //
  // One initializer per shape, as bytes has them, and an empty __init__ for
  // each: the constructor path calls both.
  func.func @LyByteArray_NewEmpty() -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.bytearray", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %self = func.call @__ly_bytearray_alloc(%zero) : (i64) -> memref<4xi64>
    func.return %self : memref<4xi64>
  }

  func.func @LyByteArray_InitEmpty(%self: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__init__"} {
    func.return
  }

  // bytearray(n): n zero bytes; "negative count" below zero, MemoryError past
  // what can be allocated.
  func.func @LyByteArray_NewZeros(%count: i64) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.bytearray", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %word = arith.constant 8 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_byte = arith.constant 0 : i8
    %negative = arith.cmpi slt, %count, %zero : i64
    scf.if %negative {
      %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
      %length = arith.constant 14 : i64
      %static = memref.get_global @__ly_bytearray_msg_negative_count : memref<14xi8>
      %message = memref.cast %static : memref<14xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%value_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.call @__ly_check_alloc_count(%count, %one, %word) : (i64, i64, i64) -> ()
    %self = func.call @__ly_bytearray_alloc(%count) : (i64) -> memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %n = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      memref.store %zero_byte, %payload[%i] : memref<?xi8>
    }
    func.return %self : memref<4xi64>
  }

  func.func @LyByteArray_InitZeros(%self: memref<4xi64> {ly.ownership.object_header}, %count: i64) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__init__"} {
    func.return
  }

  // bytearray(b) and bytearray(ba): a copy.
  func.func @LyByteArray_NewCopy(%source: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.bytearray", ly.runtime.initializer = "__new__"} {
    %self = func.call @__ly_bytearray_from_bytes(%source) : (memref<4xi64>) -> memref<4xi64>
    func.return %self : memref<4xi64>
  }

  func.func @LyByteArray_InitCopy(%self: memref<4xi64> {ly.ownership.object_header}, %source: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__init__"} {
    func.return
  }

  // bytearray([65, 66]): each int a byte.
  func.func @LyByteArray_NewFromList(%items: memref<5xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.bytearray", ly.runtime.initializer = "__new__"} {
    %self = func.call @__ly_bytearray_of_list(%items) : (memref<5xi64>) -> memref<4xi64>
    func.return %self : memref<4xi64>
  }

  func.func @LyByteArray_InitFromList(%self: memref<4xi64> {ly.ownership.object_header}, %items: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__init__"} {
    func.return
  }

  // bytearray(view): a copy of the bytes a memoryview shows.
  func.func @LyByteArray_NewFromView(%view: memref<8xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.bytearray", ly.runtime.initializer = "__new__"} {
    %bytes = func.call @LyMemoryView_ToBytes(%view) : (memref<8xi64>) -> memref<4xi64>
    %self = func.call @__ly_bytearray_from_bytes(%bytes) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%bytes) : (memref<4xi64>) -> ()
    func.return %self : memref<4xi64>
  }

  func.func @LyByteArray_InitFromView(%self: memref<4xi64> {ly.ownership.object_header}, %view: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__init__"} {
    func.return
  }

  // bytearray(text, encoding): the text encoded, as bytes(text, encoding) is.
  func.func @LyByteArray_NewEncoded(%text_header: memref<2xi64> {ly.ownership.object_header}, %text_bytes: memref<?xi8>, %encoding_header: memref<2xi64> {ly.ownership.object_header}, %encoding_bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.bytearray", ly.runtime.initializer = "__new__"} {
    %encoded = func.call @LyBytes_NewEncoded(%text_header, %text_bytes, %encoding_header, %encoding_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> memref<4xi64>
    %self = func.call @__ly_bytearray_from_bytes(%encoded) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%encoded) : (memref<4xi64>) -> ()
    func.return %self : memref<4xi64>
  }

  func.func @LyByteArray_InitEncoded(%self: memref<4xi64> {ly.ownership.object_header}, %text_header: memref<2xi64> {ly.ownership.object_header}, %text_bytes: memref<?xi8>, %encoding_header: memref<2xi64> {ly.ownership.object_header}, %encoding_bytes: memref<?xi8>) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyByteArray_DecRef(%self: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %block = func.call @__ly_bytearray_block(%self) : (memref<4xi64>) -> i64
    func.call @free_raw_i64_ptr(%block) : (i64) -> ()
    memref.dealloc %self : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== the sequence (bytearray_getitem, bytearray_ass_subscript) =====

  func.func @LyByteArray_GetItem(%self: memref<4xi64> {ly.ownership.object_header}, %raw_index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__getitem__", ly.runtime.result_contract = "builtins.int"} {
    %index = func.call @__ly_bytearray_index(%self, %raw_index) : (memref<4xi64>, i64) -> i64
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %at = arith.index_cast %index : i64 to index
    %byte = memref.load %payload[%at] : memref<?xi8>
    %wide = arith.extui %byte : i8 to i64
    %result = func.call @LyLong_FromI64(%wide) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // b[i] = v: the index is checked before the value, as CPython checks them.
  func.func @LyByteArray_SetItem(%self: memref<4xi64> {ly.ownership.object_header}, %raw_index: i64, %value: i64 {ly.runtime.clip_i64}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__setitem__"} {
    %index = func.call @__ly_bytearray_index(%self, %raw_index) : (memref<4xi64>, i64) -> i64
    func.call @__ly_bytearray_check_byte(%value) : (i64) -> ()
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %at = arith.index_cast %index : i64 to index
    %byte = arith.trunci %value : i64 to i8
    memref.store %byte, %payload[%at] : memref<?xi8>
    func.return
  }

  func.func @LyByteArray_DelItem(%self: memref<4xi64> {ly.ownership.object_header}, %raw_index: i64) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__delitem__"} {
    %one = arith.constant 1 : i64
    %length_slot = arith.constant 3 : index
    %index = func.call @__ly_bytearray_index(%self, %raw_index) : (memref<4xi64>, i64) -> i64
    func.call @__ly_bytearray_check_resizable(%self) : (memref<4xi64>) -> ()
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %next = arith.addi %index, %one : i64
    %tail = arith.subi %length, %next : i64
    func.call @__ly_bytearray_move(%payload, %index, %next, %tail) : (memref<?xi8>, i64, i64, i64) -> ()
    %shorter = arith.subi %length, %one : i64
    func.call @__ly_bytearray_resize(%self, %shorter) : (memref<4xi64>, i64) -> ()
    func.return
  }

  func.func @LyByteArray_GetSlice(%self: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__getslice__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %length_slot = arith.constant 3 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    // The throw does not return; the substitute keeps the IR division-safe.
    %step = arith.select %step_zero, %one, %step_raw : i64
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %start, %count = func.call @__ly_slice_adjust(%length, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %result = func.call @__ly_bytearray_alloc(%count) : (i64) -> memref<4xi64>
    %from = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %to = func.call @__ly_bytes_payload(%result) : (memref<4xi64>) -> memref<?xi8>
    %n = arith.index_cast %count : i64 to index
    scf.for %k = %c0 to %n step %c1 {
      %k64 = arith.index_cast %k : index to i64
      %offset = arith.muli %k64, %step : i64
      %src64 = arith.addi %start, %offset : i64
      %src = arith.index_cast %src64 : i64 to index
      %byte = memref.load %from[%src] : memref<?xi8>
      memref.store %byte, %to[%k] : memref<?xi8>
    }
    func.return %result : memref<4xi64>
  }

  func.func @LyByteArray_SliceSubscript(%self: memref<4xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__getslice__"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    %result = func.call @LyByteArray_GetSlice(%self, %start, %stop, %step, %mask) : (memref<4xi64>, i64, i64, i64, i64) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  // Removes the `count` bytes a slice selects -- from `start`, `step` apart --
  // moving what follows down (bytearray_ass_subscript's deletion).
  func.func private @__ly_bytearray_delete(%self: memref<4xi64>, %start: i64, %count: i64, %step: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %length_slot = arith.constant 3 : index
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %nothing = arith.cmpi eq, %count, %zero : i64
    %contiguous = arith.cmpi eq, %step, %one : i64
    // An extended slice is checked even when it selects nothing: CPython's
    // deletion asks _canresize before it looks at the slice's length.
    %true = arith.constant true
    %something = arith.xori %nothing, %true : i1
    %extended = arith.xori %contiguous, %true : i1
    %checked = arith.ori %something, %extended : i1
    scf.if %checked {
      func.call @__ly_bytearray_check_resizable(%self) : (memref<4xi64>) -> ()
    }
    scf.if %nothing {
    } else {
      scf.if %contiguous {
        %after = arith.addi %start, %count : i64
        %tail = arith.subi %length, %after : i64
        func.call @__ly_bytearray_move(%payload, %start, %after, %tail) : (memref<?xi8>, i64, i64, i64) -> ()
      } else {
        // The selected positions from the lowest, `stride` apart; every other
        // byte from there on moves down over them.
        %negative = arith.cmpi slt, %step, %zero : i64
        %last = arith.subi %count, %one : i64
        %span = arith.muli %last, %step : i64
        %lowest_back = arith.addi %start, %span : i64
        %lowest = arith.select %negative, %lowest_back, %start : i64
        %flipped = arith.subi %zero, %step : i64
        %stride = arith.select %negative, %flipped, %step : i64
        %selected_end_offset = arith.muli %count, %stride : i64
        %from = arith.index_cast %lowest : i64 to index
        %end = arith.index_cast %length : i64 to index
        %written = scf.for %read = %from to %end step %c1 iter_args(%write = %lowest) -> (i64) {
          %read64 = arith.index_cast %read : index to i64
          %offset = arith.subi %read64, %lowest : i64
          %phase = arith.remsi %offset, %stride : i64
          %on_stride = arith.cmpi eq, %phase, %zero : i64
          %within = arith.cmpi slt, %offset, %selected_end_offset : i64
          %selected = arith.andi %on_stride, %within : i1
          %next = scf.if %selected -> (i64) {
            scf.yield %write : i64
          } else {
            %byte = memref.load %payload[%read] : memref<?xi8>
            %at = arith.index_cast %write : i64 to index
            memref.store %byte, %payload[%at] : memref<?xi8>
            %moved = arith.addi %write, %one : i64
            scf.yield %moved : i64
          }
          scf.yield %next : i64
        }
      }
      %shorter = arith.subi %length, %count : i64
      func.call @__ly_bytearray_resize(%self, %shorter) : (memref<4xi64>, i64) -> ()
    }
    func.return
  }

  // ValueError "attempt to assign bytes of size N to extended slice of size M".
  func.func private @__ly_bytearray_raise_extended(%given: i64, %count: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %length = arith.constant 32 : i64
    %static = memref.get_global @__ly_bytearray_msg_assign_prefix : memref<32xi8>
    %prefix = memref.cast %static : memref<32xi8> to memref<?xi8>
    func.call @__ly_slice_raise_extended_mismatch(%prefix, %length, %given, %count) : (memref<?xi8>, i64, i64, i64) -> ()
    func.return
  }

  // b[i:j:k] = values, `values` a bytes-shaped handle that is not `self`
  // (bytearray_ass_subscript): step 1 splices (bytearray_setslice_linear);
  // another step writes one byte per selected position, and an empty value
  // deletes the selection -- how CPython reads `b[::2] = b""`.
  func.func private @__ly_bytearray_assign(%self: memref<4xi64>, %start_raw: i64, %stop_raw: i64, %step_raw: i64, %mask: i64, %values: memref<4xi64>) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %length_slot = arith.constant 3 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    %step = arith.select %step_zero, %one, %step_raw : i64
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %start, %count = func.call @__ly_slice_adjust(%length, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %needed = memref.load %values[%length_slot] : memref<4xi64>
    %contiguous = arith.cmpi eq, %step, %one : i64
    scf.if %contiguous {
      %hi = arith.addi %start, %count : i64
      %tail = arith.subi %length, %hi : i64
      %new_hi = arith.addi %start, %needed : i64
      %growth = arith.subi %needed, %count : i64
      %shrinks = arith.cmpi slt, %growth, %zero : i64
      %grows = arith.cmpi sgt, %growth, %zero : i64
      %new_length = arith.addi %length, %growth : i64
      scf.if %shrinks {
        func.call @__ly_bytearray_check_resizable(%self) : (memref<4xi64>) -> ()
        %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
        func.call @__ly_bytearray_move(%payload, %new_hi, %hi, %tail) : (memref<?xi8>, i64, i64, i64) -> ()
        func.call @__ly_bytearray_resize(%self, %new_length) : (memref<4xi64>, i64) -> ()
      }
      scf.if %grows {
        func.call @__ly_bytearray_resize(%self, %new_length) : (memref<4xi64>, i64) -> ()
        %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
        func.call @__ly_bytearray_move(%payload, %new_hi, %hi, %tail) : (memref<?xi8>, i64, i64, i64) -> ()
      }
      %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
      %source = func.call @__ly_bytes_payload(%values) : (memref<4xi64>) -> memref<?xi8>
      func.call @__ly_bytearray_copy(%payload, %start, %source, %zero, %needed) : (memref<?xi8>, i64, memref<?xi8>, i64, i64) -> ()
    } else {
      %empty = arith.cmpi eq, %needed, %zero : i64
      scf.if %empty {
        func.call @__ly_bytearray_delete(%self, %start, %count, %step) : (memref<4xi64>, i64, i64, i64) -> ()
      } else {
        %mismatch = arith.cmpi ne, %needed, %count : i64
        scf.if %mismatch {
          func.call @__ly_bytearray_raise_extended(%needed, %count) : (i64, i64) -> ()
        }
        %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
        %source = func.call @__ly_bytes_payload(%values) : (memref<4xi64>) -> memref<?xi8>
        %n = arith.index_cast %count : i64 to index
        scf.for %k = %c0 to %n step %c1 {
          %k64 = arith.index_cast %k : index to i64
          %offset = arith.muli %k64, %step : i64
          %dst64 = arith.addi %start, %offset : i64
          %dst = arith.index_cast %dst64 : i64 to index
          %byte = memref.load %source[%k] : memref<?xi8>
          memref.store %byte, %payload[%dst] : memref<?xi8>
        }
      }
    }
    func.return
  }

  // b[i:j:k] = values for bytes or a bytearray; the values copied first when
  // they are `b` itself.
  func.func @LyByteArray_SetSlice(%self: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64, %values: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__setslice__"} {
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<4xi64> -> index
    %values_index = memref.extract_aligned_pointer_as_index %values : memref<4xi64> -> index
    %same = arith.cmpi eq, %self_index, %values_index : index
    scf.if %same {
      %copy = func.call @__ly_bytearray_from_bytes(%values) : (memref<4xi64>) -> memref<4xi64>
      func.call @__ly_bytearray_assign(%self, %start_raw, %stop_raw, %step_raw, %mask, %copy) : (memref<4xi64>, i64, i64, i64, i64, memref<4xi64>) -> ()
      func.call @LyByteArray_DecRef(%copy) : (memref<4xi64>) -> ()
    } else {
      func.call @__ly_bytearray_assign(%self, %start_raw, %stop_raw, %step_raw, %mask, %values) : (memref<4xi64>, i64, i64, i64, i64, memref<4xi64>) -> ()
    }
    func.return
  }

  // b[i:j:k] = [ints]: a zero step refused before the ints are read, as the
  // slice is unpacked before the values are.
  func.func @LyByteArray_SetSliceList(%self: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64, %items: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__setslice__"} {
    %zero = arith.constant 0 : i64
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    %values = func.call @__ly_bytearray_of_list(%items) : (memref<5xi64>) -> memref<4xi64>
    func.call @__ly_bytearray_assign(%self, %start_raw, %stop_raw, %step_raw, %mask, %values) : (memref<4xi64>, i64, i64, i64, i64, memref<4xi64>) -> ()
    func.call @LyByteArray_DecRef(%values) : (memref<4xi64>) -> ()
    func.return
  }

  func.func @LyByteArray_SliceAssign(%self: memref<4xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}, %values: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__setslice__"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    func.call @LyByteArray_SetSlice(%self, %start, %stop, %step, %mask, %values) : (memref<4xi64>, i64, i64, i64, i64, memref<4xi64>) -> ()
    func.return
  }

  func.func @LyByteArray_SliceAssignList(%self: memref<4xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}, %items: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__setslice__"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    func.call @LyByteArray_SetSliceList(%self, %start, %stop, %step, %mask, %items) : (memref<4xi64>, i64, i64, i64, i64, memref<5xi64>) -> ()
    func.return
  }

  func.func @LyByteArray_DelSlice(%self: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__delslice__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %length_slot = arith.constant 3 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    %step = arith.select %step_zero, %one, %step_raw : i64
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %start, %count = func.call @__ly_slice_adjust(%length, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    func.call @__ly_bytearray_delete(%self, %start, %count, %step) : (memref<4xi64>, i64, i64, i64) -> ()
    func.return
  }

  func.func @LyByteArray_SliceDelete(%self: memref<4xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__delslice__"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    func.call @LyByteArray_DelSlice(%self, %start, %stop, %step, %mask) : (memref<4xi64>, i64, i64, i64, i64) -> ()
    func.return
  }

  // v in b for an int: "byte must be in range(0, 256)" for one that is not a
  // byte, as CPython's _getbytevalue answers.
  func.func @LyByteArray_ContainsInt(%self: memref<4xi64> {ly.ownership.object_header}, %value: i64 {ly.runtime.clip_i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__contains__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %length_slot = arith.constant 3 : index
    func.call @__ly_bytearray_check_byte(%value) : (i64) -> ()
    %byte = arith.trunci %value : i64 to i8
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %n = arith.index_cast %length : i64 to index
    %found = scf.for %i = %c0 to %n step %c1 iter_args(%seen = %false) -> (i1) {
      %at = memref.load %payload[%i] : memref<?xi8>
      %match = arith.cmpi eq, %at, %byte : i8
      %either = arith.ori %seen, %match : i1
      scf.yield %either : i1
    }
    func.return %found : i1
  }

  // ===== mutation (bytearray methods) =====

  // `count` more bytes at the end, from a bytes-shaped handle (the source
  // payload read after the resize: when it is `self`, it moved).
  func.func private @__ly_bytearray_extend_bytes(%self: memref<4xi64>, %other: memref<4xi64>) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %word = arith.constant 8 : i64
    %length_slot = arith.constant 3 : index
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %added = memref.load %other[%length_slot] : memref<4xi64>
    %total = arith.addi %length, %added : i64
    func.call @__ly_check_alloc_count(%total, %one, %word) : (i64, i64, i64) -> ()
    func.call @__ly_bytearray_resize(%self, %total) : (memref<4xi64>, i64) -> ()
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %source = func.call @__ly_bytes_payload(%other) : (memref<4xi64>) -> memref<?xi8>
    func.call @__ly_bytearray_copy(%payload, %length, %source, %zero, %added) : (memref<?xi8>, i64, memref<?xi8>, i64, i64) -> ()
    func.return
  }

  func.func @LyByteArray_Append(%self: memref<4xi64> {ly.ownership.object_header}, %value: i64 {ly.runtime.clip_i64}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "append"} {
    %one = arith.constant 1 : i64
    %length_slot = arith.constant 3 : index
    func.call @__ly_bytearray_check_byte(%value) : (i64) -> ()
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %longer = arith.addi %length, %one : i64
    func.call @__ly_bytearray_resize(%self, %longer) : (memref<4xi64>, i64) -> ()
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %at = arith.index_cast %length : i64 to index
    %byte = arith.trunci %value : i64 to i8
    memref.store %byte, %payload[%at] : memref<?xi8>
    func.return
  }

  func.func @LyByteArray_Extend(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "extend"} {
    func.call @__ly_bytearray_extend_bytes(%self, %other) : (memref<4xi64>, memref<4xi64>) -> ()
    func.return
  }

  func.func @LyByteArray_ExtendList(%self: memref<4xi64> {ly.ownership.object_header}, %items: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "extend"} {
    %values = func.call @__ly_bytearray_of_list(%items) : (memref<5xi64>) -> memref<4xi64>
    func.call @__ly_bytearray_extend_bytes(%self, %values) : (memref<4xi64>, memref<4xi64>) -> ()
    func.call @LyByteArray_DecRef(%values) : (memref<4xi64>) -> ()
    func.return
  }

  // b.insert(i, v): v checked first; i clamped to the ends, from the end when
  // negative (bytearray_insert_impl).
  func.func @LyByteArray_Insert(%self: memref<4xi64> {ly.ownership.object_header}, %raw_index: i64, %value: i64 {ly.runtime.clip_i64}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "insert"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %length_slot = arith.constant 3 : index
    func.call @__ly_bytearray_check_byte(%value) : (i64) -> ()
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %longer = arith.addi %length, %one : i64
    func.call @__ly_bytearray_resize(%self, %longer) : (memref<4xi64>, i64) -> ()
    %negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %length : i64
    %shifted = arith.select %negative, %from_end, %raw_index : i64
    %below = arith.cmpi slt, %shifted, %zero : i64
    %floored = arith.select %below, %zero, %shifted : i64
    %above = arith.cmpi sgt, %floored, %length : i64
    %index = arith.select %above, %length, %floored : i64
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %next = arith.addi %index, %one : i64
    %tail = arith.subi %length, %index : i64
    func.call @__ly_bytearray_move(%payload, %next, %index, %tail) : (memref<?xi8>, i64, i64, i64) -> ()
    %at = arith.index_cast %index : i64 to index
    %byte = arith.trunci %value : i64 to i8
    memref.store %byte, %payload[%at] : memref<?xi8>
    func.return
  }

  // b.pop(i): "pop from empty bytearray", then "pop index out of range".
  func.func @LyByteArray_PopAt(%self: memref<4xi64> {ly.ownership.object_header}, %raw_index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "pop", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %index_error = arith.constant {ly.class_id_of = "builtins.IndexError"} 55 : i64
    %length_slot = arith.constant 3 : index
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %empty = arith.cmpi eq, %length, %zero : i64
    scf.if %empty {
      %message_length = arith.constant 24 : i64
      %static = memref.get_global @__ly_bytearray_msg_pop_empty : memref<24xi8>
      %message = memref.cast %static : memref<24xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%index_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    %negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %length : i64
    %index = arith.select %negative, %from_end, %raw_index : i64
    %low = arith.cmpi sge, %index, %zero : i64
    %high = arith.cmpi slt, %index, %length : i64
    %valid = arith.andi %low, %high : i1
    scf.if %valid {
    } else {
      %message_length = arith.constant 22 : i64
      %static = memref.get_global @__ly_bytearray_msg_pop_index : memref<22xi8>
      %message = memref.cast %static : memref<22xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%index_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.call @__ly_bytearray_check_resizable(%self) : (memref<4xi64>) -> ()
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %at = arith.index_cast %index : i64 to index
    %byte = memref.load %payload[%at] : memref<?xi8>
    %value = arith.extui %byte : i8 to i64
    %next = arith.addi %index, %one : i64
    %tail = arith.subi %length, %next : i64
    func.call @__ly_bytearray_move(%payload, %index, %next, %tail) : (memref<?xi8>, i64, i64, i64) -> ()
    %shorter = arith.subi %length, %one : i64
    func.call @__ly_bytearray_resize(%self, %shorter) : (memref<4xi64>, i64) -> ()
    %result = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func @LyByteArray_Pop(%self: memref<4xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "pop", ly.runtime.result_contract = "builtins.int"} {
    %last = arith.constant -1 : i64
    %result = func.call @LyByteArray_PopAt(%self, %last) : (memref<4xi64>, i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // b.remove(v): the first v, or "value not found in bytearray".
  func.func @LyByteArray_Remove(%self: memref<4xi64> {ly.ownership.object_header}, %value: i64 {ly.runtime.clip_i64}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "remove"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %length_slot = arith.constant 3 : index
    func.call @__ly_bytearray_check_byte(%value) : (i64) -> ()
    %byte = arith.trunci %value : i64 to i8
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %n = arith.index_cast %length : i64 to index
    %found = scf.for %i = %c0 to %n step %c1 iter_args(%first = %minus_one) -> (i64) {
      %at = memref.load %payload[%i] : memref<?xi8>
      %match = arith.cmpi eq, %at, %byte : i8
      %unset = arith.cmpi slt, %first, %zero : i64
      %take = arith.andi %match, %unset : i1
      %i64 = arith.index_cast %i : index to i64
      %next = arith.select %take, %i64, %first : i64
      scf.yield %next : i64
    }
    %missing = arith.cmpi slt, %found, %zero : i64
    scf.if %missing {
      %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
      %message_length = arith.constant 28 : i64
      %static = memref.get_global @__ly_bytearray_msg_not_found : memref<28xi8>
      %message = memref.cast %static : memref<28xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%value_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.call @__ly_bytearray_check_resizable(%self) : (memref<4xi64>) -> ()
    %next = arith.addi %found, %one : i64
    %tail = arith.subi %length, %next : i64
    func.call @__ly_bytearray_move(%payload, %found, %next, %tail) : (memref<?xi8>, i64, i64, i64) -> ()
    %shorter = arith.subi %length, %one : i64
    func.call @__ly_bytearray_resize(%self, %shorter) : (memref<4xi64>, i64) -> ()
    func.return
  }

  func.func @LyByteArray_Clear(%self: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "clear"} {
    %zero = arith.constant 0 : i64
    func.call @__ly_bytearray_resize(%self, %zero) : (memref<4xi64>, i64) -> ()
    func.return
  }

  func.func @LyByteArray_Reverse(%self: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %length_slot = arith.constant 3 : index
    %length = memref.load %self[%length_slot] : memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %half = arith.divsi %length, %two : i64
    %n = arith.index_cast %half : i64 to index
    %last = arith.subi %length, %one : i64
    scf.for %i = %c0 to %n step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %j64 = arith.subi %last, %i64 : i64
      %j = arith.index_cast %j64 : i64 to index
      %front = memref.load %payload[%i] : memref<?xi8>
      %back = memref.load %payload[%j] : memref<?xi8>
      memref.store %back, %payload[%i] : memref<?xi8>
      memref.store %front, %payload[%j] : memref<?xi8>
    }
    func.return
  }

  func.func @LyByteArray_Copy(%self: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "copy"} {
    %copy = func.call @__ly_bytearray_from_bytes(%self) : (memref<4xi64>) -> memref<4xi64>
    func.return %copy : memref<4xi64>
  }

  // ===== operators =====

  func.func @LyByteArray_Concat(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__add__"} {
    %result = func.call @__ly_bytearray_from_bytes(%self) : (memref<4xi64>) -> memref<4xi64>
    func.call @__ly_bytearray_extend_bytes(%result, %other) : (memref<4xi64>, memref<4xi64>) -> ()
    func.return %result : memref<4xi64>
  }

  // b += other: the bytes appended in place and `b` handed back. `b += b` is
  // CPython's BufferError -- its export of the argument forbids resizing the
  // object it is -- and kept as one.
  func.func @LyByteArray_InplaceConcat(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__iadd__"} {
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<4xi64> -> index
    %other_index = memref.extract_aligned_pointer_as_index %other : memref<4xi64> -> index
    %same = arith.cmpi eq, %self_index, %other_index : index
    scf.if %same {
      %buffer_error = arith.constant {ly.class_id_of = "builtins.BufferError"} 105 : i64
      %message_length = arith.constant 51 : i64
      %static = memref.get_global @__ly_bytearray_msg_exported : memref<51xi8>
      %message = memref.cast %static : memref<51xi8> to memref<?xi8>
      func.call @__ly_bytearray_raise(%buffer_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.call @__ly_bytearray_extend_bytes(%self, %other) : (memref<4xi64>, memref<4xi64>) -> ()
    %refcount_view = memref.subview %self[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %self : memref<4xi64>
  }

  // `times` copies of the payload into `self` (already `times` times long).
  func.func private @__ly_bytearray_replicate(%self: memref<4xi64>, %unit: i64, %times: i64) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c1_start = arith.constant 1 : index
    %payload = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
    %n = arith.index_cast %times : i64 to index
    scf.for %copy = %c1_start to %n step %c1 {
      %copy64 = arith.index_cast %copy : index to i64
      %into = arith.muli %copy64, %unit : i64
      func.call @__ly_bytearray_copy(%payload, %into, %payload, %zero, %unit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64) -> ()
    }
    func.return
  }

  // The length `times` copies take, or MemoryError past what can be
  // allocated (bytearray_repeat's check).
  func.func private @__ly_bytearray_repeat_length(%unit: i64, %times: i64) -> i64 attributes {ly.runtime.contract = "builtins.bytearray"} {
    %zero = arith.constant 0 : i64
    %word = arith.constant 8 : i64
    %negative = arith.cmpi slt, %times, %zero : i64
    %count = arith.select %negative, %zero, %times : i64
    func.call @__ly_check_alloc_count(%count, %unit, %word) : (i64, i64, i64) -> ()
    %total = arith.muli %unit, %count : i64
    func.return %total : i64
  }

  func.func @LyByteArray_Repeat(%self: memref<4xi64> {ly.ownership.object_header}, %times: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__mul__"} {
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 3 : index
    %unit = memref.load %self[%length_slot] : memref<4xi64>
    %total = func.call @__ly_bytearray_repeat_length(%unit, %times) : (i64, i64) -> i64
    %result = func.call @__ly_bytearray_alloc(%total) : (i64) -> memref<4xi64>
    %empty = arith.cmpi eq, %total, %zero : i64
    scf.if %empty {
    } else {
      %payload = func.call @__ly_bytes_payload(%result) : (memref<4xi64>) -> memref<?xi8>
      %source = func.call @__ly_bytes_payload(%self) : (memref<4xi64>) -> memref<?xi8>
      func.call @__ly_bytearray_copy(%payload, %zero, %source, %zero, %unit) : (memref<?xi8>, i64, memref<?xi8>, i64, i64) -> ()
      %times_kept = arith.divsi %total, %unit : i64
      func.call @__ly_bytearray_replicate(%result, %unit, %times_kept) : (memref<4xi64>, i64, i64) -> ()
    }
    func.return %result : memref<4xi64>
  }

  func.func @LyByteArray_RRepeat(%self: memref<4xi64> {ly.ownership.object_header}, %times: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__rmul__"} {
    %result = func.call @LyByteArray_Repeat(%self, %times) : (memref<4xi64>, i64) -> memref<4xi64>
    func.return %result : memref<4xi64>
  }

  func.func @LyByteArray_InplaceRepeat(%self: memref<4xi64> {ly.ownership.object_header}, %times: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytearray"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__imul__"} {
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 3 : index
    %unit = memref.load %self[%length_slot] : memref<4xi64>
    %total = func.call @__ly_bytearray_repeat_length(%unit, %times) : (i64, i64) -> i64
    func.call @__ly_bytearray_resize(%self, %total) : (memref<4xi64>, i64) -> ()
    %empty = arith.cmpi eq, %total, %zero : i64
    scf.if %empty {
    } else {
      %times_kept = arith.divsi %total, %unit : i64
      func.call @__ly_bytearray_replicate(%self, %unit, %times_kept) : (memref<4xi64>, i64, i64) -> ()
    }
    %refcount_view = memref.subview %self[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %self : memref<4xi64>
  }

  // "bytearray("
  memref.global "private" constant @__ly_bytearray_repr_open : memref<10xi8> = dense<[98, 121, 116, 101, 97, 114, 114, 97, 121, 40]>

  // repr(b): "bytearray(" + bytes' repr of the payload + ")".
  func.func @LyByteArray_Repr(%self: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %ten = arith.constant 10 : i64
    %one = arith.constant 1 : i64
    %open_static = memref.get_global @__ly_bytearray_repr_open : memref<10xi8>
    %open = memref.cast %open_static : memref<10xi8> to memref<?xi8>
    %close_static = memref.get_global @__ly_repr_rparen : memref<1xi8>
    %close = memref.cast %close_static : memref<1xi8> to memref<?xi8>
    %head_h, %head_b = func.call @__ly_unicode_from_valid_utf8(%open, %c0, %ten) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %inner_h, %inner_b = func.call @LyBytes_Repr(%self) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    %a_h, %a_b = func.call @LyUnicode_Concat(%head_h, %head_b, %inner_h, %inner_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%head_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%inner_h) : (memref<2xi64>) -> ()
    %tail_h, %tail_b = func.call @__ly_unicode_from_valid_utf8(%close, %c0, %one) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %r_h, %r_b = func.call @LyUnicode_Concat(%a_h, %a_b, %tail_h, %tail_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%a_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%tail_h) : (memref<2xi64>) -> ()
    func.return %r_h, %r_b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyByteArray_Str(%self: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %h, %b = func.call @LyByteArray_Repr(%self) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // The bytes a bytes-returning method made, as bytearrays, in place in the
  // list it returned (split).
  func.func private @__ly_bytearray_convert_list(%list: memref<5xi64>) attributes {ly.runtime.contract = "builtins.bytearray"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %four = arith.constant 4 : i64
    %length = func.call @LyList_Len(%list) : (memref<5xi64>) -> i64
    %slots = func.call @__ly_list_items(%list) : (memref<5xi64>) -> memref<?xi64>
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %handle_words_index = arith.index_cast %handle_words : i64 to index
    %n = arith.index_cast %length : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      %at = arith.muli %i, %handle_words_index : index
      %entity = memref.load %slots[%at] : memref<?xi64>
      %view = func.call @__ly_global_view_i64(%entity, %four) : (i64, i64) -> memref<?xi64>
      %bytes = memref.cast %view : memref<?xi64> to memref<4xi64>
      %converted = func.call @__ly_bytearray_from_bytes(%bytes) : (memref<4xi64>) -> memref<4xi64>
      func.call @LyBytes_DecRef(%bytes) : (memref<4xi64>) -> ()
      %converted_index = memref.extract_aligned_pointer_as_index %converted : memref<4xi64> -> index
      %converted_word = arith.index_cast %converted_index : index to i64
      memref.store %converted_word, %slots[%at] : memref<?xi64>
    }
    func.return
  }

  // ===== iteration (bytearrayiterobject) =====
  //
  // [refcount, class 28, index, the bytearray's address -- 0 once exhausted,
  // when CPython drops its reference too]. Each step reads the bytearray's
  // length then, so bytes appended while iterating are reached.
  func.func @LyByteArray_Iter(%self: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.bytearray_iterator"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %class_iterator = arith.constant {ly.class_id_of = "builtins.bytearray_iterator"} 28 : i64
    %header_bytes = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %refcount_slot = arith.constant 0 : index
    %class_slot = arith.constant 1 : index
    %index_slot = arith.constant 2 : index
    %source_slot = arith.constant 3 : index
    %raw = memref.alloc(%header_bytes) {alignment = 16 : i64} : memref<?xi8>
    %iterator = memref.view %raw[%c0][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<4xi64>
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<4xi64> -> index
    %self_word = arith.index_cast %self_index : index to i64
    memref.store %one, %iterator[%refcount_slot] : memref<4xi64>
    memref.store %class_iterator, %iterator[%class_slot] : memref<4xi64>
    memref.store %zero, %iterator[%index_slot] : memref<4xi64>
    memref.store %self_word, %iterator[%source_slot] : memref<4xi64>
    %refcount_view = memref.subview %self[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iterator : memref<4xi64>
  }

  func.func @LyByteArrayIterator_Iter(%self: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.bytearray_iterator"} {
    %refcount_view = memref.subview %self[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %self : memref<4xi64>
  }

  func.func @LyByteArrayIterator_Next(%self: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, i1, memref<4xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "builtins.bytearray_iterator"], ly.ownership.owned_results = [0, 2], ly.runtime.contract = "builtins.bytearray_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.int", ly.runtime.next_contract = "builtins.bytearray_iterator", ly.runtime.valid_result_index = 1 : i64} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %false = arith.constant false
    %index_slot = arith.constant 2 : index
    %source_slot = arith.constant 3 : index
    %length_slot = arith.constant 3 : index
    %index = memref.load %self[%index_slot] : memref<4xi64>
    %source_word = memref.load %self[%source_slot] : memref<4xi64>
    %attached = arith.cmpi ne, %source_word, %zero : i64
    %valid, %value = scf.if %attached -> (i1, i64) {
      %view = func.call @__ly_global_view_i64(%source_word, %four) : (i64, i64) -> memref<?xi64>
      %source = memref.cast %view : memref<?xi64> to memref<4xi64>
      %length = memref.load %source[%length_slot] : memref<4xi64>
      %more = arith.cmpi slt, %index, %length : i64
      %byte_value = scf.if %more -> (i64) {
        %payload = func.call @__ly_bytes_payload(%source) : (memref<4xi64>) -> memref<?xi8>
        %at = arith.index_cast %index : i64 to index
        %byte = memref.load %payload[%at] : memref<?xi8>
        %wide = arith.extui %byte : i8 to i64
        scf.yield %wide : i64
      } else {
        memref.store %zero, %self[%source_slot] : memref<4xi64>
        func.call @LyByteArray_DecRef(%source) : (memref<4xi64>) -> ()
        scf.yield %zero : i64
      }
      scf.yield %more, %byte_value : i1, i64
    } else {
      scf.yield %false, %zero : i1, i64
    }
    %advanced = arith.addi %index, %one : i64
    %next_index = arith.select %valid, %advanced, %index : i64
    memref.store %next_index, %self[%index_slot] : memref<4xi64>
    %refcount_view = memref.subview %self[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    %element = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %element, %valid, %self : memref<2xi64>, i1, memref<4xi64>
  }

  func.func @LyByteArrayIterator_DecRef(%self: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytearray_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %zero = arith.constant 0 : i64
    %four = arith.constant 4 : i64
    %source_slot = arith.constant 3 : index
    %source_word = memref.load %self[%source_slot] : memref<4xi64>
    %attached = arith.cmpi ne, %source_word, %zero : i64
    scf.if %attached {
      %view = func.call @__ly_global_view_i64(%source_word, %four) : (i64, i64) -> memref<?xi64>
      %source = memref.cast %view : memref<?xi64> to memref<4xi64>
      func.call @LyByteArray_DecRef(%source) : (memref<4xi64>) -> ()
    }
    memref.dealloc %self : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== bytes' functions under bytearray's contract =====
  //
  // Each takes the signature of the bytes function it calls, on the same
  // handle; a bytes result becomes a bytearray (__ly_bytearray_from_bytes), a
  // list of them a list of bytearrays.

  func.func @LyByteArray_Len(%header: memref<4xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__len__"} {
    %r0 = func.call @LyBytes_Len(%header) : (memref<4xi64>) -> i64
    func.return %r0 : i64
  }

  func.func @LyByteArray_Bool(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__bool__"} {
    %r0 = func.call @LyBytes_Bool(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_Decode(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %r0, %r1 = func.call @LyBytes_Decode(%header) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %r0, %r1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyByteArray_DecodeEncoding(%header: memref<4xi64> {ly.ownership.object_header}, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %r0, %r1 = func.call @LyBytes_DecodeEncoding(%header, %enc_header, %enc_bytes) : (memref<4xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %r0, %r1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyByteArray_DecodeEncodingErrors(%header: memref<4xi64> {ly.ownership.object_header}, %enc_header: memref<2xi64> {ly.ownership.object_header}, %enc_bytes: memref<?xi8>, %err_header: memref<2xi64> {ly.ownership.object_header}, %err_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "decode", ly.runtime.result_contract = "builtins.str"} {
    %r0, %r1 = func.call @LyBytes_DecodeEncodingErrors(%header, %enc_header, %enc_bytes, %err_header, %err_bytes) : (memref<4xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %r0, %r1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyByteArray_Hex(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "hex", ly.runtime.result_contract = "builtins.str"} {
    %r0, %r1 = func.call @LyBytes_Hex(%header) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %r0, %r1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyByteArray_Find(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "find", ly.runtime.result_contract = "builtins.int"} {
    %r0 = func.call @LyBytes_Find(%header, %sub_header, %start_raw, %end_raw) : (memref<4xi64>, memref<4xi64>, i64, i64) -> memref<2xi64>
    func.return %r0 : memref<2xi64>
  }

  func.func @LyByteArray_CountSub(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "count", ly.runtime.result_contract = "builtins.int"} {
    %r0 = func.call @LyBytes_CountSub(%header, %sub_header, %start_raw, %end_raw) : (memref<4xi64>, memref<4xi64>, i64, i64) -> memref<2xi64>
    func.return %r0 : memref<2xi64>
  }

  func.func @LyByteArray_StartsWith(%header: memref<4xi64> {ly.ownership.object_header}, %prefix_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "startswith"} {
    %r0 = func.call @LyBytes_StartsWith(%header, %prefix_header, %start_raw, %end_raw) : (memref<4xi64>, memref<4xi64>, i64, i64) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_EndsWith(%header: memref<4xi64> {ly.ownership.object_header}, %suffix_header: memref<4xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 0 : i64}, %end_raw: i64 {ly.runtime.clip_i64, ly.runtime.default_i64 = 9223372036854775807 : i64}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "endswith"} {
    %r0 = func.call @LyBytes_EndsWith(%header, %suffix_header, %start_raw, %end_raw) : (memref<4xi64>, memref<4xi64>, i64, i64) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_IsAlpha(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "isalpha"} {
    %r0 = func.call @LyBytes_IsAlpha(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_IsDigit(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "isdigit"} {
    %r0 = func.call @LyBytes_IsDigit(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_IsAlnum(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "isalnum"} {
    %r0 = func.call @LyBytes_IsAlnum(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_IsSpace(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "isspace"} {
    %r0 = func.call @LyBytes_IsSpace(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_IsAscii(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "isascii"} {
    %r0 = func.call @LyBytes_IsAscii(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_IsLower(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "islower"} {
    %r0 = func.call @LyBytes_IsLower(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_IsUpper(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "isupper"} {
    %r0 = func.call @LyBytes_IsUpper(%header) : (memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_ContainsBytes(%header: memref<4xi64> {ly.ownership.object_header}, %sub_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__contains__"} {
    %r0 = func.call @LyBytes_ContainsBytes(%header, %sub_header) : (memref<4xi64>, memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_EqBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__eq__"} {
    %r0 = func.call @LyBytes_EqBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_NeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__ne__"} {
    %r0 = func.call @LyBytes_NeBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_LtBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__lt__"} {
    %r0 = func.call @LyBytes_LtBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_LeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__le__"} {
    %r0 = func.call @LyBytes_LeBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_GtBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__gt__"} {
    %r0 = func.call @LyBytes_GtBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_GeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__ge__"} {
    %r0 = func.call @LyBytes_GeBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    func.return %r0 : i1
  }

  func.func @LyByteArray_Upper(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "upper", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Upper(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_Lower(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "lower", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Lower(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_SwapCase(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "swapcase", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_SwapCase(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_Capitalize(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "capitalize", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Capitalize(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_Title(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "title", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Title(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_LStrip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_LStrip(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_LStripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "lstrip", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_LStripChars(%header, %chars_header) : (memref<4xi64>, memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_RStrip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_RStrip(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_RStripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "rstrip", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_RStripChars(%header, %chars_header) : (memref<4xi64>, memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_Strip(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Strip(%header) : (memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_StripChars(%header: memref<4xi64> {ly.ownership.object_header}, %chars_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "strip", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_StripChars(%header, %chars_header) : (memref<4xi64>, memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_RemovePrefix(%header: memref<4xi64> {ly.ownership.object_header}, %affix_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "removeprefix", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_RemovePrefix(%header, %affix_header) : (memref<4xi64>, memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_RemoveSuffix(%header: memref<4xi64> {ly.ownership.object_header}, %affix_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "removesuffix", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_RemoveSuffix(%header, %affix_header) : (memref<4xi64>, memref<4xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_Replace(%header: memref<4xi64> {ly.ownership.object_header}, %old_header: memref<4xi64> {ly.ownership.object_header}, %new_header: memref<4xi64> {ly.ownership.object_header}, %limit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "replace", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Replace(%header, %old_header, %new_header, %limit) : (memref<4xi64>, memref<4xi64>, memref<4xi64>, i64) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_Join(%sep_header: memref<4xi64> {ly.ownership.object_header}, %n: i64, %seq_items: memref<?xi64>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "join", ly.runtime.result_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Join(%sep_header, %n, %seq_items) : (memref<4xi64>, i64, memref<?xi64>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }

  func.func @LyByteArray_Split(%header: memref<4xi64> {ly.ownership.object_header}, %sep_header: memref<4xi64> {ly.ownership.object_header}, %maxsplit: i64 {ly.runtime.default_i64 = -1 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_Split(%header, %sep_header, %maxsplit) : (memref<4xi64>, memref<4xi64>, i64) -> memref<5xi64>
    func.call @__ly_bytearray_convert_list(%r0) : (memref<5xi64>) -> ()
    func.return %r0 : memref<5xi64>
  }

  func.func @LyByteArray_SplitWS(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "split", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.bytearray"} {
    %r0 = func.call @LyBytes_SplitWS(%header) : (memref<4xi64>) -> memref<5xi64>
    func.call @__ly_bytearray_convert_list(%r0) : (memref<5xi64>) -> ()
    func.return %r0 : memref<5xi64>
  }

  func.func @LyByteArray_FromHex(%str_header: memref<2xi64> {ly.ownership.object_header}, %str_bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytearray", ly.runtime.initializer = "fromhex"} {
    %r0 = func.call @LyBytes_FromHex(%str_header, %str_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<4xi64>
    %converted = func.call @__ly_bytearray_from_bytes(%r0) : (memref<4xi64>) -> memref<4xi64>
    func.call @LyBytes_DecRef(%r0) : (memref<4xi64>) -> ()
    func.return %converted : memref<4xi64>
  }
}
