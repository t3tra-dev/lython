// `bool` -- CPython's Objects/boolobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.bool"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  memref.global "private" constant @__ly_fmt_msg_name_bool : memref<4xi8>
  func.func private @__ly_fmt_parse_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %out: memref<?xi64>) -> i1
  func.func private @__ly_fmt_raise_invalid_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %name: memref<?xi8>, %name_len: i64)
  func.func private @__ly_long_format_impl(%header: memref<2xi64>, %spec: memref<?xi64>, %tname: memref<?xi8>, %tname_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64

  py.class @bool attributes {
    base_names = ["int"], ly.typing.final,
    method_names = ["__new__", "__repr__", "__str__", "__bool__", "__and__",
                    "__or__", "__xor__", "__hash__", "__format__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.bool">>, !py.contract<"builtins.object">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.bool">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.bool">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance"]
  } {}

  memref.global "private" constant @__ly_bool_repr_true : memref<4xi8> = dense<[84, 114, 117, 101]>
  memref.global "private" constant @__ly_bool_repr_false : memref<5xi8> = dense<[70, 97, 108, 115, 101]>

  func.func private @LyBool_Shape() -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.shape}

  func.func @LyBool_Repr(%value: i1) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %result:2 = scf.if %value -> (memref<2xi64>, memref<?xi8>) {
      %true_static = memref.get_global @__ly_bool_repr_true : memref<4xi8>
      %true_bytes = memref.cast %true_static : memref<4xi8> to memref<?xi8>
      %true_len = arith.constant 4 : i64
      %header, %bytes = func.call @__ly_unicode_from_valid_utf8(%true_bytes, %c0, %true_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    } else {
      %false_static = memref.get_global @__ly_bool_repr_false : memref<5xi8>
      %false_bytes = memref.cast %false_static : memref<5xi8> to memref<?xi8>
      %false_len = arith.constant 5 : i64
      %header, %bytes = func.call @__ly_unicode_from_valid_utf8(%false_bytes, %c0, %false_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // Boxed bool: two immortal singletons (CPython's True/False semantics).
  // Layout follows the shared header contract: [refcount, class_id, value].
  // The refcount starts at the immortal marker; generic retains/releases may
  // drift it but it never reaches zero, and the deallocator is a no-op.
  memref.global "private" @__ly_bool_box_true : memref<3xi64> = dense<[9223372036854775807, 22, 1]>
  memref.global "private" @__ly_bool_box_false : memref<3xi64> = dense<[9223372036854775807, 22, 0]>

  // box: canonical i1 -> the boxed singleton (no allocation).
  func.func @LyBool_Box(%value: i1) -> memref<3xi64> attributes {ly.runtime.class_id = 22 : i64, ly.runtime.contract = "builtins.bool", ly.runtime.primitive = "box"} {
    %true_box = memref.get_global @__ly_bool_box_true : memref<3xi64>
    %false_box = memref.get_global @__ly_bool_box_false : memref<3xi64>
    %box = arith.select %value, %true_box, %false_box : memref<3xi64>
    func.return %box : memref<3xi64>
  }

  // unbox: boxed singleton -> canonical i1.
  func.func @LyBool_Unbox(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.primitive = "unbox"} {
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %word = memref.load %header[%c2] : memref<3xi64>
    %value = arith.cmpi ne, %word, %c0_i64 : i64
    func.return %value : i1
  }

  // Boxed-conforming __repr__ (erased-element dispatch through the repr hook).
  func.func @LyBool_BoxedRepr(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %value = func.call @LyBool_Unbox(%header) : (memref<3xi64>) -> i1
    %result_header, %result_bytes = func.call @LyBool_Repr(%value) : (i1) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // Immortal singletons never deallocate; the release hook still needs a
  // conforming deallocator so boxed slots release without a miss.
  func.func @LyBool_DecRef(%header: memref<3xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bool", ly.runtime.deallocator} {
    func.return
  }

  func.func @LyBool_Str(%value: i1) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %header, %bytes = func.call @LyBool_Repr(%value) : (i1) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // bool.__bool__ / __and__ / __or__ / __xor__. CPython's bool_and/or/xor
  // return bool only when *both* operands are bool and otherwise defer to
  // long_and/or/xor; the mixed cases are already int methods here, so these
  // four cover exactly the bool-bool shape the contract declares. Not folded
  // into the int primitive path: an i1 pair never reaches the boxed-long
  // arithmetic that primitiveI64ArithmeticKind selects.
  func.func @LyBool_Bool(%value: i1) -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.method = "__bool__"} {
    func.return %value : i1
  }

  func.func @LyBool_And(%lhs: i1, %rhs: i1) -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.method = "__and__"} {
    %result = arith.andi %lhs, %rhs : i1
    func.return %result : i1
  }

  func.func @LyBool_Or(%lhs: i1, %rhs: i1) -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.method = "__or__"} {
    %result = arith.ori %lhs, %rhs : i1
    func.return %result : i1
  }

  func.func @LyBool_Xor(%lhs: i1, %rhs: i1) -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.method = "__xor__"} {
    %result = arith.xori %lhs, %rhs : i1
    func.return %result : i1
  }

  func.func @LyBool_Format(%value: i1, %spec_header: memref<2xi64> {ly.ownership.object_header}, %spec_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__format__", ly.runtime.result_contract = "builtins.str"} {
    %n = func.call @__ly_unicode_count(%spec_header, %spec_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %empty = arith.cmpi eq, %n, %zero : i64
    cf.cond_br %empty, ^as_str, ^as_int

  ^as_str:
    %sh, %sb = func.call @LyBool_Str(%value) : (i1) -> (memref<2xi64>, memref<?xi8>)
    func.return %sh, %sb : memref<2xi64>, memref<?xi8>

  ^as_int:
    %one = arith.constant 1 : i64
    %iv = arith.select %value, %one, %zero : i64
    %ih = func.call @LyLong_FromI64(%iv) : (i64) -> memref<2xi64>
    %im, %id = func.call @__ly_long_parts(%ih) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %spec_store = memref.alloca() : memref<10xi64>
    %spec = memref.cast %spec_store : memref<10xi64> to memref<?xi64>
    %ok = func.call @__ly_fmt_parse_spec(%spec_header, %spec_bytes, %spec) : (memref<2xi64>, memref<?xi8>, memref<?xi64>) -> i1
    %true_bf = arith.constant true
    %bad = arith.xori %ok, %true_bf : i1
    scf.if %bad {
      %names = memref.get_global @__ly_fmt_msg_name_bool : memref<4xi8>
      %name = memref.cast %names : memref<4xi8> to memref<?xi8>
      %nlen = arith.constant 4 : i64
      // The int goes before the raise: the release below is on the path this one
      // replaces, and the raise does not return. Found by
      // RuntimeRaisePathTests -- a text scan missed it because `%ih, %im, %id =`
      // is a multi-result definition and the pattern only matched single results.
      func.call @LyLong_DecRef(%ih) : (memref<2xi64>) -> ()
      func.call @__ly_fmt_raise_invalid_spec(%spec_header, %spec_bytes, %name, %nlen) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> ()
    }
    %names2 = memref.get_global @__ly_fmt_msg_name_bool : memref<4xi8>
    %name2 = memref.cast %names2 : memref<4xi8> to memref<?xi8>
    %nlen2 = arith.constant 4 : i64
    %h, %b = func.call @__ly_long_format_impl(%ih, %spec, %name2, %nlen2) : (memref<2xi64>, memref<?xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyLong_DecRef(%ih) : (memref<2xi64>) -> ()
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // bool.__hash__ (canonical i1 receiver and boxed-conforming variants):
  // hash(False) == 0, hash(True) == 1 == hash(1).
  func.func @LyBool_Hash(%value: i1) -> i64 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.method = "__hash__"} {
    %h = arith.extui %value : i1 to i64
    func.return %h : i64
  }

  func.func @LyBool_BoxedHash(%header: memref<3xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.method = "__hash__"} {
    %value = func.call @LyBool_Unbox(%header) : (memref<3xi64>) -> i1
    %h = arith.extui %value : i1 to i64
    func.return %h : i64
  }
}
