// `type` -- CPython's Objects/typeobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// A class is its type object (lowering/Common/TypeObjects.h): an immortal
// static whose word 1 is `type`'s own type object, so a class VALUE is a
// reference to that image like a reference to any other object. Its refcount
// word is INT64_MAX, which retain and release skip, so holding one costs no
// write.
//
// What is implemented is what a class value answers by itself, from its image:
// the name (word 3, "__main__.f.<locals>.A" or "int", and word 4, which says
// how much of it is the module), identity, its hash and `repr`, and the
// subclass walk over its MRO array (word 5). Construction and class
// attributes are not here: which class a value is decides them, and the
// emitter dispatches on that.
//
// Deviations from CPython:
// - the hash is CPython's `_Py_HashPointer` of the image address, so it is
//   stable for a run and differs between runs, as CPython's does.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.type"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyType_Name(%class_word: i64) -> !llvm.ptr
  func.func private @LyType_NameLength(%class_word: i64) -> i64
  func.func private @LyType_ModuleLength(%class_word: i64) -> i64
  func.func private @LyType_IsSubtype(%class: i64, %base: i64) -> i1
  func.func private @LyObject_ClassNameFromId(%class_word: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "class_name_from_id", ly.runtime.result_contract = "builtins.str"}

  py.class @type attributes {
    base_names = ["object"], ly.typing.params = ["T", "P"],
    method_names = ["__call__", "__new__", "__or__", "__ror__", "__repr__",
                    "__str__", "__hash__", "__eq__", "__ne__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"$T">>, !py.paramspec<"P">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.type">>, !py.contract<"builtins.object">] -> [!py.type<!py.contract<"builtins.object">>]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.object">>, !py.contract<"typing.Any">] -> [!py.contract<"types.UnionType">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.object">>, !py.contract<"typing.Any">] -> [!py.contract<"types.UnionType">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.type">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.type">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.type">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.type">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.type">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "classmethod", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance"]
  } {}

  // A class value is a view of its type object's words (TypeObjects.h).
  // ⛔ `ly.runtime.class` is what makes this contract a runtime CLASS to the
  // manifest index, and the boxed hooks place their arms on that class's type
  // object: without it a class value held in a container renders through no
  // arm at all.
  func.func private @LyType_Shape() -> memref<16xi64> attributes {ly.runtime.class, ly.runtime.contract = "builtins.type", ly.runtime.shape}

  // The class word: the image's address, which is what an instance's header
  // word 1 holds for the same class.
  func.func private @__ly_type_word(%type: memref<16xi64> {ly.ownership.object_header}) -> i64 {
    %index = memref.extract_aligned_pointer_as_index %type : memref<16xi64> -> index
    %word = arith.index_cast %index : index to i64
    func.return %word : i64
  }

  memref.global "private" constant @__ly_type_repr_prefix : memref<8xi8> = dense<[60, 99, 108, 97, 115, 115, 32, 39]>
  memref.global "private" constant @__ly_type_repr_suffix : memref<2xi8> = dense<[39, 62]>
  memref.global "private" constant @__ly_type_builtins_module : memref<8xi8> = dense<[98, 117, 105, 108, 116, 105, 110, 115]>

  // `<class 'NAME'>`, NAME being the type object's own name -- qualified for
  // a class with a module ("__main__.A"), bare for a builtin ("int"), which is
  // CPython's type_repr.
  func.func @LyType_Repr(%type: memref<16xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %word = func.call @__ly_type_word(%type) : (memref<16xi64>) -> i64
    %name_ptr = func.call @LyType_Name(%word) : (i64) -> !llvm.ptr
    %length = func.call @LyType_NameLength(%word) : (i64) -> i64
    %length_index = arith.index_cast %length : i64 to index
    %total = arith.addi %length_index, %c10 : index
    %buffer = memref.alloc(%total) : memref<?xi8>
    %prefix = memref.get_global @__ly_type_repr_prefix : memref<8xi8>
    scf.for %i = %c0 to %c8 step %c1 {
      %byte = memref.load %prefix[%i] : memref<8xi8>
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }
    scf.for %i = %c0 to %length_index step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %slot = llvm.getelementptr %name_ptr[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      %at = arith.addi %i, %c8 : index
      memref.store %byte, %buffer[%at] : memref<?xi8>
    }
    %suffix = memref.get_global @__ly_type_repr_suffix : memref<2xi8>
    %tail = arith.addi %length_index, %c8 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %byte = memref.load %suffix[%i] : memref<2xi8>
      %at = arith.addi %i, %tail : index
      memref.store %byte, %buffer[%at] : memref<?xi8>
    }
    %total_i64 = arith.index_cast %total : index to i64
    %h, %b = func.call @__ly_unicode_from_valid_utf8(%buffer, %c0, %total_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // `str(cls)` is its repr: type defines no __str__ of its own.
  func.func @LyType_Str(%type: memref<16xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %h, %b = func.call @LyType_Repr(%type) : (memref<16xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // CPython's _Py_HashPointer: the address rotated right by four, -1 being
  // reserved for "error".
  func.func @LyType_Hash(%type: memref<16xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.type", ly.runtime.method = "__hash__"} {
    %word = func.call @__ly_type_word(%type) : (memref<16xi64>) -> i64
    %four = arith.constant 4 : i64
    %sixty = arith.constant 60 : i64
    %low = arith.shrui %word, %four : i64
    %high = arith.shli %word, %sixty : i64
    %rotated = arith.ori %low, %high : i64
    %minus_one = arith.constant -1 : i64
    %minus_two = arith.constant -2 : i64
    %is_reserved = arith.cmpi eq, %rotated, %minus_one : i64
    %hash = arith.select %is_reserved, %minus_two, %rotated : i64
    func.return %hash : i64
  }

  // Two class values are equal exactly when they are the same class: type
  // compares by identity.
  func.func @LyType_Eq(%lhs: memref<16xi64> {ly.ownership.object_header}, %rhs: memref<16xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.type", ly.runtime.method = "__eq__"} {
    %a = func.call @__ly_type_word(%lhs) : (memref<16xi64>) -> i64
    %b = func.call @__ly_type_word(%rhs) : (memref<16xi64>) -> i64
    %same = arith.cmpi eq, %a, %b : i64
    func.return %same : i1
  }

  func.func @LyType_Ne(%lhs: memref<16xi64> {ly.ownership.object_header}, %rhs: memref<16xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.type", ly.runtime.method = "__ne__"} {
    %a = func.call @__ly_type_word(%lhs) : (memref<16xi64>) -> i64
    %b = func.call @__ly_type_word(%rhs) : (memref<16xi64>) -> i64
    %differ = arith.cmpi ne, %a, %b : i64
    func.return %differ : i1
  }

  // `cls.__name__`: the name after the last dot.
  func.func @LyType_GetName(%type: memref<16xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.primitive = "name", ly.runtime.result_contract = "builtins.str"} {
    %word = func.call @__ly_type_word(%type) : (memref<16xi64>) -> i64
    %h, %b = func.call @LyObject_ClassNameFromId(%word) : (i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // The bytes [start, start + length) of the class's name, as a str.
  func.func private @__ly_type_name_slice(%word: i64, %start: i64, %length: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %name_ptr = func.call @LyType_Name(%word) : (i64) -> !llvm.ptr
    %length_index = arith.index_cast %length : i64 to index
    %buffer = memref.alloc(%length_index) : memref<?xi8>
    scf.for %i = %c0 to %length_index step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %at = arith.addi %i_i64, %start : i64
      %slot = llvm.getelementptr %name_ptr[%at] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }
    %h, %b = func.call @__ly_unicode_from_valid_utf8(%buffer, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // `cls.__module__`: the part of the name word 4 says is the module, or
  // "builtins" for a class whose name has none.
  func.func @LyType_GetModule(%type: memref<16xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.primitive = "module", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %word = func.call @__ly_type_word(%type) : (memref<16xi64>) -> i64
    %module_len = func.call @LyType_ModuleLength(%word) : (i64) -> i64
    %has_module = arith.cmpi ne, %module_len, %zero : i64
    cf.cond_br %has_module, ^qualified, ^builtin

  ^qualified:
    %qh, %qb = func.call @__ly_type_name_slice(%word, %zero, %module_len) : (i64, i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %qh, %qb : memref<2xi64>, memref<?xi8>

  ^builtin:
    %builtins = memref.get_global @__ly_type_builtins_module : memref<8xi8>
    %builtins_dyn = memref.cast %builtins : memref<8xi8> to memref<?xi8>
    %eight = arith.constant 8 : i64
    %bh, %bb = func.call @__ly_unicode_from_valid_utf8(%builtins_dyn, %c0, %eight) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %bh, %bb : memref<2xi64>, memref<?xi8>
  }

  // `cls.__qualname__`: the name past its module and the dot after it --
  // "f.<locals>.A" of "__main__.f.<locals>.A" -- or all of it for a builtin.
  func.func @LyType_GetQualname(%type: memref<16xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.primitive = "qualname", ly.runtime.result_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %word = func.call @__ly_type_word(%type) : (memref<16xi64>) -> i64
    %length = func.call @LyType_NameLength(%word) : (i64) -> i64
    %module_len = func.call @LyType_ModuleLength(%word) : (i64) -> i64
    %has_module = arith.cmpi ne, %module_len, %zero : i64
    %past_dot = arith.addi %module_len, %one : i64
    %start = arith.select %has_module, %past_dot, %zero : i64
    %rest = arith.subi %length, %start : i64
    %h, %b = func.call @__ly_type_name_slice(%word, %start, %rest) : (i64, i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // `issubclass(sub, base)` for two class values, over the class words
  // (CPython's PyType_IsSubtype walks tp_mro; so does LyType_IsSubtype).
  func.func @LyType_SubclassCheck(%sub: memref<16xi64> {ly.ownership.object_header}, %base: memref<16xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.type", ly.runtime.primitive = "is_subtype"} {
    %sub_word = func.call @__ly_type_word(%sub) : (memref<16xi64>) -> i64
    %base_word = func.call @__ly_type_word(%base) : (memref<16xi64>) -> i64
    %answer = func.call @LyType_IsSubtype(%sub_word, %base_word) : (i64, i64) -> i1
    func.return %answer : i1
  }
}
