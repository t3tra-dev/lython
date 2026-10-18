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
// the name (word 3, "__main__.A" or "int"), identity, its hash and `repr`, and
// the subclass walk over its MRO array (word 5). Construction and class
// attributes are not here: which class a value is decides them, and the
// emitter dispatches on that.
//
// Deviations from CPython:
// - `__qualname__` is `__name__`: a class here is always a module-level
//   statement, so the two are the same string.
// - the hash is CPython's `_Py_HashPointer` of the image address, so it is
//   stable for a run and differs between runs, as CPython's does.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.type"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyType_Name(%class_word: i64) -> !llvm.ptr
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

  // A class value is a view of its type object's fifteen words.
  // ⛔ `ly.runtime.class` is what makes this contract a runtime CLASS to the
  // manifest index, and the boxed hooks place their arms on that class's type
  // object: without it a class value held in a container renders through no
  // arm at all.
  func.func private @LyType_Shape() -> memref<15xi64> attributes {ly.runtime.class, ly.runtime.contract = "builtins.type", ly.runtime.shape}

  // The class word: the image's address, which is what an instance's header
  // word 1 holds for the same class.
  func.func private @__ly_type_word(%type: memref<15xi64> {ly.ownership.object_header}) -> i64 {
    %index = memref.extract_aligned_pointer_as_index %type : memref<15xi64> -> index
    %word = arith.index_cast %index : index to i64
    func.return %word : i64
  }

  memref.global "private" constant @__ly_type_repr_prefix : memref<8xi8> = dense<[60, 99, 108, 97, 115, 115, 32, 39]>
  memref.global "private" constant @__ly_type_repr_suffix : memref<2xi8> = dense<[39, 62]>
  memref.global "private" constant @__ly_type_builtins_module : memref<8xi8> = dense<[98, 117, 105, 108, 116, 105, 110, 115]>

  // `<class 'NAME'>`, NAME being the type object's own name -- qualified for
  // a class with a module ("__main__.A"), bare for a builtin ("int"), which is
  // CPython's type_repr.
  func.func @LyType_Repr(%type: memref<15xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %word = func.call @__ly_type_word(%type) : (memref<15xi64>) -> i64
    %name_ptr = func.call @LyType_Name(%word) : (i64) -> !llvm.ptr
    %length = memref.load %type[%c4] : memref<15xi64>
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
  func.func @LyType_Str(%type: memref<15xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %h, %b = func.call @LyType_Repr(%type) : (memref<15xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // CPython's _Py_HashPointer: the address rotated right by four, -1 being
  // reserved for "error".
  func.func @LyType_Hash(%type: memref<15xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.type", ly.runtime.method = "__hash__"} {
    %word = func.call @__ly_type_word(%type) : (memref<15xi64>) -> i64
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
  func.func @LyType_Eq(%lhs: memref<15xi64> {ly.ownership.object_header}, %rhs: memref<15xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.type", ly.runtime.method = "__eq__"} {
    %a = func.call @__ly_type_word(%lhs) : (memref<15xi64>) -> i64
    %b = func.call @__ly_type_word(%rhs) : (memref<15xi64>) -> i64
    %same = arith.cmpi eq, %a, %b : i64
    func.return %same : i1
  }

  func.func @LyType_Ne(%lhs: memref<15xi64> {ly.ownership.object_header}, %rhs: memref<15xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.type", ly.runtime.method = "__ne__"} {
    %a = func.call @__ly_type_word(%lhs) : (memref<15xi64>) -> i64
    %b = func.call @__ly_type_word(%rhs) : (memref<15xi64>) -> i64
    %differ = arith.cmpi ne, %a, %b : i64
    func.return %differ : i1
  }

  // `cls.__name__`: the name after the last dot.
  func.func @LyType_GetName(%type: memref<15xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.primitive = "name", ly.runtime.result_contract = "builtins.str"} {
    %word = func.call @__ly_type_word(%type) : (memref<15xi64>) -> i64
    %h, %b = func.call @LyObject_ClassNameFromId(%word) : (i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // `cls.__module__`: the name before the last dot, or "builtins" for a class
  // whose name has none.
  func.func @LyType_GetModule(%type: memref<15xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.type", ly.runtime.primitive = "module", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dot = arith.constant 46 : i8
    %word = func.call @__ly_type_word(%type) : (memref<15xi64>) -> i64
    %name_ptr = func.call @LyType_Name(%word) : (i64) -> !llvm.ptr
    %length = memref.load %type[%c4] : memref<15xi64>
    %length_index = arith.index_cast %length : i64 to index
    %last_dot = scf.for %i = %c0 to %length_index step %c1 iter_args(%found = %length_index) -> (index) {
      %i_i64 = arith.index_cast %i : index to i64
      %slot = llvm.getelementptr %name_ptr[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      %is_dot = arith.cmpi eq, %byte, %dot : i8
      %next = arith.select %is_dot, %i, %found : index
      scf.yield %next : index
    }
    %has_module = arith.cmpi ult, %last_dot, %length_index : index
    cf.cond_br %has_module, ^qualified, ^builtin

  ^qualified:
    %buffer = memref.alloc(%last_dot) : memref<?xi8>
    scf.for %i = %c0 to %last_dot step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %slot = llvm.getelementptr %name_ptr[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }
    %module_len = arith.index_cast %last_dot : index to i64
    %qh, %qb = func.call @__ly_unicode_from_valid_utf8(%buffer, %c0, %module_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %qh, %qb : memref<2xi64>, memref<?xi8>

  ^builtin:
    %builtins = memref.get_global @__ly_type_builtins_module : memref<8xi8>
    %builtins_dyn = memref.cast %builtins : memref<8xi8> to memref<?xi8>
    %eight = arith.constant 8 : i64
    %bh, %bb = func.call @__ly_unicode_from_valid_utf8(%builtins_dyn, %c0, %eight) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %bh, %bb : memref<2xi64>, memref<?xi8>
  }

  // `issubclass(sub, base)` for two class values, over the class words
  // (CPython's PyType_IsSubtype walks tp_mro; so does LyType_IsSubtype).
  func.func @LyType_SubclassCheck(%sub: memref<15xi64> {ly.ownership.object_header}, %base: memref<15xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.type", ly.runtime.primitive = "is_subtype"} {
    %sub_word = func.call @__ly_type_word(%sub) : (memref<15xi64>) -> i64
    %base_word = func.call @__ly_type_word(%base) : (memref<15xi64>) -> i64
    %answer = func.call @LyType_IsSubtype(%sub_word, %base_word) : (i64, i64) -> i1
    func.return %answer : i1
  }
}
