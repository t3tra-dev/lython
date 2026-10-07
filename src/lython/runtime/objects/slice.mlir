// `slice` -- CPython's Objects/sliceobject.c: the index arithmetic every
// sequence's `__getslice__` shares (PySlice_Unpack + PySlice_AdjustIndices).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_Str(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)

  py.class @slice attributes {
    base_names = ["object"], ly.typing.final,
    ly.typing.params = ["StartT", "StopT", "StepT"],
    method_names = ["indices"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.tuple">]>
    ],
    method_kinds = ["instance"]
  } {}

  memref.global "private" constant @__ly_slice_msg_zero_step : memref<25xi8> = dense<[115, 108, 105, 99, 101, 32, 115, 116, 101, 112, 32, 99, 97, 110, 110, 111, 116, 32, 98, 101, 32, 122, 101, 114, 111]>

  // ValueError("slice step cannot be zero") -- shared by every __getslice__.
  func.func private @__ly_slice_raise_zero_step() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 25 : i64
    %message_static = memref.get_global @__ly_slice_msg_zero_step : memref<25xi8>
    %message = memref.cast %message_static : memref<25xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython PySlice_Unpack + PySlice_AdjustIndices over a length: absent
  // bounds (mask bit0 = start present, bit1 = stop present) default by the
  // step's sign, explicit bounds normalize (+len) and clamp into the window
  // the sign allows. Returns (start, slicelength); the caller iterates
  // start, start+step, ... slicelength times.
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %minus_one = arith.constant -1 : i64
    %neg_step = arith.cmpi slt, %step, %zero : i64
    %len_minus_one = arith.subi %len, %one : i64
    %neg_floor = arith.select %neg_step, %minus_one, %zero : i1, i64
    %hi_clamp = arith.select %neg_step, %len_minus_one, %len : i1, i64

    %has_start_bit = arith.andi %mask, %one : i64
    %has_start = arith.cmpi ne, %has_start_bit, %zero : i64
    %s_plus = arith.addi %start_in, %len : i64
    %s_isneg = arith.cmpi slt, %start_in, %zero : i64
    %s_plus_isneg = arith.cmpi slt, %s_plus, %zero : i64
    %s_neg_val = arith.select %s_plus_isneg, %neg_floor, %s_plus : i1, i64
    %s_big = arith.cmpi sge, %start_in, %len : i64
    %s_pos_val = arith.select %s_big, %hi_clamp, %start_in : i1, i64
    %s_adj = arith.select %s_isneg, %s_neg_val, %s_pos_val : i1, i64
    %s_default = arith.select %neg_step, %len_minus_one, %zero : i1, i64
    %start = arith.select %has_start, %s_adj, %s_default : i1, i64

    %has_stop_bit = arith.andi %mask, %two : i64
    %has_stop = arith.cmpi ne, %has_stop_bit, %zero : i64
    %e_plus = arith.addi %stop_in, %len : i64
    %e_isneg = arith.cmpi slt, %stop_in, %zero : i64
    %e_plus_isneg = arith.cmpi slt, %e_plus, %zero : i64
    %e_neg_val = arith.select %e_plus_isneg, %neg_floor, %e_plus : i1, i64
    %e_big = arith.cmpi sge, %stop_in, %len : i64
    %e_pos_val = arith.select %e_big, %hi_clamp, %stop_in : i1, i64
    %e_adj = arith.select %e_isneg, %e_neg_val, %e_pos_val : i1, i64
    %e_default = arith.select %neg_step, %minus_one, %len : i1, i64
    %stop = arith.select %has_stop, %e_adj, %e_default : i1, i64

    %pos_diff = arith.subi %stop, %start : i64
    %neg_diff = arith.subi %start, %stop : i64
    %pos_nonempty = arith.cmpi slt, %start, %stop : i64
    %neg_nonempty = arith.cmpi slt, %stop, %start : i64
    %neg_step_abs = arith.subi %zero, %step : i64
    %step_abs = arith.select %neg_step, %neg_step_abs, %step : i1, i64
    %diff = arith.select %neg_step, %neg_diff, %pos_diff : i1, i64
    %diff_m1 = arith.subi %diff, %one : i64
    %quot = arith.divsi %diff_m1, %step_abs : i64
    %count_raw = arith.addi %quot, %one : i64
    %nonempty = arith.select %neg_step, %neg_nonempty, %pos_nonempty : i1, i1
    %count = arith.select %nonempty, %count_raw, %zero : i1, i64
    func.return %start, %count : i64, i64
  }

  // "attempt to assign sequence of size "
  memref.global "private" constant @__ly_slice_msg_extended_prefix : memref<35xi8> = dense<[97, 116, 116, 101, 109, 112, 116, 32, 116, 111, 32, 97, 115, 115, 105, 103, 110, 32, 115, 101, 113, 117, 101, 110, 99, 101, 32, 111, 102, 32, 115, 105, 122, 101, 32]>
  // " to extended slice of size "
  memref.global "private" constant @__ly_slice_msg_extended_middle : memref<27xi8> = dense<[32, 116, 111, 32, 101, 120, 116, 101, 110, 100, 101, 100, 32, 115, 108, 105, 99, 101, 32, 111, 102, 32, 115, 105, 122, 101, 32]>

  // ValueError for `a[i:j:k] = xs` with k != 1 and len(xs) != slicelength,
  // message text matching CPython list_ass_subscript.
  func.func private @__ly_slice_raise_extended_mismatch(%src_len: i64, %slice_len: i64) {
    %class_id = arith.constant 53 : i64
    %start = arith.constant 0 : index
    %prefix_static = memref.get_global @__ly_slice_msg_extended_prefix : memref<35xi8>
    %prefix = memref.cast %prefix_static : memref<35xi8> to memref<?xi8>
    %prefix_len = arith.constant 35 : i64
    %middle_static = memref.get_global @__ly_slice_msg_extended_middle : memref<27xi8>
    %middle = memref.cast %middle_static : memref<27xi8> to memref<?xi8>
    %middle_len = arith.constant 27 : i64
    %ph, %pb = func.call @LyUnicode_FromBytes(%prefix, %start, %prefix_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %src_int = func.call @LyLong_FromI64(%src_len) : (i64) -> memref<2xi64>
    %src_str:2 = func.call @LyLong_Str(%src_int) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi8>)
    %m1:2 = func.call @LyUnicode_Concat(%ph, %pb, %src_str#0, %src_str#1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %mh, %mb = func.call @LyUnicode_FromBytes(%middle, %start, %middle_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %m2:2 = func.call @LyUnicode_Concat(%m1#0, %m1#1, %mh, %mb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %cnt_int = func.call @LyLong_FromI64(%slice_len) : (i64) -> memref<2xi64>
    %cnt_str:2 = func.call @LyLong_Str(%cnt_int) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi8>)
    %m3:2 = func.call @LyUnicode_Concat(%m2#0, %m2#1, %cnt_str#0, %cnt_str#1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %m3#0, %m3#1) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }
}
