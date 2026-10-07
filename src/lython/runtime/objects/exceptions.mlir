// The exception hierarchy -- CPython's Objects/exceptions.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// Deviations from CPython:
//   - An empty message and no message share one payload: `repr(X(''))`
//     renders as `X()` and `X('').args` is `()`.
//   - The codec errors take their codec arguments and nothing else:
//     `UnicodeDecodeError("message")` is refused at compile time where CPython
//     raises TypeError when it runs, and a decode error's `object` is bytes
//     rather than any buffer.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.BaseException", "builtins.Exception", "builtins.RuntimeError", "builtins.TypeError", "builtins.ValueError", "builtins.ArithmeticError", "builtins.LookupError", "builtins.ZeroDivisionError", "builtins.KeyError", "builtins.IndexError", "builtins.AssertionError", "builtins.StopIteration", "builtins.StopAsyncIteration", "builtins.SystemExit", "builtins.GeneratorExit", "builtins.OSError", "builtins.FileNotFoundError", "builtins.KeyboardInterrupt", "builtins.BaseExceptionGroup", "builtins.ExceptionGroup", "builtins.FloatingPointError", "builtins.OverflowError", "builtins.BufferError", "builtins.EOFError", "builtins.ImportError", "builtins.ModuleNotFoundError", "builtins.MemoryError", "builtins.NameError", "builtins.UnboundLocalError", "builtins.AttributeError", "builtins.ReferenceError", "builtins.NotImplementedError", "builtins.RecursionError", "builtins.PythonFinalizationError", "builtins.SyntaxError", "builtins.IndentationError", "builtins.TabError", "builtins.SystemError", "builtins.UnicodeError", "builtins.UnicodeDecodeError", "builtins.UnicodeEncodeError", "builtins.UnicodeTranslateError", "builtins.Warning", "builtins.BytesWarning", "builtins.DeprecationWarning", "builtins.EncodingWarning", "builtins.FutureWarning", "builtins.ImportWarning", "builtins.PendingDeprecationWarning", "builtins.ResourceWarning", "builtins.RuntimeWarning", "builtins.SyntaxWarning", "builtins.UnicodeWarning", "builtins.UserWarning", "builtins.BlockingIOError", "builtins.ChildProcessError", "builtins.ConnectionError", "builtins.BrokenPipeError", "builtins.ConnectionAbortedError", "builtins.ConnectionRefusedError", "builtins.ConnectionResetError", "builtins.FileExistsError", "builtins.InterruptedError", "builtins.IsADirectoryError", "builtins.NotADirectoryError", "builtins.PermissionError", "builtins.ProcessLookupError", "builtins.TimeoutError"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_slot_word_is_immediate(%word: i64) -> i1
  func.func private @__ly_int_from_immediate(%word: i64) -> i64
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64
  func.func private @__ly_bytes_payload(%self: memref<4xi64>) -> memref<?xi8> attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.interior_word, ly.runtime.primitive = "payload_view"}
  func.func private @LyLong_SlotWordAsI64(%word: i64) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_as_i64"}
  func.func private @LyLong_SlotWordFromI64(%value: i64) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_from_i64"}
  func.func private @__ly_unicode_from_hex(%value: i64, %digits: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyBool_Unbox(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.primitive = "unbox"}
  func.func private @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyObject_RetainBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyTuple_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 11 : i64, ly.runtime.contract = "builtins.tuple", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.tuple"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"}
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"}
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_entity_word_get(%ptr: i64, %slot: i64) -> i64
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_global_view_i8(%pointer: i64, %size: i64) -> memref<?xi8>
  func.func private @__ly_handle_retain_raw(%entity: i64)
  func.func private @__ly_repr_boxed_by_contract(%box: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>, i1) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  memref.global "private" constant @__ly_repr_comma : memref<2xi8>
  memref.global "private" constant @__ly_repr_lbracket : memref<1xi8>
  memref.global "private" constant @__ly_repr_lparen : memref<1xi8>
  memref.global "private" constant @__ly_repr_rbracket : memref<1xi8>
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8>
  func.func private @__ly_slot_class(%word: i64) -> i64
  func.func private @__ly_str_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "str_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  func.func private @__ly_tuple_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_unicode_alloc(%count: i64, %width: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "alloc"}
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_data_offset() -> i64
  func.func private @__ly_unicode_lane_words(%hdr_ptr: i64) -> (i64, i64) attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "lane_words"}
  func.func private @__ly_unicode_put(%bytes: memref<?xi8>, %width: i64, %i: index, %cp: i64)
  func.func private @__ly_unicode_raw_bytes(%hdr_ptr: i64) -> i64
  func.func private @__ly_unicode_retain_self(%header: memref<2xi64>, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_unicode_store_item(%items: memref<?xi64>, %slot: i64, %eh: memref<2xi64> {ly.ownership.object_header}, %eb: memref<?xi8>) attributes {ly.ownership.transfer_args = [2]}

  py.class @BaseException attributes {
    base_names = ["object"],
    field_names = ["args", "__cause__", "__context__", "__suppress_context__",
                   "__traceback__"],
    field_contract_types = [
      !py.contract<"builtins.tuple">,
      !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>,
      !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>,
      !py.contract<"builtins.bool">,
      !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>
    ],
    method_names = ["__init__", "with_traceback", "__str__", "__repr__",
                    "add_note"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">, !py.paramspec<"P">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.BaseException">, !py.contract<"builtins.str">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}
  py.class @Exception attributes {base_names = ["BaseException"]} {}
  py.class @RuntimeError attributes {base_names = ["Exception"]} {}
  py.class @TypeError attributes {base_names = ["Exception"]} {}
  py.class @ValueError attributes {base_names = ["Exception"]} {}
  py.class @ArithmeticError attributes {base_names = ["Exception"]} {}
  py.class @LookupError attributes {base_names = ["Exception"]} {}
  py.class @ZeroDivisionError attributes {base_names = ["ArithmeticError"]} {}
  py.class @KeyError attributes {base_names = ["LookupError"]} {}
  py.class @IndexError attributes {base_names = ["LookupError"]} {}
  py.class @AssertionError attributes {base_names = ["Exception"]} {}
  py.class @StopIteration attributes {
    base_names = ["Exception"],
    field_names = ["value"],
    field_contract_types = [!py.contract<"typing.Any">]
  } {}
  py.class @StopAsyncIteration attributes {base_names = ["Exception"]} {}
  // SystemExit derives from BaseException directly (never caught by
  // `except Exception`); the top-level runner converts it to the process
  // exit status instead of printing a traceback.
  py.class @SystemExit attributes {base_names = ["BaseException"]} {}
  // GeneratorExit derives from BaseException directly (never caught by
  // `except Exception`); generator.close() injects it at the suspension
  // point so the body's finally blocks run.
  py.class @GeneratorExit attributes {base_names = ["BaseException"]} {}
  py.class @OSError attributes {base_names = ["Exception"]} {}
  py.class @FileNotFoundError attributes {base_names = ["OSError"]} {}
  py.class @KeyboardInterrupt attributes {base_names = ["BaseException"]} {}
  py.class @BaseExceptionGroup attributes {
    base_names = ["BaseException"],
    field_names = ["message", "exceptions"],
    field_contract_types = [
      !py.contract<"builtins.str">,
      !py.contract<"builtins.tuple">
    ]
  } {}
  py.class @ExceptionGroup attributes {base_names = ["BaseExceptionGroup"]} {}
  py.class @FloatingPointError attributes {base_names = ["ArithmeticError"]} {}
  py.class @OverflowError attributes {base_names = ["ArithmeticError"]} {}
  py.class @BufferError attributes {base_names = ["Exception"]} {}
  py.class @EOFError attributes {base_names = ["Exception"]} {}
  py.class @ImportError attributes {base_names = ["Exception"]} {}
  py.class @ModuleNotFoundError attributes {base_names = ["ImportError"]} {}
  py.class @MemoryError attributes {base_names = ["Exception"]} {}
  py.class @NameError attributes {base_names = ["Exception"]} {}
  py.class @UnboundLocalError attributes {base_names = ["NameError"]} {}
  py.class @AttributeError attributes {base_names = ["Exception"]} {}
  py.class @ReferenceError attributes {base_names = ["Exception"]} {}
  py.class @NotImplementedError attributes {base_names = ["RuntimeError"]} {}
  py.class @RecursionError attributes {base_names = ["RuntimeError"]} {}
  py.class @PythonFinalizationError attributes {base_names = ["RuntimeError"]} {}
  py.class @SyntaxError attributes {base_names = ["Exception"]} {}
  py.class @IndentationError attributes {base_names = ["SyntaxError"]} {}
  py.class @TabError attributes {base_names = ["IndentationError"]} {}
  py.class @SystemError attributes {base_names = ["Exception"]} {}
  py.class @UnicodeError attributes {base_names = ["ValueError"]} {}
  // The codec errors take the codec's own arguments, as typeshed declares
  // them (`object` is bytes for a decode: the buffer protocol is not
  // modelled), and keep them as these attributes.
  py.class @UnicodeDecodeError attributes {
    base_names = ["UnicodeError"],
    field_names = ["encoding", "object", "start", "end", "reason"],
    field_contract_types = [
      !py.contract<"builtins.str">,
      !py.contract<"builtins.bytes">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.str">
    ],
    method_names = ["__init__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.UnicodeDecodeError">, !py.contract<"builtins.str">, !py.contract<"builtins.bytes">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @UnicodeEncodeError attributes {
    base_names = ["UnicodeError"],
    field_names = ["encoding", "object", "start", "end", "reason"],
    field_contract_types = [
      !py.contract<"builtins.str">,
      !py.contract<"builtins.str">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.str">
    ],
    method_names = ["__init__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.UnicodeEncodeError">, !py.contract<"builtins.str">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @UnicodeTranslateError attributes {
    base_names = ["UnicodeError"],
    field_names = ["object", "start", "end", "reason"],
    field_contract_types = [
      !py.contract<"builtins.str">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.str">
    ],
    method_names = ["__init__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.UnicodeTranslateError">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @Warning attributes {base_names = ["Exception"]} {}
  py.class @BytesWarning attributes {base_names = ["Warning"]} {}
  py.class @DeprecationWarning attributes {base_names = ["Warning"]} {}
  py.class @EncodingWarning attributes {base_names = ["Warning"]} {}
  py.class @FutureWarning attributes {base_names = ["Warning"]} {}
  py.class @ImportWarning attributes {base_names = ["Warning"]} {}
  py.class @PendingDeprecationWarning attributes {base_names = ["Warning"]} {}
  py.class @ResourceWarning attributes {base_names = ["Warning"]} {}
  py.class @RuntimeWarning attributes {base_names = ["Warning"]} {}
  py.class @SyntaxWarning attributes {base_names = ["Warning"]} {}
  py.class @UnicodeWarning attributes {base_names = ["Warning"]} {}
  py.class @UserWarning attributes {base_names = ["Warning"]} {}
  py.class @BlockingIOError attributes {base_names = ["OSError"]} {}
  py.class @ChildProcessError attributes {base_names = ["OSError"]} {}
  py.class @ConnectionError attributes {base_names = ["OSError"]} {}
  py.class @BrokenPipeError attributes {base_names = ["ConnectionError"]} {}
  py.class @ConnectionAbortedError attributes {base_names = ["ConnectionError"]} {}
  py.class @ConnectionRefusedError attributes {base_names = ["ConnectionError"]} {}
  py.class @ConnectionResetError attributes {base_names = ["ConnectionError"]} {}
  py.class @FileExistsError attributes {base_names = ["OSError"]} {}
  py.class @InterruptedError attributes {base_names = ["OSError"]} {}
  py.class @IsADirectoryError attributes {base_names = ["OSError"]} {}
  py.class @NotADirectoryError attributes {base_names = ["OSError"]} {}
  py.class @PermissionError attributes {base_names = ["OSError"]} {}
  py.class @ProcessLookupError attributes {base_names = ["OSError"]} {}
  py.class @TimeoutError attributes {base_names = ["OSError"]} {}

  func.func @LyBaseException_DecRef(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %header_ptr_index = memref.extract_aligned_pointer_as_index %header : memref<3xi64> -> index
    %header_word = arith.index_cast %header_ptr_index : index to i64
    %header_ptr = llvm.inttoptr %header_word : i64 to !llvm.ptr
    func.call @release_exception_extras(%header_ptr) : (!llvm.ptr) -> ()
    func.call @LyUnicode_DecRef(%message_header) : (memref<2xi64>) -> ()
    memref.dealloc %header : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // Extended exception words (see LyBaseException_New): [3] payload block
  // (ExceptionGroup members / multi-value args, as a [count, count x box16]
  // i64 block), [4] user-exception field block (same shape). Raw pointer
  // access on purpose -- the 3-word header view must not widen.
  func.func private @__ly_exc_ext_get(%header: memref<3xi64>, %slot: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "ext_get"} {
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<3xi64> -> index
    %ptr_word = arith.index_cast %ptr_index : index to i64
    %ptr = llvm.inttoptr %ptr_word : i64 to !llvm.ptr
    %slot_ptr = llvm.getelementptr %ptr[%slot] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %value = llvm.load %slot_ptr : !llvm.ptr -> i64
    func.return %value : i64
  }

  func.func private @__ly_exc_ext_set(%header: memref<3xi64>, %slot: i64, %value: i64) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "ext_set"} {
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<3xi64> -> index
    %ptr_word = arith.index_cast %ptr_index : index to i64
    %ptr = llvm.inttoptr %ptr_word : i64 to !llvm.ptr
    %slot_ptr = llvm.getelementptr %ptr[%slot] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %value, %slot_ptr : i64, !llvm.ptr
    func.return
  }

  // Records the message lane in the object (extended word 6). A cache, not a
  // reference: the lane's owner releases it, and this word is only how a reader
  // that holds the OBJECT alone finds the same string.
  func.func private @__ly_exc_set_message(%header: memref<3xi64>, %message_header: memref<2xi64>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "set_message"} {
    %slot = arith.constant 6 : i64
    %ptr_index = memref.extract_aligned_pointer_as_index %message_header : memref<2xi64> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    func.call @__ly_exc_ext_set(%header, %slot, %ptr) : (memref<3xi64>, i64, i64) -> ()
    func.return
  }

  func.func private @__ly_exc_lane_words(%obj_ptr: i64) -> (i64, i64, i64, i64) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "lane_words"} {
    %message_slot = arith.constant 6 : i64
    %two = arith.constant 2 : i64
    %msg_ptr = func.call @__ly_entity_word_get(%obj_ptr, %message_slot) : (i64, i64) -> i64
    // ⛔ Not followed when it is 0: the immortal dead header an absent union
    // member is read from carries no message, and a union read takes every
    // member's lanes before its tag selects one (`BaseException | None`
    // holding None segfaulted here).
    %zero = arith.constant 0 : i64
    %has_message = arith.cmpi ne, %msg_ptr, %zero : i64
    %bytes_ptr, %byte_len = scf.if %has_message -> (i64, i64) {
      %ptr, %len = func.call @__ly_unicode_lane_words(%msg_ptr) : (i64) -> (i64, i64)
      scf.yield %ptr, %len : i64, i64
    } else {
      scf.yield %zero, %zero : i64, i64
    }
    func.return %msg_ptr, %two, %bytes_ptr, %byte_len : i64, i64, i64, i64
  }

  // The message lanes of an exception, from the object alone.
  func.func private @__ly_exc_message_parts(%header: memref<3xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "message_parts", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 6 : i64
    %two = arith.constant 2 : i64
    %prefix = func.call @__ly_unicode_data_offset() : () -> i64
    %msg_ptr = func.call @__ly_exc_ext_get(%header, %slot) : (memref<3xi64>, i64) -> i64
    %words = func.call @__ly_global_view_i64(%msg_ptr, %two) : (i64, i64) -> memref<?xi64>
    %msg_header = memref.cast %words : memref<?xi64> to memref<2xi64>
    %bytes_ptr = arith.addi %msg_ptr, %prefix : i64
    %byte_len = func.call @__ly_unicode_raw_bytes(%msg_ptr) : (i64) -> i64
    %msg_bytes = func.call @__ly_global_view_i8(%bytes_ptr, %byte_len) : (i64, i64) -> memref<?xi8>
    func.return %msg_header, %msg_bytes : memref<2xi64>, memref<?xi8>
  }

  // Allocate a payload block for %count boxed entries: word 0 = count, then
  // count x 16 zeroed box words. Returned as a raw pointer word (the block is
  // reached only through the extended exception words).
  func.func private @__ly_exc_payload_alloc(%count: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_alloc"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %boxes = arith.muli %count, %c16 : i64
    %words = arith.addi %boxes, %one : i64
    %words_index = arith.index_cast %words : i64 to index
    %block = memref.alloc(%words_index) : memref<?xi64>
    scf.for %i = %c0 to %words_index step %c1 {
      memref.store %zero, %block[%i] : memref<?xi64>
    }
    %count_slot = arith.constant 0 : index
    memref.store %count, %block[%count_slot] : memref<?xi64>
    %ptr_index = memref.extract_aligned_pointer_as_index %block : memref<?xi64> -> index
    %ptr_word = arith.index_cast %ptr_index : index to i64
    func.return %ptr_word : i64
  }

  // Store one exception (header + message) into payload box %slot, retaining
  // the exception entity for the block (the caller keeps its own reference).
  // Box layout mirrors objectPayloadHandleWords (BoxLayout.h): word 1 is the
  // header's layout word (5 = the shared BaseException dispatch class), the
  // precise class id stays in the exception header itself.
  func.func private @__ly_exc_payload_store(%block_word: i64, %slot: i64, %eh: memref<3xi64>, %mh: memref<2xi64>, %mb: memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_store"} {
    %layout_slot = arith.constant 1 : index
    %layout = memref.load %eh[%layout_slot] : memref<3xi64>
    %eh_index = memref.extract_aligned_pointer_as_index %eh : memref<3xi64> -> index
    %eh_word = arith.index_cast %eh_index : index to i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    // The message lanes are not written: `__ly_exc_lane_words` reads them out
    // of the exception the entity word names.
    func.call @__ly_exc_payload_store_words(%block_word, %slot, %eh_word) : (i64, i64, i64) -> ()
    func.call @__ly_handle_retain_raw(%eh_word) : (i64) -> ()
    func.return
  }

  // Borrowed views of payload box %slot: the sub-exception triple, rebuilt
  // from the box's pointer/size words.
  func.func private @__ly_exc_payload_view(%block_word: i64, %slot: i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_view"} {
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %one = arith.constant 1 : i64
    %block_ptr = llvm.inttoptr %block_word : i64 to !llvm.ptr
    %boxes_base = arith.muli %slot, %c16 : i64
    %box_base = arith.addi %boxes_base, %one : i64
    %box_ptr = llvm.getelementptr %block_ptr[%box_base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // The member's message comes from the member, not from the box beside it.
    %entity_slot = arith.constant 0 : i64
    %w2 = llvm.getelementptr %box_ptr[%entity_slot] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %eh_word = llvm.load %w2 : !llvm.ptr -> i64
    %mh_word, %mh_size, %mb_word, %mb_len = func.call @__ly_exc_lane_words(%eh_word) : (i64) -> (i64, i64, i64, i64)
    %three = arith.constant 3 : i64
    %two = arith.constant 2 : i64
    %eh_dyn = func.call @__ly_global_view_i64(%eh_word, %three) : (i64, i64) -> memref<?xi64>
    %eh = memref.cast %eh_dyn : memref<?xi64> to memref<3xi64>
    %mh_dyn = func.call @__ly_global_view_i64(%mh_word, %two) : (i64, i64) -> memref<?xi64>
    %mh = memref.cast %mh_dyn : memref<?xi64> to memref<2xi64>
    %mb = func.call @__ly_global_view_i8(%mb_word, %mb_len) : (i64, i64) -> memref<?xi8>
    func.return %eh, %mh, %mb : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // Store one pre-built box (objectPayloadHandleWords layout) into
  // payload box %slot -- the multi-value exception args path boxes arbitrary
  // contracts in the lowering and hands the words across.
  // ⛔ THE ONE PLACE THE WIDTH IS STILL SPELLED OUT, and the only one where that
  // is safe: it takes the slot's words as separate arguments, so the width is
  // its ARITY -- and an arity that disagrees with `objectPayloadHandleWords` is
  // a verifier error at build time, not a wrong word at run time. A slot is
  // one word now (BoxLayout.h).
  func.func private @__ly_exc_payload_store_words(%block_word: i64, %slot: i64, %w0: i64) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_store_words"} {
    %words = func.call @__ly_box_word_count() : () -> i64
    %one = arith.constant 1 : i64
    %block_ptr = llvm.inttoptr %block_word : i64 to !llvm.ptr
    %boxes_base = arith.muli %slot, %words : i64
    %box_base = arith.addi %boxes_base, %one : i64
    %box_ptr = llvm.getelementptr %block_ptr[%box_base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %w0, %box_ptr : i64, !llvm.ptr
    func.return
  }

  // Store a str as boxed payload arg %slot: the same box layout
  // __ly_unicode_store_item writes into a tuple slot, aimed at the payload
  // block. The block owns one reference per slot, so the str is retained
  // here rather than by the caller.
  func.func private @__ly_exc_payload_store_unicode(%block: i64, %slot: i64, %eh: memref<2xi64> {ly.ownership.object_header}, %eb: memref<?xi8>) attributes {ly.ownership.transfer_args = [2]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %str_class = arith.constant 4 : i64
    %hdr_idx = memref.extract_aligned_pointer_as_index %eh : memref<2xi64> -> index
    %hdr_ptr = arith.index_cast %hdr_idx : index to i64
    func.call @__ly_exc_payload_store_words(%block, %slot, %hdr_ptr) : (i64, i64, i64) -> ()
    func.return
  }

  // Store an already-boxed object as payload arg %slot: copy the box's words
  // and retain the entity, the same pairing LyBaseException_Args uses when it
  // copies a slot back out into a tuple.
  func.func private @__ly_exc_payload_store_box(%block: i64, %slot: i64, %box: !llvm.ptr) {
    %v2 = llvm.load %box : !llvm.ptr -> i64
    func.call @__ly_exc_payload_store_words(%block, %slot, %v2) : (i64, i64, i64) -> ()
    func.call @__ly_handle_retain_raw(%v2) : (i64) -> ()
    func.return
  }

  // ===== the codec errors' messages (UnicodeDecodeError_str and its two
  // siblings in Objects/exceptions.c) =====
  // "'"
  memref.global "private" constant @__ly_exc_codec_quote : memref<1xi8> = dense<[39]>
  // "' codec can't decode byte 0x"
  memref.global "private" constant @__ly_exc_codec_decode_byte : memref<28xi8> = dense<[39, 32, 99, 111, 100, 101, 99, 32, 99, 97, 110, 39, 116, 32, 100, 101, 99, 111, 100, 101, 32, 98, 121, 116, 101, 32, 48, 120]>
  // "' codec can't decode bytes in position "
  memref.global "private" constant @__ly_exc_codec_decode_bytes : memref<39xi8> = dense<[39, 32, 99, 111, 100, 101, 99, 32, 99, 97, 110, 39, 116, 32, 100, 101, 99, 111, 100, 101, 32, 98, 121, 116, 101, 115, 32, 105, 110, 32, 112, 111, 115, 105, 116, 105, 111, 110, 32]>
  // "' codec can't encode character '\"
  memref.global "private" constant @__ly_exc_codec_encode_char : memref<33xi8> = dense<[39, 32, 99, 111, 100, 101, 99, 32, 99, 97, 110, 39, 116, 32, 101, 110, 99, 111, 100, 101, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 32, 39, 92]>
  // "' codec can't encode characters in position "
  memref.global "private" constant @__ly_exc_codec_encode_chars : memref<44xi8> = dense<[39, 32, 99, 111, 100, 101, 99, 32, 99, 97, 110, 39, 116, 32, 101, 110, 99, 111, 100, 101, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 115, 32, 105, 110, 32, 112, 111, 115, 105, 116, 105, 111, 110, 32]>
  // "can't translate character '\"
  memref.global "private" constant @__ly_exc_codec_translate_char : memref<28xi8> = dense<[99, 97, 110, 39, 116, 32, 116, 114, 97, 110, 115, 108, 97, 116, 101, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 32, 39, 92]>
  // "can't translate characters in position "
  memref.global "private" constant @__ly_exc_codec_translate_chars : memref<39xi8> = dense<[99, 97, 110, 39, 116, 32, 116, 114, 97, 110, 115, 108, 97, 116, 101, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 115, 32, 105, 110, 32, 112, 111, 115, 105, 116, 105, 111, 110, 32]>
  // " in position "
  memref.global "private" constant @__ly_exc_codec_in_position : memref<13xi8> = dense<[32, 105, 110, 32, 112, 111, 115, 105, 116, 105, 111, 110, 32]>
  // "' in position "
  memref.global "private" constant @__ly_exc_codec_char_in_position : memref<14xi8> = dense<[39, 32, 105, 110, 32, 112, 111, 115, 105, 116, 105, 111, 110, 32]>
  // ": "
  memref.global "private" constant @__ly_exc_codec_colon : memref<2xi8> = dense<[58, 32]>
  // "-"
  memref.global "private" constant @__ly_exc_codec_dash : memref<1xi8> = dense<[45]>
  // "x"
  memref.global "private" constant @__ly_exc_codec_x : memref<1xi8> = dense<[120]>
  // "u"
  memref.global "private" constant @__ly_exc_codec_u : memref<1xi8> = dense<[117]>
  // "U"
  memref.global "private" constant @__ly_exc_codec_U : memref<1xi8> = dense<[85]>

  // acc + piece, as a new str; both inputs are released -- the one step a
  // codec error's message is built by.
  func.func private @__ly_exc_append(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %piece_h: memref<2xi64> {ly.ownership.object_header}, %piece_b: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0, 2]} {
    %h, %b = func.call @LyUnicode_Concat(%acc_h, %acc_b, %piece_h, %piece_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%acc_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%piece_h) : (memref<2xi64>) -> ()
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + a str the caller only borrows (an argument the payload block still
  // holds); acc is released.
  func.func private @__ly_exc_append_borrowed(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %piece_h: memref<2xi64> {ly.ownership.object_header}, %piece_b: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %h, %b = func.call @LyUnicode_Concat(%acc_h, %acc_b, %piece_h, %piece_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%acc_h) : (memref<2xi64>) -> ()
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + `length` bytes of ASCII text; acc is released.
  func.func private @__ly_exc_append_text(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %text: memref<?xi8>, %length: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %c0 = arith.constant 0 : index
    %piece_h, %piece_b = func.call @__ly_unicode_from_valid_utf8(%text, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %h, %b = func.call @__ly_exc_append(%acc_h, %acc_b, %piece_h, %piece_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + the decimal digits of `value` (PyUnicode_FromFormat's %zd); acc is
  // released.
  func.func private @__ly_exc_append_int(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %piece_h, %piece_b = func.call @LyUnicode_FromI64(%value) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %h, %b = func.call @__ly_exc_append(%acc_h, %acc_b, %piece_h, %piece_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + `digits` lowercase hex digits of `value` (%02x, %04x, %08x); acc is
  // released.
  func.func private @__ly_exc_append_hex(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %value: i64, %digits: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %piece_h, %piece_b = func.call @__ly_unicode_from_hex(%value, %digits) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %h, %b = func.call @__ly_exc_append(%acc_h, %acc_b, %piece_h, %piece_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + a character the way the encode and translate messages escape it:
  // \xhh up to 0xff, \uhhhh up to 0xffff, \Uhhhhhhhh beyond (the backslash
  // is already in the text before it); acc is released.
  func.func private @__ly_exc_append_escape(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %ch: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %eight = arith.constant 8 : i64
    %byte_max = arith.constant 255 : i64
    %bmp_max = arith.constant 65535 : i64
    %is_byte = arith.cmpi ule, %ch, %byte_max : i64
    %is_bmp = arith.cmpi ule, %ch, %bmp_max : i64
    %x_ref = memref.get_global @__ly_exc_codec_x : memref<1xi8>
    %x = memref.cast %x_ref : memref<1xi8> to memref<?xi8>
    %u_ref = memref.get_global @__ly_exc_codec_u : memref<1xi8>
    %u = memref.cast %u_ref : memref<1xi8> to memref<?xi8>
    %big_u_ref = memref.get_global @__ly_exc_codec_U : memref<1xi8>
    %big_u = memref.cast %big_u_ref : memref<1xi8> to memref<?xi8>
    %wide_letter = arith.select %is_bmp, %u, %big_u : memref<?xi8>
    %letter = arith.select %is_byte, %x, %wide_letter : memref<?xi8>
    %wide_digits = arith.select %is_bmp, %four, %eight : i64
    %digits = arith.select %is_byte, %two, %wide_digits : i64
    %lettered_h, %lettered_b = func.call @__ly_exc_append_text(%acc_h, %acc_b, %letter, %one) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    %h, %b = func.call @__ly_exc_append_hex(%lettered_h, %lettered_b, %ch, %digits) : (memref<2xi64>, memref<?xi8>, i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + "<start>: <reason>"; acc is released.
  func.func private @__ly_exc_append_position(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %start: i64, %reason_h: memref<2xi64> {ly.ownership.object_header}, %reason_b: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %two = arith.constant 2 : i64
    %colon_ref = memref.get_global @__ly_exc_codec_colon : memref<2xi8>
    %colon = memref.cast %colon_ref : memref<2xi8> to memref<?xi8>
    %a_h, %a_b = func.call @__ly_exc_append_int(%acc_h, %acc_b, %start) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    %b_h, %b_b = func.call @__ly_exc_append_text(%a_h, %a_b, %colon, %two) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    %h, %b = func.call @__ly_exc_append_borrowed(%b_h, %b_b, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + "<start>-<end - 1>: <reason>"; acc is released.
  func.func private @__ly_exc_append_span(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %start: i64, %end: i64, %reason_h: memref<2xi64> {ly.ownership.object_header}, %reason_b: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %one = arith.constant 1 : i64
    %dash_ref = memref.get_global @__ly_exc_codec_dash : memref<1xi8>
    %dash = memref.cast %dash_ref : memref<1xi8> to memref<?xi8>
    %a_h, %a_b = func.call @__ly_exc_append_int(%acc_h, %acc_b, %start) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    %b_h, %b_b = func.call @__ly_exc_append_text(%a_h, %a_b, %dash, %one) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    %last = arith.subi %end, %one : i64
    %h, %b = func.call @__ly_exc_append_position(%b_h, %b_b, %last, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // The str an argument slot holds, as views the payload block keeps alive.
  func.func private @__ly_exc_slot_str(%word: i64) -> (memref<2xi64>, memref<?xi8>) {
    %two = arith.constant 2 : i64
    %header_dyn = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
    %header = memref.cast %header_dyn : memref<?xi64> to memref<2xi64>
    %bytes_ptr, %byte_len = func.call @__ly_unicode_lane_words(%word) : (i64) -> (i64, i64)
    %bytes = func.call @__ly_global_view_i8(%bytes_ptr, %byte_len) : (i64, i64) -> memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Whether argument `slot` is an object of class `class_id`.
  func.func private @__ly_exc_slot_is(%block: i64, %slot: i64, %class_id: i64) -> i1 {
    %zero = arith.constant 0 : i64
    %word = func.call @__ly_exc_payload_box_word(%block, %slot, %zero) : (i64, i64, i64) -> i64
    %class = func.call @__ly_slot_class(%word) : (i64) -> i64
    %is = arith.cmpi eq, %class, %class_id : i64
    func.return %is : i1
  }

  // Whether argument `slot` is an int that is a word (the Py_ssize_t the
  // "n" of the codec errors' __init__ format takes).
  func.func private @__ly_exc_slot_is_index(%block: i64, %slot: i64) -> i1 {
    %zero = arith.constant 0 : i64
    %int_class = arith.constant 1 : i64
    %false = arith.constant false
    %is_int = func.call @__ly_exc_slot_is(%block, %slot, %int_class) : (i64, i64, i64) -> i1
    %fits = scf.if %is_int -> (i1) {
      %word = func.call @__ly_exc_payload_box_word(%block, %slot, %zero) : (i64, i64, i64) -> i64
      %value, %ok = func.call @LyLong_SlotWordAsI64(%word) : (i64) -> (i64, i1)
      scf.yield %ok : i1
    } else {
      scf.yield %false : i1
    }
    func.return %fits : i1
  }

  // Which codec error's message the arguments can spell: 1 for
  // UnicodeDecodeError's, 2 for UnicodeEncodeError's, 3 for
  // UnicodeTranslateError's, 0 for none -- the class is none of the three
  // (their subclasses count, as CPython's __str__ is inherited) or the
  // arguments are not what their __init__ takes: (str, bytes, int, int,
  // str), (str, str, int, int, str), (str, int, int, str). Lython also builds
  // these with a single message, which renders as BaseException does.
  func.func private @__ly_exc_codec_error_kind(%header: memref<3xi64>, %block: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %str_class = arith.constant 4 : i64
    %bytes_class = arith.constant 70 : i64
    %decode_root = arith.constant 122 : i64
    %encode_root = arith.constant 123 : i64
    %translate_root = arith.constant 124 : i64
    %false = arith.constant false
    %class_slot = arith.constant 2 : index
    %class = memref.load %header[%class_slot] : memref<3xi64>
    %is_decode = func.call @LyEH_ClassIdMatches(%class, %decode_root) : (i64, i64) -> i1
    %is_encode = func.call @LyEH_ClassIdMatches(%class, %encode_root) : (i64, i64) -> i1
    %is_translate = func.call @LyEH_ClassIdMatches(%class, %translate_root) : (i64, i64) -> i1
    %count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %five_args = arith.cmpi eq, %count, %five : i64
    %four_args = arith.cmpi eq, %count, %four : i64
    %codec = arith.ori %is_decode, %is_encode : i1
    %codec_shaped = arith.andi %codec, %five_args : i1
    %translate_shaped = arith.andi %is_translate, %four_args : i1
    %kind = scf.if %codec_shaped -> (i64) {
      %encoding_ok = func.call @__ly_exc_slot_is(%block, %zero, %str_class) : (i64, i64, i64) -> i1
      %object_class = arith.select %is_decode, %bytes_class, %str_class : i64
      %object_ok = func.call @__ly_exc_slot_is(%block, %one, %object_class) : (i64, i64, i64) -> i1
      %start_ok = func.call @__ly_exc_slot_is_index(%block, %two) : (i64, i64) -> i1
      %end_ok = func.call @__ly_exc_slot_is_index(%block, %three) : (i64, i64) -> i1
      %reason_ok = func.call @__ly_exc_slot_is(%block, %four, %str_class) : (i64, i64, i64) -> i1
      %a = arith.andi %encoding_ok, %object_ok : i1
      %b = arith.andi %a, %start_ok : i1
      %c = arith.andi %b, %end_ok : i1
      %all = arith.andi %c, %reason_ok : i1
      %which = arith.select %is_decode, %one, %two : i64
      %answer = arith.select %all, %which, %zero : i64
      scf.yield %answer : i64
    } else {
      %t = scf.if %translate_shaped -> (i64) {
        %object_ok = func.call @__ly_exc_slot_is(%block, %zero, %str_class) : (i64, i64, i64) -> i1
        %start_ok = func.call @__ly_exc_slot_is_index(%block, %one) : (i64, i64) -> i1
        %end_ok = func.call @__ly_exc_slot_is_index(%block, %two) : (i64, i64) -> i1
        %reason_ok = func.call @__ly_exc_slot_is(%block, %three, %str_class) : (i64, i64, i64) -> i1
        %a = arith.andi %object_ok, %start_ok : i1
        %b = arith.andi %a, %end_ok : i1
        %all = arith.andi %b, %reason_ok : i1
        %answer = arith.select %all, %three, %zero : i64
        scf.yield %answer : i64
      } else {
        scf.yield %zero : i64
      }
      scf.yield %t : i64
    }
    func.return %kind : i64
  }

  // CPython's UnicodeDecodeError_str / UnicodeEncodeError_str /
  // UnicodeTranslateError_str for arguments __ly_exc_codec_error_kind
  // accepted: one byte or character in range is named, anything else is a
  // span "start-(end - 1)".
  func.func private @__ly_exc_render_codec_error(%block: i64, %kind: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %is_decode = arith.cmpi eq, %kind, %one : i64
    %is_translate = arith.cmpi eq, %kind, %three : i64
    // The translate error has no encoding: its arguments start a slot earlier.
    %object_slot = arith.select %is_translate, %zero, %one : i64
    %start_slot = arith.addi %object_slot, %one : i64
    %end_slot = arith.addi %object_slot, %two : i64
    %reason_slot = arith.addi %object_slot, %three : i64
    %start_word = func.call @__ly_exc_payload_box_word(%block, %start_slot, %zero) : (i64, i64, i64) -> i64
    %start, %start_fits = func.call @LyLong_SlotWordAsI64(%start_word) : (i64) -> (i64, i1)
    %end_word = func.call @__ly_exc_payload_box_word(%block, %end_slot, %zero) : (i64, i64, i64) -> i64
    %end, %end_fits = func.call @LyLong_SlotWordAsI64(%end_word) : (i64) -> (i64, i1)
    %reason_word = func.call @__ly_exc_payload_box_word(%block, %reason_slot, %zero) : (i64, i64, i64) -> i64
    %reason_h, %reason_b = func.call @__ly_exc_slot_str(%reason_word) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %object_word = func.call @__ly_exc_payload_box_word(%block, %object_slot, %zero) : (i64, i64, i64) -> i64
    %start_low = arith.cmpi sge, %start, %zero : i64
    // The object's length, and the byte or character at `start` when there is
    // one to read.
    %length, %bad = scf.if %is_decode -> (i64, i64) {
      %bytes_header_dyn = func.call @__ly_global_view_i64(%object_word, %four) : (i64, i64) -> memref<?xi64>
      %bytes_header = memref.cast %bytes_header_dyn : memref<?xi64> to memref<4xi64>
      %payload = func.call @__ly_bytes_payload(%bytes_header) : (memref<4xi64>) -> memref<?xi8>
      %dim = memref.dim %payload, %c0 : memref<?xi8>
      %len = arith.index_cast %dim : index to i64
      %start_high = arith.cmpi slt, %start, %len : i64
      %in = arith.andi %start_low, %start_high : i1
      %byte = scf.if %in -> (i64) {
        %at = arith.index_cast %start : i64 to index
        %raw = memref.load %payload[%at] : memref<?xi8>
        %wide = arith.extui %raw : i8 to i64
        scf.yield %wide : i64
      } else {
        scf.yield %zero : i64
      }
      scf.yield %len, %byte : i64, i64
    } else {
      %object_h, %object_b = func.call @__ly_exc_slot_str(%object_word) : (i64) -> (memref<2xi64>, memref<?xi8>)
      %len = func.call @__ly_unicode_count(%object_h, %object_b) : (memref<2xi64>, memref<?xi8>) -> i64
      %width = func.call @__ly_unicode_width(%object_h) : (memref<2xi64>) -> i64
      %start_high = arith.cmpi slt, %start, %len : i64
      %in = arith.andi %start_low, %start_high : i1
      %ch = scf.if %in -> (i64) {
        %at = arith.index_cast %start : i64 to index
        %code = func.call @__ly_unicode_get(%object_b, %width, %at) : (memref<?xi8>, i64, index) -> i64
        scf.yield %code : i64
      } else {
        scf.yield %zero : i64
      }
      scf.yield %len, %ch : i64, i64
    }
    %start_high = arith.cmpi slt, %start, %length : i64
    %end_low = arith.cmpi sge, %end, %zero : i64
    %end_high = arith.cmpi sle, %end, %length : i64
    %next = arith.addi %start, %one : i64
    %one_wide = arith.cmpi eq, %end, %next : i64
    %s1 = arith.andi %start_low, %start_high : i1
    %s2 = arith.andi %s1, %end_low : i1
    %s3 = arith.andi %s2, %end_high : i1
    %single = arith.andi %s3, %one_wide : i1
    %message:2 = scf.if %is_translate -> (memref<2xi64>, memref<?xi8>) {
      %t:2 = scf.if %single -> (memref<2xi64>, memref<?xi8>) {
        %head_ref = memref.get_global @__ly_exc_codec_translate_char : memref<28xi8>
        %head = memref.cast %head_ref : memref<28xi8> to memref<?xi8>
        %head_len = arith.constant 28 : i64
        %a_h, %a_b = func.call @__ly_unicode_from_valid_utf8(%head, %c0, %head_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %b_h, %b_b = func.call @__ly_exc_append_escape(%a_h, %a_b, %bad) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
        %pos_ref = memref.get_global @__ly_exc_codec_char_in_position : memref<14xi8>
        %pos = memref.cast %pos_ref : memref<14xi8> to memref<?xi8>
        %pos_len = arith.constant 14 : i64
        %c_h, %c_b = func.call @__ly_exc_append_text(%b_h, %b_b, %pos, %pos_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
        %d_h, %d_b = func.call @__ly_exc_append_position(%c_h, %c_b, %start, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %d_h, %d_b : memref<2xi64>, memref<?xi8>
      } else {
        %head_ref = memref.get_global @__ly_exc_codec_translate_chars : memref<39xi8>
        %head = memref.cast %head_ref : memref<39xi8> to memref<?xi8>
        %head_len = arith.constant 39 : i64
        %a_h, %a_b = func.call @__ly_unicode_from_valid_utf8(%head, %c0, %head_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %d_h, %d_b = func.call @__ly_exc_append_span(%a_h, %a_b, %start, %end, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, i64, i64, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %d_h, %d_b : memref<2xi64>, memref<?xi8>
      }
      scf.yield %t#0, %t#1 : memref<2xi64>, memref<?xi8>
    } else {
      // "'" + encoding, then the rest by kind.
      %quote_ref = memref.get_global @__ly_exc_codec_quote : memref<1xi8>
      %quote = memref.cast %quote_ref : memref<1xi8> to memref<?xi8>
      %q_h, %q_b = func.call @__ly_unicode_from_valid_utf8(%quote, %c0, %one) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %encoding_word = func.call @__ly_exc_payload_box_word(%block, %zero, %zero) : (i64, i64, i64) -> i64
      %encoding_h, %encoding_b = func.call @__ly_exc_slot_str(%encoding_word) : (i64) -> (memref<2xi64>, memref<?xi8>)
      %e_h, %e_b = func.call @__ly_exc_append_borrowed(%q_h, %q_b, %encoding_h, %encoding_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      %r:2 = scf.if %is_decode -> (memref<2xi64>, memref<?xi8>) {
        %d:2 = scf.if %single -> (memref<2xi64>, memref<?xi8>) {
          %text_ref = memref.get_global @__ly_exc_codec_decode_byte : memref<28xi8>
          %text = memref.cast %text_ref : memref<28xi8> to memref<?xi8>
          %text_len = arith.constant 28 : i64
          %a_h, %a_b = func.call @__ly_exc_append_text(%e_h, %e_b, %text, %text_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
          %b_h, %b_b = func.call @__ly_exc_append_hex(%a_h, %a_b, %bad, %two) : (memref<2xi64>, memref<?xi8>, i64, i64) -> (memref<2xi64>, memref<?xi8>)
          %pos_ref = memref.get_global @__ly_exc_codec_in_position : memref<13xi8>
          %pos = memref.cast %pos_ref : memref<13xi8> to memref<?xi8>
          %pos_len = arith.constant 13 : i64
          %c_h, %c_b = func.call @__ly_exc_append_text(%b_h, %b_b, %pos, %pos_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
          %f_h, %f_b = func.call @__ly_exc_append_position(%c_h, %c_b, %start, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          scf.yield %f_h, %f_b : memref<2xi64>, memref<?xi8>
        } else {
          %text_ref = memref.get_global @__ly_exc_codec_decode_bytes : memref<39xi8>
          %text = memref.cast %text_ref : memref<39xi8> to memref<?xi8>
          %text_len = arith.constant 39 : i64
          %a_h, %a_b = func.call @__ly_exc_append_text(%e_h, %e_b, %text, %text_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
          %f_h, %f_b = func.call @__ly_exc_append_span(%a_h, %a_b, %start, %end, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, i64, i64, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          scf.yield %f_h, %f_b : memref<2xi64>, memref<?xi8>
        }
        scf.yield %d#0, %d#1 : memref<2xi64>, memref<?xi8>
      } else {
        %d:2 = scf.if %single -> (memref<2xi64>, memref<?xi8>) {
          %text_ref = memref.get_global @__ly_exc_codec_encode_char : memref<33xi8>
          %text = memref.cast %text_ref : memref<33xi8> to memref<?xi8>
          %text_len = arith.constant 33 : i64
          %a_h, %a_b = func.call @__ly_exc_append_text(%e_h, %e_b, %text, %text_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
          %b_h, %b_b = func.call @__ly_exc_append_escape(%a_h, %a_b, %bad) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
          %pos_ref = memref.get_global @__ly_exc_codec_char_in_position : memref<14xi8>
          %pos = memref.cast %pos_ref : memref<14xi8> to memref<?xi8>
          %pos_len = arith.constant 14 : i64
          %c_h, %c_b = func.call @__ly_exc_append_text(%b_h, %b_b, %pos, %pos_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
          %f_h, %f_b = func.call @__ly_exc_append_position(%c_h, %c_b, %start, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, i64, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          scf.yield %f_h, %f_b : memref<2xi64>, memref<?xi8>
        } else {
          %text_ref = memref.get_global @__ly_exc_codec_encode_chars : memref<44xi8>
          %text = memref.cast %text_ref : memref<44xi8> to memref<?xi8>
          %text_len = arith.constant 44 : i64
          %a_h, %a_b = func.call @__ly_exc_append_text(%e_h, %e_b, %text, %text_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
          %f_h, %f_b = func.call @__ly_exc_append_span(%a_h, %a_b, %start, %end, %reason_h, %reason_b) : (memref<2xi64>, memref<?xi8>, i64, i64, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          scf.yield %f_h, %f_b : memref<2xi64>, memref<?xi8>
        }
        scf.yield %d#0, %d#1 : memref<2xi64>, memref<?xi8>
      }
      scf.yield %r#0, %r#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %message#0, %message#1 : memref<2xi64>, memref<?xi8>
  }

  // The codec errors' attributes are their payload slots: the arguments
  // __init__ took, in its order (exceptions.c keeps them in fields). Each
  // reader hands back a reference of its own.
  func.func private @__ly_exc_codec_slot_word(%header: memref<3xi64>, %slot: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %payload_slot = arith.constant 3 : i64
    %block = func.call @__ly_exc_ext_get(%header, %payload_slot) : (memref<3xi64>, i64) -> i64
    %word = func.call @__ly_exc_payload_box_word(%block, %slot, %zero) : (i64, i64, i64) -> i64
    func.return %word : i64
  }

  func.func private @__ly_exc_codec_str(%header: memref<3xi64>, %slot: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %word = func.call @__ly_exc_codec_slot_word(%header, %slot) : (memref<3xi64>, i64) -> i64
    func.call @__ly_handle_retain_raw(%word) : (i64) -> ()
    %h, %b = func.call @__ly_exc_slot_str(%word) : (i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_exc_codec_bytes(%header: memref<3xi64>, %slot: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0]} {
    %four = arith.constant 4 : i64
    %word = func.call @__ly_exc_codec_slot_word(%header, %slot) : (memref<3xi64>, i64) -> i64
    func.call @__ly_handle_retain_raw(%word) : (i64) -> ()
    %view = func.call @__ly_global_view_i64(%word, %four) : (i64, i64) -> memref<?xi64>
    %object = memref.cast %view : memref<?xi64> to memref<4xi64>
    func.return %object : memref<4xi64>
  }

  func.func private @__ly_exc_codec_int(%header: memref<3xi64>, %slot: i64) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %word = func.call @__ly_exc_codec_slot_word(%header, %slot) : (memref<3xi64>, i64) -> i64
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %value = scf.if %immediate -> (memref<2xi64>) {
      %v = func.call @__ly_int_from_immediate(%word) : (i64) -> i64
      %fresh = func.call @LyLong_FromI64(%v) : (i64) -> memref<2xi64>
      scf.yield %fresh : memref<2xi64>
    } else {
      %two = arith.constant 2 : i64
      func.call @__ly_handle_retain_raw(%word) : (i64) -> ()
      %view = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
      %held = memref.cast %view : memref<?xi64> to memref<2xi64>
      scf.yield %held : memref<2xi64>
    }
    func.return %value : memref<2xi64>
  }

  func.func @LyUnicodeDecodeError_Encoding(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "encoding", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 0 : i64
    %h, %b = func.call @__ly_exc_codec_str(%header, %slot) : (memref<3xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeDecodeError_Object(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "object", ly.runtime.result_contract = "builtins.bytes"} {
    %slot = arith.constant 1 : i64
    %object = func.call @__ly_exc_codec_bytes(%header, %slot) : (memref<3xi64>, i64) -> memref<4xi64>
    func.return %object : memref<4xi64>
  }
  func.func @LyUnicodeDecodeError_Start(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "start", ly.runtime.result_contract = "builtins.int"} {
    %slot = arith.constant 2 : i64
    %value = func.call @__ly_exc_codec_int(%header, %slot) : (memref<3xi64>, i64) -> memref<2xi64>
    func.return %value : memref<2xi64>
  }
  func.func @LyUnicodeDecodeError_End(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "end", ly.runtime.result_contract = "builtins.int"} {
    %slot = arith.constant 3 : i64
    %value = func.call @__ly_exc_codec_int(%header, %slot) : (memref<3xi64>, i64) -> memref<2xi64>
    func.return %value : memref<2xi64>
  }
  func.func @LyUnicodeDecodeError_Reason(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "reason", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 4 : i64
    %h, %b = func.call @__ly_exc_codec_str(%header, %slot) : (memref<3xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeEncodeError_Encoding(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "encoding", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 0 : i64
    %h, %b = func.call @__ly_exc_codec_str(%header, %slot) : (memref<3xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeEncodeError_Object(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "object", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 1 : i64
    %h, %b = func.call @__ly_exc_codec_str(%header, %slot) : (memref<3xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeEncodeError_Start(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "start", ly.runtime.result_contract = "builtins.int"} {
    %slot = arith.constant 2 : i64
    %value = func.call @__ly_exc_codec_int(%header, %slot) : (memref<3xi64>, i64) -> memref<2xi64>
    func.return %value : memref<2xi64>
  }
  func.func @LyUnicodeEncodeError_End(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "end", ly.runtime.result_contract = "builtins.int"} {
    %slot = arith.constant 3 : i64
    %value = func.call @__ly_exc_codec_int(%header, %slot) : (memref<3xi64>, i64) -> memref<2xi64>
    func.return %value : memref<2xi64>
  }
  func.func @LyUnicodeEncodeError_Reason(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "reason", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 4 : i64
    %h, %b = func.call @__ly_exc_codec_str(%header, %slot) : (memref<3xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeTranslateError_Object(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.primitive = "object", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 0 : i64
    %h, %b = func.call @__ly_exc_codec_str(%header, %slot) : (memref<3xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeTranslateError_Start(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.primitive = "start", ly.runtime.result_contract = "builtins.int"} {
    %slot = arith.constant 1 : i64
    %value = func.call @__ly_exc_codec_int(%header, %slot) : (memref<3xi64>, i64) -> memref<2xi64>
    func.return %value : memref<2xi64>
  }
  func.func @LyUnicodeTranslateError_End(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.primitive = "end", ly.runtime.result_contract = "builtins.int"} {
    %slot = arith.constant 2 : i64
    %value = func.call @__ly_exc_codec_int(%header, %slot) : (memref<3xi64>, i64) -> memref<2xi64>
    func.return %value : memref<2xi64>
  }
  func.func @LyUnicodeTranslateError_Reason(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.primitive = "reason", ly.runtime.result_contract = "builtins.str"} {
    %slot = arith.constant 3 : i64
    %h, %b = func.call @__ly_exc_codec_str(%header, %slot) : (memref<3xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // The payload message of a codec error: what went wrong where, as CPython's
  // UnicodeDecodeError_str and its siblings say it, when the arguments are
  // the codec's; BaseException's rendering otherwise.
  // ⛔ Not a branch in __ly_exc_render_args, which every exception built
  // with arguments shares: a KeyError(key) then carried the codec messages'
  // whole construction (fib.wasm +29%). The lowering picks this init by the
  // exception's class, so only a program that builds a codec error has it.
  //
  // The message's reference moves into the header's lane (set_message stores
  // it and takes nothing), which the release insertion cannot follow -- like
  // LyBaseException_InitPayloadMessage this is a manifest function, whose
  // ownership is the one written here.
  func.func private @__ly_exc_init_codec_payload_message(%header: memref<3xi64> {ly.ownership.object_header}, %old_mh: memref<2xi64> {ly.ownership.object_header}, %old_mb: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.UnicodeError"} {
    %zero = arith.constant 0 : i64
    %payload_slot = arith.constant 3 : i64
    %block = func.call @__ly_exc_ext_get(%header, %payload_slot) : (memref<3xi64>, i64) -> i64
    %kind = func.call @__ly_exc_codec_error_kind(%header, %block) : (memref<3xi64>, i64) -> i64
    %is_codec = arith.cmpi ne, %kind, %zero : i64
    cf.cond_br %is_codec, ^codec, ^plain

  ^codec:
    %codec_h, %codec_b = func.call @__ly_exc_render_codec_error(%block, %kind) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_exc_set_message(%header, %codec_h) : (memref<3xi64>, memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%old_mh) : (memref<2xi64>) -> ()
    func.return %header, %codec_h, %codec_b : memref<3xi64>, memref<2xi64>, memref<?xi8>

  ^plain:
    %plain_h, %plain_b = func.call @__ly_exc_render_args(%header) : (memref<3xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_exc_set_message(%header, %plain_h) : (memref<3xi64>, memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%old_mh) : (memref<2xi64>) -> ()
    func.return %header, %plain_h, %plain_b : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeDecodeError_InitPayloadMessage(%header: memref<3xi64> {ly.ownership.object_header}, %old_mh: memref<2xi64> {ly.ownership.object_header}, %old_mb: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.UnicodeDecodeError"], ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "init_payload_message"} {
    %e:3 = func.call @__ly_exc_init_codec_payload_message(%header, %old_mh, %old_mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %e#0, %e#1, %e#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeEncodeError_InitPayloadMessage(%header: memref<3xi64> {ly.ownership.object_header}, %old_mh: memref<2xi64> {ly.ownership.object_header}, %old_mb: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.UnicodeEncodeError"], ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "init_payload_message"} {
    %e:3 = func.call @__ly_exc_init_codec_payload_message(%header, %old_mh, %old_mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %e#0, %e#1, %e#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeTranslateError_InitPayloadMessage(%header: memref<3xi64> {ly.ownership.object_header}, %old_mh: memref<2xi64> {ly.ownership.object_header}, %old_mb: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.UnicodeTranslateError"], ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.primitive = "init_payload_message"} {
    %e:3 = func.call @__ly_exc_init_codec_payload_message(%header, %old_mh, %old_mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %e#0, %e#1, %e#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // CPython PyUnicodeDecodeError_Create: a UnicodeDecodeError built with its
  // five arguments, and its message rendered from them. The encoding, the
  // object and the reason are the block's from here on.
  func.func @LyUnicodeDecodeError_Create(%encoding_h: memref<2xi64> {ly.ownership.object_header}, %encoding_b: memref<?xi8>, %object: memref<4xi64> {ly.ownership.object_header}, %start: i64, %end: i64, %reason_h: memref<2xi64> {ly.ownership.object_header}, %reason_b: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.UnicodeDecodeError"], ly.ownership.owned_results = [0], ly.ownership.transfer_args = [0, 2, 5]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %five = arith.constant 5 : i64
    %class_id = arith.constant 122 : i64
    %exception:3 = func.call @LyUnicodeDecodeError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %block = func.call @LyBaseExceptionGroup_MembersAlloc(%exception#0, %exception#1, %exception#2, %five) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64) -> i64
    func.call @__ly_exc_payload_store_unicode(%block, %zero, %encoding_h, %encoding_b) : (i64, i64, memref<2xi64>, memref<?xi8>) -> ()
    %object_index = memref.extract_aligned_pointer_as_index %object : memref<4xi64> -> index
    %object_word = arith.index_cast %object_index : index to i64
    func.call @__ly_exc_payload_store_words(%block, %one, %object_word) : (i64, i64, i64) -> ()
    %start_word = func.call @LyLong_SlotWordFromI64(%start) : (i64) -> i64
    func.call @__ly_exc_payload_store_words(%block, %two, %start_word) : (i64, i64, i64) -> ()
    %end_word = func.call @LyLong_SlotWordFromI64(%end) : (i64) -> i64
    func.call @__ly_exc_payload_store_words(%block, %three, %end_word) : (i64, i64, i64) -> ()
    func.call @__ly_exc_payload_store_unicode(%block, %four, %reason_h, %reason_b) : (i64, i64, memref<2xi64>, memref<?xi8>) -> ()
    %initialized:3 = func.call @LyUnicodeDecodeError_InitPayloadMessage(%exception#0, %exception#1, %exception#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %initialized#0, %initialized#1, %initialized#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // Multi-value args message: CPython's str(e) for len(args) > 1 is
  // repr(args) -- "(r0, r1, ...)". Renders from the payload boxes, replaces
  // the empty construction-time message, and returns the receiver triple.
  func.func @LyBaseException_InitPayloadMessage(%header: memref<3xi64> {ly.ownership.object_header}, %old_mh: memref<2xi64> {ly.ownership.object_header}, %old_mb: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.BaseException"], ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "init_payload_message"} {
    %msg_h, %msg_b = func.call @__ly_exc_render_args(%header) : (memref<3xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_exc_set_message(%header, %msg_h) : (memref<3xi64>, memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%old_mh) : (memref<2xi64>) -> ()
    func.return %header, %msg_h, %msg_b : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // str(e) off the payload block. ONE argument renders as str(arg) and only
  // zero or two-and-up render the "(a, b)" tuple: that is what CPython's
  // BaseException.__str__ does, and it is why `str(ValueError(42))` is "42"
  // rather than "(42,)".
  func.func private @__ly_exc_render_args(%header: memref<3xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "render_args", ly.runtime.result_contract = "builtins.str"} {
    %sc0 = arith.constant 0 : index
    %sc1_i64 = arith.constant 1 : i64
    %spayload_slot = arith.constant 3 : i64
    %sblock = func.call @__ly_exc_ext_get(%header, %spayload_slot) : (memref<3xi64>, i64) -> i64
    %scount = func.call @__ly_exc_payload_count(%sblock) : (i64) -> i64
    %stuple = func.call @__ly_exc_payload_tuple_flag(%sblock) : (i64) -> i1
    %strue = arith.constant true
    %snot_tuple = arith.xori %stuple, %strue : i1
    %sone = arith.cmpi eq, %scount, %sc1_i64 : i64
    %ssingle = arith.andi %sone, %snot_tuple : i1
    %sout:2 = scf.if %ssingle -> (memref<2xi64>, memref<?xi8>) {
      %sblock_ptr = llvm.inttoptr %sblock : i64 to !llvm.ptr
      %sbox_ptr = llvm.getelementptr %sblock_ptr[%sc1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %sclass_word = llvm.load %sbox_ptr : !llvm.ptr -> i64
      %sclass_id = func.call @__ly_slot_class(%sclass_word) : (i64) -> i64
      // ⛔ KeyError.__str__ IS repr(args[0]) IN CPYTHON, and it is inherited, so
      // the taxonomy walk decides rather than an equality test: routing a
      // non-str argument through the generic payload path would otherwise lose
      // the override that the str path keeps in LyKeyError_Init, and
      // `str(KeyError(p))` printed p's __str__ where CPython prints its __repr__.
      %sexc_class_slot = arith.constant 2 : index
      %sexc_class = memref.load %header[%sexc_class_slot] : memref<3xi64>
      %skey_root = arith.constant 54 : i64
      %sis_key = func.call @LyEH_ClassIdMatches(%sexc_class, %skey_root) : (i64, i64) -> i1
      %spicked:2 = scf.if %sis_key -> (memref<2xi64>, memref<?xi8>) {
        %krh, %krb = func.call @__ly_repr_boxed_or_default(%sbox_ptr, %sclass_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %krh, %krb : memref<2xi64>, memref<?xi8>
      } else {
        %sh, %sb = func.call @__ly_str_boxed_or_default(%sbox_ptr, %sclass_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %sh, %sb : memref<2xi64>, memref<?xi8>
      }
      scf.yield %spicked#0, %spicked#1 : memref<2xi64>, memref<?xi8>
    } else {
      %th, %tb = func.call @__ly_exc_render_args_tuple(%header) : (memref<3xi64>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %th, %tb : memref<2xi64>, memref<?xi8>
    }
    func.return %sout#0, %sout#1 : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_exc_render_args_tuple(%header: memref<3xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "render_args_tuple", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %payload_slot = arith.constant 3 : i64
    %block = func.call @__ly_exc_ext_get(%header, %payload_slot) : (memref<3xi64>, i64) -> i64
    %count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %block_ptr = llvm.inttoptr %block : i64 to !llvm.ptr
    %open_ref = memref.get_global @__ly_repr_lparen : memref<1xi8>
    %open_dyn = memref.cast %open_ref : memref<1xi8> to memref<?xi8>
    %r0_h, %r0_b = func.call @__ly_unicode_from_valid_utf8(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %comma_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
    %comma_dyn = memref.cast %comma_ref : memref<2xi8> to memref<?xi8>
    %count_index = arith.index_cast %count : i64 to index
    %loop:2 = scf.for %i = %c0 to %count_index step %c1 iter_args(%rh = %r0_h, %rb = %r0_b) -> (memref<2xi64>, memref<?xi8>) {
      %i_i64 = arith.index_cast %i : index to i64
      %is_pos = arith.cmpi sgt, %i_i64, %c0_i64 : i64
      %sep:2 = scf.if %is_pos -> (memref<2xi64>, memref<?xi8>) {
        %sh, %sb = func.call @__ly_unicode_from_valid_utf8(%comma_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %jh, %jb = func.call @LyUnicode_Concat(%rh, %rb, %sh, %sb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%rh) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%sh) : (memref<2xi64>) -> ()
        scf.yield %jh, %jb : memref<2xi64>, memref<?xi8>
      } else {
        scf.yield %rh, %rb : memref<2xi64>, memref<?xi8>
      }
      %box_words = arith.muli %i_i64, %c2_i64 : i64
      %sixteen = func.call @__ly_box_word_count() : () -> i64
      %box_off = arith.muli %i_i64, %sixteen : i64
      %box_base = arith.addi %box_off, %c1_i64 : i64
      %box_ptr = llvm.getelementptr %block_ptr[%box_base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %class_word = llvm.load %box_ptr : !llvm.ptr -> i64
      %class_id = func.call @__ly_slot_class(%class_word) : (i64) -> i64
      %erh, %erb, %ok = func.call @__ly_repr_boxed_by_contract(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
      cf.assert %ok, "exception args: boxed value has no conforming __repr__"
      %nh, %nb = func.call @LyUnicode_Concat(%sep#0, %sep#1, %erh, %erb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%sep#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%erh) : (memref<2xi64>) -> ()
      scf.yield %nh, %nb : memref<2xi64>, memref<?xi8>
    }
    %close_ref = memref.get_global @__ly_repr_rparen : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %cl_h, %cl_b = func.call @__ly_unicode_from_valid_utf8(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%loop#0, %loop#1, %cl_h, %cl_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%cl_h) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  // Payload block entry count (0 for an absent block). Bit 62 of the count
  // word flags "repr as a tuple" (the PEP 654 naked-exception wrap shows
  // `('', (exc,))`); every count consumer masks it off.
  func.func private @__ly_exc_payload_count(%block_word: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_count"} {
    %zero = arith.constant 0 : i64
    %mask = arith.constant 4611686018427387903 : i64
    %absent = arith.cmpi eq, %block_word, %zero : i64
    %count = scf.if %absent -> (i64) {
      scf.yield %zero : i64
    } else {
      %block_ptr = llvm.inttoptr %block_word : i64 to !llvm.ptr
      %loaded = llvm.load %block_ptr : !llvm.ptr -> i64
      %masked = arith.andi %loaded, %mask : i64
      scf.yield %masked : i64
    }
    func.return %count : i64
  }

  // Whether the payload block asked for tuple-style repr (bit 62).
  func.func private @__ly_exc_payload_tuple_flag(%block_word: i64) -> i1 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_tuple_flag"} {
    %zero = arith.constant 0 : i64
    %flag_bit = arith.constant 4611686018427387904 : i64
    %absent = arith.cmpi eq, %block_word, %zero : i64
    %flag = scf.if %absent -> (i1) {
      %false_v = arith.constant false
      scf.yield %false_v : i1
    } else {
      %block_ptr = llvm.inttoptr %block_word : i64 to !llvm.ptr
      %loaded = llvm.load %block_ptr : !llvm.ptr -> i64
      %bit = arith.andi %loaded, %flag_bit : i64
      %set = arith.cmpi ne, %bit, %zero : i64
      scf.yield %set : i1
    }
    func.return %flag : i1
  }

  // ---- User-exception field block (extended word 4) -------------------
  // A user exception class declares instance fields; the exception object's
  // layout is fixed by the taxonomy (3-word header + message), so the fields
  // live in a separate [count, count x box16] block hung off word 4 -- the
  // same shape word 3 uses for group members. Reached only through these
  // primitives; `release_exception_extras` already frees the block and
  // releases each owning slot.

  // The field block for %header, allocating it on first use. Idempotent:
  // whichever store runs first materializes the block, so no construction
  // path has to know the field count up front.
  func.func private @__ly_exc_fields_block(%header: memref<3xi64>, %count: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.interior_word, ly.runtime.primitive = "fields_block"} {
    %zero = arith.constant 0 : i64
    %fields_slot = arith.constant 4 : i64
    %existing = func.call @__ly_exc_ext_get(%header, %fields_slot) : (memref<3xi64>, i64) -> i64
    %absent = arith.cmpi eq, %existing, %zero : i64
    %block = scf.if %absent -> (i64) {
      %fresh = func.call @__ly_exc_payload_alloc(%count) : (i64) -> i64
      func.call @__ly_exc_ext_set(%header, %fields_slot, %fresh) : (memref<3xi64>, i64, i64) -> ()
      scf.yield %fresh : i64
    } else {
      scf.yield %existing : i64
    }
    func.return %block : i64
  }

  // One word of the %slot-th box of a payload block. The generic reader that
  // pairs with payload_store_words: the lowering rebuilds a payload's memref
  // group from the box's pointer/size words (BoxLayout.h offsets).
  func.func private @__ly_exc_payload_box_word(%block_word: i64, %slot: i64, %word: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.interior_word, ly.runtime.primitive = "payload_box_word"} {
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %one = arith.constant 1 : i64
    %block_ptr = llvm.inttoptr %block_word : i64 to !llvm.ptr
    %boxes_base = arith.muli %slot, %c16 : i64
    %box_base = arith.addi %boxes_base, %one : i64
    %box_offset = arith.addi %box_base, %word : i64
    %word_ptr = llvm.getelementptr %block_ptr[%box_offset] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %value = llvm.load %word_ptr : !llvm.ptr -> i64
    func.return %value : i64
  }

  // Address of the %slot-th box. An erased-`object` field stores the box
  // itself as its value (the box words ARE the canonical object handle), so
  // that read needs the box address rather than the payload words.
  func.func private @__ly_exc_payload_box_ptr(%block_word: i64, %slot: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.interior_word, ly.runtime.primitive = "payload_box_ptr"} {
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %one = arith.constant 1 : i64
    %block_ptr = llvm.inttoptr %block_word : i64 to !llvm.ptr
    %boxes_base = arith.muli %slot, %c16 : i64
    %box_base = arith.addi %boxes_base, %one : i64
    %box_ptr = llvm.getelementptr %block_ptr[%box_base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %box_word = llvm.ptrtoint %box_ptr : !llvm.ptr to i64
    func.return %box_word : i64
  }

  // Release whatever the %slot-th box owns (a no-op while its owned flag is
  // zero), so a field rebind drops the previous payload exactly once.
  func.func private @__ly_exc_payload_release_slot(%block_word: i64, %slot: i64) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_release_slot"} {
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %one = arith.constant 1 : i64
    %block_ptr = llvm.inttoptr %block_word : i64 to !llvm.ptr
    %boxes_base = arith.muli %slot, %c16 : i64
    %box_base = arith.addi %boxes_base, %one : i64
    %box_ptr = llvm.getelementptr %block_ptr[%box_base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    func.call @release_payload_slot_ptr(%box_ptr) : (!llvm.ptr) -> ()
    func.return
  }

  // Zeroed dummies for the absent sides of a star split result: the paired
  // flag gates every consumer, so the views are never dereferenced.
  memref.global "private" @__ly_exc_dummy_header : memref<3xi64> = dense<0>
  memref.global "private" @__ly_exc_dummy_str : memref<2xi64> = dense<0>

  // Fresh group carrying %count member slots, derived from %eh's class and
  // message (PEP 654 split keeps the original metadata on both halves).
  func.func private @__ly_exc_derive_group(%eh: memref<3xi64>, %mh: memref<2xi64>, %mb: memref<?xi8>, %count: i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "derive_group"} {
    %class_slot = arith.constant 2 : index
    %payload_slot = arith.constant 3 : i64
    %class_id = memref.load %eh[%class_slot] : memref<3xi64>
    %fresh:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    // Adopt the original message: release the fresh empty one, retain ours.
    // ⛔ AND RECORD IT. This is a message producer like `__init__` is, and it
    // is the one that does not look like one -- it returns four values rather
    // than the taxonomy's triple. Leaving word 6 on the empty message it just
    // released made the split hand a box a dangling record.
    func.call @LyUnicode_DecRef(%fresh#1) : (memref<2xi64>) -> ()
    %retained:2 = func.call @__ly_unicode_retain_self(%mh, %mb) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_exc_set_message(%fresh#0, %retained#0) : (memref<3xi64>, memref<2xi64>) -> ()
    %block = func.call @__ly_exc_payload_alloc(%count) : (i64) -> i64
    func.call @__ly_exc_ext_set(%fresh#0, %payload_slot, %block) : (memref<3xi64>, i64, i64) -> ()
    func.return %fresh#0, %retained#0, %retained#1, %block : memref<3xi64>, memref<2xi64>, memref<?xi8>, i64
  }

  // PEP 654 split: partition %e against handler class %handler. Returns
  // (has_matched, matched triple, has_rest, rest triple); present sides are
  // OWNED by the caller, absent sides are the zero dummies. A naked matching
  // exception comes back wrapped in an ExceptionGroup('', (exc,)) (the
  // BaseExceptionGroup variant when it is not an Exception).
  func.func private @__ly_exc_star_split_rec(%eh: memref<3xi64>, %mh: memref<2xi64>, %mb: memref<?xi8>, %handler: i64) -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "star_split_rec"} {
    %true_v = arith.constant true
    %false_v = arith.constant false
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %class_slot = arith.constant 2 : index
    %payload_slot = arith.constant 3 : i64
    %group_root = arith.constant 101 : i64
    %exception_root = arith.constant 50 : i64
    %group_id = arith.constant 102 : i64
    %base_group_id = arith.constant 101 : i64
    %tuple_bit = arith.constant 4611686018427387904 : i64
    %dummy_h = memref.get_global @__ly_exc_dummy_header : memref<3xi64>
    %dummy_s = memref.get_global @__ly_exc_dummy_str : memref<2xi64>
    %dummy_b = memref.alloca(%c0) : memref<?xi8>

    %class_id = memref.load %eh[%class_slot] : memref<3xi64>
    %block = func.call @__ly_exc_ext_get(%eh, %payload_slot) : (memref<3xi64>, i64) -> i64
    %member_count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %is_group = func.call @LyEH_ClassIdMatches(%class_id, %group_root) : (i64, i64) -> i1
    %has_members = arith.cmpi sgt, %member_count, %zero : i64
    %grouped = arith.andi %is_group, %has_members : i1

    %result:8 = scf.if %grouped -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) {
      // First pass: how many members land on each side (recursively).
      %count_index = arith.index_cast %member_count : i64 to index
      %tally:2 = scf.for %i = %c0 to %count_index step %c1 iter_args(%m = %zero, %r = %zero) -> (i64, i64) {
        %i64v = arith.index_cast %i : index to i64
        %member:3 = func.call @__ly_exc_payload_view(%block, %i64v) : (i64, i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
        %sub:8 = func.call @__ly_exc_star_split_rec(%member#0, %member#1, %member#2, %handler) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64) -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>)
        // Tally only: drop the split halves again.
        scf.if %sub#0 {
          func.call @LyBaseException_DecRef(%sub#1, %sub#2, %sub#3) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
        }
        scf.if %sub#4 {
          func.call @LyBaseException_DecRef(%sub#5, %sub#6, %sub#7) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
        }
        %m_inc = arith.extui %sub#0 : i1 to i64
        %r_inc = arith.extui %sub#4 : i1 to i64
        %m_next = arith.addi %m, %m_inc : i64
        %r_next = arith.addi %r, %r_inc : i64
        scf.yield %m_next, %r_next : i64, i64
      }
      %no_match = arith.cmpi eq, %tally#0, %zero : i64
      %full_match = arith.cmpi eq, %tally#1, %zero : i64
      %whole:8 = scf.if %no_match -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) {
        %sub0 = memref.subview %eh[0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1]>>
        %view0 = memref.cast %sub0 : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
        func.call @Ly_IncRef(%view0) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
        scf.yield %false_v, %dummy_h, %dummy_s, %dummy_b, %true_v, %eh, %mh, %mb : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
      } else {
        %all:8 = scf.if %full_match -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) {
          %subw = memref.subview %eh[0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1]>>
          %vieww = memref.cast %subw : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
          func.call @Ly_IncRef(%vieww) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
          scf.yield %true_v, %eh, %mh, %mb, %false_v, %dummy_h, %dummy_s, %dummy_b : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
        } else {
          // Real split: derive a group per side and distribute the halves.
          %mg:4 = func.call @__ly_exc_derive_group(%eh, %mh, %mb, %tally#0) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64)
          %rg:4 = func.call @__ly_exc_derive_group(%eh, %mh, %mb, %tally#1) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64)
          %fill:2 = scf.for %i = %c0 to %count_index step %c1 iter_args(%mi = %zero, %ri = %zero) -> (i64, i64) {
            %i64v = arith.index_cast %i : index to i64
            %member:3 = func.call @__ly_exc_payload_view(%block, %i64v) : (i64, i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
            %sub:8 = func.call @__ly_exc_star_split_rec(%member#0, %member#1, %member#2, %handler) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64) -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>)
            %mi_next = scf.if %sub#0 -> (i64) {
              func.call @__ly_exc_payload_store(%mg#3, %mi, %sub#1, %sub#2, %sub#3) : (i64, i64, memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
              func.call @LyBaseException_DecRef(%sub#1, %sub#2, %sub#3) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
              %n = arith.addi %mi, %one : i64
              scf.yield %n : i64
            } else {
              scf.yield %mi : i64
            }
            %ri_next = scf.if %sub#4 -> (i64) {
              func.call @__ly_exc_payload_store(%rg#3, %ri, %sub#5, %sub#6, %sub#7) : (i64, i64, memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
              func.call @LyBaseException_DecRef(%sub#5, %sub#6, %sub#7) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
              %n = arith.addi %ri, %one : i64
              scf.yield %n : i64
            } else {
              scf.yield %ri : i64
            }
            scf.yield %mi_next, %ri_next : i64, i64
          }
          scf.yield %true_v, %mg#0, %mg#1, %mg#2, %true_v, %rg#0, %rg#1, %rg#2 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
        }
        scf.yield %all#0, %all#1, %all#2, %all#3, %all#4, %all#5, %all#6, %all#7 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
      }
      scf.yield %whole#0, %whole#1, %whole#2, %whole#3, %whole#4, %whole#5, %whole#6, %whole#7 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
    } else {
      // Naked exception (or a group without members): all-or-nothing.
      %matches = func.call @LyEH_ClassIdMatches(%class_id, %handler) : (i64, i64) -> i1
      %naked:8 = scf.if %matches -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) {
        // A matching leaf moves into the matched half as ITSELF; the
        // synthesized wrap applies only to a top-level naked exception
        // (star_split below).
        %subm = memref.subview %eh[0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1]>>
        %viewm = memref.cast %subm : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
        func.call @Ly_IncRef(%viewm) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
        scf.yield %true_v, %eh, %mh, %mb, %false_v, %dummy_h, %dummy_s, %dummy_b : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
      } else {
        %sub1 = memref.subview %eh[0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1]>>
        %view1 = memref.cast %sub1 : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
        func.call @Ly_IncRef(%view1) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
        scf.yield %false_v, %dummy_h, %dummy_s, %dummy_b, %true_v, %eh, %mh, %mb : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
      }
      scf.yield %naked#0, %naked#1, %naked#2, %naked#3, %naked#4, %naked#5, %naked#6, %naked#7 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1, %result#2, %result#3, %result#4, %result#5, %result#6, %result#7 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // Top-level star split (the manifest entry): a naked exception either
  // misses whole or comes back wrapped in ExceptionGroup('', (exc,)) --
  // BaseExceptionGroup when it is not an Exception. Groups delegate to the
  // recursive split, whose leaves stay unwrapped.
  func.func private @__ly_exc_star_split(%eh: memref<3xi64>, %mh: memref<2xi64>, %mb: memref<?xi8>, %handler: i64) -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "star_split"} {
    %true_v = arith.constant true
    %false_v = arith.constant false
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %class_slot = arith.constant 2 : index
    %payload_slot = arith.constant 3 : i64
    %group_root = arith.constant 101 : i64
    %exception_root = arith.constant 50 : i64
    %group_id = arith.constant 102 : i64
    %base_group_id = arith.constant 101 : i64
    %tuple_bit = arith.constant 4611686018427387904 : i64
    %dummy_h = memref.get_global @__ly_exc_dummy_header : memref<3xi64>
    %dummy_s = memref.get_global @__ly_exc_dummy_str : memref<2xi64>
    %dummy_b = memref.alloca(%c0) : memref<?xi8>
    %class_id = memref.load %eh[%class_slot] : memref<3xi64>
    %block = func.call @__ly_exc_ext_get(%eh, %payload_slot) : (memref<3xi64>, i64) -> i64
    %member_count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %is_group = func.call @LyEH_ClassIdMatches(%class_id, %group_root) : (i64, i64) -> i1
    %has_members = arith.cmpi sgt, %member_count, %zero : i64
    %grouped = arith.andi %is_group, %has_members : i1
    %result:8 = scf.if %grouped -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) {
      %sub:8 = func.call @__ly_exc_star_split_rec(%eh, %mh, %mb, %handler) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, i64) -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>)
      scf.yield %sub#0, %sub#1, %sub#2, %sub#3, %sub#4, %sub#5, %sub#6, %sub#7 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
    } else {
      %matches = func.call @LyEH_ClassIdMatches(%class_id, %handler) : (i64, i64) -> i1
      %naked:8 = scf.if %matches -> (i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>) {
        %is_exception = func.call @LyEH_ClassIdMatches(%class_id, %exception_root) : (i64, i64) -> i1
        %wrap_id = arith.select %is_exception, %group_id, %base_group_id : i64
        %wrap:3 = func.call @LyBaseException_New(%wrap_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
        %wblock = func.call @__ly_exc_payload_alloc(%one) : (i64) -> i64
        func.call @__ly_exc_ext_set(%wrap#0, %payload_slot, %wblock) : (memref<3xi64>, i64, i64) -> ()
        func.call @__ly_exc_payload_store(%wblock, %zero, %eh, %mh, %mb) : (i64, i64, memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
        // Tuple-style repr for the synthesized wrap.
        %wblock_ptr = llvm.inttoptr %wblock : i64 to !llvm.ptr
        %flagged = arith.ori %one, %tuple_bit : i64
        llvm.store %flagged, %wblock_ptr : i64, !llvm.ptr
        scf.yield %true_v, %wrap#0, %wrap#1, %wrap#2, %false_v, %dummy_h, %dummy_s, %dummy_b : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
      } else {
        %subn = memref.subview %eh[0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1]>>
        %viewn = memref.cast %subn : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
        func.call @Ly_IncRef(%viewn) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
        scf.yield %false_v, %dummy_h, %dummy_s, %dummy_b, %true_v, %eh, %mh, %mb : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
      }
      scf.yield %naked#0, %naked#1, %naked#2, %naked#3, %naked#4, %naked#5, %naked#6, %naked#7 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1, %result#2, %result#3, %result#4, %result#5, %result#6, %result#7 : i1, memref<3xi64>, memref<2xi64>, memref<?xi8>, i1, memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // Combine the star frame's leftovers into one fresh group: the collected
  // chain-node exceptions (in raise order) followed by the residual. Member
  // references are retained here; the caller still releases the nodes (their
  // reference moves to the group, net one owner).
  func.func private @__ly_exc_star_combine(%nodes_ptr: !llvm.ptr, %count: i64, %has_res: i1, %res_eh: memref<3xi64>, %res_mh: memref<2xi64>, %res_mb: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "star_combine"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %exception_root = arith.constant 50 : i64
    %group_id = arith.constant 102 : i64
    %base_group_id = arith.constant 101 : i64
    %payload_slot = arith.constant 3 : i64
    %res_inc = arith.extui %has_res : i1 to i64
    %total = arith.addi %count, %res_inc : i64
    %true_v = arith.constant true
    %c2 = arith.constant 2 : i64
    %c3 = arith.constant 3 : i64
    %part0 = arith.constant 0 : i64
    %part1 = arith.constant 1 : i64
    %field_aligned = arith.constant 1 : i64
    %field_size = arith.constant 3 : i64

    // Borrowed member views out of a parked chain node, each field's address
    // from `__ly_chain_node_part_field` (the node's own struct type, laid out
    // for the target). These were word indices -- 2, 7, 12, 14 -- which are
    // the LP64 offsets and nothing else: on armv7 word 2 is the header's
    // offset field, not its pointer.
    //
    // The clause cells and the node's aligned members hold POINTERS, so they
    // are loaded as pointers. The narrowing at the view calls is the one place
    // left where an address becomes a word, and it is
    // `__ly_global_view_*`'s signature that forces it.
    %count_index = arith.index_cast %count : i64 to index

    // ExceptionGroup unless any member falls outside Exception.
    %all_exc = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %true_v) -> (i1) {
      %i64v = arith.index_cast %i : index to i64
      %node_slot = llvm.getelementptr %nodes_ptr[%i64v] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %node_ptr = llvm.load %node_slot : !llvm.ptr -> !llvm.ptr
      %eh_slot = func.call @__ly_chain_node_part_field(%node_ptr, %part0, %field_aligned) : (!llvm.ptr, i64, i64) -> !llvm.ptr
      %eh_ptr = llvm.load %eh_slot : !llvm.ptr -> !llvm.ptr
      %class_slot = llvm.getelementptr %eh_ptr[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %class_id = llvm.load %class_slot : !llvm.ptr -> i64
      %is_exc = func.call @LyEH_ClassIdMatches(%class_id, %exception_root) : (i64, i64) -> i1
      %next = arith.andi %acc, %is_exc : i1
      scf.yield %next : i1
    }
    %res_is_exc = scf.if %has_res -> (i1) {
      %class_slot_r = arith.constant 2 : index
      %res_class = memref.load %res_eh[%class_slot_r] : memref<3xi64>
      %is_exc = func.call @LyEH_ClassIdMatches(%res_class, %exception_root) : (i64, i64) -> i1
      scf.yield %is_exc : i1
    } else {
      scf.yield %true_v : i1
    }
    %all = arith.andi %all_exc, %res_is_exc : i1
    %wrap_id = arith.select %all, %group_id, %base_group_id : i64

    %wrap:3 = func.call @LyBaseException_New(%wrap_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %block = func.call @__ly_exc_payload_alloc(%total) : (i64) -> i64
    func.call @__ly_exc_ext_set(%wrap#0, %payload_slot, %block) : (memref<3xi64>, i64, i64) -> ()
    scf.for %i = %c0 to %count_index step %c1 {
      %i64v = arith.index_cast %i : index to i64
      %node_slot = llvm.getelementptr %nodes_ptr[%i64v] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %node_ptr = llvm.load %node_slot : !llvm.ptr -> !llvm.ptr
      %eh_slot = func.call @__ly_chain_node_part_field(%node_ptr, %part0, %field_aligned) : (!llvm.ptr, i64, i64) -> !llvm.ptr
      %eh_aligned = llvm.load %eh_slot : !llvm.ptr -> !llvm.ptr
      %eh_word = llvm.ptrtoint %eh_aligned : !llvm.ptr to i64
      %mh_slot = func.call @__ly_chain_node_part_field(%node_ptr, %part1, %field_aligned) : (!llvm.ptr, i64, i64) -> !llvm.ptr
      %mh_aligned = llvm.load %mh_slot : !llvm.ptr -> !llvm.ptr
      %mh_word = llvm.ptrtoint %mh_aligned : !llvm.ptr to i64
      %mb_slot = func.call @__ly_chain_node_part_field(%node_ptr, %c2, %field_aligned) : (!llvm.ptr, i64, i64) -> !llvm.ptr
      %mb_aligned = llvm.load %mb_slot : !llvm.ptr -> !llvm.ptr
      %mb_word = llvm.ptrtoint %mb_aligned : !llvm.ptr to i64
      %len_slot = func.call @__ly_chain_node_part_field(%node_ptr, %c2, %field_size) : (!llvm.ptr, i64, i64) -> !llvm.ptr
      %mb_len = llvm.load %len_slot : !llvm.ptr -> i64
      %eh_dyn = func.call @__ly_global_view_i64(%eh_word, %c3) : (i64, i64) -> memref<?xi64>
      %eh_view = memref.cast %eh_dyn : memref<?xi64> to memref<3xi64>
      %two = arith.constant 2 : i64
      %mh_dyn = func.call @__ly_global_view_i64(%mh_word, %two) : (i64, i64) -> memref<?xi64>
      %mh_view = memref.cast %mh_dyn : memref<?xi64> to memref<2xi64>
      %mb_view = func.call @__ly_global_view_i8(%mb_word, %mb_len) : (i64, i64) -> memref<?xi8>
      func.call @__ly_exc_payload_store(%block, %i64v, %eh_view, %mh_view, %mb_view) : (i64, i64, memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    }
    scf.if %has_res {
      func.call @__ly_exc_payload_store(%block, %count, %res_eh, %res_mh, %res_mb) : (i64, i64, memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    }
    func.return %wrap#0, %wrap#1, %wrap#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // ===== impls: exception =====
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyEH_BorrowCurrentException() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "borrow_current"}
  // Native support (RuntimeSupportBuilder): extended-word release for the
  // deallocator, raw view rebuilders for the payload boxes, and the taxonomy
  // subtree matcher for the group-aware paths.
  func.func private @release_exception_extras(%header_ptr: !llvm.ptr)
  func.func private @release_payload_slot_ptr(%slot: !llvm.ptr)
  func.func private @__ly_chain_node_part_field(%node: !llvm.ptr, %section: i64, %field: i64) -> !llvm.ptr
  func.func private @LyEH_ClassIdMatches(%raised: i64, %handler: i64) -> i1

  func.func private @LyException_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.Exception", ly.runtime.shape}
  func.func private @LyRuntimeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.RuntimeError", ly.runtime.shape}
  func.func private @LyTypeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.TypeError", ly.runtime.shape}
  func.func private @LyValueError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ValueError", ly.runtime.shape}
  func.func private @LyArithmeticError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.shape}
  func.func private @LyLookupError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.LookupError", ly.runtime.shape}
  func.func private @LyZeroDivisionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.shape}
  func.func private @LyKeyError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.KeyError", ly.runtime.shape}
  func.func private @LyIndexError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.IndexError", ly.runtime.shape}
  func.func private @LyAssertionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.AssertionError", ly.runtime.shape}
  func.func private @LyKeyboardInterrupt_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.shape}
  func.func private @LyBaseExceptionGroup_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.shape}
  func.func private @LyExceptionGroup_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.shape}
  func.func private @LyFloatingPointError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.shape}
  func.func private @LyOverflowError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.OverflowError", ly.runtime.shape}
  func.func private @LyBufferError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BufferError", ly.runtime.shape}
  func.func private @LyEOFError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.EOFError", ly.runtime.shape}
  func.func private @LyImportError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ImportError", ly.runtime.shape}
  func.func private @LyModuleNotFoundError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.shape}
  func.func private @LyMemoryError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.MemoryError", ly.runtime.shape}
  func.func private @LyNameError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.NameError", ly.runtime.shape}
  func.func private @LyUnboundLocalError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.shape}
  func.func private @LyAttributeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.AttributeError", ly.runtime.shape}
  func.func private @LyReferenceError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ReferenceError", ly.runtime.shape}
  func.func private @LyNotImplementedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.shape}
  func.func private @LyRecursionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.RecursionError", ly.runtime.shape}
  func.func private @LyPythonFinalizationError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.shape}
  func.func private @LySyntaxError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SyntaxError", ly.runtime.shape}
  func.func private @LyIndentationError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.IndentationError", ly.runtime.shape}
  func.func private @LyTabError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.TabError", ly.runtime.shape}
  func.func private @LySystemError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SystemError", ly.runtime.shape}
  func.func private @LyUnicodeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeError", ly.runtime.shape}
  func.func private @LyUnicodeDecodeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.shape}
  func.func private @LyUnicodeEncodeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.shape}
  func.func private @LyUnicodeTranslateError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.shape}
  func.func private @LyWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.Warning", ly.runtime.shape}
  func.func private @LyBytesWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BytesWarning", ly.runtime.shape}
  func.func private @LyDeprecationWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.shape}
  func.func private @LyEncodingWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.shape}
  func.func private @LyFutureWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FutureWarning", ly.runtime.shape}
  func.func private @LyImportWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ImportWarning", ly.runtime.shape}
  func.func private @LyPendingDeprecationWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.shape}
  func.func private @LyResourceWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.shape}
  func.func private @LyRuntimeWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.shape}
  func.func private @LySyntaxWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.shape}
  func.func private @LyUnicodeWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.shape}
  func.func private @LyUserWarning_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.UserWarning", ly.runtime.shape}
  func.func private @LyBlockingIOError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.shape}
  func.func private @LyChildProcessError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.shape}
  func.func private @LyConnectionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionError", ly.runtime.shape}
  func.func private @LyBrokenPipeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.shape}
  func.func private @LyConnectionAbortedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.shape}
  func.func private @LyConnectionRefusedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.shape}
  func.func private @LyConnectionResetError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.shape}
  func.func private @LyFileExistsError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FileExistsError", ly.runtime.shape}
  func.func private @LyInterruptedError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.InterruptedError", ly.runtime.shape}
  func.func private @LyIsADirectoryError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.shape}
  func.func private @LyNotADirectoryError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.shape}
  func.func private @LyPermissionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.PermissionError", ly.runtime.shape}
  func.func private @LyProcessLookupError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.shape}
  func.func private @LyTimeoutError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.TimeoutError", ly.runtime.shape}
  func.func private @LyStopIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopIteration", ly.runtime.shape}
  func.func private @LyStopAsyncIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.shape}
  func.func private @LySystemExit_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.SystemExit", ly.runtime.shape}
  func.func private @LyGeneratorExit_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.shape}
  func.func private @LyOSError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.OSError", ly.runtime.shape}
  func.func private @LyFileNotFoundError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.shape}
  func.func @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    func.call @__ly_exc_set_message(%header, %message_header) : (memref<3xi64>, memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%old_message_header) : (memref<2xi64>) -> ()
    func.return %header, %message_header, %message_bytes : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    // str(KeyError(x)) is repr(x) in CPython, so the stored message IS the
    // argument repr -- the traceback and str() both read the message lane
    // verbatim, so the repr cannot move to __str__.
    //
    // The argument itself therefore also goes into the payload block: .args
    // reads back from the message lane otherwise, and `KeyError("zz").args[0]`
    // was "'zz'". Un-quoting a repr is not a thing; keeping the value is.
    // An empty message means no argument (the same rule LyBaseException_Args
    // applies), and an empty payload block would turn args () into ('',).
    %zero_len = arith.constant 0 : i64
    %arg_len = func.call @__ly_unicode_count(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %has_arg = arith.cmpi sgt, %arg_len, %zero_len : i64
    %repr_header, %repr_bytes = func.call @LyUnicode_Repr(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %repr_header, %repr_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    scf.if %has_arg {
      %one_arg = arith.constant 1 : i64
      %slot_zero = arith.constant 0 : i64
      %payload_slot = arith.constant 3 : i64
      %block = func.call @__ly_exc_payload_alloc(%one_arg) : (i64) -> i64
      func.call @__ly_exc_ext_set(%result#0, %payload_slot, %block) : (memref<3xi64>, i64, i64) -> ()
      %kept:2 = func.call @__ly_unicode_retain_self(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @__ly_exc_payload_store_unicode(%block, %slot_zero, %kept#0, %kept#1) : (i64, i64, memref<2xi64>, memref<?xi8>) -> ()
    }
    func.call @LyUnicode_DecRef(%message_header) : (memref<2xi64>) -> ()
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyGeneratorExit_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // The exception ALLOCATION is 5 words even though the header view (and
  // every signature) stays memref<3xi64>: words [3] (group/args payload
  // block) and [4] (user-exception field block) ride inside the allocation
  // so they survive the TLS throw path (g_current_parts stores only the
  // three descriptors) without widening every exception signature in the
  // manifests. The extended words are reached only through
  // @__ly_exc_ext_get / @__ly_exc_ext_set, never through the 3-word view.
  // SystemExit's exit code, recorded on the object at construction because that
  // is where its static type is still known: a boxed value in the payload block
  // cannot be read back as an i64 without a per-class unboxer, and the top-level
  // runner reads raw words. Biased by one; slot 0 means "no int code".
  func.func private @__ly_systemexit_set_code(%header: memref<3xi64>, %value_header: memref<2xi64>) attributes {ly.runtime.contract = "builtins.SystemExit", ly.runtime.primitive = "set_code"} {
    %value = func.call @LyLong_AsI64(%value_header) : (memref<2xi64>) -> i64
    func.call @__ly_systemexit_set_code_i64(%header, %value) : (memref<3xi64>, i64) -> ()
    func.return
  }

  // ⛔ bool NEEDS ITS OWN ENTRY POINT rather than a coercion at the call: a bool
  // is one i1 lane, not the int triple, so it cannot reach the unboxer above --
  // and CPython counts it as an int, so `raise SystemExit(True)` exits 1 in
  // silence rather than printing "True".
  func.func private @__ly_systemexit_set_code_bool(%header: memref<3xi64>, %boxed: memref<3xi64>) attributes {ly.runtime.contract = "builtins.SystemExit", ly.runtime.primitive = "set_code_bool"} {
    %value = func.call @LyBool_Unbox(%boxed) : (memref<3xi64>) -> i1
    %widened = arith.extui %value : i1 to i64
    func.call @__ly_systemexit_set_code_i64(%header, %widened) : (memref<3xi64>, i64) -> ()
    func.return
  }

  func.func private @__ly_systemexit_set_code_i64(%header: memref<3xi64>, %value: i64) attributes {ly.runtime.contract = "builtins.SystemExit", ly.runtime.primitive = "set_code_i64"} {
    %one = arith.constant 1 : i64
    %code_slot = arith.constant 5 : i64
    %biased = arith.addi %value, %one : i64
    func.call @__ly_exc_ext_set(%header, %code_slot, %biased) : (memref<3xi64>, i64, i64) -> ()
    func.return
  }

  func.func @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_exception = arith.constant 5 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %class_slot = arith.constant 2 : index
    %payload_slot = arith.constant 3 : index
    %fields_slot = arith.constant 4 : index
    // The exit code rides the exception object, the way CPython's .code does.
    // It is stored biased by one so that zero means "no code": SystemExit(0)
    // and SystemExit() are both silent but only one of them is an int.
    %code_slot = arith.constant 5 : index
    %zero_index = arith.constant 0 : index
    %zero_len = arith.constant 0 : i64

    // ⭐ WORD 6 IS THE MESSAGE, and it is what lets a box hold an exception. The
    // message is a physical LANE of the contract and the object never named it,
    // so a box had to cache all three lanes to hand the exception back. The
    // word is a cache of the lane, kept in sync at the three places that
    // produce one (New, Init, InitPayloadMessage) and released by the lane's
    // own owner -- LyBaseException_DecRef -- so it adds no reference.
    %message_slot = arith.constant 6 : index
    %block_bytes = arith.constant 56 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header = memref.view %block[%zero_index][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<3xi64>
    %extended = memref.view %block[%zero_index][] : memref<?xi8> to memref<7xi64>
    %empty_bytes = memref.alloca(%zero_index) : memref<?xi8>
    %message_header, %message_bytes = func.call @__ly_unicode_from_valid_utf8(%empty_bytes, %zero_index, %zero_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %message_ptr_index = memref.extract_aligned_pointer_as_index %message_header : memref<2xi64> -> index
    %message_ptr = arith.index_cast %message_ptr_index : index to i64

    memref.store %one, %header[%refcount_slot] : memref<3xi64>
    memref.store %layout_exception, %header[%layout_slot] : memref<3xi64>
    memref.store %class_id, %header[%class_slot] : memref<3xi64>
    memref.store %zero, %extended[%payload_slot] : memref<7xi64>
    memref.store %zero, %extended[%fields_slot] : memref<7xi64>
    memref.store %zero, %extended[%code_slot] : memref<7xi64>
    memref.store %message_ptr, %extended[%message_slot] : memref<7xi64>

    func.return %header, %message_header, %message_bytes : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 50 : i64, ly.runtime.contract = "builtins.Exception", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 51 : i64, ly.runtime.contract = "builtins.RuntimeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 52 : i64, ly.runtime.contract = "builtins.TypeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 53 : i64, ly.runtime.contract = "builtins.ValueError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 59 : i64, ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 60 : i64, ly.runtime.contract = "builtins.LookupError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 61 : i64, ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 54 : i64, ly.runtime.contract = "builtins.KeyError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 55 : i64, ly.runtime.contract = "builtins.IndexError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 56 : i64, ly.runtime.contract = "builtins.AssertionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 57 : i64, ly.runtime.contract = "builtins.StopIteration", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 58 : i64, ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 64 : i64, ly.runtime.contract = "builtins.SystemExit", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyGeneratorExit_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 68 : i64, ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 66 : i64, ly.runtime.contract = "builtins.OSError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 67 : i64, ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  // Per-program builtin/user exception-class name table (synthesized by the
  // lowering's support builder; user-defined ids resolve through its
  // fallback hook). Returns a NUL-terminated ASCII name.
  func.func private @exception_class_name(%class_id: i64) -> !llvm.ptr

  // ⭐ `type(e).__name__`, keyed by the DYNAMIC class id the way the repr below
  // is: an instance caught through a base-class handler answers the class it
  // WAS RAISED AS, which is the one thing the emitter's static fold cannot do
  // for an exception (the handler's static class is the one caught).
  //
  // ⛔ Not on the typed manifest surface -- CPython has no
  // BaseException.__class_name__ -- so, like str.__int__, it is reachable only
  // through the emitter's own op and never appears in the class declaration.
  func.func @LyBaseException_ClassName(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__class_name__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero_i8 = arith.constant 0 : i8
    %cap = arith.constant 64 : index
    %class_id = memref.load %header[%c2] : memref<3xi64>
    %name_ptr = func.call @exception_class_name(%class_id) : (i64) -> !llvm.ptr
    // The same bounded copy the repr does: taxonomy names are short ASCII, and
    // 64 caps a runaway pointer rather than a real name.
    %buffer = memref.alloca() : memref<64xi8>
    %true_scan = arith.constant true
    %dot_byte = arith.constant 46 : i8
    // ⛔ The taxonomy name is MODULE-QUALIFIED ("lib.SubErr") because the
    // traceback prints it that way, and CPython's `__name__` and `repr(e)` are
    // the LEAF -- so the scan also records where the last '.' left off, and the
    // leaf is shifted down to offset 0 for every reader below.
    %scan:3 = scf.while (%i = %c0, %go = %true_scan, %start = %c0) : (index, i1, index) -> (index, i1, index) {
      %in_bounds = arith.cmpi ult, %i, %cap : index
      %continue = arith.andi %in_bounds, %go : i1
      scf.condition(%continue) %i, %go, %start : index, i1, index
    } do {
    ^bb0(%i: index, %go: i1, %start: index):
      %i_i64 = arith.index_cast %i : index to i64
      %slot = llvm.getelementptr %name_ptr[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      %is_nul = arith.cmpi eq, %byte, %zero_i8 : i8
      %true_x = arith.constant true
      %not_nul = arith.xori %is_nul, %true_x : i1
      scf.if %not_nul {
        memref.store %byte, %buffer[%i] : memref<64xi8>
      }
      %next = arith.addi %i, %c1 : index
      %kept = arith.select %is_nul, %i, %next : index
      %is_dot = arith.cmpi eq, %byte, %dot_byte : i8
      %next_start = arith.select %is_dot, %next, %start : index
      scf.yield %kept, %not_nul, %next_start : index, i1, index
    }
    scf.for %i = %scan#2 to %scan#0 step %c1 {
      %byte = memref.load %buffer[%i] : memref<64xi8>
      %at = arith.subi %i, %scan#2 : index
      memref.store %byte, %buffer[%at] : memref<64xi8>
    }
    %leaf_len = arith.subi %scan#0, %scan#2 : index
    %name_len = arith.index_cast %leaf_len : index to i64
    %name_dyn = memref.cast %buffer : memref<64xi8> to memref<?xi8>
    %name_h, %name_b = func.call @__ly_unicode_from_valid_utf8(%name_dyn, %c0, %name_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %name_h, %name_b : memref<2xi64>, memref<?xi8>
  }

  // repr(e) keyed by the DYNAMIC class id in the exception header, so an
  // instance caught through a base-class handler (or a user subclass, once
  // the class hook resolves it) still renders its own class name.
  func.func private @__ly_exception_repr_by_id(%header: memref<3xi64>, %message_header: memref<2xi64>, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "repr_by_id", ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero_i8 = arith.constant 0 : i8
    %cap = arith.constant 64 : index
    %class_id = memref.load %header[%c2] : memref<3xi64>
    %name_ptr = func.call @exception_class_name(%class_id) : (i64) -> !llvm.ptr
    // Copy the NUL-terminated name into a bounded local buffer (taxonomy
    // names are short ASCII; 64 caps runaway pointers, not real names).
    %buffer = memref.alloca() : memref<64xi8>
    %true_scan = arith.constant true
    %dot_byte = arith.constant 46 : i8
    // ⛔ The taxonomy name is MODULE-QUALIFIED ("lib.SubErr") because the
    // traceback prints it that way, and CPython's `__name__` and `repr(e)` are
    // the LEAF -- so the scan also records where the last '.' left off, and the
    // leaf is shifted down to offset 0 for every reader below.
    %scan:3 = scf.while (%i = %c0, %go = %true_scan, %start = %c0) : (index, i1, index) -> (index, i1, index) {
      %in_bounds = arith.cmpi ult, %i, %cap : index
      %continue = arith.andi %in_bounds, %go : i1
      scf.condition(%continue) %i, %go, %start : index, i1, index
    } do {
    ^bb0(%i: index, %go: i1, %start: index):
      %i_i64 = arith.index_cast %i : index to i64
      %slot = llvm.getelementptr %name_ptr[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      %is_nul = arith.cmpi eq, %byte, %zero_i8 : i8
      %true_x = arith.constant true
      %not_nul = arith.xori %is_nul, %true_x : i1
      scf.if %not_nul {
        memref.store %byte, %buffer[%i] : memref<64xi8>
      }
      %next = arith.addi %i, %c1 : index
      %kept = arith.select %is_nul, %i, %next : index
      %is_dot = arith.cmpi eq, %byte, %dot_byte : i8
      %next_start = arith.select %is_dot, %next, %start : index
      scf.yield %kept, %not_nul, %next_start : index, i1, index
    }
    scf.for %i = %scan#2 to %scan#0 step %c1 {
      %byte = memref.load %buffer[%i] : memref<64xi8>
      %at = arith.subi %i, %scan#2 : index
      memref.store %byte, %buffer[%at] : memref<64xi8>
    }
    %leaf_len = arith.subi %scan#0, %scan#2 : index
    %name_len = arith.index_cast %leaf_len : index to i64
    %name_dyn = memref.cast %buffer : memref<64xi8> to memref<?xi8>
    // ExceptionGroup carrying members renders CPython's two-argument form:
    // Name('msg', [member reprs...]); every other exception (and a group
    // whose member block is absent) keeps the message-only form.
    %zero_i64 = arith.constant 0 : i64
    %payload_slot = arith.constant 3 : i64
    %group_root = arith.constant 101 : i64
    %block = func.call @__ly_exc_ext_get(%header, %payload_slot) : (memref<3xi64>, i64) -> i64
    %count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %is_group = func.call @LyEH_ClassIdMatches(%class_id, %group_root) : (i64, i64) -> i1
    %has_members = arith.cmpi sgt, %count, %zero_i64 : i64
    %grouped = arith.andi %is_group, %has_members : i1
    %result:2 = scf.if %grouped -> (memref<2xi64>, memref<?xi8>) {
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %name_h, %name_b = func.call @__ly_unicode_from_valid_utf8(%name_dyn, %c0, %name_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %lparen_ref = memref.get_global @__ly_exc_lparen : memref<1xi8>
      %lparen_dyn = memref.cast %lparen_ref : memref<1xi8> to memref<?xi8>
      %lp_h, %lp_b = func.call @__ly_unicode_from_valid_utf8(%lparen_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %head_h, %head_b = func.call @LyUnicode_Concat(%name_h, %name_b, %lp_h, %lp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%name_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%lp_h) : (memref<2xi64>) -> ()
      %msg_h, %msg_b = func.call @LyUnicode_Repr(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      %a_h, %a_b = func.call @LyUnicode_Concat(%head_h, %head_b, %msg_h, %msg_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%head_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%msg_h) : (memref<2xi64>) -> ()
      %comma_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
      %comma_dyn = memref.cast %comma_ref : memref<2xi8> to memref<?xi8>
      %sep0_h, %sep0_b = func.call @__ly_unicode_from_valid_utf8(%comma_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %b_h, %b_b = func.call @LyUnicode_Concat(%a_h, %a_b, %sep0_h, %sep0_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%a_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%sep0_h) : (memref<2xi64>) -> ()
      // The synthesized naked-exception wrap repr-renders its members as the
      // tuple CPython passes: ('', (exc,)); source-constructed groups keep
      // the list form.
      %tuple_style = func.call @__ly_exc_payload_tuple_flag(%block) : (i64) -> i1
      %open_ref = memref.get_global @__ly_repr_lbracket : memref<1xi8>
      %open_paren_ref = memref.get_global @__ly_repr_lparen : memref<1xi8>
      %open_pick = arith.select %tuple_style, %open_paren_ref, %open_ref : memref<1xi8>
      %open_dyn = memref.cast %open_pick : memref<1xi8> to memref<?xi8>
      %ob_h, %ob_b = func.call @__ly_unicode_from_valid_utf8(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %c_h, %c_b = func.call @LyUnicode_Concat(%b_h, %b_b, %ob_h, %ob_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%b_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%ob_h) : (memref<2xi64>) -> ()
      %count_index = arith.index_cast %count : i64 to index
      %loop:2 = scf.for %i = %c0 to %count_index step %c1 iter_args(%rh = %c_h, %rb = %c_b) -> (memref<2xi64>, memref<?xi8>) {
        %i_i64 = arith.index_cast %i : index to i64
        %is_pos = arith.cmpi sgt, %i_i64, %zero_i64 : i64
        %sep:2 = scf.if %is_pos -> (memref<2xi64>, memref<?xi8>) {
          %sh, %sb = func.call @__ly_unicode_from_valid_utf8(%comma_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
          %jh, %jb = func.call @LyUnicode_Concat(%rh, %rb, %sh, %sb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          func.call @LyUnicode_DecRef(%rh) : (memref<2xi64>) -> ()
          func.call @LyUnicode_DecRef(%sh) : (memref<2xi64>) -> ()
          scf.yield %jh, %jb : memref<2xi64>, memref<?xi8>
        } else {
          scf.yield %rh, %rb : memref<2xi64>, memref<?xi8>
        }
        %member:3 = func.call @__ly_exc_payload_view(%block, %i_i64) : (i64, i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
        %mr_h, %mr_b = func.call @__ly_exception_repr_by_id(%member#0, %member#1, %member#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        %nh, %nb = func.call @LyUnicode_Concat(%sep#0, %sep#1, %mr_h, %mr_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%sep#0) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%mr_h) : (memref<2xi64>) -> ()
        scf.yield %nh, %nb : memref<2xi64>, memref<?xi8>
      }
      // Tuple-style single member keeps CPython's trailing comma: (exc,).
      %is_single = arith.cmpi eq, %count, %c1_i64 : i64
      %needs_comma = arith.andi %tuple_style, %is_single : i1
      %joined:2 = scf.if %needs_comma -> (memref<2xi64>, memref<?xi8>) {
        %comma1_h, %comma1_b = func.call @__ly_unicode_from_valid_utf8(%comma_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %jc_h, %jc_b = func.call @LyUnicode_Concat(%loop#0, %loop#1, %comma1_h, %comma1_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%comma1_h) : (memref<2xi64>) -> ()
        scf.yield %jc_h, %jc_b : memref<2xi64>, memref<?xi8>
      } else {
        %kept:2 = func.call @__ly_unicode_retain_self(%loop#0, %loop#1) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %kept#0, %kept#1 : memref<2xi64>, memref<?xi8>
      }
      func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
      %close_ref = memref.get_global @__ly_repr_rbracket : memref<1xi8>
      %close_paren_ref = memref.get_global @__ly_repr_rparen : memref<1xi8>
      %close_pick = arith.select %tuple_style, %close_paren_ref, %close_ref : memref<1xi8>
      %close_dyn = memref.cast %close_pick : memref<1xi8> to memref<?xi8>
      %cb_h, %cb_b = func.call @__ly_unicode_from_valid_utf8(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %d_h, %d_b = func.call @LyUnicode_Concat(%joined#0, %joined#1, %cb_h, %cb_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%joined#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%cb_h) : (memref<2xi64>) -> ()
      %rparen_ref = memref.get_global @__ly_exc_rparen : memref<1xi8>
      %rparen_dyn = memref.cast %rparen_ref : memref<1xi8> to memref<?xi8>
      %rp_h, %rp_b = func.call @__ly_unicode_from_valid_utf8(%rparen_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %e_h, %e_b = func.call @LyUnicode_Concat(%d_h, %d_b, %rp_h, %rp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%d_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%rp_h) : (memref<2xi64>) -> ()
      scf.yield %e_h, %e_b : memref<2xi64>, memref<?xi8>
    } else {
      // Multi-value args (non-group payload): Name(r0, r1, ...) from the
      // boxed values; the message lane already carries str(e)'s "(r0, r1)".
      %args_repr = arith.cmpi sgt, %count, %zero_i64 : i64
      %preplain:2 = scf.if %args_repr -> (memref<2xi64>, memref<?xi8>) {
        %ac1 = arith.constant 1 : i64
        %ac2 = arith.constant 2 : i64
        %an_h, %an_b = func.call @__ly_unicode_from_valid_utf8(%name_dyn, %c0, %name_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %alp_ref = memref.get_global @__ly_exc_lparen : memref<1xi8>
        %alp_dyn = memref.cast %alp_ref : memref<1xi8> to memref<?xi8>
        %alp_h, %alp_b = func.call @__ly_unicode_from_valid_utf8(%alp_dyn, %c0, %ac1) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %ah_h, %ah_b = func.call @LyUnicode_Concat(%an_h, %an_b, %alp_h, %alp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%an_h) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%alp_h) : (memref<2xi64>) -> ()
        %acomma_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
        %acomma_dyn = memref.cast %acomma_ref : memref<2xi8> to memref<?xi8>
        %ablock_ptr = llvm.inttoptr %block : i64 to !llvm.ptr
        %acount_index = arith.index_cast %count : i64 to index
        %aloop:2 = scf.for %ai = %c0 to %acount_index step %c1 iter_args(%arh = %ah_h, %arb = %ah_b) -> (memref<2xi64>, memref<?xi8>) {
          %ai_i64 = arith.index_cast %ai : index to i64
          %ais_pos = arith.cmpi sgt, %ai_i64, %zero_i64 : i64
          %asep:2 = scf.if %ais_pos -> (memref<2xi64>, memref<?xi8>) {
            %ash, %asb = func.call @__ly_unicode_from_valid_utf8(%acomma_dyn, %c0, %ac2) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
            %ajh, %ajb = func.call @LyUnicode_Concat(%arh, %arb, %ash, %asb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
            func.call @LyUnicode_DecRef(%arh) : (memref<2xi64>) -> ()
            func.call @LyUnicode_DecRef(%ash) : (memref<2xi64>) -> ()
            scf.yield %ajh, %ajb : memref<2xi64>, memref<?xi8>
          } else {
            scf.yield %arh, %arb : memref<2xi64>, memref<?xi8>
          }
          %asixteen = func.call @__ly_box_word_count() : () -> i64
          %abox_off = arith.muli %ai_i64, %asixteen : i64
          %abox_base = arith.addi %abox_off, %ac1 : i64
          %abox_ptr = llvm.getelementptr %ablock_ptr[%abox_base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %aclass_word = llvm.load %abox_ptr : !llvm.ptr -> i64
          %aclass_id = func.call @__ly_slot_class(%aclass_word) : (i64) -> i64
          %aer_h, %aer_b, %aok = func.call @__ly_repr_boxed_by_contract(%abox_ptr, %aclass_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
          cf.assert %aok, "exception repr: boxed arg has no conforming __repr__"
          %anh, %anb = func.call @LyUnicode_Concat(%asep#0, %asep#1, %aer_h, %aer_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
          func.call @LyUnicode_DecRef(%asep#0) : (memref<2xi64>) -> ()
          func.call @LyUnicode_DecRef(%aer_h) : (memref<2xi64>) -> ()
          scf.yield %anh, %anb : memref<2xi64>, memref<?xi8>
        }
        %arp_ref = memref.get_global @__ly_exc_rparen : memref<1xi8>
        %arp_dyn = memref.cast %arp_ref : memref<1xi8> to memref<?xi8>
        %arp_h, %arp_b = func.call @__ly_unicode_from_valid_utf8(%arp_dyn, %c0, %ac1) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %aout_h, %aout_b = func.call @LyUnicode_Concat(%aloop#0, %aloop#1, %arp_h, %arp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%aloop#0) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%arp_h) : (memref<2xi64>) -> ()
        scf.yield %aout_h, %aout_b : memref<2xi64>, memref<?xi8>
      } else {
        %fallback:2 = func.call @__ly_exception_repr(%message_header, %message_bytes, %name_dyn, %name_len) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %fallback#0, %fallback#1 : memref<2xi64>, memref<?xi8>
      }
      scf.yield %preplain#0, %preplain#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // CPython repr(e): ClassName(<repr of the message>) -- ClassName() when
  // the message is empty. A no-argument construction and an explicit empty
  // string are indistinguishable in the 3-word payload, so the empty case
  // renders as ClassName() (the CPython-visible difference is only
  // ClassName('')).
  memref.global "private" constant @__ly_exc_lparen : memref<1xi8> = dense<40>
  memref.global "private" constant @__ly_exc_rparen : memref<1xi8> = dense<41>
  func.func private @__ly_exception_repr(%message_header: memref<2xi64>, %message_bytes: memref<?xi8>, %name: memref<?xi8>, %name_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %name_h, %name_b = func.call @__ly_unicode_from_valid_utf8(%name, %c0, %name_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %lparen_ref = memref.get_global @__ly_exc_lparen : memref<1xi8>
    %lparen_dyn = memref.cast %lparen_ref : memref<1xi8> to memref<?xi8>
    %lp_h, %lp_b = func.call @__ly_unicode_from_valid_utf8(%lparen_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %rparen_ref = memref.get_global @__ly_exc_rparen : memref<1xi8>
    %rparen_dyn = memref.cast %rparen_ref : memref<1xi8> to memref<?xi8>
    %rp_h, %rp_b = func.call @__ly_unicode_from_valid_utf8(%rparen_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %head_h, %head_b = func.call @LyUnicode_Concat(%name_h, %name_b, %lp_h, %lp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%name_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%lp_h) : (memref<2xi64>) -> ()
    %msg_len = func.call @__ly_unicode_count(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %has_message = arith.cmpi sgt, %msg_len, %zero : i64
    // Both branches yield a fresh owned str so the releases stay outside the
    // regions (region-crossing ownership of the head would trip the
    // affine-ownership verifier).
    %quoted:2 = scf.if %has_message -> (memref<2xi64>, memref<?xi8>) {
      %repr_h, %repr_b = func.call @LyUnicode_Repr(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %repr_h, %repr_b : memref<2xi64>, memref<?xi8>
    } else {
      %one_w = arith.constant 1 : i64
      %empty_h, %empty_b = func.call @__ly_unicode_alloc(%zero, %one_w) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %empty_h, %empty_b : memref<2xi64>, memref<?xi8>
    }
    %body_h, %body_b = func.call @LyUnicode_Concat(%head_h, %head_b, %quoted#0, %quoted#1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%head_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%quoted#0) : (memref<2xi64>) -> ()
    %out_h, %out_b = func.call @LyUnicode_Concat(%body_h, %body_b, %rp_h, %rp_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%body_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%rp_h) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  // BaseException.args: the boxed payload values when the exception was
  // constructed with several arguments; otherwise () for an empty message
  // and (message,) for the single-argument fast path. (Deviation, noted:
  // an exception GROUP renders args as (message,) -- the second CPython
  // element, the original member sequence object, is not retained.)
  func.func @LyBaseException_Args(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "args", ly.runtime.result_contract = "builtins.tuple", ly.runtime.element_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %class_slot = arith.constant 2 : index
    %payload_slot = arith.constant 3 : i64
    %group_root = arith.constant 101 : i64
    %class_id = memref.load %header[%class_slot] : memref<3xi64>
    %block = func.call @__ly_exc_ext_get(%header, %payload_slot) : (memref<3xi64>, i64) -> i64
    %payload_count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %is_group = func.call @LyEH_ClassIdMatches(%class_id, %group_root) : (i64, i64) -> i1
    %true_c = arith.constant true
    %not_group = arith.xori %is_group, %true_c : i1
    %has_payload = arith.cmpi sgt, %payload_count, %zero : i64
    %payload_args = arith.andi %has_payload, %not_group : i1
    %result = scf.if %payload_args -> memref<5xi64> {
      %tuple = func.call @LyTuple_FromLength(%payload_count) : (i64) -> memref<5xi64>
      %tuple_items = func.call @__ly_tuple_items(%tuple) : (memref<5xi64>) -> memref<?xi64>
      %block_ptr = llvm.inttoptr %block : i64 to !llvm.ptr
      %count_index = arith.index_cast %payload_count : i64 to index
      scf.for %i = %c0 to %count_index step %c1 {
        %i64v = arith.index_cast %i : index to i64
        %sixteen = func.call @__ly_box_word_count() : () -> i64
        %one_off = arith.constant 1 : i64
        %box_words = arith.muli %i64v, %sixteen : i64
        %box_base = arith.addi %box_words, %one_off : i64
        %dst_base = arith.muli %i, %c16 : index
        scf.for %w = %c0 to %c16 step %c1 {
          %w64 = arith.index_cast %w : index to i64
          %src_off = arith.addi %box_base, %w64 : i64
          %src_ptr = llvm.getelementptr %block_ptr[%src_off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %word = llvm.load %src_ptr : !llvm.ptr -> i64
          %dst = arith.addi %dst_base, %w : index
          memref.store %word, %tuple_items[%dst] : memref<?xi64>
        }
        func.call @LyObject_RetainBoxedPayloadArraySlotRaw(%tuple_items, %i64v) : (memref<?xi64>, i64) -> ()
      }
      scf.yield %tuple : memref<5xi64>
    } else {
      %msg_len = func.call @__ly_unicode_count(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
      %has_message = arith.cmpi sgt, %msg_len, %zero : i64
      %count = arith.select %has_message, %one, %zero : i64
      %tuple = func.call @LyTuple_FromLength(%count) : (i64) -> memref<5xi64>
      %tuple_items = func.call @__ly_tuple_items(%tuple) : (memref<5xi64>) -> memref<?xi64>
      scf.if %has_message {
        %retained:2 = func.call @__ly_unicode_retain_self(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @__ly_unicode_store_item(%tuple_items, %zero, %retained#0, %retained#1) : (memref<?xi64>, i64, memref<2xi64>, memref<?xi8>) -> ()
      }
      scf.yield %tuple : memref<5xi64>
    }
    func.return %result : memref<5xi64>
  }

  func.func @LyKeyboardInterrupt_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 100 : i64, ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyKeyboardInterrupt_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyKeyboardInterrupt_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyboardInterrupt_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseExceptionGroup_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 101 : i64, ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBaseExceptionGroup_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // "msg (N sub-exception[s])" -- CPython's BaseExceptionGroup.__str__. A
  // group without a member block (never constructible from source, but the
  // dead-placeholder lanes can reach str) falls back to the plain message.
  memref.global "private" constant @__ly_excgroup_str_open : memref<2xi8> = dense<[32, 40]>
  memref.global "private" constant @__ly_excgroup_str_word : memref<14xi8> = dense<[32, 115, 117, 98, 45, 101, 120, 99, 101, 112, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_excgroup_str_plural : memref<1xi8> = dense<115>
  memref.global "private" constant @__ly_excgroup_str_close : memref<1xi8> = dense<41>

  // ASCII decimal rendering of a non-negative count (fits any payload size).
  func.func private @__ly_exc_count_str(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "count_str", ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one_w = arith.constant 1 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buffer = memref.alloca() : memref<24xi8>
    %loop:2 = scf.while (%v = %value, %i = %zero) : (i64, i64) -> (i64, i64) {
      %first = arith.cmpi eq, %i, %zero : i64
      %more = arith.cmpi sgt, %v, %zero : i64
      %continue = arith.ori %first, %more : i1
      scf.condition(%continue) %v, %i : i64, i64
    } do {
    ^bb0(%v: i64, %i: i64):
      %digit = arith.remsi %v, %ten : i64
      %char64 = arith.addi %digit, %ascii_zero : i64
      %char = arith.trunci %char64 : i64 to i8
      %slot = arith.index_cast %i : i64 to index
      memref.store %char, %buffer[%slot] : memref<24xi8>
      %next_v = arith.divsi %v, %ten : i64
      %next_i = arith.addi %i, %one_w : i64
      scf.yield %next_v, %next_i : i64, i64
    }
    %out_h, %out_b = func.call @__ly_unicode_alloc(%loop#1, %one_w) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %len_index = arith.index_cast %loop#1 : i64 to index
    scf.for %k = %c0 to %len_index step %c1 {
      %k64 = arith.index_cast %k : index to i64
      %minus_one = arith.constant 1 : i64
      %rev64 = arith.subi %loop#1, %k64 : i64
      %rev64m = arith.subi %rev64, %minus_one : i64
      %rev = arith.index_cast %rev64m : i64 to index
      %byte = memref.load %buffer[%rev] : memref<24xi8>
      %cp = arith.extui %byte : i8 to i64
      func.call @__ly_unicode_put(%out_b, %one_w, %k, %cp) : (memref<?xi8>, i64, index, i64) -> ()
    }
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  func.func private @__ly_exception_group_str(%header: memref<3xi64>, %message_header: memref<2xi64>, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "group_str", ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %fourteen = arith.constant 14 : i64
    %three_slot = arith.constant 3 : i64
    %c0 = arith.constant 0 : index
    %block = func.call @__ly_exc_ext_get(%header, %three_slot) : (memref<3xi64>, i64) -> i64
    %count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %empty = arith.cmpi eq, %count, %zero : i64
    %result:2 = scf.if %empty -> (memref<2xi64>, memref<?xi8>) {
      %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
      func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
      scf.yield %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
    } else {
      %open_ref = memref.get_global @__ly_excgroup_str_open : memref<2xi8>
      %open_dyn = memref.cast %open_ref : memref<2xi8> to memref<?xi8>
      %open_h, %open_b = func.call @__ly_unicode_from_valid_utf8(%open_dyn, %c0, %two) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %count_h, %count_b = func.call @__ly_exc_count_str(%count) : (i64) -> (memref<2xi64>, memref<?xi8>)
      %word_ref = memref.get_global @__ly_excgroup_str_word : memref<14xi8>
      %word_dyn = memref.cast %word_ref : memref<14xi8> to memref<?xi8>
      %word_h, %word_b = func.call @__ly_unicode_from_valid_utf8(%word_dyn, %c0, %fourteen) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %a_h, %a_b = func.call @LyUnicode_Concat(%message_header, %message_bytes, %open_h, %open_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%open_h) : (memref<2xi64>) -> ()
      %b_h, %b_b = func.call @LyUnicode_Concat(%a_h, %a_b, %count_h, %count_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%a_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%count_h) : (memref<2xi64>) -> ()
      %c_h, %c_b = func.call @LyUnicode_Concat(%b_h, %b_b, %word_h, %word_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%b_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%word_h) : (memref<2xi64>) -> ()
      %is_plural = arith.cmpi sgt, %count, %one : i64
      %d:2 = scf.if %is_plural -> (memref<2xi64>, memref<?xi8>) {
        %s_ref = memref.get_global @__ly_excgroup_str_plural : memref<1xi8>
        %s_dyn = memref.cast %s_ref : memref<1xi8> to memref<?xi8>
        %s_h, %s_b = func.call @__ly_unicode_from_valid_utf8(%s_dyn, %c0, %one) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %p_h, %p_b = func.call @LyUnicode_Concat(%c_h, %c_b, %s_h, %s_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%s_h) : (memref<2xi64>) -> ()
        scf.yield %p_h, %p_b : memref<2xi64>, memref<?xi8>
      } else {
        %retained:2 = func.call @__ly_unicode_retain_self(%c_h, %c_b) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        scf.yield %retained#0, %retained#1 : memref<2xi64>, memref<?xi8>
      }
      func.call @LyUnicode_DecRef(%c_h) : (memref<2xi64>) -> ()
      %close_ref = memref.get_global @__ly_excgroup_str_close : memref<1xi8>
      %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
      %close_h, %close_b = func.call @__ly_unicode_from_valid_utf8(%close_dyn, %c0, %one) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %e_h, %e_b = func.call @LyUnicode_Concat(%d#0, %d#1, %close_h, %close_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%d#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%close_h) : (memref<2xi64>) -> ()
      scf.yield %e_h, %e_b : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseExceptionGroup_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_group_str(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseExceptionGroup_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyExceptionGroup_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 102 : i64, ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyExceptionGroup_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyExceptionGroup_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_group_str(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyExceptionGroup_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // ExceptionGroup member block: allocated by the 2-argument construction
  // lowering (the exception's extended word 3), one member stored per call.
  // The store borrows the member triple and retains the entity for the
  // block; the block itself is released by release_exception_extras.
  func.func @LyBaseExceptionGroup_MembersAlloc(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>, %count: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.primitive = "members_alloc"} {
    %block = func.call @__ly_exc_payload_alloc(%count) : (i64) -> i64
    %payload_slot = arith.constant 3 : i64
    func.call @__ly_exc_ext_set(%header, %payload_slot, %block) : (memref<3xi64>, i64, i64) -> ()
    func.return %block : i64
  }

  func.func @LyBaseExceptionGroup_MemberStore(%block: i64, %slot: i64, %eh: memref<3xi64> {ly.ownership.object_header}, %mh: memref<2xi64> {ly.ownership.object_header}, %mb: memref<?xi8>) attributes {ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.primitive = "member_store"} {
    func.call @__ly_exc_payload_store(%block, %slot, %eh, %mh, %mb) : (i64, i64, memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // BaseExceptionGroup.message: the bare message (str() adds the count).
  func.func @LyBaseExceptionGroup_Message(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.primitive = "message", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  // BaseExceptionGroup.exceptions: a fresh tuple whose boxes duplicate the
  // member block (each copy retains the member entity).
  func.func @LyBaseExceptionGroup_Exceptions(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.primitive = "exceptions", ly.runtime.result_contract = "builtins.tuple", ly.runtime.element_contract = "builtins.BaseException"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %one = arith.constant 1 : i64
    %payload_slot = arith.constant 3 : i64
    %block = func.call @__ly_exc_ext_get(%header, %payload_slot) : (memref<3xi64>, i64) -> i64
    %count = func.call @__ly_exc_payload_count(%block) : (i64) -> i64
    %tuple = func.call @LyTuple_FromLength(%count) : (i64) -> memref<5xi64>
    %tuple_items = func.call @__ly_tuple_items(%tuple) : (memref<5xi64>) -> memref<?xi64>
    %block_ptr = llvm.inttoptr %block : i64 to !llvm.ptr
    %count_index = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %count_index step %c1 {
      %i64v = arith.index_cast %i : index to i64
      %c16_i64 = func.call @__ly_box_word_count() : () -> i64
      %one_off = arith.constant 1 : i64
      %box_words = arith.muli %i64v, %c16_i64 : i64
      %box_base = arith.addi %box_words, %one_off : i64
      %dst_base = arith.muli %i, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %w64 = arith.index_cast %w : index to i64
        %src_off = arith.addi %box_base, %w64 : i64
        %src_ptr = llvm.getelementptr %block_ptr[%src_off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %word = llvm.load %src_ptr : !llvm.ptr -> i64
        %dst = arith.addi %dst_base, %w : index
        memref.store %word, %tuple_items[%dst] : memref<?xi64>
      }
      func.call @LyObject_RetainBoxedPayloadArraySlotRaw(%tuple_items, %i64v) : (memref<?xi64>, i64) -> ()
    }
    func.return %tuple : memref<5xi64>
  }

  func.func @LyFloatingPointError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 103 : i64, ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyFloatingPointError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyFloatingPointError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFloatingPointError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOverflowError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 104 : i64, ly.runtime.contract = "builtins.OverflowError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyOverflowError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.OverflowError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyOverflowError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OverflowError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOverflowError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OverflowError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBufferError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 105 : i64, ly.runtime.contract = "builtins.BufferError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBufferError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BufferError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBufferError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BufferError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBufferError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BufferError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyEOFError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 106 : i64, ly.runtime.contract = "builtins.EOFError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyEOFError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.EOFError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyEOFError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EOFError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyEOFError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EOFError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyImportError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 107 : i64, ly.runtime.contract = "builtins.ImportError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyImportError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ImportError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyImportError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyImportError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyModuleNotFoundError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 108 : i64, ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyModuleNotFoundError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyModuleNotFoundError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyModuleNotFoundError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyMemoryError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 109 : i64, ly.runtime.contract = "builtins.MemoryError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyMemoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.MemoryError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyMemoryError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.MemoryError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyMemoryError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.MemoryError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNameError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 110 : i64, ly.runtime.contract = "builtins.NameError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyNameError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.NameError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyNameError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NameError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNameError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NameError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnboundLocalError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 111 : i64, ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnboundLocalError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnboundLocalError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnboundLocalError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAttributeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 112 : i64, ly.runtime.contract = "builtins.AttributeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyAttributeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.AttributeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyAttributeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AttributeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAttributeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AttributeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyReferenceError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 113 : i64, ly.runtime.contract = "builtins.ReferenceError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyReferenceError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyReferenceError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyReferenceError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNotImplementedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 114 : i64, ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyNotImplementedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyNotImplementedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNotImplementedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRecursionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 115 : i64, ly.runtime.contract = "builtins.RecursionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyRecursionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.RecursionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyRecursionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RecursionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRecursionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RecursionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPythonFinalizationError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 116 : i64, ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyPythonFinalizationError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyPythonFinalizationError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPythonFinalizationError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LySyntaxError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 117 : i64, ly.runtime.contract = "builtins.SyntaxError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LySyntaxError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LySyntaxError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySyntaxError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndentationError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 118 : i64, ly.runtime.contract = "builtins.IndentationError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyIndentationError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.IndentationError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyIndentationError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndentationError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndentationError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndentationError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTabError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 119 : i64, ly.runtime.contract = "builtins.TabError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyTabError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.TabError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyTabError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TabError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTabError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TabError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 120 : i64, ly.runtime.contract = "builtins.SystemError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LySystemError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SystemError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LySystemError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 121 : i64, ly.runtime.contract = "builtins.UnicodeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeDecodeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 122 : i64, ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeDecodeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeDecodeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeDecodeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeEncodeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 123 : i64, ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeEncodeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeEncodeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeEncodeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeTranslateError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 124 : i64, ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeTranslateError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeTranslateError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeTranslateError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 125 : i64, ly.runtime.contract = "builtins.Warning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.Warning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Warning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Warning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytesWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 126 : i64, ly.runtime.contract = "builtins.BytesWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBytesWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBytesWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytesWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyDeprecationWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 127 : i64, ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyDeprecationWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyDeprecationWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyDeprecationWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyEncodingWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 128 : i64, ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyEncodingWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyEncodingWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyEncodingWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFutureWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 129 : i64, ly.runtime.contract = "builtins.FutureWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyFutureWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyFutureWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFutureWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyImportWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 130 : i64, ly.runtime.contract = "builtins.ImportWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyImportWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyImportWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyImportWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPendingDeprecationWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 131 : i64, ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyPendingDeprecationWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyPendingDeprecationWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPendingDeprecationWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyResourceWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 132 : i64, ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyResourceWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyResourceWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyResourceWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 133 : i64, ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyRuntimeWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyRuntimeWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LySyntaxWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 134 : i64, ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LySyntaxWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LySyntaxWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySyntaxWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 135 : i64, ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnicodeWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnicodeWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUserWarning_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 136 : i64, ly.runtime.contract = "builtins.UserWarning", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUserWarning_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.UserWarning", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUserWarning_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UserWarning", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUserWarning_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.UserWarning", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBlockingIOError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 137 : i64, ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBlockingIOError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBlockingIOError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBlockingIOError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyChildProcessError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 138 : i64, ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyChildProcessError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyChildProcessError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyChildProcessError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 139 : i64, ly.runtime.contract = "builtins.ConnectionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBrokenPipeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 140 : i64, ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyBrokenPipeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyBrokenPipeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBrokenPipeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionAbortedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 141 : i64, ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionAbortedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionAbortedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionAbortedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionRefusedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 142 : i64, ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionRefusedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionRefusedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionRefusedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionResetError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 143 : i64, ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyConnectionResetError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyConnectionResetError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyConnectionResetError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileExistsError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 144 : i64, ly.runtime.contract = "builtins.FileExistsError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyFileExistsError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyFileExistsError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileExistsError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyInterruptedError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 145 : i64, ly.runtime.contract = "builtins.InterruptedError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyInterruptedError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyInterruptedError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyInterruptedError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIsADirectoryError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 146 : i64, ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyIsADirectoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyIsADirectoryError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIsADirectoryError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNotADirectoryError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 147 : i64, ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyNotADirectoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyNotADirectoryError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyNotADirectoryError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPermissionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 148 : i64, ly.runtime.contract = "builtins.PermissionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyPermissionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.PermissionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyPermissionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PermissionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPermissionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.PermissionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyProcessLookupError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 149 : i64, ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyProcessLookupError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyProcessLookupError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyProcessLookupError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTimeoutError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 150 : i64, ly.runtime.contract = "builtins.TimeoutError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyTimeoutError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyTimeoutError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTimeoutError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyboardInterrupt_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.KeyboardInterrupt", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBaseExceptionGroup_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseExceptionGroup", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyExceptionGroup_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ExceptionGroup", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyFloatingPointError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FloatingPointError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyOverflowError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.OverflowError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBufferError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BufferError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyEOFError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.EOFError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyImportError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ImportError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyModuleNotFoundError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ModuleNotFoundError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyMemoryError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.MemoryError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyNameError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.NameError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnboundLocalError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnboundLocalError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyAttributeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.AttributeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyReferenceError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ReferenceError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyNotImplementedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.NotImplementedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyRecursionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.RecursionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyPythonFinalizationError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.PythonFinalizationError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LySyntaxError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SyntaxError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyIndentationError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.IndentationError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyTabError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.TabError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LySystemError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SystemError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeDecodeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeDecodeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeEncodeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeEncodeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeTranslateError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeTranslateError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.Warning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBytesWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BytesWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyDeprecationWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.DeprecationWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyEncodingWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.EncodingWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyFutureWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FutureWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyImportWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ImportWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyPendingDeprecationWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.PendingDeprecationWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyResourceWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ResourceWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyRuntimeWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.RuntimeWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LySyntaxWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.SyntaxWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUnicodeWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UnicodeWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyUserWarning_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.UserWarning", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBlockingIOError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BlockingIOError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyChildProcessError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ChildProcessError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyBrokenPipeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BrokenPipeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionAbortedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionAbortedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionRefusedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionRefusedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyConnectionResetError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ConnectionResetError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyFileExistsError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.FileExistsError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyInterruptedError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.InterruptedError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyIsADirectoryError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.IsADirectoryError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyNotADirectoryError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.NotADirectoryError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyPermissionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.PermissionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyProcessLookupError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ProcessLookupError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }
  func.func @LyTimeoutError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.TimeoutError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyException_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.Exception", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyRuntimeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyTypeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.TypeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyValueError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ValueError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyArithmeticError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyLookupError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.LookupError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyZeroDivisionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyKeyError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.KeyError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyIndexError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.IndexError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyAssertionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.AssertionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyStopIteration_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.StopIteration", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyStopAsyncIteration_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LySystemExit_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.SystemExit", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }
  func.func private @LyGeneratorExit_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyOSError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.OSError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyFileNotFoundError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // __str__/__repr__ return the exception message as an owned str.

  // Deviation from CPython: repr(err) also yields the bare message (not

  // "Cls('msg')"), so that print(err) -- which resolves __repr__ -- matches

  // CPython's str-based print output.

  // ⛔ THE MESSAGE COMES FROM THE OBJECT AND NOT FROM THE LANES IT WAS HANDED.
  // The two are the same string, and reading the recorded one here is what
  // keeps the record honest: every `str(e)` in the suite compares it against
  // the lane a producer passed, so a producer that forgot to record fails
  // loudly instead of waiting for a box to read the stale word.
  func.func @LyBaseException_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %recorded_header, %recorded_bytes = func.call @__ly_exc_message_parts(%header) : (memref<3xi64>) -> (memref<2xi64>, memref<?xi8>)
    %view = memref.cast %recorded_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %recorded_header, %recorded_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseException_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LySystemExit_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.SystemExit", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyGeneratorExit_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyGeneratorExit_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.GeneratorExit", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyOSError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.OSError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFileNotFoundError_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.FileNotFoundError", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %result:2 = func.call @__ly_exception_repr_by_id(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }
}
