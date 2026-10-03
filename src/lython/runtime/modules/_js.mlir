// Contract manifest for the JavaScript host's values: what `from js import
// ...` reaches (Pyodide's `js` module).
//
// The program is typed by the `js` stub's classes (runtime/lib/js.pyi); once
// it is, every one of them is `_js.JsProxy` here -- a handle into the host's
// value table, kept alive while a Python reference holds it. Values cross the
// boundary by the STATIC type at each end: a member the stub declares `float`
// is read as a number and anything else raises TypeError, so a stub that
// disagrees with the host is caught where it is read, not believed.
//
// ⛔ No memref crosses into the host: a memref is a descriptor of several
// words, and the host imports (runtime/js/lython_js.js) take addresses and
// lengths as `index`, the target's size_t.
//
// ⛔ Arguments are PUSHED, one host call each, and the call takes them all:
// no array of handles to lay out in wasm memory, and no temporary handle for a
// number or a string that would need dropping after the call.

module attributes {
  ly.runtime.only_with = "ly.js.host",
  ly.runtime.contracts = ["_js.JsProxy"],
  ly.typing.module = "_js",
  ly.typing.class_exports = ["_js.JsProxy=_js.JsProxy"],
  // What runtime/lib/_js_bridge.py keeps callbacks with.
  ly.typing.callable_exports = ["_js.function_for", "_js.callback_slot", "_js.fail_callback", "_js.wait_for_host"],
  ly.typing.function_names = ["_js.function_for", "_js.callback_slot", "_js.fail_callback", "_js.wait_for_host"],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["slot"], arg_defaults = [false], returns = [!py.contract<"_js.JsProxy">]>,
    !py.callable<[], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["message"], arg_defaults = [false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["timeout_ms"], arg_defaults = [false], returns = [!py.contract<"builtins.bool">]>
  ]
} {
  // The internal methods are what a read typed with a union dispatches on
  // (ModuleEmitter::adaptJsHostResult): a test per member, then the
  // conversion to the one that held.
  py.class @JsProxy attributes {
    base_names = ["object"], ly.typing.final,
    ly.runtime.contract = "_js.JsProxy",
    method_names = ["__ly_js_is_none__", "__ly_js_is_bool__", "__ly_js_is_int__",
                    "__ly_js_is_float__", "__ly_js_is_str__", "__ly_js_as_bool__",
                    "__ly_js_as_int__", "__ly_js_as_float__", "__ly_js_as_str__",
                    "__ly_js_as_proxy__", "__ly_js_instanceof__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">] -> [!py.contract<"_js.JsProxy">]>,
      !py.protocol<"Callable", [!py.contract<"_js.JsProxy">, !py.contract<"_js.JsProxy">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  // ===== host imports =====
  // A handle is an i32 index into the host's table; -1 is "the host threw",
  // with the exception parked in the host until LyJs_TakeError.
  func.func private @LyJs_Global(index, index) -> i32
  func.func private @LyJs_Get(i32, index, index) -> i32
  func.func private @LyJs_Set(i32, index, index) -> i32
  func.func private @LyJs_CallMethod(i32, index, index) -> i32
  // `new` on the value with every value pushed since the last call.
  func.func private @LyJs_Construct(i32) -> i32
  func.func private @LyJs_Drop(i32)
  func.func private @LyJs_PushHandle(i32)
  func.func private @LyJs_PushF64(f64)
  func.func private @LyJs_PushI64(i64)
  func.func private @LyJs_PushStr(index, index, index)
  // An ASCII decimal, pushed as a BigInt.
  func.func private @LyJs_PushBigInt(index, index)
  // 1 when the value is what `kind` names (LyJsKind in lython_js.js);
  // otherwise 0 with a TypeError's message parked like an exception.
  func.func private @LyJs_Expect(i32, i32) -> i32
  func.func private @LyJs_IsNullish(i32) -> i32
  // Expect's test without the parked message.
  func.func private @LyJs_Is(i32, i32) -> i32
  // A second handle to the same value.
  func.func private @LyJs_Dup(i32) -> i32
  // `value instanceof constructor`; -1 when the host throws (a constructor
  // that is not callable).
  func.func private @LyJs_InstanceOf(i32, i32) -> i32
  func.func private @LyJs_ToF64(i32) -> f64
  func.func private @LyJs_ToI64(i32) -> i64
  // ⛔ i32 and not `index`: on wasm64 an `index` result is a BigInt the host
  // would have to build, and no host string is 2^31 code points long.
  func.func private @LyJs_StrCount(i32) -> i32
  func.func private @LyJs_StrWidth(i32) -> i32
  func.func private @LyJs_StrWrite(i32, index, index)
  // The parked message as a host string, and clears it.
  func.func private @LyJs_TakeError() -> i32
  // A host function that calls the program back with `slot`; the slot of the
  // callback being run; that callback's exception, as the one value pushed.
  func.func private @LyJs_MakeFunction(i32) -> i32
  func.func private @LyJs_CurrentSlot() -> i32
  func.func private @LyJs_SetCallbackError()
  // Suspends the program until the host has called it back or `ms` have
  // passed (negative: no limit), and says whether it did suspend: 0 where it
  // cannot -- no JSPI, or inside a callback the host is running.
  func.func private @LyJs_WaitForHost(f64) -> i32

  // ===== from builtins =====
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1
  func.func private @__ly_raise_message_object(%class_id: i64, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>)
  func.func private @__ly_unicode_alloc(%count: i64, %width: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64
  func.func private @LyLong_TryAsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> (i64, i1)
  func.func private @LyLong_Repr(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0]}
  func.func private @LyLong_FromI64(%value: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0]}

  // ===== the proxy object =====
  // Words: refcount, class id, handle. Width 17 is this contract's alone
  // (HandleWidthRegistry.h): a release chosen by shape cannot be another
  // contract's, which would drop the object without dropping the handle.
  func.func @LyJsProxy_New(%handle: i32) -> memref<17xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 151 : i64, ly.runtime.contract = "_js.JsProxy", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %class_id = arith.constant 151 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %handle_slot = arith.constant 2 : index
    %proxy = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<17xi64>
    memref.store %one, %proxy[%refcount_slot] : memref<17xi64>
    memref.store %class_id, %proxy[%layout_slot] : memref<17xi64>
    %word = arith.extui %handle : i32 to i64
    memref.store %word, %proxy[%handle_slot] : memref<17xi64>
    func.return %proxy : memref<17xi64>
  }

  func.func @LyJsProxy_DecRef(%proxy: memref<17xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_js.JsProxy", ly.runtime.deallocator} {
    %storage = memref.cast %proxy : memref<17xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %handle = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    func.call @LyJs_Drop(%handle) : (i32) -> ()
    memref.dealloc %proxy : memref<17xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyJsProxy_Handle(%proxy: memref<17xi64> {ly.ownership.object_header}) -> i32 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "handle"} {
    %handle_slot = arith.constant 2 : index
    %word = memref.load %proxy[%handle_slot] : memref<17xi64>
    %handle = arith.trunci %word : i64 to i32
    func.return %handle : i32
  }

  // ===== failure =====
  // Raises `class_id` with the host's parked message.
  func.func private @__ly_js_raise_parked(%class_id: i64) {
    %message = func.call @LyJs_TakeError() : () -> i32
    %header, %bytes = func.call @__ly_js_string(%message) : (i32) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyJs_Drop(%message) : (i32) -> ()
    func.call @__ly_raise_message_object(%class_id, %header, %bytes) : (i64, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // A handle the host returned, or RuntimeError with what the host threw.
  // ⛔ RuntimeError and not Pyodide's JsException: that is a class of its
  // own, and naming it is the program's way to catch exactly the host's
  // exceptions -- not a spelling to fake with another class.
  func.func private @__ly_js_checked(%handle: i32) -> i32 {
    %failed_value = arith.constant -1 : i32
    %failed = arith.cmpi eq, %handle, %failed_value : i32
    scf.if %failed {
      %runtime_error = arith.constant 51 : i64
      func.call @__ly_js_raise_parked(%runtime_error) : (i64) -> ()
    }
    func.return %handle : i32
  }

  // TypeError unless the value is of `kind`; the handle is dropped on the
  // raise, so a conversion that fails leaks nothing.
  func.func private @__ly_js_expect(%handle: i32, %kind: i32) {
    %ok = func.call @LyJs_Expect(%handle, %kind) : (i32, i32) -> i32
    %zero = arith.constant 0 : i32
    %bad = arith.cmpi eq, %ok, %zero : i32
    scf.if %bad {
      func.call @LyJs_Drop(%handle) : (i32) -> ()
      %type_error = arith.constant 52 : i64
      func.call @__ly_js_raise_parked(%type_error) : (i64) -> ()
    }
    func.return
  }

  func.func private @__ly_js_address(%bytes: memref<?xi8>) -> index {
    %address = memref.extract_aligned_pointer_as_index %bytes : memref<?xi8> -> index
    func.return %address : index
  }

  // ===== member access =====
  // ⛔ The receiver is the PROXY, not its handle: a handle read out of it is
  // an i32 the ownership pass cannot see, so the proxy's last use would be
  // that read and its release -- which drops the handle -- would land before
  // the host call that needs it.
  func.func @LyJsProxy_Global(%name: memref<?xi8>, %length: i64) -> i32 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "global"} {
    %address = func.call @__ly_js_address(%name) : (memref<?xi8>) -> index
    %count = arith.index_cast %length : i64 to index
    %raw = func.call @LyJs_Global(%address, %count) : (index, index) -> i32
    %handle = func.call @__ly_js_checked(%raw) : (i32) -> i32
    func.return %handle : i32
  }

  func.func @LyJsProxy_Get(%proxy: memref<17xi64> {ly.ownership.object_header}, %name: memref<?xi8>, %length: i64) -> i32 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "get"} {
    %object = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    %address = func.call @__ly_js_address(%name) : (memref<?xi8>) -> index
    %count = arith.index_cast %length : i64 to index
    %raw = func.call @LyJs_Get(%object, %address, %count) : (i32, index, index) -> i32
    %handle = func.call @__ly_js_checked(%raw) : (i32) -> i32
    func.return %handle : i32
  }

  // Sets the member to the one value pushed before it.
  func.func @LyJsProxy_Set(%proxy: memref<17xi64> {ly.ownership.object_header}, %name: memref<?xi8>, %length: i64) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "set"} {
    %object = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    %address = func.call @__ly_js_address(%name) : (memref<?xi8>) -> index
    %count = arith.index_cast %length : i64 to index
    %raw = func.call @LyJs_Set(%object, %address, %count) : (i32, index, index) -> i32
    %checked = func.call @__ly_js_checked(%raw) : (i32) -> i32
    func.return
  }

  // Calls the member with every value pushed since the last call.
  func.func @LyJsProxy_CallMethod(%proxy: memref<17xi64> {ly.ownership.object_header}, %name: memref<?xi8>, %length: i64) -> i32 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "call_method"} {
    %object = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    %address = func.call @__ly_js_address(%name) : (memref<?xi8>) -> index
    %count = arith.index_cast %length : i64 to index
    %raw = func.call @LyJs_CallMethod(%object, %address, %count) : (i32, index, index) -> i32
    %handle = func.call @__ly_js_checked(%raw) : (i32) -> i32
    func.return %handle : i32
  }

  func.func @LyJsProxy_Construct(%proxy: memref<17xi64> {ly.ownership.object_header}) -> i32 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "construct"} {
    %object = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    %raw = func.call @LyJs_Construct(%object) : (i32) -> i32
    %handle = func.call @__ly_js_checked(%raw) : (i32) -> i32
    func.return %handle : i32
  }

  // ===== Python -> host =====
  // The host's stack holds the value itself, so the proxy may go once this
  // returns.
  func.func @LyJsProxy_PushProxy(%proxy: memref<17xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "push.proxy"} {
    %handle = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    func.call @LyJs_PushHandle(%handle) : (i32) -> ()
    func.return
  }

  func.func @LyJsProxy_PushF64(%value: f64) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "push.f64"} {
    func.call @LyJs_PushF64(%value) : (f64) -> ()
    func.return
  }

  // Pyodide's rule: a number when a double holds it exactly, a BigInt
  // otherwise -- an int past 64 bits as its decimal digits, so no width of
  // the crossing bounds it.
  func.func @LyJsProxy_PushInt(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "push.int"} {
    %value, %fits = func.call @LyLong_TryAsI64(%header) : (memref<2xi64>) -> (i64, i1)
    scf.if %fits {
      func.call @LyJs_PushI64(%value) : (i64) -> ()
    } else {
      %digits_header, %digits = func.call @LyLong_Repr(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi8>)
      %c0 = arith.constant 0 : index
      %count = memref.dim %digits, %c0 : memref<?xi8>
      %address = func.call @__ly_js_address(%digits) : (memref<?xi8>) -> index
      func.call @LyJs_PushBigInt(%address, %count) : (index, index) -> ()
      func.call @LyUnicode_DecRef(%digits_header) : (memref<2xi64>) -> ()
    }
    func.return
  }

  // The host's fixed handles: 0 undefined, 2 true, 3 false.
  func.func @LyJsProxy_PushBool(%value: i1) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "push.bool"} {
    %true_handle = arith.constant 2 : i32
    %false_handle = arith.constant 3 : i32
    %handle = arith.select %value, %true_handle, %false_handle : i32
    func.call @LyJs_PushHandle(%handle) : (i32) -> ()
    func.return
  }

  // None is undefined, as Pyodide sends it.
  func.func @LyJsProxy_PushNone() attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "push.none"} {
    %undefined = arith.constant 0 : i32
    func.call @LyJs_PushHandle(%undefined) : (i32) -> ()
    func.return
  }

  // The str's code units as stored (1, 2 or 4 bytes each): the host builds
  // its string from them, so nothing is re-encoded and a lone surrogate
  // survives the crossing.
  func.func @LyJsProxy_PushStr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "push.str"} {
    %c0 = arith.constant 0 : index
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %width_index = arith.index_cast %width : i64 to index
    %byte_count = memref.dim %bytes, %c0 : memref<?xi8>
    %count = arith.divui %byte_count, %width_index : index
    %address = func.call @__ly_js_address(%bytes) : (memref<?xi8>) -> index
    func.call @LyJs_PushStr(%address, %count, %width_index) : (index, index, index) -> ()
    func.return
  }

  // ===== host -> Python =====
  // Each takes the handle a member access returned and drops it.
  func.func @LyJsProxy_TakeF64(%handle: i32) -> f64 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "take.f64"} {
    %number = arith.constant 1 : i32
    func.call @__ly_js_expect(%handle, %number) : (i32, i32) -> ()
    %value = func.call @LyJs_ToF64(%handle) : (i32) -> f64
    func.call @LyJs_Drop(%handle) : (i32) -> ()
    func.return %value : f64
  }

  // An int is a number with no fractional part that a double holds exactly,
  // or a BigInt that fits 64 bits.
  func.func @LyJsProxy_TakeI64(%handle: i32) -> i64 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "take.i64"} {
    %integer = arith.constant 5 : i32
    func.call @__ly_js_expect(%handle, %integer) : (i32, i32) -> ()
    %value = func.call @LyJs_ToI64(%handle) : (i32) -> i64
    func.call @LyJs_Drop(%handle) : (i32) -> ()
    func.return %value : i64
  }

  func.func @LyJsProxy_TakeBool(%handle: i32) -> i1 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "take.bool"} {
    %boolean = arith.constant 2 : i32
    func.call @__ly_js_expect(%handle, %boolean) : (i32, i32) -> ()
    %value = func.call @LyJs_ToF64(%handle) : (i32) -> f64
    %zero = arith.constant 0.0 : f64
    %truth = arith.cmpf one, %value, %zero : f64
    func.call @LyJs_Drop(%handle) : (i32) -> ()
    func.return %truth : i1
  }

  func.func @LyJsProxy_TakeStr(%handle: i32) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "take.str"} {
    %string = arith.constant 3 : i32
    func.call @__ly_js_expect(%handle, %string) : (i32, i32) -> ()
    %header, %bytes = func.call @__ly_js_string(%handle) : (i32) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyJs_Drop(%handle) : (i32) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // None for undefined and null, which is Pyodide's reading of both.
  func.func @LyJsProxy_TakeNone(%handle: i32) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "take.none"} {
    %nullish = arith.constant 4 : i32
    func.call @__ly_js_expect(%handle, %nullish) : (i32, i32) -> ()
    func.call @LyJs_Drop(%handle) : (i32) -> ()
    func.return
  }

  func.func @LyJsProxy_IsNullish(%handle: i32) -> i1 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "is_nullish"} {
    %raw = func.call @LyJs_IsNullish(%handle) : (i32) -> i32
    %zero = arith.constant 0 : i32
    %nullish = arith.cmpi ne, %raw, %zero : i32
    func.return %nullish : i1
  }

  func.func @LyJsProxy_Drop(%handle: i32) attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "drop"} {
    func.call @LyJs_Drop(%handle) : (i32) -> ()
    func.return
  }

  // A host string as a str of the narrowest width that holds it, written
  // by the host straight into the new str's storage.
  func.func private @__ly_js_string(%handle: i32) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %count32 = func.call @LyJs_StrCount(%handle) : (i32) -> i32
    %width32 = func.call @LyJs_StrWidth(%handle) : (i32) -> i32
    %count = arith.extui %count32 : i32 to i64
    %width = arith.extui %width32 : i32 to i64
    %width_index = arith.index_cast %width : i64 to index
    %header, %bytes = func.call @__ly_unicode_alloc(%count, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %address = func.call @__ly_js_address(%bytes) : (memref<?xi8>) -> index
    func.call @LyJs_StrWrite(%handle, %address, %width_index) : (i32, index, index) -> ()
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // ===== a union's dispatch =====
  // On the proxy, which keeps its handle; the conversions raise TypeError
  // like the reads they stand in for.
  func.func @LyJsProxy_Is(%proxy: memref<17xi64> {ly.ownership.object_header}, %kind: i32) -> i1 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "is"} {
    %handle = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    %raw = func.call @LyJs_Is(%handle, %kind) : (i32, i32) -> i32
    %zero = arith.constant 0 : i32
    %is = arith.cmpi ne, %raw, %zero : i32
    func.return %is : i1
  }

  // A conversion reads from a second handle and lets `take.*` drop it, so
  // the proxy's own handle stays the proxy's.
  func.func @LyJsProxy_Duplicate(%proxy: memref<17xi64> {ly.ownership.object_header}) -> i32 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "duplicate"} {
    %handle = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    %copy = func.call @LyJs_Dup(%handle) : (i32) -> i32
    func.return %copy : i32
  }

  func.func @LyJsProxy_InstanceOf(%proxy: memref<17xi64> {ly.ownership.object_header}, %constructor: memref<17xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "instanceof"} {
    %value = func.call @LyJsProxy_Handle(%proxy) : (memref<17xi64>) -> i32
    %class = func.call @LyJsProxy_Handle(%constructor) : (memref<17xi64>) -> i32
    %raw = func.call @LyJs_InstanceOf(%value, %class) : (i32, i32) -> i32
    %answer = func.call @__ly_js_checked(%raw) : (i32) -> i32
    %zero = arith.constant 0 : i32
    %is = arith.cmpi ne, %answer, %zero : i32
    func.return %is : i1
  }

  // ===== callbacks (runtime/lib/_js_bridge.py) =====
  func.func @LyJs_FunctionFor(%slot: i64) -> memref<17xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_js.function_for", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "function_for", ly.runtime.result_contract = "_js.JsProxy"} {
    %slot32 = arith.trunci %slot : i64 to i32
    %handle = func.call @LyJs_MakeFunction(%slot32) : (i32) -> i32
    %proxy = func.call @LyJsProxy_New(%handle) : (i32) -> memref<17xi64>
    func.return %proxy : memref<17xi64>
  }

  func.func @LyJs_CallbackSlot() -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_js.callback_slot", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "callback_slot", ly.runtime.result_contract = "builtins.int"} {
    %slot32 = func.call @LyJs_CurrentSlot() : () -> i32
    %slot = arith.extsi %slot32 : i32 to i64
    %boxed = func.call @LyLong_FromI64(%slot) : (i64) -> memref<2xi64>
    func.return %boxed : memref<2xi64>
  }

  func.func @LyJs_WaitForHostBuiltin(%timeout_ms: i64) -> i1 attributes {ly.runtime.builtin = "_js.wait_for_host", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "wait_for_host", ly.runtime.result_contract = "builtins.bool"} {
    %ms = arith.sitofp %timeout_ms : i64 to f64
    %waited = func.call @LyJs_WaitForHost(%ms) : (f64) -> i32
    %zero = arith.constant 0 : i32
    %did = arith.cmpi ne, %waited, %zero : i32
    func.return %did : i1
  }

  func.func @LyJs_FailCallback(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.builtin = "_js.fail_callback", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "_js.JsProxy", ly.runtime.primitive = "fail_callback", ly.runtime.result_contract = "types.NoneType"} {
    func.call @LyJsProxy_PushStr(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    func.call @LyJs_SetCallbackError() : () -> ()
    func.return
  }
}
