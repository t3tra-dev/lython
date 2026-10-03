// The host half of `from js import ...`: the functions a Lython program
// imports to reach JavaScript values (runtime/modules/_js.mlir declares them).
//
// Host-neutral on purpose. It knows the wasm memory only through the accessor
// it is given and nothing of the loader that instantiated the module, so the
// same object serves an Emscripten build (through the adapter lyc generates
// with --js-library) and a module instantiated by any other loader that hands
// these functions over as imports.
//
// A JavaScript value lives in `values` and the program holds its index -- a
// handle -- for as long as a Python reference does; the program drops it when
// the last one goes. Handles 0..3 are undefined, null, true and false, never
// dropped.
//
// Conventions, all of them from the Lython side:
//   - an address or a length is `index`, the target's size_t: a Number on
//     wasm32, a BigInt on wasm64;
//   - an i64 is a BigInt (WASM_BIGINT);
//   - a function that can throw returns -1 and parks the exception's text,
//     which LyJs_TakeError hands back as a string.
// eslint-disable-next-line no-unused-vars
var LythonJs = {
  // `exports` gives the module's exports once it is instantiated:
  // LyJs_Dispatch and LyJs_Release, which a callback calls back in through.
  // `options.suspending`: the loader wraps LyJs_WaitForHost in
  // WebAssembly.Suspending and runs the program under WebAssembly.promising
  // (JSPI), so a promise it returns suspends the program.
  create(memory, exports, options = {}) {
    const values = [undefined, null, true, false];
    const freeHandles = [];
    let pending = [];
    let parked = null;

    const view = () => new DataView(memory().buffer);
    const address = (index) => Number(index);
    const intern = (value) => {
      if (value === undefined) return 0;
      if (value === null) return 1;
      if (value === true) return 2;
      if (value === false) return 3;
      const handle = freeHandles.length ? freeHandles.pop() : values.length;
      values[handle] = value;
      return handle;
    };
    const fail = (error) => {
      parked =
        error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      return -1;
    };
    // An ASCII member name.
    const memberName = (pointer, length) => {
      const bytes = new Uint8Array(memory().buffer, address(pointer), Number(length));
      return String.fromCharCode(...bytes);
    };
    const takePending = () => {
      const taken = pending;
      pending = [];
      return taken;
    };
    // Python's code points as stored -- 1, 2 or 4 bytes each -- to a string.
    const codePoints = (pointer, count, width) => {
      const data = view();
      const base = address(pointer);
      let text = "";
      for (let i = 0; i < Number(count); ++i) {
        const at = base + i * Number(width);
        const point =
          width == 1 ? data.getUint8(at)
          : width == 2 ? data.getUint16(at, true)
          : data.getUint32(at, true);
        text += String.fromCodePoint(point);
      }
      return text;
    };
    // A string's code points, a lone surrogate counting as one, as Python
    // stores them.
    const pointsOf = (text) => Array.from(text, (unit) => unit.codePointAt(0));
    const describe = (value) =>
      value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
    // A Python callable handed to the host is a slot in the program's
    // callback table (runtime/lib/_js_bridge.py). The function the host gets
    // pushes a frame -- which slot, what arguments -- and calls the program's
    // one entry point; the program reads the frame through the `$`-named
    // globals below and leaves its result or its exception in it.
    const frames = [];
    // A program suspended in LyJs_WaitForHost, waiting for the host to call
    // it back.
    const waiters = new Set();
    const enter = (frame, entry) => {
      frames.push(frame);
      try {
        entry();
      } finally {
        frames.pop();
        if (frames.length === 0) for (const wake of [...waiters]) wake();
      }
    };
    const callbacks = new FinalizationRegistry((slot) =>
      enter({ slot }, () => exports().LyJs_Release()),
    );
    const makeFunction = (slot) => {
      const callback = function (...args) {
        const frame = {
          slot,
          args,
          result: undefined,
          error: null,
          set_result(value) {
            this.result = value;
          },
        };
        enter(frame, () => exports().LyJs_Dispatch());
        if (frame.error !== null) throw new Error(frame.error);
        return frame.result;
      };
      callbacks.register(callback, slot);
      return callback;
    };
    const frameGlobal = (name) => {
      const frame = frames[frames.length - 1];
      if (name == "$frame") return frame;
      // `$arg$<index>$<kind>`: the kind only keeps the program's globals
      // apart; the index is the argument.
      return frame.args[Number(name.split("$")[2])];
    };
    // LyJsKind: what `kind` asks of a value in LyJs_Expect.
    const kinds = {
      1: ["a number", (value) => typeof value === "number"],
      2: ["a boolean", (value) => typeof value === "boolean"],
      3: ["a string", (value) => typeof value === "string"],
      4: ["undefined or null", (value) => value == null],
      5: [
        "an integer",
        (value) =>
          (typeof value === "number" && Number.isSafeInteger(value)) ||
          (typeof value === "bigint" &&
            value >= -(2n ** 63n) && value < 2n ** 63n),
      ],
    };

    return {
      LyJs_Global(pointer, length) {
        const name = memberName(pointer, length);
        if (name.startsWith("$")) return intern(frameGlobal(name));
        // ⛔ Not `globalThis[name]` unchecked: a global the host does not
        // have is JavaScript's ReferenceError, not an undefined to carry on
        // with.
        if (!(name in globalThis))
          return fail(new ReferenceError(`${name} is not defined`));
        try {
          return intern(globalThis[name]);
        } catch (error) {
          return fail(error);
        }
      },
      LyJs_Get(object, pointer, length) {
        try {
          return intern(values[object][memberName(pointer, length)]);
        } catch (error) {
          return fail(error);
        }
      },
      LyJs_Set(object, pointer, length) {
        const [value] = takePending();
        try {
          values[object][memberName(pointer, length)] = value;
          return 0;
        } catch (error) {
          return fail(error);
        }
      },
      LyJs_CallMethod(object, pointer, length) {
        const args = takePending();
        try {
          const receiver = values[object];
          const name = memberName(pointer, length);
          const method = receiver[name];
          if (typeof method !== "function")
            throw new TypeError(`${name} is not a function`);
          return intern(Reflect.apply(method, receiver, args));
        } catch (error) {
          return fail(error);
        }
      },
      LyJs_Construct(object) {
        const args = takePending();
        try {
          return intern(Reflect.construct(values[object], args));
        } catch (error) {
          return fail(error);
        }
      },
      LyJs_MakeFunction(slot) {
        return intern(makeFunction(slot));
      },
      LyJs_CurrentSlot() {
        return frames[frames.length - 1].slot;
      },
      LyJs_SetCallbackError() {
        frames[frames.length - 1].error = String(takePending()[0]);
      },
      LyJs_Drop(handle) {
        if (handle < 4) return;
        values[handle] = undefined;
        freeHandles.push(handle);
      },
      LyJs_PushHandle(handle) {
        pending.push(values[handle]);
      },
      LyJs_PushF64(value) {
        pending.push(value);
      },
      // Pyodide's rule: a Number when it holds the int exactly.
      LyJs_PushI64(value) {
        const number = Number(value);
        pending.push(Number.isSafeInteger(number) ? number : value);
      },
      LyJs_PushBigInt(pointer, length) {
        pending.push(BigInt(memberName(pointer, length)));
      },
      LyJs_PushStr(pointer, count, width) {
        pending.push(codePoints(pointer, count, Number(width)));
      },
      LyJs_Expect(handle, kind) {
        const [expected, test] = kinds[kind];
        const value = values[handle];
        if (test(value)) return 1;
        parked = `JavaScript returned ${describe(value)} where ${expected} was declared`;
        return 0;
      },
      LyJs_Is(handle, kind) {
        return kinds[kind][1](values[handle]) ? 1 : 0;
      },
      LyJs_Dup(handle) {
        return intern(values[handle]);
      },
      LyJs_InstanceOf(value, constructor) {
        try {
          return values[value] instanceof values[constructor] ? 1 : 0;
        } catch (error) {
          return fail(error);
        }
      },
      LyJs_IsNullish(handle) {
        return values[handle] == null ? 1 : 0;
      },
      LyJs_ToF64(handle) {
        return Number(values[handle]);
      },
      LyJs_ToI64(handle) {
        return BigInt(values[handle]);
      },
      LyJs_StrCount(handle) {
        return pointsOf(values[handle]).length;
      },
      LyJs_StrWidth(handle) {
        const widest = Math.max(0, ...pointsOf(values[handle]));
        return widest < 0x100 ? 1 : widest < 0x10000 ? 2 : 4;
      },
      LyJs_StrWrite(handle, pointer, width) {
        const data = view();
        const base = address(pointer);
        const size = Number(width);
        pointsOf(values[handle]).forEach((point, i) => {
          const at = base + i * size;
          if (size == 1) data.setUint8(at, point);
          else if (size == 2) data.setUint16(at, point, true);
          else data.setUint32(at, point, true);
        });
      },
      // Whether LyJs_WaitForHost may be called: JSPI is there, and the
      // program is not inside a callback the host is running -- that is the
      // host's own stack, which no promise can suspend.
      // ⛔ Asked by a plain import first, because LyJs_WaitForHost is a
      // Suspending one, and calling it outside WebAssembly.promising traps
      // even when it would not suspend.
      LyJs_CanWaitForHost() {
        return options.suspending && frames.length === 0 ? 1 : 0;
      },
      LyJs_WaitForHost(ms) {
        return new Promise((resolve) => {
          let timer = null;
          const wake = () => {
            if (timer !== null) clearTimeout(timer);
            waiters.delete(wake);
            resolve(1);
          };
          waiters.add(wake);
          if (ms >= 0) timer = setTimeout(wake, ms);
        });
      },
      LyJs_TakeError() {
        const message = parked ?? "";
        parked = null;
        return intern(message);
      },
    };
  },
};
