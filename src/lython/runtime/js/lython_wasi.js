// The loader of a WASI program run under a JavaScript host
// (`lyc --target wasm32-wasip1 --js-host -o prog.js`): the WASI preview-1
// calls the program makes, answered here, and the host's half of `js`
// (lython_js.js, which lyc puts ahead of this file).
//
// With JSPI (`WebAssembly.Suspending` and `WebAssembly.promising`) the
// program can wait on the host: LyJs_WaitForHost returns a promise and the
// program is suspended until it settles, so timers fire and promises settle
// while asyncio waits. Without JSPI it is never called (LyJs_CanWaitForHost
// says so) and asyncio raises where it would have waited.
//
// ⛔ `poll_oneoff` -- time.sleep -- does not suspend. wasi-libc calls it from
// wherever the program sleeps, a callback the host is running included, and a
// Suspending import called outside WebAssembly.promising traps even when it
// returns no promise. It spins, as a sleep does on a thread nothing else
// runs on.
//
// Under node the file system is the host's: node's own WASI implementation
// (`node:wasi`) answers the calls these do not, with the working directory
// and `/` preopened and the process's environment, as a native program sees
// them. In a page only the calls a program without a file system makes are
// answered: there is nothing preopened, so opening a file fails with ENOENT
// and the program sees the OSError CPython would.
// eslint-disable-next-line no-unused-vars
var LythonWasi = (() => {
  const SUCCESS = 0;
  const EBADF = 8;
  const EINVAL = 28;
  const ENOENT = 44;
  const ENOSYS = 52;
  const isNode =
    typeof process === "object" && process.versions && process.versions.node;

  class Exit {
    constructor(code) {
      this.code = code;
    }
  }

  // Bytes written to fd 1 and 2. Node writes them as they come; a browser
  // console takes lines, so it holds a partial one.
  const makeOutput = () => {
    if (isNode)
      return {
        1: (bytes) => process.stdout.write(bytes),
        2: (bytes) => process.stderr.write(bytes),
      };
    const decoder = new TextDecoder();
    const pending = { 1: "", 2: "" };
    const sink = (fd, log) => (bytes) => {
      pending[fd] += decoder.decode(bytes, { stream: true });
      const lines = pending[fd].split("\n");
      pending[fd] = lines.pop();
      for (const line of lines) log(line);
    };
    return { 1: sink(1, console.log), 2: sink(2, console.error) };
  };

  // Nanoseconds on clock `id`: 0 is the realtime clock, every other the
  // monotonic one.
  const now = (id) =>
    id === 0
      ? BigInt(Date.now()) * 1000000n
      : BigInt(Math.round(performance.now() * 1e6));

  const makeWasi = (memory, args) => {
    const view = () => new DataView(memory().buffer);
    const bytes = () => new Uint8Array(memory().buffer);
    const encoder = new TextEncoder();
    const argBytes = args.map((arg) => encoder.encode(arg + "\0"));
    const output = makeOutput();
    const readStdin = (buffer) => {
      if (!isNode) return 0;
      try {
        return require("fs").readSync(0, buffer);
      } catch (error) {
        if (error.code === "EAGAIN" || error.code === "EOF") return 0;
        throw error;
      }
    };
    return {
      args_sizes_get(count, size) {
        view().setUint32(count, argBytes.length, true);
        view().setUint32(
          size,
          argBytes.reduce((total, arg) => total + arg.length, 0),
          true,
        );
        return SUCCESS;
      },
      args_get(argv, buffer) {
        let at = buffer;
        argBytes.forEach((arg, index) => {
          view().setUint32(argv + 4 * index, at, true);
          bytes().set(arg, at);
          at += arg.length;
        });
        return SUCCESS;
      },
      environ_sizes_get(count, size) {
        view().setUint32(count, 0, true);
        view().setUint32(size, 0, true);
        return SUCCESS;
      },
      environ_get() {
        return SUCCESS;
      },
      clock_res_get(id, resolution) {
        view().setBigUint64(resolution, 1000n, true);
        return SUCCESS;
      },
      clock_time_get(id, precision, time) {
        view().setBigUint64(time, now(id), true);
        return SUCCESS;
      },
      random_get(buffer, length) {
        for (let at = 0; at < length; at += 65536)
          crypto.getRandomValues(
            new Uint8Array(memory().buffer, buffer + at,
                           Math.min(65536, length - at)),
          );
        return SUCCESS;
      },
      fd_close(fd) {
        return fd <= 2 ? SUCCESS : EBADF;
      },
      // The three standard streams are character devices, so the C library
      // line-buffers them as it does a terminal's.
      fd_fdstat_get(fd, stat) {
        if (fd > 2) return EBADF;
        const data = view();
        data.setUint8(stat, 2);
        data.setUint16(stat + 2, 0, true);
        data.setBigUint64(stat + 8, 0n, true);
        data.setBigUint64(stat + 16, 0n, true);
        return SUCCESS;
      },
      fd_fdstat_set_flags(fd) {
        return fd <= 2 ? SUCCESS : EBADF;
      },
      fd_prestat_get() {
        return EBADF;
      },
      fd_prestat_dir_name() {
        return EBADF;
      },
      fd_read(fd, iovs, count, read) {
        if (fd !== 0) return EBADF;
        let total = 0;
        for (let index = 0; index < count; ++index) {
          const data = view();
          const at = data.getUint32(iovs + 8 * index, true);
          const length = data.getUint32(iovs + 8 * index + 4, true);
          const got = readStdin(new Uint8Array(memory().buffer, at, length));
          total += got;
          if (got < length) break;
        }
        view().setUint32(read, total, true);
        return SUCCESS;
      },
      fd_write(fd, iovs, count, written) {
        const write = output[fd];
        if (!write) return EBADF;
        let total = 0;
        for (let index = 0; index < count; ++index) {
          const data = view();
          const at = data.getUint32(iovs + 8 * index, true);
          const length = data.getUint32(iovs + 8 * index + 4, true);
          write(bytes().slice(at, at + length));
          total += length;
        }
        view().setUint32(written, total, true);
        return SUCCESS;
      },
      fd_seek() {
        return ENOSYS;
      },
      path_open() {
        return ENOENT;
      },
      // Clock subscriptions sleep until the earliest one; an fd one is ready
      // at once.
      poll_oneoff(subscriptions, events, count, ready) {
        if (count === 0) return EINVAL;
        const data = view();
        let delayMs = 0;
        let sawFd = false;
        for (let index = 0; index < count; ++index) {
          const at = subscriptions + 48 * index;
          if (data.getUint8(at + 8) !== 0) {
            sawFd = true;
            continue;
          }
          const id = data.getUint32(at + 16, true);
          let timeout = data.getBigUint64(at + 24, true);
          if (data.getUint16(at + 40, true) & 1) timeout -= now(id);
          const ms = Number(timeout > 0n ? timeout : 0n) / 1e6;
          delayMs = index === 0 || ms < delayMs ? ms : delayMs;
        }
        if (sawFd) delayMs = 0;
        const answer = () => {
          const out = view();
          for (let index = 0; index < count; ++index) {
            const at = subscriptions + 48 * index;
            const event = events + 32 * index;
            out.setBigUint64(event, out.getBigUint64(at, true), true);
            out.setUint16(event + 8, 0, true);
            out.setUint8(event + 10, out.getUint8(at + 8));
          }
          out.setUint32(ready, count, true);
          return SUCCESS;
        };
        const end = performance.now() + delayMs;
        while (performance.now() < end) {
          // ⛔ A spin, because nothing else can wait here: the program holds
          // the host's thread.
        }
        return answer();
      },
      sched_yield() {
        return SUCCESS;
      },
      proc_exit(code) {
        throw new Exit(code);
      },
    };
  };

  // Runs the module and resolves to its exit status.
  const run = async (wasmBytes, args) => {
    const module = await WebAssembly.compile(wasmBytes);
    const suspending =
      typeof WebAssembly.Suspending === "function" &&
      typeof WebAssembly.promising === "function";
    let instance = null;
    const memory = () => instance.exports.memory;
    const js = LythonJs.create(memory, () => instance.exports, { suspending });
    const wasi = makeWasi(memory, args);
    // node's WASI for everything this file does not answer itself: the sleep
    // has to be a promise and the exit an exception this loader catches, and
    // the standard streams stay character devices -- line-buffered, so
    // `print` and the host's console.log come out in the order they ran.
    let nodeWasi = null;
    if (isNode) {
      // ⛔ Quiet: node calls its WASI experimental, on stderr, and the
      // program's stderr is the program's.
      const warn = process.emitWarning;
      process.emitWarning = (warning, ...rest) =>
        String(warning).includes("WASI") ? undefined
                                         : warn.call(process, warning, ...rest);
      const { WASI } = require("node:wasi");
      nodeWasi = new WASI({
        version: "preview1",
        args,
        // PWD is where the program starts (enterHostWorkingDirectory in
        // the runtime): the process's own, whatever the shell left there.
        env: { ...process.env, PWD: process.cwd() },
        preopens: { ".": process.cwd(), "/": "/" },
      });
      process.emitWarning = warn;
      const own = new Set(["poll_oneoff", "proc_exit", "fd_fdstat_get",
                           "fd_write"]);
      for (const [name, fn] of Object.entries(nodeWasi.wasiImport))
        if (!own.has(name)) wasi[name] = fn;
      const standard = wasi.fd_fdstat_get;
      const fdstat = nodeWasi.wasiImport.fd_fdstat_get;
      wasi.fd_fdstat_get = (fd, stat) =>
        fd <= 2 ? standard(fd, stat) : fdstat(fd, stat);
      const write = wasi.fd_write;
      const nodeWrite = nodeWasi.wasiImport.fd_write;
      wasi.fd_write = (fd, iovs, count, written) =>
        fd <= 2 ? write(fd, iovs, count, written)
                : nodeWrite(fd, iovs, count, written);
    }
    const imports = { wasi_snapshot_preview1: {}, lython_js: {} };
    for (const { module: from, name, kind } of WebAssembly.Module.imports(
      module,
    )) {
      if (kind !== "function") continue;
      const table = from === "lython_js" ? js : wasi;
      let fn = table[name];
      if (!fn) {
        if (from !== "wasi_snapshot_preview1")
          throw new Error(`the program imports ${from}.${name}, which this ` +
                          "host does not have");
        fn = () => ENOSYS;
      }
      if (suspending && name === "LyJs_WaitForHost")
        fn = new WebAssembly.Suspending(fn);
      imports[from][name] = fn;
    }
    instance = await WebAssembly.instantiate(module, imports);
    // ⛔ `initialize` rather than `start`: it only binds the memory, and the
    // program has to be started under WebAssembly.promising, which `start`
    // cannot do. A program exports `_start`, which `initialize` refuses, so
    // it is handed the memory alone.
    if (nodeWasi)
      nodeWasi.initialize({
        exports: { memory: instance.exports.memory, _initialize() {} },
      });
    const start = suspending
      ? WebAssembly.promising(instance.exports._start)
      : instance.exports._start;
    try {
      await start();
      return 0;
    } catch (error) {
      if (error instanceof Exit) return error.code;
      throw error;
    }
  };

  // The program beside this loader, by its file name; its exit status is the
  // process's. An exit from a callback after the program's main body has
  // returned ends the process there too.
  const main = (wasmFile) => {
    if (isNode) {
      process.on("uncaughtException", (error) => {
        if (error instanceof Exit) process.exit(error.code);
        throw error;
      });
      const path = require("path");
      const bytes = require("fs").readFileSync(path.join(__dirname, wasmFile));
      run(bytes, [wasmFile, ...process.argv.slice(2)]).then(
        (code) => {
          if (code !== 0) process.exit(code);
        },
        (error) => {
          console.error(error);
          process.exit(1);
        },
      );
      return;
    }
    const base =
      (typeof document === "object" && document.currentScript &&
       document.currentScript.src) || location.href;
    fetch(new URL(wasmFile, base))
      .then((response) => response.arrayBuffer())
      .then((bytes) => run(bytes, [wasmFile]))
      .then((code) => {
        if (code !== 0) console.error(`exit status ${code}`);
      });
  };

  return { run, main, Exit };
})();
