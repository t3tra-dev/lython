#pragma once

// The JavaScript host's module (`js`, as Pyodide spells it): the names the
// emitter and the lowering agree on.

#include "llvm/ADT/StringRef.h"

namespace py {

// Unit attribute on the program's module: it imports the host's `js`, so a
// global spelled `js.<name>` is `globalThis[name]` and every `js.*` contract
// is a JavaScript value.
inline constexpr llvm::StringLiteral kJsHostModuleAttr = "ly.js.host";

// The one runtime contract every JavaScript value has: a handle into the
// host's table. The stub's classes type the program; at run time they are all
// this.
inline constexpr llvm::StringLiteral kJsProxyContract = "_js.JsProxy";

// The module whose contracts are the stub's classes.
inline constexpr llvm::StringLiteral kJsHostModule = "js";

// True where the program runs with a JavaScript host -- WASI linked with
// `--js-host` -- and folded statically, so a
// module can import `js` in a branch it guards with it. Lython's: CPython has
// no such attribute, and `sys.platform` cannot say it, since a WASI program
// may or may not have a host.
inline constexpr llvm::StringLiteral kJsHostBinding = "sys._js_host";

} // namespace py
