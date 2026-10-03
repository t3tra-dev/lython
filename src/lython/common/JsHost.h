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

} // namespace py
