#ifndef LYTHON_COMMON_MANGLING_H
#define LYTHON_COMMON_MANGLING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace py::mangling {

// The symbol a function the program DEFINES is emitted under.
//
// A Python name is not a C name. `def abort()` at a module's top level and
// libc's `abort` were one symbol, and so were `fmod`, `free` and `puts`: the
// runtime declares the C functions it calls and is merged into the program
// while it is lowered, and a program function of the same name was refused
// there -- or, for a runtime DEFINITION, replaced by it. So a definition is
// named by its module and its qualified name, in the Itanium C++ nested-name
// form, which no C or runtime symbol has and every C++ demangler reads:
//
//     __main__.free               _ZN8__main__4freeE      __main__::free
//     pkg.mod.f                   _ZN3pkg3mod1fE          pkg::mod::f
//     C.m                         _ZN8__main__1C1mE       __main__::C::m
//     outer.<locals>.inner        _ZN8__main__5outer8<locals>5innerE
//
// A second definition with the same qualified name -- a redefinition, a
// specialization, the two halves of a rewritten method -- takes `.N`, which
// is how LLVM names a clone of one entity.
// ⛔ Not the Itanium grammar for parameters or local entities: Python has no
// overloads to tell apart, and its qualified name already spells the scopes
// (`<locals>`), so a nested name with no parameter list is the whole of it.
inline void appendSourceName(std::string &out, llvm::StringRef component) {
  out += std::to_string(component.size());
  out += component;
}

inline std::string pythonDefinitionSymbol(llvm::StringRef module,
                                          llvm::StringRef qualname,
                                          unsigned discriminator = 0) {
  std::string out = "_ZN";
  llvm::SmallVector<llvm::StringRef, 4> parts;
  module.split(parts, '.');
  for (llvm::StringRef part : parts)
    appendSourceName(out, part);
  parts.clear();
  qualname.split(parts, '.');
  for (llvm::StringRef part : parts)
    appendSourceName(out, part);
  out += 'E';
  if (discriminator != 0) {
    out += '.';
    out += std::to_string(discriminator);
  }
  return out;
}

// The attributes a definition's function carries for what a traceback and a
// generator's repr print: CPython's `co_name` and `co_qualname`.
inline constexpr llvm::StringLiteral kFunctionNameAttr{"ly.function.name"};
inline constexpr llvm::StringLiteral kFunctionQualnameAttr{
    "ly.function.qualname"};

} // namespace py::mangling

#endif // LYTHON_COMMON_MANGLING_H
