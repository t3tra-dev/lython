#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <optional>
#include <string>

namespace py::platform_constants {

inline bool isStaticStringBinding(llvm::StringRef binding) {
  // os.name and the os path constants live in the stdlib source module
  // (runtime/lib/os.py) as folded module constants, not here. Target-
  // independent sys constants (sys.version etc.) live in the sys.mlir
  // manifest str/int constant channels, not here: this table is only for
  // values the manifest cannot know because they depend on the target triple.
  return binding == "sys.platform" || binding == "sys.byteorder";
}

inline bool isStaticStringCallable(llvm::StringRef binding) {
  return binding == "platform.system" ||
         binding == "sys.getdefaultencoding" ||
         binding == "sys.getfilesystemencoding";
}

inline bool isStaticIntBinding(llvm::StringRef binding) {
  return binding == "sys.maxsize";
}

inline std::string effectiveTriple(llvm::StringRef targetTriple) {
  if (!targetTriple.empty())
    return llvm::Triple(targetTriple).normalize();
  return llvm::Triple(llvm::sys::getDefaultTargetTriple()).normalize();
}

inline std::optional<std::string> sysPlatform(const llvm::Triple &triple) {
  if (triple.isOSDarwin())
    return std::string("darwin");
  if (triple.isOSLinux())
    return std::string("linux");
  if (triple.isOSWindows())
    return std::string("win32");
  if (triple.isOSFreeBSD())
    return std::string("freebsd");
  if (triple.isOSOpenBSD())
    return std::string("openbsd");
  if (triple.isOSNetBSD())
    return std::string("netbsd");
  return std::nullopt;
}

inline std::optional<std::string> platformSystem(const llvm::Triple &triple) {
  if (triple.isOSDarwin())
    return std::string("Darwin");
  if (triple.isOSLinux())
    return std::string("Linux");
  if (triple.isOSWindows())
    return std::string("Windows");
  if (triple.isOSFreeBSD())
    return std::string("FreeBSD");
  if (triple.isOSOpenBSD())
    return std::string("OpenBSD");
  if (triple.isOSNetBSD())
    return std::string("NetBSD");
  return std::nullopt;
}

inline std::optional<std::string> sysByteorder(const llvm::Triple &triple) {
  return std::string(triple.isLittleEndian() ? "little" : "big");
}

inline std::optional<std::string>
staticStringValue(llvm::StringRef binding, llvm::StringRef targetTriple) {
  llvm::Triple triple(effectiveTriple(targetTriple));
  return llvm::StringSwitch<std::optional<std::string>>(binding)
      .Case("sys.platform", sysPlatform(triple))
      .Case("sys.byteorder", sysByteorder(triple))
      // Lython strings are UTF-8 on every target (PEP 529 made this true for
      // the filesystem encoding on Windows too), so both encodings are
      // triple-independent; they live here rather than in the sys.mlir
      // manifest because the manifest has no zero-arg callable fold channel.
      .Case("sys.getdefaultencoding", std::string("utf-8"))
      .Case("sys.getfilesystemencoding", std::string("utf-8"))
      .Case("platform.system", platformSystem(triple))
      .Default(std::nullopt);
}

inline std::optional<long long>
staticIntValue(llvm::StringRef binding, llvm::StringRef targetTriple) {
  llvm::Triple triple(effectiveTriple(targetTriple));
  if (binding == "sys.maxsize")
    return triple.isArch64Bit() ? 9223372036854775807LL : 2147483647LL;
  return std::nullopt;
}

} // namespace py::platform_constants
