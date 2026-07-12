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
  // (runtime/lib/os.py) as folded module constants, not here.
  return binding == "sys.platform";
}

inline bool isStaticStringCallable(llvm::StringRef binding) {
  return binding == "platform.system";
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

inline std::optional<std::string>
staticStringValue(llvm::StringRef binding, llvm::StringRef targetTriple) {
  llvm::Triple triple(effectiveTriple(targetTriple));
  return llvm::StringSwitch<std::optional<std::string>>(binding)
      .Case("sys.platform", sysPlatform(triple))
      .Case("platform.system", platformSystem(triple))
      .Default(std::nullopt);
}

} // namespace py::platform_constants
