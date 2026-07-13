#include "SanitizerSupport.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <unistd.h>
#endif

#ifndef LYTHON_CLANG_RESOURCE_DIR
#define LYTHON_CLANG_RESOURCE_DIR ""
#endif

namespace lython::driver {
namespace {

struct SanitizerAction {
  bool enable = false;
  std::string value;
};

std::vector<SanitizerAction> sanitizerActions;

void clearSanitizers(SanitizerConfig &config) { config = SanitizerConfig{}; }

void setSanitizer(SanitizerConfig &config, llvm::StringRef name, bool enabled) {
  if (name == "address") {
    config.address = enabled;
  } else if (name == "leak") {
    config.leak = enabled;
  } else if (name == "thread") {
    config.thread = enabled;
  } else if (name == "undefined") {
    config.undefined = enabled;
  }
}

bool applySanitizerToken(SanitizerConfig &config, llvm::StringRef token,
                         bool enable) {
  token = token.trim();
  if (token.empty()) {
    llvm::errs() << "error: empty sanitizer name in "
                 << (enable ? "-fsanitize" : "-fno-sanitize") << "\n";
    return false;
  }

  if (token == "all") {
    if (enable) {
      config.address = true;
      config.leak = true;
      config.thread = true;
      config.undefined = true;
    } else {
      clearSanitizers(config);
    }
    return true;
  }

  if (token == "address" || token == "leak" || token == "thread" ||
      token == "undefined") {
    setSanitizer(config, token, enable);
    return true;
  }

  llvm::errs() << "error: unsupported sanitizer '" << token
               << "'; supported sanitizers are address, leak, thread, and "
                  "undefined\n";
  return false;
}

std::optional<llvm::StringRef>
jitSanitizerPreloadEnvName(const llvm::Triple &targetTriple) {
  if (targetTriple.isOSDarwin())
    return llvm::StringRef("DYLD_INSERT_LIBRARIES");
  if (targetTriple.isOSLinux())
    return llvm::StringRef("LD_PRELOAD");
  return std::nullopt;
}

bool pathListContains(llvm::StringRef pathList, llvm::StringRef needle) {
  llvm::SmallVector<llvm::StringRef, 8> entries;
  pathList.split(entries, ":", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef entry : entries)
    if (entry == needle)
      return true;
  return false;
}

std::string prependPathList(llvm::ArrayRef<std::string> paths,
                            llvm::StringRef existing) {
  std::string result;
  for (const std::string &path : paths) {
    if (!result.empty())
      result += ":";
    result += path;
  }
  if (!existing.empty()) {
    if (!result.empty())
      result += ":";
    result += existing.str();
  }
  return result;
}

bool jitSanitizerRuntimeIsPreloaded(llvm::StringRef path,
                                    const llvm::Triple &targetTriple) {
  std::optional<llvm::StringRef> envName =
      jitSanitizerPreloadEnvName(targetTriple);
  if (!envName)
    return false;
  const std::string envNameStorage = envName->str();
  const char *current = std::getenv(envNameStorage.c_str());
  return current && pathListContains(current, path);
}

const char *sanitizerLoadedSymbolName(llvm::StringRef runtime) {
  if (runtime == "asan")
    return "__asan_init";
  if (runtime == "tsan")
    return "__tsan_init";
  if (runtime == "lsan")
    return "__lsan_init";
  return nullptr;
}

bool sanitizerRuntimeIsLoaded(llvm::StringRef runtime) {
  const char *symbolName = sanitizerLoadedSymbolName(runtime);
  if (!symbolName)
    return false;
#if defined(_WIN32)
  return false;
#else
  return dlsym(RTLD_DEFAULT, symbolName) != nullptr;
#endif
}

std::optional<std::string>
sanitizerRuntimeLibraryPath(llvm::StringRef runtime,
                            const llvm::Triple &targetTriple) {
  llvm::StringRef resourceDir = LYTHON_CLANG_RESOURCE_DIR;
  if (resourceDir.empty())
    return std::nullopt;

  llvm::SmallString<256> path(resourceDir);
  llvm::sys::path::append(path, "lib");

  if (targetTriple.isOSDarwin()) {
    llvm::sys::path::append(path, "darwin");
    std::string filename =
        (llvm::Twine("libclang_rt.") + runtime + "_osx_dynamic.dylib").str();
    llvm::sys::path::append(path, filename);
    if (llvm::sys::fs::exists(path))
      return path.str().str();
    return std::nullopt;
  }

  return std::nullopt;
}

mlir::LogicalResult addJITSanitizerRuntime(llvm::orc::LLJIT &jit,
                                           llvm::StringRef runtime,
                                           const llvm::Triple &targetTriple) {
  if (sanitizerRuntimeIsLoaded(runtime))
    return mlir::success();

  std::optional<std::string> path =
      sanitizerRuntimeLibraryPath(runtime, targetTriple);
  if (!path) {
    llvm::errs() << "error: cannot find dynamic compiler-rt runtime for "
                 << "-fsanitize=" << runtime << " and target '"
                 << targetTriple.str() << "' under "
                 << LYTHON_CLANG_RESOURCE_DIR << "\n";
    return mlir::failure();
  }

  if (jitSanitizerRuntimeIsPreloaded(*path, targetTriple))
    return mlir::success();

  auto generator = llvm::orc::DynamicLibrarySearchGenerator::Load(
      path->c_str(), jit.getDataLayout().getGlobalPrefix());
  if (!generator) {
    llvm::errs() << "error: failed to load sanitizer runtime '" << *path
                 << "': " << llvm::toString(generator.takeError()) << "\n";
    return mlir::failure();
  }
  jit.getMainJITDylib().addGenerator(std::move(*generator));
  return mlir::success();
}

mlir::LogicalResult collectJITSanitizerRuntimePaths(
    const SanitizerConfig &sanitizers, const llvm::Triple &targetTriple,
    llvm::SmallVectorImpl<std::string> &runtimePaths) {
  auto appendRuntime = [&](llvm::StringRef sanitizerName,
                           llvm::StringRef runtimeName) -> mlir::LogicalResult {
    if (sanitizerRuntimeIsLoaded(runtimeName))
      return mlir::success();

    std::optional<std::string> path =
        sanitizerRuntimeLibraryPath(runtimeName, targetTriple);
    if (!path) {
      llvm::errs() << "error: cannot find dynamic compiler-rt runtime for "
                   << "-fsanitize=" << sanitizerName << " and target '"
                   << targetTriple.str() << "' under "
                   << LYTHON_CLANG_RESOURCE_DIR << "\n";
      return mlir::failure();
    }
    runtimePaths.push_back(std::move(*path));
    return mlir::success();
  };

  if (sanitizers.address && mlir::failed(appendRuntime("address", "asan")))
    return mlir::failure();
  if (sanitizers.thread && mlir::failed(appendRuntime("thread", "tsan")))
    return mlir::failure();
  if (sanitizers.leak && mlir::failed(appendRuntime("leak", "lsan")))
    return mlir::failure();
  return mlir::success();
}

} // namespace

void recordSanitizerAction(bool enable, const std::string &value) {
  sanitizerActions.push_back(SanitizerAction{enable, value});
}

mlir::LogicalResult buildSanitizerConfig(SanitizerConfig &config) {
  clearSanitizers(config);

  for (const SanitizerAction &action : sanitizerActions) {
    llvm::SmallVector<llvm::StringRef, 8> tokens;
    llvm::StringRef(action.value)
        .split(tokens, ",", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (tokens.empty()) {
      llvm::errs() << "error: empty sanitizer list in "
                   << (action.enable ? "-fsanitize" : "-fno-sanitize") << "\n";
      return mlir::failure();
    }
    for (llvm::StringRef token : tokens)
      if (!applySanitizerToken(config, token, action.enable))
        return mlir::failure();
  }

  if (config.thread && config.address) {
    llvm::errs() << "error: invalid argument '-fsanitize=address' not allowed "
                    "with '-fsanitize=thread'\n";
    return mlir::failure();
  }
  if (config.thread && config.leak) {
    llvm::errs() << "error: invalid argument '-fsanitize=leak' not allowed "
                    "with '-fsanitize=thread'\n";
    return mlir::failure();
  }

  return mlir::success();
}

std::string sanitizerClangList(const SanitizerConfig &config) {
  std::string result;
  auto append = [&](llvm::StringRef name) {
    if (!result.empty())
      result += ",";
    result += name.str();
  };
  if (config.address)
    append("address");
  if (config.leak)
    append("leak");
  if (config.thread)
    append("thread");
  if (config.undefined)
    append("undefined");
  return result;
}

void addSanitizerInstrumentationPasses(llvm::ModulePassManager &modulePM,
                                       const SanitizerConfig &sanitizers) {
  if (sanitizers.address) {
    llvm::AddressSanitizerOptions options;
    options.UseAfterScope = true;
    modulePM.addPass(llvm::AddressSanitizerPass(options));
  }

  if (sanitizers.thread) {
    modulePM.addPass(llvm::ModuleThreadSanitizerPass());
    modulePM.addPass(
        llvm::createModuleToFunctionPassAdaptor(llvm::ThreadSanitizerPass()));
  }
}

mlir::LogicalResult
addJITSanitizerRuntimes(llvm::orc::LLJIT &jit,
                        const SanitizerConfig &sanitizers,
                        const llvm::Triple &targetTriple) {
  if (!sanitizers.requiresJITRuntimePreload())
    return mlir::success();

  if (sanitizers.address &&
      mlir::failed(addJITSanitizerRuntime(jit, "asan", targetTriple)))
    return mlir::failure();
  if (sanitizers.thread &&
      mlir::failed(addJITSanitizerRuntime(jit, "tsan", targetTriple)))
    return mlir::failure();
  if (sanitizers.leak &&
      mlir::failed(addJITSanitizerRuntime(jit, "lsan", targetTriple)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult
ensureJITSanitizerRuntimesPreloaded(char **argv,
                                    const SanitizerConfig &sanitizers,
                                    const llvm::Triple &targetTriple) {
  if (!sanitizers.any())
    return mlir::success();
  if (!sanitizers.requiresJITRuntimePreload())
    return mlir::success();

  std::optional<llvm::StringRef> envName =
      jitSanitizerPreloadEnvName(targetTriple);
  if (!envName) {
    llvm::errs() << "error: JIT -fsanitize requires a dynamic compiler-rt "
                    "runtime preloaded at process startup for target '"
                 << targetTriple.str() << "'\n";
    return mlir::failure();
  }

  llvm::SmallVector<std::string, 3> runtimePaths;
  if (mlir::failed(collectJITSanitizerRuntimePaths(sanitizers, targetTriple,
                                                   runtimePaths)))
    return mlir::failure();

  const std::string envNameStorage = envName->str();
  const char *currentValue = std::getenv(envNameStorage.c_str());
  llvm::StringRef current =
      currentValue ? llvm::StringRef(currentValue) : llvm::StringRef();
  llvm::SmallVector<std::string, 3> missingPaths;
  for (const std::string &path : runtimePaths)
    if (!pathListContains(current, path))
      missingPaths.push_back(path);
  if (missingPaths.empty())
    return mlir::success();

#if defined(_WIN32)
  llvm::errs() << "error: automatic JIT sanitizer runtime preload is not "
                  "implemented on Windows\n";
  return mlir::failure();
#else
  std::string updatedValue = prependPathList(missingPaths, current);
  if (setenv(envNameStorage.c_str(), updatedValue.c_str(), /*overwrite=*/1) !=
      0) {
    llvm::errs() << "error: failed to set " << envNameStorage << ": "
                 << std::strerror(errno) << "\n";
    return mlir::failure();
  }

  execvp(argv[0], argv);
  llvm::errs() << "error: failed to re-exec '" << argv[0] << "' with "
               << envNameStorage << "=" << updatedValue
               << ": " << std::strerror(errno) << "\n";
  return mlir::failure();
#endif
}

void callLeakSanitizerHook(const char *symbolName) {
#if defined(_WIN32)
  (void)symbolName;
#else
  using HookFn = void (*)();
  if (void *symbol = dlsym(RTLD_DEFAULT, symbolName))
    reinterpret_cast<HookFn>(symbol)();
#endif
}

} // namespace lython::driver
