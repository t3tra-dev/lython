#include "Driver.h"
#include "DriverCodeGen.h"

#include "Common/Instrumentation.h"
#include "Emitter.h"
#include "Parser.h"
#include "embedded.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace lython::driver {
using py::PerfScope;

OwningOpRef<ModuleOp> parseModuleFromBuffer(StringRef buffer,
                                            MLIRContext &context,
                                            llvm::raw_ostream &diag) {
  auto module = parseSourceString<ModuleOp>(buffer, &context);
  if (!module)
    diag << "error: failed to parse MLIR source\n";
  return module;
}

static const std::vector<lython::parser::NodePtr> *
nodeListField(const lython::parser::Node &node, StringRef name) {
  const lython::parser::Field *field =
      lython::parser::findField(node, name.str());
  if (!field)
    return nullptr;
  return std::get_if<std::vector<lython::parser::NodePtr>>(&field->value);
}

static std::optional<std::string>
stringField(const lython::parser::Node &node, StringRef name) {
  const lython::parser::Field *field =
      lython::parser::findField(node, name.str());
  if (!field)
    return std::nullopt;
  if (const auto *value = std::get_if<std::string>(&field->value))
    return *value;
  return std::nullopt;
}

static std::int64_t
integerField(const lython::parser::Node &node, StringRef name,
                          std::int64_t fallback = 0) {
  const lython::parser::Field *field =
      lython::parser::findField(node, name.str());
  if (!field)
    return fallback;
  if (const auto *value = std::get_if<std::int64_t>(&field->value))
    return *value;
  return fallback;
}

static std::string joinModuleName(StringRef prefix, StringRef suffix) {
  if (prefix.empty())
    return suffix.str();
  if (suffix.empty())
    return prefix.str();
  return (Twine(prefix) + "." + suffix).str();
}

struct LocalSourceModulePath {
  std::string path;
  bool isStub = false;
  bool isPackage = false;
  bool isEmbedded = false;
};

static const py::runtime_library::embedded::StdlibSourceModule *
embeddedStdlibSource(StringRef moduleName) {
  namespace embedded = py::runtime_library::embedded;
  for (std::size_t index = 0; index < embedded::stdlibSourceModuleCount();
       ++index) {
    const embedded::StdlibSourceModule &entry =
        embedded::stdlibSourceModules()[index];
    if (moduleName == entry.name)
      return &entry;
  }
  return nullptr;
}

static std::optional<LocalSourceModulePath>
localSourceModulePath(StringRef baseDir,
                                                           StringRef moduleName) {
  llvm::SmallVector<StringRef, 8> parts;
  moduleName.split(parts, '.');
  if (parts.empty())
    return std::nullopt;
  llvm::SmallString<256> path(baseDir.empty() ? "." : baseDir);
  for (StringRef part : parts) {
    if (part.empty())
      return std::nullopt;
    llvm::sys::path::append(path, part);
  }
  llvm::SmallString<256> sourcePath(path);
  sourcePath += ".py";
  if (llvm::sys::fs::exists(sourcePath))
    return LocalSourceModulePath{sourcePath.str().str(), false, false};

  llvm::SmallString<256> stubPath(path);
  stubPath += ".pyi";
  if (llvm::sys::fs::exists(stubPath))
    return LocalSourceModulePath{stubPath.str().str(), true, false};

  llvm::SmallString<256> packageSourcePath(path);
  llvm::sys::path::append(packageSourcePath, "__init__.py");
  if (llvm::sys::fs::exists(packageSourcePath))
    return LocalSourceModulePath{packageSourcePath.str().str(), false, true};

  llvm::SmallString<256> packageStubPath(path);
  llvm::sys::path::append(packageStubPath, "__init__.pyi");
  if (llvm::sys::fs::exists(packageStubPath))
    return LocalSourceModulePath{packageStubPath.str().str(), true, true};

  // Embedded stdlib source (runtime/lib/*.py): resolved after user files,
  // before module manifests; compiled with the program like any source module.
  if (embeddedStdlibSource(moduleName))
    return LocalSourceModulePath{("<stdlib>/" + moduleName + ".py").str(),
                                 false, false, /*isEmbedded=*/true};
  return std::nullopt;
}

static bool packageInitExists(StringRef directory) {
  if (directory.empty())
    return false;
  llvm::SmallString<256> initPath(directory);
  llvm::sys::path::append(initPath, "__init__.py");
  if (llvm::sys::fs::exists(initPath))
    return true;
  llvm::SmallString<256> stubPath(directory);
  llvm::sys::path::append(stubPath, "__init__.pyi");
  return llvm::sys::fs::exists(stubPath);
}

static std::string staticPackageNameForSourcePath(StringRef sourcePath) {
  llvm::SmallVector<std::string, 8> reversedParts;
  llvm::SmallString<256> directory(llvm::sys::path::parent_path(sourcePath));
  while (!directory.empty() && packageInitExists(directory)) {
    reversedParts.push_back(llvm::sys::path::filename(directory).str());
    std::string parent = llvm::sys::path::parent_path(directory).str();
    directory = parent;
  }

  std::string packageName;
  for (auto it = reversedParts.rbegin(), end = reversedParts.rend(); it != end;
       ++it)
    packageName = joinModuleName(packageName, *it);
  return packageName;
}

static std::string packageNameForModuleName(StringRef moduleName) {
  std::pair<StringRef, StringRef> split = moduleName.rsplit('.');
  if (split.second.empty())
    return {};
  return split.first.str();
}

static std::optional<std::string> relativeBasePackage(StringRef packageName,
                                               std::int64_t level) {
  if (level <= 0)
    return packageName.str();
  if (packageName.empty())
    return std::nullopt;
  llvm::SmallVector<StringRef, 8> parts;
  packageName.split(parts, '.');
  if (level > static_cast<std::int64_t>(parts.size()))
    return std::nullopt;

  std::string resolved;
  std::size_t keep = parts.size() - static_cast<std::size_t>(level - 1);
  for (std::size_t index = 0; index < keep; ++index)
    resolved = joinModuleName(resolved, parts[index]);
  return resolved;
}

static std::optional<std::string> relativeBaseDirectory(StringRef baseDir,
                                                 std::int64_t level) {
  llvm::SmallString<256> directory(baseDir.empty() ? "." : baseDir);
  for (std::int64_t index = 1; index < level; ++index) {
    std::string parent = llvm::sys::path::parent_path(directory).str();
    if (parent.empty())
      return std::nullopt;
    directory = parent;
  }
  return directory.str().str();
}

struct SourceImportRequest {
  std::string moduleName;
  std::string sourcePath;
  bool isStub = false;
  bool isPackage = false;
  bool isEmbedded = false;
};

static bool appendLocalSourceRequest(
    llvm::SmallVectorImpl<SourceImportRequest> &requests,
    StringRef moduleName, std::optional<LocalSourceModulePath> sourcePath,
    std::set<std::string> *requestedModules = nullptr) {
  if (!sourcePath)
    return false;
  std::string moduleNameText = moduleName.str();
  if (requestedModules && !requestedModules->insert(moduleNameText).second)
    return false;
  requests.push_back(SourceImportRequest{std::move(moduleNameText),
                                         sourcePath->path,
                                         sourcePath->isStub,
                                         sourcePath->isPackage,
                                         sourcePath->isEmbedded});
  return true;
}

static void appendDottedImportSourceRequests(
    llvm::SmallVectorImpl<SourceImportRequest> &requests, StringRef baseDir,
    StringRef moduleName, std::set<std::string> *requestedModules = nullptr) {
  llvm::SmallVector<StringRef, 8> parts;
  moduleName.split(parts, '.');
  if (parts.empty())
    return;

  std::string prefix;
  for (std::size_t index = 0; index + 1 < parts.size(); ++index) {
    prefix = joinModuleName(prefix, parts[index]);
    std::optional<LocalSourceModulePath> sourcePath =
        localSourceModulePath(baseDir, prefix);
    if (sourcePath && sourcePath->isPackage)
      appendLocalSourceRequest(requests, prefix, sourcePath, requestedModules);
  }

  appendLocalSourceRequest(requests, moduleName,
                           localSourceModulePath(baseDir, moduleName),
                           requestedModules);
}

static void collectImportedModuleRequests(
    const lython::parser::Node &module, StringRef baseDir,
    StringRef packageName,
    llvm::SmallVectorImpl<SourceImportRequest> &requests) {
  std::set<std::string> requestedModules;
  auto appendIfLocal = [&](StringRef moduleName,
                           std::optional<LocalSourceModulePath> sourcePath) {
    appendLocalSourceRequest(requests, moduleName, std::move(sourcePath),
                             &requestedModules);
  };

  const auto *body = nodeListField(module, "body");
  if (!body)
    return;
  // Module-level `if` bodies participate (the platform-switch idiom, e.g.
  // `if os.name == "posix": from posix import *`). Discovery collects BOTH
  // branches: the emitter later folds the test and binds only the taken one,
  // and dead modules cost only their (DCE-able) compilation.
  std::vector<const lython::parser::Node *> statements;
  std::function<void(const std::vector<lython::parser::NodePtr> &)> flatten =
      [&](const std::vector<lython::parser::NodePtr> &list) {
        for (const lython::parser::NodePtr &statement : list) {
          if (!statement)
            continue;
          if (statement->kind == "If") {
            if (const auto *thenBody = nodeListField(*statement, "body"))
              flatten(*thenBody);
            if (const auto *elseBody = nodeListField(*statement, "orelse"))
              flatten(*elseBody);
            continue;
          }
          statements.push_back(statement.get());
        }
      };
  flatten(*body);
  for (const lython::parser::Node *statementPtr : statements) {
    const lython::parser::Node *statement = statementPtr;
    if (statement->kind == "Import") {
      const auto *aliases = nodeListField(*statement, "names");
      if (!aliases)
        continue;
      for (const lython::parser::NodePtr &alias : *aliases) {
        if (!alias)
          continue;
        if (std::optional<std::string> name = stringField(*alias, "name"))
          appendDottedImportSourceRequests(requests, baseDir, *name,
                                           &requestedModules);
      }
      continue;
    }
    if (statement->kind == "ImportFrom") {
      std::int64_t level = integerField(*statement, "level");
      std::optional<std::string> moduleName = stringField(*statement, "module");
      if (level == 0) {
        if (moduleName) {
          appendIfLocal(*moduleName,
                        localSourceModulePath(baseDir, *moduleName));
          const auto *aliases = nodeListField(*statement, "names");
          if (!aliases)
            continue;
          for (const lython::parser::NodePtr &alias : *aliases) {
            if (!alias)
              continue;
            std::optional<std::string> name = stringField(*alias, "name");
            if (!name || *name == "*")
              continue;
            std::string submodule = joinModuleName(*moduleName, *name);
            appendIfLocal(submodule, localSourceModulePath(baseDir, submodule));
          }
        }
        continue;
      }

      std::optional<std::string> relativePackage =
          relativeBasePackage(packageName, level);
      std::optional<std::string> relativeDirectory =
          relativeBaseDirectory(baseDir, level);
      if (!relativePackage || !relativeDirectory)
        continue;
      if (moduleName) {
        appendIfLocal(joinModuleName(*relativePackage, *moduleName),
                      localSourceModulePath(*relativeDirectory, *moduleName));
        const auto *aliases = nodeListField(*statement, "names");
        if (!aliases)
          continue;
        for (const lython::parser::NodePtr &alias : *aliases) {
          if (!alias)
            continue;
          std::optional<std::string> name = stringField(*alias, "name");
          if (!name || *name == "*")
            continue;
          std::string relativeSubmodule = joinModuleName(*moduleName, *name);
          appendIfLocal(joinModuleName(*relativePackage, relativeSubmodule),
                        localSourceModulePath(*relativeDirectory,
                                              relativeSubmodule));
        }
        continue;
      }

      const auto *aliases = nodeListField(*statement, "names");
      if (!aliases)
        continue;
      for (const lython::parser::NodePtr &alias : *aliases) {
        if (!alias)
          continue;
        std::optional<std::string> name = stringField(*alias, "name");
        if (!name || *name == "*")
          continue;
        appendIfLocal(joinModuleName(*relativePackage, *name),
                      localSourceModulePath(*relativeDirectory, *name));
      }
    }
  }
}

struct ParsedLocalSourceModule {
  std::string moduleName;
  std::string packageName;
  std::string sourceName;
  lython::parser::ParseResult parsed;
  bool isStub = false;
  bool isPackage = false;
};

static LogicalResult
collectLocalSourceModules(const lython::parser::Node &module, StringRef baseDir,
                          StringRef packageName, StringRef mainPath,
                          std::vector<ParsedLocalSourceModule> &sources,
                          std::set<std::string> &seen,
                          std::set<std::string> &visiting, bool releaseMode,
                          llvm::raw_ostream &diag) {
  llvm::SmallVector<SourceImportRequest, 8> imports;
  collectImportedModuleRequests(module, baseDir, packageName, imports);
  for (const SourceImportRequest &request : imports) {
    if (llvm::StringRef(request.sourcePath) == mainPath)
      continue;
    if (seen.find(request.moduleName) != seen.end())
      continue;
    if (visiting.find(request.moduleName) != visiting.end()) {
      diag << request.sourcePath
           << ": emit error: local source import cycle "
           << "involving module '" << request.moduleName << "'\n";
      return failure();
    }

    std::unique_ptr<llvm::MemoryBuffer> buffer;
    if (request.isEmbedded) {
      const py::runtime_library::embedded::StdlibSourceModule *entry =
          embeddedStdlibSource(request.moduleName);
      if (!entry) {
        diag << "error: embedded stdlib module '" << request.moduleName
             << "' disappeared from the registry\n";
        return failure();
      }
      buffer = llvm::MemoryBuffer::getMemBuffer(
          StringRef(reinterpret_cast<const char *>(entry->source),
                    entry->size),
          request.sourcePath, /*RequiresNullTerminator=*/false);
    } else {
      auto file = llvm::MemoryBuffer::getFile(request.sourcePath);
      if (!file) {
        diag << "error: could not open input file '" << request.sourcePath
             << "'\n";
        return failure();
      }
      buffer = std::move(*file);
    }

    lython::parser::ParseOptions options;
    options.typeComments = true;
    lython::parser::ParseResult parsed = lython::parser::parse(
        buffer->getBuffer(), request.sourcePath, options);
    if (!parsed.ok()) {
      for (const lython::parser::Diagnostic &diagnostic : parsed.diagnostics) {
        diag << request.sourcePath << ':' << diagnostic.location.line << ':'
             << diagnostic.location.column
             << ": parse error: " << diagnostic.message << "\n";
      }
      return failure();
    }

    visiting.insert(request.moduleName);
    std::string nestedBaseDir =
        llvm::sys::path::parent_path(request.sourcePath).str();
    std::string nestedPackageName = request.isPackage
                                        ? request.moduleName
                                        : packageNameForModuleName(
                                              request.moduleName);
    if (failed(collectLocalSourceModules(*parsed.tree, nestedBaseDir,
                                         nestedPackageName, mainPath, sources,
                                         seen, visiting, releaseMode, diag)))
      return failure();
    visiting.erase(request.moduleName);
    seen.insert(request.moduleName);
    sources.push_back(ParsedLocalSourceModule{
        request.moduleName, nestedPackageName,
        pythonTracebackPath(request.sourcePath, releaseMode),
        std::move(parsed), request.isStub, request.isPackage});
  }
  return success();
}

LogicalResult emitMLIRFromSource(StringRef source, StringRef sourcePath,
                                 StringRef importBaseDir,
                                 const DriverOptions &driverOptions,
                                 MLIRContext &context,
                                 OwningOpRef<ModuleOp> &module,
                                 llvm::raw_ostream &diag) {
  lython::parser::ParseOptions options;
  options.typeComments = true;
  lython::parser::ParseResult parsed;
  {
    PerfScope perf("parse");
    parsed = lython::parser::parse(source, sourcePath.str(), options);
  }
  if (!parsed.ok()) {
    for (const lython::parser::Diagnostic &diagnostic : parsed.diagnostics) {
      diag << sourcePath << ':' << diagnostic.location.line << ':'
           << diagnostic.location.column
           << ": parse error: " << diagnostic.message << "\n";
    }
    return failure();
  }

  std::vector<ParsedLocalSourceModule> localSources;
  std::set<std::string> seenSourceModules;
  std::set<std::string> visitingSourceModules;
  std::string mainPackageName = staticPackageNameForSourcePath(sourcePath);
  if (failed(collectLocalSourceModules(
          *parsed.tree, importBaseDir, mainPackageName, sourcePath,
          localSources, seenSourceModules, visitingSourceModules,
          driverOptions.releaseMode, diag)))
    return failure();

  lython::emitter::EmitResult emitted;
  {
    PerfScope perf("ir-generation");
    lython::emitter::EmitOptions emitOptions;
    emitOptions.sanitizeUndefined = driverOptions.sanitizers.undefined;
    emitOptions.mainPackageName = mainPackageName;
    emitOptions.targetTriple =
        codeGenTripleForTarget({}, driverOptions).normalize();
    emitOptions.sourceModules.reserve(localSources.size());
    for (const ParsedLocalSourceModule &source : localSources) {
      emitOptions.sourceModules.push_back(
          lython::emitter::EmitOptions::SourceModule{
              source.moduleName, source.packageName, source.sourceName,
              source.parsed.tree.get(), source.isStub});
    }
    emitted = lython::emitter::emitModule(
        *parsed.tree, context, "__main__",
        pythonTracebackPath(sourcePath, driverOptions.releaseMode),
        emitOptions);
  }
  if (!emitted.ok()) {
    for (const lython::parser::Diagnostic &diagnostic : emitted.diagnostics) {
      StringRef diagnosticFile =
          diagnostic.filename.empty() ? sourcePath : diagnostic.filename;
      diag << diagnosticFile << ':' << diagnostic.location.line << ':'
           << diagnostic.location.column
           << ": emit error: " << diagnostic.message << "\n";
    }
    return failure();
  }

  module = std::move(emitted.module);
  return success();
}

LogicalResult emitMLIRFromFile(StringRef pythonFile,
                               const DriverOptions &driverOptions,
                               MLIRContext &context,
                               OwningOpRef<ModuleOp> &module,
                               llvm::raw_ostream &diag) {
  auto file = llvm::MemoryBuffer::getFile(pythonFile);
  if (!file) {
    diag << "error: could not open input file '" << pythonFile << "'\n";
    return failure();
  }
  return emitMLIRFromSource(file->get()->getBuffer(), pythonFile,
                            llvm::sys::path::parent_path(pythonFile),
                            driverOptions, context, module, diag);
}

std::string pythonTracebackPath(StringRef inputPath, bool releaseMode) {
  if (releaseMode) {
    auto dotPath = [](StringRef path) {
      llvm::SmallString<256> result(".");
      llvm::sys::path::append(result, path);
      return result.str().str();
    };

    llvm::SmallString<256> absolute(inputPath);
    if (!llvm::sys::fs::make_absolute(absolute)) {
      llvm::sys::path::remove_dots(absolute, /*remove_dot_dot=*/true);

      llvm::SmallString<256> current;
      if (!llvm::sys::fs::current_path(current)) {
        llvm::sys::path::remove_dots(current, /*remove_dot_dot=*/true);

        llvm::SmallVector<llvm::StringRef, 16> absoluteParts;
        llvm::SmallVector<llvm::StringRef, 16> currentParts;
        for (auto it = llvm::sys::path::begin(absolute),
                  end = llvm::sys::path::end(absolute);
             it != end; ++it)
          absoluteParts.push_back(*it);
        for (auto it = llvm::sys::path::begin(current),
                  end = llvm::sys::path::end(current);
             it != end; ++it)
          currentParts.push_back(*it);

        bool underCurrent = absoluteParts.size() >= currentParts.size();
        for (unsigned index = 0; underCurrent && index < currentParts.size();
             ++index)
          underCurrent = absoluteParts[index] == currentParts[index];

        if (underCurrent) {
          llvm::SmallString<256> relative(".");
          for (unsigned index = currentParts.size();
               index < absoluteParts.size(); ++index)
            llvm::sys::path::append(relative, absoluteParts[index]);
          return relative.str().str();
        }
      }
    }

    StringRef filename = llvm::sys::path::filename(inputPath);
    if (filename.empty())
      filename = "<unknown>";
    return dotPath(filename);
  }

  if (llvm::sys::path::is_absolute(inputPath))
    return inputPath.str();
  llvm::SmallString<256> current;
  if (llvm::sys::fs::current_path(current))
    return inputPath.str();
  llvm::sys::path::append(current, inputPath);
  return current.str().str();
}

} // namespace lython::driver
