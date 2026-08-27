#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace py {

// One traceback frame the emitter INLINED away: the call that brought control
// into a method body written into its caller, and the name of the function that
// call is written in. An empty `functionName` means the enclosing LLVM
// function, which only the lowering can name.
struct PythonInlineFrame {
  std::string functionName;
  std::int32_t line = 0;
  std::int32_t column = 0;
  std::int32_t endLine = 0;
  std::int32_t endColumn = 0;
  bool noAnchor = false;
};

struct PythonSourceRange {
  std::string filename;
  std::int32_t line = 0;
  std::int32_t column = 0;
  std::int32_t endLine = 0;
  std::int32_t endColumn = 0;
  // CPython draws no `~~~^^^` under this range; the emitter decided it from the
  // statement's AST (EmitterStatements.cpp, `anchorlessCallOf`).
  bool noAnchor = false;
  // The function this range is written in, when that is NOT the enclosing LLVM
  // function -- i.e. when the emitter inlined a method body here.
  std::string functionName;
  // The call sites this range is nested inside, innermost first. Each is a
  // traceback frame CPython would show and an inlined body cannot produce.
  std::vector<PythonInlineFrame> inlinedAt;
};

inline std::optional<mlir::FileLineColLoc>
findPythonSourceLoc(mlir::Location loc) {
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    if (fileLoc.getFilename().getValue().ends_with(".py"))
      return fileLoc;
    return std::nullopt;
  }
  if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc))
    return findPythonSourceLoc(nameLoc.getChildLoc());
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location child : fused.getLocations())
      if (auto found = findPythonSourceLoc(child))
        return found;
  }
  return std::nullopt;
}

inline std::optional<std::int32_t>
pythonSourceI32Attr(mlir::DictionaryAttr dict, llvm::StringRef name) {
  auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(dict.get(name));
  if (!attr)
    return std::nullopt;
  return static_cast<std::int32_t>(attr.getInt());
}

inline std::optional<PythonSourceRange>
sourceRangeFromDict(mlir::DictionaryAttr dict) {
  auto startLine = pythonSourceI32Attr(dict, "ly.source.start_line");
  auto startCol = pythonSourceI32Attr(dict, "ly.source.start_col");
  auto endLine = pythonSourceI32Attr(dict, "ly.source.end_line");
  auto endCol = pythonSourceI32Attr(dict, "ly.source.end_col");
  if (!startLine || !startCol || !endLine || !endCol)
    return std::nullopt;
  PythonSourceRange range;
  range.line = *startLine;
  range.column = *startCol;
  range.endLine = *endLine;
  range.endColumn = *endCol;
  range.noAnchor = dict.get("ly.source.no_anchor") != nullptr;
  if (auto function =
          mlir::dyn_cast_or_null<mlir::StringAttr>(dict.get("ly.source.function")))
    range.functionName = function.getValue().str();
  if (auto frames =
          mlir::dyn_cast_or_null<mlir::ArrayAttr>(dict.get("ly.source.inline_at")))
    for (mlir::Attribute entry : frames) {
      auto frameDict = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
      if (!frameDict)
        continue;
      PythonInlineFrame frame;
      if (auto function = mlir::dyn_cast_or_null<mlir::StringAttr>(
              frameDict.get("function")))
        frame.functionName = function.getValue().str();
      frame.line = pythonSourceI32Attr(frameDict, "start_line").value_or(0);
      frame.column = pythonSourceI32Attr(frameDict, "start_col").value_or(0);
      frame.endLine = pythonSourceI32Attr(frameDict, "end_line").value_or(0);
      frame.endColumn = pythonSourceI32Attr(frameDict, "end_col").value_or(0);
      frame.noAnchor = frameDict.get("no_anchor") != nullptr;
      range.inlinedAt.push_back(std::move(frame));
    }
  return range;
}

inline std::optional<PythonSourceRange>
findSourceRangeMetadata(mlir::Location loc) {
  if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc))
    return findSourceRangeMetadata(nameLoc.getChildLoc());
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    if (auto dict =
            mlir::dyn_cast_or_null<mlir::DictionaryAttr>(fused.getMetadata()))
      if (auto range = sourceRangeFromDict(dict))
        return range;
    for (mlir::Location child : fused.getLocations())
      if (auto range = findSourceRangeMetadata(child))
        return range;
  }
  return std::nullopt;
}

inline std::optional<PythonSourceRange> pythonSourceRange(mlir::Location loc) {
  std::optional<mlir::FileLineColLoc> fileLoc = findPythonSourceLoc(loc);
  if (!fileLoc)
    return std::nullopt;

  PythonSourceRange range;
  range.filename = fileLoc->getFilename().getValue().str();
  range.line = static_cast<std::int32_t>(fileLoc->getLine());
  range.column = static_cast<std::int32_t>(fileLoc->getColumn());
  range.endLine = range.line;
  range.endColumn = range.column;
  if (auto metadata = findSourceRangeMetadata(loc)) {
    metadata->filename = range.filename;
    return metadata;
  }
  return range;
}

} // namespace py
