#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py {

struct PythonSourceRange {
  std::string filename;
  std::int32_t line = 0;
  std::int32_t column = 0;
  std::int32_t endLine = 0;
  std::int32_t endColumn = 0;
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
  auto startLine = pythonSourceI32Attr(dict, "lython.source.start_line");
  auto startCol = pythonSourceI32Attr(dict, "lython.source.start_col");
  auto endLine = pythonSourceI32Attr(dict, "lython.source.end_line");
  auto endCol = pythonSourceI32Attr(dict, "lython.source.end_col");
  if (!startLine || !startCol || !endLine || !endCol)
    return std::nullopt;
  PythonSourceRange range;
  range.line = *startLine;
  range.column = *startCol;
  range.endLine = *endLine;
  range.endColumn = *endCol;
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

inline bool sourcePointLessOrEqual(std::int32_t lhsLine, std::int32_t lhsColumn,
                                   std::int32_t rhsLine,
                                   std::int32_t rhsColumn) {
  return lhsLine < rhsLine || (lhsLine == rhsLine && lhsColumn <= rhsColumn);
}

inline bool sourceRangeContains(const PythonSourceRange &outer,
                                const PythonSourceRange &inner) {
  if (outer.filename != inner.filename)
    return false;
  if (outer.line <= 0 || inner.line <= 0 || outer.endLine <= 0 ||
      inner.endLine <= 0)
    return false;
  return sourcePointLessOrEqual(outer.line, outer.column, inner.line,
                                inner.column) &&
         sourcePointLessOrEqual(inner.endLine, inner.endColumn, outer.endLine,
                                outer.endColumn);
}

} // namespace py
