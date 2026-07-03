#!/usr/bin/env python3

import argparse
import dataclasses
import re
from pathlib import Path


@dataclasses.dataclass
class ClassInfo:
    name: str
    attrs: str
    base_names: list[str]
    base_args: list[list[str]]
    params: list[str]
    variances: list[str]
    is_protocol: bool


def find_matching(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unmatched {open_char!r}")


def attr_value(attrs: str, key: str) -> str | None:
    match = re.search(rf"(?<![A-Za-z0-9_.]){re.escape(key)}(?![A-Za-z0-9_.])", attrs)
    if not match:
        return None
    index = match.end()
    while index < len(attrs) and attrs[index].isspace():
        index += 1
    if index >= len(attrs) or attrs[index] != "=":
        return None
    index += 1
    while index < len(attrs) and attrs[index].isspace():
        index += 1
    if index >= len(attrs):
        return None
    if attrs[index] == "[":
        end = find_matching(attrs, index, "[", "]")
        return attrs[index : end + 1]
    start = index
    angle_depth = 0
    in_string = False
    while index < len(attrs):
        char = attrs[index]
        if in_string:
            if char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char == "," and angle_depth == 0:
            break
        index += 1
    return attrs[start:index].strip()


def has_marker(attrs: str, key: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_.]){re.escape(key)}(?![A-Za-z0-9_.])", attrs))


def string_list(value: str | None) -> list[str]:
    if not value:
        return []
    return re.findall(r'"([^"]*)"', value)


def split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    square_depth = 0
    angle_depth = 0
    in_string = False
    for index, char in enumerate(value):
        if in_string:
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char == "," and square_depth == 0 and angle_depth == 0:
            part = value[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def base_arg_groups(value: str | None) -> list[list[str]]:
    if not value:
        return []
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError(f"invalid ly.typing.base_args: {value}")
    inner = value[1:-1]
    groups: list[list[str]] = []
    index = 0
    while index < len(inner):
        while index < len(inner) and (inner[index].isspace() or inner[index] == ","):
            index += 1
        if index >= len(inner):
            break
        if inner[index] != "[":
            raise ValueError(f"expected base argument group in {value}")
        end = find_matching(inner, index, "[", "]")
        group = inner[index + 1 : end].strip()
        groups.append(split_top_level(group) if group else [])
        index = end + 1
    return groups


def parse_classes(text: str) -> list[ClassInfo]:
    classes: list[ClassInfo] = []
    cursor = 0
    pattern = re.compile(r"\bpy\.class\s+@([A-Za-z_.$][A-Za-z0-9_.$]*)")
    while True:
        match = pattern.search(text, cursor)
        if not match:
            return classes
        name = match.group(1)
        attr_keyword = text.find("attributes", match.end())
        if attr_keyword < 0:
            cursor = match.end()
            continue
        attr_start = text.find("{", attr_keyword)
        if attr_start < 0:
            cursor = match.end()
            continue
        attr_end = find_matching(text, attr_start, "{", "}")
        attrs = text[attr_start + 1 : attr_end]
        classes.append(
            ClassInfo(
                name=name,
                attrs=attrs,
                base_names=string_list(attr_value(attrs, "base_names")),
                base_args=base_arg_groups(attr_value(attrs, "ly.typing.base_args")),
                params=string_list(attr_value(attrs, "ly.typing.params")),
                variances=string_list(attr_value(attrs, "ly.typing.param_variance")),
                is_protocol=has_marker(attrs, "ly.typing.protocol"),
            )
        )
        cursor = attr_end + 1


def projection_for_type(type_expr: str, params: list[str]) -> tuple[str, int]:
    normalized = re.sub(r"\s+", "", type_expr)
    type_var = re.fullmatch(r'!py\.(?:class|contract)<"\$([^"]+)">', normalized)
    if type_var:
        name = type_var.group(1)
        if name not in params:
            raise ValueError(f"unknown type parameter ${name}")
        return ("Argument", params.index(name))
    constants = {
        '!py.contract<"typing.Any">': "Object",
        '!py.contract<"builtins.object">': "Object",
        '!py.contract<"builtins.int">': "Int",
        '!py.contract<"builtins.float">': "Float",
        '!py.contract<"builtins.bool">': "Bool",
        '!py.contract<"builtins.str">': "Str",
        '!py.contract<"types.NoneType">': "None",
        '!py.literal<None>': "None",
        '!py.contract<"builtins.BaseException">': "Exception",
        '!py.contract<"types.TracebackType">': "Traceback",
        '!py.contract<"builtins.tuple">': "Object",
        '!py.contract<"builtins.list">': "Object",
        '!py.contract<"builtins.dict">': "Object",
        "!py.object": "Object",
        "!py.int": "Int",
        "!py.float": "Float",
        "!py.bool": "Bool",
        "!py.str": "Str",
        "!py.none": "None",
        "!py.exception": "Exception",
        "!py.traceback": "Traceback",
    }
    if normalized in constants:
        return (constants[normalized], 0)
    raise ValueError(f"unsupported protocol base argument: {type_expr}")


def cpp_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit_variance_rules(classes: list[ClassInfo]) -> str:
    mapping = {"covariant": "C", "contravariant": "X", "invariant": "I"}
    entries: list[str] = [
        '{"Callable", {X, C, C, C, C, C, C, C}, 2, true}',
    ]
    for info in classes:
        if not info.is_protocol or not info.variances:
            continue
        values = [mapping[v] for v in info.variances]
        padded = values + ["C"] * (8 - len(values))
        entries.append(
            f'      {{{cpp_string(info.name)}, {{{", ".join(padded)}}}, {len(values)}, false}}'
        )
    return ",\n".join("      " + e if not e.startswith("      ") else e for e in entries)


def emit_base_tables(classes: list[ClassInfo]) -> tuple[str, str]:
    protocols = {info.name for info in classes if info.is_protocol}
    projections: list[tuple[str, list[tuple[str, int]]]] = []
    rules: list[tuple[str, int, int, int]] = []
    for info in classes:
        first = len(projections)
        for index, base in enumerate(info.base_names):
            if base not in protocols:
                continue
            groups = info.base_args[index] if index < len(info.base_args) else []
            projections.append((base, [projection_for_type(arg, info.params) for arg in groups]))
        count = len(projections) - first
        if count:
            rules.append((info.name, len(info.params), first, count))

    projection_rows: list[str] = []
    for name, args in projections:
        padded = args + [("Argument", 0)] * (3 - len(args))
        fields = ", ".join(f"{{ProjectionKind::{kind}, {index}}}" for kind, index in padded)
        projection_rows.append(f"      {{{cpp_string(name)}, {{{{{fields}}}}}, {len(args)}}}")

    rule_rows = [
        f"      {{{cpp_string(name)}, {arity}, {first}, {count}}}"
        for name, arity, first, count in rules
    ]
    return ",\n".join(projection_rows), ",\n".join(rule_rows)


def generate(text: str, sources: list[Path]) -> str:
    classes = parse_classes(text)
    variance_rules = emit_variance_rules(classes)
    projections, rules = emit_base_tables(classes)
    source_comment = ", ".join(str(source) for source in sources)
    return f"""#pragma once

// Generated from {source_comment}. Do not edit.

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>

namespace py::protocol_relations {{

enum class Variance {{ Covariant, Contravariant, Invariant }};

struct VarianceRule {{
  llvm::StringLiteral protocolName;
  std::array<Variance, 8> variance;
  unsigned count = 0;
  bool repeatLast = false;
}};

inline Variance parameterVariance(llvm::StringRef protocolName,
                                  unsigned index) {{
  static constexpr Variance C = Variance::Covariant;
  static constexpr Variance X = Variance::Contravariant;
  static constexpr Variance I = Variance::Invariant;
  static constexpr VarianceRule rules[] = {{
{variance_rules}
  }};
  for (const VarianceRule &rule : rules) {{
    if (protocolName != rule.protocolName)
      continue;
    if (index < rule.count)
      return rule.variance[index];
    if (rule.repeatLast && rule.count > 0)
      return rule.variance[rule.count - 1];
    return Variance::Covariant;
  }}
  return Variance::Covariant;
}}

enum class ProjectionKind {{
  Argument,
  Object,
  Int,
  Float,
  Bool,
  Str,
  None,
  Exception,
  Traceback
}};

struct BaseArgumentProjection {{
  ProjectionKind kind = ProjectionKind::Argument;
  unsigned index = 0;
}};

struct BaseProjection {{
  llvm::StringLiteral protocolName;
  std::array<BaseArgumentProjection, 3> arguments;
  unsigned argumentCount = 0;
}};

struct DirectBaseRule {{
  llvm::StringLiteral protocolName;
  unsigned arity = 0;
  unsigned firstProjection = 0;
  unsigned projectionCount = 0;
}};

inline mlir::Type projectedArgument(mlir::MLIRContext *ctx,
                                    BaseArgumentProjection projection,
                                    llvm::ArrayRef<mlir::Type> args) {{
  switch (projection.kind) {{
  case ProjectionKind::Argument:
    if (projection.index >= args.size())
      return {{}};
    return args[projection.index];
  case ProjectionKind::Object:
    return py::ContractType::get(ctx, "builtins.object");
  case ProjectionKind::Int:
    return py::ContractType::get(ctx, "builtins.int");
  case ProjectionKind::Float:
    return py::ContractType::get(ctx, "builtins.float");
  case ProjectionKind::Bool:
    return py::ContractType::get(ctx, "builtins.bool");
  case ProjectionKind::Str:
    return py::ContractType::get(ctx, "builtins.str");
  case ProjectionKind::None:
    return py::LiteralType::get(ctx, "None");
  case ProjectionKind::Exception:
    return py::ContractType::get(ctx, "builtins.BaseException");
  case ProjectionKind::Traceback:
    return py::ContractType::get(ctx, "types.TracebackType");
  }}
  return {{}};
}}

template <typename EmitBase>
void forEachDirectBase(llvm::StringRef name, mlir::MLIRContext *ctx,
                       llvm::ArrayRef<mlir::Type> args,
                       EmitBase emit) {{
  static constexpr BaseProjection projections[] = {{
{projections}
  }};
  static constexpr DirectBaseRule rules[] = {{
{rules}
  }};

  for (const DirectBaseRule &rule : rules) {{
    if (name != rule.protocolName || args.size() != rule.arity)
      continue;
    for (const BaseProjection &projection :
         llvm::ArrayRef<BaseProjection>(projections)
             .slice(rule.firstProjection, rule.projectionCount)) {{
      llvm::SmallVector<mlir::Type, 3> baseArgs;
      baseArgs.reserve(projection.argumentCount);
      llvm::ArrayRef<BaseArgumentProjection> projected(
          projection.arguments.data(), projection.argumentCount);
      for (BaseArgumentProjection argument : projected) {{
        mlir::Type type = projectedArgument(ctx, argument, args);
        if (!type)
          return;
        baseArgs.push_back(type);
      }}
      emit(projection.protocolName, baseArgs);
    }}
    return;
  }}
}}

}} // namespace py::protocol_relations
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = [Path(path) for path in args.input]
    text = "\n".join(path.read_text() for path in inputs)
    Path(args.output).write_text(generate(text, inputs))


if __name__ == "__main__":
    main()
