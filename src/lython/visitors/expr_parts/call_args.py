from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprCallArgsMixin(VisitorRuntime):
    """Helpers for packing positional and keyword call operands."""

    def _build_direct_vectorcall_operands(
        self,
        *,
        positional_args: list[ir.Value],
        keywords: list[ast.keyword],
        keyword_values: dict[str, ir.Value] | None = None,
        positional_param_types: tuple[ir.Type, ...] | list[ir.Type],
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_param_types: tuple[ir.Type, ...] | list[ir.Type],
        kwonly_names: tuple[str, ...] | list[str],
        kwdefault_names: tuple[str, ...] | list[str],
        defaults_count: int,
        loc: ir.Location,
        leading_args: list[ir.Value] | None = None,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        positional_param_types = list(positional_param_types)
        positional_param_names = list(positional_param_names)
        kwonly_param_types = list(kwonly_param_types)
        kwonly_names = list(kwonly_names)
        kwdefault_names = list(kwdefault_names)
        leading_args = list(leading_args or [])

        if len(positional_args) > len(positional_param_types):
            raise ValueError(
                f"Expected at most {len(positional_param_types)} positional "
                f"arguments, got {len(positional_args)}"
            )

        coerced_positional = [
            self.coerce_value_to_type(arg, expected, loc)
            for arg, expected in zip(positional_args, positional_param_types)
        ]
        posargs = self.build_tuple(leading_args + coerced_positional, loc=loc)

        empty_tuple_type = self.get_py_type("!py.tuple<>")
        if not keywords:
            min_positional_count = len(positional_param_types) - defaults_count
            if len(positional_args) < min_positional_count:
                raise ValueError(
                    f"Expected at least {min_positional_count} positional "
                    f"arguments, got {len(positional_args)}"
                )
            with loc, self.insertion_point():
                return (
                    posargs,
                    py_ops.TupleEmptyOp(empty_tuple_type).result,
                    py_ops.TupleEmptyOp(empty_tuple_type).result,
                )

        positional_types = {
            name: (index, ty)
            for index, (name, ty) in enumerate(
                zip(positional_param_names, positional_param_types)
            )
        }
        kwonly_types = {name: ty for name, ty in zip(kwonly_names, kwonly_param_types)}
        kwdefault_name_set = set(kwdefault_names)
        seen_keywords: set[str] = set()
        kwname_values: list[ir.Value] = []
        kwvalue_values: list[ir.Value] = []

        with loc, self.insertion_point():
            for keyword in keywords:
                if keyword.arg is None:
                    raise NotImplementedError(
                        "Keyword argument unpacking is not supported yet"
                    )
                if keyword.arg in seen_keywords:
                    raise ValueError(f"Duplicate keyword argument '{keyword.arg}'")
                seen_keywords.add(keyword.arg)
                if keyword.arg in positional_types:
                    slot_index, expected_type = positional_types[keyword.arg]
                    if slot_index < len(positional_args):
                        raise ValueError(
                            f"Argument '{keyword.arg}' was provided both "
                            "positionally and by keyword"
                        )
                elif keyword.arg in kwonly_types:
                    expected_type = kwonly_types[keyword.arg]
                else:
                    raise NotImplementedError(
                        f"Keyword argument '{keyword.arg}' is not supported"
                    )
                if keyword_values is not None and keyword.arg in keyword_values:
                    value = keyword_values[keyword.arg]
                else:
                    value = self.require_value(keyword.value, self.visit(keyword.value))
                value = self.coerce_value_to_type(
                    value, expected_type, self._loc(keyword.value)
                )
                kwname_values.append(
                    py_ops.StrConstantOp(
                        self.get_py_type("!py.str"),
                        ir.StringAttr.get(keyword.arg, self.ctx),
                    ).result
                )
                kwvalue_values.append(value)

        required_positional_count = len(positional_param_types) - defaults_count
        for index, name in enumerate(positional_param_names):
            if index < len(positional_args):
                continue
            if name in seen_keywords:
                continue
            if index < required_positional_count:
                raise ValueError(f"Missing required argument '{name}'")

        for name in kwonly_names:
            if name in seen_keywords or name in kwdefault_name_set:
                continue
            raise ValueError(f"Missing required keyword-only argument '{name}'")

        return (
            posargs,
            self.build_tuple(kwname_values, loc=loc),
            self.build_tuple(kwvalue_values, loc=loc),
        )
