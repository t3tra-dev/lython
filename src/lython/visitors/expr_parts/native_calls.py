from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...frontend.symbols import FunctionInfo
from ...mlir import ir
from ...mlir.dialects import func as func_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprNativeCallMixin(VisitorRuntime):
    """Calls to @native primitive functions."""

    def _handle_native_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value:
        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]

        if len(arg_values) != len(func_info.arg_types):
            raise ValueError(
                f"Native function '{func_info.symbol}' expects {len(func_info.arg_types)} "
                f"arguments, got {len(arg_values)}"
            )
        coerced_args = [
            self.coerce_value_to_type(arg, expected, loc)
            for arg, expected in zip(arg_values, func_info.arg_types)
        ]

        result_types = list(func_info.result_types)

        with loc, self.insertion_point():
            call = func_ops.CallOp(result_types, func_info.symbol, coerced_args)
            return call.results[0]
