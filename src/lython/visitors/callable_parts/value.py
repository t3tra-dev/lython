from __future__ import annotations

from typing import TYPE_CHECKING

from ...frontend.symbols import FunctionInfo
from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ..mlir_access import (
    op_name,
    op_operand_segment_sizes,
    op_operands,
    value_operation,
)

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class CallableValueMixin(VisitorRuntime):
    """Callable value attributes, cloning, and materialization."""

    def maybe_define_callable_parameter_binding(
        self, name: str, type_spec: str, value: ir.Value
    ) -> FunctionInfo | None:
        if not type_spec.startswith("!py.func<"):
            return None
        info = self.build_function_info_from_callable_type(
            name, value.type, maythrow=True
        )
        if info is not None:
            self.define_function_binding(name, info)
        return info

    def annotate_known_callable_value(
        self,
        value: ir.Value,
        info: FunctionInfo | None,
        *,
        loc: ir.Location,
    ) -> ir.Value:
        if info is None:
            return value
        self._callable_value_info[id(value)] = info
        op = self._value_operation(value)
        if op is None:
            return value
        self._attach_returned_callable_info(op, info)
        return value

    def _attach_returned_callable_metadata(
        self,
        op: ir.Operation,
        func_info: FunctionInfo,
    ) -> None:
        self._attach_returned_callable_info(op, func_info.returned_function_info)

    def _attach_returned_callable_info(
        self,
        op: ir.Operation,
        returned: FunctionInfo | None,
    ) -> None:
        if returned is None:
            return
        op.attributes["ly.returned_callable_symbol"] = ir.FlatSymbolRefAttr.get(
            returned.symbol, self.ctx
        )
        op.attributes["ly.returned_callable_defaults_count"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64, context=self.ctx),
            returned.defaults_count,
        )
        if returned.kwdefault_names:
            op.attributes["ly.returned_callable_kwdefault_names"] = self.array_attr(
                [
                    ir.StringAttr.get(name, self.ctx)
                    for name in returned.kwdefault_names
                ],
            )

    def _value_operation(self, value: ir.Value) -> ir.Operation | None:
        return value_operation(value)

    def _extract_make_function_operands(self, op: ir.Operation) -> tuple[
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
    ]:
        operands = op_operands(op)
        segment_sizes = op_operand_segment_sizes(op)
        if not segment_sizes:
            return (None, None, None, None, None)
        segment_index = 0
        operand_index = 0

        def take_optional() -> ir.Value | None:
            nonlocal segment_index, operand_index
            if segment_index >= len(segment_sizes):
                return None
            size = segment_sizes[segment_index]
            result = operands[operand_index] if size else None
            segment_index += 1
            operand_index += size
            return result

        return (
            take_optional(),
            take_optional(),
            take_optional(),
            take_optional(),
            take_optional(),
        )

    def _materialize_known_callable_result(
        self,
        value: ir.Value,
        info: FunctionInfo | None,
        loc: ir.Location,
    ) -> ir.Value:
        if info is None or not str(value.type).startswith("!py.func<"):
            return value
        return self._build_python_callable(
            symbol=info.symbol,
            func_type=value.type,
            defaults=info.defaults,
            kwdefaults=info.kwdefaults,
            closure=info.closure,
            loc=loc,
            force_make_function=bool(
                info.defaults is not None
                or info.kwdefaults is not None
                or info.closure is not None
            ),
        )

    def _build_python_callable(
        self,
        *,
        symbol: str,
        func_type: ir.Type,
        defaults: ir.Value | None,
        kwdefaults: ir.Value | None,
        closure: ir.Value | None,
        loc: ir.Location,
        force_make_function: bool = False,
    ) -> ir.Value:
        with loc, self.insertion_point():
            if (
                not force_make_function
                and defaults is None
                and kwdefaults is None
                and closure is None
            ):
                return py_ops.FuncObjectOp(func_type, symbol).result
            return py_ops.MakeFunctionOp(
                func_type,
                ir.FlatSymbolRefAttr.get(symbol, self.ctx),
                defaults=(
                    self._clone_bound_callable_metadata(defaults, loc)
                    if defaults is not None
                    else None
                ),
                kwdefaults=(
                    self._clone_bound_callable_metadata(kwdefaults, loc)
                    if kwdefaults is not None
                    else None
                ),
                closure=(
                    self._clone_bound_callable_metadata(closure, loc)
                    if closure is not None
                    else None
                ),
            ).result

    def _build_method_callable(
        self,
        *,
        symbol: str,
        func_type: ir.Type,
        defaults: ir.Value | None,
        kwdefaults: ir.Value | None,
        loc: ir.Location,
        force_make_function: bool = False,
    ) -> ir.Value:
        return self._build_python_callable(
            symbol=symbol,
            func_type=func_type,
            defaults=defaults,
            kwdefaults=kwdefaults,
            closure=None,
            loc=loc,
            force_make_function=force_make_function,
        )

    def _needs_keyword_callable_materialization(self, value: ir.Value) -> bool:
        current = value
        while True:
            op = self._value_operation(current)
            if op is None:
                return False
            name = op_name(op)
            if name == "py.publish":
                operands = op_operands(op)
                if not operands:
                    return False
                current = operands[0]
                continue
            return name == "py.func.object"
