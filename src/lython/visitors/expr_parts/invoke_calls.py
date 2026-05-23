from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ...frontend.symbols import FunctionInfo
from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ..models import RegionBlocks

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprInvokeMixin(VisitorRuntime):
    """Helpers for py.invoke result seeds and control-flow emission."""

    def _split_type_specs(self, specs: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in specs:
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
            elif ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)
        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    def _build_invoke_result_seed(
        self, result_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        type_str = str(result_type)

        if type_str == "!py.none":
            with loc, self.insertion_point():
                return py_ops.NoneOp(result_type).result
        if type_str == "!py.exception":
            with loc, self.insertion_point():
                return py_ops.ExceptionNullOp(result_type).result
        if type_str == "!py.traceback":
            with loc, self.insertion_point():
                return py_ops.TracebackNullOp(result_type).result
        if type_str == "!py.location":
            with loc, self.insertion_point():
                return py_ops.LocationCurrentOp(result_type).result
        if type_str == "!py.object":
            with loc, self.insertion_point():
                none_val = py_ops.NoneOp(self.get_py_type("!py.none")).result
            return self.ensure_object(none_val, loc=loc)
        if type_str in {"!py.int", "!py.bool"}:
            with loc, self.insertion_point():
                return py_ops.IntConstantOp(result_type, "0").result
        if type_str == "!py.float":
            with loc, self.insertion_point():
                return py_ops.FloatConstantOp(result_type, 0.0).result
        if type_str == "!py.str":
            with loc, self.insertion_point():
                return py_ops.StrConstantOp(
                    result_type, ir.StringAttr.get("", self.ctx)
                ).result
        if type_str == "!py.tuple<>":
            with loc, self.insertion_point():
                return py_ops.TupleEmptyOp(result_type).result
        if type_str.startswith("!py.tuple<") and type_str.endswith(">"):
            inner = type_str[len("!py.tuple<") : -1].strip()
            if not inner:
                with loc, self.insertion_point():
                    return py_ops.TupleEmptyOp(result_type).result
            elements = [
                self._build_invoke_result_seed(self.get_py_type(spec), loc)
                for spec in self._split_type_specs(inner)
            ]
            return self.build_tuple(elements, loc=loc)
        if type_str.startswith("!py.dict<"):
            with loc, self.insertion_point():
                return py_ops.DictEmptyOp(result_type).result
        if type_str.startswith("!py.list<"):
            with loc, self.insertion_point():
                return py_ops.ListNewOp(result_type).result
        if type_str.startswith('!py.class<"') and type_str.endswith('">'):
            class_name = type_str[len('!py.class<"') : -len('">')]
            with loc, self.insertion_point():
                return py_ops.ClassNewOp(result_type, class_name).result
        if type_str.startswith("i"):
            return self._build_primitive_scalar(0, result_type, loc)
        if type_str.startswith("f"):
            return self._build_primitive_scalar(0.0, result_type, loc)

        raise NotImplementedError(
            f"py.invoke result placeholder for {result_type} is not implemented yet"
        )

    def _emit_none_returning_invoke(
        self,
        callee: ir.Value,
        posargs: ir.Value,
        kwnames: ir.Value,
        kwvalues: ir.Value,
        loc: ir.Location,
    ) -> ir.Block:
        with loc:
            if self.current_block is None:
                raise RuntimeError("Insertion block is not set")
            parent_region = self.current_block.region
            blocks = cast(RegionBlocks, parent_region.blocks)
            normal_block = blocks.append()
            unwind_block = blocks.append(self.get_py_type("!py.exception"))
        with loc, self.insertion_point():
            exc_null = py_ops.ExceptionNullOp(self.get_py_type("!py.exception")).result
            py_ops.InvokeOp(
                callee,
                posargs,
                kwnames,
                kwvalues,
                [],
                [exc_null],
                normal_block,
                unwind_block,
            )
        with ir.InsertionPoint(unwind_block), loc:
            py_ops.RaiseCurrentOp()
        return normal_block

    def _emit_value_returning_invoke(
        self,
        callee: ir.Value,
        posargs: ir.Value,
        kwnames: ir.Value,
        kwvalues: ir.Value,
        result_type: ir.Type,
        loc: ir.Location,
        returned_function_info: FunctionInfo | None = None,
    ) -> ir.Value:
        with loc:
            if self.current_block is None:
                raise RuntimeError("Insertion block is not set")
            parent_region = self.current_block.region
            blocks = cast(RegionBlocks, parent_region.blocks)
            normal_block = blocks.append(result_type)
            unwind_block = blocks.append(self.get_py_type("!py.exception"))
        with loc, self.insertion_point():
            if returned_function_info is not None and str(result_type).startswith(
                "!py.func<"
            ):
                seed = py_ops.FuncObjectOp(
                    result_type,
                    ir.FlatSymbolRefAttr.get(
                        returned_function_info.symbol,
                        self.ctx,
                    ),
                ).result
            else:
                seed = self._build_invoke_result_seed(result_type, loc)
            exc_null = py_ops.ExceptionNullOp(self.get_py_type("!py.exception")).result
            py_ops.InvokeOp(
                callee,
                posargs,
                kwnames,
                kwvalues,
                [seed],
                [exc_null],
                normal_block,
                unwind_block,
            )
        with ir.InsertionPoint(unwind_block), loc:
            py_ops.RaiseCurrentOp()
        self._set_insertion_block(normal_block)
        result = normal_block.arguments[0]
        if returned_function_info is not None and str(result_type).startswith(
            "!py.func<"
        ):
            result = self._materialize_known_callable_result(
                result, returned_function_info, loc
            )
        return result
