# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import func as func_ops


class StmtExceptionMixin:
    """Statement lowering for return and exception-related AST nodes."""

    def visit_Return(self, node: ast.Return) -> None:
        loc = self._loc(node)
        if node.value is None:
            if self._in_native_func:
                raise ValueError("@native function cannot return None implicitly")
            with loc, self.insertion_point():
                value = py_ops.NoneOp(self.get_py_type("!py.none")).result
        else:
            value = self.require_value(node.value, self.visit(node.value))

        with loc, self.insertion_point():
            expected_return = self.current_return_type()
            if expected_return is not None:
                value = self.coerce_value_to_type(value, expected_return, loc)
                if node.value is not None and str(expected_return).startswith(
                    "!py.func<"
                ):
                    returned_function_info = self.resolve_function_info_from_expression(
                        node.value, value
                    )
                    returned_callable_arg_index = (
                        self.resolve_current_function_parameter_index_from_expression(
                            node.value
                        )
                    )
                    if returned_callable_arg_index is not None:
                        returned_function_info = None
                    self._record_returned_callable_summary(
                        returned_function_info,
                        returned_callable_arg_index,
                    )

            if self._in_native_func:
                func_ops.ReturnOp([value])
            else:
                py_ops.ReturnOp([value])
        self._advance_block_after_terminator()

    def visit_Raise(self, node: ast.Raise) -> None:
        if self._in_native_func:
            raise NotImplementedError("raise in @native is not supported")

        with self._loc(node), self.insertion_point():
            self._note_maythrow()
            if node.exc is None:
                py_ops.RaiseCurrentOp()
                self._advance_block_after_terminator()
                return

            if node.cause is not None:
                raise NotImplementedError("raise ... from ... is not supported")

            if isinstance(node.exc, ast.Constant) and isinstance(node.exc.value, str):
                exc_value = self.build_exception_value(
                    message=node.exc.value, loc=self._loc(node)
                )
                py_ops.RaiseOp(exc_value)
                self._advance_block_after_terminator()
                return

            if isinstance(node.exc, ast.Call):
                if (
                    isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id == "Exception"
                ):
                    if len(node.exc.args) != 1 or not isinstance(
                        node.exc.args[0], ast.Constant
                    ):
                        raise NotImplementedError(
                            "Exception(...) requires a single string literal"
                        )
                    if not isinstance(node.exc.args[0].value, str):
                        raise NotImplementedError(
                            "Exception(...) requires a string literal"
                        )
                    exc_value = self.build_exception_value(
                        message=node.exc.args[0].value, loc=self._loc(node)
                    )
                    py_ops.RaiseOp(exc_value)
                    self._advance_block_after_terminator()
                    return

            raise NotImplementedError(
                "raise requires a string literal or bare raise (for now)"
            )

    def visit_Try(self, node: ast.Try) -> None:
        raise NotImplementedError("Try statement not implemented")

    def visit_TryStar(self, node: ast.TryStar) -> None:
        raise NotImplementedError("Try star statement not implemented")

    def visit_Assert(self, node: ast.Assert) -> None:
        raise NotImplementedError("Assert statement not implemented")
