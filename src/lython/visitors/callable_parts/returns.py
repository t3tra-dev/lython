from __future__ import annotations

from typing import TYPE_CHECKING

from ...frontend.symbols import FunctionInfo

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class CallableReturnSummaryMixin(VisitorRuntime):
    """Track whether a function consistently returns a known callable."""

    def _record_returned_callable_summary(
        self, info: FunctionInfo | None, arg_index: int | None
    ) -> None:
        if (
            not self._returned_function_info_stack
            or not self._returned_callable_arg_index_stack
            or not self._returned_function_info_valid_stack
        ):
            return
        if not self._returned_function_info_valid_stack[-1]:
            return
        if info is None and arg_index is None:
            self._returned_function_info_stack[-1] = None
            self._returned_callable_arg_index_stack[-1] = None
            self._returned_function_info_valid_stack[-1] = False
            return
        current = self._returned_function_info_stack[-1]
        current_arg_index = self._returned_callable_arg_index_stack[-1]
        if current is None and current_arg_index is None:
            self._returned_function_info_stack[-1] = info
            self._returned_callable_arg_index_stack[-1] = arg_index
            return
        current_symbol = current.symbol if current is not None else None
        new_symbol = info.symbol if info is not None else None
        if current_symbol != new_symbol or current_arg_index != arg_index:
            self._returned_function_info_stack[-1] = None
            self._returned_callable_arg_index_stack[-1] = None
            self._returned_function_info_valid_stack[-1] = False
