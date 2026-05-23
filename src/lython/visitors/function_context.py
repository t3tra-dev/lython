from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..frontend.symbols import FunctionInfo
from ..mlir import ir

if TYPE_CHECKING:
    from .contracts import VisitorRuntime
else:
    VisitorRuntime = object


class FunctionContextMixin(VisitorRuntime):
    """Function, return, async, and exception context stacks for emission."""

    def current_function_name(self) -> str | None:
        if not self._function_name_stack:
            return None
        return self._function_name_stack[-1]

    def current_function_ast(
        self,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        if not self._function_ast_stack:
            return None
        return self._function_ast_stack[-1]

    def is_nested_function_context(self) -> bool:
        current = self.current_function_name()
        return current is not None and current != "main"

    def next_nested_function_symbol(self, lexical_name: str) -> str:
        self._nested_function_counter += 1
        for visitor in self.subvisitors.values():
            visitor._nested_function_counter = self._nested_function_counter
        parent = self.current_function_name() or self._module_name
        return f"{parent}.{lexical_name}${self._nested_function_counter}"

    def _enter_py_function(self, name: str) -> None:
        self._function_effect_stack.append(False)
        self._function_name_stack.append(name)
        self._returned_function_info_stack.append(None)
        self._returned_callable_arg_index_stack.append(None)
        self._returned_function_info_valid_stack.append(True)
        for visitor in self.subvisitors.values():
            visitor._function_effect_stack = self._function_effect_stack
            visitor._function_name_stack = self._function_name_stack
            visitor._returned_function_info_stack = self._returned_function_info_stack
            visitor._returned_callable_arg_index_stack = (
                self._returned_callable_arg_index_stack
            )
            visitor._returned_function_info_valid_stack = (
                self._returned_function_info_valid_stack
            )

    def _exit_py_function(self) -> tuple[bool, FunctionInfo | None, int | None]:
        if (
            not self._function_effect_stack
            or not self._function_name_stack
            or not self._returned_function_info_stack
            or not self._returned_callable_arg_index_stack
            or not self._returned_function_info_valid_stack
        ):
            raise RuntimeError("Function effect stack underflow")
        maythrow = self._function_effect_stack.pop()
        self._function_name_stack.pop()
        returned_info = self._returned_function_info_stack.pop()
        returned_arg_index = self._returned_callable_arg_index_stack.pop()
        returned_info_valid = self._returned_function_info_valid_stack.pop()
        for visitor in self.subvisitors.values():
            visitor._function_effect_stack = self._function_effect_stack
            visitor._function_name_stack = self._function_name_stack
            visitor._returned_function_info_stack = self._returned_function_info_stack
            visitor._returned_callable_arg_index_stack = (
                self._returned_callable_arg_index_stack
            )
            visitor._returned_function_info_valid_stack = (
                self._returned_function_info_valid_stack
            )
        return (
            maythrow,
            returned_info if returned_info_valid else None,
            returned_arg_index if returned_info_valid else None,
        )

    def push_function_ast(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        saved_finally_returns = (
            self._finally_return_stack if self._finally_return_stack else None
        )
        self._finally_return_stack_save.append(saved_finally_returns)
        if saved_finally_returns is not None:
            self._set_finally_return_stack([])
        saved_exception_context = (
            self._exception_context_stack if self._exception_context_stack else None
        )
        self._exception_context_stack_save.append(saved_exception_context)
        if saved_exception_context is not None:
            self._set_exception_context_stack([])
        self._function_ast_stack.append(node)
        for visitor in self.subvisitors.values():
            visitor._function_ast_stack = self._function_ast_stack

    def pop_function_ast(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        if not self._function_ast_stack:
            raise RuntimeError("Function AST stack underflow")
        node = self._function_ast_stack.pop()
        saved_finally_returns = self._finally_return_stack_save.pop()
        if saved_finally_returns is not None:
            self._set_finally_return_stack(saved_finally_returns)
        saved_exception_context = self._exception_context_stack_save.pop()
        if saved_exception_context is not None:
            self._set_exception_context_stack(saved_exception_context)
        for visitor in self.subvisitors.values():
            visitor._function_ast_stack = self._function_ast_stack
        return node

    def _note_maythrow(self) -> None:
        if not self._function_effect_stack:
            return
        self._function_effect_stack[-1] = True

    def push_return_type(self, return_type: ir.Type) -> None:
        self._return_type_stack.append(return_type)
        for visitor in self.subvisitors.values():
            visitor._return_type_stack = self._return_type_stack

    def pop_return_type(self) -> ir.Type:
        if not self._return_type_stack:
            raise RuntimeError("Return type stack underflow")
        return_type = self._return_type_stack.pop()
        for visitor in self.subvisitors.values():
            visitor._return_type_stack = self._return_type_stack
        return return_type

    def current_return_type(self) -> ir.Type | None:
        if not self._return_type_stack:
            return None
        return self._return_type_stack[-1]

    def push_async_function(self, is_async: bool) -> None:
        self._async_function_stack.append(is_async)
        for visitor in self.subvisitors.values():
            visitor._async_function_stack = self._async_function_stack

    def pop_async_function(self) -> bool:
        if not self._async_function_stack:
            raise RuntimeError("Async function stack underflow")
        value = self._async_function_stack.pop()
        for visitor in self.subvisitors.values():
            visitor._async_function_stack = self._async_function_stack
        return value

    def in_async_function(self) -> bool:
        return bool(self._async_function_stack and self._async_function_stack[-1])

    def get_awaitable_payload_type(self, awaitable_type: ir.Type) -> ir.Type | None:
        return self._type_resolver.awaitable_payload_type(awaitable_type)

    def push_exception_context(self, exception: ir.Value) -> None:
        self._exception_context_stack.append(exception)

    def pop_exception_context(self) -> ir.Value:
        if not self._exception_context_stack:
            raise RuntimeError("Exception context stack underflow")
        return self._exception_context_stack.pop()

    def current_exception_context(self) -> ir.Value | None:
        if not self._exception_context_stack:
            return None
        return self._exception_context_stack[-1]
