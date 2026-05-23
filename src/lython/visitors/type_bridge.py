from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..frontend.symbols import FunctionInfo
from ..mlir import ir

if TYPE_CHECKING:
    from .contracts import VisitorRuntime
else:
    VisitorRuntime = object


class TypeBridgeMixin(VisitorRuntime):
    """Bridge AST annotations and resolver facts to concrete MLIR types."""

    def annotation_to_static_class_type(self, annotation: ast.expr | None) -> ir.Type:
        return self._type_resolver.annotation_to_static_type(annotation)

    def annotation_to_py_type(self, annotation: ast.expr | None) -> str:
        return self._type_resolver.annotation_to_py_type(annotation)

    def build_funcsig(
        self,
        arg_types: list[str],
        result_types: list[str],
        *,
        kwonly_types: list[str] | None = None,
        vararg_type: str | None = None,
        kwargs_type: str | None = None,
    ) -> str:
        return self._type_resolver.build_funcsig(
            arg_types,
            result_types,
            kwonly_types=kwonly_types,
            vararg_type=vararg_type,
            kwargs_type=kwargs_type,
        )

    def build_function_info_from_callable_type(
        self,
        name: str,
        func_type: ir.Type,
        *,
        maythrow: bool = True,
    ) -> FunctionInfo | None:
        return self._type_resolver.build_function_info_from_callable_type(
            name,
            func_type,
            maythrow=maythrow,
        )
