from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..mlir import ir
from .models import TypedProgram

if TYPE_CHECKING:
    from .contracts import VisitorRuntime
else:
    VisitorRuntime = object


class TypedOverlayMixin(VisitorRuntime):
    """Bridge from finalized frontend type analysis to visitor emission."""

    def _set_typed_program(self, typed_program: TypedProgram) -> None:
        self._typed_program = typed_program
        for visitor in self.subvisitors.values():
            visitor._typed_program = typed_program

    def typed_node_type(self, node: ast.AST) -> ir.Type:
        typed_program = self._typed_program
        if typed_program is None:
            raise TypeError(
                "Typed frontend analysis result is required before emission"
            )
        term = typed_program.type_of(node)
        return self._type_resolver.term_to_ir_type(term)

    def typed_node_type_or_none(self, node: ast.AST) -> ir.Type | None:
        return self.typed_node_type(node)
