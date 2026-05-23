from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprContainerMixin(VisitorRuntime):
    """Expression lowering for typed list/dict/tuple containers."""

    def visit_Dict(self, node: ast.Dict) -> ir.Value:
        if not node.keys:
            raise NotImplementedError("Empty dict literal is not supported yet")

        entries: list[tuple[ir.Value, ir.Value]] = []
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise NotImplementedError("Dict unpacking is not supported yet")
            key = self.require_value(key_node, self.visit(key_node))
            value = self.require_value(value_node, self.visit(value_node))
            entries.append((key, value))

        key_type = entries[0][0].type
        value_type = entries[0][1].type
        dict_type = self.get_py_type(f"!py.dict<{key_type}, {value_type}>")

        with self._loc(node), self.insertion_point():
            result = py_ops.DictEmptyOp(dict_type).result

        for key, value in entries:
            if key.type != key_type:
                key = self.coerce_value_to_type(key, key_type, self._loc(node))
            if value.type != value_type:
                value = self.coerce_value_to_type(value, value_type, self._loc(node))
            with self._loc(node), self.insertion_point():
                py_ops.DictInsertOp(result, key, value)
        return result

    def visit_Subscript(self, node: ast.Subscript) -> ir.Value:
        container = self.require_value(node.value, self.visit(node.value))
        result_type = self.typed_node_type(node)
        element_type = self.get_list_element_type(container.type)
        if element_type is not None:
            index = self.require_value(node.slice, self.visit(node.slice))
            with self._loc(node), self.insertion_point():
                return py_ops.ListGetOp(result_type, container, index).result

        dict_types = self.get_dict_key_value_types(container.type)
        if dict_types is not None:
            key_type, _ = dict_types
            key = self.require_value(node.slice, self.visit(node.slice))
            key = self.coerce_value_to_type(key, key_type, self._loc(node))
            with self._loc(node), self.insertion_point():
                return py_ops.DictGetOp(result_type, container, key).result

        raise NotImplementedError(
            f"Subscript access is only supported on !py.list or !py.dict values, got {container.type}"
        )

    def visit_List(self, node: ast.List) -> ir.Value:
        if not node.elts:
            raise NotImplementedError("Empty list literal is not supported yet")

        values = [self.require_value(elt, self.visit(elt)) for elt in node.elts]
        element_type = values[0].type
        list_type = self.get_py_type(f"!py.list<{element_type}>")

        with self._loc(node), self.insertion_point():
            result = py_ops.ListNewOp(list_type).result

        for value in values:
            prepared = self._prepare_list_element_for_storage(
                value, element_type, self._loc(node)
            )
            with self._loc(node), self.insertion_point():
                py_ops.ListAppendOp(result, prepared)
        return result

    def visit_Tuple(self, node: ast.Tuple) -> ir.Value:
        if not isinstance(node.ctx, ast.Load):
            raise NotImplementedError("Tuple assignment target is not implemented")
        values = [self.require_value(elt, self.visit(elt)) for elt in node.elts]
        return self.build_tuple(values, loc=self._loc(node))

    def _prepare_list_element_for_storage(
        self, value: ir.Value, element_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        if value.type != element_type:
            value = self.coerce_value_to_type(value, element_type, loc)
        return value
