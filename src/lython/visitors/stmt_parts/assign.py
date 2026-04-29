# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops


class StmtAssignMixin:
    """Statement lowering for assignment-like AST nodes."""

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")

        target = node.targets[0]

        expr_visitor = self.subvisitors.get("Expr")
        if expr_visitor:
            expr_visitor._pending_prim_const = None

        value = self.require_value(node.value, self.visit(node.value))

        if isinstance(target, ast.Name):
            if str(value.type).startswith('!py.class<"'):
                allow_fresh_class = (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and self.lookup_class(node.value.func.id) is not None
                )
                if not allow_fresh_class:
                    value = self.copy_static_class_value(value, loc=self._loc(node))
            alias_info = self.resolve_function_info_from_expression(node.value, value)
            if alias_info is not None and str(value.type).startswith("!py.func<"):
                value = self.annotate_known_callable_value(
                    value, alias_info, loc=self._loc(node)
                )
            self._check_prim_overwrite(target.id, self._loc(node))
            self.define_symbol(target.id, value)
            if alias_info is not None:
                self.define_function_binding(target.id, alias_info)
            else:
                self.undefine_function_binding(target.id)

            if expr_visitor:
                pending = getattr(expr_visitor, "_pending_prim_const", None)
                if pending is not None:
                    mlir_type, const_value = pending
                    self.register_prim_constant(target.id, mlir_type, const_value)
                    expr_visitor._pending_prim_const = None
            return

        if isinstance(target, ast.Attribute):
            obj = self.require_value(target.value, self.visit(target.value))
            attr_type = self.get_attribute_type(obj.type, target.attr)
            pending_attrs = getattr(self, "_pending_attributes", None)
            if (
                pending_attrs is not None
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                current_method = getattr(self, "_current_method", None)
                if current_method is None:
                    raise RuntimeError("Internal error: self assignment outside method")
                if current_method != "__init__" and target.attr not in pending_attrs:
                    raise ValueError(
                        f"Dynamic field introduction outside __init__ is not supported: "
                        f"self.{target.attr}"
                    )
                inferred_type = value.type
                if (
                    target.attr in pending_attrs
                    and pending_attrs[target.attr] != inferred_type
                ):
                    raise TypeError(
                        f"Field '{target.attr}' type mismatch: "
                        f"{pending_attrs[target.attr]} vs {inferred_type}"
                    )
                pending_attrs[target.attr] = inferred_type
                attr_type = inferred_type
                self._current_method_mutates_self = True
            if str(obj.type).startswith('!py.class<"'):
                value = self.coerce_value_to_type(value, attr_type, self._loc(node))
            with self._loc(node), self.insertion_point():
                py_ops.AttrSetOp(obj, target.attr, value)
            return

        if isinstance(target, ast.Subscript):
            container = self.require_value(target.value, self.visit(target.value))
            dict_types = self.get_dict_key_value_types(container.type)
            if dict_types is None:
                raise NotImplementedError(
                    "Subscript assignment is only supported for typed dict values"
                )
            key_type, value_type = dict_types
            key = self.require_value(target.slice, self.visit(target.slice))
            key = self.coerce_value_to_type(key, key_type, self._loc(node))
            value = self.coerce_value_to_type(value, value_type, self._loc(node))
            with self._loc(node), self.insertion_point():
                updated = py_ops.DictInsertOp(
                    container.type, container, key, value
                ).result
            if isinstance(target.value, ast.Name):
                self.define_symbol(target.value.id, updated)
            return

        raise NotImplementedError(
            f"Assignment target type {type(target).__name__} not supported"
        )

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        raise NotImplementedError("Type alias statement not implemented")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        rhs = self.require_value(node.value, self.visit(node.value))
        loc = self._loc(node)

        if isinstance(node.target, ast.Name):
            current = self.lookup_symbol(node.target.id)
            with loc, self.insertion_point():
                result = self._apply_binop(node.op, current, rhs, loc)
            self.define_symbol(node.target.id, result)
            self.undefine_function_binding(node.target.id)
            return

        if isinstance(node.target, ast.Attribute):
            obj = self.require_value(node.target.value, self.visit(node.target.value))
            attr_type = self.get_attribute_type(obj.type, node.target.attr)
            pending_attrs = getattr(self, "_pending_attributes", None)
            if (
                pending_attrs is not None
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
            ):
                if node.target.attr not in pending_attrs:
                    raise ValueError(
                        f"Unknown field '{node.target.attr}' in self mutation"
                    )
                self._current_method_mutates_self = True

            rhs = self.coerce_value_to_type(rhs, attr_type, loc)
            with loc, self.insertion_point():
                current = py_ops.AttrGetOp(attr_type, obj, node.target.attr).result
                result = self._apply_binop(node.op, current, rhs, loc)
                py_ops.AttrSetOp(obj, node.target.attr, result)
            return

        raise NotImplementedError(
            f"Augmented assignment target type {type(node.target).__name__} not supported"
        )

    def _apply_binop(
        self, op: ast.operator, lhs: ir.Value, rhs: ir.Value, loc: ir.Location
    ) -> ir.Value:
        lhs, rhs = self.coerce_operands_for_binary(lhs, rhs, loc)
        if not self.is_py_type(lhs.type) and not self.is_py_type(rhs.type):
            if isinstance(op, ast.Add):
                if self.is_primitive_float_type(lhs.type):
                    return arith_ops.AddFOp(lhs, rhs).result
                if self.is_primitive_int_type(lhs.type):
                    return arith_ops.AddIOp(lhs, rhs).result
            elif isinstance(op, ast.Sub):
                if self.is_primitive_float_type(lhs.type):
                    return arith_ops.SubFOp(lhs, rhs).result
                if self.is_primitive_int_type(lhs.type):
                    return arith_ops.SubIOp(lhs, rhs).result
        if isinstance(op, ast.Add):
            return py_ops.NumAddOp(lhs, rhs).result
        if isinstance(op, ast.Sub):
            return py_ops.NumSubOp(lhs, rhs).result
        raise NotImplementedError(f"Operator {type(op).__name__} not supported")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        raise NotImplementedError(
            "An assignment with a type annotation is not implemented"
        )
