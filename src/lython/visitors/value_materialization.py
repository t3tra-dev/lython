from __future__ import annotations

from typing import TYPE_CHECKING

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from .contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ValueMaterializationMixin(VisitorRuntime):
    """Helpers for constructing and coercing Python-level values."""

    def build_tuple(
        self, values: list[ir.Value], *, loc: ir.Location | None = None
    ) -> ir.Value:
        location = loc or ir.Location.unknown(self.ctx)
        if not values:
            tuple_type = self.get_py_type("!py.tuple<>")
            with location, self.insertion_point():
                return py_ops.TupleEmptyOp(tuple_type).result
        spec = ", ".join(str(value.type) for value in values)
        tuple_type = self.get_py_type(f"!py.tuple<{spec}>")
        with location, self.insertion_point():
            return py_ops.TupleCreateOp(tuple_type, values).result

    def ensure_object(
        self, value: ir.Value, *, loc: ir.Location | None = None
    ) -> ir.Value:
        object_type = self.get_py_type("!py.object")
        if value.type == object_type:
            return value
        if str(value.type).startswith('!py.class<"'):
            raise TypeError("Static class instances cannot be converted to !py.object")
        if not self.is_py_type(value.type):
            raise TypeError(
                f"Cannot convert primitive value {value.type} to !py.object"
            )
        location = loc or ir.Location.unknown(self.ctx)
        with location, self.insertion_point():
            return py_ops.UpcastOp(object_type, value).result

    def coerce_value_to_type(
        self, value: ir.Value, expected_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        if value.type == expected_type:
            return value

        object_type = self.get_py_type("!py.object")
        if expected_type == object_type:
            return self.ensure_object(value, loc=loc)

        raise TypeError(f"Cannot coerce {value.type} to {expected_type} at {loc}")

    def coerce_operands_for_binary(
        self, lhs: ir.Value, rhs: ir.Value, loc: ir.Location
    ) -> tuple[ir.Value, ir.Value]:
        return lhs, rhs

    def get_closure_storage_type(self, value_type: ir.Type) -> ir.Type:
        if self.is_py_type(value_type):
            return value_type
        if isinstance(value_type, ir.ShapedType):
            raise NotImplementedError(
                f"Capturing shaped primitive values in nested closures is not supported yet: {value_type}"
            )
        if self.is_primitive_float_type(value_type):
            return self.get_py_type("!py.float")
        if self.is_primitive_int_type(value_type):
            elem_type = self.get_primitive_element_type(value_type)
            if str(elem_type) == "i1":
                return self.get_py_type("!py.bool")
            return self.get_py_type("!py.int")
        raise NotImplementedError(
            f"Capturing values of type {value_type} in nested closures is not supported yet"
        )

    def materialize_closure_storage_value(
        self, value: ir.Value, *, loc: ir.Location
    ) -> ir.Value:
        storage_type = self.get_closure_storage_type(value.type)
        if storage_type == value.type:
            return value
        with loc, self.insertion_point():
            return py_ops.CastFromPrimOp(storage_type, value).result

    def materialize_captured_value_from_storage(
        self,
        storage_value: ir.Value,
        original_type: ir.Type,
        *,
        loc: ir.Location,
    ) -> ir.Value:
        if storage_value.type == original_type:
            return storage_value
        if not self.is_py_type(storage_value.type):
            raise TypeError(
                f"Cannot restore captured value of type {original_type} from storage type {storage_value.type}"
            )
        with loc, self.insertion_point():
            return py_ops.CastToPrimOp(
                original_type,
                storage_value,
                ir.StringAttr.get("exact", self.ctx),
            ).result

    def copy_static_class_value(
        self, value: ir.Value, *, loc: ir.Location | None = None
    ) -> ir.Value:
        class_info = self.get_class_info_from_type(value.type)
        if class_info is None:
            raise TypeError(f"Value is not a static class instance: {value.type}")
        location = loc or ir.Location.unknown(self.ctx)
        with location, self.insertion_point():
            copy_value = py_ops.ClassNewOp(
                class_info.class_type, class_info.name
            ).result
            for attr_name, attr_type in class_info.attributes.items():
                field_value = py_ops.AttrGetOp(attr_type, value, attr_name).result
                py_ops.AttrSetOp(copy_value, attr_name, field_value)
            return copy_value
