from __future__ import annotations

import ast
from typing import TYPE_CHECKING, TypeAlias, cast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops
from ...mlir.dialects import linalg as linalg_ops
from ...mlir.dialects import tensor as tensor_ops
from ..models import (
    AffineMapFactory,
    ArrayAttrFactory,
    AttrBuilderFactory,
    BlockFactory,
)

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object

# Base primitive type names(lyrt.prim)
# Int[N] supports 1 - 8388608 bits(LLVM limit)
# Float[N] supports 16, 32, 64, 128 bits only
PRIMITIVE_BASE_TYPES: set[str] = {"Int", "Float"}

# Valid bit widths for Float type
FLOAT_VALID_BITS: set[int] = {16, 32, 64, 128}

# LLVM maximum integer bit width
INT_MAX_BITS: int = 8388608

PrimitiveScalar: TypeAlias = int | float
PrimitiveLiteral: TypeAlias = PrimitiveScalar | list["PrimitiveLiteral"]


class PrimitiveMixin(VisitorRuntime):
    """Helpers for lyrt primitive/native frontend support."""

    def get_primitive_type_from_spec(self, base_type: str, bits: int) -> ir.Type:
        if base_type == "Int":
            if not (1 <= bits <= INT_MAX_BITS):
                raise ValueError(f"Int bit width must be 1-{INT_MAX_BITS}, got {bits}")
            return ir.IntegerType.get_signless(bits, context=self.ctx)
        if base_type == "Float":
            if bits not in FLOAT_VALID_BITS:
                raise ValueError(
                    f"Float bit width must be one of {sorted(FLOAT_VALID_BITS)}, got {bits}"
                )
            if bits == 16:
                return ir.F16Type.get(context=self.ctx)
            if bits == 32:
                return ir.F32Type.get(context=self.ctx)
            if bits == 64:
                return ir.F64Type.get(context=self.ctx)
            if bits == 128:
                return ir.Type.parse("f128", self.ctx)
        raise ValueError(f"Unsupported primitive base type: {base_type}")

    def _handle_prim_constructor(
        self, node: ast.Call, base_type: str, loc: ir.Location
    ) -> ir.Value:
        if len(node.args) != 1:
            raise ValueError(f"{base_type}[N]() requires exactly 1 argument")

        assert isinstance(node.func, ast.Subscript)
        if not isinstance(node.func.slice, ast.Constant) or not isinstance(
            node.func.slice.value, int
        ):
            raise ValueError(f"{base_type} requires an integer bit width")
        bits = node.func.slice.value
        prim_type = self.get_primitive_type_from_spec(base_type, bits)
        value_node = node.args[0]

        if isinstance(value_node, ast.Constant):
            value = value_node.value
            with loc, self.insertion_point():
                if base_type == "Int" and isinstance(value, int):
                    attr = ir.IntegerAttr.get(prim_type, value)
                    result = arith_ops.ConstantOp(prim_type, attr).result
                    self._pending_prim_const = (prim_type, value)
                    return result
                if base_type == "Float" and isinstance(value, (int, float)):
                    attr = ir.FloatAttr.get(prim_type, float(value))
                    result = arith_ops.ConstantOp(prim_type, attr).result
                    self._pending_prim_const = (prim_type, float(value))
                    return result
            raise ValueError(
                f"Cannot convert {type(value).__name__} to {base_type}[{bits}]"
            )

        py_value = self.require_value(value_node, self.visit(value_node))
        with loc, self.insertion_point():
            return py_ops.CastToPrimOp(
                prim_type, py_value, ir.StringAttr.get("exact", self.ctx)
            ).result

    def _handle_from_prim(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        self._ensure_not_in_native("from_prim", loc)
        if len(node.args) != 1:
            raise ValueError("from_prim() requires exactly 1 argument")

        prim_value = self.require_value(node.args[0], self.visit(node.args[0]))
        prim_type = prim_value.type
        prim_type_str = str(prim_type)

        if prim_type_str.startswith("i"):
            result_type = self.get_py_type("!py.int")
        elif prim_type_str.startswith("f"):
            result_type = self.get_py_type("!py.float")
        elif isinstance(prim_type, ir.RankedTensorType):
            result_type = self.get_py_type("!py.str")
        else:
            raise ValueError(
                f"Cannot convert primitive type {prim_type_str} to Python object"
            )

        with loc, self.insertion_point():
            return py_ops.CastFromPrimOp(result_type, prim_value).result

    def is_primitive_base_type(self, type_name: str) -> bool:
        return type_name in self._prim_types or type_name in PRIMITIVE_BASE_TYPES

    def get_primitive_element_type(self, prim_type: ir.Type) -> ir.Type:
        if isinstance(prim_type, ir.ShapedType):
            return prim_type.element_type
        return prim_type

    def is_primitive_float_type(self, prim_type: ir.Type) -> bool:
        elem_type = self.get_primitive_element_type(prim_type)
        return str(elem_type).startswith("f")

    def is_primitive_int_type(self, prim_type: ir.Type) -> bool:
        elem_type = self.get_primitive_element_type(prim_type)
        return str(elem_type).startswith("i")

    def is_py_type(self, mlir_type: ir.Type) -> bool:
        return str(mlir_type).startswith("!py.")

    def _parse_primitive_shaped_type(
        self, base_type: str, slice_expr: ast.expr
    ) -> ir.Type:
        elem_type, dims, _ = self._parse_primitive_shaped_spec(base_type, slice_expr)
        shape = [
            ir.ShapedType.get_dynamic_size() if dim is None else dim for dim in dims
        ]
        with self._loc(slice_expr):
            return ir.RankedTensorType.get(shape, elem_type)

    def _parse_primitive_shaped_spec(
        self, base_type: str, slice_expr: ast.expr
    ) -> tuple[ir.Type, list[int | None], bool]:
        if isinstance(slice_expr, ast.Tuple):
            items = slice_expr.elts
        else:
            items = [slice_expr]

        if not items:
            raise ValueError(f"{base_type} requires element type specification")

        elem_type = self.annotation_to_primitive_type(items[0])
        if elem_type is None:
            raise ValueError(f"{base_type} element type must be primitive")

        dims: list[int | None] = []
        has_ellipsis = False
        for dim in items[1:]:
            if isinstance(dim, ast.Constant) and isinstance(dim.value, int):
                dims.append(dim.value)
            elif isinstance(dim, ast.Constant) and dim.value is Ellipsis:
                dims.append(None)
                has_ellipsis = True
            else:
                raise ValueError(f"{base_type} dimensions must be integers or ...")

        if base_type == "Vector" and len(dims) != 1 and not has_ellipsis:
            raise ValueError("Vector requires exactly 1 dimension")
        if base_type == "Matrix" and len(dims) != 2 and not has_ellipsis:
            raise ValueError("Matrix requires exactly 2 dimensions")

        return elem_type, dims, has_ellipsis

    def _extract_primitive_literal(self, node: ast.AST) -> PrimitiveLiteral:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Primitive tensor constructor requires numeric literals")
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
            if isinstance(node.op, ast.USub):
                if isinstance(node.operand.value, (int, float)):
                    return -node.operand.value
                raise ValueError("Unary minus expects numeric literal")
            if isinstance(node.op, ast.UAdd):
                if isinstance(node.operand.value, (int, float)):
                    return +node.operand.value
                raise ValueError("Unary plus expects numeric literal")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._extract_primitive_literal(elt) for elt in node.elts]
        raise ValueError("Primitive tensor constructor requires literal values")

    def _infer_primitive_shape(self, value: PrimitiveLiteral) -> list[int]:
        if isinstance(value, list):
            length = len(value)
            if length == 0:
                return [0]
            subshape = self._infer_primitive_shape(value[0])
            for elem in value[1:]:
                if self._infer_primitive_shape(elem) != subshape:
                    raise ValueError("Inconsistent tensor literal shape")
            return [length] + subshape
        return []

    def _flatten_primitive_literal(
        self, value: PrimitiveLiteral, out: list[PrimitiveScalar]
    ) -> None:
        if isinstance(value, list):
            for elem in value:
                self._flatten_primitive_literal(elem, out)
            return
        out.append(value)

    def _build_primitive_scalar(
        self, value: object, elem_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        with loc, self.insertion_point():
            if str(elem_type).startswith("f") and isinstance(value, (int, float)):
                attr = ir.FloatAttr.get(elem_type, float(value))
                return arith_ops.ConstantOp(elem_type, attr).result
            if str(elem_type).startswith("i") and isinstance(value, int):
                attr = ir.IntegerAttr.get(elem_type, value)
                return arith_ops.ConstantOp(elem_type, attr).result
        raise ValueError("Tensor element type/value mismatch")

    def _populate_linalg_fill_region(
        self, op: linalg_ops.FillOp, elem_type: ir.Type, loc: ir.Location
    ) -> None:
        region = op.operation.regions[0]
        block_cls = cast(BlockFactory, ir.Block)
        block = block_cls.create_at_start(region, [elem_type, elem_type])
        with ir.InsertionPoint(block), loc:
            linalg_ops.YieldOp([block.arguments[0]])

    def _populate_linalg_matmul_region(
        self, op: linalg_ops.GenericOp, elem_type: ir.Type, loc: ir.Location
    ) -> None:
        region = op.operation.regions[0]
        block_cls = cast(BlockFactory, ir.Block)
        block = block_cls.create_at_start(region, [elem_type, elem_type, elem_type])
        with ir.InsertionPoint(block), loc:
            prod = arith_ops.MulFOp(block.arguments[0], block.arguments[1]).result
            acc = arith_ops.AddFOp(prod, block.arguments[2]).result
            linalg_ops.YieldOp([acc])

    def _build_linalg_fill(
        self, value: ir.Value, init: ir.Value, elem_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        op = linalg_ops.FillOp([init.type], [value], [init])
        self._populate_linalg_fill_region(op, elem_type, loc)
        return op.result

    def _build_linalg_matmul(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
        elem_type: ir.Type,
        loc: ir.Location,
    ) -> ir.Value:
        d0 = ir.AffineExpr.get_dim(0)
        d1 = ir.AffineExpr.get_dim(1)
        d2 = ir.AffineExpr.get_dim(2)
        affine_map_cls = cast(AffineMapFactory, ir.AffineMap)
        map_a = affine_map_cls.get(3, 0, [d0, d2])
        map_b = affine_map_cls.get(3, 0, [d2, d1])
        map_c = affine_map_cls.get(3, 0, [d0, d1])
        attr_builder_cls = cast(AttrBuilderFactory, ir.AttrBuilder)
        iterator_builder = attr_builder_cls.get("IteratorTypeEnum")
        array_attr_cls = cast(ArrayAttrFactory, ir.ArrayAttr)
        iterator_types = array_attr_cls.get(
            [
                iterator_builder(linalg_ops.IteratorType.parallel, context=self.ctx),
                iterator_builder(linalg_ops.IteratorType.parallel, context=self.ctx),
                iterator_builder(linalg_ops.IteratorType.reduction, context=self.ctx),
            ],
            context=self.ctx,
        )
        op = linalg_ops.GenericOp(
            [init.type],
            [lhs, rhs],
            [init],
            [map_a, map_b, map_c],
            iterator_types,
        )
        self._populate_linalg_matmul_region(op, elem_type, loc)
        return op.result

    def build_primitive_tensor_constructor(
        self,
        base_type: str,
        slice_expr: ast.expr,
        value_node: ast.AST,
        loc: ir.Location,
    ) -> ir.Value:
        elem_type, dims_spec, has_ellipsis = self._parse_primitive_shaped_spec(
            base_type, slice_expr
        )
        literal = self._extract_primitive_literal(value_node)
        inferred_shape = self._infer_primitive_shape(literal)

        if has_ellipsis:
            if any(dim is not None for dim in dims_spec):
                raise ValueError("Mixed explicit dims and ... are not supported")
            dims = inferred_shape
        else:
            dims = [dim for dim in dims_spec if dim is not None]
            if dims != inferred_shape:
                raise ValueError("Tensor literal shape does not match type")

        if base_type == "Vector" and len(dims) != 1:
            raise ValueError("Vector literal must be 1-D")
        if base_type == "Matrix" and len(dims) != 2:
            raise ValueError("Matrix literal must be 2-D")

        flat: list[PrimitiveScalar] = []
        self._flatten_primitive_literal(literal, flat)
        if len(flat) == 0 and dims:
            raise ValueError("Tensor literal cannot be empty")

        with loc:
            tensor_type = ir.RankedTensorType.get(dims, elem_type)
        with loc, self.insertion_point():
            elements = [
                self._build_primitive_scalar(val, elem_type, loc) for val in flat
            ]
            return tensor_ops.FromElementsOp(tensor_type, elements).result

    def build_primitive_tensor_fill(
        self,
        base_type: str,
        slice_expr: ast.expr,
        fill_value: object,
        loc: ir.Location,
        shape_args: list[ast.expr] | None = None,
    ) -> ir.Value:
        elem_type, dims_spec, has_ellipsis = self._parse_primitive_shaped_spec(
            base_type, slice_expr
        )
        if has_ellipsis:
            raise ValueError(f"{base_type}.zeros/ones does not support ...")
        if shape_args:
            raise ValueError(f"{base_type}.zeros/ones takes no dimensions")
        dims = [dim for dim in dims_spec if dim is not None]

        if base_type == "Vector" and len(dims) != 1:
            raise ValueError("Vector requires exactly 1 dimension")
        if base_type == "Matrix" and len(dims) != 2:
            raise ValueError("Matrix requires exactly 2 dimensions")

        total = 1
        for dim in dims:
            total *= dim

        with loc:
            tensor_type = ir.RankedTensorType.get(dims, elem_type)
        with loc, self.insertion_point():
            elements = [
                self._build_primitive_scalar(fill_value, elem_type, loc)
                for _ in range(total)
            ]
            return tensor_ops.FromElementsOp(tensor_type, elements).result

    def build_exception_value(
        self,
        *,
        message: ir.Value | None,
        loc: ir.Location,
        exc_type_name: str = "Exception",
        context: ir.Value | None = None,
    ) -> ir.Value:
        del exc_type_name, context
        with loc, self.insertion_point():
            args = [] if message is None else [message]
            return py_ops.ExceptionNewOp(
                self.get_py_type("!py.exception"),
                args,
            ).result

    def _set_in_native_func(self, value: bool) -> None:
        self._in_native_func = value
        for visitor in self.subvisitors.values():
            visitor._in_native_func = value

    def _set_native_gc_mode(self, mode: str | None) -> None:
        self._native_gc_mode = mode
        for visitor in self.subvisitors.values():
            visitor._native_gc_mode = mode

    def _enter_native_context(self, gc_mode: str) -> None:
        self._set_native_gc_mode(gc_mode)
        self._prim_allocated.clear()
        self._prim_deallocated.clear()
        self._prim_alloc_sites.clear()
        self._set_in_native_func(True)

    def _exit_native_context(self, loc: ir.Location) -> None:
        self._finalize_native_allocs(loc)
        self._set_in_native_func(False)
        self._set_native_gc_mode(None)
        self._prim_allocated.clear()
        self._prim_deallocated.clear()
        self._prim_alloc_sites.clear()

    def _is_py_type(self, ty: ir.Type) -> bool:
        return str(ty).startswith("!py.")

    def _ensure_native_only(self, name: str, loc: ir.Location) -> None:
        if not self._in_native_func:
            raise ValueError(f"{name}() is only available in @native functions: {loc}")

    def _ensure_not_in_native(self, name: str, loc: ir.Location) -> None:
        if self._in_native_func:
            raise ValueError(f"{name}() is not allowed in @native functions: {loc}")

    def _ensure_primitive_value(self, value: ir.Value, loc: ir.Location) -> None:
        if self._is_py_type(value.type):
            raise ValueError(f"Expected a primitive value, got {value.type} at {loc}")

    def _note_prim_alloc(self, value: ir.Value, loc: ir.Location) -> None:
        if value in self._prim_allocated and value not in self._prim_deallocated:
            raise ValueError(f"Double alloc detected at {loc}")
        self._prim_allocated.add(value)
        self._prim_alloc_sites[value] = loc

    def _note_prim_dealloc(self, value: ir.Value, loc: ir.Location) -> None:
        if value not in self._prim_allocated:
            raise ValueError(f"Dealloc of non-alloc value at {loc}")
        if value in self._prim_deallocated:
            raise ValueError(f"Double dealloc detected at {loc}")
        self._prim_deallocated.add(value)

    def _check_prim_overwrite(self, name: str, loc: ir.Location) -> None:
        if not self._in_native_func:
            return
        try:
            existing = self.lookup_symbol(name)
        except NameError:
            return
        if existing in self._prim_allocated and existing not in self._prim_deallocated:
            raise ValueError(
                f"Overwriting allocated value '{name}' without dealloc at {loc}"
            )

    def _finalize_native_allocs(self, loc: ir.Location) -> None:
        leaked = self._prim_allocated - self._prim_deallocated
        if leaked:
            raise ValueError(
                f"Leaked allocations detected at {loc}: {len(leaked)} value(s)"
            )

    def annotation_to_primitive_type(
        self, annotation: ast.expr | None
    ) -> ir.Type | None:
        if annotation is None:
            return None
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                base_type = annotation.value.id
                if base_type in self._prim_types:
                    base_type = self._prim_types[base_type]

                if base_type in PRIMITIVE_BASE_TYPES:
                    if isinstance(annotation.slice, ast.Constant) and isinstance(
                        annotation.slice.value, int
                    ):
                        bits = annotation.slice.value
                        return self.get_primitive_type_from_spec(base_type, bits)
                    raise ValueError(
                        f"Primitive type {base_type} requires an integer bit width"
                    )
                if base_type in ("Vector", "Matrix", "Tensor"):
                    return self._parse_primitive_shaped_type(
                        base_type, annotation.slice
                    )
        return None

    def _handle_to_prim_call(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        self._ensure_not_in_native("to_prim", loc)
        if len(node.args) != 2 or node.keywords:
            raise ValueError("to_prim() requires exactly 2 positional arguments")

        value = self.require_value(node.args[0], self.visit(node.args[0]))
        if not self._is_py_type(value.type):
            raise ValueError(f"to_prim() expects a Python object, got {value.type}")

        prim_type = self.annotation_to_primitive_type(node.args[1])
        if prim_type is None:
            raise ValueError(
                "to_prim() requires a primitive type as the second argument"
            )
        if isinstance(prim_type, ir.RankedTensorType):
            raise ValueError("to_prim() does not accept tensor primitive types")

        with loc, self.insertion_point():
            return py_ops.CastToPrimOp(
                prim_type, value, ir.StringAttr.get("exact", self.ctx)
            ).result

    def _handle_alloc_call(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        self._ensure_native_only("alloc", loc)
        if len(node.args) != 1 or node.keywords:
            raise ValueError("alloc() requires exactly 1 positional argument")

        value = self.require_value(node.args[0], self.visit(node.args[0]))
        self._ensure_primitive_value(value, loc)
        self._note_prim_alloc(value, loc)
        return value

    def _handle_dealloc_call(self, node: ast.Call, loc: ir.Location) -> None:
        self._ensure_native_only("dealloc", loc)
        if len(node.args) != 1 or node.keywords:
            raise ValueError("dealloc() requires exactly 1 positional argument")

        value = self.require_value(node.args[0], self.visit(node.args[0]))
        self._ensure_primitive_value(value, loc)
        self._note_prim_dealloc(value, loc)
        return None

    def register_prim_constant(
        self, name: str, mlir_type: ir.Type, value: int | float
    ) -> None:
        self._prim_constants[name] = (mlir_type, value)

    def get_prim_constant(self, name: str) -> tuple[ir.Type, int | float] | None:
        return self._prim_constants.get(name)
