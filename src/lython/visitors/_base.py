from __future__ import annotations

import ast
import sys
from typing import Any, NamedTuple, NoReturn, TypeAlias

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from ..mlir.dialects import arith as arith_ops
from ..mlir.dialects import linalg as linalg_ops
from ..mlir.dialects import tensor as tensor_ops

__all__ = ["BaseVisitor"]


# Base primitive type names (lyrt.prim)
# Int[N] supports 1-8388608 bits (LLVM limit)
# Float[N] supports 16, 32, 64, 128 bits only
PRIMITIVE_BASE_TYPES: set[str] = {"Int", "Float"}

# Valid bit widths for Float type
FLOAT_VALID_BITS: set[int] = {16, 32, 64, 128}

# LLVM maximum integer bit width
INT_MAX_BITS: int = 8388608

PrimitiveScalar: TypeAlias = int | float
PrimitiveLiteral: TypeAlias = PrimitiveScalar | list["PrimitiveLiteral"]


class FunctionInfo(NamedTuple):
    symbol: str
    func_type: ir.Type
    arg_types: tuple[ir.Type, ...]
    result_types: tuple[ir.Type, ...]
    has_vararg: bool


class MethodInfo(NamedTuple):
    """Information about a class method."""

    name: str
    arg_types: tuple[ir.Type, ...]  # Including self
    result_types: tuple[ir.Type, ...]


class ClassInfo(NamedTuple):
    """Information about a class definition."""

    name: str
    class_type: ir.Type
    methods: dict[str, MethodInfo]
    attributes: dict[str, ir.Type]  # Instance attribute name -> type


class BaseVisitor:
    """
    ベースとなるVisitorクラス

    すべてのASTノードに対して visit_<NodeType>() メソッドをディスパッチする
    また、専用のサブビジター (AliasVisitor, ArgVisitor, etc. など) が存在する場合は、
    self.subvisitors に登録されている対応するVisitorへ転送。
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, "BaseVisitor"] | None = None,
    ) -> None:
        self.module: ir.Module
        self.ctx = ctx
        self.ctx.allow_unregistered_dialects = True
        self._type_cache: dict[str, ir.Type] = {}
        self.current_block: ir.Block | None = None
        self._scope_stack: list[dict[str, ir.Value]] = []
        self._module_name: str = "__main__"
        self._functions: dict[str, FunctionInfo] = {}
        self._classes: dict[str, ClassInfo] = {}
        # Primitive world support
        self._in_native_func: bool = False  # True when inside @native function
        self._native_gc_mode: str | None = None
        self._prim_types: dict[str, str] = {}  # Imported primitive types: name -> kind
        self._lyrt_builtins: set[str] = (
            set()
        )  # Imported lyrt builtins (native, to_prim, from_prim)
        # Primitive allocation tracking (native world only)
        self._prim_allocated: set[ir.Value] = set()
        self._prim_deallocated: set[ir.Value] = set()
        self._prim_alloc_sites: dict[ir.Value, ir.Location] = {}
        # Track primitive constants for cross-region access: name -> (mlir_type, python_value)
        self._prim_constants: dict[str, tuple[ir.Type, int | float]] = {}
        # Temporary storage for primitive constant info during assignment
        self._pending_prim_const: tuple[ir.Type, int | float] | None = None
        self._ensure_mlir_dialect_aliases()

        if subvisitors is not None:
            self.subvisitors = subvisitors
            return

        subvisitors = {}
        self.subvisitors = subvisitors

        from .expr import ExprVisitor
        from .mod import ModVisitor
        from .stmt import StmtVisitor

        subvisitors["Module"] = ModVisitor(ctx, subvisitors=subvisitors)
        subvisitors["Stmt"] = StmtVisitor(ctx, subvisitors=subvisitors)
        subvisitors["Expr"] = ExprVisitor(ctx, subvisitors=subvisitors)
        for visitor in subvisitors.values():
            visitor.subvisitors = subvisitors
            visitor._type_cache = self._type_cache
            visitor._scope_stack = self._scope_stack
            visitor._module_name = self._module_name
            visitor._functions = self._functions
            visitor._classes = self._classes
            visitor._prim_types = self._prim_types
            visitor._lyrt_builtins = self._lyrt_builtins
            visitor._native_gc_mode = self._native_gc_mode
            visitor._prim_allocated = self._prim_allocated
            visitor._prim_deallocated = self._prim_deallocated
            visitor._prim_alloc_sites = self._prim_alloc_sites
            visitor._prim_constants = self._prim_constants
            # Note: _pending_prim_const is not shared - each visitor has its own

    def _ensure_mlir_dialect_aliases(self) -> None:
        sys.modules.setdefault("mlir.dialects.linalg", linalg_ops)
        linalg_gen = sys.modules.get("lython.mlir.dialects._linalg_ops_gen")
        if linalg_gen is not None:
            sys.modules.setdefault("mlir.dialects._linalg_ops_gen", linalg_gen)
        linalg_enum = sys.modules.get("lython.mlir.dialects._linalg_enum_gen")
        if linalg_enum is not None:
            sys.modules.setdefault("mlir.dialects._linalg_enum_gen", linalg_enum)

    def visit(self, node: ast.AST) -> Any:
        method_name = f"visit_{type(node).__name__}"

        # 1) このビジター自身が実装していれば呼び出し
        if hasattr(self, method_name):
            visitor = getattr(self, method_name)
            return visitor(node)

        # 2) 同一クラス名に対する直接委譲
        name = type(node).__name__
        visitor = self.subvisitors.get(name)
        if visitor is not None and visitor is not self:
            result = visitor.visit(node)
            if isinstance(node, ast.mod):
                self.module = getattr(visitor, "module")
            return result

        # 3) カテゴリ委譲
        if isinstance(node, ast.stmt):
            v = self.subvisitors.get("Stmt")
            if v is not None and v is not self:
                return v.visit(node)
        if isinstance(node, ast.mod):
            v = self.subvisitors.get("Module")
            if v is not None and v is not self:
                result = v.visit(node)
                self.module = getattr(v, "module")
                return result
        if isinstance(node, ast.expr):
            v = self.subvisitors.get("Expr")
            if v is not None and v is not self:
                return v.visit(node)

        # 4) それでも無ければ未実装
        return self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> NoReturn:
        """
        各ノードに対応する visit_* が未定義の場合、ここにフォールバック
        未実装の構文要素があればエラーを出す
        """
        raise NotImplementedError(
            f"Node type {type(node).__name__} not implemented by {self.__class__.__name__}"
        )

    def _set_module(self, module: ir.Module) -> None:
        self.module = module
        for visitor in self.subvisitors.values():
            visitor.module = module

    def _set_module_name(self, name: str) -> None:
        self._module_name = name
        for visitor in self.subvisitors.values():
            visitor._module_name = name

    def _set_insertion_block(self, block: ir.Block | None) -> None:
        self.current_block = block
        for visitor in self.subvisitors.values():
            visitor.current_block = block

    def get_py_type(self, type_spec: str) -> ir.Type:
        cached = self._type_cache.get(type_spec)
        if cached is None:
            cached = ir.Type.parse(type_spec, self.ctx)
            self._type_cache[type_spec] = cached
        return cached

    def require_value(self, node: ast.AST, result: Any) -> ir.Value:
        if isinstance(result, ir.Value):
            if self._in_native_func and result in self._prim_deallocated:
                loc = self._loc(node)
                raise ValueError(
                    f"Use-after-dealloc detected in @native(gc={self._native_gc_mode}): "
                    f"{loc}"
                )
            return result
        raise TypeError(
            f"Visitor for {type(node).__name__} must return an MLIR value, got {type(result)!r}"
        )

    def _loc(self, node: ast.AST) -> ir.Location:
        lineno = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        if lineno is None or col is None:
            return ir.Location.unknown(self.ctx)
        return ir.Location.file(self._module_name, int(lineno), int(col) + 1, self.ctx)

    def insertion_point(self) -> ir.InsertionPoint:
        if self.current_block is None:
            raise RuntimeError("Insertion block is not set")
        return ir.InsertionPoint(self.current_block)

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
        location = loc or ir.Location.unknown(self.ctx)
        with location, self.insertion_point():
            return py_ops.UpcastOp(object_type, value).result

    def annotation_to_py_type(self, annotation: ast.expr | None) -> str:
        if annotation is None:
            return "!py.object"
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return "!py.none"
        if isinstance(annotation, ast.Name):
            mapping = {
                "int": "!py.int",
                "float": "!py.float",
                "bool": "!py.bool",
                "str": "!py.str",
                "None": "!py.none",
            }
            if annotation.id in mapping:
                return mapping[annotation.id]
        raise NotImplementedError(f"Unsupported annotation {ast.dump(annotation)}")

    def build_funcsig(self, arg_types: list[str], result_types: list[str]) -> str:
        args = ", ".join(arg_types)
        rets = ", ".join(result_types)
        arg_part = f"[{args}]" if args else "[]"
        ret_part = f"[{rets}]" if rets else "[]"
        return f"!py.funcsig<{arg_part} -> {ret_part}>"

    # --- Scope management -------------------------------------------------
    def push_scope(self) -> None:
        self._scope_stack.append({})
        for visitor in self.subvisitors.values():
            visitor._scope_stack = self._scope_stack

    def pop_scope(self) -> dict[str, ir.Value]:
        if not self._scope_stack:
            raise RuntimeError("Scope stack underflow")
        scope = self._scope_stack.pop()
        for visitor in self.subvisitors.values():
            visitor._scope_stack = self._scope_stack
        return scope

    def current_scope(self) -> dict[str, ir.Value]:
        if not self._scope_stack:
            raise RuntimeError("Scope stack is empty")
        return self._scope_stack[-1]

    def define_symbol(self, name: str, value: ir.Value) -> None:
        self.current_scope()[name] = value

    def lookup_symbol(self, name: str) -> ir.Value:
        for scope in reversed(self._scope_stack):
            if name in scope:
                return scope[name]
        raise NameError(f"Undefined symbol '{name}'")

    def register_function(
        self,
        name: str,
        func_type: ir.Type,
        arg_types: list[ir.Type],
        result_types: list[ir.Type],
        *,
        symbol: str | None = None,
        has_vararg: bool = False,
    ) -> None:
        info = FunctionInfo(
            symbol or name,
            func_type,
            tuple(arg_types),
            tuple(result_types),
            has_vararg,
        )
        self._functions[name] = info

    def lookup_function(self, name: str) -> FunctionInfo:
        if name not in self._functions:
            raise NameError(f"Unknown function '{name}'")
        return self._functions[name]

    def register_class(
        self,
        name: str,
        class_type: ir.Type,
        methods: dict[str, MethodInfo],
        attributes: dict[str, ir.Type] | None = None,
    ) -> None:
        info = ClassInfo(name, class_type, methods, attributes or {})
        self._classes[name] = info

    def lookup_class(self, name: str) -> ClassInfo | None:
        return self._classes.get(name)

    def get_attribute_type(self, obj_type: ir.Type, attr_name: str) -> ir.Type:
        """
        オブジェクト型と属性名から属性の型を推論する。

        クラス処理中は _pending_attributes を、それ以外は登録済みの ClassInfo を参照する。
        型が不明な場合は !py.object を返す。
        """
        obj_type_str = str(obj_type)

        # Extract class name from type like !py.class<"Counter">
        if not (obj_type_str.startswith('!py.class<"') and obj_type_str.endswith('">')):
            return self.get_py_type("!py.object")

        class_name = obj_type_str[len('!py.class<"') : -len('">')]  # noqa

        # First check _pending_attributes (during class processing)
        stmt_visitor = self.subvisitors.get("Stmt") or self
        pending_attrs = getattr(stmt_visitor, "_pending_attributes", None)
        if pending_attrs is not None and attr_name in pending_attrs:
            return pending_attrs[attr_name]

        # Fall back to registered class info
        class_info = self.lookup_class(class_name)
        if class_info is not None and attr_name in class_info.attributes:
            return class_info.attributes[attr_name]

        return self.get_py_type("!py.object")

    def _block_terminated(self, block: ir.Block) -> bool:
        ops = list(block.operations)
        if not ops:
            return False
        terminators = {
            "py.return",
            "func.return",
            "cf.br",
            "cf.cond_br",
        }
        return ops[-1].operation.name in terminators

    # --- Primitive world support -------------------------------------------
    def get_primitive_type_from_spec(self, base_type: str, bits: int) -> ir.Type:
        """Get MLIR primitive type from base type name and bit width.

        Args:
            base_type: "Int" or "Float"
            bits: bit width (1-8388608 for Int, 16/32/64/128 for Float)
        """
        if base_type == "Int":
            if not (1 <= bits <= INT_MAX_BITS):
                raise ValueError(f"Int bit width must be 1-{INT_MAX_BITS}, got {bits}")
            return ir.IntegerType.get_signless(bits, context=self.ctx)
        elif base_type == "Float":
            if bits not in FLOAT_VALID_BITS:
                raise ValueError(
                    f"Float bit width must be one of {sorted(FLOAT_VALID_BITS)}, got {bits}"
                )
            if bits == 16:
                return ir.F16Type.get(context=self.ctx)
            elif bits == 32:
                return ir.F32Type.get(context=self.ctx)
            elif bits == 64:
                return ir.F64Type.get(context=self.ctx)
            elif bits == 128:
                return ir.Type.parse("f128", self.ctx)
        raise ValueError(f"Unknown primitive base type: {base_type}")

    def is_primitive_base_type(self, type_name: str) -> bool:
        """Check if a type name refers to a primitive base type (Int or Float)."""
        return type_name in self._prim_types or type_name in PRIMITIVE_BASE_TYPES

    def get_primitive_element_type(self, prim_type: ir.Type) -> ir.Type:
        """Return element type for shaped primitives, or the type itself."""
        if isinstance(prim_type, ir.ShapedType):
            return prim_type.element_type
        return prim_type

    def is_primitive_float_type(self, prim_type: ir.Type) -> bool:
        """Check if a primitive type (scalar or shaped) has a float element type."""
        elem_type = self.get_primitive_element_type(prim_type)
        return str(elem_type).startswith("f")

    def is_primitive_int_type(self, prim_type: ir.Type) -> bool:
        """Check if a primitive type (scalar or shaped) has an integer element type."""
        elem_type = self.get_primitive_element_type(prim_type)
        return str(elem_type).startswith("i")

    def is_py_type(self, mlir_type: ir.Type) -> bool:
        """Best-effort check for !py.* types in Python frontend."""
        return str(mlir_type).startswith("!py.")

    def _parse_primitive_shaped_type(
        self, base_type: str, slice_expr: ast.expr
    ) -> ir.Type:
        """Parse Vector/Matrix/Tensor type annotations into RankedTensorType."""
        elem_type, dims, _ = self._parse_primitive_shaped_spec(base_type, slice_expr)
        shape = [
            ir.ShapedType.get_dynamic_size() if dim is None else dim for dim in dims
        ]
        return ir.RankedTensorType.get(shape, elem_type)

    def _parse_primitive_shaped_spec(
        self, base_type: str, slice_expr: ast.expr
    ) -> tuple[ir.Type, list[int | None], bool]:
        """Parse Vector/Matrix/Tensor type spec into element type + dims."""
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
        """Extract nested python literals from AST nodes for prim tensors."""
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
        """Infer tensor shape from nested list literals."""
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
        """Flatten nested list literals in row-major order."""
        if isinstance(value, list):
            for elem in value:
                self._flatten_primitive_literal(elem, out)
            return
        out.append(value)

    def _build_primitive_scalar(
        self, value: object, elem_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        """Create a scalar constant for primitive tensor elements."""
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
        block = ir.Block.create_at_start(region, [elem_type, elem_type])  # type: ignore
        with ir.InsertionPoint(block), loc:
            linalg_ops.YieldOp([block.arguments[0]])

    def _populate_linalg_matmul_region(
        self, op: linalg_ops.GenericOp, elem_type: ir.Type, loc: ir.Location
    ) -> None:
        region = op.operation.regions[0]
        block = ir.Block.create_at_start(  # type: ignore
            region, [elem_type, elem_type, elem_type]
        )
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
        map_a = ir.AffineMap.get(3, 0, [d0, d2])  # type: ignore
        map_b = ir.AffineMap.get(3, 0, [d2, d1])  # type: ignore
        map_c = ir.AffineMap.get(3, 0, [d0, d1])  # type: ignore
        iterator_builder = ir.AttrBuilder.get("IteratorTypeEnum")  # type: ignore
        iterator_types = ir.ArrayAttr.get(  # type: ignore
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
        """Build a ranked tensor from a nested literal value."""
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
        """Build a ranked tensor filled with a scalar value."""
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

    def _set_in_native_func(self, value: bool) -> None:
        """Set native function mode across all subvisitors."""
        self._in_native_func = value
        for visitor in self.subvisitors.values():
            visitor._in_native_func = value

    def _set_native_gc_mode(self, mode: str | None) -> None:
        """Set native GC mode across all subvisitors."""
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
        """Convert a type annotation to a primitive MLIR type, or None if not primitive.

        Handles Int[N] and Float[N] subscript syntax.
        """
        if annotation is None:
            return None

        # Handle Int[32], Float[64], etc.
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                base_type = annotation.value.id
                # Check if it's an aliased import
                if base_type in self._prim_types:
                    base_type = self._prim_types[base_type]

                if base_type in PRIMITIVE_BASE_TYPES:
                    # Get the bit width from the slice
                    if isinstance(annotation.slice, ast.Constant) and isinstance(
                        annotation.slice.value, int
                    ):
                        bits = annotation.slice.value
                        return self.get_primitive_type_from_spec(base_type, bits)
                    else:
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
        """Register a primitive constant for cross-region access."""
        self._prim_constants[name] = (mlir_type, value)

    def get_prim_constant(self, name: str) -> tuple[ir.Type, int | float] | None:
        """Get a registered primitive constant, or None if not found."""
        return self._prim_constants.get(name)
