from __future__ import annotations

import ast
from typing import Any, NamedTuple, NoReturn

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops

__all__ = ["BaseVisitor"]


# Mapping from lyrt.prim type names to MLIR type constructors
PRIMITIVE_TYPE_MAP: dict[str, tuple[str, int]] = {
    # Integer types: (kind, bits)
    "i1": ("int", 1),
    "i8": ("int", 8),
    "i16": ("int", 16),
    "i32": ("int", 32),
    "i64": ("int", 64),
    "i128": ("int", 128),
    "i256": ("int", 256),
    # Float types
    "f16": ("float", 16),
    "f32": ("float", 32),
    "f64": ("float", 64),
    "f128": ("float", 128),
    "f256": ("float", 256),
}


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
        self._prim_types: dict[str, str] = {}  # Imported primitive types: name -> kind
        self._lyrt_builtins: set[str] = (
            set()
        )  # Imported lyrt builtins (native, to_prim, from_prim)
        # Track primitive constants for cross-region access: name -> (mlir_type, python_value)
        self._prim_constants: dict[str, tuple[ir.Type, int | float]] = {}
        # Temporary storage for primitive constant info during assignment
        self._pending_prim_const: tuple[ir.Type, int | float] | None = None

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
            visitor._prim_constants = self._prim_constants
            # Note: _pending_prim_const is not shared - each visitor has its own

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
    def get_primitive_type(self, type_name: str) -> ir.Type:
        """Get MLIR primitive type from a lyrt.prim type name."""
        if type_name not in PRIMITIVE_TYPE_MAP:
            raise ValueError(f"Unknown primitive type: {type_name}")
        kind, bits = PRIMITIVE_TYPE_MAP[type_name]
        if kind == "int":
            return ir.IntegerType.get_signless(bits, context=self.ctx)
        elif kind == "float":
            if bits == 16:
                return ir.F16Type.get(context=self.ctx)
            elif bits == 32:
                return ir.F32Type.get(context=self.ctx)
            elif bits == 64:
                return ir.F64Type.get(context=self.ctx)
            else:
                raise ValueError(f"Unsupported float bit width: {bits}")
        raise ValueError(f"Unknown primitive kind: {kind}")

    def is_primitive_type(self, type_name: str) -> bool:
        """Check if a type name refers to a primitive type."""
        return type_name in self._prim_types or type_name in PRIMITIVE_TYPE_MAP

    def _set_in_native_func(self, value: bool) -> None:
        """Set native function mode across all subvisitors."""
        self._in_native_func = value
        for visitor in self.subvisitors.values():
            visitor._in_native_func = value

    def annotation_to_primitive_type(
        self, annotation: ast.expr | None
    ) -> ir.Type | None:
        """Convert a type annotation to a primitive MLIR type, or None if not primitive."""
        if annotation is None:
            return None
        if isinstance(annotation, ast.Name):
            name = annotation.id
            if name in self._prim_types:
                return self.get_primitive_type(name)
            if name in PRIMITIVE_TYPE_MAP:
                return self.get_primitive_type(name)
        return None

    def register_prim_constant(
        self, name: str, mlir_type: ir.Type, value: int | float
    ) -> None:
        """Register a primitive constant for cross-region access."""
        self._prim_constants[name] = (mlir_type, value)

    def get_prim_constant(self, name: str) -> tuple[ir.Type, int | float] | None:
        """Get a registered primitive constant, or None if not found."""
        return self._prim_constants.get(name)
