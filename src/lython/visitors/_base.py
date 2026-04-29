from __future__ import annotations

import ast
import sys
from typing import Any, NoReturn

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from ..mlir.dialects import linalg as linalg_ops
from .base import PrimitiveMixin
from .base import primitive as _primitive
from .models import ClassInfo, FunctionInfo, MethodInfo

PRIMITIVE_BASE_TYPES = _primitive.PRIMITIVE_BASE_TYPES
FLOAT_VALID_BITS = _primitive.FLOAT_VALID_BITS
INT_MAX_BITS = _primitive.INT_MAX_BITS
PrimitiveScalar = _primitive.PrimitiveScalar
PrimitiveLiteral = _primitive.PrimitiveLiteral

__all__ = [
    "BaseVisitor",
    "FLOAT_VALID_BITS",
    "INT_MAX_BITS",
    "PRIMITIVE_BASE_TYPES",
    "PrimitiveLiteral",
    "PrimitiveScalar",
    "ClassInfo",
    "FunctionInfo",
    "MethodInfo",
]


class BaseVisitor(PrimitiveMixin):
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
        self._function_scope_stack: list[dict[str, FunctionInfo]] = []
        self._module_name: str = "__main__"
        self._functions: dict[str, FunctionInfo] = {}
        self._classes: dict[str, ClassInfo] = {}
        self._function_effect_stack: list[bool] = []
        self._function_name_stack: list[str] = []
        self._returned_function_info_stack: list[FunctionInfo | None] = []
        self._returned_callable_arg_index_stack: list[int | None] = []
        self._returned_function_info_valid_stack: list[bool] = []
        self._function_ast_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self._return_type_stack: list[ir.Type] = []
        self._nested_function_counter: int = 0
        self._current_method_mutates_self: bool = False
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
        # Track primitive constants for cross-region access:
        # name -> (mlir_type, python_value)
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
            visitor._function_scope_stack = self._function_scope_stack
            visitor._module_name = self._module_name
            visitor._functions = self._functions
            visitor._classes = self._classes
            visitor._function_effect_stack = self._function_effect_stack
            visitor._function_name_stack = self._function_name_stack
            visitor._returned_function_info_stack = self._returned_function_info_stack
            visitor._returned_callable_arg_index_stack = (
                self._returned_callable_arg_index_stack
            )
            visitor._returned_function_info_valid_stack = (
                self._returned_function_info_valid_stack
            )
            visitor._function_ast_stack = self._function_ast_stack
            visitor._return_type_stack = self._return_type_stack
            visitor._nested_function_counter = self._nested_function_counter
            visitor._prim_types = self._prim_types
            visitor._lyrt_builtins = self._lyrt_builtins
            visitor._native_gc_mode = self._native_gc_mode
            visitor._prim_allocated = self._prim_allocated
            visitor._prim_deallocated = self._prim_deallocated
            visitor._prim_alloc_sites = self._prim_alloc_sites
            visitor._prim_constants = self._prim_constants

    # Note : _pending_prim_const is not shared - each visitor has its own

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

    def _set_func_effect(self, func: Any, maythrow: bool) -> None:
        if "nothrow" in func.attributes:
            del func.attributes["nothrow"]
        if "maythrow" in func.attributes:
            del func.attributes["maythrow"]
        if maythrow:
            func.attributes["maythrow"] = ir.UnitAttr.get(self.ctx)
        else:
            func.attributes["nothrow"] = ir.UnitAttr.get(self.ctx)

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
        if str(value.type).startswith('!py.class<"'):
            raise TypeError("Static class instances cannot be converted to !py.object")
        if not self.is_py_type(value.type):
            raise TypeError(
                f"Cannot convert primitive value {value.type} to !py.object"
            )
        location = loc or ir.Location.unknown(self.ctx)
        with location, self.insertion_point():
            return py_ops.UpcastOp(object_type, value).result

    def annotation_to_static_class_type(self, annotation: ast.expr | None) -> ir.Type:
        prim_type = self.annotation_to_primitive_type(annotation)
        if prim_type is not None:
            return prim_type
        return self.get_py_type(self.annotation_to_py_type(annotation))

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
            if self.lookup_class(annotation.id) is not None:
                return f'!py.class<"{annotation.id}">'
        if isinstance(annotation, ast.Subscript):
            if (
                isinstance(annotation.value, ast.Name)
                and annotation.value.id == "Callable"
            ):
                return self._callable_annotation_to_py_type(annotation)
            if isinstance(annotation.value, ast.Name) and annotation.value.id == "list":
                element_spec = self.annotation_to_py_type(annotation.slice)
                return f"!py.list<{element_spec}>"
            if (
                isinstance(annotation.value, ast.Name)
                and annotation.value.id == "tuple"
            ):
                if isinstance(annotation.slice, ast.Tuple):
                    if (
                        len(annotation.slice.elts) == 2
                        and isinstance(annotation.slice.elts[1], ast.Constant)
                        and annotation.slice.elts[1].value is Ellipsis
                    ):
                        element_spec = self.annotation_to_py_type(
                            annotation.slice.elts[0]
                        )
                        return f"!py.tuple<{element_spec}>"
                    element_specs = [
                        self.annotation_to_py_type(element)
                        for element in annotation.slice.elts
                    ]
                    return f"!py.tuple<{', '.join(element_specs)}>"
                element_spec = self.annotation_to_py_type(annotation.slice)
                return f"!py.tuple<{element_spec}>"
            if isinstance(annotation.value, ast.Name) and annotation.value.id == "dict":
                if (
                    not isinstance(annotation.slice, ast.Tuple)
                    or len(annotation.slice.elts) != 2
                ):
                    raise NotImplementedError(
                        "dict annotation must have the form dict[K, V]"
                    )
                key_spec = self.annotation_to_py_type(annotation.slice.elts[0])
                value_spec = self.annotation_to_py_type(annotation.slice.elts[1])
                return f"!py.dict<{key_spec}, {value_spec}>"
        raise NotImplementedError(f"Unsupported annotation {ast.dump(annotation)}")

    def _callable_annotation_to_py_type(self, annotation: ast.Subscript) -> str:
        slice_expr = annotation.slice
        if not isinstance(slice_expr, ast.Tuple) or len(slice_expr.elts) != 2:
            raise NotImplementedError(
                "Callable annotation must have the form Callable[[...], Ret]"
            )

        args_expr, result_expr = slice_expr.elts
        if isinstance(args_expr, ast.List):
            arg_exprs = list(args_expr.elts)
        elif isinstance(args_expr, ast.Tuple):
            arg_exprs = list(args_expr.elts)
        else:
            raise NotImplementedError(
                "Callable argument list must be written as [T1, T2, ...]"
            )

        arg_types = [self.annotation_to_py_type(arg) for arg in arg_exprs]
        result_type = self.annotation_to_py_type(result_expr)
        funcsig = self.build_funcsig(arg_types, [result_type])
        return f"!py.func<{funcsig}>"

    def build_funcsig(
        self,
        arg_types: list[str],
        result_types: list[str],
        *,
        kwonly_types: list[str] | None = None,
        vararg_type: str | None = None,
        kwargs_type: str | None = None,
    ) -> str:
        args = ", ".join(arg_types)
        rets = ", ".join(result_types)
        arg_part = f"[{args}]" if args else "[]"
        ret_part = f"[{rets}]" if rets else "[]"
        extras: list[str] = []
        if vararg_type is not None:
            extras.append(f"vararg = {vararg_type}")
        if kwonly_types:
            extras.append(f"kwonly = [{', '.join(kwonly_types)}]")
        if kwargs_type is not None:
            extras.append(f"kwargs = {kwargs_type}")
        suffix = f", {', '.join(extras)}" if extras else ""
        return f"!py.funcsig<{arg_part}{suffix} -> {ret_part}>"

    def _split_top_level_specs(self, text: str) -> list[str]:
        parts: list[str] = []
        depth_angle = 0
        depth_square = 0
        current: list[str] = []
        for ch in text:
            if ch == "<":
                depth_angle += 1
            elif ch == ">":
                depth_angle -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
            elif ch == "," and depth_angle == 0 and depth_square == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)
        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    def _split_top_level_arrow(self, text: str) -> tuple[str, str]:
        depth_angle = 0
        depth_square = 0
        for index in range(len(text) - 1):
            ch = text[index]
            if ch == "<":
                depth_angle += 1
            elif ch == ">":
                depth_angle -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
            elif (
                ch == "-"
                and text[index + 1] == ">"
                and depth_angle == 0
                and depth_square == 0
            ):
                return text[:index].strip(), text[index + 2 :].strip()
        raise ValueError(f"Malformed callable signature: {text}")

    def _parse_type_list(self, text: str) -> list[str]:
        stripped = text.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            raise ValueError(f"Expected type list, got: {text}")
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return self._split_top_level_specs(inner)

    def build_function_info_from_callable_type(
        self,
        name: str,
        func_type: ir.Type,
        *,
        maythrow: bool = True,
    ) -> FunctionInfo | None:
        type_spec = str(func_type)
        prefix = "!py.func<!py.funcsig<"
        if not (type_spec.startswith(prefix) and type_spec.endswith(">>")):
            return None
        inner = type_spec[len(prefix) : -2]
        lhs, rhs = self._split_top_level_arrow(inner)
        positional_specs: list[str] = []
        kwonly_specs: list[str] = []
        has_vararg = False
        has_kwargs = False

        lhs_parts = self._split_top_level_specs(lhs)
        if not lhs_parts:
            raise ValueError(f"Malformed callable type: {type_spec}")
        positional_specs = self._parse_type_list(lhs_parts[0])
        for extra in lhs_parts[1:]:
            if extra.startswith("kwonly = "):
                kwonly_specs = self._parse_type_list(extra[len("kwonly = ") :])
            elif extra.startswith("vararg = "):
                has_vararg = True
            elif extra.startswith("kwargs = "):
                has_kwargs = True

        result_specs = self._parse_type_list(rhs)
        return FunctionInfo(
            symbol=f"<param {name}>",
            func_type=func_type,
            arg_types=tuple(self.get_py_type(spec) for spec in positional_specs),
            result_types=tuple(self.get_py_type(spec) for spec in result_specs),
            has_vararg=has_vararg,
            maythrow=maythrow,
            kwonly_arg_types=tuple(self.get_py_type(spec) for spec in kwonly_specs),
            has_kwargs=has_kwargs,
        )

    def maybe_define_callable_parameter_binding(
        self, name: str, type_spec: str, value: ir.Value
    ) -> FunctionInfo | None:
        if not type_spec.startswith("!py.func<"):
            return None
        info = self.build_function_info_from_callable_type(
            name, value.type, maythrow=True
        )
        if info is not None:
            self.define_function_binding(name, info)
        return info

    def collect_callable_default_infos(
        self, defaults: list[ast.expr | None] | tuple[ast.expr | None, ...]
    ) -> tuple[FunctionInfo | None, ...]:
        infos: list[FunctionInfo | None] = []
        for default in defaults:
            if default is None:
                infos.append(None)
            else:
                infos.append(self.resolve_function_info_from_ast(default))
        return tuple(infos)

    def annotate_known_callable_value(
        self,
        value: ir.Value,
        info: FunctionInfo | None,
        *,
        loc: ir.Location,
    ) -> ir.Value:
        if info is None:
            return value
        with loc, self.insertion_point():
            wrapped = py_ops.CastIdentityOp(value.type, value).result
        op = self._value_operation(wrapped)
        if op is None:
            return wrapped
        op.attributes["lython.returned_callable_symbol"] = ir.FlatSymbolRefAttr.get(
            info.symbol, self.ctx
        )
        op.attributes["lython.returned_callable_defaults_count"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64, context=self.ctx),
            info.defaults_count,
        )
        if info.kwdefault_names:
            op.attributes["lython.returned_callable_kwdefault_names"] = self.array_attr(
                [ir.StringAttr.get(name, self.ctx) for name in info.kwdefault_names],
            )
        return wrapped

    def array_attr(self, attributes: list[Any]) -> ir.ArrayAttr:
        return ir.ArrayAttr.get(  # pyright: ignore[reportUnknownMemberType]
            attributes, context=self.ctx
        )

    # --- Scope management -------------------------------------------------
    def push_scope(self) -> None:
        self._scope_stack.append({})
        self._function_scope_stack.append({})
        for visitor in self.subvisitors.values():
            visitor._scope_stack = self._scope_stack
            visitor._function_scope_stack = self._function_scope_stack

    def pop_scope(self) -> dict[str, ir.Value]:
        if not self._scope_stack:
            raise RuntimeError("Scope stack underflow")
        scope = self._scope_stack.pop()
        if not self._function_scope_stack:
            raise RuntimeError("Function scope stack underflow")
        self._function_scope_stack.pop()
        for visitor in self.subvisitors.values():
            visitor._scope_stack = self._scope_stack
            visitor._function_scope_stack = self._function_scope_stack
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
        maythrow: bool = False,
        arg_names: list[str] | None = None,
        kwonly_arg_types: list[ir.Type] | None = None,
        kwonly_names: list[str] | None = None,
        kwdefault_names: list[str] | None = None,
        defaults_count: int = 0,
        positional_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None] | None
        ) = None,
        kwonly_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None] | None
        ) = None,
        defaults: ir.Value | None = None,
        kwdefaults: ir.Value | None = None,
        has_kwargs: bool = False,
        returned_function_info: FunctionInfo | None = None,
        returned_callable_arg_index: int | None = None,
        closure: ir.Value | None = None,
        closure_capture_arg_indices: (
            tuple[int | None, ...] | list[int | None] | None
        ) = None,
    ) -> None:
        info = FunctionInfo(
            symbol or name,
            func_type,
            tuple(arg_types),
            tuple(result_types),
            has_vararg,
            maythrow,
            tuple(arg_names or ()),
            tuple(kwonly_arg_types or ()),
            tuple(kwonly_names or ()),
            tuple(kwdefault_names or ()),
            defaults_count,
            tuple(positional_default_callable_infos or ()),
            tuple(kwonly_default_callable_infos or ()),
            defaults,
            kwdefaults,
            has_kwargs,
            returned_function_info,
            returned_callable_arg_index,
            closure,
            tuple(closure_capture_arg_indices or ()),
        )
        self._functions[name] = info

    def define_function_binding(self, name: str, info: FunctionInfo) -> None:
        if not self._function_scope_stack:
            raise RuntimeError("Function scope stack is empty")
        self._function_scope_stack[-1][name] = info

    def undefine_function_binding(self, name: str) -> None:
        if not self._function_scope_stack:
            raise RuntimeError("Function scope stack is empty")
        self._function_scope_stack[-1].pop(name, None)

    def lookup_function_binding(self, name: str) -> FunctionInfo:
        for scope in reversed(self._function_scope_stack):
            if name in scope:
                return scope[name]
        raise NameError(f"Unknown bound function '{name}'")

    def lookup_function(self, name: str) -> FunctionInfo:
        if name not in self._functions:
            raise NameError(f"Unknown function '{name}'")
        return self._functions[name]

    def lookup_function_by_symbol(self, symbol: str) -> FunctionInfo:
        for scope in reversed(self._function_scope_stack):
            for info in scope.values():
                if info.symbol == symbol:
                    return info
        for info in self._functions.values():
            if info.symbol == symbol:
                return info
        raise NameError(f"Unknown function symbol '{symbol}'")

    def _value_operation(self, value: ir.Value) -> ir.Operation | None:
        owner = getattr(value, "owner", None)
        opview = getattr(owner, "opview", None)
        return getattr(opview, "operation", owner)

    def _extract_make_function_operands(self, op: ir.Operation) -> tuple[
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
    ]:
        operands = list(getattr(op, "operands", ()))
        attributes = getattr(op, "attributes", {})
        segment_attr = (
            attributes["operandSegmentSizes"]
            if "operandSegmentSizes" in attributes
            else None
        )
        if segment_attr is None:
            return (None, None, None, None, None)
        segment_sizes = [int(size) for size in segment_attr]
        segment_index = 0
        operand_index = 0

        def take_optional() -> ir.Value | None:
            nonlocal segment_index, operand_index
            if segment_index >= len(segment_sizes):
                return None
            size = segment_sizes[segment_index]
            result = operands[operand_index] if size else None
            segment_index += 1
            operand_index += size
            return result

        return (
            take_optional(),
            take_optional(),
            take_optional(),
            take_optional(),
            take_optional(),
        )

    def resolve_function_info_from_value(self, value: ir.Value) -> FunctionInfo | None:
        if not str(value.type).startswith("!py.func<"):
            return None

        current = value
        while True:
            op = self._value_operation(current)
            if op is None:
                return None

            op_name = str(getattr(op, "name", ""))
            if op_name == "py.cast.identity":
                operands = list(getattr(op, "operands", ()))
                if operands:
                    inner_info = self.resolve_function_info_from_value(operands[0])
                    if inner_info is not None:
                        return inner_info
                attributes = getattr(op, "attributes", {})
                returned_symbol_attr = (
                    attributes["lython.returned_callable_symbol"]
                    if "lython.returned_callable_symbol" in attributes
                    else None
                )
                if returned_symbol_attr is None:
                    return None
                symbol = str(returned_symbol_attr)
                if symbol.startswith("@"):
                    symbol = symbol[1:]
                try:
                    return self.lookup_function_by_symbol(symbol)
                except NameError:
                    return None
            if op_name == "py.publish":
                operands = list(getattr(op, "operands", ()))
                if not operands:
                    return None
                current = operands[0]
                continue

            attributes = getattr(op, "attributes", {})
            if op_name in {"py.func.object", "py.make_function"}:
                target_attr = attributes["target"] if "target" in attributes else None
                if target_attr is None:
                    return None
                symbol = str(target_attr)
                if symbol.startswith("@"):
                    symbol = symbol[1:]
                try:
                    info = self.lookup_function_by_symbol(symbol)
                except NameError:
                    return None
                if op_name == "py.func.object":
                    return info
                defaults, kwdefaults, closure, _, _ = (
                    self._extract_make_function_operands(op)
                )
                return info._replace(
                    func_type=value.type,
                    defaults=defaults,
                    kwdefaults=kwdefaults,
                    closure=closure,
                )

            return None

    def resolve_function_info_from_expression(
        self, value_node: ast.expr, value: ir.Value
    ) -> FunctionInfo | None:
        value_info = self.resolve_function_info_from_value(value)
        ast_info = self.resolve_function_info_from_ast(value_node)
        if value_info is None:
            return ast_info
        if ast_info is None:
            return value_info
        if value_info.symbol != ast_info.symbol:
            return value_info
        return value_info._replace(
            func_type=ast_info.func_type or value_info.func_type,
            arg_types=ast_info.arg_types or value_info.arg_types,
            result_types=ast_info.result_types or value_info.result_types,
            has_vararg=ast_info.has_vararg or value_info.has_vararg,
            maythrow=ast_info.maythrow or value_info.maythrow,
            arg_names=ast_info.arg_names or value_info.arg_names,
            kwonly_arg_types=ast_info.kwonly_arg_types or value_info.kwonly_arg_types,
            kwonly_names=ast_info.kwonly_names or value_info.kwonly_names,
            kwdefault_names=ast_info.kwdefault_names or value_info.kwdefault_names,
            defaults_count=ast_info.defaults_count or value_info.defaults_count,
            positional_default_callable_infos=(
                ast_info.positional_default_callable_infos
                or value_info.positional_default_callable_infos
            ),
            kwonly_default_callable_infos=(
                ast_info.kwonly_default_callable_infos
                or value_info.kwonly_default_callable_infos
            ),
            defaults=(
                value_info.defaults
                if value_info.defaults is not None
                else ast_info.defaults
            ),
            kwdefaults=(
                value_info.kwdefaults
                if value_info.kwdefaults is not None
                else ast_info.kwdefaults
            ),
            has_kwargs=ast_info.has_kwargs or value_info.has_kwargs,
            returned_function_info=(
                ast_info.returned_function_info
                if ast_info.returned_function_info is not None
                else value_info.returned_function_info
            ),
            returned_callable_arg_index=(
                ast_info.returned_callable_arg_index
                if ast_info.returned_callable_arg_index is not None
                else value_info.returned_callable_arg_index
            ),
            closure=(
                value_info.closure
                if value_info.closure is not None
                else ast_info.closure
            ),
            closure_capture_arg_indices=(
                value_info.closure_capture_arg_indices
                or ast_info.closure_capture_arg_indices
            ),
        )

    def _select_argument_expression_for_callable_summary(
        self,
        call_node: ast.Call,
        *,
        returned_callable_arg_index: int | None,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        leading_exprs: list[ast.expr] | None = None,
    ) -> ast.expr | None:
        if returned_callable_arg_index is None:
            return None

        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)
        leading_exprs = list(leading_exprs or [])
        positional_exprs = [*leading_exprs, *list(call_node.args)]

        if returned_callable_arg_index < len(positional_exprs):
            return positional_exprs[returned_callable_arg_index]

        if returned_callable_arg_index < len(positional_param_names):
            param_name = positional_param_names[returned_callable_arg_index]
        else:
            kw_index = returned_callable_arg_index - len(positional_param_names)
            if kw_index < 0 or kw_index >= len(kwonly_names):
                return None
            param_name = kwonly_names[kw_index]

        for keyword in call_node.keywords:
            if keyword.arg == param_name:
                return keyword.value
        return None

    def _resolve_omitted_default_callable_info(
        self,
        *,
        returned_callable_arg_index: int | None,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        defaults_count: int = 0,
        positional_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None]
        ) = (),
        kwonly_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None]
        ) = (),
    ) -> FunctionInfo | None:
        if returned_callable_arg_index is None:
            return None

        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)
        positional_default_callable_infos = list(positional_default_callable_infos)
        kwonly_default_callable_infos = list(kwonly_default_callable_infos)

        if returned_callable_arg_index < len(positional_param_names):
            default_start = len(positional_param_names) - defaults_count
            if (
                defaults_count <= 0
                or default_start < 0
                or returned_callable_arg_index < default_start
            ):
                return None
            default_index = returned_callable_arg_index - default_start
            if 0 <= default_index < len(positional_default_callable_infos):
                return positional_default_callable_infos[default_index]
            return None

        kw_index = returned_callable_arg_index - len(positional_param_names)
        if kw_index < 0 or kw_index >= len(kwonly_names):
            return None
        if 0 <= kw_index < len(kwonly_default_callable_infos):
            return kwonly_default_callable_infos[kw_index]
        return None

    def resolve_function_info_from_ast(
        self, value_node: ast.expr
    ) -> FunctionInfo | None:

        if isinstance(value_node, ast.Name):
            try:
                return self.lookup_function_binding(value_node.id)
            except NameError:
                try:
                    return self.lookup_function(value_node.id)
                except NameError:
                    return None

        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name):
            try:
                callee_info = self.lookup_function_binding(value_node.func.id)
            except NameError:
                try:
                    callee_info = self.lookup_function(value_node.func.id)
                except NameError:
                    callee_info = None
            if callee_info is not None:
                if callee_info.returned_function_info is not None:
                    return callee_info.returned_function_info
                selected = self._select_argument_expression_for_callable_summary(
                    value_node,
                    returned_callable_arg_index=callee_info.returned_callable_arg_index,
                    positional_param_names=callee_info.arg_names,
                    kwonly_names=callee_info.kwonly_names,
                )
                if selected is not None:
                    return self.resolve_function_info_from_ast(selected)
                return self._resolve_omitted_default_callable_info(
                    returned_callable_arg_index=callee_info.returned_callable_arg_index,
                    positional_param_names=callee_info.arg_names,
                    kwonly_names=callee_info.kwonly_names,
                    defaults_count=callee_info.defaults_count,
                    positional_default_callable_infos=callee_info.positional_default_callable_infos,
                    kwonly_default_callable_infos=callee_info.kwonly_default_callable_infos,
                )

        if isinstance(value_node, ast.Call) and isinstance(
            value_node.func, ast.Attribute
        ):
            receiver_type = self.resolve_static_expression_type(value_node.func.value)
            if receiver_type is not None:
                class_info = self.get_class_info_from_type(receiver_type)
                if class_info is not None:
                    method_info = class_info.methods.get(value_node.func.attr)
                    if method_info is not None:
                        if method_info.returned_function_info is not None:
                            return method_info.returned_function_info
                        selected = self._select_argument_expression_for_callable_summary(
                            value_node,
                            returned_callable_arg_index=method_info.returned_callable_arg_index,
                            positional_param_names=method_info.arg_names,
                            kwonly_names=method_info.kwonly_names,
                            leading_exprs=[value_node.func.value],
                        )
                        if selected is not None:
                            return self.resolve_function_info_from_ast(selected)
                        return self._resolve_omitted_default_callable_info(
                            returned_callable_arg_index=method_info.returned_callable_arg_index,
                            positional_param_names=method_info.arg_names,
                            kwonly_names=method_info.kwonly_names,
                            defaults_count=method_info.defaults_count,
                            positional_default_callable_infos=method_info.positional_default_callable_infos,
                            kwonly_default_callable_infos=method_info.kwonly_default_callable_infos,
                        )

        return None

    def resolve_current_function_parameter_index_from_value(
        self, value: ir.Value
    ) -> int | None:
        current = value
        while True:
            op = self._value_operation(current)
            if op is None:
                break
            if str(getattr(op, "name", "")) != "py.cast.identity":
                break
            operands = list(getattr(op, "operands", ()))
            if not operands:
                break
            current = operands[0]

        if hasattr(current, "arg_number"):
            try:
                return int(getattr(current, "arg_number"))
            except Exception:
                pass
        owner = getattr(current, "owner", None)
        if owner is None:
            return None
        arguments = getattr(owner, "arguments", None)
        if arguments is not None:
            try:
                return list(arguments).index(current)
            except Exception:
                pass
        if not hasattr(owner, "arg_number"):
            return None
        try:
            return int(owner.arg_number)
        except Exception:
            return None

    def resolve_current_function_parameter_index_from_expression(
        self, expr: ast.expr
    ) -> int | None:
        if isinstance(expr, ast.Name):
            current = self.current_function_ast()
            if current is None:
                return None
            for index, arg in enumerate(current.args.args):
                if arg.arg == expr.id:
                    return index
            kwonly_offset = len(current.args.args)
            for index, arg in enumerate(current.args.kwonlyargs):
                if arg.arg == expr.id:
                    return kwonly_offset + index
            return None

        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            try:
                callee_info = self.lookup_function_binding(expr.func.id)
            except NameError:
                try:
                    callee_info = self.lookup_function(expr.func.id)
                except NameError:
                    callee_info = None
            if callee_info is None:
                return None
            selected = self._select_argument_expression_for_callable_summary(
                expr,
                returned_callable_arg_index=callee_info.returned_callable_arg_index,
                positional_param_names=callee_info.arg_names,
                kwonly_names=callee_info.kwonly_names,
            )
            if selected is None:
                return None
            return self.resolve_current_function_parameter_index_from_expression(
                selected
            )

        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            receiver_type = self.resolve_static_expression_type(expr.func.value)
            if receiver_type is None:
                return None
            class_info = self.get_class_info_from_type(receiver_type)
            if class_info is None:
                return None
            method_info = class_info.methods.get(expr.func.attr)
            if method_info is None:
                return None
            selected = self._select_argument_expression_for_callable_summary(
                expr,
                returned_callable_arg_index=method_info.returned_callable_arg_index,
                positional_param_names=method_info.arg_names,
                kwonly_names=method_info.kwonly_names,
                leading_exprs=[expr.func.value],
            )
            if selected is None:
                return None
            return self.resolve_current_function_parameter_index_from_expression(
                selected
            )

        return None

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

    def push_function_ast(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_ast_stack.append(node)
        for visitor in self.subvisitors.values():
            visitor._function_ast_stack = self._function_ast_stack

    def pop_function_ast(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        if not self._function_ast_stack:
            raise RuntimeError("Function AST stack underflow")
        node = self._function_ast_stack.pop()
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

    def lookup_method_by_symbol(self, symbol: str) -> MethodInfo | None:
        for class_info in self._classes.values():
            for method in class_info.methods.values():
                if f"{class_info.name}.{method.name}" == symbol:
                    return method
        return None

    def get_class_info_from_type(self, obj_type: ir.Type) -> ClassInfo | None:
        obj_type_str = str(obj_type)
        if not (obj_type_str.startswith('!py.class<"') and obj_type_str.endswith('">')):
            return None
        class_name = obj_type_str[len('!py.class<"') : -len('">')]  # noqa
        return self.lookup_class(class_name)

    def get_list_element_type(self, list_type: ir.Type) -> ir.Type | None:
        list_type_str = str(list_type)
        if not (list_type_str.startswith("!py.list<") and list_type_str.endswith(">")):
            return None
        element_spec = list_type_str[len("!py.list<") : -1]
        return self.get_py_type(element_spec)

    def get_dict_key_value_types(
        self, dict_type: ir.Type
    ) -> tuple[ir.Type, ir.Type] | None:
        dict_type_str = str(dict_type)
        if not (dict_type_str.startswith("!py.dict<") and dict_type_str.endswith(">")):
            return None
        inner = dict_type_str[len("!py.dict<") : -1]
        parts = self._split_top_level_specs(inner)
        if len(parts) != 2:
            return None
        return self.get_py_type(parts[0]), self.get_py_type(parts[1])

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

    def resolve_static_expression_type(self, expr: ast.expr) -> ir.Type | None:
        if isinstance(expr, ast.Name):
            try:
                return self.lookup_symbol(expr.id).type
            except NameError:
                class_info = self.lookup_class(expr.id)
                if class_info is not None:
                    return class_info.class_type
                try:
                    return self.lookup_function_binding(expr.id).func_type
                except NameError:
                    try:
                        return self.lookup_function(expr.id).func_type
                    except NameError:
                        return None

        if isinstance(expr, ast.Attribute):
            base_type = self.resolve_static_expression_type(expr.value)
            if base_type is None:
                return None
            return self.get_attribute_type(base_type, expr.attr)

        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                class_info = self.lookup_class(expr.func.id)
                if class_info is not None:
                    return class_info.class_type
                try:
                    func_info = self.lookup_function_binding(expr.func.id)
                except NameError:
                    try:
                        func_info = self.lookup_function(expr.func.id)
                    except NameError:
                        func_info = None
                if func_info is not None and len(func_info.result_types) == 1:
                    return func_info.result_types[0]
            if isinstance(expr.func, ast.Attribute):
                receiver_type = self.resolve_static_expression_type(expr.func.value)
                if receiver_type is None:
                    return None
                class_info = self.get_class_info_from_type(receiver_type)
                if class_info is None:
                    return None
                method_info = class_info.methods.get(expr.func.attr)
                if method_info is None or len(method_info.result_types) != 1:
                    return None
                return method_info.result_types[0]

        return None

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

    def _block_terminated(self, block: ir.Block) -> bool:
        ops = list(block.operations)
        if not ops:
            return False
        terminators = {
            "py.return",
            "py.raise",
            "py.raise.current",
            "py.invoke",
            "func.return",
            "cf.br",
            "cf.cond_br",
        }
        return ops[-1].operation.name in terminators

    def _advance_block_after_terminator(self) -> None:
        """Move insertion to a fresh block if the current block is terminated."""
        block = self.current_block
        if block is None:
            return
        ops = list(block.operations)
        if not ops:
            return
        terminators = {
            "py.return",
            "py.raise",
            "py.raise.current",
            "py.invoke",
            "func.return",
            "cf.br",
            "cf.cond_br",
        }
        if ops[-1].operation.name not in terminators:
            return
        # Do not create an unreachable empty block. Callers should
        # explicitly set a new insertion block when control can continue.
        self._set_insertion_block(None)
