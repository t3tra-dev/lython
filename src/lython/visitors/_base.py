from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Mapping
from typing import NoReturn, cast

from ..frontend.symbols import ClassInfo, FunctionInfo, MethodInfo
from ..frontend.types import TypeResolver
from ..mlir import ir
from ..mlir.dialects import linalg as linalg_ops
from .base import PrimitiveMixin
from .base import primitive as _primitive
from .callable_metadata import CallableMetadataMixin
from .contracts import VisitorRuntime
from .emission import EmissionMixin
from .function_context import FunctionContextMixin
from .models import FinallyReturnContext, TypedProgram, VisitResult
from .symbols import SymbolTableMixin
from .type_bridge import TypeBridgeMixin
from .typed import TypedOverlayMixin
from .value_materialization import ValueMaterializationMixin

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


class BaseVisitor(
    CallableMetadataMixin,
    FunctionContextMixin,
    SymbolTableMixin,
    TypedOverlayMixin,
    TypeBridgeMixin,
    ValueMaterializationMixin,
    EmissionMixin,
    PrimitiveMixin,
):
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
        subvisitors: Mapping[str, VisitorRuntime] | None = None,
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
        self._callable_value_info: dict[int, FunctionInfo] = {}
        self._function_ast_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self._return_type_stack: list[ir.Type] = []
        self._async_function_stack: list[bool] = []
        self._finally_return_stack: list[FinallyReturnContext] = []
        self._finally_return_stack_save: list[list[FinallyReturnContext] | None] = []
        self._exception_context_stack: list[ir.Value] = []
        self._exception_context_stack_save: list[list[ir.Value] | None] = []
        self._nested_function_counter: int = 0
        self._current_class: str | None = None
        self._current_class_definition_block: ir.Block | None = None
        self._current_method: str | None = None
        self._current_method_mutates_self: bool = False
        self._pending_attributes: dict[str, ir.Type] | None = None
        # Primitive world support
        self._in_native_func: bool = False  # True when inside @native function
        self._native_gc_mode: str | None = None
        self._prim_types: dict[str, str] = {}  # Imported primitive types: name -> kind
        self._static_modules: dict[str, str] = {}
        self._static_module_symbols: dict[str, tuple[str, str]] = {}
        self._typed_program: TypedProgram | None = None
        self._type_resolver = TypeResolver(
            ctx=self.ctx,
            parse_type=self.get_py_type,
            primitive_annotation=self.annotation_to_primitive_type,
            class_lookup=self.lookup_class,
            symbol_type_lookup=self._lookup_symbol_type_or_none,
            function_binding_lookup=self._lookup_function_binding_or_none,
            function_lookup=self._lookup_function_or_none,
            static_modules=self._static_modules,
        )
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
            self.subvisitors = dict(subvisitors)
            return

        created_subvisitors: dict[str, VisitorRuntime] = {}
        self.subvisitors = created_subvisitors

        from .expr import ExprVisitor
        from .mod import ModVisitor
        from .stmt import StmtVisitor

        created_subvisitors["Module"] = ModVisitor(ctx, subvisitors=created_subvisitors)
        created_subvisitors["Stmt"] = StmtVisitor(ctx, subvisitors=created_subvisitors)
        created_subvisitors["Expr"] = ExprVisitor(ctx, subvisitors=created_subvisitors)
        for visitor in created_subvisitors.values():
            visitor.subvisitors = created_subvisitors
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
            visitor._callable_value_info = self._callable_value_info
            visitor._function_ast_stack = self._function_ast_stack
            visitor._return_type_stack = self._return_type_stack
            visitor._async_function_stack = self._async_function_stack
            visitor._finally_return_stack = self._finally_return_stack
            visitor._finally_return_stack_save = self._finally_return_stack_save
            visitor._exception_context_stack = self._exception_context_stack
            visitor._exception_context_stack_save = self._exception_context_stack_save
            visitor._nested_function_counter = self._nested_function_counter
            visitor._current_class = self._current_class
            visitor._current_class_definition_block = (
                self._current_class_definition_block
            )
            visitor._current_method = self._current_method
            visitor._current_method_mutates_self = self._current_method_mutates_self
            visitor._pending_attributes = self._pending_attributes
            visitor._prim_types = self._prim_types
            visitor._static_modules = self._static_modules
            visitor._static_module_symbols = self._static_module_symbols
            visitor._typed_program = self._typed_program
            visitor._type_resolver = self._type_resolver
            visitor._lyrt_builtins = self._lyrt_builtins
            visitor._native_gc_mode = self._native_gc_mode
            visitor._prim_allocated = self._prim_allocated
            visitor._prim_deallocated = self._prim_deallocated
            visitor._prim_alloc_sites = self._prim_alloc_sites
            visitor._prim_constants = self._prim_constants

    # Note: _pending_prim_const is not shared; each visitor has its own.

    def _ensure_mlir_dialect_aliases(self) -> None:
        sys.modules.setdefault("mlir.dialects.linalg", linalg_ops)
        linalg_gen = sys.modules.get("lython.mlir.dialects._linalg_ops_gen")
        if linalg_gen is not None:
            sys.modules.setdefault("mlir.dialects._linalg_ops_gen", linalg_gen)
        linalg_enum = sys.modules.get("lython.mlir.dialects._linalg_enum_gen")
        if linalg_enum is not None:
            sys.modules.setdefault("mlir.dialects._linalg_enum_gen", linalg_enum)

    def visit(self, node: ast.AST) -> VisitResult:
        method_name = f"visit_{type(node).__name__}"

        # 1) このビジター自身が実装していれば呼び出し
        visit_method: Callable[[BaseVisitor, ast.AST], VisitResult] | None = None
        for cls in type(self).__mro__:
            method = cls.__dict__.get(method_name)
            if method is not None:
                visit_method = cast(
                    Callable[[BaseVisitor, ast.AST], VisitResult], method
                )
                break
        if visit_method is not None:
            return visit_method(self, node)

        # 2) 同一クラス名に対する直接委譲
        name = type(node).__name__
        visitor = self.subvisitors.get(name)
        if visitor is not None and visitor is not self:
            result = visitor.visit(node)
            if isinstance(node, ast.mod):
                self.module = visitor.module
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
                self.module = v.module
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
