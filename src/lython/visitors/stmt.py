from __future__ import annotations

import ast
from typing import Any

from lython.mlir.dialects import _lython_ops_gen as py_ops
from lython.mlir.dialects import arith as arith_ops
from lython.mlir.dialects import cf as cf_ops
from lython.mlir.dialects import func as func_ops

from ..mlir import ir
from ._base import PRIMITIVE_TYPE_MAP, BaseVisitor, MethodInfo  # noqa: F401

__all__ = ["StmtVisitor"]


class StmtVisitor(BaseVisitor):
    """
    文(stmt)ノードを訪問し、
    MLIRを生成するクラス。

    ```asdl
    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns,
                       string? type_comment, type_param* type_params)
          | AsyncFunctionDef(identifier name, arguments args,
                             stmt* body, expr* decorator_list, expr? returns,
                             string? type_comment, type_param* type_params)

          | ClassDef(identifier name,
             expr* bases,
             keyword* keywords,
             stmt* body,
             expr* decorator_list,
             type_param* type_params)
          | Return(expr? value)

          | Delete(expr* targets)
          | Assign(expr* targets, expr value, string? type_comment)
          | TypeAlias(expr name, type_param* type_params, expr value)
          | AugAssign(expr target, operator op, expr value)
          -- 'simple' indicates that we annotate simple name without parens
          | AnnAssign(expr target, expr annotation, expr? value, int simple)

          -- use 'orelse' because else is a keyword in target languages
          | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
          | With(withitem* items, stmt* body, string? type_comment)
          | AsyncWith(withitem* items, stmt* body, string? type_comment)

          | Match(expr subject, match_case* cases)

          | Raise(expr? exc, expr? cause)
          | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | TryStar(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | Assert(expr test, expr? msg)

          | Import(alias* names)
          | ImportFrom(identifier? module, alias* names, int? level)

          | Global(identifier* names)
          | Nonlocal(identifier* names)
          | Expr(expr value)
          | Pass | Break | Continue

          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def _get_native_decorator(
        self, decorators: list[ast.expr]
    ) -> dict[str, Any] | None:
        """
        Check if function has @native decorator and extract its arguments.
        Returns dict with gc mode or None if not a native function.
        """
        for dec in decorators:
            # @native(gc="none") is ast.Call
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                if dec.func.id == "native" and "native" in self._lyrt_builtins:
                    result: dict[str, Any] = {"gc": "none"}  # default
                    for kw in dec.keywords:
                        if kw.arg == "gc" and isinstance(kw.value, ast.Constant):
                            result["gc"] = kw.value.value
                    return result
            # @native without parens is ast.Name
            if isinstance(dec, ast.Name) and dec.id == "native":
                if "native" in self._lyrt_builtins:
                    return {"gc": "none"}
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        関数定義:
        def hoge(n: int) -> int:
            ...
        などを受け取り、静的型のIRを生成する

        ```asdl
        FunctionDef(
            identifier name,
            arguments args,
            stmt* body,
            expr* decorator_list,
            expr? returns,
            string? type_comment,
            type_param* type_params
        )
        ```
        """
        # Check for @native decorator
        native_info = self._get_native_decorator(node.decorator_list)
        if native_info is not None:
            return self._visit_native_function_def(node, native_info)

        # Regular Python function
        arg_type_specs = [
            self.annotation_to_py_type(arg.annotation) for arg in node.args.args
        ]
        result_type_spec = self.annotation_to_py_type(node.returns)
        result_ir_type = self.get_py_type(result_type_spec)
        funcsig = self.build_funcsig(arg_type_specs, [result_type_spec])
        py_func_sig = self.get_py_type(funcsig)
        py_func_type = self.get_py_type(f"!py.func<{funcsig}>")
        arg_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.args
        ]
        arg_names_attr = (
            ir.ArrayAttr.get(arg_name_attrs, context=self.ctx)
            if arg_name_attrs
            else None
        )
        loc = self._loc(node)
        with loc, ir.InsertionPoint(self.module.body):
            func = py_ops.FuncOp(
                node.name,
                ir.TypeAttr.get(py_func_sig),
                arg_names=arg_names_attr,
            )
        entry_arg_types = [self.get_py_type(spec) for spec in arg_type_specs]
        self.register_function(
            node.name,
            py_func_type,
            entry_arg_types,
            [result_ir_type],
        )
        with loc:
            if entry_arg_types:
                entry_block = func.body.blocks.append(*entry_arg_types)
            else:
                entry_block = func.body.blocks.append()
        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()
        for arg, value in zip(node.args.args, entry_block.arguments):
            self.define_symbol(arg.arg, value)
        for stmt in node.body:
            self.visit(stmt)
        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            if result_type_spec != "!py.none":
                raise NotImplementedError(
                    f"Function '{node.name}' must explicitly return {result_type_spec}"
                )
            with ir.Location.unknown(self.ctx), ir.InsertionPoint(active_block):
                none_val = py_ops.NoneOp(self.get_py_type("!py.none")).result
                py_ops.ReturnOp([none_val])
        self.pop_scope()
        self._set_insertion_block(prev_block)

    def _visit_native_function_def(
        self, node: ast.FunctionDef, native_info: dict[str, Any]
    ) -> None:
        """
        Generate func.func for @native decorated functions.

        Native functions operate in the Primitive World:
        - Use MLIR primitive types (i8, i32, f64, etc.)
        - Generate arith.* operations instead of py.num.*
        - No GC involvement
        """
        # Get primitive types for arguments
        arg_types: list[ir.Type] = []
        for arg in node.args.args:
            prim_type = self.annotation_to_primitive_type(arg.annotation)
            if prim_type is None:
                raise ValueError(
                    f"@native function '{node.name}' argument '{arg.arg}' "
                    f"must have a primitive type annotation"
                )
            arg_types.append(prim_type)

        # Get primitive return type
        result_type = self.annotation_to_primitive_type(node.returns)
        if result_type is None:
            raise ValueError(
                f"@native function '{node.name}' must have a primitive return type annotation"
            )
        result_types = [result_type]

        loc = self._loc(node)

        # Create func.func operation with 'native' attribute
        # This attribute marks functions operating in the Primitive World (𝒫)
        # and enables static verification that no py.* types are used inside
        func_type = ir.FunctionType.get(arg_types, result_types, context=self.ctx)
        with loc, ir.InsertionPoint(self.module.body):
            func = func_ops.FuncOp(node.name, func_type)
            func.attributes["native"] = ir.UnitAttr.get(self.ctx)

        # Register the function with primitive types
        # Note: We use a special marker to indicate this is a native function
        self._register_native_function(node.name, arg_types, result_types)

        # Create entry block
        with loc:
            if arg_types:
                entry_block = func.body.blocks.append(*arg_types)
            else:
                entry_block = func.body.blocks.append()

        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()

        # Enter native mode
        self._set_in_native_func(True)

        # Register arguments in scope
        for arg, value in zip(node.args.args, entry_block.arguments):
            self.define_symbol(arg.arg, value)

        # Process function body
        for stmt in node.body:
            self.visit(stmt)

        # Check for missing return
        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            raise NotImplementedError(
                f"@native function '{node.name}' must explicitly return"
            )

        # Exit native mode
        self._set_in_native_func(False)

        self.pop_scope()
        self._set_insertion_block(prev_block)

    def _register_native_function(
        self,
        name: str,
        arg_types: list[ir.Type],
        result_types: list[ir.Type],
    ) -> None:
        """Register a native function in the function table."""
        from ._base import FunctionInfo

        # Create a fake py.func type for compatibility
        # Native functions are identified by their types being primitives
        func_type = ir.FunctionType.get(arg_types, result_types, context=self.ctx)
        info = FunctionInfo(
            symbol=name,
            func_type=func_type,
            arg_types=tuple(arg_types),
            result_types=tuple(result_types),
            has_vararg=False,
        )
        self._functions[name] = info

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        非同期関数の定義を処理する

        ```asdl
        AsyncFunctionDef(
            identifier name,
            arguments args,
            stmt* body,
            expr* decorator_list,
            expr? returns,
            string? type_comment,
            type_param* type_params
        )
        ```
        """
        raise NotImplementedError("Async function definition not supported")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        クラス定義

        ```asdl
        ClassDef(
            identifier name,
            expr* bases,
            keyword* keywords,
            stmt* body,
            expr* decorator_list,
            type_param* type_params
        )
        ```
        """
        if node.bases:
            raise NotImplementedError("Class inheritance not yet supported")
        if node.decorator_list:
            raise NotImplementedError("Class decorators not yet supported")

        class_name = node.name
        loc = self._loc(node)

        # Create py.class operation
        with loc, ir.InsertionPoint(self.module.body):
            class_op = py_ops.ClassOp(class_name)

        # Create a block in the class body for methods
        class_body_block = class_op.body.blocks.append()

        # Register the class type (class name must be quoted in type syntax)
        class_type = self.get_py_type(f'!py.class<"{class_name}">')

        # Process class body - only method definitions for now
        prev_block = self.current_block
        self._set_insertion_block(class_body_block)
        self.push_scope()

        # Set current class context for method processing
        prev_class = getattr(self, "_current_class", None)
        self._current_class = class_name

        # Track instance attributes and their types during class processing
        prev_pending_attrs = getattr(self, "_pending_attributes", None)
        self._pending_attributes: dict[str, ir.Type] = {}

        # Collect method info
        methods: dict[str, MethodInfo] = {}

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                method_info = self._visit_method_def(stmt, class_name, class_type)
                if method_info is not None:
                    methods[stmt.name] = method_info
            elif isinstance(stmt, ast.Pass):
                pass  # Ignore pass statements
            else:
                raise NotImplementedError(
                    f"Class body can only contain method definitions, got {type(stmt).__name__}"
                )

        # Capture collected attributes
        attributes = self._pending_attributes
        self._pending_attributes = prev_pending_attrs  # type: ignore
        self._current_class = prev_class
        self.pop_scope()
        self._set_insertion_block(prev_block)

        # Register the class with methods and attributes
        self.register_class(class_name, class_type, methods, attributes)

    def _visit_method_def(
        self, node: ast.FunctionDef, class_name: str, class_type: ir.Type
    ) -> "MethodInfo | None":
        """
        クラス内のメソッド定義を処理する
        Returns MethodInfo for the method.

        Methods are created at module scope with qualified names (e.g., Counter.__init__)
        to allow FuncObjectOp to reference them using flat symbol references.
        """
        from ._base import MethodInfo

        # First argument should be 'self'
        if not node.args.args:
            raise ValueError(f"Method '{node.name}' must have 'self' parameter")

        self_arg = node.args.args[0]
        if self_arg.arg != "self":
            raise ValueError(
                f"First parameter of method '{node.name}' must be 'self', got '{self_arg.arg}'"
            )

        # Build argument types - self is the class type
        arg_type_specs: list[str] = [f'!py.class<"{class_name}">']
        for arg in node.args.args[1:]:
            arg_type_specs.append(self.annotation_to_py_type(arg.annotation))

        result_type_spec = self.annotation_to_py_type(node.returns)
        result_ir_type = self.get_py_type(result_type_spec)
        funcsig = self.build_funcsig(arg_type_specs, [result_type_spec])
        py_func_sig = self.get_py_type(funcsig)

        arg_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.args
        ]
        arg_names_attr = (
            ir.ArrayAttr.get(arg_name_attrs, context=self.ctx)
            if arg_name_attrs
            else None
        )

        # Qualified method name at module scope (e.g., Counter.__init__)
        qualified_name = f"{class_name}.{node.name}"

        loc = self._loc(node)
        # Methods are created at module scope for flat symbol reference
        with loc, ir.InsertionPoint(self.module.body):
            func = py_ops.FuncOp(
                qualified_name,
                ir.TypeAttr.get(py_func_sig),
                arg_names=arg_names_attr,
            )

        entry_arg_types = [self.get_py_type(spec) for spec in arg_type_specs]
        with loc:
            if entry_arg_types:
                entry_block = func.body.blocks.append(*entry_arg_types)
            else:
                entry_block = func.body.blocks.append()

        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()

        # Register arguments in scope
        for arg, value in zip(node.args.args, entry_block.arguments):
            self.define_symbol(arg.arg, value)

        # Process method body
        for stmt in node.body:
            self.visit(stmt)

        # Handle implicit return
        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            if result_type_spec != "!py.none":
                raise NotImplementedError(
                    f"Method '{node.name}' must explicitly return {result_type_spec}"
                )
            with ir.Location.unknown(self.ctx), ir.InsertionPoint(active_block):
                none_val = py_ops.NoneOp(self.get_py_type("!py.none")).result
                py_ops.ReturnOp([none_val])

        self.pop_scope()
        self._set_insertion_block(prev_block)

        # Return method info
        return MethodInfo(
            name=node.name,
            arg_types=tuple(entry_arg_types),
            result_types=(result_ir_type,),
        )

    def visit_Return(self, node: ast.Return) -> None:
        """
        関数の返り値を定めるreturn文を処理する

        ```asdl
        Return(expr? value)
        ```
        """
        with self._loc(node), self.insertion_point():
            if node.value is None:
                if self._in_native_func:
                    raise ValueError("@native function cannot return None implicitly")
                value = py_ops.NoneOp(self.get_py_type("!py.none")).result
            else:
                value = self.require_value(node.value, self.visit(node.value))

            # Use func.return for native functions, py.return for Python functions
            if self._in_native_func:
                func_ops.ReturnOp([value])
            else:
                py_ops.ReturnOp([value])

    def visit_Delete(self, node: ast.Delete) -> None:
        """
        変数を削除するdelete文を処理する

        ```asdl
        Delete(expr* targets)
        ```
        """
        raise NotImplementedError("Delete statement not supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        代入演算子 = を処理する

        ```asdl
        Assign(expr* targets, expr value, string? type_comment)
        ```
        """
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")

        target = node.targets[0]

        # Clear any pending primitive constant from previous assignment
        expr_visitor = self.subvisitors.get("Expr")
        if expr_visitor:
            expr_visitor._pending_prim_const = None

        value = self.require_value(node.value, self.visit(node.value))

        if isinstance(target, ast.Name):
            # Simple assignment: x = value
            self.define_symbol(target.id, value)

            # Check if this was a to_prim() call with a constant value
            # If so, register the constant for cross-region access in @native functions
            if expr_visitor:
                pending = getattr(expr_visitor, "_pending_prim_const", None)
                if pending is not None:
                    mlir_type, const_value = pending
                    self.register_prim_constant(target.id, mlir_type, const_value)
                    expr_visitor._pending_prim_const = None
        elif isinstance(target, ast.Attribute):
            # Attribute assignment: obj.attr = value
            obj = self.require_value(target.value, self.visit(target.value))
            with self._loc(node), self.insertion_point():
                py_ops.AttrSetOp(obj, target.attr, value)

            # Track attribute type if this is a self.attr assignment in a class
            pending_attrs = getattr(self, "_pending_attributes", None)
            if (
                pending_attrs is not None
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                pending_attrs[target.attr] = value.type
        else:
            raise NotImplementedError(
                f"Assignment target type {type(target).__name__} not supported"
            )

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        """
        型エイリアスを処理する

        ```asdl
        TypeAlias(
            expr name,
            type_param* type_params,
            expr value
        )
        """
        raise NotImplementedError("Type alias statement not implemented")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """
        a += 1 のような累積代入を処理する

        ```asdl
        AugAssign(
            expr target,
            operator op,
            expr value
        )
        """
        rhs = self.require_value(node.value, self.visit(node.value))
        loc = self._loc(node)

        if isinstance(node.target, ast.Name):
            # Simple augmented assignment: x += value
            current = self.lookup_symbol(node.target.id)
            with loc, self.insertion_point():
                result = self._apply_binop(node.op, current, rhs)
            self.define_symbol(node.target.id, result)

        elif isinstance(node.target, ast.Attribute):
            # Attribute augmented assignment: obj.attr += value
            obj = self.require_value(node.target.value, self.visit(node.target.value))
            attr_type = self.get_attribute_type(obj.type, node.target.attr)

            with loc, self.insertion_point():
                current = py_ops.AttrGetOp(attr_type, obj, node.target.attr).result
                result = self._apply_binop(node.op, current, rhs)
                py_ops.AttrSetOp(obj, node.target.attr, result)

        else:
            raise NotImplementedError(
                f"Augmented assignment target type {type(node.target).__name__} not supported"
            )

    def _apply_binop(self, op: ast.operator, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """二項演算子を適用し、結果を返す"""
        if isinstance(op, ast.Add):
            return py_ops.NumAddOp(lhs, rhs).result
        elif isinstance(op, ast.Sub):
            return py_ops.NumSubOp(lhs, rhs).result
        else:
            raise NotImplementedError(f"Operator {type(op).__name__} not supported")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """
        c: int のような型注釈を持つ代入を処理する

        ```asdl
        -- 'simple' indicates that we annotate simple name without parens
        AnnAssign(
            expr target,
            expr annotation,
            expr? value,
            int simple
        )
        ```
        """
        raise NotImplementedError(
            "An assignment with a type annotation is not implemented"
        )

    def visit_For(self, node: ast.For) -> None:
        """
        for文の処理をする

        ```asdl
        For(
            expr target,
            expr iter,
            stmt* body,
            stmt* orelse,
            string? type_comment
        )
        ```
        """
        raise NotImplementedError("For statement not implemented")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """
        非同期for文の処理をする

        ```asdl
        AsyncFor(
            expr target,
            expr iter,
            stmt* body,
            stmt* orelse,
            string? type_comment
        )
        ```
        """
        raise NotImplementedError("Async for statement not implemented")

    def visit_While(self, node: ast.While) -> None:
        """
        while文を処理する

        ```asdl
        While(
            expr test,
            stmt* body,
            stmt* orelse
        )
        ```
        """
        raise NotImplementedError("While statement not implemented")

    def visit_If(self, node: ast.If) -> None:
        """
        if文:
          if test:
              ...
          else:
              ...

        ```asdl
        If(expr test, stmt* body, stmt* orelse)
        ```
        """
        cond_value = self.require_value(node.test, self.visit(node.test))
        i1 = ir.IntegerType.get_signless(1, context=self.ctx)

        # In native mode, the condition should already be i1 (from arith.cmpi)
        # In object mode, we need to cast from !py.bool to i1
        if self._in_native_func:
            cond = cond_value  # Already i1 from primitive comparison
        else:
            with self._loc(node), self.insertion_point():
                cond = py_ops.CastToPrimOp(
                    i1, cond_value, ir.StringAttr.get("exact", self.ctx)
                ).result

        assert self.current_block is not None
        parent_region = self.current_block.region
        true_block = parent_region.blocks.append()
        false_block = parent_region.blocks.append()
        merge_block = parent_region.blocks.append()
        with self._loc(node), self.insertion_point():
            cf_ops.CondBranchOp(cond, [], [], true_block, false_block)

        def handle_branch(block: ir.Block, statements: list[ast.stmt]) -> None:
            self._set_insertion_block(block)
            self.push_scope()
            for stmt in statements:
                self.visit(stmt)
            if not self._block_terminated(block):
                with self._loc(node), ir.InsertionPoint(block):
                    cf_ops.BranchOp([], merge_block)
            self.pop_scope()

        handle_branch(true_block, node.body)
        handle_branch(false_block, node.orelse or [])

        self._set_insertion_block(merge_block)

    def visit_With(self, node: ast.With) -> None:
        """
        with文を処理する

        ```asdl
        With(withitem* items, stmt* body, string? type_comment)
        ```
        """
        raise NotImplementedError("With statement not implemented")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """
        非同期with文を処理する

        ```asdl
        AsyncWith(withitem* items, stmt* body, string? type_comment)
        ```
        """
        raise NotImplementedError("Async with statement not implemented")

    def visit_Match(self, node: ast.Match) -> None:
        """
        match文を処理する

        ```asdl
        Match(expr subject, match_case* cases)
        ```
        """
        raise NotImplementedError("Match statement not implemented")

    def visit_Raise(self, node: ast.Raise) -> None:
        """
        raise文を処理する

        ```asdl
        Raise(expr? exc, expr? cause)
        ```
        """
        raise NotImplementedError("Raise statement not implemented")

    def visit_Try(self, node: ast.Try) -> None:
        """
        Try文を処理する

        ```asdl
        Try(
            stmt* body,
            excepthandler* handlers,
            stmt* orelse,
            stmt* finalbody
        )
        ```
        """
        raise NotImplementedError("Try statement not implemented")

    def visit_TryStar(self, node: ast.TryStar) -> None:
        """
        except*節が続くtryブロックを処理する

        ```asdl
        TryStar(
            stmt* body,
            excepthandler* handlers,
            stmt* orelse,
            stmt* finalbody
        )
        ```
        """
        raise NotImplementedError("Try star statement not implemented")

    def visit_Assert(self, node: ast.Assert) -> None:
        """
        assert文を処理する

        ```asdl
        Assert(expr test, expr? msg)
        ```
        """
        raise NotImplementedError("Assert statement not implemented")

    def visit_Import(self, node: ast.Import) -> None:
        """
        import文を処理する

        ```asdl
        Import(alias* names)
        ```
        """
        raise NotImplementedError("Import statement not implemented")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        from ... import文を処理する

        ```asdl
        ImportFrom(identifier? module, alias* names, int? level)
        ```

        Handles:
        - from lyrt import native, to_prim, from_prim
        - from lyrt.prim import i8, i16, ...
        """
        from ._base import PRIMITIVE_TYPE_MAP

        module = node.module

        if module == "lyrt":
            # Handle lyrt builtins: native, to_prim, from_prim
            for alias in node.names:
                name = alias.name
                if name in ("native", "to_prim", "from_prim"):
                    self._lyrt_builtins.add(name)
                else:
                    raise NotImplementedError(f"Unknown lyrt import: {name}")
            return

        if module == "lyrt.prim":
            # Handle primitive type imports: i8, i16, i32, ...
            for alias in node.names:
                name = alias.name
                if name in PRIMITIVE_TYPE_MAP:
                    # Store the imported name (may be aliased)
                    local_name = alias.asname or name
                    self._prim_types[local_name] = name
                else:
                    raise NotImplementedError(f"Unknown lyrt.prim type: {name}")
            return

        raise NotImplementedError(f"Import from '{module}' not implemented")

    def visit_Global(self, node: ast.Global) -> None:
        """
        global文を処理する

        ```asdl
        Global(identifier* names)
        ```
        """
        raise NotImplementedError("Global statement not implemented")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """
        nonlocal文を処理する

        ```asdl
        Nonlocal(identifier* names)
        ```
        """
        raise NotImplementedError("Nonlocal statement not implemented")

    def visit_Expr(self, node: ast.Expr) -> Any:
        """
        式文
        戻り値は破棄してよいので、一応計算はするが特に変数には入れない

        ```asdl
        Expr(expr value)
        ```
        """
        expr_visitor = self.subvisitors.get("Expr")
        if expr_visitor is None:
            raise NotImplementedError("Expression visitor not available")
        expr_visitor.current_block = self.current_block
        expr_visitor.visit(node.value)
        return None

    def visit_Pass(self, node: ast.Pass) -> None:
        """
        pass文を処理する

        ```asdl
        Pass
        ```
        """
        raise NotImplementedError("Pass statement not implemented")

    def visit_Break(self, node: ast.Break) -> None:
        """
        break文を処理する

        ```asdl
        Break
        ```
        """
        raise NotImplementedError("Break statement not implemented")

    def visit_Continue(self, node: ast.Continue) -> None:
        """
        continue文を処理する

        ```asdl
        Continue
        ```
        """
        raise NotImplementedError("Continue statement not implemented")
