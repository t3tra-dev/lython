from __future__ import annotations

import ast
from collections.abc import Iterable

from .annotations import AnnotationResolver
from .inference import AlgorithmM, InferenceError, TypeCon, TypeTerm
from .locations import source_position
from .program import (
    ClassSig,
    FunctionSig,
    SourceTypeOracle,
    TypedProgram,
    bind_typed_program,
    compute_c3_mro,
)


class TypeAnalyzer(ast.NodeVisitor):
    def __init__(self, *, filename: str = "<module>") -> None:
        self.filename = filename
        self.functions: dict[str, FunctionSig] = {}
        self.classes: dict[str, ClassSig] = {}
        self.imported_static_modules: dict[str, str] = {}
        self.imported_primitives: set[str] = set()
        self.lyrt_builtins: set[str] = set()
        self.annotations = AnnotationResolver(
            classes=self.classes,
            static_modules=self.imported_static_modules,
            imported_primitives=self.imported_primitives,
        )
        self.node_types: dict[int, TypeTerm] = {}
        self.m = AlgorithmM(SourceTypeOracle(self.functions, self.classes))
        self.scopes: list[dict[str, TypeTerm]] = [{}]
        self.current_class: ClassSig | None = None
        self.current_function: FunctionSig | None = None
        self.current_return: TypeTerm | None = None
        self.in_async_function = False
        self._install_builtin_classes()

    def analyze(self, module: ast.Module) -> TypedProgram:
        self._scan_declarations(module.body)
        for stmt in module.body:
            if isinstance(stmt, ast.ClassDef):
                self._collect_class_fields(stmt)
        self._inherit_class_members()
        self.visit(module)
        self.m.solve()
        finalized = {
            node_id: self._finalize(term, f"node {node_id}")
            for node_id, term in self.node_types.items()
        }
        for class_sig in self.classes.values():
            class_sig.fields = {
                name: self._finalize(term, f"{class_sig.name}.{name}")
                for name, term in class_sig.fields.items()
            }
        return TypedProgram(finalized, self.functions, self.classes)

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "asyncio":
                self.imported_static_modules[alias.asname or alias.name] = "asyncio"

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "asyncio":
            for alias in node.names:
                self.imported_static_modules[alias.asname or alias.name] = "asyncio"
            return
        if node.module == "lyrt":
            for alias in node.names:
                self.lyrt_builtins.add(alias.asname or alias.name)
            return
        if node.module == "lyrt.prim":
            for alias in node.names:
                self.imported_primitives.add(alias.asname or alias.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_sig = self.classes[node.name]
        prev_class = self.current_class
        self.current_class = class_sig
        self._push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self._pop_scope()
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_body(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_body(node, is_async=True)

    def visit_Return(self, node: ast.Return) -> None:
        if self.current_return is None:
            raise InferenceError(f"return outside function: {self._loc(node)}")
        value_type = TypeCon("None")
        if node.value is not None:
            value_type = self.expr(node.value)
        self.m.require_equal(
            value_type,
            self.current_return,
            "return type must match function annotation",
            self._loc(node),
        )

    def visit_Expr(self, node: ast.Expr) -> None:
        self.expr(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise InferenceError(
                f"multiple assignment is unsupported: {self._loc(node)}"
            )
        value_type = self.expr(node.value)
        self._assign(node.targets[0], value_type)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target_type = self.expr(node.target)
        value_type = self.expr(node.value)
        self.m.require_equal(
            target_type,
            value_type,
            "augmented assignment operands must match",
            self._loc(node),
        )
        self._record(node, target_type)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        annotation_type = self.annotation(node.annotation)
        if node.value is not None:
            value_type = self.expr(node.value)
            self.m.require_equal(
                value_type,
                annotation_type,
                "annotated assignment value must match annotation",
                self._loc(node),
            )
        self._assign(node.target, annotation_type)

    def visit_If(self, node: ast.If) -> None:
        self.expr(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        self.expr(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Pass(self, node: ast.Pass) -> None:
        return None

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            self.expr(node.exc)
        if node.cause is not None:
            self.expr(node.cause)

    def visit_Try(self, node: ast.Try) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            if handler.type is not None:
                self.expr(handler.type)
            if handler.name is not None:
                self._bind(handler.name, TypeCon("Exception"))
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def generic_visit(self, node: ast.AST) -> None:
        raise InferenceError(
            f"Type analysis does not support {type(node).__name__}: {self._loc(node)}"
        )

    def expr(self, node: ast.expr) -> TypeTerm:
        if id(node) in self.node_types:
            return self.node_types[id(node)]
        if isinstance(node, ast.Constant):
            if node.value is None:
                return self._record(node, TypeCon("None"))
            if isinstance(node.value, bool):
                return self._record(node, TypeCon("bool"))
            if isinstance(node.value, int):
                return self._record(node, TypeCon("int"))
            if isinstance(node.value, float):
                return self._record(node, TypeCon("float"))
            if isinstance(node.value, str):
                return self._record(node, TypeCon("str"))
        if isinstance(node, ast.Name):
            return self._record(node, self._lookup(node.id))
        if isinstance(node, ast.List):
            return self._list_expr(node)
        if isinstance(node, ast.Tuple):
            values = tuple(self.expr(elt) for elt in node.elts)
            return self._record(node, TypeCon("tuple", values))
        if isinstance(node, ast.Dict):
            return self._dict_expr(node)
        if isinstance(node, ast.Attribute):
            owner = self.expr(node.value)
            result = self.m.fresh_var(f"{node.attr}_attr")
            self.m.require_attribute(
                owner,
                node.attr,
                result,
                "attribute access must resolve statically",
                self._loc(node),
            )
            return self._record(node, result)
        if isinstance(node, ast.Subscript):
            if isinstance(node.ctx, ast.Load) and self._is_type_constructor(node):
                return self._record(node, self.annotation(node))
            container = self.expr(node.value)
            index = self.expr(node.slice)
            result = self.m.fresh_var("subscript")
            self.m.require_subscript(
                container,
                index,
                result,
                "subscript must resolve statically",
                self._loc(node),
            )
            return self._record(node, result)
        if isinstance(node, ast.Call):
            return self._call_expr(node)
        if isinstance(node, ast.BinOp):
            lhs = self.expr(node.left)
            rhs = self.expr(node.right)
            if isinstance(node.op, ast.MatMult):
                return self._record(node, self._matmul_type(lhs, rhs, node))
            self.m.require_equal(
                lhs,
                rhs,
                "binary operands must have the same static type",
                self._loc(node),
            )
            return self._record(node, lhs)
        if isinstance(node, ast.UnaryOp):
            return self._record(node, self.expr(node.operand))
        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            for comparator in node.comparators:
                right = self.expr(comparator)
                self.m.require_equal(
                    left,
                    right,
                    "comparison operands must have the same static type",
                    self._loc(node),
                )
            return self._record(node, TypeCon("bool"))
        if isinstance(node, ast.Await):
            if not self.in_async_function:
                raise InferenceError(f"await outside async function: {self._loc(node)}")
            awaitable = self.expr(node.value)
            result = self.m.fresh_var("await")
            self.m.require_await(
                awaitable,
                result,
                "await operand must be awaitable",
                self._loc(node),
            )
            return self._record(node, result)
        raise InferenceError(
            f"Type analysis does not support expression {type(node).__name__}: "
            f"{self._loc(node)}"
        )

    def annotation(self, node: ast.expr | None) -> TypeTerm:
        return self.annotations.resolve(node)

    def _scan_declarations(self, body: Iterable[ast.stmt]) -> None:
        self._install_builtin_classes()
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                base_names = self._class_base_names(stmt)
                self.classes[stmt.name] = ClassSig(stmt.name, base_names=base_names)
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[stmt.name] = self._function_sig(
                    stmt, is_async=isinstance(stmt, ast.AsyncFunctionDef)
                )
            if isinstance(stmt, ast.ClassDef):
                class_sig = self.classes[stmt.name]
                for child in stmt.body:
                    if isinstance(child, ast.FunctionDef):
                        class_sig.methods[child.name] = self._function_sig(
                            child,
                            is_async=False,
                            owning_class=class_sig,
                        )

    def _collect_class_fields(self, node: ast.ClassDef) -> None:
        class_sig = self.classes[node.name]
        init = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"
            ),
            None,
        )
        if init is None:
            return
        self._push_scope()
        self._bind("self", class_sig.term())
        for arg in init.args.args[1:]:
            self._bind(arg.arg, self.annotation(arg.annotation))
        for stmt in init.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                class_sig.fields[target.attr] = self.expr(stmt.value)
        self._pop_scope()

    def _visit_function_body(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool
    ) -> None:
        if self.current_class is not None and isinstance(node, ast.FunctionDef):
            sig = self.current_class.methods[node.name]
        else:
            sig = self.functions[node.name]
        prev_function = self.current_function
        prev_return = self.current_return
        prev_async = self.in_async_function
        self.current_function = sig
        self.current_return = sig.ret
        self.in_async_function = is_async
        self._push_scope()
        for name, term in zip(sig.arg_names, sig.args):
            self._bind(name, term)
        for name, term in zip(sig.kwonly_names, sig.kwonly):
            self._bind(name, term)
        for stmt in node.body:
            self.visit(stmt)
        self._pop_scope()
        self.current_function = prev_function
        self.current_return = prev_return
        self.in_async_function = prev_async

    def _function_sig(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_async: bool,
        owning_class: ClassSig | None = None,
    ) -> FunctionSig:
        args: list[TypeTerm] = []
        arg_names: list[str] = []
        for index, arg in enumerate(node.args.args):
            if owning_class is not None and index == 0 and arg.arg == "self":
                args.append(owning_class.term())
            else:
                args.append(self.annotation(arg.annotation))
            arg_names.append(arg.arg)
        kwonly = [self.annotation(arg.annotation) for arg in node.args.kwonlyargs]
        if node.name == "__init__" and node.returns is None:
            ret = TypeCon("None")
        else:
            ret = self.annotation(node.returns)
        return FunctionSig(
            node.name,
            tuple(args),
            ret,
            tuple(arg_names),
            tuple(kwonly),
            tuple(arg.arg for arg in node.args.kwonlyargs),
            defaults_count=len(node.args.defaults),
            kwdefault_names=tuple(
                arg.arg
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if default is not None
            ),
            is_async=is_async,
        )

    def _call_expr(self, node: ast.Call) -> TypeTerm:
        if isinstance(node.func, ast.Subscript) and self._is_type_constructor(
            node.func
        ):
            result = self.annotation(node.func)
            for arg in node.args:
                self.expr(arg)
            return self._record(node, result)
        if isinstance(node.func, ast.Name):
            special = self._special_name_call(node)
            if special is not None:
                return self._record(node, special)
        if isinstance(node.func, ast.Attribute):
            special_attr = self._special_attribute_call(node)
            if special_attr is not None:
                return self._record(node, special_attr)

        callee = self.expr(node.func)
        if node.keywords:
            raise InferenceError(
                f"keyword calls require a statically known callable: {self._loc(node)}"
            )
        args = tuple(self.expr(arg) for arg in node.args)
        result = self.m.fresh_var("call")
        self.m.require_call(
            callee,
            args,
            result,
            "call target must resolve statically",
            self._loc(node),
        )
        return self._record(node, result)

    def _special_name_call(self, node: ast.Call) -> TypeTerm | None:
        assert isinstance(node.func, ast.Name)
        name = node.func.id
        if name == "print":
            for arg in node.args:
                self.expr(arg)
            return TypeCon("None")
        if name == "from_prim":
            if len(node.args) != 1:
                raise InferenceError(
                    f"from_prim expects one argument: {self._loc(node)}"
                )
            arg = self.expr(node.args[0])
            if isinstance(arg, TypeCon):
                if arg.name.startswith("Int"):
                    return TypeCon("int")
                if arg.name.startswith("Float"):
                    return TypeCon("float")
                if arg.name in {"Matrix", "Tensor"}:
                    return TypeCon("str")
            return self.m.fresh_var("from_prim")
        if name in self.classes:
            class_sig = self.classes[name]
            init = class_sig.methods.get("__init__")
            if init is not None:
                self._check_signature_call(init, node, "__init__", skip_args=1)
            return class_sig.term()
        if name in self.functions:
            sig = self.functions[name]
            self._check_signature_call(sig, node, name)
            return TypeCon("coro", (sig.ret,)) if sig.is_async else sig.ret
        return None

    def _special_attribute_call(self, node: ast.Call) -> TypeTerm | None:
        assert isinstance(node.func, ast.Attribute)
        func = node.func
        if (
            isinstance(func.value, ast.Name)
            and self.imported_static_modules.get(func.value.id) == "asyncio"
        ):
            if func.attr == "run":
                if len(node.args) != 1:
                    raise InferenceError(
                        f"asyncio.run expects one argument: {self._loc(node)}"
                    )
                awaitable = self.expr(node.args[0])
                result = self.m.fresh_var("asyncio_run")
                self.m.require_await(
                    awaitable,
                    result,
                    "asyncio.run operand must be awaitable",
                    self._loc(node),
                )
                return result
            if func.attr in {"create_task", "ensure_future"}:
                if len(node.args) != 1:
                    raise InferenceError(
                        f"asyncio.{func.attr} expects one argument: {self._loc(node)}"
                    )
                awaitable = self.expr(node.args[0])
                payload = self.m.fresh_var("task_payload")
                self.m.require_equal(
                    awaitable,
                    TypeCon("coro", (payload,)),
                    f"asyncio.{func.attr} operand must be a coroutine",
                    self._loc(node),
                )
                return TypeCon("task", (payload,))
            if func.attr == "gather":
                payloads: list[TypeTerm] = []
                for arg in node.args:
                    awaitable = self.expr(arg)
                    payload = self.m.fresh_var("gather_payload")
                    self.m.require_await(
                        awaitable,
                        payload,
                        "asyncio.gather operands must be awaitable",
                        self._loc(arg),
                    )
                    payloads.append(payload)
                return TypeCon("future", (TypeCon("tuple", tuple(payloads)),))
            if func.attr == "sleep":
                for arg in node.args:
                    self.expr(arg)
                return TypeCon("future", (TypeCon("None"),))

        owner = self.expr(func.value)
        if isinstance(owner, TypeCon) and owner.name == "task":
            if func.attr != "cancel":
                raise InferenceError(
                    f"Unsupported task method '{func.attr}': {self._loc(func)}"
                )
            if node.args or node.keywords:
                raise InferenceError(
                    f"task.cancel() expects no arguments: {self._loc(node)}"
                )
            return TypeCon("bool")

        method = self.m.fresh_var(func.attr)
        self.m.require_attribute(
            owner,
            func.attr,
            method,
            "method lookup must resolve statically",
            self._loc(func),
        )
        args = tuple(self.expr(arg) for arg in node.args)
        if node.keywords:
            raise InferenceError(
                f"keyword method calls require a statically known method: {self._loc(node)}"
            )
        result = self.m.fresh_var("method_call")
        self.m.require_call(
            method,
            args,
            result,
            "method call must resolve statically",
            self._loc(node),
        )
        return result

    def _list_expr(self, node: ast.List) -> TypeTerm:
        if not node.elts:
            raise InferenceError(f"empty list literal needs a type: {self._loc(node)}")
        element = self.expr(node.elts[0])
        for elt in node.elts[1:]:
            self.m.require_equal(
                self.expr(elt),
                element,
                "list elements must have the same static type",
                self._loc(elt),
            )
        return self._record(node, TypeCon("list", (element,)))

    def _dict_expr(self, node: ast.Dict) -> TypeTerm:
        if not node.keys:
            raise InferenceError(f"empty dict literal needs a type: {self._loc(node)}")
        first_key = node.keys[0]
        if first_key is None:
            raise InferenceError(f"dict unpacking is unsupported: {self._loc(node)}")
        key_type = self.expr(first_key)
        value_type = self.expr(node.values[0])
        for key, value in zip(node.keys[1:], node.values[1:]):
            if key is None:
                raise InferenceError(
                    f"dict unpacking is unsupported: {self._loc(node)}"
                )
            self.m.require_equal(
                self.expr(key),
                key_type,
                "dict keys must have the same static type",
                self._loc(key),
            )
            self.m.require_equal(
                self.expr(value),
                value_type,
                "dict values must have the same static type",
                self._loc(value),
            )
        return self._record(node, TypeCon("dict", (key_type, value_type)))

    def _matmul_type(self, lhs: TypeTerm, rhs: TypeTerm, node: ast.AST) -> TypeTerm:
        if (
            isinstance(lhs, TypeCon)
            and isinstance(rhs, TypeCon)
            and lhs.name == "Matrix"
            and rhs.name == "Matrix"
            and len(lhs.args) == 3
            and len(rhs.args) == 3
        ):
            self.m.require_equal(
                lhs.args[0],
                rhs.args[0],
                "matrix multiplication element types must match",
                self._loc(node),
            )
            self.m.require_equal(
                lhs.args[2],
                rhs.args[1],
                "matrix multiplication inner dimensions must match",
                self._loc(node),
            )
            return TypeCon("Matrix", (lhs.args[0], lhs.args[1], rhs.args[2]))
        raise InferenceError(
            f"matrix multiplication requires Matrix[T, M, N] operands: {self._loc(node)}"
        )

    def _is_type_constructor(self, node: ast.Subscript) -> bool:
        return self.annotations.is_type_constructor(node)

    def _assign(self, target: ast.expr, value_type: TypeTerm) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id, value_type)
            self._record(target, value_type)
            return
        if isinstance(target, ast.Attribute):
            owner = self.expr(target.value)
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and self.current_class is not None
            ):
                field = self.current_class.fields.setdefault(target.attr, value_type)
                self.m.require_equal(
                    value_type,
                    field,
                    "field assignment must match static class layout",
                    self._loc(target),
                )
                self._record(target, field)
                return
            expected = self.m.fresh_var(target.attr)
            self.m.require_attribute(
                owner,
                target.attr,
                expected,
                "attribute assignment must resolve statically",
                self._loc(target),
            )
            self.m.require_equal(
                value_type,
                expected,
                "attribute assignment must match field type",
                self._loc(target),
            )
            self._record(target, expected)
            return
        if isinstance(target, ast.Subscript):
            container = self.expr(target.value)
            key = self.expr(target.slice)
            expected = self.m.fresh_var("subscript_set")
            self.m.require_subscript(
                container,
                key,
                expected,
                "subscript assignment must resolve statically",
                self._loc(target),
            )
            self.m.require_equal(
                value_type,
                expected,
                "subscript assignment must match element type",
                self._loc(target),
            )
            self._record(target, expected)
            return
        raise InferenceError(
            f"unsupported assignment target {type(target).__name__}: {self._loc(target)}"
        )

    def _check_signature_call(
        self,
        sig: FunctionSig,
        node: ast.Call,
        callee: str,
        *,
        skip_args: int = 0,
    ) -> None:
        expected = sig.args[skip_args:]
        arg_names = sig.arg_names[skip_args:]
        if len(node.args) > len(expected):
            raise InferenceError(
                f"{callee} expects at most {len(expected)} positional args, "
                f"got {len(node.args)}: {self._loc(node)}"
            )
        for expected_type, arg in zip(expected, node.args):
            self.m.require_equal(
                self.expr(arg),
                expected_type,
                f"{callee} argument type mismatch",
                self._loc(arg),
            )
        positional_by_name = {name: index for index, name in enumerate(arg_names)}
        kwonly_by_name = {name: index for index, name in enumerate(sig.kwonly_names)}
        seen_keywords: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg is None:
                raise InferenceError(
                    f"{callee} does not support keyword unpacking: {self._loc(keyword)}"
                )
            if keyword.arg in seen_keywords:
                raise InferenceError(
                    f"{callee} received duplicate keyword '{keyword.arg}': "
                    f"{self._loc(keyword)}"
                )
            seen_keywords.add(keyword.arg)
            if keyword.arg in positional_by_name:
                index = positional_by_name[keyword.arg]
                if index < len(node.args):
                    raise InferenceError(
                        f"{callee} argument '{keyword.arg}' was provided both "
                        f"positionally and by keyword: {self._loc(keyword)}"
                    )
                expected_type = expected[index]
            elif keyword.arg in kwonly_by_name:
                expected_type = sig.kwonly[kwonly_by_name[keyword.arg]]
            else:
                raise InferenceError(
                    f"{callee} got unexpected keyword '{keyword.arg}': "
                    f"{self._loc(keyword)}"
                )
            self.m.require_equal(
                self.expr(keyword.value),
                expected_type,
                f"{callee} keyword argument '{keyword.arg}' type mismatch",
                self._loc(keyword.value),
            )

        min_positional = max(0, len(expected) - sig.defaults_count)
        for index, name in enumerate(arg_names):
            if index < len(node.args) or name in seen_keywords:
                continue
            if index < min_positional:
                raise InferenceError(
                    f"{callee} missing required argument '{name}': {self._loc(node)}"
                )

        kwdefault_names = set(sig.kwdefault_names)
        for name in sig.kwonly_names:
            if name in seen_keywords or name in kwdefault_names:
                continue
            raise InferenceError(
                f"{callee} missing required keyword-only argument '{name}': "
                f"{self._loc(node)}"
            )

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        if len(self.scopes) == 1:
            raise RuntimeError("cannot pop root type scope")
        self.scopes.pop()

    def _bind(self, name: str, term: TypeTerm) -> None:
        self.scopes[-1][name] = term

    def _lookup(self, name: str) -> TypeTerm:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        if name in self.functions:
            return self.functions[name].callable_type()
        if name in self.classes:
            return self.classes[name].term()
        raise InferenceError(f"unknown symbol '{name}'")

    def _install_builtin_classes(self) -> None:
        self.classes.setdefault("BaseException", ClassSig("BaseException"))
        self.classes.setdefault(
            "Exception", ClassSig("Exception", base_names=("BaseException",))
        )

    @staticmethod
    def _builtin_class_bases(name: str) -> tuple[str, ...]:
        if name == "Exception":
            return ("BaseException",)
        return ()

    def _class_base_names(self, node: ast.ClassDef) -> tuple[str, ...]:
        if not node.bases:
            return self._builtin_class_bases(node.name)
        base_names: list[str] = []
        for base in node.bases:
            if not isinstance(base, ast.Name):
                raise InferenceError(
                    f"class bases must be statically named classes: {self._loc(base)}"
                )
            if base.id not in self.classes:
                raise InferenceError(
                    f"unknown base class '{base.id}': {self._loc(base)}"
                )
            base_names.append(base.id)
        return tuple(base_names)

    def _inherit_class_members(self) -> None:
        for class_name, class_sig in list(self.classes.items()):
            mro = compute_c3_mro(self.classes, class_name)
            inherited_fields: dict[str, TypeTerm] = {}
            inherited_methods: dict[str, FunctionSig] = {}
            for ancestor_name in reversed(mro[1:]):
                ancestor = self.classes[ancestor_name]
                self._merge_inherited_fields(
                    inherited_fields, ancestor.fields, class_name, ancestor_name
                )
                inherited_methods.update(ancestor.methods)

            own_fields = dict(class_sig.fields)
            self._merge_inherited_fields(
                inherited_fields, own_fields, class_name, class_name
            )
            inherited_methods.update(class_sig.methods)
            class_sig.fields = inherited_fields
            class_sig.methods = inherited_methods

    def _merge_inherited_fields(
        self,
        target: dict[str, TypeTerm],
        source: dict[str, TypeTerm],
        class_name: str,
        source_name: str,
    ) -> None:
        for field_name, field_type in source.items():
            existing = target.get(field_name)
            if existing is not None:
                self.m.require_equal(
                    existing,
                    field_type,
                    f"inherited field '{field_name}' in class '{class_name}' "
                    f"must keep the same static type when merged from '{source_name}'",
                )
            target[field_name] = field_type

    def _record(self, node: ast.AST, term: TypeTerm) -> TypeTerm:
        self.node_types[id(node)] = term
        return term

    def _finalize(self, term: TypeTerm, what: str) -> TypeTerm:
        result = self.m.apply(term)
        if self._free_vars(result):
            raise InferenceError(f"unresolved type for {what}: {result.display()}")
        return result

    def _free_vars(self, term: TypeTerm) -> set[int]:
        from .inference import type_free_vars

        return set(type_free_vars(term))

    def _loc(self, node: ast.AST) -> str:
        position = source_position(node)
        if position is None:
            return self.filename
        line, col = position
        return f"{self.filename}:{line}:{int(col) + 1}"


def analyze_module_types(
    module: ast.Module, *, filename: str = "<module>"
) -> TypedProgram:
    analyzer = TypeAnalyzer(filename=filename)
    result = analyzer.analyze(module)
    bind_typed_program(module, result)
    return result
