from __future__ import annotations

import ast
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from .inference import InferenceError, TypeCon, TypeTerm

_BUILTIN_ANNOTATIONS = {
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
    "None": "None",
}


class AnnotationResolver:
    def __init__(
        self,
        *,
        classes: Mapping[str, object],
        static_modules: Mapping[str, str],
        imported_primitives: AbstractSet[str],
    ) -> None:
        self.classes = classes
        self.static_modules = static_modules
        self.imported_primitives = imported_primitives

    def resolve(self, node: ast.expr | None) -> TypeTerm:
        if node is None:
            raise InferenceError("missing type annotation")
        if isinstance(node, ast.Constant) and node.value is None:
            return TypeCon("None")
        if isinstance(node, ast.Name):
            if node.id == "object":
                raise InferenceError(
                    "generic object annotations are not supported; use a "
                    "concrete Python-level type"
                )
            builtin = _BUILTIN_ANNOTATIONS.get(node.id)
            if builtin is not None:
                return TypeCon(builtin)
            if node.id in self.classes:
                return TypeCon(f"class:{node.id}")
            if node.id in self.imported_primitives:
                return TypeCon(node.id)
        if isinstance(node, ast.Subscript):
            return self._subscript(node)
        raise InferenceError(f"unsupported annotation {ast.dump(node)}")

    def is_type_constructor(self, node: ast.Subscript) -> bool:
        base_name = self._base_name(node.value)
        return base_name in self.imported_primitives or base_name in {
            "list",
            "dict",
            "tuple",
            "Callable",
            "Coroutine",
            "Coro",
            "Task",
            "Future",
        }

    def _subscript(self, node: ast.Subscript) -> TypeTerm:
        base_name = self._base_name(node.value)
        if base_name in {"list", "List"}:
            return TypeCon("list", (self.resolve(node.slice),))
        if base_name in {"dict", "Dict"}:
            args = self._tuple_args(node.slice, expected=2)
            return TypeCon("dict", (self.resolve(args[0]), self.resolve(args[1])))
        if base_name in {"tuple", "Tuple"}:
            if isinstance(node.slice, ast.Tuple):
                return TypeCon(
                    "tuple", tuple(self.resolve(arg) for arg in node.slice.elts)
                )
            return TypeCon("tuple", (self.resolve(node.slice),))
        if base_name in {"Coroutine", "Coro"}:
            return TypeCon("coro", (self.resolve(node.slice),))
        if base_name == "Task":
            return TypeCon("task", (self.resolve(node.slice),))
        if base_name == "Future":
            return TypeCon("future", (self.resolve(node.slice),))
        if base_name in {"Int", "Float"}:
            bits = self._constant_int(node.slice)
            return TypeCon(f"{base_name}[{bits}]")
        if base_name in {"Matrix", "Tensor"}:
            args = self._tuple_args(node.slice, min_expected=1)
            elem = self.resolve(args[0])
            dims = tuple(TypeCon(str(self._constant_int(arg))) for arg in args[1:])
            return TypeCon(base_name, (elem, *dims))
        raise InferenceError(f"unsupported annotation {ast.dump(node)}")

    def _base_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and self.static_modules.get(node.value.id) == "asyncio"
        ):
            return node.attr
        return None

    def _tuple_args(
        self,
        node: ast.expr,
        *,
        expected: int | None = None,
        min_expected: int | None = None,
    ) -> tuple[ast.expr, ...]:
        args = tuple(node.elts) if isinstance(node, ast.Tuple) else (node,)
        if expected is not None and len(args) != expected:
            raise InferenceError(f"expected {expected} type arguments")
        if min_expected is not None and len(args) < min_expected:
            raise InferenceError(f"expected at least {min_expected} type arguments")
        return args

    def _constant_int(self, node: ast.expr) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        raise InferenceError(f"expected integer type parameter: {ast.dump(node)}")
