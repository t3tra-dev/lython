from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .inference import FuncType, InferenceError, TypeCon, TypeTerm


def _empty_fields() -> dict[str, TypeTerm]:
    return {}


def _empty_methods() -> dict[str, "FunctionSig"]:
    return {}


@dataclass(slots=True)
class FunctionSig:
    name: str
    args: tuple[TypeTerm, ...]
    ret: TypeTerm
    arg_names: tuple[str, ...]
    kwonly: tuple[TypeTerm, ...] = ()
    kwonly_names: tuple[str, ...] = ()
    defaults_count: int = 0
    kwdefault_names: tuple[str, ...] = ()
    is_async: bool = False

    def callable_type(self) -> FuncType:
        result = TypeCon("coro", (self.ret,)) if self.is_async else self.ret
        return FuncType(self.args, result, self.kwonly)


@dataclass(slots=True)
class ClassSig:
    name: str
    base_names: tuple[str, ...] = ()
    fields: dict[str, TypeTerm] = field(default_factory=_empty_fields)
    methods: dict[str, FunctionSig] = field(default_factory=_empty_methods)

    def term(self) -> TypeCon:
        return TypeCon(f"class:{self.name}")


def compute_c3_mro(classes: Mapping[str, ClassSig], class_name: str) -> tuple[str, ...]:
    visiting: set[str] = set()

    def mro(name: str) -> list[str]:
        if name in visiting:
            raise InferenceError(f"class inheritance cycle through '{name}'")
        class_sig = classes.get(name)
        if class_sig is None:
            raise InferenceError(f"unknown base class '{name}'")
        visiting.add(name)
        base_mros = [mro(base) for base in class_sig.base_names]
        visiting.remove(name)
        return [name, *_c3_merge([*base_mros, list(class_sig.base_names)], name)]

    return tuple(mro(class_name))


def _c3_merge(sequences: Sequence[list[str]], class_name: str) -> list[str]:
    pending = [list(seq) for seq in sequences if seq]
    result: list[str] = []
    while pending:
        candidate = None
        for seq in pending:
            head = seq[0]
            if not any(head in other[1:] for other in pending):
                candidate = head
                break
        if candidate is None:
            raise InferenceError(f"inconsistent C3 MRO for class '{class_name}'")
        result.append(candidate)
        next_pending: list[list[str]] = []
        for seq in pending:
            if seq and seq[0] == candidate:
                seq = seq[1:]
            if seq:
                next_pending.append(seq)
        pending = next_pending
    return result


@dataclass(slots=True)
class TypedProgram:
    node_types: dict[int, TypeTerm]
    functions: dict[str, FunctionSig]
    classes: dict[str, ClassSig]

    def type_of(self, node: ast.AST) -> TypeTerm:
        try:
            return self.node_types[id(node)]
        except KeyError as exc:
            raise InferenceError(
                f"No finalized type for {type(node).__name__}"
            ) from exc


_TYPED_PROGRAM_REGISTRY: dict[int, TypedProgram] = {}


def bind_typed_program(module: ast.Module, program: TypedProgram) -> None:
    _TYPED_PROGRAM_REGISTRY[id(module)] = program


def typed_program_for_module(module: ast.Module) -> TypedProgram | None:
    return _TYPED_PROGRAM_REGISTRY.get(id(module))


class SourceTypeOracle:
    def __init__(
        self,
        functions: dict[str, FunctionSig],
        classes: dict[str, ClassSig],
    ) -> None:
        self.functions = functions
        self.classes = classes

    def attribute(self, owner: TypeTerm, name: str) -> TypeTerm | None:
        if isinstance(owner, TypeCon):
            class_name = self._class_name(owner)
            if class_name is not None:
                for class_sig in self._class_mro(class_name):
                    field = class_sig.fields.get(name)
                    if field is not None:
                        return field
                    method = class_sig.methods.get(name)
                    if method is not None:
                        return FuncType(method.args[1:], method.ret, method.kwonly)
            if owner.name == "list" and len(owner.args) == 1:
                element = owner.args[0]
                if name in {"append", "remove"}:
                    return FuncType((element,), TypeCon("None"))
            if owner.name == "str" and name == "__repr__":
                return FuncType((), TypeCon("str"))
        return None

    def call(self, callee: TypeTerm, args: tuple[TypeTerm, ...]) -> TypeTerm | None:
        if isinstance(callee, FuncType):
            if len(args) != len(callee.args):
                return None
            return callee.ret
        if isinstance(callee, TypeCon):
            class_name = self._class_name(callee)
            if class_name is not None:
                return callee
        return None

    def subscript(self, container: TypeTerm, index: TypeTerm) -> TypeTerm | None:
        if not isinstance(container, TypeCon):
            return None
        if container.name == "list" and len(container.args) == 1:
            return container.args[0]
        if container.name == "dict" and len(container.args) == 2:
            return container.args[1]
        if container.name == "tuple" and len(container.args) == 1:
            return container.args[0]
        return None

    def awaitable(self, awaitable: TypeTerm) -> TypeTerm | None:
        if isinstance(awaitable, TypeCon) and awaitable.name in {
            "coro",
            "task",
            "future",
            "async.value",
        }:
            if len(awaitable.args) == 1:
                return awaitable.args[0]
        return None

    def _class_name(self, term: TypeCon) -> str | None:
        if not term.name.startswith("class:"):
            return None
        return term.name[len("class:") :]

    def _class_mro(
        self, class_name: str, seen: set[str] | None = None
    ) -> Iterable[ClassSig]:
        if seen is None:
            seen = set()
        if class_name in seen:
            return
        seen.add(class_name)
        class_sig = self.classes.get(class_name)
        if class_sig is None:
            return
        yield class_sig
        for base_name in class_sig.base_names:
            yield from self._class_mro(base_name, seen)
