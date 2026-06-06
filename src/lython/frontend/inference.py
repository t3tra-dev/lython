from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class TypeVar:
    id: int
    name: str | None = None

    def display(self) -> str:
        if self.name is not None:
            return self.name
        return f"t{self.id}"


@dataclass(frozen=True, slots=True)
class TypeCon:
    name: str
    args: tuple["TypeTerm", ...] = ()

    def display(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}<{', '.join(arg.display() for arg in self.args)}>"


@dataclass(frozen=True, slots=True)
class FuncType:
    args: tuple["TypeTerm", ...]
    ret: "TypeTerm"
    kwonly: tuple["TypeTerm", ...] = ()

    def display(self) -> str:
        args = ", ".join(arg.display() for arg in self.args)
        ret = self.ret.display()
        if not self.kwonly:
            return f"func<[{args}] -> {ret}>"
        kwonly = ", ".join(arg.display() for arg in self.kwonly)
        return f"func<[{args}], kwonly = [{kwonly}] -> {ret}>"


TypeTerm: TypeAlias = TypeVar | TypeCon | FuncType


@dataclass(frozen=True, slots=True)
class TypeScheme:
    variables: frozenset[int]
    body: TypeTerm


@dataclass(frozen=True, slots=True)
class ConstraintReason:
    message: str
    location: object | None = None

    def describe(self) -> str:
        if self.location is None:
            return self.message
        return f"{self.message} at {self.location}"


@dataclass(frozen=True, slots=True)
class Equal:
    lhs: TypeTerm
    rhs: TypeTerm
    reason: ConstraintReason


@dataclass(frozen=True, slots=True)
class Attribute:
    owner: TypeTerm
    name: str
    result: TypeTerm
    reason: ConstraintReason


@dataclass(frozen=True, slots=True)
class Call:
    callee: TypeTerm
    args: tuple[TypeTerm, ...]
    result: TypeTerm
    reason: ConstraintReason


@dataclass(frozen=True, slots=True)
class Subscript:
    container: TypeTerm
    index: TypeTerm
    result: TypeTerm
    reason: ConstraintReason


@dataclass(frozen=True, slots=True)
class Await:
    awaitable: TypeTerm
    result: TypeTerm
    reason: ConstraintReason


Constraint: TypeAlias = Equal | Attribute | Call | Subscript | Await
Substitution: TypeAlias = dict[int, TypeTerm]


class InferenceError(TypeError):
    pass


class SemanticOracle(Protocol):
    """Lython-specific semantic facts used by the Algorithm M kernel."""

    def attribute(self, owner: TypeTerm, name: str) -> TypeTerm | None: ...

    def call(self, callee: TypeTerm, args: tuple[TypeTerm, ...]) -> TypeTerm | None: ...

    def subscript(self, container: TypeTerm, index: TypeTerm) -> TypeTerm | None: ...

    def awaitable(self, awaitable: TypeTerm) -> TypeTerm | None: ...


def _empty_attribute_map() -> dict[tuple[str, str], TypeTerm]:
    return {}


def _empty_call_map() -> dict[str, FuncType]:
    return {}


@dataclass(slots=True)
class DefaultOracle:
    """Constructor-based semantics for builtin Lython type constructors."""

    attributes: Mapping[tuple[str, str], TypeTerm] = field(
        default_factory=_empty_attribute_map
    )
    calls: Mapping[str, FuncType] = field(default_factory=_empty_call_map)

    def attribute(self, owner: TypeTerm, name: str) -> TypeTerm | None:
        if not isinstance(owner, TypeCon):
            return None
        return self.attributes.get((owner.name, name))

    def call(self, callee: TypeTerm, args: tuple[TypeTerm, ...]) -> TypeTerm | None:
        if isinstance(callee, FuncType):
            if len(args) != len(callee.args):
                return None
            return callee.ret
        if isinstance(callee, TypeCon):
            signature = self.calls.get(callee.name)
            if signature is not None and len(args) == len(signature.args):
                return signature.ret
        return None

    def subscript(self, container: TypeTerm, index: TypeTerm) -> TypeTerm | None:
        if not isinstance(container, TypeCon):
            return None
        if container.name == "list" and len(container.args) == 1:
            return container.args[0]
        if container.name == "dict" and len(container.args) == 2:
            return container.args[1]
        if container.name == "tuple" and container.args:
            return container.args[0] if len(container.args) == 1 else None
        return None

    def awaitable(self, awaitable: TypeTerm) -> TypeTerm | None:
        if not isinstance(awaitable, TypeCon):
            return None
        if awaitable.name in {"coro", "task", "future", "async.value"}:
            if len(awaitable.args) == 1:
                return awaitable.args[0]
        return None


@dataclass(slots=True)
class TypeEnvironment:
    scopes: list[dict[str, TypeScheme]] = field(default_factory=lambda: [{}])

    def push(self) -> None:
        self.scopes.append({})

    def pop(self) -> dict[str, TypeScheme]:
        if len(self.scopes) == 1:
            raise RuntimeError("Cannot pop root type environment scope")
        return self.scopes.pop()

    def bind(self, name: str, scheme: TypeScheme) -> None:
        self.scopes[-1][name] = scheme

    def lookup(self, name: str) -> TypeScheme:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise InferenceError(f"Unresolved type binding '{name}'")

    def free_vars(self) -> frozenset[int]:
        free: set[int] = set()
        for scope in self.scopes:
            for scheme in scope.values():
                free.update(type_free_vars(scheme.body) - scheme.variables)
        return frozenset(free)


def type_free_vars(term: TypeTerm) -> frozenset[int]:
    if isinstance(term, TypeVar):
        return frozenset({term.id})
    if isinstance(term, TypeCon):
        free: set[int] = set()
        for arg in term.args:
            free.update(type_free_vars(arg))
        return frozenset(free)
    free = set(type_free_vars(term.ret))
    for arg in term.args:
        free.update(type_free_vars(arg))
    for arg in term.kwonly:
        free.update(type_free_vars(arg))
    return frozenset(free)


def apply_substitution(term: TypeTerm, subst: Mapping[int, TypeTerm]) -> TypeTerm:
    if isinstance(term, TypeVar):
        replacement = subst.get(term.id)
        if replacement is None:
            return term
        return apply_substitution(replacement, subst)
    if isinstance(term, TypeCon):
        if not term.args:
            return term
        return TypeCon(
            term.name, tuple(apply_substitution(arg, subst) for arg in term.args)
        )
    return FuncType(
        tuple(apply_substitution(arg, subst) for arg in term.args),
        apply_substitution(term.ret, subst),
        tuple(apply_substitution(arg, subst) for arg in term.kwonly),
    )


def apply_constraint(
    constraint: Constraint, subst: Mapping[int, TypeTerm]
) -> Constraint:
    if isinstance(constraint, Equal):
        return Equal(
            apply_substitution(constraint.lhs, subst),
            apply_substitution(constraint.rhs, subst),
            constraint.reason,
        )
    if isinstance(constraint, Attribute):
        return Attribute(
            apply_substitution(constraint.owner, subst),
            constraint.name,
            apply_substitution(constraint.result, subst),
            constraint.reason,
        )
    if isinstance(constraint, Call):
        return Call(
            apply_substitution(constraint.callee, subst),
            tuple(apply_substitution(arg, subst) for arg in constraint.args),
            apply_substitution(constraint.result, subst),
            constraint.reason,
        )
    if isinstance(constraint, Subscript):
        return Subscript(
            apply_substitution(constraint.container, subst),
            apply_substitution(constraint.index, subst),
            apply_substitution(constraint.result, subst),
            constraint.reason,
        )
    return Await(
        apply_substitution(constraint.awaitable, subst),
        apply_substitution(constraint.result, subst),
        constraint.reason,
    )


def _empty_constraints() -> list[Constraint]:
    return []


def _empty_substitution() -> Substitution:
    return {}


@dataclass(slots=True)
class AlgorithmM:
    oracle: SemanticOracle = field(default_factory=DefaultOracle)
    env: TypeEnvironment = field(default_factory=TypeEnvironment)
    constraints: list[Constraint] = field(default_factory=_empty_constraints)
    substitution: Substitution = field(default_factory=_empty_substitution)
    _next_type_var: int = 0

    def fresh_var(self, name: str | None = None) -> TypeVar:
        var = TypeVar(self._next_type_var, name)
        self._next_type_var += 1
        return var

    def mono(self, name: str, *args: TypeTerm) -> TypeCon:
        return TypeCon(name, tuple(args))

    def scheme(self, term: TypeTerm, variables: Iterable[int] = ()) -> TypeScheme:
        return TypeScheme(frozenset(variables), term)

    def bind(self, name: str, term: TypeTerm, *, generalized: bool = False) -> None:
        if generalized:
            self.env.bind(name, self.generalize(term))
            return
        self.env.bind(name, self.scheme(term))

    def lookup(self, name: str) -> TypeTerm:
        return self.instantiate(self.env.lookup(name))

    def generalize(self, term: TypeTerm) -> TypeScheme:
        term = self.apply(term)
        variables = type_free_vars(term) - self.env.free_vars()
        return TypeScheme(variables, term)

    def instantiate(self, scheme: TypeScheme) -> TypeTerm:
        if not scheme.variables:
            return scheme.body
        replacements = {var_id: self.fresh_var() for var_id in scheme.variables}
        return apply_substitution(scheme.body, replacements)

    def require_equal(
        self, lhs: TypeTerm, rhs: TypeTerm, message: str, location: object | None = None
    ) -> None:
        self.constraints.append(Equal(lhs, rhs, ConstraintReason(message, location)))

    def require_attribute(
        self,
        owner: TypeTerm,
        name: str,
        result: TypeTerm,
        message: str,
        location: object | None = None,
    ) -> None:
        self.constraints.append(
            Attribute(owner, name, result, ConstraintReason(message, location))
        )

    def require_call(
        self,
        callee: TypeTerm,
        args: Iterable[TypeTerm],
        result: TypeTerm,
        message: str,
        location: object | None = None,
    ) -> None:
        self.constraints.append(
            Call(callee, tuple(args), result, ConstraintReason(message, location))
        )

    def require_subscript(
        self,
        container: TypeTerm,
        index: TypeTerm,
        result: TypeTerm,
        message: str,
        location: object | None = None,
    ) -> None:
        self.constraints.append(
            Subscript(container, index, result, ConstraintReason(message, location))
        )

    def require_await(
        self,
        awaitable: TypeTerm,
        result: TypeTerm,
        message: str,
        location: object | None = None,
    ) -> None:
        self.constraints.append(
            Await(awaitable, result, ConstraintReason(message, location))
        )

    def solve(self) -> Substitution:
        pending = [apply_constraint(c, self.substitution) for c in self.constraints]
        self.constraints.clear()

        while pending:
            constraint = apply_constraint(pending.pop(0), self.substitution)
            if isinstance(constraint, Equal):
                self.unify(constraint.lhs, constraint.rhs, constraint.reason)
                continue
            if isinstance(constraint, Attribute):
                resolved = self.oracle.attribute(constraint.owner, constraint.name)
                if resolved is None:
                    raise InferenceError(
                        f"Cannot resolve attribute '{constraint.name}' on "
                        f"{constraint.owner.display()}: {constraint.reason.describe()}"
                    )
                pending.insert(0, Equal(constraint.result, resolved, constraint.reason))
                continue
            if isinstance(constraint, Call):
                resolved = self.oracle.call(constraint.callee, constraint.args)
                if resolved is None:
                    raise InferenceError(
                        f"Cannot resolve call of {constraint.callee.display()}: "
                        f"{constraint.reason.describe()}"
                    )
                pending.insert(0, Equal(constraint.result, resolved, constraint.reason))
                continue
            if isinstance(constraint, Subscript):
                resolved = self.oracle.subscript(constraint.container, constraint.index)
                if resolved is None:
                    raise InferenceError(
                        f"Cannot resolve subscript of {constraint.container.display()}: "
                        f"{constraint.reason.describe()}"
                    )
                pending.insert(0, Equal(constraint.result, resolved, constraint.reason))
                continue
            resolved = self.oracle.awaitable(constraint.awaitable)
            if resolved is None:
                raise InferenceError(
                    f"Cannot await {constraint.awaitable.display()}: "
                    f"{constraint.reason.describe()}"
                )
            pending.insert(0, Equal(constraint.result, resolved, constraint.reason))

        return dict(self.substitution)

    def apply(self, term: TypeTerm) -> TypeTerm:
        return apply_substitution(term, self.substitution)

    def resolve(self, term: TypeTerm) -> TypeTerm:
        self.solve()
        result = self.apply(term)
        if type_free_vars(result):
            raise InferenceError(f"Unresolved type variable in {result.display()}")
        return result

    def unify(
        self, lhs: TypeTerm, rhs: TypeTerm, reason: ConstraintReason | None = None
    ) -> None:
        lhs = self.apply(lhs)
        rhs = self.apply(rhs)
        if lhs == rhs:
            return
        if isinstance(lhs, TypeVar):
            self.bind_var(lhs, rhs, reason)
            return
        if isinstance(rhs, TypeVar):
            self.bind_var(rhs, lhs, reason)
            return
        if isinstance(lhs, TypeCon) and isinstance(rhs, TypeCon):
            if lhs.name != rhs.name or len(lhs.args) != len(rhs.args):
                self.fail_unify(lhs, rhs, reason)
            for lhs_arg, rhs_arg in zip(lhs.args, rhs.args):
                self.unify(lhs_arg, rhs_arg, reason)
            return
        if isinstance(lhs, FuncType) and isinstance(rhs, FuncType):
            if len(lhs.args) != len(rhs.args) or len(lhs.kwonly) != len(rhs.kwonly):
                self.fail_unify(lhs, rhs, reason)
            for lhs_arg, rhs_arg in zip(lhs.args, rhs.args):
                self.unify(lhs_arg, rhs_arg, reason)
            for lhs_arg, rhs_arg in zip(lhs.kwonly, rhs.kwonly):
                self.unify(lhs_arg, rhs_arg, reason)
            self.unify(lhs.ret, rhs.ret, reason)
            return
        self.fail_unify(lhs, rhs, reason)

    def bind_var(
        self,
        var: TypeVar,
        term: TypeTerm,
        reason: ConstraintReason | None = None,
    ) -> None:
        if isinstance(term, TypeVar) and term.id == var.id:
            return
        if var.id in type_free_vars(term):
            detail = f"Recursive type {var.display()} occurs in {term.display()}"
            if reason is not None:
                detail = f"{detail}: {reason.describe()}"
            raise InferenceError(detail)
        self.substitution[var.id] = term

    def fail_unify(
        self,
        lhs: TypeTerm,
        rhs: TypeTerm,
        reason: ConstraintReason | None,
    ) -> None:
        detail = f"Cannot unify {lhs.display()} with {rhs.display()}"
        if reason is not None:
            detail = f"{detail}: {reason.describe()}"
        raise InferenceError(detail)


def parse_py_type_spec(spec: str, split_args: Callable[[str], list[str]]) -> TypeTerm:
    spec = spec.strip()
    class_prefix = '!py.class<"'
    class_suffix = '">'
    func_prefix = "!py.func<!py.funcsig<"
    scalar = {
        "!py.int": "int",
        "!py.float": "float",
        "!py.bool": "bool",
        "!py.str": "str",
        "!py.none": "None",
    }
    if spec in scalar:
        return TypeCon(scalar[spec])
    if spec == "!py.exception":
        return TypeCon("Exception")
    if spec.startswith(func_prefix) and spec.endswith(">>"):
        inner = spec[len(func_prefix) : -2]
        lhs, rhs = _split_top_level_arrow(inner)
        lhs_parts = split_args(lhs)
        if not lhs_parts:
            raise InferenceError(f"Malformed function type: {spec}")
        args = tuple(
            parse_py_type_spec(arg, split_args)
            for arg in _parse_type_list(lhs_parts[0], split_args)
        )
        kwonly: tuple[TypeTerm, ...] = ()
        for extra in lhs_parts[1:]:
            if extra.startswith("kwonly = "):
                kwonly = tuple(
                    parse_py_type_spec(arg, split_args)
                    for arg in _parse_type_list(extra[len("kwonly = ") :], split_args)
                )
        results = [
            parse_py_type_spec(ret, split_args)
            for ret in _parse_type_list(rhs, split_args)
        ]
        if len(results) != 1:
            return_type: TypeTerm = TypeCon("tuple", tuple(results))
        else:
            return_type = results[0]
        return FuncType(args, return_type, kwonly)
    if spec.startswith(class_prefix) and spec.endswith(class_suffix):
        return TypeCon(f"class:{spec[len(class_prefix) : -len(class_suffix)]}")
    for prefix, name in (
        ("!py.list<", "list"),
        ("!py.tuple<", "tuple"),
        ("!py.dict<", "dict"),
        ("!py.coro<", "coro"),
        ("!py.task<", "task"),
        ("!py.future<", "future"),
        ("!async.value<", "async.value"),
    ):
        if spec.startswith(prefix) and spec.endswith(">"):
            inner = spec[len(prefix) : -1].strip()
            if not inner:
                return TypeCon(name)
            return TypeCon(
                name,
                tuple(parse_py_type_spec(arg, split_args) for arg in split_args(inner)),
            )
    if spec.startswith("!py."):
        raise InferenceError(f"Unsupported py dialect type spec: {spec}")
    return TypeCon(spec)


def _parse_type_list(text: str, split_args: Callable[[str], list[str]]) -> list[str]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise InferenceError(f"Expected type list, got: {text}")
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    return split_args(inner)


def _split_top_level_arrow(text: str) -> tuple[str, str]:
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
    raise InferenceError(f"Malformed function signature: {text}")
