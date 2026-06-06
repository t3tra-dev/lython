from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..mlir import ir
from .inference import (
    AlgorithmM,
    DefaultOracle,
    FuncType,
    TypeCon,
    TypeTerm,
    parse_py_type_spec,
)
from .symbols import ClassInfo, FunctionInfo

TypeParser = Callable[[str], ir.Type]
PrimitiveAnnotationResolver = Callable[[ast.expr | None], ir.Type | None]
ClassLookup = Callable[[str], ClassInfo | None]
SymbolTypeLookup = Callable[[str], ir.Type | None]
FunctionLookup = Callable[[str], FunctionInfo | None]


@dataclass(slots=True)
class TypeResolver:
    """Resolve source-level Python types before emitting the highest py IR."""

    ctx: ir.Context
    parse_type: TypeParser
    primitive_annotation: PrimitiveAnnotationResolver
    class_lookup: ClassLookup
    symbol_type_lookup: SymbolTypeLookup
    function_binding_lookup: FunctionLookup
    function_lookup: FunctionLookup
    static_modules: Mapping[str, str]

    def new_inference(self) -> AlgorithmM:
        return AlgorithmM(DefaultOracle())

    def py_type_to_term(self, type_spec: str) -> TypeTerm:
        return parse_py_type_spec(type_spec, self.split_top_level_specs)

    def term_to_ir_type(self, term: TypeTerm) -> ir.Type:
        return self.parse_type(self.term_to_type_spec(term))

    def term_to_type_spec(self, term: TypeTerm) -> str:
        if isinstance(term, FuncType):
            arg_types = [self.term_to_type_spec(arg) for arg in term.args]
            result_type = self.term_to_type_spec(term.ret)
            kwonly_types = [self.term_to_type_spec(arg) for arg in term.kwonly]
            funcsig = self.build_funcsig(
                arg_types,
                [result_type],
                kwonly_types=kwonly_types,
            )
            return f"!py.func<{funcsig}>"
        if not isinstance(term, TypeCon):
            raise TypeError(f"Unresolved type variable cannot be lowered: {term}")
        scalar = {
            "int": "!py.int",
            "float": "!py.float",
            "bool": "!py.bool",
            "str": "!py.str",
            "None": "!py.none",
        }
        if term.name in scalar:
            return scalar[term.name]
        if term.name == "Exception":
            return "!py.exception"
        if term.name == "object":
            raise TypeError(
                "Generic object type is not part of the statically typed "
                "lowering ABI; use a concrete Python-level type"
            )
        if term.name.startswith("class:"):
            return f'!py.class<"{term.name[len("class:") :]}">'
        if term.name == "list" and len(term.args) == 1:
            return f"!py.list<{self.term_to_type_spec(term.args[0])}>"
        if term.name == "dict" and len(term.args) == 2:
            key = self.term_to_type_spec(term.args[0])
            value = self.term_to_type_spec(term.args[1])
            return f"!py.dict<{key}, {value}>"
        if term.name == "tuple":
            elements = ", ".join(self.term_to_type_spec(arg) for arg in term.args)
            return f"!py.tuple<{elements}>"
        if term.name in {"coro", "task", "future"} and len(term.args) == 1:
            return f"!py.{term.name}<{self.term_to_type_spec(term.args[0])}>"
        if term.name == "async.value" and len(term.args) == 1:
            return f"!async.value<{self.term_to_type_spec(term.args[0])}>"
        primitive = self._primitive_term_to_type_spec(term)
        if primitive is not None:
            return primitive
        return term.name

    def _primitive_term_to_type_spec(self, term: TypeCon) -> str | None:
        if term.name.startswith("Int[") and term.name.endswith("]"):
            return f"i{term.name[len('Int[') : -1]}"
        if term.name.startswith("Float[") and term.name.endswith("]"):
            bits = term.name[len("Float[") : -1]
            return f"f{bits}"
        if term.name in {"Matrix", "Tensor"} and len(term.args) >= 2:
            element = self.term_to_type_spec(term.args[0])
            dims = [self._dimension_term_to_spec(arg) for arg in term.args[1:]]
            return f"tensor<{'x'.join([*dims, element])}>"
        return None

    def _dimension_term_to_spec(self, term: TypeTerm) -> str:
        if isinstance(term, TypeCon):
            return term.name
        raise TypeError(f"Unresolved tensor dimension cannot be lowered: {term}")

    def annotation_to_term(self, annotation: ast.expr | None) -> TypeTerm:
        prim_type = self.primitive_annotation(annotation)
        if prim_type is not None:
            return self.py_type_to_term(str(prim_type))
        return self.py_type_to_term(self.annotation_to_py_type(annotation))

    def annotation_to_static_type(self, annotation: ast.expr | None) -> ir.Type:
        prim_type = self.primitive_annotation(annotation)
        if prim_type is not None:
            return prim_type
        return self.parse_type(self.annotation_to_py_type(annotation))

    def annotation_to_py_type(self, annotation: ast.expr | None) -> str:
        if annotation is None:
            raise TypeError(
                "Missing type annotation; Lython requires statically known "
                "Python-level types before lowering"
            )
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
            if annotation.id == "object":
                raise TypeError(
                    "Generic object annotations are not supported; use a "
                    "concrete Python-level type"
                )
            if self.class_lookup(annotation.id) is not None:
                return f'!py.class<"{annotation.id}">'
        if isinstance(annotation, ast.Subscript):
            async_kind: str | None = None
            if isinstance(annotation.value, ast.Name):
                async_kind = annotation.value.id
            elif (
                isinstance(annotation.value, ast.Attribute)
                and isinstance(annotation.value.value, ast.Name)
                and self.static_modules.get(annotation.value.value.id) == "asyncio"
            ):
                async_kind = annotation.value.attr
            async_mapping = {
                "Coroutine": "!py.coro",
                "Coro": "!py.coro",
                "Task": "!py.task",
                "Future": "!py.future",
            }
            if async_kind in async_mapping:
                element_spec = self.annotation_to_py_type(annotation.slice)
                return f"{async_mapping[async_kind]}<{element_spec}>"
            if (
                isinstance(annotation.value, ast.Name)
                and annotation.value.id == "Callable"
            ):
                return self.callable_annotation_to_py_type(annotation)
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

    def callable_annotation_to_py_type(self, annotation: ast.Subscript) -> str:
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

    def split_top_level_specs(self, text: str) -> list[str]:
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

    def split_top_level_arrow(self, text: str) -> tuple[str, str]:
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

    def parse_type_list(self, text: str) -> list[str]:
        stripped = text.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            raise ValueError(f"Expected type list, got: {text}")
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return self.split_top_level_specs(inner)

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
        lhs, rhs = self.split_top_level_arrow(inner)
        kwonly_specs: list[str] = []
        has_vararg = False
        has_kwargs = False

        lhs_parts = self.split_top_level_specs(lhs)
        if not lhs_parts:
            raise ValueError(f"Malformed callable type: {type_spec}")
        positional_specs = self.parse_type_list(lhs_parts[0])
        for extra in lhs_parts[1:]:
            if extra.startswith("kwonly = "):
                kwonly_specs = self.parse_type_list(extra[len("kwonly = ") :])
            elif extra.startswith("vararg = "):
                has_vararg = True
            elif extra.startswith("kwargs = "):
                has_kwargs = True

        result_specs = self.parse_type_list(rhs)
        return FunctionInfo(
            symbol=f"<param {name}>",
            func_type=func_type,
            arg_types=tuple(self.parse_type(spec) for spec in positional_specs),
            result_types=tuple(self.parse_type(spec) for spec in result_specs),
            has_vararg=has_vararg,
            maythrow=maythrow,
            kwonly_arg_types=tuple(self.parse_type(spec) for spec in kwonly_specs),
            has_kwargs=has_kwargs,
        )

    def awaitable_payload_type(self, awaitable_type: ir.Type) -> ir.Type | None:
        type_spec = str(awaitable_type)
        for prefix in ("!py.coro<", "!py.task<", "!py.future<", "!async.value<"):
            if type_spec.startswith(prefix) and type_spec.endswith(">"):
                return self.parse_type(type_spec[len(prefix) : -1])
        return None

    def class_info_from_type(self, obj_type: ir.Type) -> ClassInfo | None:
        obj_type_str = str(obj_type)
        if not (obj_type_str.startswith('!py.class<"') and obj_type_str.endswith('">')):
            return None
        class_name = obj_type_str[len('!py.class<"') : -len('">')]
        return self.class_lookup(class_name)

    def list_element_type(self, list_type: ir.Type) -> ir.Type | None:
        list_type_str = str(list_type)
        if not (list_type_str.startswith("!py.list<") and list_type_str.endswith(">")):
            return None
        element_spec = list_type_str[len("!py.list<") : -1]
        return self.parse_type(element_spec)

    def dict_key_value_types(
        self, dict_type: ir.Type
    ) -> tuple[ir.Type, ir.Type] | None:
        dict_type_str = str(dict_type)
        if not (dict_type_str.startswith("!py.dict<") and dict_type_str.endswith(">")):
            return None
        inner = dict_type_str[len("!py.dict<") : -1]
        parts = self.split_top_level_specs(inner)
        if len(parts) != 2:
            return None
        return self.parse_type(parts[0]), self.parse_type(parts[1])

    def attribute_type(
        self,
        obj_type: ir.Type,
        attr_name: str,
        *,
        pending_attributes: Mapping[str, ir.Type] | None = None,
    ) -> ir.Type:
        obj_type_str = str(obj_type)
        if not (obj_type_str.startswith('!py.class<"') and obj_type_str.endswith('">')):
            raise TypeError(
                f"Cannot statically resolve attribute '{attr_name}' on {obj_type}"
            )

        class_name = obj_type_str[len('!py.class<"') : -len('">')]
        if pending_attributes is not None and attr_name in pending_attributes:
            return pending_attributes[attr_name]

        class_info = self.class_lookup(class_name)
        if class_info is not None and attr_name in class_info.attributes:
            return class_info.attributes[attr_name]

        raise TypeError(
            f"Unresolved field '{attr_name}' for static class '{class_name}'"
        )

    def static_expression_type(self, expr: ast.expr) -> ir.Type | None:
        if isinstance(expr, ast.Name):
            symbol_type = self.symbol_type_lookup(expr.id)
            if symbol_type is not None:
                return symbol_type
            class_info = self.class_lookup(expr.id)
            if class_info is not None:
                return class_info.class_type
            binding_info = self.function_binding_lookup(expr.id)
            if binding_info is not None:
                return binding_info.func_type
            func_info = self.function_lookup(expr.id)
            if func_info is not None:
                return func_info.func_type
            return None

        if isinstance(expr, ast.Attribute):
            base_type = self.static_expression_type(expr.value)
            if base_type is None:
                return None
            return self.attribute_type(base_type, expr.attr)

        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                class_info = self.class_lookup(expr.func.id)
                if class_info is not None:
                    return class_info.class_type
                func_info = self.function_binding_lookup(expr.func.id)
                if func_info is None:
                    func_info = self.function_lookup(expr.func.id)
                if func_info is not None and len(func_info.result_types) == 1:
                    return func_info.result_types[0]
            if isinstance(expr.func, ast.Attribute):
                receiver_type = self.static_expression_type(expr.func.value)
                if receiver_type is None:
                    return None
                class_info = self.class_info_from_type(receiver_type)
                if class_info is None:
                    return None
                method_info = class_info.methods.get(expr.func.attr)
                if method_info is None or len(method_info.result_types) != 1:
                    return None
                return method_info.result_types[0]

        return None
