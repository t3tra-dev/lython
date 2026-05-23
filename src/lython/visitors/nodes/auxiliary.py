from __future__ import annotations

import ast

from .._base import BaseVisitor

__all__ = [
    "AliasVisitor",
    "ArgVisitor",
    "ArgumentsVisitor",
    "ComprehensionVisitor",
    "ExprContextVisitor",
    "KeywordVisitor",
    "MatchCaseVisitor",
    "TypeIgnoreVisitor",
    "TypeParamVisitor",
    "WithitemVisitor",
]


class AliasVisitor(BaseVisitor):
    """
    ```asdl
    alias = (identifier name, identifier? asname)
    ```
    """

    def visit_alias(self, node: ast.alias) -> None:
        raise NotImplementedError("alias not implemented")


class ArgVisitor(BaseVisitor):
    """
    ```asdl
    arg = (identifier arg, expr? annotation, string? type_comment)
    ```
    """

    def visit_arg(self, node: ast.arg) -> None:
        raise NotImplementedError("arg not implemented")


class ArgumentsVisitor(BaseVisitor):
    """
    ```asdl
    arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                 expr* kw_defaults, arg? kwarg, expr* defaults)
    ```
    """

    def visit_arguments(self, node: ast.arguments) -> None:
        raise NotImplementedError("arguments not implemented")


class ComprehensionVisitor(BaseVisitor):
    """
    ```asdl
    comprehension = (expr target, expr iter, expr* ifs, int is_async)
    ```
    """

    def visit_comprehension(self, node: ast.comprehension) -> None:
        raise NotImplementedError("Comprehension not implemented")


class ExprContextVisitor(BaseVisitor):
    """
    ```asdl
    expr_context = Load | Store | Del
    ```
    """

    def visit_Load(self, node: ast.Load) -> None:
        raise NotImplementedError("Load expression context not implemented")

    def visit_Store(self, node: ast.Store) -> None:
        raise NotImplementedError("Store expression context not implemented")

    def visit_Del(self, node: ast.Del) -> None:
        raise NotImplementedError("Del expression context not implemented")


class KeywordVisitor(BaseVisitor):
    """
    ```asdl
    keyword = (identifier? arg, expr value)
    ```
    """

    def visit_keyword(self, node: ast.keyword) -> None:
        raise NotImplementedError("keyword not implemented")


class MatchCaseVisitor(BaseVisitor):
    """
    ```asdl
    match_case = (pattern pattern, expr? guard, stmt* body)
    ```
    """

    def visit_match_case(self, node: ast.match_case) -> None:
        raise NotImplementedError("match_case not implemented")


class TypeIgnoreVisitor(BaseVisitor):
    """
    ```asdl
    type_ignore = TypeIgnore(int lineno, string tag)
    ```
    """

    def visit_TypeIgnore(self, node: ast.TypeIgnore) -> None:
        raise NotImplementedError("TypeIgnore not implemented")


class TypeParamVisitor(BaseVisitor):
    """
    ```asdl
    type_param = TypeVar(identifier name, expr? bound, expr? default_value)
    ```
    """

    def visit_type_param(self, node: ast.type_param) -> None:
        raise NotImplementedError("type_param not implemented")


class WithitemVisitor(BaseVisitor):
    """
    ```asdl
    withitem = (expr context_expr, expr? optional_vars)
    ```
    """

    def visit_withitem(self, node: ast.withitem) -> None:
        raise NotImplementedError("withitem not implemented")
