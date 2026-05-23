from __future__ import annotations

import ast

from .._base import BaseVisitor

__all__ = ["BoolOpVisitor", "CmpOpVisitor", "OperatorVisitor", "UnaryOpVisitor"]


class BoolOpVisitor(BaseVisitor):
    """
    ```asdl
    boolop = And | Or
    ```
    """

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        raise NotImplementedError(f"Unsupported boolop: {type(node.op).__name__}")

    def visit_And(self, node: ast.And) -> None:
        raise NotImplementedError("And not implemented")

    def visit_Or(self, node: ast.Or) -> None:
        raise NotImplementedError("Or not implemented")


class OperatorVisitor(BaseVisitor):
    """
    ```asdl
    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow
             | LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
    ```
    """

    def visit_Add(self, node: ast.Add) -> None:
        raise NotImplementedError("Add operator not implemented")

    def visit_Sub(self, node: ast.Sub) -> None:
        raise NotImplementedError("Sub operator not implemented")

    def visit_Mult(self, node: ast.Mult) -> None:
        raise NotImplementedError("Mult operator not implemented")

    def visit_MatMult(self, node: ast.MatMult) -> None:
        raise NotImplementedError("MatMult operator not implemented")

    def visit_Div(self, node: ast.Div) -> None:
        raise NotImplementedError("Div operator not implemented")

    def visit_Mod(self, node: ast.Mod) -> None:
        raise NotImplementedError("Mod operator not implemented")

    def visit_Pow(self, node: ast.Pow) -> None:
        raise NotImplementedError("Pow operator not implemented")

    def visit_LShift(self, node: ast.LShift) -> None:
        raise NotImplementedError("LShift operator not implemented")

    def visit_RShift(self, node: ast.RShift) -> None:
        raise NotImplementedError("RShift operator not implemented")

    def visit_BitOr(self, node: ast.BitOr) -> None:
        raise NotImplementedError("BitOr operator not implemented")

    def visit_BitXor(self, node: ast.BitXor) -> None:
        raise NotImplementedError("BitXor operator not implemented")

    def visit_BitAnd(self, node: ast.BitAnd) -> None:
        raise NotImplementedError("BitAnd operator not implemented")

    def visit_FloorDiv(self, node: ast.FloorDiv) -> None:
        raise NotImplementedError("FloorDiv operator not implemented")


class UnaryOpVisitor(BaseVisitor):
    """
    ```asdl
    unaryop = Invert | Not | UAdd | USub
    ```
    """

    def visit_Invert(self, node: ast.Invert) -> None:
        raise NotImplementedError("Invert unaryop not implemented")

    def visit_Not(self, node: ast.Not) -> None:
        raise NotImplementedError("Not unaryop not implemented")

    def visit_UAdd(self, node: ast.UAdd) -> None:
        raise NotImplementedError("UAdd unaryop not implemented")

    def visit_USub(self, node: ast.USub) -> None:
        raise NotImplementedError("USub unaryop not implemented")


class CmpOpVisitor(BaseVisitor):
    """
    ```asdl
    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    ```
    """

    def visit_Eq(self, node: ast.Eq) -> None:
        raise NotImplementedError("'==' operator not supported")

    def visit_NotEq(self, node: ast.NotEq) -> None:
        raise NotImplementedError("'!=' operator not supported")

    def visit_Lt(self, node: ast.Lt) -> None:
        raise NotImplementedError("'<' operator not supported")

    def visit_LtE(self, node: ast.LtE) -> None:
        raise NotImplementedError("'<=' operator not supported")

    def visit_Gt(self, node: ast.Gt) -> None:
        raise NotImplementedError("'>' operator not supported")

    def visit_GtE(self, node: ast.GtE) -> None:
        raise NotImplementedError("'>=' operator not supported")

    def visit_Is(self, node: ast.Is) -> None:
        raise NotImplementedError("'is' operator not supported")

    def visit_IsNot(self, node: ast.IsNot) -> None:
        raise NotImplementedError("'is not' operator not supported")

    def visit_In(self, node: ast.In) -> None:
        raise NotImplementedError("'in' operator not supported")

    def visit_NotIn(self, node: ast.NotIn) -> None:
        raise NotImplementedError("'not in' operator not supported")
