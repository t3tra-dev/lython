"""
TypedAST ノード定義

このモジュールは、CPython の ast モジュールを拡張し、
型情報を保持する TypedAST ノードを定義します。
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from typing import Any as VisitorReturn


@dataclass
class Ty:
    """型のメタ情報 (モノタイプ/ユニオン/ジェネリックの基底)"""

    py_name: str  # 'int', 'str', 'list[int]' など
    llvm: Optional[str]  # 'i32', '%pyobj*' など

    def __str__(self) -> str:
        return self.py_name

    def __repr__(self) -> str:
        return f"Ty(py_name='{self.py_name}', llvm='{self.llvm}')"

    @classmethod
    def unknown(cls) -> Ty:
        """未知の型を表す"""
        return cls("Unknown", None)

    @classmethod
    def any(cls) -> Ty:
        """Any型を表す"""
        return cls("Any", "%pyobj*")

    @classmethod
    def int(cls) -> Ty:
        """int型を表す"""
        return cls("int", "i64")

    @classmethod
    def float(cls) -> Ty:
        """float型を表す"""
        return cls("float", "double")

    @classmethod
    def str(cls) -> Ty:
        """str型を表す"""
        return cls("str", "%pyobj*")

    @classmethod
    def bool(cls) -> Ty:
        """bool型を表す"""
        return cls("bool", "i1")


@dataclass
class Span:
    """ソースコード上の位置情報"""

    lineno: int
    col_offset: int
    end_lineno: Optional[int] = None
    end_col_offset: Optional[int] = None

    def __str__(self) -> str:
        if self.end_lineno is not None:
            return (
                f"{self.lineno}:{self.col_offset}-"
                f"{self.end_lineno}:{self.end_col_offset}"
            )
        return f"{self.lineno}:{self.col_offset}"

    @classmethod
    def from_ast_node(cls, node: ast.AST) -> Span:
        """ast.ASTノードから位置情報を抽出"""
        return cls(
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
        )


@dataclass
class TypedNode(ABC):
    """全 TypedAST ノードの共通ベース"""

    node: ast.AST  # 元 CPython AST ノード
    ty: Ty  # 推論結果
    span: Span  # ソースコード位置情報

    def __post_init__(self) -> None:
        # nodeからspanが作成されていない場合は自動生成
        if not hasattr(self, "span"):
            self.span = Span.from_ast_node(self.node)

    @abstractmethod
    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        """Visitorパターンの実装"""
        pass


# 式ノード (Expression nodes)


@dataclass
class TypedExpr(TypedNode):
    """式の基底クラス"""

    pass


@dataclass
class TypedConstant(TypedExpr):
    """定数リテラル (数値、文字列、True/False/None)"""

    value: Any

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_constant(self)


@dataclass
class TypedName(TypedExpr):
    """変数名参照"""

    id: str
    ctx: ast.expr_context

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_name(self)


@dataclass
class TypedBinOp(TypedExpr):
    """二項演算 (a + b, a - b など)"""

    op: ast.operator
    left: TypedExpr
    right: TypedExpr

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_binop(self)


@dataclass
class TypedUnaryOp(TypedExpr):
    """単項演算 (-a, not a など)"""

    op: ast.unaryop
    operand: TypedExpr

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_unaryop(self)


@dataclass
class TypedCall(TypedExpr):
    """関数呼び出し"""

    func: TypedExpr
    args: List[TypedExpr]
    keywords: List["TypedKeyword"]

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_call(self)


@dataclass
class TypedAttribute(TypedExpr):
    """属性参照 (obj.attr)"""

    value: TypedExpr
    attr: str
    ctx: ast.expr_context

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_attribute(self)


@dataclass
class TypedSubscript(TypedExpr):
    """添字参照 (obj[key])"""

    value: TypedExpr
    slice: TypedExpr
    ctx: ast.expr_context

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_subscript(self)


@dataclass
class TypedList(TypedExpr):
    """リストリテラル [1, 2, 3]"""

    elts: List[TypedExpr]
    ctx: ast.expr_context

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_list(self)


@dataclass
class TypedTuple(TypedExpr):
    """タプルリテラル (1, 2, 3)"""

    elts: List[TypedExpr]
    ctx: ast.expr_context

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_tuple(self)


@dataclass
class TypedDict(TypedExpr):
    """辞書リテラル {key: value}"""

    keys: List[Optional[TypedExpr]]
    values: List[TypedExpr]

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_dict(self)


# 文ノード (Statement nodes)


@dataclass
class TypedStmt(TypedNode):
    """文の基底クラス"""

    pass


@dataclass
class TypedAssign(TypedStmt):
    """代入文 (a = b)"""

    targets: List[TypedExpr]
    value: TypedExpr
    type_comment: Optional[str] = None

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_assign(self)


@dataclass
class TypedAnnAssign(TypedStmt):
    """型注釈付き代入 (a: int = 5)"""

    target: TypedExpr
    annotation: TypedExpr
    value: Optional[TypedExpr]
    simple: int  # 1 if simple target name, 0 otherwise

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_ann_assign(self)


@dataclass
class TypedAugAssign(TypedStmt):
    """拡張代入 (a += b)"""

    target: TypedExpr
    op: ast.operator
    value: TypedExpr

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_aug_assign(self)


@dataclass
class TypedExprStmt(TypedStmt):
    """式文 (単独の式)"""

    value: TypedExpr

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_expr_stmt(self)


@dataclass
class TypedReturn(TypedStmt):
    """return文"""

    value: Optional[TypedExpr]

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_return(self)


@dataclass
class TypedIf(TypedStmt):
    """if文"""

    test: TypedExpr
    body: List[TypedStmt]
    orelse: List[TypedStmt]

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_if(self)


@dataclass
class TypedWhile(TypedStmt):
    """while文"""

    test: TypedExpr
    body: List[TypedStmt]
    orelse: List[TypedStmt]

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_while(self)


@dataclass
class TypedFor(TypedStmt):
    """for文"""

    target: TypedExpr
    iter: TypedExpr
    body: List[TypedStmt]
    orelse: List[TypedStmt]
    type_comment: Optional[str] = None

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_for(self)


@dataclass
class TypedFunctionDef(TypedStmt):
    """関数定義"""

    name: str
    args: "TypedArguments"
    body: List[TypedStmt]
    decorator_list: List[TypedExpr]
    returns: Optional[TypedExpr]
    type_comment: Optional[str] = None
    is_native: bool = False  # @native デコレータが付いているか

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_function_def(self)


@dataclass
class TypedClassDef(TypedStmt):
    """クラス定義"""

    name: str
    bases: List[TypedExpr]
    keywords: List["TypedKeyword"]
    body: List[TypedStmt]
    decorator_list: List[TypedExpr]

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_class_def(self)


# 補助的なノード


@dataclass
class TypedArguments:
    """関数の引数リスト"""

    args: List["TypedArg"]
    posonlyargs: List["TypedArg"]
    kwonlyargs: List["TypedArg"]
    kw_defaults: List[Optional[TypedExpr]]
    defaults: List[TypedExpr]
    vararg: Optional["TypedArg"]
    kwarg: Optional["TypedArg"]


@dataclass
class TypedArg:
    """関数の引数"""

    arg: str
    annotation: Optional[TypedExpr]
    type_comment: Optional[str] = None
    ty: Optional[Ty] = None  # 型推論結果


@dataclass
class TypedKeyword:
    """キーワード引数"""

    arg: Optional[str]  # None for **kwargs
    value: TypedExpr


@dataclass
class TypedModule(TypedNode):
    """モジュール全体"""

    body: List[TypedStmt]
    type_ignores: Optional[List[str]] = None

    def accept(self, visitor: "TypedASTVisitor") -> "VisitorReturn":
        return visitor.visit_module(self)


# Visitor パターンの基底クラス


class TypedASTVisitor(ABC):
    """TypedAST を走査するための Visitor パターン基底クラス"""

    def visit(self, node: TypedNode) -> "VisitorReturn":
        """ノードを訪問する"""
        return node.accept(self)

    def generic_visit(self, node: TypedNode) -> "VisitorReturn":
        """デフォルトの訪問処理"""
        return None

    # 式ノード用のメソッド
    def visit_constant(self, node: TypedConstant) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_name(self, node: TypedName) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_binop(self, node: TypedBinOp) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_unaryop(self, node: TypedUnaryOp) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_call(self, node: TypedCall) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_attribute(self, node: TypedAttribute) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_subscript(self, node: TypedSubscript) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_list(self, node: TypedList) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_tuple(self, node: TypedTuple) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_dict(self, node: TypedDict) -> "VisitorReturn":
        return self.generic_visit(node)

    # 文ノード用のメソッド
    def visit_assign(self, node: TypedAssign) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_ann_assign(self, node: TypedAnnAssign) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_aug_assign(self, node: TypedAugAssign) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_expr_stmt(self, node: TypedExprStmt) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_return(self, node: TypedReturn) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_if(self, node: TypedIf) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_while(self, node: TypedWhile) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_for(self, node: TypedFor) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_function_def(self, node: TypedFunctionDef) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_class_def(self, node: TypedClassDef) -> "VisitorReturn":
        return self.generic_visit(node)

    def visit_module(self, node: TypedModule) -> "VisitorReturn":
        return self.generic_visit(node)
