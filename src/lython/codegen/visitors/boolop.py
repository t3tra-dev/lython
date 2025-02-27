import ast

from ..ir import IRBuilder
from .base import BaseVisitor, TypedValue
from .expr import ExprVisitor

__all__ = ["BoolOpVisitor"]


class BoolOpVisitor(BaseVisitor):
    """
    論理演算子 (boolop) ノードを処理し、
    LLVM IR生成を行うクラス。
    オブジェクトシステムと連携。

    ```asdl
    boolop = And | Or
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit(self, node: ast.BoolOp, expr_visitor: ExprVisitor) -> TypedValue:
        """
        論理演算子を処理する。
        短絡評価を適切に実装する。
        """
        if isinstance(node.op, ast.And):
            return self.visit_And(node, expr_visitor)
        elif isinstance(node.op, ast.Or):
            return self.visit_Or(node, expr_visitor)
        else:
            raise NotImplementedError(f"Unsupported boolop: {type(node.op).__name__}")

    def visit_And(self, node: ast.BoolOp, expr_visitor: ExprVisitor) -> TypedValue:
        """
        AND演算子 (論理積) を処理する。
        短絡評価: 左辺が偽なら右辺は評価しない。

        ```asdl
        And
        ```
        """
        # 2つ以上の項の場合は再帰的に処理
        if len(node.values) > 2:
            left_values = node.values[:-1]
            right_value = node.values[-1]

            # 左側の値を再帰的に処理
            left_node = ast.BoolOp(op=node.op, values=left_values)
            left_node.lineno = node.lineno
            left_node.col_offset = node.col_offset

            left = self.visit(left_node, expr_visitor)
            right = expr_visitor.visit(right_value)

            # 両値をAND演算
            return self._create_and_ir(left, right)

        # 基本ケース: 2つの項の場合
        left = expr_visitor.visit(node.values[0])
        right = expr_visitor.visit(node.values[1])

        return self._create_and_ir(left, right)

    def _create_and_ir(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """AND演算のためのIR生成"""
        # ラベルを生成
        eval_right_label = f"and.right.{self.builder.get_label_counter()}"
        skip_right_label = f"and.end.{self.builder.get_label_counter()}"

        # 結果を格納する変数
        result = self.builder.get_temp_name()
        self.builder.emit(f"  {result} = alloca ptr")

        # 左辺の評価と真偽チェック
        left_obj = self.ensure_object(left)
        is_true = self.builder.get_temp_name()
        self.builder.emit(f"  {is_true} = call i32 @PyObject_IsTrue(ptr {left_obj.llvm_value})")

        # 条件分岐: 左辺が真なら右辺を評価、そうでなければ左辺を結果とする
        condition = self.builder.get_temp_name()
        self.builder.emit(f"  {condition} = icmp ne i32 {is_true}, 0")
        self.builder.emit(f"  br i1 {condition}, label %{eval_right_label}, label %{skip_right_label}")

        # 右辺を評価するブロック
        self.builder.emit(f"{eval_right_label}:")
        right_obj = self.ensure_object(right)
        self.builder.emit(f"  store ptr {right_obj.llvm_value}, ptr {result}")
        self.builder.emit(f"  br label %{skip_right_label}")

        # 結果を返すブロック
        self.builder.emit(f"{skip_right_label}:")
        self.builder.emit("  ; Short-circuit evaluation of AND operation: return left if false, right if true")
        left_is_false = self.builder.get_temp_name()
        self.builder.emit(f"  {left_is_false} = icmp eq i32 {is_true}, 0")
        phi_result = self.builder.get_temp_name()
        self.builder.emit(f"  {phi_result} = select i1 {left_is_false}, ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value}")

        return TypedValue.create_object(phi_result, "bool")

    def visit_Or(self, node: ast.BoolOp, expr_visitor: ExprVisitor) -> TypedValue:
        """
        OR演算子 (論理和) を処理する。
        短絡評価: 左辺が真なら右辺は評価しない。

        ```asdl
        Or
        ```
        """
        # 2つ以上の項の場合は再帰的に処理
        if len(node.values) > 2:
            left_values = node.values[:-1]
            right_value = node.values[-1]

            # 左側の値を再帰的に処理
            left_node = ast.BoolOp(op=node.op, values=left_values)
            left_node.lineno = node.lineno
            left_node.col_offset = node.col_offset

            left = self.visit(left_node, expr_visitor)
            right = expr_visitor.visit(right_value)

            # 両値をOR演算
            return self._create_or_ir(left, right)

        # 基本ケース: 2つの項の場合
        left = expr_visitor.visit(node.values[0])
        right = expr_visitor.visit(node.values[1])

        return self._create_or_ir(left, right)

    def _create_or_ir(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """OR演算のためのIR生成"""
        # ラベルを生成
        eval_right_label = f"or.right.{self.builder.get_label_counter()}"
        skip_right_label = f"or.end.{self.builder.get_label_counter()}"

        # 結果を格納する変数
        result = self.builder.get_temp_name()
        self.builder.emit(f"  {result} = alloca ptr")

        # 左辺の評価と真偽チェック
        left_obj = self.ensure_object(left)
        is_true = self.builder.get_temp_name()
        self.builder.emit(f"  {is_true} = call i32 @PyObject_IsTrue(ptr {left_obj.llvm_value})")

        # 条件分岐: 左辺が偽なら右辺を評価、そうでなければ左辺を結果とする
        condition = self.builder.get_temp_name()
        self.builder.emit(f"  {condition} = icmp eq i32 {is_true}, 0")
        self.builder.emit(f"  br i1 {condition}, label %{eval_right_label}, label %{skip_right_label}")

        # 右辺を評価するブロック
        self.builder.emit(f"{eval_right_label}:")
        right_obj = self.ensure_object(right)
        self.builder.emit(f"  store ptr {right_obj.llvm_value}, ptr {result}")
        self.builder.emit(f"  br label %{skip_right_label}")

        # 結果を返すブロック
        self.builder.emit(f"{skip_right_label}:")
        self.builder.emit("  ; Short-circuit evaluation of OR operation: return left if true, otherwise return right")
        left_is_true = self.builder.get_temp_name()
        self.builder.emit(f"  {left_is_true} = icmp ne i32 {is_true}, 0")
        phi_result = self.builder.get_temp_name()
        self.builder.emit(f"  {phi_result} = select i1 {left_is_true}, ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value}")

        return TypedValue.create_object(phi_result, "bool")
