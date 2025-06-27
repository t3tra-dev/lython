"""
TypedAST ノードのテスト
"""

import ast
import unittest

from lython.frontend.ast_nodes import (
    Span,
    Ty,
    TypedArg,
    TypedArguments,
    TypedAssign,
    TypedBinOp,
    TypedConstant,
    TypedFunctionDef,
    TypedModule,
    TypedName,
)


class TestTy(unittest.TestCase):
    """Ty クラスのテスト"""

    def test_create_ty(self):
        """Ty インスタンスの作成"""
        ty = Ty("int", "i64")
        self.assertEqual(ty.py_name, "int")
        self.assertEqual(ty.llvm, "i64")
        self.assertEqual(str(ty), "int")

    def test_factory_methods(self):
        """ファクトリメソッドのテスト"""
        self.assertEqual(Ty.int().py_name, "int")
        self.assertEqual(Ty.float().py_name, "float")
        self.assertEqual(Ty.str().py_name, "str")
        self.assertEqual(Ty.bool().py_name, "bool")
        self.assertEqual(Ty.any().py_name, "Any")
        self.assertEqual(Ty.unknown().py_name, "Unknown")


class TestSpan(unittest.TestCase):
    """Span クラスのテスト"""

    def test_create_span(self):
        """Span インスタンスの作成"""
        span = Span(1, 5, 1, 10)
        self.assertEqual(span.lineno, 1)
        self.assertEqual(span.col_offset, 5)
        self.assertEqual(span.end_lineno, 1)
        self.assertEqual(span.end_col_offset, 10)
        self.assertEqual(str(span), "1:5-1:10")

    def test_span_without_end(self):
        """終了位置なしのSpan"""
        span = Span(2, 3)
        self.assertEqual(str(span), "2:3")

    def test_from_ast_node(self):
        """ast.ASTノードからSpanを作成"""
        # 簡単なASTノードを作成
        node = ast.parse("x = 42").body[0]  # Assign node
        span = Span.from_ast_node(node)
        self.assertIsInstance(span, Span)
        self.assertGreaterEqual(span.lineno, 0)


class TestTypedConstant(unittest.TestCase):
    """TypedConstant のテスト"""

    def test_create_constant(self):
        """定数ノードの作成"""
        ast_node = ast.Constant(value=42)
        ty = Ty.int()
        span = Span(1, 0)

        constant = TypedConstant(node=ast_node, ty=ty, span=span, value=42)

        self.assertEqual(constant.value, 42)
        self.assertEqual(constant.ty, ty)
        self.assertEqual(constant.span, span)


class TestTypedName(unittest.TestCase):
    """TypedName のテスト"""

    def test_create_name(self):
        """名前ノードの作成"""
        ast_node = ast.Name(id="x", ctx=ast.Load())
        ty = Ty.int()
        span = Span(1, 0)

        name = TypedName(node=ast_node, ty=ty, span=span, id="x", ctx=ast.Load())

        self.assertEqual(name.id, "x")
        self.assertIsInstance(name.ctx, ast.Load)
        self.assertEqual(name.ty, ty)


class TestTypedBinOp(unittest.TestCase):
    """TypedBinOp のテスト"""

    def test_create_binop(self):
        """二項演算ノードの作成"""
        # 左辺と右辺のノード
        left_ast = ast.Name(id="a", ctx=ast.Load())
        right_ast = ast.Name(id="b", ctx=ast.Load())

        left = TypedName(left_ast, Ty.int(), Span(1, 0), "a", ast.Load())
        right = TypedName(right_ast, Ty.int(), Span(1, 4), "b", ast.Load())

        # 二項演算ノード
        binop_ast = ast.BinOp(left=left_ast, op=ast.Add(), right=right_ast)

        binop = TypedBinOp(
            node=binop_ast,
            ty=Ty.int(),
            span=Span(1, 0, 1, 5),
            op=ast.Add(),
            left=left,
            right=right,
        )

        self.assertIsInstance(binop.op, ast.Add)
        self.assertEqual(binop.left, left)
        self.assertEqual(binop.right, right)
        self.assertEqual(binop.ty.py_name, "int")


class TestTypedAssign(unittest.TestCase):
    """TypedAssign のテスト"""

    def test_create_assign(self):
        """代入ノードの作成"""
        # target: x
        target_ast = ast.Name(id="x", ctx=ast.Store())
        target = TypedName(target_ast, Ty.int(), Span(1, 0), "x", ast.Store())

        # value: 42
        value_ast = ast.Constant(value=42)
        value = TypedConstant(value_ast, Ty.int(), Span(1, 4), 42)

        # assign: x = 42
        assign_ast = ast.Assign(targets=[target_ast], value=value_ast)

        assign = TypedAssign(
            node=assign_ast,
            ty=Ty.unknown(),  # 代入文自体の型は通常unknown
            span=Span(1, 0, 1, 6),
            targets=[target],
            value=value,
        )

        self.assertEqual(len(assign.targets), 1)
        self.assertEqual(assign.targets[0], target)
        self.assertEqual(assign.value, value)


class TestTypedFunctionDef(unittest.TestCase):
    """TypedFunctionDef のテスト"""

    def test_create_function_def(self):
        """関数定義ノードの作成"""
        # 引数: a: int, b: int
        arg1 = TypedArg(arg="a", annotation=None, ty=Ty.int())
        arg2 = TypedArg(arg="b", annotation=None, ty=Ty.int())
        args = TypedArguments(
            args=[arg1, arg2],
            posonlyargs=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        )

        # 関数本体（空）
        body = []

        # 関数定義ノード
        func_ast = ast.FunctionDef(
            name="add",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="a"), ast.arg(arg="b")],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[],
            decorator_list=[],
            returns=None,
        )  # type: ignore

        func_def = TypedFunctionDef(
            node=func_ast,
            ty=Ty("(int, int) -> int", "function"),
            span=Span(1, 0, 3, 0),
            name="add",
            args=args,
            body=body,
            decorator_list=[],
            returns=None,
            is_native=False,
        )

        self.assertEqual(func_def.name, "add")
        self.assertEqual(len(func_def.args.args), 2)
        self.assertFalse(func_def.is_native)

    def test_native_function(self):
        """@native関数のテスト"""
        arg1 = TypedArg(arg="x", annotation=None, ty=Ty("i32", "i32"))
        args = TypedArguments(
            args=[arg1],
            posonlyargs=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        )

        func_ast = ast.FunctionDef(
            name="square",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="x")],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[],
            decorator_list=[],
            returns=None,
        )  # type: ignore

        func_def = TypedFunctionDef(
            node=func_ast,
            ty=Ty("@native (i32) -> i32", "function"),
            span=Span(1, 0, 3, 0),
            name="square",
            args=args,
            body=[],
            decorator_list=[],
            returns=None,
            is_native=True,
        )

        self.assertTrue(func_def.is_native)
        self.assertEqual(func_def.name, "square")


class TestTypedModule(unittest.TestCase):
    """TypedModule のテスト"""

    def test_create_module(self):
        """モジュールノードの作成"""
        # 単純な代入文
        target_ast = ast.Name(id="x", ctx=ast.Store())
        target = TypedName(target_ast, Ty.int(), Span(1, 0), "x", ast.Store())

        value_ast = ast.Constant(value=42)
        value = TypedConstant(value_ast, Ty.int(), Span(1, 4), 42)

        assign_ast = ast.Assign(targets=[target_ast], value=value_ast)
        assign = TypedAssign(
            node=assign_ast,
            ty=Ty.unknown(),
            span=Span(1, 0, 1, 6),
            targets=[target],
            value=value,
        )

        # モジュール
        module_ast = ast.Module(body=[assign_ast], type_ignores=[])
        module = TypedModule(
            node=module_ast, ty=Ty.unknown(), span=Span(1, 0, 1, 6), body=[assign]
        )

        self.assertEqual(len(module.body), 1)
        self.assertEqual(module.body[0], assign)


class TestVisitorPattern(unittest.TestCase):
    """Visitorパターンのテスト"""

    def test_visitor_pattern(self):
        """基本的なVisitorパターンの動作確認"""
        from lython.frontend.ast_nodes import TypedASTVisitor

        class CountingVisitor(TypedASTVisitor):
            def __init__(self):
                self.constant_count = 0
                self.name_count = 0

            def visit_constant(self, node):
                self.constant_count += 1
                return None

            def visit_name(self, node):
                self.name_count += 1
                return None

        # ノードを作成
        constant = TypedConstant(
            node=ast.Constant(value=42), ty=Ty.int(), span=Span(1, 0), value=42
        )

        name = TypedName(
            node=ast.Name(id="x", ctx=ast.Load()),
            ty=Ty.int(),
            span=Span(1, 0),
            id="x",
            ctx=ast.Load(),
        )

        # Visitorを実行
        visitor = CountingVisitor()
        visitor.visit(constant)
        visitor.visit(name)

        self.assertEqual(visitor.constant_count, 1)
        self.assertEqual(visitor.name_count, 1)


if __name__ == "__main__":
    unittest.main()
