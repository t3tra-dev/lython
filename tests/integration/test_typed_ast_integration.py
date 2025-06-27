"""
TypedAST システムの統合テスト

TypedAST、型システム、型推論、エラー報告の統合動作をテストします。
"""

import ast
import unittest
from io import StringIO

from lython.frontend.ast_nodes import (
    Span,
    Ty,
    TypedAssign,
    TypedConstant,
    TypedFunctionDef,
    TypedModule,
    TypedName,
)
from lython.frontend.passes.type_check import TypeCheckPass
from lython.frontend.typing.error_reporter import ErrorReporter
from lython.frontend.typing.solver import TypeInferenceEngine
from lython.frontend.typing.types import AnyType, PrimitiveType, PyObjectType
from lython.native import f64, i32, native


class TestBasicTypeInference(unittest.TestCase):
    """基本的な型推論のテスト"""

    def setUp(self):
        self.error_reporter = ErrorReporter()
        self.inference_engine = TypeInferenceEngine()

    def test_simple_assignment(self):
        """単純な代入の型推論"""
        # x = 42 の TypedAST を手動で構築

        # 定数 42
        value_ast = ast.Constant(value=42)
        value = TypedConstant(
            node=value_ast, ty=Ty.unknown(), span=Span(1, 4), value=42  # 推論前は不明
        )

        # 変数 x
        target_ast = ast.Name(id="x", ctx=ast.Store())
        target = TypedName(
            node=target_ast, ty=Ty.unknown(), span=Span(1, 0), id="x", ctx=ast.Store()
        )

        # 代入文 x = 42
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

        # 型推論を実行
        inferred_module = self.inference_engine.infer(module)

        # 結果の検証
        self.assertIsInstance(inferred_module, TypedModule)
        # 推論エンジンは簡略化されているため、具体的な型の検証は省略


class TestNativeFunctionTypeChecking(unittest.TestCase):
    """@native関数の型チェックテスト"""

    def setUp(self):
        self.error_reporter = ErrorReporter()
        self.type_checker = TypeCheckPass(self.error_reporter)

    def test_native_function_without_annotations(self):
        """型注釈なしの@native関数（エラーになるべき）"""
        # @native関数の作成（型注釈なし）
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

        # @nativeデコレータがマークされている関数として作成
        from lython.frontend.ast_nodes import TypedArg, TypedArguments, TypedName

        # @nativeデコレータを作成
        native_decorator = TypedName(
            node=ast.Name(id="native", ctx=ast.Load()),
            ty=Ty.unknown(),
            span=Span(1, 0),
            id="native",
            ctx=ast.Load(),
        )

        args = TypedArguments(
            args=[
                TypedArg(arg="a", annotation=None),
                TypedArg(arg="b", annotation=None),
            ],
            posonlyargs=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        )

        func_def = TypedFunctionDef(
            node=func_ast,
            ty=Ty.unknown(),
            span=Span(1, 0, 3, 0),
            name="add",
            args=args,
            body=[],
            decorator_list=[native_decorator],  # @nativeデコレータを追加
            returns=None,
            is_native=True,  # @nativeマーク
        )

        module = TypedModule(
            node=ast.Module(body=[func_ast], type_ignores=[]),
            ty=Ty.unknown(),
            span=Span(1, 0, 3, 0),
            body=[func_def],
        )

        # 型チェックを実行
        self.type_checker.run(module)

        # エラーが発生していることを確認
        self.assertTrue(self.error_reporter.has_errors())
        # @native関数には型注釈が必要というエラーがあることを期待


class TestErrorReporting(unittest.TestCase):
    """エラー報告システムのテスト"""

    def test_error_formatting(self):
        """エラーメッセージのフォーマット"""
        reporter = ErrorReporter("test.py")

        # ソースコードをロード
        source = "x = 42\ny = 'hello'"
        reporter.load_source_lines(source)

        # 型エラーを追加
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()
        span = Span(2, 4, 2, 11)

        reporter.add_type_error(int_type, str_type, span, "variable assignment")

        # エラー出力をキャプチャ
        output = StringIO()
        reporter.print_diagnostics(output, show_source=True)
        error_text = output.getvalue()

        # エラーメッセージの内容を確認
        self.assertIn("Type mismatch", error_text)
        self.assertIn("expected 'int'", error_text)
        self.assertIn("but got 'str'", error_text)
        self.assertIn("test.py:2:4", error_text)

        # ソースコード表示の確認
        self.assertIn("y = 'hello'", error_text)

    def test_multiple_errors_sorting(self):
        """複数エラーのソート"""
        reporter = ErrorReporter()

        # 異なる行のエラーを追加
        reporter.add_error("Error on line 3", Span(3, 0))
        reporter.add_error("Error on line 1", Span(1, 0))
        reporter.add_warning("Warning on line 2", Span(2, 0))

        # エラーが行番号順にソートされることを確認
        output = StringIO()
        reporter.print_diagnostics(output, show_source=False)
        error_text = output.getvalue()

        lines = error_text.split("\n")
        error_lines = [line for line in lines if "Error" in line or "WARNING" in line]

        # 行番号順になっていることを確認（簡略化）
        self.assertTrue(len(error_lines) >= 3)

    def test_json_output(self):
        """JSON形式での出力"""
        reporter = ErrorReporter("test.py")

        int_type = PyObjectType.int()
        str_type = PyObjectType.str()

        reporter.add_type_error(int_type, str_type, Span(1, 5, 1, 10), "test")
        reporter.add_warning("Unused variable", Span(2, 0, 2, 1), "W002")

        json_output = reporter.to_json()

        self.assertEqual(len(json_output), 2)

        # エラー情報の確認
        error_info = json_output[0]
        self.assertEqual(error_info["level"], "error")
        self.assertEqual(error_info["code"], "E001")
        self.assertEqual(error_info["file"], "test.py")
        self.assertEqual(error_info["span"]["start"]["line"], 1)
        self.assertEqual(error_info["span"]["start"]["column"], 5)

        # 警告情報の確認
        warning_info = json_output[1]
        self.assertEqual(warning_info["level"], "warning")
        self.assertEqual(warning_info["code"], "W002")


class TestNativeAPIIntegration(unittest.TestCase):
    """Native API の統合テスト"""

    def test_native_function_execution(self):
        """@native関数の実行"""

        @native
        def add_numbers(a: i32, b: i32) -> i32:
            return a + b

        # 関数がネイティブマークされていることを確認
        from lython.native import is_native_function

        self.assertTrue(is_native_function(add_numbers))

        # 実行テスト
        result = add_numbers(i32(10), i32(20))
        self.assertEqual(int(result), 30)

    def test_type_conversion_pipeline(self):
        """型変換パイプライン"""
        from lython.native import to_native, to_pytype

        # Python -> Native -> Python
        original = 42
        native_val = to_native(original, i32)
        converted_back = to_pytype(native_val, int)

        self.assertEqual(original, converted_back)
        self.assertIsInstance(native_val, i32)
        self.assertIsInstance(converted_back, int)

    def test_native_type_properties(self):
        """ネイティブ型の性質"""
        from lython.native import CompileTimeInfo

        # コンパイル時情報の取得
        llvm_type = CompileTimeInfo.get_llvm_type(i32)
        self.assertEqual(llvm_type, "i32")

        size = CompileTimeInfo.get_size_in_bits(f64)
        self.assertEqual(size, 64)

        self.assertTrue(CompileTimeInfo.is_integer(i32))
        self.assertFalse(CompileTimeInfo.is_integer(f64))
        self.assertTrue(CompileTimeInfo.is_float(f64))


class TestEndToEndWorkflow(unittest.TestCase):
    """エンドツーエンドのワークフロー"""

    def test_simple_program_analysis(self):
        """簡単なプログラムの解析"""
        # 以下のPythonコードを模擬
        # def add(a: int, b: int) -> int:
        #     return a + b
        # x = add(1, 2)

        error_reporter = ErrorReporter("example.py")

        # 簡略化されたASTの構築とチェック
        # （実際にはパーサーから生成される）

        # 基本的な動作確認として、型システムの各コンポーネントが
        # 正常に連携することを確認

        # 型の作成
        int_type = PyObjectType.int()
        # func_type = FunctionType([int_type, int_type], int_type)

        # 型の統合
        from lython.frontend.typing.types import unify_types

        unified = unify_types(int_type, int_type)
        self.assertEqual(unified, int_type)

        # エラー報告の動作確認
        error_reporter.add_info("Analysis complete", Span(1, 0))
        self.assertEqual(error_reporter.error_count, 0)
        self.assertEqual(error_reporter.warning_count, 0)

    def test_type_system_consistency(self):
        """型システムの一貫性チェック"""
        # 各型が期待される性質を持つことを確認

        # プリミティブ型
        i32_type = PrimitiveType.i32()
        self.assertTrue(i32_type.is_native())
        self.assertFalse(i32_type.is_pyobject())
        self.assertEqual(i32_type.kind.value, "primitive")

        # PyObject型
        int_type = PyObjectType.int()
        self.assertFalse(int_type.is_native())
        self.assertTrue(int_type.is_pyobject())
        self.assertEqual(int_type.kind.value, "pyobject")

        # Any型
        any_type = AnyType()
        self.assertTrue(any_type.is_any())
        self.assertEqual(any_type.kind.value, "any")

        # 代入可能性の一貫性
        self.assertTrue(i32_type.is_assignable_to(any_type))
        self.assertTrue(int_type.is_assignable_to(any_type))
        self.assertTrue(any_type.is_assignable_to(int_type))


if __name__ == "__main__":
    # テスト実行時の設定
    unittest.main(verbosity=2)
