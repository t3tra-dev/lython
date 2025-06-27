"""
Native API のテスト
"""

import unittest

from lython.native import (
    CompileTimeInfo,
    f32,
    f64,
    get_native_type_name,
    i8,
    i16,
    i32,
    i64,
    is_native_function,
    is_native_type,
    native,
    to_native,
    to_pytype,
)


class TestNativeTypes(unittest.TestCase):
    """ネイティブ型のテスト"""

    def test_i32_creation(self):
        """i32型の作成"""
        x = i32(42)
        self.assertEqual(int(x), 42)
        self.assertEqual(str(x), "i32(42)")
        self.assertEqual(repr(x), "i32(42)")

    def test_i32_range_check(self):
        """i32型の範囲チェック"""
        # 正常な値
        x = i32(2147483647)  # max i32
        self.assertEqual(int(x), 2147483647)

        y = i32(-2147483648)  # min i32
        self.assertEqual(int(y), -2147483648)

        # 範囲外の値
        with self.assertRaises(ValueError):
            i32(2147483648)  # max + 1

        with self.assertRaises(ValueError):
            i32(-2147483649)  # min - 1

    def test_i32_arithmetic(self):
        """i32型の算術演算"""
        a = i32(10)
        b = i32(5)

        # 加算
        c = a + b
        self.assertIsInstance(c, i32)
        self.assertEqual(int(c), 15)

        # 減算
        d = a - b
        self.assertIsInstance(d, i32)
        self.assertEqual(int(d), 5)

        # 乗算
        e = a * b
        self.assertIsInstance(e, i32)
        self.assertEqual(int(e), 50)

        # 除算
        f = a // b
        self.assertIsInstance(f, i32)
        self.assertEqual(int(f), 2)

        # 剰余
        g = a % b
        self.assertIsInstance(g, i32)
        self.assertEqual(int(g), 0)

    def test_i32_comparison(self):
        """i32型の比較演算"""
        a = i32(10)
        b = i32(5)
        c = i32(10)

        # 等価性
        self.assertTrue(a == c)
        self.assertFalse(a == b)

        # 大小関係
        self.assertTrue(a > b)
        self.assertTrue(a >= b)
        self.assertTrue(a >= c)
        self.assertTrue(b < a)
        self.assertTrue(b <= a)
        self.assertTrue(c <= a)

    def test_f64_creation(self):
        """f64型の作成"""
        x = f64(3.14159)
        self.assertAlmostEqual(float(x), 3.14159, places=5)
        self.assertEqual(str(x), "f64(3.14159)")

        # intからの変換
        y = f64(42)
        self.assertEqual(float(y), 42.0)

    def test_f64_arithmetic(self):
        """f64型の算術演算"""
        a = f64(3.0)
        b = f64(2.0)

        # 加算
        c = a + b
        self.assertIsInstance(c, f64)
        self.assertAlmostEqual(float(c), 5.0)

        # 減算
        d = a - b
        self.assertIsInstance(d, f64)
        self.assertAlmostEqual(float(d), 1.0)

        # 乗算
        e = a * b
        self.assertIsInstance(e, f64)
        self.assertAlmostEqual(float(e), 6.0)

        # 除算
        f = a / b
        self.assertIsInstance(f, f64)
        self.assertAlmostEqual(float(f), 1.5)

    def test_f64_equality(self):
        """f64型の等価性（浮動小数点の誤差を考慮）"""
        a = f64(3.14159)
        b = f64(3.14159)
        c = f64(3.14160)

        self.assertTrue(a == b)
        self.assertFalse(a == c)

    def test_other_integer_types(self):
        """他の整数型のテスト"""
        # i8
        x8 = i8(100)
        self.assertEqual(int(x8), 100)
        with self.assertRaises(ValueError):
            i8(128)  # 範囲外

        # i16
        x16 = i16(30000)
        self.assertEqual(int(x16), 30000)
        with self.assertRaises(ValueError):
            i16(32768)  # 範囲外

        # i64
        x64 = i64(9223372036854775807)  # max i64
        self.assertEqual(int(x64), 9223372036854775807)

    def test_f32_type(self):
        """f32型のテスト"""
        x = f32(3.14)
        self.assertAlmostEqual(float(x), 3.14, places=2)

        y = f32(42)  # intからの変換
        self.assertEqual(float(y), 42.0)

    def test_type_conversion_errors(self):
        """型変換エラーのテスト"""
        with self.assertRaises(TypeError):
            i32("not a number")

        with self.assertRaises(TypeError):
            f64("not a number")


class TestTypeConversion(unittest.TestCase):
    """型変換関数のテスト"""

    def test_to_native_integer(self):
        """整数のネイティブ型変換"""
        # Python int -> i32
        result = to_native(42, i32)
        self.assertIsInstance(result, i32)
        self.assertEqual(int(result), 42)

        # Python int -> i64
        result = to_native(12345, i64)
        self.assertIsInstance(result, i64)
        self.assertEqual(int(result), 12345)

        # Python int -> i8 (範囲チェック)
        result = to_native(100, i8)
        self.assertIsInstance(result, i8)
        self.assertEqual(int(result), 100)

    def test_to_native_float(self):
        """浮動小数点のネイティブ型変換"""
        # Python float -> f64
        result = to_native(3.14159, f64)
        self.assertIsInstance(result, f64)
        self.assertAlmostEqual(float(result), 3.14159, places=5)

        # Python float -> f32
        result = to_native(2.718, f32)
        self.assertIsInstance(result, f32)
        self.assertAlmostEqual(float(result), 2.718, places=3)

    def test_to_native_errors(self):
        """to_native のエラーケース"""
        with self.assertRaises(TypeError):
            to_native(42, str)  # type: ignore # 未対応の型

        with self.assertRaises(ValueError):
            to_native(1000, i8)  # 範囲外

    def test_to_pytype_integer(self):
        """ネイティブ整数型のPython型変換"""
        x = i32(42)

        # デフォルト変換（int）
        result = to_pytype(x)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)

        # 明示的にint
        result = to_pytype(x, int)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)

        # float変換
        result = to_pytype(x, float)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 42.0)

        # str変換
        result = to_pytype(x, str)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "42")

    def test_to_pytype_float(self):
        """ネイティブ浮動小数点型のPython型変換"""
        x = f64(3.14159)

        # デフォルト変換（float）
        result = to_pytype(x)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14159, places=5)

        # int変換
        result = to_pytype(x, int)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 3)

        # str変換
        result = to_pytype(x, str)
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("3.14"))

    def test_to_pytype_errors(self):
        """to_pytype のエラーケース"""
        with self.assertRaises(TypeError):
            to_pytype("not native type")  # type: ignore


class TestNativeDecorator(unittest.TestCase):
    """@nativeデコレータのテスト"""

    def test_native_decorator_basic(self):
        """基本的な@nativeデコレータ"""

        @native
        def add(a: i32, b: i32) -> i32:
            return a + b

        # マーキングの確認
        self.assertTrue(hasattr(add, "__ly_native__"))
        self.assertTrue(add.__ly_native__)  # type: ignore
        self.assertFalse(add.__ly_inline__)  # type: ignore

        # 実行テスト
        result = add(i32(10), i32(20))
        self.assertIsInstance(result, i32)
        self.assertEqual(int(result), 30)

    def test_native_decorator_with_inline(self):
        """インライン指定の@nativeデコレータ"""

        @native(inline=True)
        def square(x: f64) -> f64:
            return x * x

        # マーキングの確認
        self.assertTrue(hasattr(square, "__ly_native__"))
        self.assertTrue(square.__ly_native__)  # type: ignore
        self.assertTrue(square.__ly_inline__)  # type: ignore

        # 実行テスト
        result = square(f64(5.0))
        self.assertIsInstance(result, f64)
        self.assertAlmostEqual(float(result), 25.0)

    def test_native_decorator_preserve_metadata(self):
        """メタデータの保持"""

        @native
        def multiply(x: i64, y: i64) -> i64:
            """2つの数値を乗算します"""
            return x * y

        # 関数名とドキュメントの保持
        self.assertEqual(multiply.__name__, "multiply")
        self.assertEqual(multiply.__doc__, "2つの数値を乗算します")

        # 型注釈の保持
        annotations = multiply.__annotations__
        self.assertIn("x", annotations)
        self.assertIn("y", annotations)
        self.assertIn("return", annotations)

    def test_native_decorator_without_parentheses(self):
        """括弧なしの@nativeデコレータ"""

        @native
        def subtract(a: i32, b: i32) -> i32:
            return a - b

        self.assertTrue(is_native_function(subtract))

        result = subtract(i32(15), i32(5))
        self.assertEqual(int(result), 10)


class TestUtilityFunctions(unittest.TestCase):
    """ユーティリティ関数のテスト"""

    def test_is_native_type(self):
        """is_native_type 関数のテスト"""
        self.assertTrue(is_native_type(i32(42)))
        self.assertTrue(is_native_type(f64(3.14)))
        self.assertTrue(is_native_type(i8(100)))

        self.assertFalse(is_native_type(42))
        self.assertFalse(is_native_type(3.14))
        self.assertFalse(is_native_type("string"))
        self.assertFalse(is_native_type([1, 2, 3]))

    def test_is_native_function(self):
        """is_native_function 関数のテスト"""

        @native
        def native_func() -> i32:
            return i32(0)

        def regular_func():
            return 0

        self.assertTrue(is_native_function(native_func))
        self.assertFalse(is_native_function(regular_func))

    def test_get_native_type_name(self):
        """get_native_type_name 関数のテスト"""
        self.assertEqual(get_native_type_name(i32(42)), "i32")
        self.assertEqual(get_native_type_name(f64(3.14)), "f64")
        self.assertEqual(get_native_type_name(i8(100)), "i8")
        self.assertEqual(get_native_type_name(f32(2.5)), "f32")


class TestCompileTimeInfo(unittest.TestCase):
    """CompileTimeInfo のテスト"""

    def test_get_llvm_type(self):
        """LLVM型文字列の取得"""
        self.assertEqual(CompileTimeInfo.get_llvm_type(i8), "i8")
        self.assertEqual(CompileTimeInfo.get_llvm_type(i16), "i16")
        self.assertEqual(CompileTimeInfo.get_llvm_type(i32), "i32")
        self.assertEqual(CompileTimeInfo.get_llvm_type(i64), "i64")
        self.assertEqual(CompileTimeInfo.get_llvm_type(f32), "float")
        self.assertEqual(CompileTimeInfo.get_llvm_type(f64), "double")

    def test_get_size_in_bits(self):
        """型のサイズ取得"""
        self.assertEqual(CompileTimeInfo.get_size_in_bits(i8), 8)
        self.assertEqual(CompileTimeInfo.get_size_in_bits(i16), 16)
        self.assertEqual(CompileTimeInfo.get_size_in_bits(i32), 32)
        self.assertEqual(CompileTimeInfo.get_size_in_bits(i64), 64)
        self.assertEqual(CompileTimeInfo.get_size_in_bits(f32), 32)
        self.assertEqual(CompileTimeInfo.get_size_in_bits(f64), 64)

    def test_type_properties(self):
        """型の性質判定"""
        # 符号付き判定
        self.assertTrue(CompileTimeInfo.is_signed(i32))
        self.assertTrue(CompileTimeInfo.is_signed(f64))

        # 整数型判定
        self.assertTrue(CompileTimeInfo.is_integer(i32))
        self.assertTrue(CompileTimeInfo.is_integer(i64))
        self.assertFalse(CompileTimeInfo.is_integer(f32))
        self.assertFalse(CompileTimeInfo.is_integer(f64))

        # 浮動小数点型判定
        self.assertFalse(CompileTimeInfo.is_float(i32))
        self.assertFalse(CompileTimeInfo.is_float(i64))
        self.assertTrue(CompileTimeInfo.is_float(f32))
        self.assertTrue(CompileTimeInfo.is_float(f64))


class TestComplexOperations(unittest.TestCase):
    """複合的な操作のテスト"""

    def test_mixed_type_operations(self):
        """異なるネイティブ型間の操作"""
        # 基本的にはLythonでは同じ型同士の演算のみサポート
        # 異なる型同士は明示的な変換が必要

        a = i32(10)
        b = i64(20)

        # 明示的変換を使った演算
        result = to_native(int(a), i64) + b  # type: ignore
        self.assertIsInstance(result, i64)
        self.assertEqual(int(result), 30)

    def test_native_function_with_conversion(self):
        """型変換を含むネイティブ関数"""

        @native
        def process_value(x: i32) -> f64:
            # ネイティブ型間の変換
            float_x = to_native(int(x), f64)  # type: ignore
            return float_x * float_x  # type: ignore

        result = process_value(i32(5))
        self.assertIsInstance(result, f64)
        self.assertAlmostEqual(float(result), 25.0)

    def test_nested_native_calls(self):
        """ネスト化されたネイティブ関数呼び出し"""

        @native
        def add_i32(a: i32, b: i32) -> i32:
            return a + b

        @native
        def multiply_i32(a: i32, b: i32) -> i32:
            return a * b

        @native
        def complex_calculation(x: i32, y: i32) -> i32:
            sum_val = add_i32(x, y)
            return multiply_i32(sum_val, i32(2))

        result = complex_calculation(i32(3), i32(4))
        self.assertIsInstance(result, i32)
        self.assertEqual(int(result), 14)  # (3 + 4) * 2 = 14


if __name__ == "__main__":
    unittest.main()
