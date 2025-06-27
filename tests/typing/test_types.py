"""
型システムのテスト
"""

import unittest

from lython.frontend.typing.types import (
    AnyType,
    FunctionType,
    GenericType,
    OptionalType,
    PrimitiveType,
    PyObjectType,
    TupleType,
    TypeKind,
    UnionType,
    UnknownType,
    get_common_supertype,
    is_subtype,
    unify_types,
)


class TestPrimitiveType(unittest.TestCase):
    """PrimitiveType のテスト"""

    def test_create_primitive_types(self):
        """プリミティブ型の作成"""
        i32 = PrimitiveType.i32()
        self.assertEqual(i32.name, "i32")
        self.assertEqual(i32.size, 32)
        self.assertTrue(i32.signed)
        self.assertEqual(str(i32), "i32")
        self.assertEqual(i32.get_llvm_type(), "i32")
        self.assertEqual(i32.kind, TypeKind.PRIMITIVE)

        f64 = PrimitiveType.f64()
        self.assertEqual(f64.name, "f64")
        self.assertEqual(f64.size, 64)
        self.assertEqual(f64.get_llvm_type(), "double")

        bool_type = PrimitiveType.bool()
        self.assertEqual(bool_type.name, "bool")
        self.assertEqual(bool_type.size, 1)
        self.assertEqual(bool_type.get_llvm_type(), "i1")

    def test_primitive_type_equality(self):
        """プリミティブ型の等価性"""
        i32_1 = PrimitiveType.i32()
        i32_2 = PrimitiveType.i32()
        i64 = PrimitiveType.i64()

        self.assertEqual(i32_1, i32_2)
        self.assertNotEqual(i32_1, i64)

    def test_numeric_type_checks(self):
        """数値型の判定"""
        i32 = PrimitiveType.i32()
        f64 = PrimitiveType.f64()
        bool_type = PrimitiveType.bool()

        self.assertTrue(i32.is_numeric())
        self.assertTrue(i32.is_integer())
        self.assertFalse(i32.is_float())

        self.assertTrue(f64.is_numeric())
        self.assertFalse(f64.is_integer())
        self.assertTrue(f64.is_float())

        self.assertFalse(bool_type.is_numeric())

    def test_type_conversion(self):
        """型変換の可能性"""
        i32 = PrimitiveType.i32()
        i64 = PrimitiveType.i64()
        f32 = PrimitiveType.f32()
        f64 = PrimitiveType.f64()

        # 小さいサイズから大きいサイズへの変換
        self.assertTrue(i32.can_convert_to(i64))
        self.assertFalse(i64.can_convert_to(i32))

        # 整数から浮動小数点への変換
        self.assertTrue(i32.can_convert_to(f32))
        self.assertTrue(i32.can_convert_to(f64))

        # 浮動小数点間の変換
        self.assertTrue(f32.can_convert_to(f64))
        self.assertFalse(f64.can_convert_to(f32))

    def test_assignability(self):
        """代入可能性のテスト"""
        i32 = PrimitiveType.i32()
        i64 = PrimitiveType.i64()
        any_type = AnyType()

        self.assertTrue(i32.is_assignable_to(i32))  # 同じ型
        self.assertTrue(i32.is_assignable_to(i64))  # 暗黙的変換可能
        self.assertTrue(i32.is_assignable_to(any_type))  # Any型
        self.assertFalse(i64.is_assignable_to(i32))  # 縮小変換は不可


class TestPyObjectType(unittest.TestCase):
    """PyObjectType のテスト"""

    def test_create_pyobject_types(self):
        """PyObject型の作成"""
        int_type = PyObjectType.int()
        self.assertEqual(int_type.name, "int")
        self.assertEqual(str(int_type), "int")
        self.assertEqual(int_type.get_llvm_type(), "%pyobj*")
        self.assertEqual(int_type.kind, TypeKind.PYOBJECT)

        str_type = PyObjectType.str()
        self.assertEqual(str_type.name, "str")

        none_type = PyObjectType.none()
        self.assertEqual(none_type.name, "NoneType")

    def test_pyobject_equality(self):
        """PyObject型の等価性"""
        int1 = PyObjectType.int()
        int2 = PyObjectType.int()
        str_type = PyObjectType.str()

        self.assertEqual(int1, int2)
        self.assertNotEqual(int1, str_type)

    def test_subtype_relationships(self):
        """サブタイプ関係のテスト"""
        int_type = PyObjectType.int()
        bool_type = PyObjectType.bool()
        object_type = PyObjectType.object()

        # bool は int のサブタイプ
        self.assertTrue(bool_type.is_subtype_of(int_type))
        self.assertTrue(bool_type.is_subtype_of(object_type))

        # int は object のサブタイプ
        self.assertTrue(int_type.is_subtype_of(object_type))

        # 逆方向は成り立たない
        self.assertFalse(int_type.is_subtype_of(bool_type))
        self.assertFalse(object_type.is_subtype_of(int_type))


class TestGenericType(unittest.TestCase):
    """GenericType のテスト"""

    def test_create_generic_type(self):
        """ジェネリック型の作成"""
        list_base = PyObjectType.list()
        int_type = PyObjectType.int()

        list_int = GenericType(list_base, [int_type])
        self.assertEqual(str(list_int), "list[int]")
        self.assertEqual(list_int.kind, TypeKind.GENERIC)
        self.assertEqual(list_int.get_llvm_type(), "%pyobj*")

        # 複数の型引数
        dict_base = PyObjectType.dict()
        str_type = PyObjectType.str()
        dict_str_int = GenericType(dict_base, [str_type, int_type])
        self.assertEqual(str(dict_str_int), "dict[str, int]")

    def test_generic_equality(self):
        """ジェネリック型の等価性"""
        list_base = PyObjectType.list()
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()

        list_int1 = GenericType(list_base, [int_type])
        list_int2 = GenericType(list_base, [int_type])
        list_str = GenericType(list_base, [str_type])

        self.assertEqual(list_int1, list_int2)
        self.assertNotEqual(list_int1, list_str)


class TestUnionType(unittest.TestCase):
    """UnionType のテスト"""

    def test_create_union_type(self):
        """Union型の作成"""
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()

        union = UnionType([int_type, str_type])
        self.assertIn("Union[", str(union))
        self.assertIn("int", str(union))
        self.assertIn("str", str(union))
        self.assertEqual(union.kind, TypeKind.UNION)
        self.assertEqual(union.get_llvm_type(), "%pyobj*")

        # 単一要素のUnionも現在は UnionType として保持される
        single_union = UnionType([int_type])
        self.assertEqual(len(single_union.types), 1)
        self.assertIn(int_type, single_union.types)

    def test_union_contains_type(self):
        """Union型での型の包含チェック"""
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()
        float_type = PyObjectType.float()

        union = UnionType([int_type, str_type])

        self.assertTrue(union.contains_type(int_type))
        self.assertTrue(union.contains_type(str_type))
        self.assertFalse(union.contains_type(float_type))

    def test_union_add_type(self):
        """Union型への型の追加"""
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()
        float_type = PyObjectType.float()

        union = UnionType([int_type, str_type])
        new_union = union.add_type(float_type)

        self.assertTrue(new_union.contains_type(float_type))
        self.assertEqual(len(new_union.types), 3)


class TestOptionalType(unittest.TestCase):
    """OptionalType のテスト"""

    def test_create_optional_type(self):
        """Optional型の作成"""
        int_type = PyObjectType.int()
        optional_int = OptionalType(int_type)

        self.assertEqual(str(optional_int), "Optional[int]")
        self.assertEqual(optional_int.kind, TypeKind.OPTIONAL)
        self.assertEqual(optional_int.inner_type, int_type)

        # Optional[T] = Union[T, None] なので2つの型を持つ
        self.assertEqual(len(optional_int.types), 2)
        self.assertTrue(optional_int.contains_type(int_type))
        self.assertTrue(optional_int.contains_type(PyObjectType.none()))


class TestTupleType(unittest.TestCase):
    """TupleType のテスト"""

    def test_create_tuple_type(self):
        """タプル型の作成"""
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()

        tuple_type = TupleType([int_type, str_type])
        self.assertEqual(str(tuple_type), "Tuple[int, str]")
        self.assertEqual(tuple_type.kind, TypeKind.TUPLE)
        self.assertEqual(len(tuple_type.element_types), 2)

        # 空のタプル
        empty_tuple = TupleType([])
        self.assertEqual(str(empty_tuple), "Tuple[()]")


class TestFunctionType(unittest.TestCase):
    """FunctionType のテスト"""

    def test_create_function_type(self):
        """関数型の作成"""
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()

        func_type = FunctionType([int_type, str_type], int_type)
        self.assertEqual(str(func_type), "(int, str) -> int")
        self.assertEqual(func_type.kind, TypeKind.FUNCTION)
        self.assertFalse(func_type.is_native_func)

        # ネイティブ関数型
        native_func = FunctionType([int_type], int_type, is_native_func=True)
        self.assertEqual(str(native_func), "@native (int) -> int")
        self.assertTrue(native_func.is_native_func)

    def test_function_type_llvm(self):
        """関数型のLLVM型表現"""
        i32 = PrimitiveType.i32()
        i64 = PrimitiveType.i64()

        # ネイティブ関数
        native_func = FunctionType([i32, i32], i64, is_native_func=True)
        llvm_type = native_func.get_llvm_type()
        self.assertIn("i64", llvm_type)
        self.assertIn("i32", llvm_type)

        # Python関数
        int_type = PyObjectType.int()
        py_func = FunctionType([int_type], int_type, is_native_func=False)
        self.assertEqual(py_func.get_llvm_type(), "%pyobj*")


class TestAnyType(unittest.TestCase):
    """AnyType のテスト"""

    def test_any_type(self):
        """Any型の基本動作"""
        any_type = AnyType()
        int_type = PyObjectType.int()

        self.assertEqual(str(any_type), "Any")
        self.assertEqual(any_type.kind, TypeKind.ANY)
        self.assertTrue(any_type.is_any())

        # Any型は任意の型に代入可能
        self.assertTrue(any_type.is_assignable_to(int_type))
        self.assertTrue(any_type.is_assignable_to(any_type))


class TestUnknownType(unittest.TestCase):
    """UnknownType のテスト"""

    def test_unknown_type(self):
        """Unknown型の基本動作"""
        unknown = UnknownType()
        any_type = AnyType()
        int_type = PyObjectType.int()

        self.assertEqual(str(unknown), "Unknown")
        self.assertEqual(unknown.kind, TypeKind.UNKNOWN)
        self.assertTrue(unknown.is_unknown())

        # Unknown型はAny型とUnknown型にのみ代入可能
        self.assertTrue(unknown.is_assignable_to(any_type))
        self.assertTrue(unknown.is_assignable_to(unknown))
        self.assertFalse(unknown.is_assignable_to(int_type))


class TestTypeUtilities(unittest.TestCase):
    """型ユーティリティ関数のテスト"""

    def test_unify_types(self):
        """型の統合"""
        int_type = PyObjectType.int()
        str_type = PyObjectType.str()
        any_type = AnyType()
        unknown = UnknownType()

        # 同じ型の統合
        self.assertEqual(unify_types(int_type, int_type), int_type)

        # Any型との統合
        self.assertEqual(unify_types(any_type, int_type), int_type)
        self.assertEqual(unify_types(int_type, any_type), int_type)

        # Unknown型との統合
        self.assertEqual(unify_types(unknown, int_type), int_type)
        self.assertEqual(unify_types(int_type, unknown), int_type)

        # 異なる型の統合（Union型になる）
        unified = unify_types(int_type, str_type)
        self.assertIsInstance(unified, UnionType)

    def test_is_subtype(self):
        """サブタイプ関係の判定"""
        int_type = PyObjectType.int()
        bool_type = PyObjectType.bool()
        object_type = PyObjectType.object()

        self.assertTrue(is_subtype(bool_type, int_type))
        self.assertTrue(is_subtype(int_type, object_type))
        self.assertFalse(is_subtype(int_type, bool_type))

    def test_get_common_supertype(self):
        """共通スーパータイプの取得"""
        int_type = PyObjectType.int()
        bool_type = PyObjectType.bool()
        str_type = PyObjectType.str()

        # 空のリスト
        self.assertIsInstance(get_common_supertype([]), UnknownType)

        # 単一の型
        self.assertEqual(get_common_supertype([int_type]), int_type)

        # 関連のある型
        common = get_common_supertype([int_type, bool_type])
        self.assertEqual(common, int_type)  # bool は int のサブタイプなので int

        # 関連のない型（Union型になる）
        common_unrelated = get_common_supertype([int_type, str_type])
        self.assertIsInstance(common_unrelated, UnionType)


if __name__ == "__main__":
    unittest.main()
