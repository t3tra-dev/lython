"""
Lython Native API

@nativeデコレータと高性能ネイティブ型を提供します。
コンパイル時にLLVM IRに変換される型とデコレータ機能を含みます。
"""

import functools
from typing import Any, Optional, Type


class NativeType:
    """ネイティブ型の基底クラス"""

    def __init__(self, value):
        self._value = self._validate_and_convert(value)

    def _validate_and_convert(self, value):
        """値の検証と変換（サブクラスでオーバーライド）"""
        return value

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __str__(self):
        return f"{self.__class__.__name__}({self._value})"

    def __repr__(self):
        return str(self)


class i8(NativeType):
    """8ビット符号付き整数型"""

    def _validate_and_convert(self, value):
        if isinstance(value, str):
            raise TypeError("Cannot convert string to i8")
        val = int(value)
        if val < -128 or val > 127:
            raise ValueError(f"Value {val} out of range for i8 [-128, 127]")
        return val

    def __add__(self, other):
        return i8(self._value + other._value)

    def __sub__(self, other):
        return i8(self._value - other._value)

    def __mul__(self, other):
        return i8(self._value * other._value)

    def __floordiv__(self, other):
        return i8(self._value // other._value)

    def __mod__(self, other):
        return i8(self._value % other._value)

    def __eq__(self, other):
        return isinstance(other, i8) and self._value == other._value

    def __lt__(self, other):
        return self._value < other._value

    def __le__(self, other):
        return self._value <= other._value

    def __gt__(self, other):
        return self._value > other._value

    def __ge__(self, other):
        return self._value >= other._value


class i16(NativeType):
    """16ビット符号付き整数型"""

    def _validate_and_convert(self, value):
        if isinstance(value, str):
            raise TypeError("Cannot convert string to i16")
        val = int(value)
        if val < -32768 or val > 32767:
            raise ValueError(f"Value {val} out of range for i16 [-32768, 32767]")
        return val

    def __add__(self, other):
        return i16(self._value + other._value)

    def __sub__(self, other):
        return i16(self._value - other._value)

    def __mul__(self, other):
        return i16(self._value * other._value)

    def __floordiv__(self, other):
        return i16(self._value // other._value)

    def __mod__(self, other):
        return i16(self._value % other._value)

    def __eq__(self, other):
        return isinstance(other, i16) and self._value == other._value

    def __lt__(self, other):
        return self._value < other._value

    def __le__(self, other):
        return self._value <= other._value

    def __gt__(self, other):
        return self._value > other._value

    def __ge__(self, other):
        return self._value >= other._value


class i32(NativeType):
    """32ビット符号付き整数型"""

    def _validate_and_convert(self, value):
        if isinstance(value, str):
            raise TypeError("Cannot convert string to i32")
        val = int(value)
        if val < -2147483648 or val > 2147483647:
            raise ValueError(
                f"Value {val} out of range for i32 [-2147483648, 2147483647]"
            )
        return val

    def __add__(self, other):
        return i32(self._value + other._value)

    def __sub__(self, other):
        return i32(self._value - other._value)

    def __mul__(self, other):
        return i32(self._value * other._value)

    def __floordiv__(self, other):
        return i32(self._value // other._value)

    def __mod__(self, other):
        return i32(self._value % other._value)

    def __eq__(self, other):
        return isinstance(other, i32) and self._value == other._value

    def __lt__(self, other):
        return self._value < other._value

    def __le__(self, other):
        return self._value <= other._value

    def __gt__(self, other):
        return self._value > other._value

    def __ge__(self, other):
        return self._value >= other._value


class i64(NativeType):
    """64ビット符号付き整数型"""

    def _validate_and_convert(self, value):
        if isinstance(value, str):
            raise TypeError("Cannot convert string to i64")
        val = int(value)
        if val < -9223372036854775808 or val > 9223372036854775807:
            raise ValueError(f"Value {val} out of range for i64")
        return val

    def __add__(self, other):
        return i64(self._value + other._value)

    def __sub__(self, other):
        return i64(self._value - other._value)

    def __mul__(self, other):
        return i64(self._value * other._value)

    def __floordiv__(self, other):
        return i64(self._value // other._value)

    def __mod__(self, other):
        return i64(self._value % other._value)

    def __eq__(self, other):
        return isinstance(other, i64) and self._value == other._value

    def __lt__(self, other):
        return self._value < other._value

    def __le__(self, other):
        return self._value <= other._value

    def __gt__(self, other):
        return self._value > other._value

    def __ge__(self, other):
        return self._value >= other._value


class f32(NativeType):
    """32ビット浮動小数点数型"""

    def _validate_and_convert(self, value):
        if isinstance(value, str):
            raise TypeError("Cannot convert string to f32")
        return float(value)

    def __add__(self, other):
        return f32(self._value + other._value)

    def __sub__(self, other):
        return f32(self._value - other._value)

    def __mul__(self, other):
        return f32(self._value * other._value)

    def __truediv__(self, other):
        return f32(self._value / other._value)

    def __eq__(self, other):
        return isinstance(other, f32) and abs(self._value - other._value) < 1e-6

    def __lt__(self, other):
        return self._value < other._value

    def __le__(self, other):
        return self._value <= other._value

    def __gt__(self, other):
        return self._value > other._value

    def __ge__(self, other):
        return self._value >= other._value


class f64(NativeType):
    """64ビット浮動小数点数型"""

    def _validate_and_convert(self, value):
        if isinstance(value, str):
            raise TypeError("Cannot convert string to f64")
        return float(value)

    def __add__(self, other):
        return f64(self._value + other._value)

    def __sub__(self, other):
        return f64(self._value - other._value)

    def __mul__(self, other):
        return f64(self._value * other._value)

    def __truediv__(self, other):
        return f64(self._value / other._value)

    def __eq__(self, other):
        return isinstance(other, f64) and abs(self._value - other._value) < 1e-15

    def __lt__(self, other):
        return self._value < other._value

    def __le__(self, other):
        return self._value <= other._value

    def __gt__(self, other):
        return self._value > other._value

    def __ge__(self, other):
        return self._value >= other._value


def native(func=None, *, inline=False):
    """
    @nativeデコレータ

    関数をネイティブコンパイル対象としてマークします。

    Args:
        func: デコレートする関数
        inline: インライン展開を指定
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        # ネイティブ関数としてマーク
        wrapper.__ly_native__ = True
        wrapper.__ly_inline__ = inline

        return wrapper

    if func is None:
        # @native(inline=True) の形式
        return decorator
    else:
        # @native の形式
        return decorator(func)


def to_native(value: Any, target_type: Type[NativeType]) -> NativeType:
    """
    Python値をネイティブ型に変換

    Args:
        value: 変換する値
        target_type: 変換先のネイティブ型クラス

    Returns:
        ネイティブ型のインスタンス
    """
    if not issubclass(target_type, NativeType):
        raise TypeError(f"target_type must be a native type, got {target_type}")

    return target_type(value)


def to_pytype(value: NativeType, target_type: Optional[Type] = None) -> Any:
    """
    ネイティブ型をPython型に変換

    Args:
        value: 変換するネイティブ型
        target_type: 変換先のPython型（省略時は適切な型を自動選択）

    Returns:
        Python型の値
    """
    if not isinstance(value, NativeType):
        raise TypeError("value must be a native type")

    if target_type is None:
        # デフォルトの変換
        if isinstance(value, (i8, i16, i32, i64)):
            return int(value)
        elif isinstance(value, (f32, f64)):
            return float(value)
    else:
        # 明示的な変換
        if target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return str(value._value)
        else:
            raise TypeError(f"Unsupported target type: {target_type}")

    return value._value


def is_native_type(obj: Any) -> bool:
    """オブジェクトがネイティブ型かどうか判定"""
    return isinstance(obj, NativeType)


def is_native_function(func) -> bool:
    """関数が@nativeでマークされているかどうか判定"""
    return hasattr(func, "__ly_native__") and func.__ly_native__


def get_native_type_name(obj: NativeType) -> str:
    """ネイティブ型の名前を取得"""
    if not isinstance(obj, NativeType):
        raise TypeError("obj must be a native type")
    return obj.__class__.__name__


class CompileTimeInfo:
    """コンパイル時の型情報を提供するユーティリティクラス"""

    @staticmethod
    def get_llvm_type(native_type_class):
        """ネイティブ型クラスからLLVM型文字列を取得"""
        type_map = {
            i8: "i8",
            i16: "i16",
            i32: "i32",
            i64: "i64",
            f32: "float",
            f64: "double",
        }
        return type_map.get(native_type_class, "unknown")

    @staticmethod
    def get_size_in_bits(native_type_class):
        """ネイティブ型クラスのビットサイズを取得"""
        size_map = {
            i8: 8,
            i16: 16,
            i32: 32,
            i64: 64,
            f32: 32,
            f64: 64,
        }
        return size_map.get(native_type_class, 0)

    @staticmethod
    def is_signed(native_type_class):
        """符号付き型かどうか"""
        # 全ての型が符号付きとして扱う（簡略化）
        return True

    @staticmethod
    def is_integer(native_type_class):
        """整数型かどうか"""
        return native_type_class in [i8, i16, i32, i64]

    @staticmethod
    def is_float(native_type_class):
        """浮動小数点型かどうか"""
        return native_type_class in [f32, f64]


__all__ = [
    # Native型クラス
    "i8",
    "i16",
    "i32",
    "i64",
    "f32",
    "f64",
    # デコレータ
    "native",
    # 変換関数
    "to_native",
    "to_pytype",
    # ユーティリティ
    "is_native_type",
    "is_native_function",
    "get_native_type_name",
    "CompileTimeInfo",
]
