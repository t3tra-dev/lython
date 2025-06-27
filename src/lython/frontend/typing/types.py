"""
型定義システム

Lython の型システムを定義します。
Python の型システムを拡張し、ネイティブ型とPyObject型の両方をサポートします。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Set
from typing import Union as PyUnion


class TypeKind(Enum):
    """型の種類を表す列挙型"""

    PRIMITIVE = "primitive"  # i32, f64 など
    PYOBJECT = "pyobject"  # Python オブジェクト型
    GENERIC = "generic"  # List[T], Dict[K, V] など
    UNION = "union"  # Union[int, str] など
    ANY = "any"  # Any 型
    UNKNOWN = "unknown"  # 未知の型
    FUNCTION = "function"  # 関数型
    TUPLE = "tuple"  # タプル型
    OPTIONAL = "optional"  # Optional[T] = Union[T, None]


@dataclass
class Type(ABC):
    """全ての型の基底クラス"""

    @abstractmethod
    def __str__(self) -> str:
        """型の文字列表現"""
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """型の等価性判定"""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """ハッシュ値の計算（setで使用するため）"""
        pass

    @abstractmethod
    def is_assignable_to(self, other: Type) -> bool:
        """この型が other 型に代入可能かどうか"""
        pass

    @abstractmethod
    def get_llvm_type(self) -> str:
        """LLVM IR での型表現を取得"""
        pass

    @property
    @abstractmethod
    def kind(self) -> TypeKind:
        """型の種類を取得"""
        pass

    def is_native(self) -> bool:
        """ネイティブ型かどうか"""
        return self.kind == TypeKind.PRIMITIVE

    def is_pyobject(self) -> bool:
        """PyObject型かどうか"""
        return self.kind == TypeKind.PYOBJECT

    def is_any(self) -> bool:
        """Any型かどうか"""
        return self.kind == TypeKind.ANY

    def is_unknown(self) -> bool:
        """未知の型かどうか"""
        return self.kind == TypeKind.UNKNOWN


@dataclass
class PrimitiveType(Type):
    """プリミティブ型 (i32, i64, f32, f64, bool など)"""

    name: str
    size: int  # ビット数
    signed: bool = True

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PrimitiveType)
            and self.name == other.name
            and self.size == other.size
            and self.signed == other.signed
        )

    def __hash__(self) -> int:
        return hash((self.name, self.size, self.signed))

    def is_assignable_to(self, other: Type) -> bool:
        if isinstance(other, PrimitiveType):
            # 同じ型なら代入可能
            if self == other:
                return True
            # 数値型の暗黙的変換ルール
            if self.is_numeric() and other.is_numeric():
                return self.can_convert_to(other)
        elif isinstance(other, AnyType):
            return True
        elif isinstance(other, UnionType):
            return any(self.is_assignable_to(t) for t in other.types)
        return False

    def get_llvm_type(self) -> str:
        """LLVM IR での型表現"""
        if self.name == "bool":
            return "i1"
        elif self.name.startswith("i"):
            return f"i{self.size}"
        elif self.name.startswith("f"):
            if self.size == 32:
                return "float"
            elif self.size == 64:
                return "double"
        return f"i{self.size}"  # デフォルト

    @property
    def kind(self) -> TypeKind:
        return TypeKind.PRIMITIVE

    def is_numeric(self) -> bool:
        """数値型かどうか"""
        return self.name in ["i8", "i16", "i32", "i64", "f32", "f64"]

    def is_integer(self) -> bool:
        """整数型かどうか"""
        return self.name.startswith("i") and self.name != "i1"  # boolは除く

    def is_float(self) -> bool:
        """浮動小数点型かどうか"""
        return self.name.startswith("f")

    def can_convert_to(self, other: PrimitiveType) -> bool:
        """他のプリミティブ型に変換可能かどうか"""
        if not other.is_numeric():
            return False

        # 整数 -> 浮動小数点は可能
        if self.is_integer() and other.is_float():
            return True

        # 小さいサイズ -> 大きいサイズは可能
        if self.is_integer() and other.is_integer():
            return self.size <= other.size

        if self.is_float() and other.is_float():
            return self.size <= other.size

        return False

    # 標準的なプリミティブ型のファクトリメソッド
    @classmethod
    def i8(cls) -> PrimitiveType:
        return cls("i8", 8, True)

    @classmethod
    def i16(cls) -> PrimitiveType:
        return cls("i16", 16, True)

    @classmethod
    def i32(cls) -> PrimitiveType:
        return cls("i32", 32, True)

    @classmethod
    def i64(cls) -> PrimitiveType:
        return cls("i64", 64, True)

    @classmethod
    def f32(cls) -> PrimitiveType:
        return cls("f32", 32, True)

    @classmethod
    def f64(cls) -> PrimitiveType:
        return cls("f64", 64, True)

    @classmethod
    def bool(cls) -> PrimitiveType:
        return cls("bool", 1, False)


@dataclass
class PyObjectType(Type):
    """Python オブジェクト型"""

    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PyObjectType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def is_assignable_to(self, other: Type) -> bool:
        if isinstance(other, PyObjectType):
            # 同じ型または親クラスに代入可能
            return self.name == other.name or self.is_subtype_of(other)
        elif isinstance(other, AnyType):
            return True
        elif isinstance(other, UnionType):
            return any(self.is_assignable_to(t) for t in other.types)
        return False

    def get_llvm_type(self) -> str:
        return "%pyobj*"

    @property
    def kind(self) -> TypeKind:
        return TypeKind.PYOBJECT

    def is_subtype_of(self, other: PyObjectType) -> bool:
        """他のPyObject型のサブタイプかどうか"""
        # 簡単な継承関係の例
        subtype_relations = {
            "int": ["object"],
            "float": ["object"],
            "str": ["object"],
            "list": ["object"],
            "dict": ["object"],
            "bool": ["int", "object"],
        }

        if self.name in subtype_relations:
            return other.name in subtype_relations[self.name]
        return False

    # 標準的なPyObject型のファクトリメソッド
    @classmethod
    def int(cls) -> PyObjectType:
        return cls("int")

    @classmethod
    def float(cls) -> PyObjectType:
        return cls("float")

    @classmethod
    def str(cls) -> PyObjectType:
        return cls("str")

    @classmethod
    def list(cls) -> PyObjectType:
        return cls("list")

    @classmethod
    def dict(cls) -> PyObjectType:
        return cls("dict")

    @classmethod
    def bool(cls) -> PyObjectType:
        return cls("bool")

    @classmethod
    def object(cls) -> PyObjectType:
        return cls("object")

    @classmethod
    def none(cls) -> PyObjectType:
        return cls("NoneType")


@dataclass
class GenericType(Type):
    """ジェネリック型 (List[T], Dict[K, V] など)"""

    base: Type
    type_args: List[Type]

    def __str__(self) -> str:
        if len(self.type_args) == 1:
            return f"{self.base}[{self.type_args[0]}]"
        return f"{self.base}[{', '.join(str(arg) for arg in self.type_args)}]"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, GenericType)
            and self.base == other.base
            and self.type_args == other.type_args
        )

    def __hash__(self) -> int:
        return hash((self.base, tuple(self.type_args)))

    def is_assignable_to(self, other: Type) -> bool:
        if isinstance(other, GenericType):
            # 共変性・反変性は簡単化
            return self.base.is_assignable_to(other.base) and all(
                arg.is_assignable_to(other_arg)
                for arg, other_arg in zip(self.type_args, other.type_args)
            )
        elif isinstance(other, AnyType):
            return True
        elif isinstance(other, UnionType):
            return any(self.is_assignable_to(t) for t in other.types)
        return False

    def get_llvm_type(self) -> str:
        # ジェネリック型は基本的にPyObject*として扱う
        return "%pyobj*"

    @property
    def kind(self) -> TypeKind:
        return TypeKind.GENERIC


@dataclass
class UnionType(Type):
    """Union型 (Union[int, str] など)"""

    types: Set[Type]

    def __init__(self, types: PyUnion[List[Type], Set[Type]]):
        if isinstance(types, list):
            self.types = set(types)
        else:
            self.types = types

        # Union[T] は T に簡約 (簡略化のため実装を保留)

    def __str__(self) -> str:
        return f"Union[{', '.join(str(t) for t in sorted(self.types, key=str))}]"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UnionType) and self.types == other.types

    def __hash__(self) -> int:
        return hash(frozenset(self.types))

    def is_assignable_to(self, other: Type) -> bool:
        if isinstance(other, UnionType):
            # すべての型が other のいずれかに代入可能
            return all(
                any(t.is_assignable_to(ot) for ot in other.types) for t in self.types
            )
        elif isinstance(other, AnyType):
            return True
        else:
            # すべての型が other に代入可能
            return all(t.is_assignable_to(other) for t in self.types)

    def get_llvm_type(self) -> str:
        # Union型は動的にタグ付きで表現
        return "%pyobj*"

    @property
    def kind(self) -> TypeKind:
        return TypeKind.UNION

    def add_type(self, ty: Type) -> UnionType:
        """型を追加した新しいUnion型を作成"""
        new_types = self.types.copy()
        new_types.add(ty)
        return UnionType(new_types)

    def contains_type(self, ty: Type) -> bool:
        """指定された型が含まれているかどうか"""
        return ty in self.types or any(t.is_assignable_to(ty) for t in self.types)


@dataclass
class OptionalType(UnionType):
    """Optional[T] = Union[T, None]"""

    inner_type: Type

    def __init__(self, inner_type: Type):
        self.inner_type = inner_type
        super().__init__([inner_type, PyObjectType.none()])

    def __str__(self) -> str:
        return f"Optional[{self.inner_type}]"

    @property
    def kind(self) -> TypeKind:
        return TypeKind.OPTIONAL


@dataclass
class TupleType(Type):
    """タプル型"""

    element_types: List[Type]

    def __str__(self) -> str:
        if len(self.element_types) == 0:
            return "Tuple[()]"
        return f"Tuple[{', '.join(str(t) for t in self.element_types)}]"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, TupleType) and self.element_types == other.element_types
        )

    def __hash__(self) -> int:
        return hash(tuple(self.element_types))

    def is_assignable_to(self, other: Type) -> bool:
        if isinstance(other, TupleType):
            return len(self.element_types) == len(other.element_types) and all(
                t1.is_assignable_to(t2)
                for t1, t2 in zip(self.element_types, other.element_types)
            )
        elif isinstance(other, AnyType):
            return True
        elif isinstance(other, UnionType):
            return any(self.is_assignable_to(t) for t in other.types)
        return False

    def get_llvm_type(self) -> str:
        # タプルは構造体として表現するか、PyObject*として扱う
        return "%pyobj*"

    @property
    def kind(self) -> TypeKind:
        return TypeKind.TUPLE


@dataclass
class FunctionType(Type):
    """関数型"""

    param_types: List[Type]
    return_type: Type
    is_native_func: bool = False

    def __str__(self) -> str:
        params = ", ".join(str(t) for t in self.param_types)
        modifier = "@native " if self.is_native_func else ""
        return f"{modifier}({params}) -> {self.return_type}"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, FunctionType)
            and self.param_types == other.param_types
            and self.return_type == other.return_type
            and self.is_native_func == other.is_native_func
        )

    def __hash__(self) -> int:
        return hash((tuple(self.param_types), self.return_type, self.is_native_func))

    def is_assignable_to(self, other: Type) -> bool:
        if isinstance(other, FunctionType):
            # 反変性: パラメータ型は逆向き、戻り値型は同じ向き
            return all(
                ot.is_assignable_to(st)
                for st, ot in zip(self.param_types, other.param_types)
            ) and self.return_type.is_assignable_to(other.return_type)
        elif isinstance(other, AnyType):
            return True
        elif isinstance(other, UnionType):
            return any(self.is_assignable_to(t) for t in other.types)
        return False

    def get_llvm_type(self) -> str:
        if self.is_native_func:
            # ネイティブ関数は直接的なLLVM関数型
            param_types = [t.get_llvm_type() for t in self.param_types]
            return f"{self.return_type.get_llvm_type()} ({', '.join(param_types)})*"
        else:
            # Python関数はPyObject*
            return "%pyobj*"

    @property
    def kind(self) -> TypeKind:
        return TypeKind.FUNCTION


@dataclass
class AnyType(Type):
    """Any型 - 任意の型を表す"""

    def __str__(self) -> str:
        return "Any"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AnyType)

    def __hash__(self) -> int:
        return hash("Any")

    def is_assignable_to(self, other: Type) -> bool:
        # Any型は任意の型に代入可能
        return True

    def get_llvm_type(self) -> str:
        return "%pyobj*"

    @property
    def kind(self) -> TypeKind:
        return TypeKind.ANY


@dataclass
class UnknownType(Type):
    """未知の型 - 型推論中の中間状態"""

    def __str__(self) -> str:
        return "Unknown"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UnknownType)

    def __hash__(self) -> int:
        return hash("Unknown")

    def is_assignable_to(self, other: Type) -> bool:
        # 未知の型は型推論完了まで判定不可
        return isinstance(other, (UnknownType, AnyType))

    def get_llvm_type(self) -> str:
        return "%pyobj*"  # デフォルトでPyObject*

    @property
    def kind(self) -> TypeKind:
        return TypeKind.UNKNOWN


# 型ユーティリティ関数


def unify_types(t1: Type, t2: Type) -> Optional[Type]:
    """2つの型を統合する"""
    if t1 == t2:
        return t1

    if isinstance(t1, AnyType):
        return t2
    if isinstance(t2, AnyType):
        return t1

    if isinstance(t1, UnknownType):
        return t2
    if isinstance(t2, UnknownType):
        return t1

    # プリミティブ型の統合
    if isinstance(t1, PrimitiveType) and isinstance(t2, PrimitiveType):
        if t1.can_convert_to(t2):
            return t2
        elif t2.can_convert_to(t1):
            return t1

    # PyObject型のサブタイプ関係を考慮
    if isinstance(t1, PyObjectType) and isinstance(t2, PyObjectType):
        if t1.is_subtype_of(t2):
            return t2
        elif t2.is_subtype_of(t1):
            return t1

    # Union型の作成
    return UnionType([t1, t2])


def is_subtype(subtype: Type, supertype: Type) -> bool:
    """サブタイプ関係の判定"""
    return subtype.is_assignable_to(supertype)


def get_common_supertype(types: List[Type]) -> Type:
    """複数の型の共通スーパータイプを取得"""
    if not types:
        return UnknownType()

    if len(types) == 1:
        return types[0]

    result = types[0]
    for ty in types[1:]:
        unified = unify_types(result, ty)
        if unified is None:
            return UnionType(types)
        result = unified

    return result
