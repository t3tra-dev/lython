from typing import Dict, List, Optional, Set

__all__ = ["IRBuilder"]


class IRBuilder:
    """
    LLVM IRコードを構築するためのビルダークラス。

    このクラスはLLVM IRコードを生成するための様々なユーティリティメソッドを提供します。
    - グローバル文字列の管理
    - 構造体定義の管理
    - 外部関数宣言の管理
    - 一時変数とラベルの生成
    - 型情報の管理

    Attributes:
        global_strings (Dict[str, str]): グローバル文字列定数の管理
        string_counter (int): 文字列定数の連番カウンター
        label_counter (int): ラベルの連番カウンター
        temp_counter (int): 一時変数の連番カウンター
        output (List[str]): 生成されたIRコードの行リスト
        external_functions (List[str]): 外部関数宣言のリスト
        struct_definitions (List[str]): 構造体定義のリスト
        constants (List[str]): 定数定義のリスト
        type_registry (Dict[str, str]): 型名とLLVM型のマッピング
        known_python_types (Set[str]): 認識されたPython型の名前セット
    """

    def __init__(self):
        self.global_strings: Dict[str, str] = {}
        self.string_counter: int = 0
        self.label_counter: int = 0
        self.temp_counter: int = 0
        self.output: List[str] = []
        self.external_functions: List[str] = []
        self.struct_definitions: List[str] = []
        self.constants: List[str] = []
        self.constant_definitions: List[str] = []

        # 型情報の管理
        self.type_registry: Dict[str, str] = {
            'int': 'i32',
            'float': 'float',
            'bool': 'i1',
            'str': 'ptr',
            'list': 'ptr',
            'dict': 'ptr',
            'None': 'ptr',
        }

        # Python型とLLVM IR型のマッピング
        self.python_type_to_llvm: Dict[str, str] = {
            'int': 'i32',
            'float': 'float',
            'bool': 'i1',
            'str': '%struct.PyUnicodeObject*',
            'list': '%struct.PyListObject*',
            'dict': '%struct.PyDictObject*',
            'None': '%struct.PyObject*',
            'object': '%struct.PyObject*',
        }

        # 認識されたPython型名
        self.known_python_types: Set[str] = {
            'int', 'float', 'bool', 'str', 'list', 'dict', 'None', 'object'
        }

        # Python型とオブジェクト生成関数のマッピング
        self.type_constructor_map: Dict[str, str] = {
            'str': '@PyUnicode_FromString',
            'list': '@PyList_New',
            'dict': '@PyDict_New',
            'bool': '@PyBool_FromLong',
        }

    def add_external_functions(self, functions: List[str]) -> None:
        """外部関数の宣言を追加"""
        self.external_functions.extend(functions)

    def add_struct_definitions(self, definitions: List[str]) -> None:
        """構造体定義を追加"""
        self.struct_definitions.extend(definitions)

    def add_constant(self, constant: str) -> None:
        """定数の定義を追加"""
        self.constants.append(constant)

    def add_constant_definitions(self, definitions: List[str]) -> None:
        """定数定義 (グローバル変数等) を追加"""
        self.constant_definitions.extend(definitions)

    def add_global_string(self, value: str) -> str:
        """文字列定数をグローバル変数として追加し、そのidentifierを返す"""
        # NULL終端文字を追加
        value = value + "\00"
        identifier = f"@.str.{self.string_counter}"

        # バイト列にエンコード
        encoded_value = value.encode("utf-8")

        # バイト列を16進数表現に変換
        escaped_value = ""
        for b in encoded_value:
            escaped_value += f"\\{format(b, '02x')}"

        # エンコードされたバイト列の長さを計算
        encoded_length = len(encoded_value)

        self.global_strings[identifier] = (
            f"{identifier} = private unnamed_addr constant [{encoded_length} x i8] "
            f'c"{escaped_value}", align 1'
        )
        self.string_counter += 1
        return identifier

    def emit(self, instruction: str) -> None:
        """LLVM IR命令を出力に追加"""
        self.output.append(instruction)

    def get_output(self) -> str:
        """生成されたLLVM IRを文字列として返す"""
        return "\n".join(
            list(self.global_strings.values())  # noqa
            + self.constant_definitions  # noqa
            + self.constants  # noqa
            + self.external_functions  # noqa
            + self.struct_definitions  # noqa
            + self.output  # noqa
        )

    def define_function(
        self,
        name: str,
        return_type: str,
        arg_types: List[str],
        arg_names: List[str],
        body: List[str],
    ) -> None:
        """関数定義を追加"""
        args = ", ".join(f"{t} %{n}" for t, n in zip(arg_types, arg_names))
        self.output.append(f"define {return_type} @{name}({args}) {{")
        self.output.extend(body)
        self.output.append("}")

    def get_label_counter(self) -> int:
        """一意なラベル番号を取得"""
        counter = self.label_counter
        self.label_counter += 1
        return counter

    def get_temp_name(self) -> str:
        """一時変数の名前を生成"""
        name = f"%t{self.temp_counter}"
        self.temp_counter += 1
        return name

    def get_python_type_llvm(self, python_type: str) -> str:
        """Python型名からLLVM型名を取得"""
        return self.python_type_to_llvm.get(python_type, 'ptr')

    def get_type_constructor(self, python_type: str) -> Optional[str]:
        """指定されたPython型のコンストラクタ関数を取得"""
        return self.type_constructor_map.get(python_type)

    def is_known_python_type(self, type_name: str) -> bool:
        """指定された型名が認識されたPython型かどうかを確認"""
        return type_name in self.known_python_types

    def register_python_type(
            self,
            python_type: str,
            llvm_type: str,
            constructor: Optional[str] = None) -> None:
        """新しいPython型を登録"""
        self.python_type_to_llvm[python_type] = llvm_type
        self.known_python_types.add(python_type)
        if constructor:
            self.type_constructor_map[python_type] = constructor
