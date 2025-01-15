from typing import Dict, List

__all__ = ["IRBuilder"]


class IRBuilder:
    def __init__(self):
        self.global_strings: Dict[str, str] = {}
        self.string_counter: int = 0
        self.output: List[str] = []
        self.external_functions: List[str] = []
        self.struct_definitions: List[str] = []
        self.constants: List[str] = []

    def add_external_functions(self, functions: List[str]) -> None:
        """外部関数の宣言を追加"""
        self.external_functions.extend(functions)

    def add_struct_definitions(self, definitions: List[str]) -> None:
        """構造体定義を追加"""
        self.struct_definitions.extend(definitions)

    def add_constant(self, constant: str) -> None:
        """定数の定義を追加"""
        self.constants.append(constant)

    def add_global_string(self, value: str) -> str:
        """文字列定数をグローバル変数として追加"""
        # NULL終端文字を追加
        value = value + "\00"
        identifier = f"@.str.{self.string_counter}"

        # バイト列にエンコード
        encoded_value = value.encode('utf-8')

        # バイト列を16進数表現に変換
        escaped_value = ""
        for b in encoded_value:
            escaped_value += f"\\{format(b, '02x')}"

        # エンコードされたバイト列の長さを計算
        encoded_length = len(encoded_value)

        self.global_strings[identifier] = (
            f'{identifier} = private unnamed_addr constant [{encoded_length} x i8] '
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
            list(self.global_strings.values()) +  # noqa
            self.constants +  # noqa
            self.external_functions +  # noqa
            self.struct_definitions +  # noqa
            self.output
        )

    def define_function(self, name: str, return_type: str, arg_types: List[str], arg_names: List[str], body: List[str]) -> None:
        """関数定義を追加"""
        args = ", ".join(f"{t} %{n}" for t, n in zip(arg_types, arg_names))
        self.output.append(f"define {return_type} @{name}({args}) {{")
        self.output.extend(body)
        self.output.append("}")
