from typing import Dict, List

__all__ = ["IRBuilder"]


class IRBuilder:
    def __init__(self):
        self.global_strings: Dict[str, str] = {}
        self.string_counter: int = 0
        self.output: List[str] = []

        # 外部関数の宣言を追加
        self.output.extend([
            "; 外部関数の宣言",
            "declare ptr @PyInt_FromLong(i64 noundef)",
            "declare ptr @PyString_FromString(ptr noundef)",
            "declare i32 @puts(ptr nocapture readonly) local_unnamed_addr",
            ""
        ])

        # 構造体定義を追加
        self.output.extend([
            "; 構造体定義",
            "%struct.PyObject = type { ptr, i64, ptr }",
            "",
            "%struct.PyMethodTable = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }",
            "",
            "%struct.PyStringObject = type { %struct.PyObject, ptr, i64 }",
            "",
            "%struct.PyIntObject = type { %struct.PyObject, i64 }",
            ""
        ])

    def add_global_string(self, value: str) -> str:
        value = value + "\\00"
        identifier = f"@.str.{self.string_counter}"
        length = len(value)
        escaped_value = value.replace('"', '\\"')
        self.global_strings[identifier] = (
            f'{identifier} = private unnamed_addr constant [{length - 2} x i8] c"{escaped_value}", align 1'
        )
        self.string_counter += 1
        return identifier

    def emit(self, instruction: str) -> None:
        self.output.append(instruction)
