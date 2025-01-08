from typing import Dict, List

__all__ = ["IRBuilder"]


class IRBuilder:
    def __init__(self):
        self.global_strings: Dict[str, str] = {}
        self.string_counter: int = 0
        self.output: List[str] = []

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
