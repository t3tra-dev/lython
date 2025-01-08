import ast
from typing import Dict, List

__all__ = ["IRGenerator"]


class IRGenerator:
    def __init__(self):
        self.global_strings: Dict[str, str] = {}
        self.string_counter: int = 0
        self.output: List[str] = []

    def add_global_string(self, value: str) -> str:
        """文字列定数をグローバル変数として追加"""
        # NULL終端文字を追加
        value = value + "\\00"
        identifier = f"@.str.{self.string_counter}"
        length = len(value)
        # エスケープ処理
        escaped_value = value.replace('"', '\\"')
        self.global_strings[identifier] = (
            f'{identifier} = private unnamed_addr constant [{length - 2} x i8] c"{escaped_value}", align 1'
        )
        self.string_counter += 1
        return identifier

    def generate(self, node: ast.AST) -> str:
        """ASTからLLVM IRを生成"""
        self.output = []
        self.visit(node)
        return "\n".join(list(self.global_strings.values()) + self.output)

    def visit(self, node: ast.AST) -> None:
        """ASTノードを訪問"""
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> None:
        """未実装のノード用のデフォルト処理"""
        raise NotImplementedError(f"Node type {type(node).__name__} not implemented")

    def visit_Module(self, node: ast.Module) -> None:
        """モジュールの処理"""
        # 外部関数の宣言
        self.output.append(
            "declare i32 @puts(i8* nocapture readonly) local_unnamed_addr"
        )

        # main関数の開始
        self.output.append("\ndefine i32 @main(i32 %argc, i8** %argv) {")
        self.output.append("entry:")

        # 本体の処理
        for stmt in node.body:
            self.visit(stmt)

        # 関数の終了
        self.output.append("  ret i32 0")
        self.output.append("}")

    def visit_Expr(self, node: ast.Expr) -> None:
        """式文の処理"""
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """関数呼び出しの処理"""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                string_val = node.args[0].value
                str_ptr = self.add_global_string(string_val)
                self.output.append(
                    f"  %puts = call i32 @puts(i8* getelementptr inbounds ([{len(string_val) + 1} x i8], [{len(string_val) + 1} x i8]* {str_ptr}, i64 0, i64 0))"
                )
            else:
                raise NotImplementedError("Only simple string printing is supported")
        else:
            raise NotImplementedError(f"Function {node.func.id} not implemented")
