import ast

from .ir.builder import IRBuilder
from .visitors import ExprVisitor, ModVisitor
from .visitors.base import BaseVisitor

__all__ = ["IRGenerator"]


class IRGenerator(BaseVisitor):
    def __init__(self):
        self.builder = IRBuilder()
        self.module_visitor = ModVisitor(self.builder)
        self.expr_visitor = ExprVisitor(self.builder)
        self.setup_runtime()

    def setup_runtime(self):
        """
        静的型ランタイム用の外部宣言を追加する
        malloc, free, printf, print_i32, print_string, create_string など
        """
        # とりあえず最低限の宣言を追加
        self.builder.add_external_functions([
            "; ========== External runtime declarations ========== ",
            "declare ptr @malloc(i64)",    # メモリ管理用
            "declare void @free(ptr)",
            # print (String*) -> void
            "declare void @print(ptr)",
            "",
            "declare ptr @int2str(i32)",   # int => string
            "declare ptr @str2str(ptr)",  # string => string
            "",
            # 文字列生成: String* create_string(i8*)
            "declare ptr @create_string(ptr)",
            "",
        ])

        # 必要なら String 構造体の定義をIR上で書く (i64 length, i8* data)
        self.builder.add_struct_definitions([
            "%struct.String = type { i64, ptr } ; // length + data pointer",
            "",
        ])

        # attributesとか
        self.builder.add_constant('attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" }')

    def visit_Module(self, node: ast.Module) -> None:
        self.module_visitor.visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.expr_visitor.visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.expr_visitor.visit_Call(node)

    def generate(self, node: ast.AST) -> str:
        self.visit(node)
        return self.builder.get_output()
