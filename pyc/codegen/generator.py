import ast

from .ir.builder import IRBuilder
from .visitors import ModVisitor, ExprVisitor
from .visitors.base import BaseVisitor

__all__ = ["IRGenerator"]


class IRGenerator(BaseVisitor):
    def __init__(self):
        self.builder = IRBuilder()
        self.module_visitor = ModVisitor(self.builder)
        self.expr_visitor = ExprVisitor(self.builder)
        self.setup_stdlib()

    def setup_stdlib(self):
        """stdlibのセットアップ"""
        # 外部関数の宣言
        self.builder.add_external_functions(
            [
                "; 外部関数の宣言",
                "declare i32 @print(ptr noundef)",
                "declare i32 @print_object(ptr noundef)",
                "declare i32 @puts(ptr nocapture readonly) local_unnamed_addr",
                "declare ptr @PyInt_FromLong(i64)",
                "declare ptr @str(ptr noundef)",
                "declare ptr @PyString_FromString(ptr noundef)",  # この行を追加
                "",
            ]
        )

        # 構造体定義
        self.builder.add_struct_definitions(
            [
                "; 構造体定義",
                "%struct.PyObject = type { ptr, i64, ptr }",
                "",
                "%struct.PyMethodTable = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }",
                "",
                "%struct.PyStringObject = type { %struct.PyObject, ptr, i64 }",
                "",
                "%struct.PyIntObject = type { %struct.PyObject, i64 }",
                "",
                "%struct.PyBaseExceptionObject = type { %struct.PyObject, ptr }",
                "",
            ]
        )

        # 定数の定義
        self.builder.add_constant("; 定数の定義")
        self.builder.add_constant("@Py_True = global i1 true, align 1")
        self.builder.add_constant("@Py_False = global i1 false, align 1")
        self.builder.add_constant("@Py_None = global ptr null, align 8")
        self.builder.add_constant("")

        self.builder.add_constant("")
        self.builder.add_constant("attributes #0 = { noinline nounwind optnone uwtable \"frame-pointer\"=\"all\" \"min-legal-vector-width\"=\"0\" \"no-trapping-math\"=\"true\" \"stack-protector-buffer-size\"=\"8\" }")

    def visit_Module(self, node: ast.Module) -> None:
        self.module_visitor.visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.expr_visitor.visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.expr_visitor.visit_Call(node)

    def generate(self, node: ast.AST) -> str:
        self.visit(node)
        return self.builder.get_output()
