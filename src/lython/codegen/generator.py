import ast

from .ir.builder import IRBuilder
from .visitors import ExprVisitor, ModVisitor
from .visitors.base import BaseVisitor

__all__ = ["IRGenerator"]


class IRGenerator(BaseVisitor):
    """
    LLVMコード生成のメインクラス。
    Python ASTを解析し、LLVM IRを生成する。

    Attributes:
        builder (IRBuilder): LLVM IRを構築するためのビルダー
        module_visitor (ModVisitor): モジュールノード用のVisitor
        expr_visitor (ExprVisitor): 式ノード用のVisitor
    """

    def __init__(self):
        self.builder = IRBuilder()
        self.module_visitor = ModVisitor(self.builder)
        self.expr_visitor = ExprVisitor(self.builder)
        self.setup_runtime()

    def setup_runtime(self):
        """
        ランタイムライブラリの外部宣言と型定義を追加する
        """
        # 型定義と構造体
        self.builder.add_struct_definitions([
            "; ========== オブジェクトシステム構造体定義 ========== ",
            "; 基本オブジェクト構造体",
            "%struct.PyObject = type { i64, ptr }",
            "; 可変長オブジェクト構造体",
            "%struct.PyVarObject = type { %struct.PyObject, i64 }",
            "; Unicode文字列オブジェクト構造体",
            "%struct.PyUnicodeObject = type { %struct.PyObject, i64, i64, i32, ptr }",
            "; リストオブジェクト構造体",
            "%struct.PyListObject = type { %struct.PyObject, i64, ptr }",
            "; 辞書エントリ構造体",
            "%struct.PyDictEntry = type { ptr, ptr, i64 }",
            "; 辞書オブジェクト構造体",
            "%struct.PyDictObject = type { %struct.PyObject, i64, i64, ptr, i64, i64 }",
            "",
        ])

        # 定数宣言
        self.builder.add_constant_definitions([
            "; ========== オブジェクトシステム定数 ========== ",
            "@Py_None = external global %struct.PyObject",
            "@Py_True = external global %struct.PyObject",
            "@Py_False = external global %struct.PyObject",
            "@PyUnicode_Type = external global %struct.PyObject",
            "@PyList_Type = external global %struct.PyObject",
            "@PyDict_Type = external global %struct.PyObject",
            "@PyBool_Type = external global %struct.PyObject",
            "",
        ])

        # 外部関数宣言
        self.builder.add_external_functions([
            "; ========== オブジェクトシステム関数宣言 ========== ",
            "; オブジェクトシステムの初期化",
            "declare void @PyObject_InitSystem()",
            "declare void @PyObject_Init()",

            "; 基本オブジェクト操作",
            "declare ptr @PyObject_New(ptr)",
            "declare ptr @PyObject_NewVar(ptr, i64)",
            "declare ptr @PyObject_Repr(ptr)",
            "declare ptr @PyObject_Str(ptr)",
            "declare i32 @PyObject_Compare(ptr, ptr)",
            "declare ptr @PyObject_RichCompare(ptr, ptr, i32)",
            "declare i32 @PyObject_RichCompareBool(ptr, ptr, i32)",
            "declare i64 @PyObject_Hash(ptr)",
            "declare i32 @PyObject_IsTrue(ptr)",
            "declare ptr @PyObject_GetAttrString(ptr, ptr)",
            "declare i32 @PyObject_SetAttrString(ptr, ptr, ptr)",
            "declare i32 @PyObject_HasAttrString(ptr, ptr)",

            "; 型チェック・変換関数",
            "declare i32 @PyType_IsSubtype(ptr, ptr)",
            "declare void @PyType_Ready(ptr)",

            "; 数値関連関数",
            "declare ptr @PyInt_FromI32(i32)",
            "declare i32 @PyInt_AsI32(ptr)",

            "; ブール値関連",
            "declare ptr @PyBool_FromLong(i32)",

            "; Unicode文字列関連",
            "declare ptr @PyUnicode_FromString(ptr)",
            "declare ptr @PyUnicode_FromStringAndSize(ptr, i64)",
            "declare ptr @PyUnicode_FromFormat(ptr, ...)",
            "declare ptr @PyUnicode_Concat(ptr, ptr)",
            "declare i64 @PyUnicode_GetLength(ptr)",
            "declare ptr @PyUnicode_AsUTF8(ptr)",
            "declare i32 @PyUnicode_Compare(ptr, ptr)",
            "declare i32 @PyUnicode_CompareWithASCIIString(ptr, ptr)",
            "declare i64 @PyUnicode_Hash(ptr)",

            "; リスト関連",
            "declare ptr @PyList_New(i64)",
            "declare i64 @PyList_Size(ptr)",
            "declare ptr @PyList_GetItem(ptr, i64)",
            "declare i32 @PyList_SetItem(ptr, i64, ptr)",
            "declare i32 @PyList_Insert(ptr, i64, ptr)",
            "declare i32 @PyList_Append(ptr, ptr)",
            "declare ptr @PyList_GetSlice(ptr, i64, i64)",
            "declare i32 @PyList_SetSlice(ptr, i64, i64, ptr)",
            "declare i32 @PyList_Sort(ptr)",
            "declare i32 @PyList_Reverse(ptr)",

            "; 辞書関連",
            "declare ptr @PyDict_New()",
            "declare ptr @PyDict_GetItem(ptr, ptr)",
            "declare i32 @PyDict_SetItem(ptr, ptr, ptr)",
            "declare i32 @PyDict_DelItem(ptr, ptr)",
            "declare void @PyDict_Clear(ptr)",
            "declare i32 @PyDict_Next(ptr, ptr, ptr, ptr)",
            "declare ptr @PyDict_Keys(ptr)",
            "declare ptr @PyDict_Values(ptr)",
            "declare ptr @PyDict_Items(ptr)",
            "declare i64 @PyDict_Size(ptr)",
            "declare ptr @PyDict_GetItemString(ptr, ptr)",
            "declare i32 @PyDict_SetItemString(ptr, ptr, ptr)",

            "; GC関連",
            "declare void @GC_init()",
            "",
            "; 比較演算子の定数",
            "; define i32 @Py_LT() { ret i32 0 }",
            "; define i32 @Py_LE() { ret i32 1 }",
            "; define i32 @Py_EQ() { ret i32 2 }",
            "; define i32 @Py_NE() { ret i32 3 }",
            "; define i32 @Py_GT() { ret i32 4 }",
            "; define i32 @Py_GE() { ret i32 5 }",
            "",

            "; 参照カウント操作",
            "declare void @Py_INCREF(ptr)",
            "declare void @Py_DECREF(ptr)",
            "declare void @Py_XINCREF(ptr)",
            "declare void @Py_XDECREF(ptr)",

            "; 数値演算関連",
            "declare ptr @PyNumber_Add(ptr, ptr)",
            "declare ptr @PyNumber_Subtract(ptr, ptr)",
            "declare ptr @PyNumber_Multiply(ptr, ptr)",
            "declare ptr @PyNumber_TrueDivide(ptr, ptr)",
            "declare ptr @PyNumber_FloorDivide(ptr, ptr)",
            "declare ptr @PyNumber_Remainder(ptr, ptr)",
            "declare ptr @PyNumber_Power(ptr, ptr, ptr)",
            "declare ptr @PyNumber_Negative(ptr)",
            "declare ptr @PyNumber_Positive(ptr)",
            "declare ptr @PyNumber_Absolute(ptr)",
            "declare ptr @PyNumber_Invert(ptr)",
            "declare ptr @PyNumber_Lshift(ptr, ptr)",
            "declare ptr @PyNumber_Rshift(ptr, ptr)",
            "declare ptr @PyNumber_And(ptr, ptr)",
            "declare ptr @PyNumber_Xor(ptr, ptr)",
            "declare ptr @PyNumber_Or(ptr, ptr)",

            "; その他のユーティリティ関数",
            "declare void @print(ptr)",
            "declare ptr @int2str(i32)",
            "declare ptr @str2str(ptr)",
            "declare ptr @create_string(ptr)",
        ])

        # 便利なマクロ定義
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
