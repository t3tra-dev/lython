from __future__ import annotations

import ast

from ..ir import IRBuilder
from .base import BaseVisitor

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.counter = 0

        # 演算子とメソッド名のマッピング
        self.op_method_map = {
            ast.Add: "__add__",
            ast.Sub: "__sub__",
            ast.Mult: "__mul__",
            ast.Div: "__div__",
        }

    def visit_Constant(self, node: ast.Constant) -> str:
        """定数の処理"""
        result = f"%{self.counter}"
        self.counter += 1

        if isinstance(node.value, int):
            # PyInt_FromLongの呼び出し
            self.builder.emit(f"  {result} = call ptr @PyInt_FromLong(i64 noundef {node.value})")
        elif isinstance(node.value, str):
            # 文字列定数の生成
            str_const = self.builder.add_global_string(node.value)
            # PyString_FromStringの呼び出し
            self.builder.emit(f"  {result} = call ptr @PyString_FromString(ptr noundef {str_const})")
        else:
            raise NotImplementedError(f"Constant type {type(node.value)} not supported")

        return result

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """二項演算の処理"""
        # オペランドの処理
        left = self.visit(node.left)
        right = self.visit(node.right)

        # 対応するメソッド名を取得
        method_name = self.op_method_map.get(type(node.op))  # type: ignore
        if method_name is None:
            raise NotImplementedError(f"Operator {type(node.op).__name__} not supported")

        # メソッドテーブルからメソッドポインタを取得
        method_ptr = f"%{self.counter}"
        self.counter += 1
        self.builder.emit(f"""  ; 左オペランドのメソッドテーブルを取得
    %mt.{self.counter} = getelementptr inbounds %struct.PyObject, ptr {left}, i32 0, i32 2
    %mt_ptr.{self.counter} = load ptr, ptr %mt.{self.counter}
    ; {method_name}メソッドのポインタを取得
    %method.{self.counter} = getelementptr inbounds %struct.PyMethodTable, ptr %mt_ptr.{self.counter}, i32 0, i32 {self.get_method_offset(method_name)}
    {method_ptr} = load ptr, ptr %method.{self.counter}""")

        # メソッド呼び出し
        result = f"%{self.counter}"
        self.counter += 1
        self.builder.emit(f"  {result} = call ptr {method_ptr}(ptr noundef {left}, ptr noundef {right})")

        # 結果がNULLの場合は__r{method_name}__を試す
        fallback_label = f"fallback.{self.counter}"
        end_label = f"end.{self.counter}"
        self.counter += 1

        self.builder.emit(f"""  %isnull.{self.counter} = icmp eq ptr {result}, null
    br i1 %isnull.{self.counter}, label %{fallback_label}, label %{end_label}

    {fallback_label}:
    ; 右オペランドの__r{method_name[2:]}を呼び出す
    %rmt.{self.counter} = getelementptr inbounds %struct.PyObject, ptr {right}, i32 0, i32 2
    %rmt_ptr.{self.counter} = load ptr, ptr %rmt.{self.counter}
    %rmethod.{self.counter} = getelementptr inbounds %struct.PyMethodTable, ptr %rmt_ptr.{self.counter}, i32 0, i32 {self.get_method_offset('__r' + method_name[2:])}
    %rmethod_ptr.{self.counter} = load ptr, ptr %rmethod.{self.counter}
    %rresult.{self.counter} = call ptr %rmethod_ptr.{self.counter}(ptr noundef {right}, ptr noundef {left})
    br label %{end_label}

    {end_label}:
    %final.{self.counter} = phi ptr [ {result}, %entry ], [ %rresult.{self.counter}, %{fallback_label} ]""")

        return f"%final.{self.counter}"

    def visit_Call(self, node: ast.Call) -> None:
        """関数呼び出しの処理"""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            if len(node.args) == 1:
                # 引数の評価
                arg = self.visit(node.args[0])

                # 引数の__str__メソッドを呼び出す
                str_result = f"%str.{self.counter}"
                self.counter += 1
                self.builder.emit(f"""  ; __str__メソッドの取得と呼び出し
  %mt.{self.counter} = getelementptr inbounds %struct.PyObject, ptr {arg}, i32 0, i32 2
  %mt_ptr.{self.counter} = load ptr, ptr %mt.{self.counter}
  %strmethod.{self.counter} = getelementptr inbounds %struct.PyMethodTable, ptr %mt_ptr.{self.counter}, i32 0, i32 {self.get_method_offset('__str__')}
  %strmethod_ptr.{self.counter} = load ptr, ptr %strmethod.{self.counter}
  {str_result} = call ptr %strmethod_ptr.{self.counter}(ptr noundef {arg})""")

                # 文字列オブジェクトから実際の文字列を取得
                str_ptr = f"%strptr.{self.counter}"
                self.counter += 1
                self.builder.emit(f"""  {str_ptr} = getelementptr inbounds %struct.PyStringObject, ptr {str_result}, i32 0, i32 1
  %str.{self.counter} = load ptr, ptr {str_ptr}""")

                # putsの呼び出し
                self.builder.emit(f"  %puts.{self.counter} = call i32 @puts(ptr noundef %str.{self.counter})")
            else:
                raise NotImplementedError("Only single argument printing is supported")
        else:
            raise NotImplementedError(f"Function {node.func.id} not supported")  # type: ignore

    def get_method_offset(self, method_name: str) -> int:
        """メソッドテーブル内のオフセットを取得"""
        # メソッドテーブル内の順序に基づいてインデックスを返す
        method_offsets = {
            "__eq__": 0,
            "__ne__": 1,
            "__lt__": 2,
            "__le__": 3,
            "__gt__": 4,
            "__ge__": 5,
            "__add__": 6,
            "__sub__": 7,
            "__mul__": 8,
            "__div__": 9,
            "__mod__": 10,
            "__radd__": 11,
            "__rsub__": 12,
            "__rmul__": 13,
            "__rdiv__": 14,
            "__rmod__": 15,
            "__neg__": 16,
            "__pos__": 17,
            "__abs__": 18,
            "__str__": 19,
            "__repr__": 20,
            "__int__": 21,
            "__float__": 22,
            "__bool__": 23,
            # ... 他のメソッドも追加
        }
        return method_offsets.get(method_name, -1)
