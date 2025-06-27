"""
エラー報告システム

型エラーやその他のコンパイルエラーを収集し、
TypeScript風の分かりやすい形式で報告します。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TextIO

from ..ast_nodes import Span
from .types import Type


class ErrorLevel(Enum):
    """エラーレベル"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class DiagnosticMessage:
    """診断メッセージ"""

    level: ErrorLevel
    message: str
    span: Span
    code: Optional[str] = None  # エラーコード (E001, W002 など)
    help_text: Optional[str] = None  # 修正のヒント
    related_spans: List[Span] = field(default_factory=list)  # 関連する位置情報

    def __str__(self) -> str:
        level_str = self.level.value
        location = f"{self.span.lineno}:{self.span.col_offset}"
        code_str = f"[{self.code}] " if self.code else ""
        return f"{level_str}: {code_str}{self.message} at {location}"


@dataclass
class SourceLine:
    """ソースコードの行情報"""

    line_number: int
    content: str

    def get_indicator_line(self, col_start: int, col_end: Optional[int] = None) -> str:
        """エラー位置を示すインジケータ行を生成"""
        if col_end is None:
            col_end = col_start + 1

        indicator = " " * col_start + "^" * (col_end - col_start)
        return indicator


class ErrorReporter:
    """エラー報告システムのメインクラス"""

    def __init__(self, source_file: Optional[str] = None):
        self.diagnostics: List[DiagnosticMessage] = []
        self.source_file = source_file
        self.source_lines: Dict[int, str] = {}
        self.error_count = 0
        self.warning_count = 0

    def load_source_lines(self, source_content: str):
        """ソースコードの内容を読み込み"""
        lines = source_content.split("\n")
        self.source_lines = {i + 1: line for i, line in enumerate(lines)}

    def add_error(
        self,
        message: str,
        span: Span,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
    ) -> None:
        """エラーを追加"""
        diagnostic = DiagnosticMessage(
            level=ErrorLevel.ERROR,
            message=message,
            span=span,
            code=code,
            help_text=help_text,
        )
        self.diagnostics.append(diagnostic)
        self.error_count += 1

    def add_warning(
        self,
        message: str,
        span: Span,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
    ) -> None:
        """警告を追加"""
        diagnostic = DiagnosticMessage(
            level=ErrorLevel.WARNING,
            message=message,
            span=span,
            code=code,
            help_text=help_text,
        )
        self.diagnostics.append(diagnostic)
        self.warning_count += 1

    def add_info(self, message: str, span: Span, code: Optional[str] = None) -> None:
        """情報メッセージを追加"""
        diagnostic = DiagnosticMessage(
            level=ErrorLevel.INFO, message=message, span=span, code=code
        )
        self.diagnostics.append(diagnostic)

    def add_type_error(
        self, expected: Type, actual: Type, span: Span, context: str = ""
    ) -> None:
        """型エラーを追加"""
        context_str = f" in {context}" if context else ""
        message = (
            f"Type mismatch{context_str}: expected '{expected}', but got '{actual}'"
        )
        help_text = self._generate_type_help(expected, actual)
        self.add_error(message, span, "E001", help_text)

    def add_undefined_variable_error(self, var_name: str, span: Span) -> None:
        """未定義変数エラーを追加"""
        message = f"Undefined variable: '{var_name}'"
        help_text = f"Variable '{var_name}' is not defined in the current scope"
        self.add_error(message, span, "E002", help_text)

    def add_argument_count_error(
        self, expected: int, actual: int, span: Span, func_name: str = ""
    ) -> None:
        """引数の数エラーを追加"""
        func_str = f"function '{func_name}'" if func_name else "function"
        message = (
            f"Argument count mismatch: {func_str} expects {expected} arguments, "
            f"but {actual} were provided"
        )
        self.add_error(message, span, "E003")

    def add_native_annotation_required_error(
        self, func_name: str, span: Span, missing_type: str
    ) -> None:
        """@native関数の型注釈必須エラーを追加"""
        message = (
            f"@native function '{func_name}' requires type annotation for "
            f"{missing_type}"
        )
        help_text = f"Add type annotation like: {missing_type}: int"
        self.add_error(message, span, "E004", help_text)

    def add_non_callable_error(self, type_name: str, span: Span) -> None:
        """呼び出し不可能型エラーを追加"""
        message = f"'{type_name}' object is not callable"
        self.add_error(message, span, "E005")

    def add_attribute_error(self, obj_type: str, attr_name: str, span: Span) -> None:
        """属性エラーを追加"""
        message = f"'{obj_type}' object has no attribute '{attr_name}'"
        self.add_error(message, span, "E006")

    def add_index_error(self, obj_type: str, span: Span) -> None:
        """インデックスエラーを追加"""
        message = f"'{obj_type}' object is not subscriptable"
        self.add_error(message, span, "E007")

    def add_return_type_mismatch_error(
        self, expected: Type, actual: Type, span: Span, func_name: str
    ) -> None:
        """戻り値型不一致エラーを追加"""
        message = (
            f"Return type mismatch in function '{func_name}': expected "
            f"'{expected}', but got '{actual}'"
        )
        help_text = self._generate_type_help(expected, actual)
        self.add_error(message, span, "E008", help_text)

    def add_conversion_warning(
        self, from_type: Type, to_type: Type, span: Span
    ) -> None:
        """型変換警告を追加"""
        message = f"Implicit conversion from '{from_type}' to '{to_type}'"
        help_text = f"Consider explicit conversion: to_native({from_type}, {to_type})"
        self.add_warning(message, span, "W001", help_text)

    def add_unused_variable_warning(self, var_name: str, span: Span) -> None:
        """未使用変数警告を追加"""
        message = f"Unused variable: '{var_name}'"
        help_text = (
            "Remove the variable or use it, or prefix with '_' to suppress this "
            "warning"
        )
        self.add_warning(message, span, "W002", help_text)

    def has_errors(self) -> bool:
        """エラーがあるかどうか"""
        return self.error_count > 0

    def has_warnings(self) -> bool:
        """警告があるかどうか"""
        return self.warning_count > 0

    def clear(self) -> None:
        """全ての診断情報をクリア"""
        self.diagnostics.clear()
        self.error_count = 0
        self.warning_count = 0

    def get_summary(self) -> str:
        """エラー・警告の要約を取得"""
        if self.error_count == 0 and self.warning_count == 0:
            return "No errors or warnings"

        parts = []
        if self.error_count > 0:
            parts.append(
                f"{self.error_count} error{'s' if self.error_count > 1 else ''}"
            )
        if self.warning_count > 0:
            parts.append(
                f"{self.warning_count} warning{'s' if self.warning_count > 1 else ''}"
            )

        return ", ".join(parts)

    def print_diagnostics(
        self, file: TextIO = sys.stderr, show_source: bool = True
    ) -> None:
        """診断情報を出力"""
        if not self.diagnostics:
            return

        # エラーレベルでソート（ERROR > WARNING > INFO > HINT）
        level_order = {
            ErrorLevel.ERROR: 0,
            ErrorLevel.WARNING: 1,
            ErrorLevel.INFO: 2,
            ErrorLevel.HINT: 3,
        }
        sorted_diagnostics = sorted(
            self.diagnostics, key=lambda d: (d.span.lineno, level_order[d.level])
        )

        for diagnostic in sorted_diagnostics:
            self._print_diagnostic(diagnostic, file, show_source)

        # 要約を出力
        print(f"\n{self.get_summary()}", file=file)

    def _print_diagnostic(
        self, diagnostic: DiagnosticMessage, file: TextIO, show_source: bool
    ) -> None:
        """個別の診断情報を出力"""
        # レベルに応じた色付け（ANSIエスケープシーケンス）
        colors = {
            ErrorLevel.ERROR: "\033[91m",  # 赤
            ErrorLevel.WARNING: "\033[93m",  # 黄
            ErrorLevel.INFO: "\033[94m",  # 青
            ErrorLevel.HINT: "\033[92m",  # 緑
        }
        reset_color = "\033[0m"

        level_color = colors.get(diagnostic.level, "")
        level_str = diagnostic.level.value.upper()

        # ファイル名と位置情報
        location = ""
        if self.source_file:
            location = f"{self.source_file}:"
        location += f"{diagnostic.span.lineno}:{diagnostic.span.col_offset}"

        # エラーコード
        code_str = f"[{diagnostic.code}] " if diagnostic.code else ""

        # メインメッセージ
        print(
            f"{level_color}{level_str}{reset_color}: {code_str}{diagnostic.message}",
            file=file,
        )
        print(f"  --> {location}", file=file)

        # ソースコードの表示
        if show_source and diagnostic.span.lineno in self.source_lines:
            self._print_source_context(diagnostic, file)

        # ヘルプテキスト
        if diagnostic.help_text:
            print(f"  = help: {diagnostic.help_text}", file=file)

        print(file=file)  # 空行

    def _print_source_context(
        self, diagnostic: DiagnosticMessage, file: TextIO
    ) -> None:
        """ソースコードのコンテキストを表示"""
        line_num = diagnostic.span.lineno

        # 前後の行も表示（利用可能であれば）
        start_line = max(1, line_num - 1)
        end_line = min(
            max(self.source_lines.keys()) if self.source_lines else line_num,
            line_num + 1,
        )

        for i in range(start_line, end_line + 1):
            if i not in self.source_lines:
                continue

            line_content = self.source_lines[i]
            line_marker = " " if i != line_num else ">"

            print(f"  {i:3} {line_marker} {line_content}", file=file)

            # エラー位置のインジケータ
            if i == line_num:
                col_start = diagnostic.span.col_offset
                col_end = diagnostic.span.end_col_offset or (col_start + 1)

                # 行番号とマーカーのスペースを考慮
                prefix_len = 6  # "  123 > " の長さ
                indicator = " " * (prefix_len + col_start) + "^" * (col_end - col_start)
                print(indicator, file=file)

    def _generate_type_help(self, expected: Type, actual: Type) -> str:
        """型エラーのヘルプテキストを生成"""
        # 具体的なヘルプを生成
        if expected.is_native() and actual.is_pyobject():
            return (
                f"Use to_native({actual}, {expected}) to convert "
                f"PyObject to native type"
            )
        elif expected.is_pyobject() and actual.is_native():
            return (
                f"Use to_pytype({actual}, {expected}) to convert "
                f"native type to PyObject"
            )
        elif str(expected) == "int" and str(actual) == "float":
            return (
                "Consider using int() to convert float to int, "
                "or change the expected type to float"
            )
        elif str(expected) == "str" and str(actual) in ["int", "float"]:
            return f"Use str({actual}) to convert {actual} to string"
        else:
            return f"Ensure the value is of type '{expected}'"

    def to_json(self) -> List[Dict[str, Any]]:
        """診断情報をJSON形式で出力（IDE統合用）"""
        result = []
        for diagnostic in self.diagnostics:
            result.append(
                {
                    "level": diagnostic.level.value,
                    "message": diagnostic.message,
                    "span": {
                        "start": {
                            "line": diagnostic.span.lineno,
                            "column": diagnostic.span.col_offset,
                        },
                        "end": {
                            "line": diagnostic.span.end_lineno
                            or diagnostic.span.lineno,
                            "column": diagnostic.span.end_col_offset
                            or (diagnostic.span.col_offset + 1),
                        },
                    },
                    "code": diagnostic.code,
                    "help": diagnostic.help_text,
                    "file": self.source_file,
                }
            )
        return result


# グローバルエラーレポーター（簡単なアクセス用）
_global_reporter: Optional[ErrorReporter] = None


def get_global_reporter() -> ErrorReporter:
    """グローバルエラーレポーターを取得"""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = ErrorReporter()
    return _global_reporter


def set_global_reporter(reporter: ErrorReporter) -> None:
    """グローバルエラーレポーターを設定"""
    global _global_reporter
    _global_reporter = reporter


def report_error(
    message: str,
    span: Span,
    code: Optional[str] = None,
    help_text: Optional[str] = None,
) -> None:
    """グローバルレポーターにエラーを追加"""
    get_global_reporter().add_error(message, span, code, help_text)


def report_warning(
    message: str,
    span: Span,
    code: Optional[str] = None,
    help_text: Optional[str] = None,
) -> None:
    """グローバルレポーターに警告を追加"""
    get_global_reporter().add_warning(message, span, code, help_text)


def report_type_error(
    expected: Type, actual: Type, span: Span, context: str = ""
) -> None:
    """グローバルレポーターに型エラーを追加"""
    get_global_reporter().add_type_error(expected, actual, span, context)
