import ast

__all__ = ["dump_ast"]


def dump_ast(code: str, indent: int = 2) -> str:
    return ast.dump(ast.parse(code), indent=indent)
