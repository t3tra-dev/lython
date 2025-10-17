import ast
import inspect

from lython.mlir import ir
from lython.parser import Parser


def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


source = inspect.getsource(fib)
tree = ast.parse(source)
print(ast.dump(tree, indent=4))

ctx = ir.Context()
parser = Parser(ctx)
parser.visit(tree)
