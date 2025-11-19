import ast

from lython.mlir import ir
from lython.visitors._base import BaseVisitor as Parser

source = 'print("Hello, world!")\n'
tree = ast.parse(source)
print(ast.dump(tree, indent=4))

ctx = ir.Context()
parser = Parser(ctx)
parser.visit(tree)
print(parser.module, end='')
print("module is valid!" if parser.module.operation.verify() else "module is invalid!")
