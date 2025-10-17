from .mlir import ir
from .visitors import BaseVisitor

__all__ = ["Parser"]


class Parser(BaseVisitor):
    def __init__(self, ctx: ir.Context) -> None:
        super().__init__(ctx)

        from .visitors.mod import ModVisitor
        from .visitors.stmt import StmtVisitor

        subvisitors: dict[str, BaseVisitor] = {
            "Module": ModVisitor(ctx),
            "Stmt": StmtVisitor(ctx),
        }
        for v in subvisitors.values():
            v.subvisitors = subvisitors
        self.subvisitors = subvisitors
