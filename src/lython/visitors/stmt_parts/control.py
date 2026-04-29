# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import cf as cf_ops


class StmtControlMixin:
    """Statement lowering for structured control-flow AST nodes."""

    def visit_For(self, node: ast.For) -> None:
        raise NotImplementedError("For statement not implemented")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        raise NotImplementedError("Async for statement not implemented")

    def visit_While(self, node: ast.While) -> None:
        raise NotImplementedError("While statement not implemented")

    def visit_If(self, node: ast.If) -> None:
        cond_value = self.require_value(node.test, self.visit(node.test))
        i1 = ir.IntegerType.get_signless(1, context=self.ctx)

        if self._in_native_func:
            cond = cond_value
        else:
            with self._loc(node), self.insertion_point():
                cond = py_ops.CastToPrimOp(
                    i1, cond_value, ir.StringAttr.get("exact", self.ctx)
                ).result

        assert self.current_block is not None
        parent_region = self.current_block.region
        true_block = (
            parent_region.blocks.append()  # pyright: ignore[reportUnknownMemberType]
        )
        false_block = (
            parent_region.blocks.append()  # pyright: ignore[reportUnknownMemberType]
        )
        with self._loc(node), self.insertion_point():
            cf_ops.CondBranchOp(cond, [], [], true_block, false_block)

        def handle_branch(block: ir.Block, statements: list[ast.stmt]) -> bool:
            self._set_insertion_block(block)
            self.push_scope()
            for stmt in statements:
                self.visit(stmt)
            terminated = self._block_terminated(block)
            self.pop_scope()
            return terminated

        true_terminated = handle_branch(true_block, node.body)
        false_terminated = handle_branch(false_block, node.orelse or [])

        if true_terminated and false_terminated:
            self._set_insertion_block(None)
            return

        merge_block = (
            parent_region.blocks.append()  # pyright: ignore[reportUnknownMemberType]
        )
        if not true_terminated:
            with self._loc(node), ir.InsertionPoint(true_block):
                cf_ops.BranchOp([], merge_block)
        if not false_terminated:
            with self._loc(node), ir.InsertionPoint(false_block):
                cf_ops.BranchOp([], merge_block)

        self._set_insertion_block(merge_block)

    def visit_With(self, node: ast.With) -> None:
        raise NotImplementedError("With statement not implemented")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        raise NotImplementedError("Async with statement not implemented")

    def visit_Match(self, node: ast.Match) -> None:
        raise NotImplementedError("Match statement not implemented")
