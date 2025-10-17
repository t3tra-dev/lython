from lython.mlir import ir
from lython.mlir.dialects import arith, func

ctx = ir.Context()
i32 = ir.IntegerType.get_signless(32, ctx)
i32_zero = ir.IntegerAttr.get(i32, 0)


with ir.Location.unknown(ctx):
    module = ir.Module.create()

    with ir.InsertionPoint(module.body):
        f = func.FuncOp(name="add_i32", type=ir.FunctionType.get([i32, i32], [i32]))

    entry = f.add_entry_block()
    a0, a1 = entry.arguments

    with ir.InsertionPoint(entry):
        s = arith.addi(a0, a1)  # type: ignore
        func.return_([s])  # type: ignore

print(module)

"""
"builtin.module"() ({
  %0 = "func.func"() : () -> i32
}) : () -> ()
"""
