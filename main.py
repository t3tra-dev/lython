from lython.mlir import ir

ctx = ir.Context()
i32 = ir.IntegerType.get_signless(32, ctx)


with ir.Location.unknown(ctx):
    module = ir.Module.create()
    func = ir.Operation.create(
        "func.func",
        results=[ir.FunctionType.get([], [i32], ctx)],
        operands=[],
    )
    module.body.append(func)

print(module)

"""
"builtin.module"() ({
  %0 = "func.func"() : () -> (() -> i32)
}) : () -> ()
"""
