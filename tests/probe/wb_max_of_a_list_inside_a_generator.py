# `max`/`min` of a list inside a generator body dies in the lowering:
#
#   cannot adapt runtime bundle builtins.int with physical values
#   (memref<2xi64>) to expected ABI (memref<9xi64>)
#
# Every neighbour compiles (golden: a_generator_prints_what_it_computes):
# `sum`, `any`, `all`, `set`, `sorted`, `zip`, `enumerate`, a list
# comprehension and a dict comprehension all fuse inside a generator now, and
# `max`/`min` outside one is fine. Binding the call to a local first also
# works:
#
#   best = max([4, 1])
#   print(best)          # compiles
#
# ⭐ THE RUNNING BEST IS A BRANCH OPERAND THE DROP PASS DOES NOT REACH. Three
# messages, one cause, and which one appears depends on how far the pass gets:
#
#   print(max(xs))            -> cannot adapt runtime bundle builtins.int with
#                                physical values (memref<2xi64>) to expected
#                                ABI (memref<9xi64>)
#   best = max(xs)            -> lowered Py value still has non-lowered users
#                                for py.add result #0 : user=cf.cond_br
#   max(xs) inside a for      -> control-flow logical block argument still has
#                                users after runtime lowering
#
# The max fusion carries the best-so-far as a cf.cond_br DESTINATION operand,
# and `dropLogicalBranchOperands` (ABI/ControlFlowABI.cpp) removes an operand
# only for the predecessors of the block that OWNS the logical argument. Once
# the generator state machine has restructured the loop, the operand that
# holds the best is on an edge whose destination argument is not in that list,
# so it survives the lowering with a py type.
#
# ⛔ Not attempted here: this is the edge-operand indexing that has already
# produced two separate defects in this compiler, and a guess at it is the
# class of change that mis-executes rather than refuses. `sorted(xs)[-1]`
# compiles inside a generator and is the same answer.
def g():
    print(max([4, 1]))
    yield 0


print(list(g()))
