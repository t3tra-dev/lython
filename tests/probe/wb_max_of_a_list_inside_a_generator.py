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
# ⭐ AN INT LANE IS BEING FILLED WITH A LIST'S ABI. memref<9xi64> is the list
# object's shape, and the value handed to it is the running best, an int --
# so the two lanes the max fusion carries (the source list and the best so
# far) are matched to the frame in the wrong order once the state machine
# splits the loop. Not the callee-reference crossing that the same sweep
# fixed: the callee here is re-emitted, and the value that fails is an
# ARGUMENT the fusion carries, which cannot be re-emitted because it is not a
# pure lookup.
def g():
    print(max([4, 1]))
    yield 0


print(list(g()))
