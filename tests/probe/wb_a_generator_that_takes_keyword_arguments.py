# WHAT: A generator function that takes `**kwargs` is refused: the state
#   machine that runs a non-trivial generator body has no argument lane for
#   the keyword dict, and the fallback runs only straight-line int yields.
#   `*args` has a lane and works; a @contextmanager METHOD with `**params`
#   (unittest's subTest) works too, so this is the plain-function spelling.
# KIND: 未対応 (unsupported) -- loud, from the lowering
# FOUND: 2026-10-20, restoring subTest in the unittest port.
# CLASSIFICATION: loud 拒否 (診断)
#   source generator next lowering currently supports only straight-line pure
#   int yield bodies; the state machine that does support it declined this
#   generator because it takes **kwargs, which has no argument lane
# CPython 3.14 expects: 1 / {'i': 1, 'j': 2}


def g(**params: int):
    yield 1
    print(params)


for x in g(i=1, j=2):
    print(x)
