# RE-MEASURED 2026-10-16: NOT A GENERATOR DEFECT. The same refusal comes from
# an unannotated nested function that calls itself, generator or not:
#
#     def outer(n: int) -> int:
#         def f(k: int):                 # no return annotation
#             if k > 0: return k + f(k - 1)
#             return 0
#         return f(n)                    # "builtins.object is not callable"
#
# and goes away with `-> Iterator[int]` (or `-> int`) on the nested def. A
# TOP-LEVEL unannotated recursive function gets its result from the module
# fixpoint (`resultTypeOverrides`); a nested def's signature is inferred once,
# while its own name is still unbound. So this is the annotation-required
# boundary (cf. wb_an_unannotated_parameter_of_a_local_function), and what is
# wrong today is the message, which names the compiler's bookkeeping rather
# than the missing annotation. The note below is the 2026-09 reading.
#
# A nested generator that delegates to itself is refused:
#
#   emit error: static type !py.contract<"builtins.object"> is not callable:
#   no manifest __call__ contract
#
# The non-generator spelling of the same recursion works (golden:
# a_nested_function_that_calls_itself), and so does a TOP-LEVEL recursive
# generator with `yield from`.
#
# ⭐ The self-reference a nested def now gets is typed from the signature the
# body is being emitted under. For a generator that signature is the RESUME
# result, and the public one -- the GeneratorType -- is what a self-call has to
# see; the arm that chooses between them is keyed on sig.isGeneratorFunction,
# which is set, so what is left is the `yield from` delegation reading the
# generator's element type through a name whose type is not yet the public one
# at the point the walk asks. Recorded rather than guessed at: the top-level
# recursive-generator repair (`emitCallableFunction`'s "A GENERATOR CALLING
# ITSELF GETS A GENERATOR") is the shape the fix has to take one scope in.
def outer(n: int) -> "list[int]":
    def gen(k: int):
        if k > 0:
            yield k
            yield from gen(k - 1)

    return list(gen(n))


print(outer(3))
