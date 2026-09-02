# What: a def that lives inside a module-level `if` in an IMPORTED module. The
# module's binders read the module through the branch the test picks and the
# emission walk read the raw body, so the name resolved and the body was never
# emitted. The decode is that the function is CALLED and its arguments come back
# in the answer: a compiler that emitted the other branch's body would print the
# same shape with the wrong separator in it, and one that emitted neither cannot
# get past the binding.
import a_module_that_chooses_by_platform as paths

print(paths.SEP)
print(paths.joiner("a", "b"))
print([paths.joiner(x, "z") for x in ["p", "q"]])
print(paths.MISSING is None)
