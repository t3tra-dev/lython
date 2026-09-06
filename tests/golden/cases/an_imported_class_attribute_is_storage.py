# What: a class attribute of an IMPORTED class is one storage, shared and
# mutable, the way CPython's class dict is. It used to be a compile-time
# constant re-materialized per read, so a container attribute had no arm at all
# ("unsupported static class attribute expression for 'seen'", out of the
# LOWERING rather than as a diagnostic) and a write had none either -- both for
# shapes that work written in one file.
#
# Running it is the whole evidence: the list has to be the SAME list across
# calls and across modules, the counter has to survive between them, and the
# initializer has to have run before the main module's first statement -- which
# is where CPython runs an imported module's class body.
from a_module_with_a_class_registry import Registry

print("initial", Registry.seen, Registry.count, Registry.label)
print("recorded", Registry.record("a"), Registry.record("b"))
print("after", Registry.seen, Registry.count)

Registry.seen.append("direct")
Registry.count = 10
print("written", Registry.seen, Registry.count)

shared = Registry.seen
shared.append("through a name")
print("same object", Registry.seen)
