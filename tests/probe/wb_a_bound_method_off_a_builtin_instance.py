# OPEN. A method read off an INSTANCE of a manifest class, without calling it:
#
#     s = "ab"
#     m = s.upper          # inferred as the RESULT of s.upper(), i.e. str
#     print(m())           # builtins.str is not callable
#     print(s.upper)       # attr.get object type has no class schema
#
# CPython prints `<built-in method upper of str object ...>` for the read and
# `AB` for the call. The UNBOUND twin -- `str.upper` off the class -- is fixed
# (cases/an_unbound_method_off_a_builtin_class): a forwarder is synthesized for
# the value spelling and the call spelling shifts its first argument into the
# receiver. This half needs the receiver to be CARRIED, which the forwarder
# cannot do: it is a module function with no captures.
#
# ⛔ WHY THE UNBOUND REPAIR DOES NOT REACH HERE. `emitMethodObject` builds a
# closure for exactly this shape and is what makes `m = c.go` work on a SOURCE
# class -- it captures the receiver and forwards to the method's own emitted
# function. A manifest method has no emitted function to forward to, so the
# wrapper would have to be the synthesized forwarder WITH the receiver as a
# capture. That is a bound object over a synthesized def, and the piece that is
# missing is not the def (the unbound repair writes one) but the capture: the
# forwarder's parameter IS the receiver, so binding it means partial
# application, which nothing in the emitter does.
#
# ⛔ AND THE INFERENCE IS WRONG BEFORE THE EMITTER IS REACHED. `s.upper` types
# as `builtins.str` -- the RESULT of a zero-argument call -- because the
# manifest read channel folds a method to its call. TypeSystem.cpp says so at
# the arm above ("A METHOD READ WITHOUT A CALL IS THE BOUND METHOD"), which
# repaired the SOURCE-class side of the same question and left this one.
#
# Measured 2026-09-03: `m = s.upper` then `m()` is "builtins.str is not
# callable"; `print(s.upper)` dies in the lowering as "attr.get object type has
# no class schema"; `xs.sort` (a None-returning one) is "!py.literal<None> is
# not callable".
s = "ab"
m = s.upper
print(m())
