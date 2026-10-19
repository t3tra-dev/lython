# WHAT: A tuple alias defined in an IMPORTED module and used in its
#   annotations reaches them as a class named builtins.Info that does not
#   exist ("does not provide manifest method '__getitem__'"). The same alias
#   in the main module works. unittest's `_SysExcInfoType` alias has this
#   shape.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- an imported module's aliases are bound in its own scope;
#   its `type X = ...` statements are declarations too.
# CLASSIFICATION @ lang-gaps 699a549f + the fix: 1 正しい
# CPython 3.14 expects: 1a / KeyError
import sys
import ut_mod_type_alias

print(ut_mod_type_alias.pair_text((1, "a")))
try:
    raise KeyError("k")
except KeyError:
    print(ut_mod_type_alias.first_name(sys.exc_info()))
