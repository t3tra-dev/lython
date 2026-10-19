# WHAT: `traceback.format_exception_only(e)` for an exception class of an
#   IMPORTED module: CPython prefixes the module
#   ('ut_mod_lib_error.LibError: lib'), as it does for every module but
#   __main__ and builtins; here the bare class name is printed. The
#   uncaught-exception printer gets it right.
# KIND: バグ (bug) -- silent: prints the wrong text
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# differential: expect-wrong the module prefix is missing
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   cpython="['ut_mod_lib_error.LibError: lib\\n']['MainError: main\\n']['ValueError: v\\n']" lyc="['LibError: lib\\n']['MainError: main\\n']['ValueError: v\\n']"
# CPython 3.14 expects: ['ut_mod_lib_error.LibError: lib\n']['MainError: main\n']['ValueError: v\n']
import traceback
import ut_mod_lib_error


class MainError(Exception):
    pass


print(traceback.format_exception_only(ut_mod_lib_error.LibError("lib")), end="")
print(traceback.format_exception_only(MainError("main")), end="")
print(traceback.format_exception_only(ValueError("v")), end="")
