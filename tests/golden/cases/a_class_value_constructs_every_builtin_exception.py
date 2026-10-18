# WHAT: every builtin exception class that takes one message, constructed
# through a class value: its class, str, repr and args, and whether it is an
# Exception.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a call through a
# `type[BaseException]` builds every builtin exception with the bound's
# initializer and the value's class word; only those whose initializer is
# their own get an arm (`kBuiltinExceptionsWithOwnInit`, KeyError). An
# initializer of its own added to the runtime without that list would print
# the base's message here -- KeyError("m") printed `m` before it was listed.
def make(c: type[BaseException], m: str) -> BaseException:
    return c(m)


for c in [
    BaseException, Exception, RuntimeError, TypeError, ValueError, KeyError,
    IndexError, AssertionError, StopIteration, StopAsyncIteration,
    ArithmeticError, LookupError, ZeroDivisionError, SystemExit,
    GeneratorExit, OSError, FileNotFoundError, KeyboardInterrupt,
    FloatingPointError, OverflowError, BufferError, EOFError, ImportError,
    ModuleNotFoundError, MemoryError, NameError, UnboundLocalError,
    AttributeError, ReferenceError, NotImplementedError, RecursionError,
    PythonFinalizationError, SyntaxError, IndentationError, TabError,
    SystemError, UnicodeError, Warning, BytesWarning, DeprecationWarning,
    EncodingWarning, FutureWarning, ImportWarning, PendingDeprecationWarning,
    ResourceWarning, RuntimeWarning, SyntaxWarning, UnicodeWarning,
    UserWarning, BlockingIOError, ChildProcessError, ConnectionError,
    BrokenPipeError, ConnectionAbortedError, ConnectionRefusedError,
    ConnectionResetError, FileExistsError, InterruptedError,
    IsADirectoryError, NotADirectoryError, PermissionError,
    ProcessLookupError, TimeoutError
]:
    e = make(c, "m")
    print(type(e).__name__, str(e), repr(e), e.args, isinstance(e, Exception))
