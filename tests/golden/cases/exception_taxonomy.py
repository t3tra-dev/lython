try:
    raise OverflowError("too big")
except ArithmeticError as e:
    print("arith:", e)
try:
    raise NotImplementedError("later")
except RuntimeError as e:
    print("runtime:", e)
try:
    raise RecursionError("deep")
except RuntimeError as e:
    print("recursion:", e)
try:
    raise ModuleNotFoundError("no mod")
except ImportError as e:
    print("import:", e)
try:
    raise UnboundLocalError("no local")
except NameError as e:
    print("name:", e)
try:
    raise BrokenPipeError("pipe")
except ConnectionError as e:
    print("conn:", e)
try:
    raise ConnectionRefusedError("refused")
except OSError as e:
    print("os:", e)
try:
    raise TabError("bad tab")
except IndentationError as e:
    print("indent:", e)
try:
    raise TabError("bad tab 2")
except SyntaxError as e:
    print("syntax:", e)
try:
    raise DeprecationWarning("old")
except Warning as e:
    print("warning:", e)
try:
    raise UnicodeError("uni")
except ValueError as e:
    print("value:", e)
try:
    raise KeyboardInterrupt("stop")
except BaseException as e:
    print("base:", e)
try:
    raise EOFError("eof")
except Exception as e:
    print("exc:", e)
try:
    raise PermissionError("denied")
except FileNotFoundError as e:
    print("wrong handler:", e)
except PermissionError as e:
    print("right handler:", e)
print(repr(ValueError("boom")))
print(repr(OverflowError()))
print(repr(TimeoutError("late")))
print(str(AttributeError("attr")))
e2 = IndexError("range")
print(e2.args)
print(len(e2.args))
print(ValueError().args)
