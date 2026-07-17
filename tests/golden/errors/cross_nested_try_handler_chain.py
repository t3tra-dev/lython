# Cross-check (p2-ehfix x p2-traceback): a nested try/finally that completes
# on the normal path INSIDE an except handler used to skip its finally
# (lowerTry nesting bug), so the traceback track could not pin this chain.
# With the fix the handler runs the nested finally and the raise afterwards
# still chains the handled exception as __context__.
def f() -> None:
    try:
        raise ValueError("first")
    except ValueError:
        try:
            print("nested body")
        finally:
            print("nested finally")
        raise RuntimeError("second")


f()
