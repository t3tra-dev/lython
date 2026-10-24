# What was printed before an uncaught exception lands before its traceback.
import sys
print("before")
print("err", file=sys.stderr)
raise ValueError("boom")
