import sys

# write() returns the codepoint count (CPython TextIO.write semantics) and
# flush() is a no-op on the unbuffered fd path.
n = sys.stdout.write("stream\n")
print(n)
sys.stdout.flush()
sys.stderr.write("diagnostic\n")
print("done")
