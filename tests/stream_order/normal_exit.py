# A normal exit flushes what is left; stderr written last still comes first.
import sys
for i in range(3):
    print("line", i)
print("err", file=sys.stderr)
