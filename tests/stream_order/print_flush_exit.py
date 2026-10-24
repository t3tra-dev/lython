# stdout is buffered and stderr is not, so in one pipe the stderr lines come
# first until a flush; the SystemExit message comes after the buffered stdout.
import sys
print("out 1")
print("err 1", file=sys.stderr)
print("out 2")
sys.stdout.flush()
print("err 2", file=sys.stderr)
print("out 3", flush=True)
print("err 3", file=sys.stderr)
print("out 4")
raise SystemExit("bye")
