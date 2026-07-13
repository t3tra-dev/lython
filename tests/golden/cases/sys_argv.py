import sys

# The runner invokes `lyc jit <case>` with no program arguments, so argv is
# exactly [script path]. The path varies per checkout; assert the shape.
print(len(sys.argv))
count = 0
for argument in sys.argv:
    count = count + 1
print(count)
