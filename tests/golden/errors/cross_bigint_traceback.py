# Cross-check (p2-bigint-finish x p2-traceback): int ** constant negative
# int flows through the float result lane, and a heap bigint (10**30 scale)
# renders exactly in an uncaught exception's message.
print(2 ** -1)
print(10 ** -3)
print((10 ** 300) ** -1 == 1e-300)
big = 10 ** 30
raise ValueError("big value " + str(big + 7))
