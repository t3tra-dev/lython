# min()/max() fold over an iterable or over two or more operands; neither form
# takes a key. Falling out of the fold left the generic call path to look the
# name up, and a builtin is not a value there -- the report was "unresolved
# name 'min'", which points at the name instead of the keyword.
xs = [3, 1, 2]
print(min(xs, key=lambda v: -v))
