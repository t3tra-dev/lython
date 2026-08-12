# min()/max() DO take a key now -- it rides the same loop as one more carried
# accumulator -- but the key has to have a type to compare. An unannotated
# lambda has none, and the report has to say that rather than blame the
# keyword: before the fold took keys at all this was "unresolved name 'min'",
# which pointed at the name instead.
xs = [3, 1, 2]
print(min(xs, key=lambda v: -v))
