# float() applied to the bottom rung of the numeric tower. Execution is needed
# because the defect this pins was a MISSING rung, and a rung that converts to
# the wrong constant compiles just as well as one that converts correctly:
# only the printed 1.0 / 0.0 separates "True went through int.__float__" from
# "True was read as a bare truth bit".
#
# The three arms that already worked are here so a repair of the bool arm that
# reroutes the others is caught in the same file.

print(float(True))
print(float(False))
print(float(1))
print(float(2.5))
print(float("3.5"))

flag: bool = True
print(float(flag))
print(float(True) + 0.5)
