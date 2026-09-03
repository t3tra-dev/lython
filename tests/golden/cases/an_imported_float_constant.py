# An imported module's float and negative constants. The literal channel that
# carries an imported constant had arms for str, bool, int and None, and every
# other spelling widened to INT -- so a float could not be carried at all, and
# a negative literal is a `UnaryOp` over a Constant rather than a Constant, so
# it missed the channel entirely for ints as well as floats.
import a_module_of_numeric_constants as c

print(c.RATIO, c.TINY, c.BIG, c.WHOLE, c.NEG_F, c.NEG_I)
print(c.RATIO * 2, c.WHOLE + 1, round(c.RATIO), int(c.WHOLE))
print(c.NAME, c.FLAG, c.LIMIT)
print(c.Settings().summary(), c.Settings.limit)
