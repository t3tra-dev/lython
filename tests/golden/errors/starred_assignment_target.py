# `a, *rest = xs` used to COMPILE and leave `rest` holding whatever it held
# before -- the unpack walk reads target[i] for each element and a starred
# target has no index, so it was skipped in silence. It only showed up when
# the name already existed; otherwise the program was refused for an
# unrelated reason ("unresolved name 'rest'").
xs: list[int] = [1, 2, 3, 4]
rest: list[int] = [7, 7, 7]
a, *rest = xs
print(a, rest)
