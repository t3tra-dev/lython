class Row[*Ts]:
    def __init__(self, width: int) -> None:
        self.width = width


# A TypeVarTuple parameter is a parameter-LIST unknown: no instantiation fixes
# a field layout to specialize.
r: Row[int, str] = Row(2)
print(r.width)
