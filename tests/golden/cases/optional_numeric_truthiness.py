def check_int(v: int | None) -> str:
    if v:
        return "truthy"
    return "falsy"
print(check_int(None))
print(check_int(0))
print(check_int(7))
print(check_int(-3))
def check_float(v: float | None) -> str:
    if v:
        return "truthy"
    return "falsy"
print(check_float(None))
print(check_float(0.0))
print(check_float(-0.0))
print(check_float(2.5))
def pick(v: int | None) -> int:
    return v or 7
print(pick(None))
print(pick(0))
print(pick(3))
def gate(v: float | None) -> str:
    if not v:
        return "empty-or-none"
    return "value"
print(gate(0.0))
print(gate(None))
print(gate(1.25))
