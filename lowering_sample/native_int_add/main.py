from native import native, i32  # type: ignore

@native
def add(a: i32, b: i32) -> i32:
    return a + b
