from typing import List
from native import i32  # type: ignore

def sum_list(xs: List[i32]) -> i32:
    s: i32 = i32(0)
    for x in xs:
        s = s + x
    return s
