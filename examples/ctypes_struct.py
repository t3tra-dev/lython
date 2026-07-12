import ctypes
import sys


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int)]


class Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_int)]


def now_seconds() -> int:
    libc = ctypes.CDLL(None)
    gettimeofday = libc["gettimeofday"]
    gettimeofday.restype = ctypes.c_int
    gettimeofday.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    tv = Timeval()
    rc = gettimeofday(ctypes.byref(tv), None)
    if rc != 0:
        return -1
    return tv.tv_sec


print(ctypes.sizeof(Point))
p = Point(3, 4)
print(p.x * p.y)
p.x = 10
print(p.x * p.y)

cell = ctypes.c_int(41)
cell.value = cell.value + 1
print(cell.value)

if sys.platform != "win32":
    print(now_seconds() > 1500000000)
