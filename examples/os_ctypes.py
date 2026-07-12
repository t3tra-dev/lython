import ctypes
import os
import posix
import sys

pid = os.getpid()
print(pid > 0)
print(pid == os.getpid())
print(os.getppid() > 0)
print(os.getuid() >= 0)
print(os.geteuid() >= 0)
print(os.getgid() >= 0)
print(os.getegid() >= 0)
print(posix.getpid() == pid)

print(os.name)
print(os.curdir)
print(os.pardir)
print(os.extsep)
print(os.sep)
print(os.pathsep)
print(os.defpath)
print(os.devnull)
print(os.linesep == "\n")

libc = ctypes.CDLL(None)
labs = libc["labs"]
labs.restype = ctypes.c_long
labs.argtypes = [ctypes.c_long]
print(labs(-42))

print(ctypes.sizeof(ctypes.c_int))
print(sys.platform == "win32")
