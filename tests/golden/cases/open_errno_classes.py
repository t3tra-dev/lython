# open() picks its OSError subclass from errno through the same table os uses,
# so a non-ENOENT failure no longer arrives as FileNotFoundError with a
# hand-written message. Each handler catches the SPECIFIC subclass: an
# `except OSError` fallback would pass even if the class were wrong.
#
# EACCES is not exercised here (it needs os.chmod, which the manifest does not
# provide yet, and a root CI user would not get it anyway); it was verified by
# hand against CPython on an unwritable path.
import os

try:
    open("no_such_directory_here/inner.txt", "r")
except FileNotFoundError as e:
    print("FileNotFoundError", e)

# EISDIR: fopen succeeds on a directory (both glibc and the BSD libcs reach
# open(2) with O_RDONLY, which permits it), so open() stats the path itself
# rather than letting every later read answer "".
os.mkdir("open_errno_dir_case")
try:
    open("open_errno_dir_case", "r")
except IsADirectoryError as e:
    print("IsADirectoryError", e)
os.rmdir("open_errno_dir_case")

# The happy path still works, and errno is not consulted on it.
f = open("open_errno_roundtrip.tmp", "w")
f.write("ok\n")
f.close()
g = open("open_errno_roundtrip.tmp", "r")
print(g.read().strip())
g.close()
os.unlink("open_errno_roundtrip.tmp")
