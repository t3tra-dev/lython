# os's filesystem surface against a tree the case builds and removes itself.
# Every path stays RELATIVE so the OSError messages -- which carry the failing
# path -- are host-independent and match CPython 3.14's byte for byte.
import os

root = "_golden_os_fs_tmp"
sub = os.path.join(root, "sub")
deep = os.path.join(sub, "deep")
a = os.path.join(sub, "a.txt")
b = os.path.join(root, "b.txt")
c = os.path.join(root, "c.txt")
d = os.path.join(root, "d.txt")
missing = os.path.join(root, "missing")

# --- makedirs builds the intermediate levels --------------------------------
os.makedirs(deep, 0o755, True)
print(os.path.isdir(root), os.path.isdir(sub), os.path.isdir(deep))

handle = open(a, "w")
handle.write("hello")
handle.close()
handle = open(b, "w")
handle.write("bb")
handle.close()

# --- listdir drops '.' and '..' and is sorted here for determinism ----------
names = os.listdir(root)
names.sort()
print(names)
print(os.listdir(deep))

# --- the os.path predicates over real inodes --------------------------------
print(os.path.isfile(b), os.path.isdir(b))
print(os.path.exists(b), os.path.exists(missing))
print(os.path.lexists(missing), os.path.islink(b))

# --- stat / lstat -----------------------------------------------------------
info = os.stat(a)
print(info.st_size)
print(info.st_mode & 0o170000 == 0o100000)
print(info.st_nlink >= 1)
print(info.st_uid == os.getuid(), info.st_gid == os.getgid())
print(os.lstat(b).st_size)
print(os.stat(sub).st_mode & 0o170000 == 0o040000)

# --- access -----------------------------------------------------------------
print(os.access(root, os.F_OK), os.access(missing, os.F_OK))
print(os.access(b, os.R_OK), os.access(b, os.W_OK))

# --- rename / replace / remove ---------------------------------------------
os.rename(b, c)
print(os.path.isfile(b), os.path.isfile(c))
os.replace(c, d)
print(os.path.isfile(c), os.path.isfile(d))
os.remove(d)
print(os.path.exists(d))

# --- walk, top-down ---------------------------------------------------------
for triple in os.walk(root):
    print(triple[0], triple[1], triple[2])

# --- chdir / getcwd ---------------------------------------------------------
before = os.getcwd()
os.chdir(root)
print(os.getcwd() == os.path.join(before, root))
os.chdir(before)
print(os.getcwd() == before)
print(os.path.abspath(root) == os.path.join(before, root))
print(os.path.isabs(os.path.abspath(root)))

# --- errno maps to the OSError subclass, with CPython's message ------------
try:
    os.listdir(missing)
except FileNotFoundError as exc:
    print(exc)
try:
    os.stat(missing)
except FileNotFoundError as exc:
    print(exc)
try:
    os.chdir(missing)
except FileNotFoundError as exc:
    print(exc)
try:
    os.mkdir(root)
except FileExistsError as exc:
    print(exc)
try:
    os.rename(missing, os.path.join(root, "other"))
except FileNotFoundError as exc:
    print(exc)
try:
    os.listdir(a)
except NotADirectoryError as exc:
    print(exc)
try:
    os.rmdir(sub)
except OSError as exc:
    # ENOTEMPTY's NUMBER is per-libc (66 on the BSD family, 39 on Linux), so
    # only the raised class is pinned here, not the message.
    print("rmdir non-empty raises OSError")
# A FileNotFoundError is an OSError, so the broad handler catches it too.
try:
    os.unlink(missing)
except OSError as exc:
    print(exc)

# --- process identity -------------------------------------------------------
print(os.getpid() > 0, os.getppid() > 0)
print(os.getuid() >= 0, os.geteuid() >= 0)
print(os.getgid() >= 0, os.getegid() >= 0)
print(os.strerror(2))
print(os.strerror(13))
print(os.strerror(17))

# --- teardown ---------------------------------------------------------------
os.remove(a)
os.rmdir(deep)
os.rmdir(sub)
os.rmdir(root)
print(os.path.exists(root))
