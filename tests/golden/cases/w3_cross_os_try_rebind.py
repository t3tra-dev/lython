# Wave 3 cross-track: os/pathlib's errno -> OSError wiring (os-time) observed
# through a try statement whose body, handler and continuation all rebind the
# same locals (foundation).
#
# What this pins that neither track pins alone: the handler runs with the
# value the TRY BODY last stored (`stage`), the value the handler stores is
# what the continuation reads (`detail`), and the exception that carries them
# there is the errno-derived OSError subclass rather than a plain OSError. The
# body's rebind is promoted to storage for the extent of the statement, so a
# path string produced by os.path.join before the promotion and read after it
# has to survive the promotion unchanged.
#
# Every path is RELATIVE so the OSError messages, which carry the failing
# path, are host-independent and match CPython 3.14 byte for byte. The tree is
# never created: the case only needs paths that cannot exist.
import os
from pathlib import Path

root = "_golden_w3_cross_absent"
target = os.path.join(root, "nope.txt")
sibling = os.path.join(root, "sub", "other.txt")

# --- the try body's rebind reaches the handler and the continuation ---------
stage = "start"
detail = "none"
try:
    stage = "probing"
    info = os.stat(target)
    stage = "probed"
except FileNotFoundError as error:
    detail = str(error)
    stage = stage + "-failed"
print(stage)
print(detail)

# --- the same shape with a finally body reading the try body's rebind -------
phase = "before"
seen = "unset"
try:
    phase = "listing"
    names = os.listdir(root)
    print(len(names))
except OSError as error:
    seen = str(error)
finally:
    print(phase)
print(seen)

# --- else runs on the non-raising path and the continuation sees the body ---
outcome = "none"
try:
    outcome = "joined"
    joined = os.path.join(root, "sub")
except OSError as error:
    outcome = "raised"
else:
    outcome = outcome + "-ok"
print(outcome)

# --- os.path stays lexical on paths that do not exist ----------------------
print(os.path.basename(target))
print(os.path.dirname(target))
print(os.path.splitext(target))
print(os.path.split(sibling))
print(os.path.normpath(os.path.join(root, "sub", "..", "nope.txt")))
print(os.path.exists(target), os.path.isfile(target), os.path.isdir(root))

# --- pathlib agrees with os.path on the same non-existent tree -------------
# Each property read is bound to a local before use: two property reads inside
# one print() render as repr instead of str (recorded on the pathlib port).
p = Path(target)
text = str(p)
print(text)
name = p.name
print(name)
stem = p.stem
print(stem)
suffix = p.suffix
print(suffix)
parent = p.parent
parent_text = str(parent)
print(parent_text)
parts = p.parts
print(parts)
absolute = p.is_absolute()
print(absolute)
present = p.exists()
print(present)
renamed = p.with_suffix(".log")
renamed_text = str(renamed)
print(renamed_text)
child = Path(root) / "sub"
child_text = str(child)
print(child_text)

# --- reading a missing file through pathlib raises the same subclass -------
# The MESSAGE is deliberately not printed here: builtins' open() builds its own
# text in runtime/modules/_io.mlir (`No such file or directory: <path>`, no
# errno prefix and no quoting) instead of going through the errno -> OSError
# wiring the os functions above use, so it would not match CPython. The CLASS
# is what this pins.
which = "none"
try:
    which = "reading"
    opened = open(text, "r")
    opened.close()
except FileNotFoundError as error:
    which = which + "-missing"
print(which)

# --- a directory that does not exist raises through os, not Path ----------
try:
    os.rmdir(root)
except FileNotFoundError as error:
    print(str(error))
try:
    os.stat(sibling)
except FileNotFoundError as error:
    print(str(error))
print(os.strerror(2))
