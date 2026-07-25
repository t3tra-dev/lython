# pathlib's Path, lexical surface then filesystem surface.
#
# Every line matches CPython 3.14 byte for byte EXCEPT the three `parts`
# reads: pathlib.py documents parts as a list[str] rather than a tuple,
# because the component count is not static. The content is CPython's.
#
# Properties are bound to locals before use on purpose: `str(p.parent)`
# straight through segfaults, and two property reads in one print() render
# with repr (both reported to the Wave 3 foundation track).
from pathlib import Path

# --- lexical ----------------------------------------------------------------
p = Path("/x/y/z.txt")
print(str(p))
print(p.name)
print(p.stem)
print(p.suffix)
parent = p.parent
print(str(parent))
print(p.parts)
print(p.is_absolute())
child = p / "w"
print(str(child))
sibling = parent / "other.md"
print(str(sibling))
renamed = p.with_name("q.md")
print(str(renamed))
resuffixed = p.with_suffix(".md")
print(str(resuffixed))
stripped = p.with_suffix("")
print(str(stripped))
print(str(Path("a", "b", "c")))
print(str(Path("a/b/./c//d")))
ab = Path("a/b")
print(ab.parts)
print(str(Path(".")))
bare = Path("x")
bare_parent = bare.parent
print(str(bare_parent))
print(repr(Path("/tmp/a")))
print(Path("/a") == Path("/a"), Path("/a") == Path("/b"))
joined = Path("/x/y").joinpath("z", "w")
print(str(joined))
# POSIX gives two leading slashes meaning and collapses three or more to one.
print(str(Path("//a//b")))
print(str(Path("///a")))
# '..' is lexical: PurePath keeps it, because collapsing it is wrong through a
# symlink.
print(str(Path("a/b/../c")))
root = Path("/")
print(root.parts)
print(root.is_absolute())
targz = Path("/a/b.tar.gz")
print(targz.suffix)
print(targz.stem)
cshrc = Path("/a/.cshrc")
print(cshrc.suffix)
print(cshrc.stem)

# --- filesystem -------------------------------------------------------------
top = Path("_golden_pathlib_tmp")
sub = top / "sub"
note = sub / "a.txt"
log = top / "b.log"

sub.mkdir(0o755, True, True)
print(top.is_dir(), sub.is_dir())
print(note.write_text("hello"))
print(note.read_text())
print(log.write_bytes(b"bin"))
print(log.read_bytes())
print(note.exists(), note.is_file(), note.is_dir(), note.is_symlink())
info = note.stat()
print(info.st_size)

# Names are collected and sorted in the case: CPython's iterdir order is the
# directory's, not sorted, so only a sorted comparison is host-independent.
names: list[str] = []
entries = top.iterdir()
for entry in entries:
    names.append(entry.name)
names.sort()
print(names)

matched: list[str] = []
logs = top.glob("*.log")
for entry in logs:
    matched.append(entry.name)
matched.sort()
print(matched)

deep: list[str] = []
txts = top.rglob("*.txt")
for entry in txts:
    deep.append(entry.name)
deep.sort()
print(deep)

everything: list[str] = []
alls = top.glob("*")
for entry in alls:
    everything.append(entry.name)
everything.sort()
print(everything)

moved = log.rename("_golden_pathlib_tmp/c.log")
print(moved.name)
print(moved.is_file(), log.exists())
moved.unlink()
print(moved.exists())
missing = top / "nope"
missing.unlink(True)
print(missing.exists())
note.unlink()
sub.rmdir()
top.rmdir()
print(top.exists())
