# os's environment surface: `os.environ` is a snapshot taken when os is
# imported, `environ[k] = v` / `del environ[k]` write through to the process,
# and `getenv` reads the snapshot -- so a bare `putenv` is NOT visible to it.
# Everything is printed as a bool or a value the case itself set, so the output
# does not depend on the host's environment.
import os

name = "LYTHON_GOLDEN_OS_ENV"

print(name in os.environ)
print(os.getenv(name, "fallback"))

# ⛔ putenv bypasses os.environ, which is what getenv reads. This port read
# the process environment directly and printed "first" here.
os.putenv(name, "first")
print(name in os.environ)
print(os.getenv(name))
print(os.getenv(name, "fallback"))

os.environ[name] = "second"
print(os.getenv(name))
print(os.environ[name])

# An empty value is SET, which `in` distinguishes and a `getenv(...) == ""`
# test cannot.
os.environ[name] = ""
print(name in os.environ)
print(os.getenv(name) == "")
print(os.getenv(name, "fallback") == "")

del os.environ[name]
print(name in os.environ)
print(os.getenv(name, "fallback"))
# An UNSET variable reads back as None, not as ''.
print(os.getenv(name) is None)
print(os.getenv(name) == "")

try:
    os.environ[name]
except KeyError as e:
    print("KeyError", e)
try:
    del os.environ[name]
except KeyError as e:
    print("KeyError", e)

os.environ[name + "_A"] = "a"
os.environ[name + "_B"] = "b"
for key in os.environ:
    if key.startswith(name):
        print(key)
for key, value in os.environ.items():
    if key.startswith(name):
        print(key, value)
print(os.environ.setdefault(name + "_A", "z"), os.environ.setdefault(name, "c"))
print(os.environ.pop(name), name in os.environ)
snapshot = os.environ.copy()
print(type(snapshot).__name__, snapshot[name + "_A"])
print(len(os.environ) > 0, type(os.environ).__name__)
