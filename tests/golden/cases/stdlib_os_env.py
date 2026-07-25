# os's environment surface. `os.environ` is absent (a container-typed module
# global is not visible across an import boundary yet), so getenv/putenv/
# unsetenv talk to the process environment directly and `has_env` /
# `environ_entries` stand in for the mapping -- see os.py's docstring. That
# makes putenv IMMEDIATELY visible to getenv, which is where this diverges
# from CPython: there putenv bypasses os.environ, which is what getenv reads.
# Everything is printed as a bool or a value the case itself set, so the output
# does not depend on the host's environment.
import os

name = "LYTHON_GOLDEN_OS_ENV"

print(os.has_env(name))
print(os.getenv(name, "fallback"))

os.putenv(name, "first")
print(os.has_env(name))
print(os.getenv(name))
print(os.getenv(name, "fallback"))

os.putenv(name, "second")
print(os.getenv(name))

# An empty value is SET, which is exactly what has_env distinguishes and a
# `getenv(...) == ""` test cannot.
os.putenv(name, "")
print(os.has_env(name))
print(os.getenv(name) == "")
print(os.getenv(name, "fallback") == "")

os.unsetenv(name)
print(os.has_env(name))
print(os.getenv(name, "fallback"))
# The documented deviation: an unset variable reads back as '' rather than
# None, because an Optional[str] has no physical layout across the native
# boundary yet.
print(os.getenv(name) == "")

# The raw vector, every entry of which is a "KEY=VALUE" string. Scanned from a
# helper: the list has to be bound before the loop reads it, and the matched
# entry has to be copied rather than aliased.
def entry_for(key: str) -> str:
    entries = os.environ_entries()
    found = ""
    for entry in entries:
        if entry.startswith(key + "="):
            found = entry + ""
    return found


entries = os.environ_entries()
print(len(entries) > 0)
os.putenv(name, "in-vector")
print(entry_for(name))
os.unsetenv(name)
print(entry_for(name) == "")
