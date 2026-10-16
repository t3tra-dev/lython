# The two environment helpers this port adds to os: `has_env` is `key in
# os.environ` under another name, and `environ_entries()` is the raw
# "KEY=VALUE" vector of the PROCESS environment -- the one place a bare
# `putenv` shows, since os.environ is a snapshot.
import os

name = "LYTHON_GOLDEN_OS_ENV_EXTRAS"


def entry_for(key: str) -> str:
    found = ""
    for entry in os.environ_entries():
        if entry.startswith(key + "="):
            found = entry + ""
    return found


print(os.has_env(name), entry_for(name) == "")
os.putenv(name, "in-vector")
print(os.has_env(name), entry_for(name))
os.environ[name] = ""
print(os.has_env(name), entry_for(name))
del os.environ[name]
print(os.has_env(name), entry_for(name) == "")
print(len(os.environ_entries()) > 0)
