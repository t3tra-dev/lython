import sys

# Target-independent manifest constants (runtime/modules/sys.mlir).
print(sys.hexversion)
print(sys.api_version)
print(sys.maxunicode)
print(sys.version)
print(sys.copyright)
print(sys.abiflags)
print(sys.float_repr_style)

# Target-dependent platform constants (PlatformConstants.h). The suite runs
# on 64-bit little-endian hosts only, so the folded values are stable.
print(sys.maxsize)
print(sys.byteorder)

# Zero-arg encoding callables fold to "utf-8" on every target.
print(sys.getdefaultencoding())
print(sys.getfilesystemencoding())

# Folded literals participate in static comparison like sys.platform does.
if sys.byteorder == "little":
    print("le")
else:
    print("be")

# The from-import binding path resolves the same constants.
from sys import maxsize, hexversion, version

print(maxsize)
print(hexversion)
print(version)
